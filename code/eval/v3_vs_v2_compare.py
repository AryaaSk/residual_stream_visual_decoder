"""Build the v3-vs-v2.0 side-by-side comparison grid.

For each held-out prompt:
  - v2.0 best-CLIP drawing (canonical-SFT-derived)
  - v3 best-cosine drawing (pure NLA, no SFT crutch)

Two columns × 20 rows = the headline image showing what each approach
actually produces, with the metric each was optimised for printed underneath.

Honest framing:
  - v2.0 column: drawings are polished but the activation only acts as a
    concept-selector over ~220 memorized canonical templates.
  - v3 column: drawings are whatever maximises h_text↔h_image_only
    reconstruction under the SAME Qwen 3.5-4B. May be abstract; the metric
    underneath is what matters.

Output:
  findings/v3/eval/comparison_grid.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder
from render import render as stroke_render


HERO = [
    ("cat",       "I am thinking about a cat.",      "a drawing of a cat"),
    ("dog",       "I am thinking about a dog.",      "a drawing of a dog"),
    ("elephant",  "I am thinking about an elephant.","a drawing of an elephant"),
    ("fish",      "Imagine a fish.",                 "a drawing of a fish"),
    ("horse",     "I am thinking about a horse.",    "a drawing of a horse"),
    ("flower",    "Imagine a flower in bloom.",      "a drawing of a flower"),
    ("sun",       "The sun is shining.",             "a drawing of a sun"),
    ("tree",      "I am picturing a tree.",          "a drawing of a tree"),
    ("airplane",  "I am picturing an airplane.",     "a drawing of an airplane"),
    ("car",       "I am thinking about a car.",      "a drawing of a car"),
    ("apple",     "I am thinking about an apple.",   "a drawing of an apple"),
    ("pizza",     "I am thinking about a pizza.",    "a drawing of a pizza"),
    ("mountain",  "I am picturing a mountain.",      "a drawing of a mountain"),
    ("cloud",     "I am picturing a cloud in the sky.", "a drawing of a cloud"),
    ("bird",      "I am picturing a bird flying across the sky.", "a drawing of a bird"),
    ("star",      "I am picturing a star in the night sky.", "a drawing of a star"),
]


def label_box(text: str, sub: str, width: int, height: int = 60) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        f1 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        f2 = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        f1 = ImageFont.load_default()
        f2 = f1
    d.text((10, 8), text, fill=(0, 0, 0), font=f1)
    d.text((10, 32), sub, fill=(120, 120, 120), font=f2)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v2-ckpt", type=Path, default=Path("checkpoints/v2_0/L10/final"),
                   help="v2.0 SFT ckpt (canonical-drawing AV)")
    p.add_argument("--v3-ckpt", type=Path, required=True,
                   help="v3 trained AV ckpt (pure NLA)")
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--v2-layer", type=int, default=10)
    p.add_argument("--v3-layer", type=int, required=True)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--max-tokens", type=int, default=240)
    p.add_argument("--display-scale", type=float, default=2.0)
    p.add_argument("--out", type=Path,
                   default=Path("findings/v3/eval/comparison_grid.png"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cmp] loading v2 AV from {args.v2_ckpt}", flush=True)
    av2 = StrokeDecoder.from_ckpt(args.v2_ckpt, model_id=args.model_id)
    av2.model.eval()

    print(f"[cmp] loading v3 AV from {args.v3_ckpt}", flush=True)
    av3 = StrokeDecoder.from_ckpt(args.v3_ckpt, model_id=args.model_id)
    av3.model.eval()
    device = av3.device()

    print("[cmp] loading frozen Qwen (ImageTextToText) for h_text and h_image ...", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    qwen_eval = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    qwen_proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    for p_ in qwen_eval.parameters():
        p_.requires_grad = False

    print("[cmp] loading CLIP ranker for v2 column ...", flush=True)
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def h_text(text: str, layer: int) -> torch.Tensor:
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        wrap = qwen_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        inp = qwen_proc(text=[wrap], images=None, return_tensors="pt").to(device)
        out = qwen_eval(**inp, output_hidden_states=True, use_cache=False)
        return out.hidden_states[layer][0, -1, :].detach().to(torch.float32)

    @torch.no_grad()
    def h_image_only(img: Image.Image, layer: int) -> torch.Tensor | None:
        msgs = [{"role": "user", "content": [{"type": "image"}]}]
        wrap = qwen_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        try:
            inp = qwen_proc(text=[wrap], images=[img], return_tensors="pt").to(device)
            out = qwen_eval(**inp, output_hidden_states=True, use_cache=False)
            return out.hidden_states[layer][0, -1, :].detach().to(torch.float32)
        except Exception:
            return None

    @torch.no_grad()
    def clip_score(img: Image.Image, text: str) -> float:
        inp = clip_proc(text=[text], images=[img], return_tensors="pt", padding=True).to("cuda")
        return float(clip_model(**inp).logits_per_image.squeeze().item())

    @torch.no_grad()
    def best_v2(slug: str, prompt: str, clip_text: str) -> tuple[Image.Image, dict]:
        ht = h_text(prompt, args.v2_layer)
        # Use v2's layer for the activation injection
        ids_list = av2.generate_from_activation_batched(
            ht, layer_ell=args.v2_layer,
            n_samples=args.n_samples,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
        cands = []
        for ids in ids_list:
            strokes, _ = av2.vocab.decode_tokens_with_stats(ids.tolist())
            if len(strokes) < 8:
                continue
            img = stroke_render(strokes, display_scale=args.display_scale).convert("RGB")
            cs = clip_score(img, clip_text)
            cands.append({"img": img, "strokes": strokes, "clip": cs, "n_strokes": len(strokes)})
        if not cands:
            return Image.new("RGB", (448, 448), color=(240, 240, 240)), {"degenerate": True}
        cands.sort(key=lambda c: c["clip"], reverse=True)
        b = cands[0]
        # Also report v3-style cosine on the same drawing for honest comparison
        h_i = h_image_only(b["img"], args.v3_layer)
        cos = float(F.cosine_similarity(h_text(prompt, args.v3_layer), h_i, dim=0).item()) if h_i is not None else None
        return b["img"], {"clip": b["clip"], "cosine": cos, "n_strokes": b["n_strokes"]}

    @torch.no_grad()
    def best_v3(slug: str, prompt: str, clip_text: str) -> tuple[Image.Image, dict]:
        ht = h_text(prompt, args.v3_layer)
        ids_list = av3.generate_from_activation_batched(
            ht, layer_ell=args.v3_layer,
            n_samples=args.n_samples,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
        cands = []
        for ids in ids_list:
            strokes, _ = av3.vocab.decode_tokens_with_stats(ids.tolist())
            if len(strokes) < 2:
                continue
            img = stroke_render(strokes, display_scale=args.display_scale).convert("RGB")
            h_i = h_image_only(img, args.v3_layer)
            if h_i is None:
                continue
            cos = float(F.cosine_similarity(ht, h_i, dim=0).item())
            cs = clip_score(img, clip_text)
            cands.append({"img": img, "strokes": strokes, "cosine": cos, "clip": cs,
                          "n_strokes": len(strokes)})
        if not cands:
            return Image.new("RGB", (448, 448), color=(240, 240, 240)), {"degenerate": True}
        cands.sort(key=lambda c: c["cosine"], reverse=True)
        b = cands[0]
        return b["img"], {"clip": b["clip"], "cosine": b["cosine"], "n_strokes": b["n_strokes"]}

    # Build grid
    CELL = 448
    LABEL_H = 60
    HEADER_H = 60
    rows = len(HERO)
    cols = 2
    GW = cols * CELL + 30
    GH = HEADER_H + rows * (CELL + LABEL_H)
    grid = Image.new("RGB", (GW, GH), color=(255, 255, 255))
    d = ImageDraw.Draw(grid)
    try:
        fh = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        fh = ImageFont.load_default()
    d.text((20, 18), "v2.0 (CLIP-ranked from canonical SFT)", fill=(20, 20, 20), font=fh)
    d.text((20 + CELL + 30, 18), "v3 (h-reconstruction-ranked, pure NLA)", fill=(20, 20, 20), font=fh)

    results = []
    for r, (slug, prompt, ctext) in enumerate(HERO):
        t0 = time.time()
        v2_img, v2_meta = best_v2(slug, prompt, ctext)
        v3_img, v3_meta = best_v3(slug, prompt, ctext)
        y = HEADER_H + r * (CELL + LABEL_H)
        grid.paste(v2_img.resize((CELL, CELL)), (0, y))
        grid.paste(v3_img.resize((CELL, CELL)), (CELL + 30, y))
        # Labels
        v2_label = f"{slug}: CLIP {v2_meta.get('clip', 0):.1f}" + (f", v3-cos {v2_meta.get('cosine', 0):.3f}" if v2_meta.get('cosine') is not None else "")
        v3_label = f"{slug}: v3-cos {v3_meta.get('cosine', 0):+.3f}, CLIP {v3_meta.get('clip', 0):.1f}"
        d2 = ImageDraw.Draw(grid)
        try:
            f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            f = ImageFont.load_default()
        d2.text((10, y + CELL + 10), v2_label, fill=(0, 0, 0), font=f)
        d2.text((CELL + 40, y + CELL + 10), v3_label, fill=(0, 0, 0), font=f)
        results.append({"slug": slug, "prompt": prompt,
                        "v2": v2_meta, "v3": v3_meta,
                        "wallclock_s": round(time.time() - t0, 1)})
        print(f"[cmp] {slug:10s}  v2 CLIP {v2_meta.get('clip', 0):.1f}  | v3 cos {v3_meta.get('cosine', 0):+.3f}  ({time.time()-t0:.1f}s)", flush=True)

    grid.save(args.out)
    (args.out.parent / "comparison_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[cmp] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
