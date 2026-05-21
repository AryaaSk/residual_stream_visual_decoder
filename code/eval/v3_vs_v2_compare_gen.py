"""v3-vs-v2.0 side-by-side comparison (generation-likelihood version).

For each hero prompt, render:
  - v2.0 best-CLIP-of-N drawing (canonical-SFT-trained AV; concept-selector style)
  - v3   best-reward-of-N drawing (pure NLA, generation-likelihood-trained AV)

Both labelled with BOTH metrics:
  - CLIP score (against "a drawing of a {concept}")
  - v3 reward (log P(concept | image, "A drawing of a "))
  - Qwen's predicted top-1 from the 44 SFT concepts

Output:
    findings/v3/comparison/grid.png            # 2-column grid
    findings/v3/comparison/summary.json        # per-row metrics
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
    ("cat",       "I am thinking about a cat.",       "a drawing of a cat"),
    ("dog",       "I am thinking about a dog.",       "a drawing of a dog"),
    ("elephant",  "I am thinking about an elephant.", "a drawing of an elephant"),
    ("fish",      "Imagine a fish.",                  "a drawing of a fish"),
    ("horse",     "I am thinking about a horse.",     "a drawing of a horse"),
    ("flower",    "Imagine a flower in bloom.",       "a drawing of a flower"),
    ("sun",       "The sun is shining.",              "a drawing of a sun"),
    ("tree",      "I am picturing a tree.",           "a drawing of a tree"),
    ("airplane",  "I am picturing an airplane.",      "a drawing of an airplane"),
    ("car",       "I am thinking about a car.",       "a drawing of a car"),
    ("apple",     "I am thinking about an apple.",    "a drawing of an apple"),
    ("pizza",     "I am thinking about a pizza.",     "a drawing of a pizza"),
    ("mountain",  "I am picturing a mountain.",       "a drawing of a mountain"),
    ("cloud",     "I am picturing a cloud in the sky.", "a drawing of a cloud"),
    ("bird",      "I am picturing a bird flying across the sky.", "a drawing of a bird"),
    ("star",      "I am picturing a star in the night sky.", "a drawing of a star"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v2-ckpt", type=Path, default=Path("checkpoints/v2_0/L10/final"))
    p.add_argument("--v3-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--v2-layer", type=int, default=10)
    p.add_argument("--v3-layer", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--max-tokens", type=int, default=240)
    p.add_argument("--display-scale", type=float, default=2.0)
    p.add_argument("--question", default="What is this a drawing of?")
    p.add_argument("--continuation-prefix", default="A drawing of a")
    p.add_argument("--out", type=Path, default=Path("findings/v3/comparison/grid.png"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cmp-g] loading v2 AV from {args.v2_ckpt}", flush=True)
    av2 = StrokeDecoder.from_ckpt(args.v2_ckpt, model_id=args.model_id)
    av2.model.eval()

    print(f"[cmp-g] loading v3 AV from {args.v3_ckpt}", flush=True)
    av3 = StrokeDecoder.from_ckpt(args.v3_ckpt, model_id=args.model_id)
    av3.model.eval()
    device = av3.device()

    print("[cmp-g] loading frozen Qwen (ImageTextToText) + CLIP ...", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    qwen = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    qproc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    PROMPT_MSGS = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": args.question}],
    }]
    PREFIX_TEXT = qproc.apply_chat_template(PROMPT_MSGS, tokenize=False, add_generation_prompt=True) + args.continuation_prefix

    @torch.no_grad()
    def h_text(text: str, layer: int) -> torch.Tensor:
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        wrap = qproc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        inp = qproc(text=[wrap], images=None, return_tensors="pt").to(device)
        out = qwen(**inp, output_hidden_states=True, use_cache=False)
        return out.hidden_states[layer][0, -1, :].detach().to(torch.float32)

    @torch.no_grad()
    def gen_logp(image: Image.Image, concept: str) -> float | None:
        full = PREFIX_TEXT + " " + concept
        try:
            inp_full = qproc(text=[full], images=[image], return_tensors="pt").to(device)
            inp_pre = qproc(text=[PREFIX_TEXT], images=[image], return_tensors="pt")
        except Exception:
            return None
        prefix_len = inp_pre["input_ids"].shape[1]
        T = inp_full["input_ids"].shape[1]
        if T <= prefix_len:
            return None
        try:
            out = qwen(**inp_full, use_cache=False)
        except Exception:
            return None
        logits = out.logits[0, prefix_len - 1: T - 1, :]
        target = inp_full["input_ids"][0, prefix_len:]
        logp = F.log_softmax(logits.float(), dim=-1)
        chosen = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return float(chosen.sum().item())

    @torch.no_grad()
    def clip_score(image: Image.Image, text: str) -> float:
        inp = clip_proc(text=[text], images=[image], return_tensors="pt", padding=True).to("cuda")
        return float(clip_model(**inp).logits_per_image.squeeze().item())

    def best_of_n(av, slug, prompt, clip_text, rank_by: str):
        ht = h_text(prompt, args.v2_layer if av is av2 else args.v3_layer)
        ids_list = av.generate_from_activation_batched(
            ht, layer_ell=args.v2_layer if av is av2 else args.v3_layer,
            n_samples=args.n_samples,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
        cands = []
        for ids in ids_list:
            strokes, _ = av.vocab.decode_tokens_with_stats(ids.tolist())
            if len(strokes) < 2:
                continue
            img = stroke_render(strokes, display_scale=args.display_scale).convert("RGB")
            cs = clip_score(img, clip_text)
            lp = gen_logp(img, slug)
            cands.append({"img": img, "strokes": strokes, "clip": cs,
                          "lp": lp if lp is not None else -1e9,
                          "n_strokes": len(strokes)})
        if not cands:
            return Image.new("RGB", (448, 448), color=(240, 240, 240)), {"degenerate": True}
        if rank_by == "clip":
            cands.sort(key=lambda c: c["clip"], reverse=True)
        else:
            cands.sort(key=lambda c: c["lp"], reverse=True)
        return cands[0]["img"], {"clip": cands[0]["clip"], "lp": cands[0]["lp"],
                                 "n_strokes": cands[0]["n_strokes"]}

    CELL = 448
    HEAD_H = 60
    LABEL_H = 60
    GAP = 30
    GW = 2 * CELL + GAP
    GH = HEAD_H + len(HERO) * (CELL + LABEL_H)
    grid = Image.new("RGB", (GW, GH), color=(255, 255, 255))
    d = ImageDraw.Draw(grid)
    try:
        fh = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        fr = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        fh = ImageFont.load_default(); fr = fh
    d.text((20, 18), "v2.0  (CLIP-best, canonical-SFT)", fill=(20, 20, 20), font=fh)
    d.text((20 + CELL + GAP, 18), "v3   (gen-log-prob-best, pure NLA)", fill=(20, 20, 20), font=fh)

    results = []
    for r, (slug, prompt, ctext) in enumerate(HERO):
        t0 = time.time()
        v2_img, v2_meta = best_of_n(av2, slug, prompt, ctext, rank_by="clip")
        v3_img, v3_meta = best_of_n(av3, slug, prompt, ctext, rank_by="lp")
        y = HEAD_H + r * (CELL + LABEL_H)
        grid.paste(v2_img.resize((CELL, CELL)), (0, y))
        grid.paste(v3_img.resize((CELL, CELL)), (CELL + GAP, y))
        d2 = ImageDraw.Draw(grid)
        v2_lab = f"{slug}: CLIP {v2_meta.get('clip', 0):+.1f}   lp {v2_meta.get('lp', 0):+.2f}"
        v3_lab = f"{slug}: lp {v3_meta.get('lp', 0):+.2f}   CLIP {v3_meta.get('clip', 0):+.1f}"
        d2.text((10, y + CELL + 8), v2_lab, fill=(0, 0, 0), font=fr)
        d2.text((CELL + GAP + 10, y + CELL + 8), v3_lab, fill=(0, 0, 0), font=fr)
        results.append({"slug": slug, "prompt": prompt, "v2": v2_meta, "v3": v3_meta,
                        "wallclock_s": round(time.time() - t0, 1)})
        print(f"[cmp-g] {slug:10s}  v2 CLIP={v2_meta.get('clip', 0):+.1f}  v3 lp={v3_meta.get('lp', 0):+.2f}  ({time.time()-t0:.1f}s)", flush=True)

    grid.save(args.out)
    (args.out.parent / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n[cmp-g] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
