"""Render upscaled stroke animations (MP4) for each (layer × concept) pair.

For each AV checkpoint, for each hero concept:
  1. Best-of-N CLIP+gen-likelihood ranked drawing
  2. Render the winner as MP4 at display_scale=4 (4× upscale, 896x896 canvas)
  3. fps=24, one frame per stroke

Output: artefacts/v3/viral/anim/{layer_tag}/{concept}.mp4
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder
from render import render as stroke_render


HERO = [
    ("cat",      "I am thinking about a cat.",       "cat"),
    ("dog",      "I am thinking about a dog.",       "dog"),
    ("elephant", "I am thinking about an elephant.", "elephant"),
    ("fish",     "Imagine a fish.",                  "fish"),
    ("horse",    "I am thinking about a horse.",     "horse"),
    ("flower",   "Imagine a flower in bloom.",       "flower"),
    ("sun",      "The sun is shining.",              "sun"),
    ("tree",     "I am picturing a tree.",           "tree"),
    ("airplane", "I am picturing an airplane.",      "airplane"),
    ("car",      "I am thinking about a car.",       "car"),
    ("bird",     "I am picturing a bird flying.",    "bird"),
    ("mountain", "I am picturing a mountain.",       "mountain"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--layer-tag", default=None, help="Folder name (defaults to L{layer})")
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--max-tokens", type=int, default=240)
    p.add_argument("--display-scale", type=float, default=4.0,
                   help="Upscale for MP4 frames (4 = 896×896)")
    p.add_argument("--score-display-scale", type=float, default=1.0,
                   help="Scale used for Qwen scoring (1.0 = 224, native)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--out-dir", type=Path, default=Path("artefacts/v3/viral/anim"))
    p.add_argument("--question", default="What is this a drawing of?")
    p.add_argument("--continuation-prefix", default="A drawing of a")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    tag = args.layer_tag or f"L{args.layer:02d}"
    out_dir = args.out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[anim] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    device = av.device()

    print("[anim] loading frozen Qwen (ImageTextToText) for ranking ...", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    qwen = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    qproc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    for p_ in qwen.parameters():
        p_.requires_grad = False
    tok = qproc.tokenizer

    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.question}]}]
    PREFIX_TEXT = qproc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + args.continuation_prefix
    concept_ids = {c: torch.tensor([tok.encode(" " + c, add_special_tokens=False)],
                                    dtype=torch.long, device=device)
                   for _, _, c in HERO}

    @torch.no_grad()
    def gen_logp(image, concept):
        cand = concept_ids.get(concept)
        if cand is None or cand.shape[1] == 0:
            return -1e9
        try:
            inp = qproc(text=[PREFIX_TEXT], images=[image], return_tensors="pt").to(device)
        except Exception:
            return -1e9
        prefix_len = inp["input_ids"].shape[1]
        full = {k: v for k, v in inp.items()}
        full["input_ids"] = torch.cat([inp["input_ids"], cand], dim=1)
        if "attention_mask" in full:
            full["attention_mask"] = torch.cat([inp["attention_mask"], torch.ones_like(cand)], dim=1)
        if "mm_token_type_ids" in full:
            full["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(cand)], dim=1)
        try:
            out = qwen(**full, use_cache=False)
        except Exception:
            return -1e9
        T = full["input_ids"].shape[1]
        logits = out.logits[0, prefix_len - 1: T - 1, :]
        target = full["input_ids"][0, prefix_len:]
        logp = F.log_softmax(logits.float(), dim=-1)
        chosen = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return float(chosen.sum().item())

    results = []
    for slug, prompt, concept in HERO:
        t0 = time.time()
        enc = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach()
        ids_list = av.generate_from_activation_batched(
            h, layer_ell=args.layer, n_samples=args.n_samples,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
        candidates = []
        for ids in ids_list:
            strokes, _ = av.vocab.decode_tokens_with_stats(ids.tolist())
            if len(strokes) < 8:
                continue
            img_low = stroke_render(strokes, display_scale=args.score_display_scale).convert("RGB")
            lp = gen_logp(img_low, concept)
            candidates.append({"strokes": strokes, "logp": lp, "n_strokes": len(strokes)})
        if not candidates:
            print(f"[anim] {slug:10s}  all degenerate", flush=True)
            continue
        candidates.sort(key=lambda c: c["logp"], reverse=True)
        best = candidates[0]
        # Render winner as MP4 (animated stroke-by-stroke)
        mp4_path = out_dir / f"{slug}.mp4"
        png_path = out_dir / f"{slug}.png"
        # MP4 with animation
        stroke_render(best["strokes"], display_scale=args.display_scale,
                      save_animation_path=str(mp4_path), fps=args.fps)
        # Static PNG too for compositing
        final_png = stroke_render(best["strokes"], display_scale=args.display_scale).convert("RGB")
        final_png.save(png_path)
        results.append({"slug": slug, "logp": best["logp"], "n_strokes": best["n_strokes"]})
        print(f"[anim] {tag} {slug:10s}  logp={best['logp']:+.2f}  n_strokes={best['n_strokes']:3d}  → {mp4_path.name}  ({time.time()-t0:.1f}s)", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"[anim] {tag} DONE → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
