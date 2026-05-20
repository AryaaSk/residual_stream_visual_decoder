"""CLIP-ranked best-of-N gallery.

For each concept:
  1. Generate N candidates via batched AV sampling (fast).
  2. Render each candidate to a PNG.
  3. Use CLIP to compute image-text similarity vs "a drawing of a {concept}".
  4. Pick the top-K candidates by CLIP score.

This is a strictly better ranker than the heuristic in fast_best_of_n.py
because CLIP measures actual semantic resemblance to the concept, not just
"is the drawing well-formed."

Requires `transformers` (already a dep) and a CLIP model. Uses
`openai/clip-vit-base-patch32` by default (~600MB, fast).
"""

from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402
from render import render as stroke_render  # noqa: E402


HERO = [
    ("cat",      "I am thinking about a cat.",      "a drawing of a cat"),
    ("dog",      "I am thinking about a dog.",      "a drawing of a dog"),
    ("fish",     "Imagine a fish.",                 "a drawing of a fish"),
    ("bird",     "I am picturing a bird flying across the sky.", "a drawing of a bird"),
    ("horse",    "I am thinking about a horse.",    "a drawing of a horse"),
    ("elephant", "I am thinking about an elephant.","a drawing of an elephant"),
    ("flower",   "Imagine a flower in bloom.",      "a drawing of a flower"),
    ("tree",     "I am picturing a tree.",          "a drawing of a tree"),
    ("cactus",   "I am picturing a cactus in the desert.", "a drawing of a cactus"),
    ("mountain", "I am picturing a mountain.",      "a drawing of a mountain"),
    ("sun",      "The sun is shining.",             "a drawing of a sun"),
    ("cloud",    "I am picturing a cloud in the sky.", "a drawing of a cloud"),
    ("star",     "I am picturing a star in the night sky.", "a drawing of a star"),
    ("house",    "I am picturing a small house with a red roof.", "a drawing of a house"),
    ("car",      "I am thinking about a car.",      "a drawing of a car"),
    ("airplane", "I am picturing an airplane.",     "a drawing of an airplane"),
    ("apple",    "I am thinking about an apple.",   "a drawing of an apple"),
    ("pizza",    "I am thinking about a pizza.",    "a drawing of a pizza"),
    ("clock",    "I am picturing a clock on the wall.", "a drawing of a clock"),
    ("umbrella", "I am picturing an umbrella.",     "a drawing of an umbrella"),
]


def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(name)
    return model, proc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--pick-k", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=240)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--display-scale", type=float, default=4.0)
    p.add_argument("--mp4", action="store_true", default=True)
    p.add_argument("--save-all", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[clip] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    device = av.device()

    print("[clip] loading CLIP ranker ...", flush=True)
    clip_model, clip_proc = load_clip()

    summary = []
    import time
    t_all = time.time()
    for slug, prompt, clip_text in HERO:
        t0 = time.time()
        enc = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach()

        # Sample N candidates in one batched call
        gen_ids_list = av.generate_from_activation_batched(
            h, layer_ell=args.layer,
            n_samples=args.n_samples,
            alpha=args.alpha,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
        # Render each (in-memory PIL images for CLIP)
        candidates = []
        for s, gen_ids in enumerate(gen_ids_list):
            strokes, malformed = av.vocab.decode_tokens_with_stats(gen_ids.tolist())
            if len(strokes) < 8:
                # skip degenerate too-short outputs (CLIP will give junk anyway)
                continue
            img = stroke_render(strokes, display_scale=2.0).convert("RGB")
            candidates.append({"sample": s, "strokes": strokes,
                               "n_strokes": len(strokes), "malformed": malformed,
                               "img": img})

        if not candidates:
            print(f"[clip] {slug}: all candidates degenerate; skipping")
            continue

        # CLIP score each candidate vs the concept text
        imgs = [c["img"] for c in candidates]
        with torch.no_grad():
            inputs = clip_proc(text=[clip_text], images=imgs, return_tensors="pt", padding=True).to("cuda")
            out_clip = clip_model(**inputs)
            # image-text logits per image (B,1)
            logits = out_clip.logits_per_image.squeeze(-1)
        for c, score in zip(candidates, logits.tolist()):
            c["clip_score"] = float(score)

        candidates.sort(key=lambda c: c["clip_score"], reverse=True)
        top = candidates[: args.pick_k]
        elapsed = time.time() - t0
        scores_summary = [round(c["clip_score"], 2) for c in candidates[:5]]
        print(f"[clip] {slug:10s} top CLIP={top[0]['clip_score']:.2f} n_strokes={top[0]['n_strokes']} (top5={scores_summary}, {elapsed:.1f}s)", flush=True)

        # Re-render top-K at full scale + save PNG+MP4
        for rank, cand in enumerate(top):
            suffix = "" if args.pick_k == 1 else f"_top{rank}"
            png_path = args.out_dir / f"{slug}{suffix}.png"
            mp4_path = args.out_dir / f"{slug}{suffix}.mp4" if args.mp4 else None
            img = stroke_render(cand["strokes"], display_scale=args.display_scale,
                                save_animation_path=str(mp4_path) if mp4_path else None, fps=24)
            img.save(png_path)

        if args.save_all:
            ad = args.out_dir / "_all" / slug
            ad.mkdir(parents=True, exist_ok=True)
            for c in candidates:
                c["img"].save(ad / f"s{c['sample']:02d}_clip{c['clip_score']:+.1f}_n{c['n_strokes']:03d}.png")

        summary.append({"slug": slug, "prompt": prompt, "clip_text": clip_text,
                        "best_clip": top[0]["clip_score"],
                        "best_n_strokes": top[0]["n_strokes"],
                        "all_clip_scores": [c["clip_score"] for c in candidates]})

    total = time.time() - t_all
    print(f"\n[clip] total {total:.1f}s", flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[clip] DONE → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
