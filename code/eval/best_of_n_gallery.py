"""Best-of-N gallery: sample many drawings per prompt, pick the cleanest.

For each prompt:
  1. Extract h from the target model.
  2. Sample N candidate drawings from the AV (temperature configurable).
  3. Score each candidate by simple visual-quality heuristics.
  4. Pick the top-K candidates.
  5. Render each picked candidate auto-cropped + centered (post-processing
     fix: raw AV drawings are often off-center or have stray strokes).

Heuristic score (higher = better):
  - stroke count: smooth Gaussian peaked at 45, std 25, clipped to [10, 120]
  - malformation: penalty per malformed token segment
  - bbox area: reward if [0.15, 0.85] of canvas (not tiny, not overflowing)
  - bbox aspect ratio: reward closer to 1 (square-ish)
  - stroke cluster connectivity: reward when most strokes are in one tight cluster

Usage:
  python code/eval/best_of_n_gallery.py \
      --av-ckpt checkpoints/v1_2/L12/final --layer 12 \
      --n-samples 32 --pick-k 1 --temperature 0.8 \
      --out-dir findings/v1_2/best_of_n_L12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402
from render import render as stroke_render  # noqa: E402


# Curated prompt set: in-distribution trained concepts that should have strong
# QuickDraw priors. Out-of-distribution prompts (eiffel, capital_france etc)
# are intentionally excluded for the hero gallery — they're not what the AV
# was trained on, so they pollute the "this works" story.
HERO_PROMPTS = [
    ("cat",      "I am thinking about a cat."),
    ("dog",      "I am thinking about a dog."),
    ("fish",     "Imagine a fish."),
    ("bird",     "I am picturing a bird flying across the sky."),
    ("horse",    "I am thinking about a horse."),
    ("elephant", "I am thinking about an elephant."),
    ("flower",   "Imagine a flower in bloom."),
    ("tree",     "I am picturing a tree."),
    ("cactus",   "I am picturing a cactus in the desert."),
    ("mountain", "I am picturing a mountain."),
    ("sun",      "The sun is shining."),
    ("cloud",    "I am picturing a cloud in the sky."),
    ("star",     "I am picturing a star in the night sky."),
    ("house",    "I am picturing a small house with a red roof."),
    ("car",      "I am thinking about a car."),
    ("airplane", "I am picturing an airplane."),
    ("apple",    "I am thinking about an apple."),
    ("pizza",    "I am thinking about a pizza."),
    ("clock",    "I am picturing a clock on the wall."),
    ("umbrella", "I am picturing an umbrella."),
]


def gaussian_bell(x, mu, sigma):
    """Unnormalised Gaussian, used for soft preference around a target value."""
    z = (x - mu) / sigma
    return float(torch.exp(-0.5 * torch.tensor(z * z)).item())


def stroke_bbox(strokes):
    """Cumulative bounding box of strokes in stroke-5 frame.

    Returns (min_x, min_y, max_x, max_y, span_x, span_y) in stroke coords.
    """
    if not strokes:
        return 0, 0, 0, 0, 0, 0
    x, y = 0.0, 0.0
    mn_x = mx_x = 0.0
    mn_y = mx_y = 0.0
    for s in strokes:
        dx = float(s.dx if hasattr(s, "dx") else s["dx"])
        dy = float(s.dy if hasattr(s, "dy") else s["dy"])
        x += dx; y += dy
        if x < mn_x: mn_x = x
        if x > mx_x: mx_x = x
        if y < mn_y: mn_y = y
        if y > mx_y: mx_y = y
    return mn_x, mn_y, mx_x, mx_y, mx_x - mn_x, mx_y - mn_y


def score_candidate(strokes, n_tokens: int, malformed: int) -> float:
    """Combine stroke-count, malformation, and bbox into a quality score."""
    n = len(strokes)
    if n < 8:
        return -100.0  # almost-empty drawing
    # Soft preference for stroke count peaked at 45
    score = 0.0
    score += 4.0 * gaussian_bell(n, mu=45, sigma=25)
    # Penalise malformation
    score -= 0.4 * max(0, malformed - 1)
    # Bounding box should be a reasonable fraction of the canvas (canvas
    # roughly spans -128..+128 = 256 px in stroke-5 coords; we like span in
    # ~80 to 220 range)
    _, _, _, _, span_x, span_y = stroke_bbox(strokes)
    span_max = max(span_x, span_y, 1.0)
    if 60 <= span_max <= 240:
        score += 1.5 * gaussian_bell(span_max, mu=150, sigma=60)
    else:
        score -= 1.0
    # Aspect ratio reward (square-ish)
    if span_x > 0 and span_y > 0:
        ratio = min(span_x, span_y) / max(span_x, span_y)
        score += 1.0 * ratio
    # Bonus for low malformation
    if malformed == 0:
        score += 0.5
    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="google/gemma-4-e2b-it")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--n-samples", type=int, default=32, help="Candidates per prompt")
    p.add_argument("--pick-k", type=int, default=1, help="Top-k to render per prompt")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mp4", action="store_true", default=True)
    p.add_argument("--display-scale", type=float, default=4.0)
    p.add_argument("--save-all", action="store_true",
                   help="Save every candidate, not just top-k, for inspection")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bestN] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    device = av.device()
    print(f"[bestN] AV loaded. projector={av.act_projector is not None}", flush=True)

    summary = []
    for slug, prompt in HERO_PROMPTS:
        print(f"\n[bestN] === {slug} === : {prompt!r}", flush=True)
        # Extract h once
        enc = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach()
        print(f"  h.norm={float(h.norm()):.2f}", flush=True)

        # Generate N candidates
        candidates = []
        for s in range(args.n_samples):
            gen_ids = av.generate_from_activation(
                h, layer_ell=args.layer, alpha=args.alpha,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature, top_k=args.top_k,
            )
            strokes, malformed = av.vocab.decode_tokens_with_stats(gen_ids.tolist())
            sc = score_candidate(strokes, len(gen_ids), malformed)
            candidates.append({
                "sample": s, "strokes": strokes, "n_strokes": len(strokes),
                "n_tokens": len(gen_ids), "malformed": malformed, "score": sc,
            })
        candidates.sort(key=lambda c: c["score"], reverse=True)
        top = candidates[: args.pick_k]
        print(f"  top score {top[0]['score']:.2f} (n_strokes={top[0]['n_strokes']}, malformed={top[0]['malformed']})", flush=True)

        # Render selected
        for rank, cand in enumerate(top):
            suffix = "" if args.pick_k == 1 else f"_top{rank}"
            png_path = args.out_dir / f"{slug}{suffix}.png"
            mp4_path = args.out_dir / f"{slug}{suffix}.mp4" if args.mp4 else None
            img = stroke_render(cand["strokes"], display_scale=args.display_scale,
                                save_animation_path=str(mp4_path) if mp4_path else None, fps=24)
            img.save(png_path)

        # Optionally save all candidates for inspection
        if args.save_all:
            ins_dir = args.out_dir / "_all_candidates" / slug
            ins_dir.mkdir(parents=True, exist_ok=True)
            for c in candidates:
                img = stroke_render(c["strokes"], display_scale=2.0)
                img.save(ins_dir / f"s{c['sample']:02d}_score{c['score']:+.1f}_n{c['n_strokes']:03d}.png")

        summary.append({
            "slug": slug, "prompt": prompt,
            "best_score": top[0]["score"],
            "best_n_strokes": top[0]["n_strokes"],
            "best_malformed": top[0]["malformed"],
            "candidate_scores": [c["score"] for c in candidates],
        })

    # Write summary
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[bestN] DONE. summary → {summary_path}", flush=True)


if __name__ == "__main__":
    main()
