"""Fast best-of-N gallery using batched sampling.

Uses StrokeDecoder.generate_from_activation_batched to draw N candidates per
prompt in a single batched forward pass. ~10x faster than sequential best-of-N.

For each prompt:
  1. Extract h from target model (1 forward pass).
  2. Sample N candidates in a single batched generation (1 batched call).
  3. Score each by heuristics, pick top-K.
  4. Render top-K to PNG + MP4.

Heuristic score uses the same `score_candidate` from best_of_n_gallery.py.
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
from eval.best_of_n_gallery import score_candidate, HERO_PROMPTS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--pick-k", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--display-scale", type=float, default=4.0)
    p.add_argument("--mp4", action="store_true", default=True)
    p.add_argument("--save-all", action="store_true",
                   help="Save every candidate in _all/<slug>/ for inspection")
    p.add_argument("--prompts", choices=["hero", "base", "both"], default="hero",
                   help="hero=natural prompts; base='a drawing of a {X}'; both=combined")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fastN] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    device = av.device()
    print(f"[fastN] AV loaded. projector={av.act_projector is not None}", flush=True)

    # Build the prompt set
    prompts_to_run: list[tuple[str, str]] = []
    if args.prompts in ("hero", "both"):
        prompts_to_run.extend(HERO_PROMPTS)
    if args.prompts in ("base", "both"):
        VOWELS = {"apple", "umbrella", "elephant", "airplane"}
        for slug, _ in HERO_PROMPTS:
            article = "an" if slug in VOWELS else "a"
            prompts_to_run.append((f"base_{slug}", f"a drawing of {article} {slug}"))

    summary = []
    import time
    t0 = time.time()
    for idx, (slug, prompt) in enumerate(prompts_to_run):
        print(f"\n[fastN] {idx+1}/{len(prompts_to_run)} === {slug} === : {prompt!r}", flush=True)
        # Extract h
        enc = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach()

        # Batched sampling
        t_start = time.time()
        gen_ids_list = av.generate_from_activation_batched(
            h, layer_ell=args.layer,
            n_samples=args.n_samples,
            alpha=args.alpha,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
        elapsed = time.time() - t_start

        # Score
        candidates = []
        for s, gen_ids in enumerate(gen_ids_list):
            strokes, malformed = av.vocab.decode_tokens_with_stats(gen_ids.tolist())
            sc = score_candidate(strokes, len(gen_ids), malformed)
            candidates.append({"sample": s, "strokes": strokes,
                               "n_strokes": len(strokes), "malformed": malformed,
                               "score": sc, "n_tokens": len(gen_ids)})
        candidates.sort(key=lambda c: c["score"], reverse=True)
        top = candidates[:args.pick_k]
        print(f"  best score {top[0]['score']:.2f}  n_strokes={top[0]['n_strokes']}  malformed={top[0]['malformed']}  ({elapsed:.1f}s for {args.n_samples} samples)", flush=True)

        # Render top-K
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
                img = stroke_render(c["strokes"], display_scale=2.0)
                img.save(ad / f"s{c['sample']:02d}_score{c['score']:+.1f}_n{c['n_strokes']:03d}.png")

        summary.append({"slug": slug, "prompt": prompt,
                        "best_score": top[0]["score"],
                        "best_n_strokes": top[0]["n_strokes"],
                        "candidate_scores": [c["score"] for c in candidates]})

    total = time.time() - t0
    print(f"\n[fastN] total: {total:.1f}s for {len(prompts_to_run)} prompts × {args.n_samples} samples", flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[fastN] DONE → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
