"""Best-of-N gallery using BASE prompts: "a drawing of a {concept}".

This tests the upper bound of what v1.2 can do: when we prompt with the EXACT
caption template the AV saw most during Stage 1.5 training, the activation
should map cleanest. Compares to best_of_n_gallery.py (which uses diverse
natural-language prompts like "I am thinking about a cat").
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
from eval.best_of_n_gallery import score_candidate  # noqa: E402


# Use the EXACT base caption from sft_quickdraw.jsonl. These are the
# in-distribution activations the AV saw most often during Stage 1.5 SFT.
BASE_CONCEPTS = [
    "cat", "dog", "fish", "bird", "horse", "elephant", "spider", "snake",
    "apple", "banana", "carrot", "pizza", "donut", "cookie",
    "tree", "flower", "leaf", "mushroom", "cactus",
    "mountain", "cloud", "sun", "moon", "star", "rainbow",
    "house", "bridge", "tent",
    "car", "bicycle", "train", "truck",
    "book", "pencil", "scissors", "key", "clock", "umbrella",
    "chair", "table", "bed", "lamp", "door",
]
VOWEL_CONCEPTS = {"apple", "umbrella", "elephant"}


def base_caption(c: str) -> str:
    article = "an" if c in VOWEL_CONCEPTS else "a"
    return f"a drawing of {article} {c}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="google/gemma-4-e2b-it")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--display-scale", type=float, default=4.0)
    p.add_argument("--mp4", action="store_true", default=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bestN-base] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    device = av.device()

    summary = []
    for concept in BASE_CONCEPTS:
        prompt = base_caption(concept)
        print(f"\n[bestN-base] === {concept} === : {prompt!r}", flush=True)
        enc = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach()

        candidates = []
        for s in range(args.n_samples):
            gen_ids = av.generate_from_activation(
                h, layer_ell=args.layer, alpha=args.alpha,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature, top_k=args.top_k,
            )
            strokes, malformed = av.vocab.decode_tokens_with_stats(gen_ids.tolist())
            sc = score_candidate(strokes, len(gen_ids), malformed)
            candidates.append({"sample": s, "strokes": strokes,
                               "n_strokes": len(strokes), "malformed": malformed, "score": sc})
        candidates.sort(key=lambda c: c["score"], reverse=True)
        top = candidates[0]
        print(f"  best score {top['score']:.2f} n_strokes={top['n_strokes']}", flush=True)

        # Render top-1
        png_path = args.out_dir / f"{concept}.png"
        img = stroke_render(top["strokes"], display_scale=args.display_scale,
                            save_animation_path=str(args.out_dir / f"{concept}.mp4") if args.mp4 else None,
                            fps=24)
        img.save(png_path)

        summary.append({"concept": concept, "prompt": prompt,
                        "best_score": top["score"], "best_n_strokes": top["n_strokes"]})

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[bestN-base] DONE. summary → {summary_path}", flush=True)


if __name__ == "__main__":
    main()
