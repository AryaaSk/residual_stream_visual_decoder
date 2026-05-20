"""Low-temperature single-sample gallery — fast alternative to best-of-N.

For each prompt: extract h, sample one drawing at low temperature with top-k
truncation. ~5 sec per prompt vs ~5 min for best-of-N. Quality often comparable
because the noise that best-of-N filters out comes from high-temp sampling in
the first place.
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


HERO = [
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="google/gemma-4-e2b-it")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=180)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mp4", action="store_true", default=True)
    p.add_argument("--display-scale", type=float, default=4.0)
    p.add_argument("--use-base-caption", action="store_true",
                   help='If set, use "a drawing of a {x}" instead of natural prompts')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[lt] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    device = av.device()

    summary = []
    for slug, prompt in HERO:
        if args.use_base_caption:
            article = "an" if slug in {"apple", "elephant", "umbrella", "airplane"} else "a"
            prompt = f"a drawing of {article} {slug}"
        enc = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach()
        gen_ids = av.generate_from_activation(
            h, layer_ell=args.layer, alpha=args.alpha,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
        strokes, malformed = av.vocab.decode_tokens_with_stats(gen_ids.tolist())
        png_path = args.out_dir / f"{slug}.png"
        mp4_path = args.out_dir / f"{slug}.mp4" if args.mp4 else None
        img = stroke_render(strokes, display_scale=args.display_scale,
                            save_animation_path=str(mp4_path) if mp4_path else None, fps=24)
        img.save(png_path)
        print(f"[lt] {slug:10s} strokes={len(strokes):3d} malformed={malformed} → {png_path.name}", flush=True)
        summary.append({"slug": slug, "prompt": prompt, "n_strokes": len(strokes), "malformed": malformed})

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[lt] DONE → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
