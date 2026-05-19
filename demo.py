#!/usr/bin/env python3
"""One-click demo: turn a text prompt into a drawing of Gemma 4's thought.

Usage
-----
    # On the H200 (where the model + checkpoints are):
    python demo.py "The capital of France is"
    python demo.py "I am thinking about a dog." --layer 12 --display-scale 4
    python demo.py "What is 47 + 38?" --layer 12 --open

What it does
------------
1. Loads Gemma 4 E2B with our trained AV checkpoint at the chosen layer
2. Runs the prompt through Gemma, extracts the residual stream at layer ℓ
3. Injects that activation into the AV via the <ACT_TOKEN> embedding hook
4. Autoregressively samples stroke tokens
5. Renders to PNG + (optional) animated MP4
6. Saves outputs under ./demo_output/<slug>/ and optionally opens them

Files produced (per invocation, under ./demo_output/<slug>/):
    drawing.png         — 224×224 native AR-input
    drawing_4x.png      — 896×896 polished
    drawing_4x.mp4      — 4× animated stroke-by-stroke
    prompt.txt          — the input
    meta.json           — layer, alpha, model id, ckpt path, stats

Reading the result
------------------
Open `demo_output/<slug>/drawing_4x.png` or play the MP4. The drawing is
what Gemma was "thinking" in its residual stream at the chosen layer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

# Allow `python demo.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from render import render as stroke_render  # noqa: E402
from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402


def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return (s[:max_len] or "prompt") + "_" + str(int(time.time()))[-4:]


def main():
    p = argparse.ArgumentParser(description="Turn a text prompt into a drawing of Gemma 4's thought.")
    p.add_argument("prompt", nargs="+", help="The text prompt to visualize")
    p.add_argument("--av-ckpt", type=Path, default=Path("checkpoints/v1/L12/final"),
                   help="Trained AV checkpoint directory")
    p.add_argument("--model-id", default="google/gemma-4-e2b-it")
    p.add_argument("--layer", type=int, default=12, help="Which residual stream layer to decode")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--display-scale", type=float, default=4.0,
                   help="Vector upscale factor for display (PNG + MP4)")
    p.add_argument("--out-dir", type=Path, default=Path("demo_output"))
    p.add_argument("--open", action="store_true", help="Open the resulting drawing automatically")
    p.add_argument("--no-mp4", action="store_true")
    args = p.parse_args()

    prompt = " ".join(args.prompt)
    slug = slugify(prompt)
    out_dir = args.out_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo] prompt: {prompt!r}")
    print(f"[demo] layer:  {args.layer}")
    print(f"[demo] AV:     {args.av_ckpt}")
    print(f"[demo] output: {out_dir}")

    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()

    with torch.no_grad():
        enc = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(av.device())
        out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach()
        print(f"[demo] activation norm at L{args.layer}: {float(h.norm()):.2f}")

        ids = av.generate_from_activation(
            h, layer_ell=args.layer, alpha=args.alpha,
            max_new_tokens=args.max_tokens, temperature=args.temperature,
        )
        strokes, malformed = av.vocab.decode_tokens_with_stats(ids.tolist())
    print(f"[demo] generated {len(ids)} tokens → {len(strokes)} strokes (malformed: {malformed})")

    # Native and 4x outputs
    img_native = stroke_render(strokes)
    img_native.save(out_dir / "drawing.png")
    img_up = stroke_render(strokes, display_scale=args.display_scale)
    img_up.save(out_dir / f"drawing_{int(args.display_scale)}x.png")
    print(f"[demo] wrote drawing.png and drawing_{int(args.display_scale)}x.png")

    if not args.no_mp4:
        mp4_path = out_dir / f"drawing_{int(args.display_scale)}x.mp4"
        stroke_render(strokes, display_scale=args.display_scale,
                      save_animation_path=str(mp4_path), fps=24)
        print(f"[demo] wrote {mp4_path.name}")

    (out_dir / "prompt.txt").write_text(prompt + "\n")
    meta = {
        "prompt": prompt,
        "model_id": args.model_id,
        "av_ckpt": str(args.av_ckpt),
        "layer": args.layer,
        "alpha": args.alpha,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "n_strokes": len(strokes),
        "n_tokens": len(ids),
        "n_malformed": malformed,
        "h_norm": float(h.norm()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n✓ Done. Open with:")
    print(f"    open {out_dir}/drawing_{int(args.display_scale)}x.png")
    if not args.no_mp4:
        print(f"    open {out_dir}/drawing_{int(args.display_scale)}x.mp4")

    if args.open:
        target = out_dir / (f"drawing_{int(args.display_scale)}x.mp4" if not args.no_mp4
                            else f"drawing_{int(args.display_scale)}x.png")
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(target)], check=False)
            elif sys.platform == "linux":
                subprocess.run(["xdg-open", str(target)], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
