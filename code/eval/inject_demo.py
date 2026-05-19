"""End-to-end demo: extract a Gemma 4 activation, inject it into the AV,
sample a drawing, render. NO training of AR or RL — just shows the
activation-injection plumbing works.

Use to validate that the full pipeline runs before / instead of Stage 3.

Usage:
    python code/eval/inject_demo.py --av-ckpt checkpoints/av_sft/final --layer 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402
from render import render as stroke_render  # noqa: E402

DEMO_TEXTS = [
    ("capital_france", "The capital of France is"),
    ("smile", "She smiled brightly at the surprise."),
    ("dog", "I am thinking about a dog."),
    ("triangle", "Imagine a triangle inscribed in a circle."),
    ("storm", "When the storm hit, the village"),
    ("paris", "Paris, the city of lights, is famous for the Eiffel"),
]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--av-ckpt", type=Path, required=True)
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=Path("findings/inject_demo"))
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mp4", action="store_true", default=False)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[inject] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()

    device = av.device()

    rows = []
    for slug, text in DEMO_TEXTS:
        print(f"\n[inject] === {slug} === : {text!r}", flush=True)
        enc = av.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach()
        print(f"  h_ℓ={args.layer}  shape={tuple(h.shape)}  norm={float(h.norm()):.2f}", flush=True)

        gen_ids = av.generate_from_activation(
            h, layer_ell=args.layer, alpha=args.alpha,
            max_new_tokens=args.max_tokens, temperature=args.temperature,
        )
        strokes, malformed = av.vocab.decode_tokens_with_stats(gen_ids.tolist())
        png_path = args.out_dir / f"{slug}.png"
        mp4_path = args.out_dir / f"{slug}.mp4" if args.mp4 else None
        img = stroke_render(strokes, save_animation_path=str(mp4_path) if mp4_path else None, fps=24)
        img.save(png_path)
        print(f"  → {png_path.name}  strokes={len(strokes)}  tokens={len(gen_ids)}  malformed={malformed}", flush=True)
        rows.append({"slug": slug, "text": text, "n_strokes": len(strokes), "n_tokens": len(gen_ids), "malformed": malformed})

    # HTML index
    parts = ["<!doctype html><html><body><h2>Activation-Injection demo (layer "
             + str(args.layer) + ", AV from " + str(args.av_ckpt) + ")</h2>"
             "<p>Each drawing is sampled from the AV with the activation of the "
             "corresponding text input injected at &lt;ACT_TOKEN&gt;. NO RL training "
             "of faithfulness; this just shows the architecture runs end-to-end.</p>"
             "<div style='display:flex;flex-wrap:wrap;gap:1em'>"]
    for row in rows:
        parts.append(
            f"<div style='border:1px solid #ddd;padding:.5em;width:240px;text-align:center'>"
            f"<img src='{row['slug']}.png' style='width:200px'><br>"
            f"<small><b>{row['slug']}</b><br>{row['text']}<br>"
            f"strokes={row['n_strokes']}, malformed={row['malformed']}</small></div>"
        )
    parts.append("</div></body></html>")
    (args.out_dir / "index.html").write_text("".join(parts))
    print(f"\n[inject] index: {args.out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
