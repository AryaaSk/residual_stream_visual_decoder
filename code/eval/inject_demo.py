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
    # Concrete visual concepts — should produce recognisable silhouettes if the
    # decoder is working
    ("dog", "I am thinking about a dog."),
    ("cat", "I am thinking about a cat."),
    ("bird", "I am picturing a bird flying across the sky."),
    ("fish", "Imagine a fish."),
    ("tree", "I am picturing a tree."),
    ("flower", "Imagine a flower in bloom."),
    ("sun", "The sun is shining."),
    ("house", "I am picturing a small house with a red roof."),
    ("car", "I am thinking about a car."),
    ("airplane", "I am picturing an airplane."),
    # Famous structures — landmark-shaped activations
    ("eiffel", "Paris, the city of lights, is famous for the Eiffel"),
    ("eiffel_short", "The Eiffel Tower is in"),
    ("everest", "Mount Everest is in"),
    # Factual completions — interesting because the "thought" is a fact
    ("capital_france", "The capital of France is"),
    ("capital_japan", "The capital of Japan is"),
    ("largest_planet", "The largest planet in our solar system is"),
    # Geometric concepts — should produce clean shapes
    ("triangle", "Imagine a triangle inscribed in a circle."),
    ("circle", "I am picturing a circle."),
    ("smile_face", "I am picturing a smiling face."),
    # Emotional / abstract — should look qualitatively different from concrete
    ("smile", "She smiled brightly at the surprise."),
    ("storm", "When the storm hit, the village"),
    ("calm", "The lake was calm and"),
    ("sadness", "I am thinking about deep sadness."),
    # Narrative — multi-token thought building
    ("once_upon", "Once upon a time, in a kingdom far away,"),
    ("night_door", "In the middle of the night, the door"),
]

# Subset used for the alpha sweep (cost-controlled — alpha sweep multiplies cost
# by the number of alphas tried)
HERO_PROMPTS = [
    "dog", "cat", "eiffel", "capital_france", "triangle", "smile_face",
]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--av-ckpt", type=Path, required=True)
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)  # tuned via alpha_sweep on Stage-1 AV at L16
    parser.add_argument("--out-dir", type=Path, default=Path("findings/inject_demo"))
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mp4", action="store_true", default=False)
    parser.add_argument("--alpha-sweep", nargs="*", type=float, default=None,
                        help="If set, also render hero prompts at each of these alphas "
                             "(in addition to the main --alpha pass). e.g. --alpha-sweep 0.3 0.7 1.0")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Samples per prompt (each gets a _s0/_s1/... suffix)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[inject] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()

    device = av.device()

    def _render_one(slug_with_suffix: str, h: torch.Tensor, alpha: float):
        gen_ids = av.generate_from_activation(
            h, layer_ell=args.layer, alpha=alpha,
            max_new_tokens=args.max_tokens, temperature=args.temperature,
        )
        strokes, malformed = av.vocab.decode_tokens_with_stats(gen_ids.tolist())
        png_path = args.out_dir / f"{slug_with_suffix}.png"
        mp4_path = args.out_dir / f"{slug_with_suffix}.mp4" if args.mp4 else None
        img = stroke_render(strokes, save_animation_path=str(mp4_path) if mp4_path else None, fps=24)
        img.save(png_path)
        png_path_4x = args.out_dir / f"{slug_with_suffix}_4x.png"
        img_4x = stroke_render(strokes, display_scale=4.0)
        img_4x.save(png_path_4x)
        if args.mp4:
            mp4_path_4x = args.out_dir / f"{slug_with_suffix}_4x.mp4"
            stroke_render(strokes, display_scale=4.0, save_animation_path=str(mp4_path_4x), fps=24)
        print(f"  → {png_path.name}  strokes={len(strokes)}  tokens={len(gen_ids)}  malformed={malformed}", flush=True)
        return {"slug": slug_with_suffix, "n_strokes": len(strokes),
                "n_tokens": len(gen_ids), "malformed": malformed, "alpha": alpha}

    rows = []
    for slug, text in DEMO_TEXTS:
        print(f"\n[inject] === {slug} === : {text!r}", flush=True)
        enc = av.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach()
        print(f"  h_ℓ={args.layer}  shape={tuple(h.shape)}  norm={float(h.norm()):.2f}", flush=True)

        for s in range(args.n_samples):
            suffix = "" if args.n_samples == 1 else f"_s{s}"
            r = _render_one(f"{slug}{suffix}", h, args.alpha)
            r["text"] = text
            r["sample"] = s
            rows.append(r)

        if args.alpha_sweep and slug in HERO_PROMPTS:
            for alt_alpha in args.alpha_sweep:
                if abs(alt_alpha - args.alpha) < 1e-6:
                    continue
                print(f"  [sweep] alpha={alt_alpha}", flush=True)
                r = _render_one(f"{slug}_a{alt_alpha:.2f}", h, alt_alpha)
                r["text"] = text
                r["sample"] = 0
                rows.append(r)

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
            f"<small><b>{row['slug']}</b> (α={row['alpha']:.2f})<br>{row['text']}<br>"
            f"strokes={row['n_strokes']}, malformed={row['malformed']}</small></div>"
        )
    parts.append("</div></body></html>")
    (args.out_dir / "index.html").write_text("".join(parts))
    print(f"\n[inject] index: {args.out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
