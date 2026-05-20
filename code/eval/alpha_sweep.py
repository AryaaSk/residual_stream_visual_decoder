"""Phase A1: sweep the activation-injection scale (alpha) to find the right magnitude.

The activation has norm ~70 at L16 while a typical Gemma 4 embedding has norm ~10.
With alpha=1.0 we inject 7x the magnitude the model expects. Downstream layers
likely saturate. NLA learned this as a parameter; we hand-tune it here, then
optionally lock the best value as the default.

For each alpha in a chosen grid: extract h_ℓ for a fixed set of demo prompts,
run inject_demo-style generation, save artefacts under a per-alpha subdirectory,
and emit a summary (stroke count, malformation rate, variance across prompts).

Usage:
    python code/eval/alpha_sweep.py --av-ckpt checkpoints/av_sft/final --layer 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render import render as stroke_render  # noqa: E402
from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402

DEMO_TEXTS = [
    ("capital_france", "The capital of France is"),
    ("dog", "I am thinking about a dog."),
    ("triangle", "Imagine a triangle inscribed in a circle."),
    ("storm", "When the storm hit, the village"),
    ("paris", "Paris, the city of lights, is famous for the Eiffel"),
    ("math", "What is 47 + 38? The answer is"),
]

DEFAULT_ALPHAS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--av-ckpt", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=Path("findings/alpha_sweep"))
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[alpha_sweep] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    device = av.device()

    # Extract h_ℓ once per prompt (alpha doesn't affect extraction)
    print(f"[alpha_sweep] extracting activations at layer {args.layer}", flush=True)
    hs: list[torch.Tensor] = []
    for slug, text in DEMO_TEXTS:
        enc = av.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach().clone()
        hs.append(h)
    print(f"  activation norms: {[float(h.norm()) for h in hs]}", flush=True)

    summary: list[dict] = []
    for alpha in args.alphas:
        alpha_dir = args.out_dir / f"alpha_{alpha:.2f}"
        alpha_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[alpha_sweep] === alpha={alpha} ===", flush=True)
        rows = []
        all_stroke_counts: list[int] = []
        for (slug, text), h in zip(DEMO_TEXTS, hs):
            gen_ids = av.generate_from_activation(
                h, layer_ell=args.layer, alpha=alpha,
                max_new_tokens=args.max_tokens, temperature=args.temperature,
            )
            strokes, malformed = av.vocab.decode_tokens_with_stats(gen_ids.tolist())
            png = stroke_render(strokes)
            png.save(alpha_dir / f"{slug}.png")
            png_4x = stroke_render(strokes, display_scale=4.0)
            png_4x.save(alpha_dir / f"{slug}_4x.png")
            rows.append({"slug": slug, "text": text, "strokes": len(strokes),
                         "tokens": len(gen_ids), "malformed": malformed})
            all_stroke_counts.append(len(strokes))
            print(f"  {slug:20s}  strokes={len(strokes):3d}  malformed={malformed:3d}", flush=True)

        n = max(1, len(all_stroke_counts))
        mean_strokes = sum(all_stroke_counts) / n
        var_strokes = sum((c - mean_strokes) ** 2 for c in all_stroke_counts) / n
        summary.append({
            "alpha": alpha,
            "mean_strokes": mean_strokes,
            "variance_strokes": var_strokes,
            "mean_malformation_ratio": sum(r["malformed"] for r in rows) / max(1, sum(r["tokens"] for r in rows)),
            "rows": rows,
        })
        # Per-alpha HTML for quick visual sweep
        with open(alpha_dir / "index.html", "w") as f:
            f.write(f"<h2>alpha={alpha}</h2><div style='display:flex;flex-wrap:wrap;gap:1em'>")
            for r in rows:
                f.write(
                    f"<div style='border:1px solid #ddd;padding:.5em;width:240px;text-align:center'>"
                    f"<img src='{r['slug']}_4x.png' style='width:200px'><br>"
                    f"<small><b>{r['slug']}</b><br>{r['text']}<br>"
                    f"strokes={r['strokes']} malformed={r['malformed']}</small></div>"
                )
            f.write("</div>")

    # Top-level summary
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[alpha_sweep] summary:")
    for s in summary:
        print(
            f"  alpha={s['alpha']:>6.2f}  mean_strokes={s['mean_strokes']:>5.1f}  "
            f"var_strokes={s['variance_strokes']:>6.1f}  malf_ratio={s['mean_malformation_ratio']:>5.3f}"
        )

    # Build a combined HTML grid (rows = prompts, cols = alphas)
    rows_html = []
    for i, (slug, text) in enumerate(DEMO_TEXTS):
        cells = []
        for s in summary:
            alpha = s["alpha"]
            cells.append(
                f"<td style='text-align:center;font-size:11px;border:1px solid #ddd'>"
                f"<img src='alpha_{alpha:.2f}/{slug}_4x.png' style='width:140px'><br>"
                f"α={alpha} <br>str={s['rows'][i]['strokes']}</td>"
            )
        rows_html.append(f"<tr><td><b>{slug}</b><br><small>{text}</small></td>{''.join(cells)}</tr>")
    grid_html = (
        "<!doctype html><html><body><h1>Alpha sweep grid</h1><table style='border-collapse:collapse'>"
        f"<tr><th>prompt</th>{''.join('<th>α=' + str(s['alpha']) + '</th>' for s in summary)}</tr>"
        f"{''.join(rows_html)}</table></body></html>"
    )
    (args.out_dir / "grid.html").write_text(grid_html)
    print(f"\n[alpha_sweep] wrote {args.out_dir / 'grid.html'}")


if __name__ == "__main__":
    main()
