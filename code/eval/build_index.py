"""Build the final Day-1 HTML index page.

Aggregates everything in artefacts/ and findings/ into a single browsable page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def section(title: str, body_html: str) -> str:
    return f"<section style='margin:2em 0'><h2>{title}</h2>{body_html}</section>"


def img_grid(images: list[tuple[str, str]], item_width: int = 220) -> str:
    """images: list of (caption, relative_path) tuples."""
    parts = [f"<div style='display:flex;flex-wrap:wrap;gap:1em'>"]
    for caption, src in images:
        parts.append(
            f"<div style='border:1px solid #ddd;padding:.5em;width:{item_width}px;text-align:center'>"
            f"<img src='{src}' style='width:100%;background:#fff;border:1px solid #eee'><br>"
            f"<small>{caption}</small></div>"
        )
    parts.append("</div>")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("INDEX.html"))
    args = parser.parse_args()

    root = args.root.resolve()
    out_path = args.out if args.out.is_absolute() else root / args.out

    parts: list[str] = []
    parts.append("""<!doctype html><html><head><meta charset='utf-8'>
<title>Residual Stream Visual Decoder — Day 1</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1200px;
         margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.5; }
  h1 { border-bottom: 2px solid #333; padding-bottom: .3em; }
  h2 { color: #555; margin-top: 2em; }
  .meta { color: #666; font-size: 14px; }
  pre, code { background: #f4f4f4; padding: 0 4px; border-radius: 3px; }
  details { margin: 1em 0; }
  summary { cursor: pointer; font-weight: 600; }
  a { color: #06c; }
</style></head><body>""")
    parts.append("<h1>Residual Stream Visual Decoder — Day 1</h1>")
    parts.append("<p class='meta'>"
                 "Visual lens into Gemma 4 E2B's residual stream. NLA-style autoencoder, "
                 "stroke-token output, vision-pathway reconstruction. See "
                 "<a href='README.md'>README</a> + 7 design docs in repo root.</p>")

    # Day-0 alignment
    day0_png = root / "findings" / "day0_alignment.png"
    day0_json = root / "findings" / "day0_alignment.json"
    if day0_png.exists() and day0_json.exists():
        data = json.loads(day0_json.read_text())
        delta_str = ", ".join(f"L{i:02d}={d:.4f}" for i, d in enumerate(data['delta_per_layer']))
        body = (
            f"<p>Cross-modal alignment (cosine sim of text vs image residuals) per layer. "
            f"Best layer = {data['best_layer']} (delta {data['best_delta']:.4f}).</p>"
            f"<img src='findings/day0_alignment.png' style='max-width:800px'>"
            f"<details><summary>Per-layer delta numbers</summary><pre>{delta_str}</pre></details>"
        )
        parts.append(section("Day-0: cross-modal alignment", body))

    # Stage 1 samples
    stage1_dir = root / "findings" / "stage1_samples"
    if stage1_dir.exists():
        imgs = sorted(stage1_dir.glob("*.png"))
        if imgs:
            tiles = [(p.stem, str(p.relative_to(root))) for p in imgs]
            body = (
                "<p>AV (Activation Verbalizer) trained on QuickDraw text→strokes pairs. "
                "<b>NO activation injection</b>: these are text-conditioned drawings of named concepts. "
                "Validates that the AV emits real strokes after Stage-1 SFT.</p>"
                + img_grid(tiles)
            )
            parts.append(section("Stage 1 samples (text → strokes)", body))

    # Inject demo
    inject_dir = root / "findings" / "inject_demo"
    if inject_dir.exists():
        imgs = sorted(inject_dir.glob("*.png"))
        if imgs:
            tiles = [(p.stem, str(p.relative_to(root))) for p in imgs]
            body = (
                "<p>End-to-end activation-injection demo. For each text, "
                "(1) extract h_ℓ at layer 16 from Gemma 4, "
                "(2) inject h_ℓ at &lt;ACT_TOKEN&gt; embedding, "
                "(3) AV autoregressively samples stroke tokens, "
                "(4) renderer produces PNG. NO Stage-3 RL faithfulness training "
                "(without RL the drawings are not yet predictive of the activation; "
                "they show the plumbing works).</p>"
                + img_grid(tiles)
            )
            parts.append(section("Activation-injection end-to-end demo (layer 16)", body))

    # Probe sweep
    probe_root = root / "artefacts" / "per_probe"
    if probe_root.exists():
        layer_dirs = sorted([d for d in probe_root.iterdir() if d.is_dir() and d.name.startswith("L")])
        for d in layer_dirs:
            png_dir = d / "png"
            if not png_dir.exists():
                continue
            imgs = sorted(png_dir.glob("*.png"))
            if not imgs:
                continue
            tiles = [(p.stem, str(p.relative_to(root))) for p in imgs]
            body = (
                f"<p>{len(tiles)} probes × layer {d.name}. "
                f"<a href='{d.relative_to(root)}/index.html'>full per-probe index with MP4 animations</a></p>"
                + img_grid(tiles[:24], item_width=160)
                + (f"<p><small>...and {len(tiles) - 24} more</small></p>" if len(tiles) > 24 else "")
            )
            parts.append(section(f"Probe sweep ({d.name})", body))

    # Research log
    log_dir = root / "research_log"
    if log_dir.exists():
        logs = sorted(log_dir.glob("*.md"))
        if logs:
            links = "<br>".join(f"<a href='{l.relative_to(root)}'>{l.name}</a>" for l in logs)
            parts.append(section("Research log", links))

    # Footer
    parts.append("<hr><p class='meta'>Generated by code/eval/build_index.py</p>")
    parts.append("</body></html>")

    out_path.write_text("".join(parts))
    print(f"[build_index] wrote {out_path}")


if __name__ == "__main__":
    main()
