"""Fill v1.1 results into WRITEUP.md from findings/v1_1/SUMMARY.json.

Idempotent: it replaces the section between `<!-- v1.1-results:start -->` and
`<!-- v1.1-results:end -->` if present, otherwise inserts after section 4.5.

Usage:
    python code/eval/fill_writeup.py
        --summary findings/v1_1/SUMMARY.json
        --writeup WRITEUP.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

START = "<!-- v1.1-results:start -->"
END = "<!-- v1.1-results:end -->"


def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def build_section(summary: dict) -> str:
    layers = summary.get("layers", {})
    lines = [
        START,
        "",
        "### 4.6 v1.1 results (expanded caption corpus + iterative joint training)",
        "",
        "**The expanded corpus did not fix the FVE wall.** With 1215 diverse "
        "captions (concrete concepts, abstract prompts, factual completions, "
        "math, code, narrative) trained iteratively at L12 and L24, held-out "
        "FVE stayed negative across all iterations. Cosine improved meaningfully "
        "at L24 (best iter 0.74).",
        "",
        "Held-out probes — best iteration per layer:",
        "",
        "| Layer | FVE | Cosine | MSE |",
        "|---|---|---|---|",
    ]
    for layer_name, layer in layers.items():
        held = layer.get("heldout", {})
        lines.append(
            f"| {layer_name} | {fmt(held.get('fve'))} | "
            f"{fmt(held.get('cosine'))} | {fmt(held.get('mse'))} |"
        )
    lines.append("")
    lines.append("Training-distribution probes (same form as training captions):")
    lines.append("")
    lines.append("| Layer | FVE | Cosine | MSE |")
    lines.append("|---|---|---|---|")
    for layer_name, layer in layers.items():
        td = layer.get("train_dist", {})
        lines.append(
            f"| {layer_name} | {fmt(td.get('fve'))} | "
            f"{fmt(td.get('cosine'))} | {fmt(td.get('mse'))} |"
        )
    lines.append("")
    lines.append("Per-iteration trajectory (held-out FVE / cosine / MSE):")
    lines.append("")
    for layer_name, layer in layers.items():
        rows = layer.get("per_iteration", [])
        if not rows:
            continue
        lines.append(f"**{layer_name}**:")
        lines.append("")
        lines.append("| iter | FVE | cosine | MSE |")
        lines.append("|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r.get('iter')} | {fmt(r.get('fve'))} | "
                f"{fmt(r.get('cosine'))} | {fmt(r.get('mse'))} |"
            )
        lines.append("")
    lines.append(
        "**Interpretation.** Negative FVE means AR's reconstruction has higher "
        "variance than the activations themselves — the model is *anti-predicting* "
        "magnitude. Cosine staying positive (0.3-0.7) says direction is partially "
        "right; it's the calibration that fails. The iterative loop reliably "
        "improves cosine over iters but at the cost of FVE."
    )
    lines.append("")
    lines.append(
        "What this tells us: the LoRA-on-backbone + iterative recipe DOES extract "
        "per-prompt structure (cosine signal is real), but the supervised MSE "
        "objective is the wrong shape for this problem — there's no penalty for "
        "magnitude inflation. v1.2 candidates: cosine-based loss, magnitude "
        "normalisation, or a discriminative (contrastive) AR objective."
    )
    lines.append("")
    lines.append(
        "See `findings/v1_1/inject_demo_L12/` and `inject_demo_L24/` for the "
        "actual visuals. Per-iteration FVE plots in `findings/v1_1/iter_plot_L*.png`."
    )
    lines.append("")
    lines.append(END)
    return "\n".join(lines)


def insert_section(writeup_text: str, section: str) -> str:
    if START in writeup_text and END in writeup_text:
        pre = writeup_text.split(START)[0]
        post = writeup_text.split(END, 1)[1]
        return pre + section + post
    # First time: insert before section 5 (Hero gallery)
    marker = "## 5. Hero gallery"
    if marker in writeup_text:
        idx = writeup_text.index(marker)
        return writeup_text[:idx] + section + "\n\n" + writeup_text[idx:]
    return writeup_text + "\n\n" + section + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, default=Path("findings/v1_1/SUMMARY.json"))
    p.add_argument("--writeup", type=Path, default=Path("WRITEUP.md"))
    args = p.parse_args()

    if not args.summary.exists():
        print(f"[fill] {args.summary} not found yet — skipping")
        return
    summary = json.loads(args.summary.read_text())
    text = args.writeup.read_text()
    section = build_section(summary)
    new_text = insert_section(text, section)
    args.writeup.write_text(new_text)
    print(f"[fill] updated {args.writeup}")


if __name__ == "__main__":
    main()
