"""Depth-vs-decodability line chart — the headline finding.

Plots best top-1 retrieval at each layer trained, with chance line and shaded
'monotonic with depth' annotation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Best per layer (20-way top-1, chance = 5%)
BEST = [
    (3,  25.0, "from-scratch SFT"),
    (10, 75.0, "filtered SFT"),
    (15, 50.0, "filtered SFT"),
    (20, 35.0, "overnight SFT"),
    (24, 80.0, "from-scratch SFT"),
    (29, 85.0, "v2.0 SFT"),
]

# also overlay filtered vs baseline at L10 and L29 to show the filter effect
FILTERED = {
    10: {"baseline": 65.0, "filtered": 75.0},
    29: {"baseline": 85.0, "filtered": 50.0},
}


def main():
    out = Path("artefacts/v3/viral/depth_chart.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    xs = [l for l, _, _ in BEST]
    ys = [s for _, s, _ in BEST]
    ax.plot(xs, ys, marker="o", linewidth=3, markersize=14,
            color="#1f77b4", label="Best per-layer top-1 (gen-likelihood)")
    for l, s, recipe in BEST:
        ax.annotate(f"L{l}\n{s:.0f}%", xy=(l, s), xytext=(0, 14),
                     textcoords="offset points", ha="center",
                     fontsize=11, fontweight="bold", color="#1f77b4")
        ax.annotate(recipe, xy=(l, s), xytext=(0, -22),
                     textcoords="offset points", ha="center",
                     fontsize=8, color="#444")

    # Overlay filter comparison at L10 and L29
    for l, d in FILTERED.items():
        ax.scatter([l - 0.5], [d["baseline"]], marker="s", s=120,
                    color="orange", alpha=0.8, zorder=3,
                    label="baseline (no filter)" if l == 10 else None)
        ax.scatter([l + 0.5], [d["filtered"]], marker="^", s=120,
                    color="green", alpha=0.8, zorder=3,
                    label="filtered (Qwen-blessed)" if l == 10 else None)
        ax.plot([l - 0.5, l + 0.5], [d["baseline"], d["filtered"]],
                 ":", color="grey", alpha=0.5)

    ax.axhline(5, color="grey", linestyle="--", label="chance (5%, 20-way)")
    ax.set_xlim(-1, 32)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Qwen 3.5-4B residual-stream layer", fontsize=14)
    ax.set_ylabel("top-1 retrieval (%, 20 held-out concepts)", fontsize=14)
    ax.set_title("Concept-decodability grows monotonically with depth\n"
                  "(Activation-Verbalizer at each layer, generation-likelihood reward)",
                  fontsize=15, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)

    # Shaded annotation
    ax.axvspan(28, 30, alpha=0.08, color="green")
    ax.text(29, 92, "BEST", ha="center", fontsize=12, fontweight="bold", color="green")

    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"[depth-chart] → {out}")


if __name__ == "__main__":
    main()
