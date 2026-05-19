"""Plot the FVE/reward/KL trajectory across iterations from iter_log.jsonl.

Output: findings/v1/iter_plot_<layer>.png — a multi-panel chart showing the
recipe's progress per iteration. Used in WRITEUP.md and the README hero image.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True, help="Path to iter_log.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    rows = []
    with open(args.log) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))

    by_phase = defaultdict(list)
    for r in rows:
        by_phase[r.get("phase", "?")].append(r)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    title = args.title or args.log.parent.name
    fig.suptitle(f"v1.0 iterative refinement — {title}", fontsize=14)

    # Panel 1: AR loss across all iters (concatenated)
    ax = axes[0, 0]
    if by_phase["AR"]:
        xs = list(range(len(by_phase["AR"])))
        ys = [r["loss"] for r in by_phase["AR"]]
        # Color by iter
        iters = [r["iter"] for r in by_phase["AR"]]
        unique = sorted(set(iters))
        for it in unique:
            mask = [i for i, x in enumerate(iters) if x == it]
            ax.plot([xs[i] for i in mask], [ys[i] for i in mask], label=f"iter {it}", linewidth=1)
        ax.set_title("AR phase loss (MSE)")
        ax.set_xlabel("global step (across all iterations)")
        ax.set_ylabel("MSE")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    # Panel 2: AV reward EMA across all iters
    ax = axes[0, 1]
    if by_phase["AV"]:
        xs = list(range(len(by_phase["AV"])))
        ys = [r.get("reward_ema", r.get("reward", 0.0)) for r in by_phase["AV"]]
        iters = [r["iter"] for r in by_phase["AV"]]
        for it in sorted(set(iters)):
            mask = [i for i, x in enumerate(iters) if x == it]
            ax.plot([xs[i] for i in mask], [ys[i] for i in mask], label=f"iter {it}", linewidth=1.2)
        ax.set_title("AV phase reward EMA")
        ax.set_xlabel("global step (across all iterations)")
        ax.set_ylabel("reward EMA = -log MSE (EMA)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    # Panel 3: KL across iters
    ax = axes[1, 0]
    if by_phase["AV"]:
        ax.plot([r.get("kl", 0) for r in by_phase["AV"]], color="purple", linewidth=1)
        ax.set_title("AV phase KL(π ‖ π_ref)")
        ax.set_xlabel("AV step")
        ax.set_ylabel("KL")
        ax.grid(alpha=0.3)

    # Panel 4: FVE per iteration
    ax = axes[1, 1]
    if by_phase["EVAL"]:
        iters = [r["iter"] for r in by_phase["EVAL"]]
        fves = [r["fve"] for r in by_phase["EVAL"]]
        cosines = [r["cosine"] for r in by_phase["EVAL"]]
        ax.plot(iters, fves, "o-", color="darkgreen", label="FVE", linewidth=2, markersize=8)
        ax.plot(iters, cosines, "s-", color="darkorange", label="cosine", linewidth=2, markersize=8)
        ax.set_title("Held-out eval per iteration")
        ax.set_xlabel("iteration")
        ax.set_ylabel("metric")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(0.1, color="green", linestyle="--", alpha=0.5, label="FVE target 0.1")
        # annotate best
        if fves:
            best_idx = max(range(len(fves)), key=lambda i: fves[i])
            ax.annotate(f"best FVE = {fves[best_idx]:.3f}", xy=(iters[best_idx], fves[best_idx]),
                        xytext=(8, 8), textcoords="offset points", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
