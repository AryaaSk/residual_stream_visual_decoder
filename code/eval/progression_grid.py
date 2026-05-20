"""Build a "progression across checkpoints" grid.

Rows = checkpoints (step_005000, step_010000, ..., final).
Cols = concepts (cat, dog, fish, ...).
Each cell shows the top-1 best-of-N drawing for that concept at that checkpoint.

Use this to inspect how the AV's drawings evolve across the 30K training steps.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CONCEPTS = [
    "cat", "dog", "fish", "bird", "horse", "elephant", "flower", "tree",
    "cactus", "mountain", "sun", "cloud", "star", "house", "car", "airplane",
    "apple", "pizza", "clock", "umbrella",
]


def get_font(size: int):
    for c in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--findings-root", type=Path, default=Path("findings/v1_3"))
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--out", type=Path, default=Path("artefacts/v1_3/progression_L12.png"))
    p.add_argument("--cell", type=int, default=180)
    p.add_argument("--gap", type=int, default=8)
    p.add_argument("--row-label-w", type=int, default=120)
    p.add_argument("--col-label-h", type=int, default=30)
    p.add_argument("--title-h", type=int, default=60)
    p.add_argument("--concept-cap", type=int, default=12, help="max concepts per grid")
    args = p.parse_args()

    # Find checkpoints
    pattern = re.compile(rf"fastN_L{args.layer}_(step\d+|final)")
    ckpt_dirs = sorted(
        [d for d in args.findings_root.iterdir() if d.is_dir() and pattern.match(d.name)],
        key=lambda d: (0, d.name) if "final" not in d.name else (1, "final"),
    )
    if not ckpt_dirs:
        print(f"[grid] no fastN_L{args.layer}_* dirs found in {args.findings_root}")
        return
    print(f"[grid] {len(ckpt_dirs)} checkpoints: {[d.name for d in ckpt_dirs]}")

    concepts = CONCEPTS[:args.concept_cap]
    cell = args.cell
    gap = args.gap
    row_label_w = args.row_label_w
    col_label_h = args.col_label_h
    title_h = args.title_h

    cols = len(concepts)
    rows = len(ckpt_dirs)

    width = row_label_w + cols * cell + (cols + 1) * gap
    height = title_h + col_label_h + rows * cell + (rows + 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(28)
    label_font = get_font(14)
    row_font = get_font(18)

    title = f"v1.3 progression — L{args.layer}, best-of-16 at each checkpoint"
    tw = draw.textlength(title, font=title_font)
    draw.text(((width - tw) / 2, 18), title, fill="black", font=title_font)

    # Column labels (concepts)
    for c, concept in enumerate(concepts):
        x = row_label_w + gap + c * (cell + gap)
        cw = draw.textlength(concept, font=label_font)
        draw.text((x + (cell - cw) / 2, title_h + 4), concept, fill="#333333", font=label_font)

    # Rows
    for r, ckpt_dir in enumerate(ckpt_dirs):
        # Row label
        label = ckpt_dir.name.replace(f"fastN_L{args.layer}_", "")
        if label.startswith("step"):
            n = int(label.replace("step", ""))
            label = f"step {n//1000}K"
        y = title_h + col_label_h + gap + r * (cell + gap)
        draw.text((4, y + cell // 2 - 8), label, fill="black", font=row_font)
        for c, concept in enumerate(concepts):
            x = row_label_w + gap + c * (cell + gap)
            png_path = ckpt_dir / f"{concept}_top0.png"
            if png_path.exists():
                img = Image.open(png_path).convert("RGB")
                img.thumbnail((cell, cell))
                cx = x + (cell - img.width) // 2
                cy = y + (cell - img.height) // 2
                canvas.paste(img, (cx, cy))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"[grid] wrote {args.out}  size={canvas.size}")


if __name__ == "__main__":
    main()
