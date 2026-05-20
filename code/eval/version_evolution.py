"""Build a 'concept evolution across versions' image.

Rows = concepts (cat, dog, fish, ...)
Cols = versions (v1.1, v1.2, v1.3, v1.4, v1.5)
Shows how each prompt's drawing evolved across project versions.

Useful both as an internal sanity check and as a viral-ready visual that
makes the progression undeniable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CONCEPTS = ["cat", "dog", "fish", "horse", "elephant", "flower", "mountain", "sun"]

# (version label, image dir, suffix used in filename)
VERSIONS = [
    ("v1.1", "findings/v1_1/inject_demo_L24", "_4x"),
    ("v1.2", "findings/v1_2/inject_demo_L12", "_4x"),
    ("v1.3", "findings/v1_2/clip_L12", ""),  # v1.3 was just CLIP-ranking on v1.2 ckpts
    ("v1.4", "findings/v1_4/clip_L12_step005000", ""),
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
    p.add_argument("--out", type=Path, default=Path("artefacts/v1_5/evolution.png"))
    p.add_argument("--cell", type=int, default=220)
    p.add_argument("--gap", type=int, default=10)
    p.add_argument("--col-label-h", type=int, default=44)
    p.add_argument("--row-label-w", type=int, default=120)
    p.add_argument("--title-h", type=int, default=70)
    p.add_argument("--extra-v15", type=Path, default=None,
                   help="Optional extra version dir for v1.5 (final ckpt clip outputs)")
    args = p.parse_args()

    versions = list(VERSIONS)
    if args.extra_v15 is not None and args.extra_v15.exists():
        versions.append(("v1.5", str(args.extra_v15), ""))

    cell = args.cell
    gap = args.gap
    n_rows = len(CONCEPTS)
    n_cols = len(versions)

    width = args.row_label_w + n_cols * cell + (n_cols + 1) * gap
    height = args.title_h + args.col_label_h + n_rows * cell + (n_rows + 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(28)
    col_font = get_font(20)
    row_font = get_font(18)

    title = "What Gemma 4 thinks, drawn — across project versions"
    tw = draw.textlength(title, font=title_font)
    draw.text(((width - tw) / 2, 22), title, fill="black", font=title_font)

    # Column labels
    for ci, (label, _, _) in enumerate(versions):
        x = args.row_label_w + gap + ci * (cell + gap)
        cw = draw.textlength(label, font=col_font)
        draw.text((x + (cell - cw) / 2, args.title_h + 12), label, fill="#222222", font=col_font)

    # Rows
    for ri, concept in enumerate(CONCEPTS):
        y = args.title_h + args.col_label_h + gap + ri * (cell + gap)
        # Row label
        draw.text((10, y + cell // 2 - 10), concept, fill="black", font=row_font)
        for ci, (label, src_dir_str, suffix) in enumerate(versions):
            x = args.row_label_w + gap + ci * (cell + gap)
            src_dir = Path(src_dir_str)
            # Try suffixed (e.g. cat_4x.png) and several other variants
            candidates = [
                src_dir / f"{concept}{suffix}.png",
                src_dir / f"{concept}.png",
                src_dir / f"{concept}_top0.png",
                src_dir / f"{concept}_4x.png",
            ]
            img_path = next((p for p in candidates if p.exists()), None)
            if img_path is not None:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((cell, cell))
                cx = x + (cell - img.width) // 2
                cy = y + (cell - img.height) // 2
                canvas.paste(img, (cx, cy))
            else:
                draw.text((x + 8, y + 8), "—", fill="#999999", font=row_font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"[evol] wrote {args.out}  size={canvas.size}")


if __name__ == "__main__":
    main()
