"""Build a clean hero gallery image (no comparison) for the v1.2 README.

Layout: 4 cols × 3 rows of (drawing + prompt caption underneath).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


GALLERY = [
    ("cat",      "I am thinking about a cat"),
    ("dog",      "I am thinking about a dog"),
    ("fish",     "Imagine a fish"),
    ("flower",   "Imagine a flower in bloom"),
    ("cactus",   "A cactus in the desert"),
    ("mountain", "I am picturing a mountain"),
    ("elephant", "I am thinking about an elephant"),
    ("horse",    "I am thinking about a horse"),
    ("sun",      "The sun is shining"),
    ("tree",     "I am picturing a tree"),
    ("cloud",    "A cloud in the sky"),
    ("pizza",    "I am thinking about a pizza"),
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
    p.add_argument("--src", type=Path, default=Path("findings/v1_2/inject_demo_L12"))
    p.add_argument("--out", type=Path, default=Path("artefacts/v1_2/gallery.png"))
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--cell", type=int, default=280)
    p.add_argument("--gap", type=int, default=16)
    p.add_argument("--cap-h", type=int, default=36)
    p.add_argument("--title-h", type=int, default=70)
    args = p.parse_args()

    cols = args.cols
    n = len(GALLERY)
    rows = (n + cols - 1) // cols
    cell = args.cell
    gap = args.gap
    cap_h = args.cap_h
    title_h = args.title_h

    width = cols * cell + (cols + 1) * gap
    height = title_h + rows * (cell + cap_h) + (rows + 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(30)
    cap_font = get_font(14)

    title = "What Gemma 4 thinks, drawn"
    tw = draw.textlength(title, font=title_font)
    draw.text(((width - tw) / 2, 20), title, fill="black", font=title_font)

    for i, (slug, prompt) in enumerate(GALLERY):
        r, c = divmod(i, cols)
        x = gap + c * (cell + gap)
        y = title_h + gap + r * (cell + cap_h + gap)
        img_path = args.src / f"{slug}_4x.png"
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((cell, cell))
            cx = x + (cell - img.width) // 2
            cy = y + (cell - img.height) // 2
            canvas.paste(img, (cx, cy))
        # Caption
        cy_text = y + cell + 4
        caption = prompt
        while draw.textlength(caption, font=cap_font) > cell - 6 and len(caption) > 5:
            caption = caption[:-2] + "…"
        cw = draw.textlength(caption, font=cap_font)
        draw.text((x + (cell - cw) / 2, cy_text), caption, fill="#222222", font=cap_font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"[gallery] wrote {args.out}  size={canvas.size}")


if __name__ == "__main__":
    main()
