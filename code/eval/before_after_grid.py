"""Build the v1.1 vs v1.2 before/after grid image.

For each hero concept, places v1.1's drawing next to v1.2's drawing,
labelled with the prompt. Output: PNG suitable for tweet embedding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# (slug, prompt) pairs. We require both v1_1/inject_demo_L24/<slug>_4x.png and
# v1_2/inject_demo_L12/<slug>_4x.png to exist; missing entries are skipped.
HERO_PAIRS = [
    ("cat", "I am thinking about a cat."),
    ("dog", "I am thinking about a dog."),
    ("fish", "Imagine a fish."),
    ("flower", "Imagine a flower in bloom."),
    ("cactus", "I am picturing a cactus in the desert."),
    ("mountain", "I am picturing a mountain."),
    ("elephant", "I am thinking about an elephant."),
    ("horse", "I am thinking about a horse."),
]


def load_or_blank(path: Path, size: int) -> Image.Image:
    if path.exists():
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), "white")
        x = (size - img.width) // 2
        y = (size - img.height) // 2
        canvas.paste(img, (x, y))
        return canvas
    return Image.new("RGB", (size, size), "white")


def get_font(size: int):
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1_1-dir", type=Path,
                   default=Path("findings/v1_1/inject_demo_L24"))
    p.add_argument("--v1_2-dir", type=Path,
                   default=Path("findings/v1_2/inject_demo_L12"))
    p.add_argument("--out", type=Path, default=Path("artefacts/v1_2/before_after.png"))
    p.add_argument("--cell-size", type=int, default=320)
    p.add_argument("--gap", type=int, default=20)
    p.add_argument("--header-h", type=int, default=80)
    p.add_argument("--row-label-h", type=int, default=40)
    args = p.parse_args()

    # Try both _4x.png (old format) and .png (CLIP-ranker output format)
    def exists_in(d: Path, slug: str) -> Path | None:
        for variant in (f"{slug}_4x.png", f"{slug}.png", f"{slug}_top0.png"):
            p = d / variant
            if p.exists():
                return p
        return None
    available = [(s, p) for s, p in HERO_PAIRS if exists_in(args.v1_2_dir, s) is not None]
    if not available:
        print(f"[grid] no matching pairs found; need files in {args.v1_2_dir}")
        return
    n = len(available)
    print(f"[grid] {n} concept pairs available")

    cell = args.cell_size
    gap = args.gap
    header_h = args.header_h
    row_label_h = args.row_label_h
    cols = n
    rows = 2  # v1.1, v1.2

    width = cols * cell + (cols + 1) * gap
    height = header_h + rows * (cell + row_label_h) + (rows + 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(36)
    subtitle_font = get_font(18)
    label_font = get_font(14)
    row_font = get_font(22)

    # Header
    title = "Drawing what Gemma 4 is thinking — v1.1 vs v1.5"
    sub = "Same prompts, same model, same activation injection layer."
    tw = draw.textlength(title, font=title_font)
    draw.text(((width - tw) / 2, 12), title, fill="black", font=title_font)
    sw = draw.textlength(sub, font=subtitle_font)
    draw.text(((width - sw) / 2, 50), sub, fill="#555555", font=subtitle_font)

    # Rows
    for row_i, (label, src_dir) in enumerate([
        ("v1.1 (no projector, no AV-LoRA)", args.v1_1_dir),
        ("v1.5 (24-layer LoRA + top-5 canonical + CLIP-32)", args.v1_2_dir),
    ]):
        y_label = header_h + gap + row_i * (cell + row_label_h + gap)
        y_img = y_label + row_label_h
        # Row label on the left margin
        draw.text((gap, y_label + 8), label, fill="black", font=row_font)
        for col_i, (slug, prompt) in enumerate(available):
            x = gap + col_i * (cell + gap)
            found = exists_in(src_dir, slug)
            img = load_or_blank(found if found else (src_dir / f"{slug}.png"), cell)
            canvas.paste(img, (x, y_img))
            # Caption: prompt under the image (only on bottom row)
            if row_i == 1:
                cy = y_img + cell + 4
                # Truncate caption to fit
                caption = prompt
                while draw.textlength(caption, font=label_font) > cell - 6 and len(caption) > 5:
                    caption = caption[:-2] + "…"
                draw.text((x + 4, cy), caption, fill="#333333", font=label_font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"[grid] wrote {args.out}  size={canvas.size}")


if __name__ == "__main__":
    main()
