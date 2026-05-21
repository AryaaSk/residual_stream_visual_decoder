"""Build 7-layer cross-depth strips (L3, L10, L15, L20, L24, L27, L29) using
the static pngs from artefacts/v3/viral/anim/L{NN}/{concept}.png.

Output: artefacts/v3/viral/strips_7layer/{concept}_strip.png
        artefacts/v3/viral/strips_7layer/grid.png  (8 concepts × 7 layers)
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


LAYERS = [("L03", "L3"), ("L10", "L10"), ("L15", "L15"), ("L20", "L20"),
          ("L24", "L24"), ("L27", "L27"), ("L29", "L29")]
HERO = ["cat", "elephant", "sun", "dog", "horse", "flower", "car", "fish",
         "tree", "airplane", "bird", "mountain"]
ANIM_DIR = Path("artefacts/v3/viral/anim")
OUT_DIR = Path("artefacts/v3/viral/strips_7layer")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def font(size, bold=False):
    for p in ["/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_strip(concept: str, cell: int = 280, label_h: int = 50) -> Image.Image | None:
    imgs = []
    for tag, label in LAYERS:
        p = ANIM_DIR / tag / f"{concept}.png"
        if p.exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell, cell))
            imgs.append((label, im))
    if not imgs:
        return None
    n = len(imgs)
    canvas = Image.new("RGB", (n * cell, cell + label_h), color=(255, 255, 255))
    d = ImageDraw.Draw(canvas)
    f = font(24, bold=True)
    for i, (label, im) in enumerate(imgs):
        x = i * cell + (cell - im.size[0]) // 2
        y = (cell - im.size[1]) // 2
        canvas.paste(im, (x, y))
        bbox = d.textbbox((0, 0), label, font=f)
        d.text((i * cell + (cell - (bbox[2] - bbox[0])) // 2, cell + 12),
                label, fill=(0, 0, 0), font=f)
    return canvas


def main():
    strips = []
    for c in HERO:
        s = build_strip(c)
        if s is None:
            continue
        out = OUT_DIR / f"{c}_strip.png"
        s.save(out)
        strips.append((c, s))
        print(f"[7layer] {c} -> {out}  ({s.size[0]}x{s.size[1]})")

    if not strips:
        return

    # Grid
    cell_w, cell_h = strips[0][1].size
    LABEL_W = 220
    TITLE_H = 130
    canvas = Image.new("RGB", (LABEL_W + cell_w, TITLE_H + len(strips) * cell_h),
                        color=(255, 255, 255))
    d = ImageDraw.Draw(canvas)
    f_title = font(46, bold=True)
    f_sub = font(22)
    f_row = font(28, bold=True)
    d.text((24, 24), "Cross-depth grid: 8+ concepts x 7 layers",
            fill=(0, 0, 0), font=f_title)
    d.text((24, 82), "Qwen 3.5-4B residual stream  |  best-of-N gen-likelihood-ranked drawing per cell",
            fill=(80, 80, 80), font=f_sub)
    for i, (c, s) in enumerate(strips):
        y = TITLE_H + i * cell_h
        canvas.paste(s, (LABEL_W, y))
        d.text((24, y + cell_h // 2 - 18), c, fill=(0, 0, 0), font=f_row)
    out = OUT_DIR / "grid.png"
    canvas.save(out)
    print(f"[7layer] grid -> {out}  ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
