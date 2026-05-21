"""Compose OOD grids at the best layer (L29) showing what the model 'draws'
for never-trained concepts: emotions, abstract, math, philosophical.

Output: artefacts/v3/viral/ood_grid_L29.png (static)
        artefacts/v3/viral/ood_grid_L29.mp4 (5s slow zoom)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


GROUPS = {
    "emotions": ["happy", "sad", "anger", "love", "loneliness", "nostalgia",
                 "grief", "hope", "fear", "wonder"],
    "abstract / philosophical": ["god", "death", "dreams", "the_universe",
                                  "consciousness", "nothingness", "forever",
                                  "freedom", "memory", "silence"],
    "math / numbers": ["infinity", "two_plus_two", "triangle", "pi", "zero", "many"],
    "places / people": ["eiffel", "paris", "tokyo", "the_beatles", "shakespeare",
                         "einstein", "napoleon"],
    "sensory / quality": ["colour_purple", "colour_red", "warmth", "rain_sound",
                           "midnight", "morning"],
}


def font(size, bold=False):
    paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_grid(layer_tag: str, src_dir: Path, out_path: Path,
               cell_w: int = 280, cell_h: int = 280):
    src = src_dir / layer_tag
    rows = []
    for group, slugs in GROUPS.items():
        items = []
        for slug in slugs:
            p = src / f"{slug}.png"
            if p.exists():
                items.append((slug, Image.open(p).convert("RGB")))
        if items:
            rows.append((group, items))

    if not rows:
        raise SystemExit(f"no pngs found in {src}")

    n_cols = max(len(items) for _, items in rows)
    TITLE_H = 130
    GROUP_LABEL_H = 50
    LABEL_H = 36
    PAD = 8
    total_w = n_cols * (cell_w + PAD) + PAD + 200
    total_h = TITLE_H + sum(GROUP_LABEL_H + cell_h + LABEL_H + 16 for _ in rows) + 40

    canvas = Image.new("RGB", (total_w, total_h), color=(252, 252, 252))
    d = ImageDraw.Draw(canvas)
    f_title = font(48, bold=True)
    f_sub = font(22)
    f_group = font(28, bold=True)
    f_label = font(15)

    d.text((40, 22), f"What the model draws for ideas it was never trained on", fill=(0, 0, 0), font=f_title)
    d.text((40, 84), f"Layer {layer_tag} of frozen Qwen 3.5-4B residual stream. No concept labels seen for any of these prompts.",
           fill=(80, 80, 80), font=f_sub)

    y = TITLE_H
    for group, items in rows:
        d.text((40, y), group.upper(), fill=(40, 40, 40), font=f_group)
        y += GROUP_LABEL_H
        x = 200 + PAD
        for slug, img in items:
            img = img.copy()
            img.thumbnail((cell_w, cell_h))
            iw, ih = img.size
            ox = x + (cell_w - iw) // 2
            oy = y + (cell_h - ih) // 2
            canvas.paste(img, (ox, oy))
            d.rectangle([x, y, x + cell_w, y + cell_h], outline=(220, 220, 220), width=1)
            label = slug.replace("_", " ")
            bbox = d.textbbox((0, 0), label, font=f_label)
            d.text((x + (cell_w - (bbox[2] - bbox[0])) // 2, y + cell_h + 4),
                   label, fill=(60, 60, 60), font=f_label)
            x += cell_w + PAD
        y += cell_h + LABEL_H + 16

    canvas.save(out_path)
    print(f"[ood-grid] {layer_tag} → {out_path} ({canvas.size[0]}×{canvas.size[1]})")


def png_to_mp4(png: Path, mp4: Path, seconds: float = 6.0, fps: int = 30):
    """Ken-burns slow zoom on the grid."""
    cmd = (
        f"ffmpeg -y -hide_banner -loglevel error -loop 1 -i {png} "
        f"-vf 'scale=2560:-2,zoompan=z=zoom+0.0008:d={int(seconds*fps)}:s=1920x1080' "
        f"-c:v libx264 -pix_fmt yuv420p -t {seconds} -r {fps} {mp4}"
    )
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    out_dir = Path("artefacts/v3/viral")
    src_dir = Path("artefacts/v3/viral/ood")
    for tag in ["L29", "L27", "L24", "L20", "L15", "L10", "L03"]:
        png = out_dir / f"ood_grid_{tag}.png"
        build_grid(tag, src_dir, png)
        mp4 = out_dir / f"ood_grid_{tag}.mp4"
        png_to_mp4(png, mp4, seconds=8.0)
        print(f"[ood-grid] {tag} mp4 → {mp4}")
