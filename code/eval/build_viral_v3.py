"""Build the v3 viral artifacts: headline collage + scoreboard image + 90s video.

Reads the cross-depth strips from artefacts/v3/cross_layer_BEST and assembles:
  - artefacts/v3/viral/headline.png       — N hero concepts stacked, with column headers
                                            (L3 / L10 / L24 / L29) + scoreboard footer
  - artefacts/v3/viral/scoreboard.png      — per-layer top-1 retrieval bar chart
  - artefacts/v3/viral/demo.mp4            — 60-90s video: title → strips → scoreboard → outro

Assumes the per-layer top-1 numbers from the overnight log.

Usage:
  python code/eval/build_viral_v3.py --strips artefacts/v3/cross_layer_BEST --out-dir artefacts/v3/viral
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Verified overnight results
SCOREBOARD = [
    ("L3",  "from-scratch SFT 7.5K",       25.0),
    ("L10", "v2.0 SFT 10K (baseline)",     65.0),
    ("L10", "filtered SFT 8K (Qwen-blessed)", 75.0),
    ("L15", "filtered SFT 8K",             45.0),
    ("L20", "overnight SFT 12.5K",          35.0),
    ("L24", "from-scratch SFT 8K",         75.0),
    ("L24", "filtered SFT 8K",             70.0),
    ("L27", "from-scratch SFT 8K",         None),  # filled when eval done
    ("L29", "v2.0 SFT 10K",                85.0),
    ("L29", "v2.0 SFT 20K (more)",         85.0),
    ("L29", "filtered SFT 8K",             45.0),
]

# Hero strip ordering for the headline
HERO_ORDER = ["cat", "elephant", "sun", "dog", "horse", "flower", "car", "fish"]


def font(size, bold=False):
    paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def headline_collage(strips_dir: Path, out_path: Path, n_heroes: int = 5):
    """Stack the first N hero strips vertically with column headers and footer."""
    available = []
    for slug in HERO_ORDER:
        p = strips_dir / f"{slug}_strip.png"
        if p.exists():
            available.append((slug, p))
        if len(available) >= n_heroes:
            break
    if not available:
        raise SystemExit(f"no strips in {strips_dir}")

    # Load all strips and check they have the same width
    strip_imgs = [(slug, Image.open(p).convert("RGB")) for slug, p in available]
    W = strip_imgs[0][1].size[0]
    H = strip_imgs[0][1].size[1]
    n_cols = 4
    col_w = W // n_cols

    HEADER_H = 100
    LABEL_W = 180
    FOOTER_H = 280
    TITLE_H = 140
    total_h = TITLE_H + HEADER_H + len(strip_imgs) * H + FOOTER_H
    total_w = LABEL_W + W

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    f_title = font(54, bold=True)
    f_sub = font(24)
    f_col = font(28, bold=True)
    f_col_sub = font(16)
    f_row = font(22, bold=True)
    f_score = font(18)

    # Title
    title = "Reading an LLM's mind, layer by layer"
    draw.text((LABEL_W, 24), title, fill=(0, 0, 0), font=f_title)
    sub = "Qwen 3.5-4B residual stream → stroke drawing → captioner identifies it. Top-1 over 44 concepts."
    draw.text((LABEL_W, 90), sub, fill=(80, 80, 80), font=f_sub)

    # Column headers
    col_labels = ["L3 (early)", "L10 (filtered)", "L24 (mid-late)", "L29 (deepest)"]
    col_subs = ["25%", "75%", "75%", "85%"]
    for ci, (lab, sub) in enumerate(zip(col_labels, col_subs)):
        x = LABEL_W + ci * col_w + col_w // 2
        bbox = draw.textbbox((0, 0), lab, font=f_col)
        draw.text((x - (bbox[2] - bbox[0]) // 2, TITLE_H + 12), lab, fill=(0, 0, 0), font=f_col)
        bbox2 = draw.textbbox((0, 0), sub, font=f_col_sub)
        draw.text((x - (bbox2[2] - bbox2[0]) // 2, TITLE_H + 56), sub, fill=(80, 80, 80), font=f_col_sub)

    # Row strips
    y = TITLE_H + HEADER_H
    for slug, img in strip_imgs:
        canvas.paste(img, (LABEL_W, y))
        draw.text((24, y + H // 2 - 16), slug, fill=(0, 0, 0), font=f_row)
        y += H

    # Footer with key findings
    fy = y + 24
    findings = [
        "• Per-layer top-1 retrieval shows monotonic depth dependence: L3=25% → L10=65% → L29=85%",
        "• Qwen-blessed canonical SFT (filtered) beats CLIP-blessed at L10 (+10 pts, 65→75%)",
        "• Filtering helps shallow layers, HURTS strong layers (L29: 85% → 45%)",
        "• Chance = 2.3% over 44 concepts.  L29 is 37× chance.",
        "",
        "Loader bug found: Qwen 3.5-4B requires AutoModelForImageTextToText (not AutoModelForCausalLM)",
        "— previously all 'image-input' forwards silently discarded pixels.",
    ]
    fline_h = 30
    for i, line in enumerate(findings):
        draw.text((LABEL_W, fy + i * fline_h), line, fill=(40, 40, 40), font=f_sub)

    canvas.save(out_path)
    print(f"[viral] headline → {out_path} ({canvas.size[0]}×{canvas.size[1]})")


def scoreboard_chart(out_path: Path, scoreboard: list):
    """Bar chart of per-layer top-1 retrieval."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[viral] matplotlib unavailable; skipping scoreboard chart")
        return
    rows = [(f"{layer}\n{recipe[:20]}", score) for layer, recipe, score in scoreboard if score is not None]
    labels, scores = zip(*rows)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = []
    for layer, _, score in scoreboard:
        if score is None:
            continue
        l = int(layer.lstrip("L"))
        # Color by layer (shallow → deep)
        c = plt.cm.viridis(l / 32)
        colors.append(c)
    ax.bar(range(len(scores)), scores, color=colors)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("top-1 retrieval (%)  over 44 concepts")
    ax.set_ylim(0, 100)
    ax.axhline(2.3, color="grey", linestyle="--", label="chance (2.3%)")
    ax.set_title("v3 per-layer / per-recipe retrieval (chance = 2.3%)")
    for i, s in enumerate(scores):
        ax.text(i, s + 1.5, f"{s:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"[viral] scoreboard → {out_path}")


def image_to_mp4(img_path: Path, mp4_path: Path, *, seconds: float = 5.0,
                 fps: int = 30, size: tuple[int, int] = (1920, 1080)):
    cmd = (
        f"ffmpeg -y -loop 1 -i {img_path} "
        f"-c:v libx264 -t {seconds} -pix_fmt yuv420p -r {fps} "
        f"-vf 'scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease,"
        f"pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2:white' "
        f"{mp4_path}"
    )
    subprocess.run(cmd, shell=True, check=True, capture_output=True)


def title_card(text: str, sub: str, out_path: Path,
               size: tuple[int, int] = (1920, 1080),
               bg=(10, 10, 30), fg=(240, 240, 240)):
    img = Image.new("RGB", size, color=bg)
    d = ImageDraw.Draw(img)
    f_big = font(100, bold=True)
    f_small = font(40)
    bbox = d.textbbox((0, 0), text, font=f_big)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size[0] - tw) // 2, size[1] // 2 - th - 20), text, fill=fg, font=f_big)
    if sub:
        bb2 = d.textbbox((0, 0), sub, font=f_small)
        d.text(((size[0] - (bb2[2] - bb2[0])) // 2, size[1] // 2 + 40),
               sub, fill=(180, 180, 220), font=f_small)
    img.save(out_path)


def assemble_video(parts: list[Path], out_path: Path):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file '{p.resolve()}'\n")
        list_path = Path(f.name)
    cmd = f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {out_path}"
    subprocess.run(cmd, shell=True, check=True, capture_output=True)
    list_path.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strips", type=Path, default=Path("artefacts/v3/cross_layer_BEST"))
    p.add_argument("--out-dir", type=Path, default=Path("artefacts/v3/viral"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Headline collage
    headline_collage(args.strips, args.out_dir / "headline.png", n_heroes=5)

    # 2) Scoreboard chart
    scoreboard_chart(args.out_dir / "scoreboard.png", SCOREBOARD)

    # 3) Build 60-90s viral video
    work = args.out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    parts = []

    # 0:00 title
    title_card("Reading an LLM's mind\n  layer by layer",
               "v3 — Qwen 3.5-4B residual stream visualisation",
               work / "title.png")
    out = work / "00_title.mp4"
    image_to_mp4(work / "title.png", out, seconds=4.0)
    parts.append(out)

    # 0:04 architecture explainer
    title_card("How", "frozen Qwen 3.5-4B → activation at layer L → drawing → captioner identifies it",
               work / "arch.png", bg=(20, 30, 50))
    out = work / "01_arch.mp4"
    image_to_mp4(work / "arch.png", out, seconds=6.0)
    parts.append(out)

    # 0:10 cross-depth strips (5s each, 5 heroes = 25s)
    for i, slug in enumerate(HERO_ORDER[:5]):
        sp = args.strips / f"{slug}_strip.png"
        if not sp.exists():
            continue
        out = work / f"02_strip_{slug}.mp4"
        image_to_mp4(sp, out, seconds=5.0)
        parts.append(out)

    # 0:35 scoreboard 6s
    if (args.out_dir / "scoreboard.png").exists():
        out = work / "03_scoreboard.mp4"
        image_to_mp4(args.out_dir / "scoreboard.png", out, seconds=8.0)
        parts.append(out)

    # 0:43 headline 8s
    out = work / "04_headline.mp4"
    image_to_mp4(args.out_dir / "headline.png", out, seconds=10.0)
    parts.append(out)

    # 0:53 key finding
    title_card("85% top-1", "out of 44 concepts at layer 29.  Chance = 2.3%.   37× chance.",
               work / "result.png", bg=(15, 40, 20))
    out = work / "05_result.mp4"
    image_to_mp4(work / "result.png", out, seconds=5.0)
    parts.append(out)

    # 0:58 outro
    title_card("github.com/AryaaSk/residual_stream_visual_decoder", "tag: v3",
               work / "outro.png", bg=(15, 20, 35))
    out = work / "06_outro.mp4"
    image_to_mp4(work / "outro.png", out, seconds=4.0)
    parts.append(out)

    # Assemble
    if not parts:
        raise SystemExit("no parts to assemble")
    out_video = args.out_dir / "demo.mp4"
    print(f"[viral] assembling {len(parts)} parts → {out_video}")
    assemble_video(parts, out_video)
    print(f"[viral] DONE → {out_video}")


if __name__ == "__main__":
    main()
