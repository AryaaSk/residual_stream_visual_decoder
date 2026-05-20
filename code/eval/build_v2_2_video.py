"""Compose the v2.2 viral video.

Stitches together the six demos into a single ~90-second MP4:
    0:00  title card "Can we read an LLM's mind?"
    0:03  hero gallery shot
    0:08  architecture diagram
    0:18  Demo 3: per-token trajectory (best one)
    0:30  Demo 1: cross-layer 4-panel for 4 hero concepts (5s each = 20s)
    0:50  Demo 2: interpolation morphs (~15s)
    1:05  Demo 4 (random-h) + Demo 6 (probe accuracy)
    1:15  Demo 5: OOD demo grid
    1:25  outro card
    1:30  end

Each segment is normalised to 720x720 at 24fps before concat via ffmpeg.

Note: requires the per-demo MP4s + images to already exist at:
    artefacts/v2_2/per_token/{slug}.mp4
    artefacts/v2_2/cross_layer/{slug}.mp4
    artefacts/v2_2/morph/{tag}.mp4
    findings/v2_2/random_h_baseline/grid_*.png
    findings/v2_2/probe_accuracy.png
    findings/v2_2/ood/{slug}.png
    artefacts/v2_0/best_of_best/{slug}.png   (gallery shot fallback)

Output:
    artefacts/v2_2/demo.mp4
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_title_card(text: str, sub: str = "", out_path: Path = None,
                    size: tuple[int, int] = (1080, 1080),
                    bg: tuple[int, int, int] = (10, 10, 30),
                    fg: tuple[int, int, int] = (240, 240, 240)) -> Path:
    img = Image.new("RGB", size, color=bg)
    d = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 72)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = font_big
    bbox = d.textbbox((0, 0), text, font=font_big)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size[0] - tw) / 2, size[1] / 2 - th - 30), text, fill=fg, font=font_big)
    if sub:
        bb2 = d.textbbox((0, 0), sub, font=font_small)
        d.text(((size[0] - (bb2[2] - bb2[0])) / 2, size[1] / 2 + 30),
               sub, fill=(180, 180, 220), font=font_small)
    img.save(out_path)
    return out_path


def image_to_mp4(img_path: Path, mp4_path: Path, *, seconds: float, fps: int = 24,
                 size: tuple[int, int] = (1080, 1080)):
    # ffmpeg loops the image for `seconds`.
    cmd = (
        f"ffmpeg -y -loop 1 -i {shlex.quote(str(img_path))} "
        f"-c:v libx264 -t {seconds} -pix_fmt yuv420p -r {fps} "
        f"-vf 'scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease,"
        f"pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2:white' "
        f"{shlex.quote(str(mp4_path))}"
    )
    subprocess.run(cmd, shell=True, check=True, capture_output=True)


def normalise_mp4(in_mp4: Path, out_mp4: Path, *, seconds: float | None = None,
                  fps: int = 24, size: tuple[int, int] = (1080, 1080)):
    duration = f"-t {seconds}" if seconds else ""
    cmd = (
        f"ffmpeg -y -i {shlex.quote(str(in_mp4))} "
        f"{duration} "
        f"-vf 'scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease,"
        f"pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2:white,fps={fps},setpts=PTS-STARTPTS' "
        f"-c:v libx264 -pix_fmt yuv420p -an "
        f"{shlex.quote(str(out_mp4))}"
    )
    subprocess.run(cmd, shell=True, check=True, capture_output=True)


def concat_mp4s(parts: list[Path], out_path: Path):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file {shlex.quote(str(p.resolve()))}\n")
        list_path = Path(f.name)
    cmd = (
        f"ffmpeg -y -f concat -safe 0 -i {shlex.quote(str(list_path))} "
        f"-c copy {shlex.quote(str(out_path))}"
    )
    subprocess.run(cmd, shell=True, check=True, capture_output=True)
    list_path.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("artefacts/v2_2"))
    p.add_argument("--findings-dir", type=Path, default=Path("findings/v2_2"))
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--frame-size", type=int, nargs=2, default=[1080, 1080])
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = args.out_dir / "_video_work"
    work.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []

    size = tuple(args.frame_size)

    def add_image_seg(path: Path, seconds: float, name: str):
        if not path.exists():
            print(f"[viral] MISSING image: {path}; skipping segment {name}", flush=True)
            return
        norm = work / f"seg_{name}.mp4"
        image_to_mp4(path, norm, seconds=seconds, fps=args.fps, size=size)
        parts.append(norm)
        print(f"[viral] + {name} ({seconds}s) from image", flush=True)

    def add_mp4_seg(path: Path, seconds: float | None, name: str):
        if not path.exists():
            print(f"[viral] MISSING mp4: {path}; skipping segment {name}", flush=True)
            return
        norm = work / f"seg_{name}.mp4"
        normalise_mp4(path, norm, seconds=seconds, fps=args.fps, size=size)
        parts.append(norm)
        print(f"[viral] + {name} ({seconds or 'full'}s) from mp4", flush=True)

    # 0:00 — title card
    title = work / "title.png"
    make_title_card("Can we read an LLM's mind?", sub="v2.2 — cross-layer interpretability",
                    out_path=title, size=size)
    add_image_seg(title, 3.0, "00_title")

    # 0:03 — hero gallery: cat from v2.0
    add_image_seg(Path("artefacts/v2_0/best_of_best/cat.png"), 5.0, "01_hero_cat")

    # 0:08 — architecture diagram (we'll generate this on the fly if missing)
    arch = work / "arch.png"
    make_title_card("Qwen 3.5-4B → activation at layer L → decoder draws",
                    sub="frozen LLM emits a residual stream • we read it • a verbalizer draws what's there",
                    out_path=arch, size=size, bg=(20, 30, 50))
    add_image_seg(arch, 8.0, "02_arch")

    # 0:18 — per-token trajectory (best one)
    add_mp4_seg(args.out_dir / "per_token" / "paris_eiffel.mp4", 12.0, "03_per_token")

    # 0:30 — cross-layer 4-panel for 4 hero concepts (5s each = 20s)
    for slug in ["cat", "elephant", "flower", "sun"]:
        add_mp4_seg(args.out_dir / "cross_layer" / f"{slug}.mp4", 5.0, f"04_xlayer_{slug}")

    # 0:50 — morphs (~15s for 3 pairs)
    for tag in ["cat_to_elephant", "fish_to_bird", "sun_to_cloud"]:
        add_mp4_seg(args.out_dir / "morph" / f"{tag}.mp4", 5.0, f"05_morph_{tag}")

    # 1:05 — random-h baseline + probe accuracy
    add_image_seg(args.findings_dir / "random_h_baseline" / "grid_random_iso.png", 5.0, "06_random_h")
    add_image_seg(args.findings_dir / "probe_accuracy.png", 5.0, "07_probe")

    # 1:15 — OOD demo (compose into a grid if individual PNGs)
    ood_grid = work / "ood_grid.png"
    ood_paths = sorted((args.findings_dir / "ood").glob("*.png")) if (args.findings_dir / "ood").exists() else []
    if ood_paths:
        # 4-up grid
        from PIL import Image as Im
        n = min(8, len(ood_paths))
        first = Im.open(ood_paths[0]).resize((448, 448))
        cell = first.size
        cols = 4
        rows = (n + cols - 1) // cols
        grid = Im.new("RGB", (cols * cell[0], rows * cell[1]), color=(255, 255, 255))
        for i, p_ in enumerate(ood_paths[:n]):
            im = Im.open(p_).convert("RGB").resize(cell)
            grid.paste(im, ((i % cols) * cell[0], (i // cols) * cell[1]))
        grid.save(ood_grid)
        add_image_seg(ood_grid, 10.0, "08_ood")

    # 1:25 — outro
    outro = work / "outro.png"
    make_title_card("github.com/AryaaSk/residual_stream_visual_decoder",
                    sub="tag: v2.2  •  cross-layer interpretability",
                    out_path=outro, size=size, bg=(15, 20, 35))
    add_image_seg(outro, 5.0, "09_outro")

    # Concat all
    out_path = args.out_dir / "demo.mp4"
    if not parts:
        print("[viral] no parts assembled; aborting", flush=True)
        sys.exit(1)
    print(f"[viral] concatenating {len(parts)} parts → {out_path}", flush=True)
    concat_mp4s(parts, out_path)
    print(f"[viral] DONE → {out_path}", flush=True)


if __name__ == "__main__":
    main()
