"""Short social-media-friendly cut, ~30-45s.

Just: hook -> one hero cross-layer anim -> result card.
Vertical-friendly 1080x1920 with letterbox.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("artefacts/v3/viral")
WORK = OUT_DIR / "_work_social"
WORK.mkdir(parents=True, exist_ok=True)

SIZE = (1080, 1920)
FPS = 30


def font(size, bold=False):
    for p in ["/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def title_card(text: str, sub: str, out_png: Path,
               bg=(10, 10, 30), fg=(240, 240, 240)):
    img = Image.new("RGB", SIZE, color=bg)
    d = ImageDraw.Draw(img)
    f_big = font(96, bold=True)
    f_small = font(40)
    lines = text.split("\n")
    total_h = len(lines) * 110
    y0 = SIZE[1] // 2 - total_h // 2 - 80
    for i, line in enumerate(lines):
        bbox = d.textbbox((0, 0), line, font=f_big)
        d.text(((SIZE[0] - (bbox[2] - bbox[0])) // 2, y0 + i * 110),
                line, fill=fg, font=f_big)
    if sub:
        for j, s_line in enumerate(sub.split("\n")):
            bbox2 = d.textbbox((0, 0), s_line, font=f_small)
            d.text(((SIZE[0] - (bbox2[2] - bbox2[0])) // 2, y0 + total_h + 30 + j * 56),
                    s_line, fill=(190, 190, 220), font=f_small)
    img.save(out_png)


def encode_image(png: Path, mp4: Path, seconds: float, bg="black"):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-loop", "1", "-i", str(png),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-t", str(seconds), "-r", str(FPS),
        "-vf", f"scale={SIZE[0]}:{SIZE[1]}:force_original_aspect_ratio=decrease,"
                f"pad={SIZE[0]}:{SIZE[1]}:(ow-iw)/2:(oh-ih)/2:{bg}",
        str(mp4),
    ]
    subprocess.run(cmd, check=True)


def encode_video(src: Path, mp4: Path, seconds: float | None = None, bg="white"):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", str(FPS),
        "-an",
        "-vf", f"scale={SIZE[0]}:{SIZE[1]}:force_original_aspect_ratio=decrease,"
                f"pad={SIZE[0]}:{SIZE[1]}:(ow-iw)/2:(oh-ih)/2:{bg}",
    ]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += [str(mp4)]
    subprocess.run(cmd, check=True)


def concat(parts, out: Path):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file '{Path(p).resolve()}'\n")
        list_path = Path(f.name)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c", "copy", str(out)]
    subprocess.run(cmd, check=True)
    list_path.unlink()


def main():
    parts = []

    title_card("What does an LLM\n'see'?",
                "Qwen 3.5-4B residual stream\nbecomes a drawing\nyou can recognise",
                WORK / "00_hook.png")
    encode_image(WORK / "00_hook.png", WORK / "00_hook.mp4", 3.5)
    parts.append(WORK / "00_hook.mp4")

    title_card("Same activation, 7 layers",
                "(L3 / L10 / L15 / L20 / L24 / L27 / L29)\nThe model's 'idea' crystallises\nas depth grows.",
                WORK / "01_setup.png", bg=(20, 30, 50))
    encode_image(WORK / "01_setup.png", WORK / "01_setup.mp4", 3.5)
    parts.append(WORK / "01_setup.mp4")

    for slug in ["cat", "elephant", "flower"]:
        src = OUT_DIR / "cross_layer_anim" / f"{slug}.mp4"
        if src.exists():
            encode_video(src, WORK / f"02_{slug}.mp4", seconds=4.0)
            parts.append(WORK / f"02_{slug}.mp4")

    title_card("85% top-1",
                "20 held-out concepts at L29.\nChance = 5%.   17 times chance.",
                WORK / "03_result.png", bg=(15, 40, 20))
    encode_image(WORK / "03_result.png", WORK / "03_result.mp4", 4.0)
    parts.append(WORK / "03_result.mp4")

    title_card("github.com/AryaaSk\n/residual_stream\n_visual_decoder",
                "tag: v3",
                WORK / "04_outro.png", bg=(15, 20, 35))
    encode_image(WORK / "04_outro.png", WORK / "04_outro.mp4", 3.0)
    parts.append(WORK / "04_outro.mp4")

    out = OUT_DIR / "demo_social.mp4"
    print(f"[social] {len(parts)} parts -> {out}")
    concat(parts, out)
    import json
    info = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", str(out)],
        capture_output=True, text=True)
    dur = float(json.loads(info.stdout)["format"]["duration"])
    print(f"[social] duration {dur:.1f}s, file: {out}")


if __name__ == "__main__":
    main()
