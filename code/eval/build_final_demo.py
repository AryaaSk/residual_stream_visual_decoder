"""Final viral demo composer — uses ALL the local assets, not just static images.

Plan (90-110s target):
  0:00  title card                                                        4s
  0:04  architecture card                                                 6s
  0:10  depth chart (the headline finding)                                7s
  0:17  scoreboard chart                                                  6s
  0:23  cross-layer per-concept anims (cat, elephant, sun, dog, flower) 25s  (5 x 5s)
  0:48  headline collage                                                  8s
  0:56  OOD grid L29                                                      8s
  1:04  cross-token trajectory (paris_eiffel + capital_japan)            20s  (2 x 10s)
  1:24  result card                                                       5s
  1:29  outro card                                                        4s

Re-encodes each clip to 1920x1080 white-letterbox so concat-copy works.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("artefacts/v3/viral")
WORK = OUT_DIR / "_work_final"
WORK.mkdir(parents=True, exist_ok=True)

SIZE = (1920, 1080)
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
               bg=(10, 10, 30), fg=(240, 240, 240), sub_color=(180, 180, 220)):
    img = Image.new("RGB", SIZE, color=bg)
    d = ImageDraw.Draw(img)
    f_big = font(96, bold=True)
    f_small = font(36)
    lines = text.split("\n")
    total_h = len(lines) * 110
    y0 = SIZE[1] // 2 - total_h // 2 - 40
    for i, line in enumerate(lines):
        bbox = d.textbbox((0, 0), line, font=f_big)
        tw = bbox[2] - bbox[0]
        d.text(((SIZE[0] - tw) // 2, y0 + i * 110), line, fill=fg, font=f_big)
    if sub:
        for j, s_line in enumerate(sub.split("\n")):
            bbox2 = d.textbbox((0, 0), s_line, font=f_small)
            tw2 = bbox2[2] - bbox2[0]
            d.text(((SIZE[0] - tw2) // 2, y0 + total_h + 30 + j * 50),
                    s_line, fill=sub_color, font=f_small)
    img.save(out_png)


def encode_image(png: Path, mp4: Path, seconds: float, bg="white"):
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
    """Re-encode source MP4 to canonical 1920x1080 white-letterbox, optionally trimmed."""
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
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-c", "copy", str(out),
    ]
    subprocess.run(cmd, check=True)
    list_path.unlink()


def main():
    parts = []

    # 0:00 title
    title_card("Reading an LLM's mind\nlayer by layer",
                "v3 - Qwen 3.5-4B residual stream visualisation",
                WORK / "00_title.png")
    encode_image(WORK / "00_title.png", WORK / "00_title.mp4", 4.0, bg="black")
    parts.append(WORK / "00_title.mp4")

    # 0:04 architecture
    title_card("How it works",
                "frozen Qwen 3.5-4B -> activation at layer L -> stroke decoder -> drawing\n"
                "frozen Qwen scores 'A drawing of a __' on the rendered image",
                WORK / "01_arch.png", bg=(20, 30, 50))
    encode_image(WORK / "01_arch.png", WORK / "01_arch.mp4", 6.0, bg="black")
    parts.append(WORK / "01_arch.mp4")

    # 0:10 depth chart (the headline finding)
    if (OUT_DIR / "depth_chart.png").exists():
        encode_image(OUT_DIR / "depth_chart.png", WORK / "02_depth.mp4", 7.0, bg="white")
        parts.append(WORK / "02_depth.mp4")

    # 0:17 scoreboard
    if (OUT_DIR / "scoreboard.png").exists():
        encode_image(OUT_DIR / "scoreboard.png", WORK / "03_score.mp4", 6.0, bg="white")
        parts.append(WORK / "03_score.mp4")

    # 0:23 BIG REVEAL grid animation (12 concepts × 7 layers, all at once)
    if (OUT_DIR / "grid_anim_loop.mp4").exists():
        title_card("12 concepts x 7 layers, all at once",
                    "every cell is a real activation->drawing.\n"
                    "columns = concepts, rows = layer depth (L3 up to L29).",
                    WORK / "20_grid_title.png", bg=(25, 25, 50))
        encode_image(WORK / "20_grid_title.png", WORK / "20_grid_title.mp4", 3.0, bg="black")
        parts.append(WORK / "20_grid_title.mp4")
        encode_video(OUT_DIR / "grid_anim_loop.mp4", WORK / "20_grid.mp4",
                      seconds=12.0, bg="white")
        parts.append(WORK / "20_grid.mp4")

    # then cross-layer per-concept anims
    HERO_VIDEOS = ["cat", "elephant", "sun", "dog", "flower"]
    for slug in HERO_VIDEOS:
        src = OUT_DIR / "cross_layer_anim" / f"{slug}.mp4"
        if not src.exists():
            continue
        # label header overlay via title before each anim (small caption frame)
        title_card(slug.upper(),
                    "watch the same activation get decoded at L3, L10, L15, L20, L24, L27, L29",
                    WORK / f"04_lbl_{slug}.png", bg=(15, 15, 25))
        encode_image(WORK / f"04_lbl_{slug}.png", WORK / f"04_lbl_{slug}.mp4", 1.5, bg="black")
        parts.append(WORK / f"04_lbl_{slug}.mp4")
        # encode the anim itself
        encode_video(src, WORK / f"04_anim_{slug}.mp4", seconds=4.0, bg="white")
        parts.append(WORK / f"04_anim_{slug}.mp4")

    # 0:48 headline collage
    if (OUT_DIR / "headline.png").exists():
        encode_image(OUT_DIR / "headline.png", WORK / "05_headline.mp4", 8.0, bg="white")
        parts.append(WORK / "05_headline.mp4")

    # 0:56 OOD grid L29 with title
    title_card("Generalisation",
                "OOD prompts: emotions, math, philosophy.  None seen during training.",
                WORK / "06_ood_title.png", bg=(20, 35, 55))
    encode_image(WORK / "06_ood_title.png", WORK / "06_ood_title.mp4", 2.0, bg="black")
    parts.append(WORK / "06_ood_title.mp4")
    if (OUT_DIR / "ood_grid_L29.png").exists():
        encode_image(OUT_DIR / "ood_grid_L29.png", WORK / "06_ood_grid.mp4", 8.0, bg="white")
        parts.append(WORK / "06_ood_grid.mp4")

    # 1:04 cross-token trajectory
    title_card("Per-token trajectory",
                "as the model generates tokens, the activation evolves.\n"
                "L10 / L24 / L29 simultaneously decode each step.",
                WORK / "07_xtoken_title.png", bg=(20, 40, 60))
    encode_image(WORK / "07_xtoken_title.png", WORK / "07_xtoken_title.mp4", 2.0, bg="black")
    parts.append(WORK / "07_xtoken_title.mp4")
    XT_ORDER = ["paris_eiffel", "capital_japan", "ocean_color", "storm_village",
                 "sad_funeral", "dog_thinking", "face_smile", "triangle_geom",
                 "once_upon", "rainbow"]
    for slug in XT_ORDER:
        src = OUT_DIR / "cross_token" / f"{slug}.mp4"
        if src.exists():
            encode_video(src, WORK / f"07_xt_{slug}.mp4", bg="white")
            parts.append(WORK / f"07_xt_{slug}.mp4")

    # 1:24 result card
    title_card("85% top-1",
                "20 held-out concepts at layer 29.  Chance = 5%.  17 times chance.",
                WORK / "08_result.png", bg=(15, 40, 20))
    encode_image(WORK / "08_result.png", WORK / "08_result.mp4", 5.0, bg="black")
    parts.append(WORK / "08_result.mp4")

    # 1:29 outro
    title_card("github.com/AryaaSk\n/residual_stream_visual_decoder",
                "tag: v3",
                WORK / "09_outro.png", bg=(15, 20, 35))
    encode_image(WORK / "09_outro.png", WORK / "09_outro.mp4", 4.0, bg="black")
    parts.append(WORK / "09_outro.mp4")

    # Assemble
    out = OUT_DIR / "demo_final.mp4"
    print(f"[final] {len(parts)} parts -> {out}")
    concat(parts, out)
    # Total duration
    import json
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(out)],
        capture_output=True, text=True,
    )
    info = json.loads(probe.stdout)
    dur = float(info["format"]["duration"])
    print(f"[final] duration: {dur:.1f}s")
    print(f"[final] DONE -> {out}")


if __name__ == "__main__":
    main()
