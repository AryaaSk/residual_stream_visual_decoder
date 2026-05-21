"""Viral demo v2 — drawings-first, prompt -> drawing format, with morphs and old AR-loss clips.

No opening graph. Opens on a drawing reveal.

Structure (~110s):
  0:00  silent text card: "what does a language model 'see' inside?"           3s
  0:03  prompt -> drawing reveals (5 concepts x ~4s)                          20s
  0:23  architecture card: prompt -> Qwen -> activation h -> decoder -> draw   5s
  0:28  cross-layer reveal (3 concepts using cross_layer_anim, ~7s each)      21s
  0:49  morphs (3 from v2_2/morph, ~5s each)                                  15s
  1:04  per-token trajectory (2 prompts from cross_token, ~8s each)           16s
  1:20  OOD section title + 4 OOD prompts -> drawing (3s each)                14s
  1:34  honesty card: snap-to-nearest disclaimer                               5s
  1:39  old vs new (v1_5 Gemma clip vs v3 Qwen clip side-by-side)              8s
  1:47  outro: github link                                                     4s

All clips re-encoded to 1920x1080 white bg for clean concat.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("artefacts/v3/viral")
WORK = OUT_DIR / "_work_v2"
WORK.mkdir(parents=True, exist_ok=True)

SIZE = (1920, 1080)
FPS = 30


def font(size, bold=False):
    # Arial Unicode has the → glyph; Helvetica doesn't always.
    for p in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def title_card(text: str, sub: str, out_png: Path,
               bg=(10, 10, 30), fg=(240, 240, 240), sub_color=(180, 180, 220),
               big_size=88, small_size=36):
    img = Image.new("RGB", SIZE, color=bg)
    d = ImageDraw.Draw(img)
    f_big = font(big_size, bold=True)
    f_small = font(small_size)
    lines = text.split("\n")
    line_h = int(big_size * 1.15)
    total_h = len(lines) * line_h
    y0 = SIZE[1] // 2 - total_h // 2 - (40 if sub else 0)
    for i, line in enumerate(lines):
        bbox = d.textbbox((0, 0), line, font=f_big)
        d.text(((SIZE[0] - (bbox[2] - bbox[0])) // 2, y0 + i * line_h),
                line, fill=fg, font=f_big)
    if sub:
        for j, s_line in enumerate(sub.split("\n")):
            bbox2 = d.textbbox((0, 0), s_line, font=f_small)
            d.text(((SIZE[0] - (bbox2[2] - bbox2[0])) // 2,
                    y0 + total_h + 30 + j * int(small_size * 1.3)),
                    s_line, fill=sub_color, font=f_small)
    img.save(out_png)


def prompt_panel(prompt: str, out_png: Path,
                  size=(720, 1080), bg=(245, 245, 250)):
    """Left panel: the prompt text in big quoted font."""
    img = Image.new("RGB", size, color=bg)
    d = ImageDraw.Draw(img)
    f_label = font(28)
    f_quote = font(54, bold=True)
    f_arrow = font(80, bold=True)
    d.text((40, 60), "prompt", fill=(140, 140, 150), font=f_label)
    # Wrap prompt text to fit
    words = prompt.split()
    lines = []
    cur = ""
    max_w = size[0] - 80
    for w in words:
        test = (cur + " " + w).strip()
        bbox = d.textbbox((0, 0), test, font=f_quote)
        if (bbox[2] - bbox[0]) > max_w and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)
    y = 130
    line_h = 70
    d.text((40, y), '"', fill=(60, 60, 70), font=f_quote)
    y += 10
    for line in lines:
        d.text((40, y + 50), line, fill=(20, 20, 30), font=f_quote)
        y += line_h
    d.text((max_w - 40, y + 30), '"', fill=(60, 60, 70), font=f_quote)
    # Arrow at bottom right
    arrow_bbox = d.textbbox((0, 0), "→", font=f_arrow)
    arrow_w = arrow_bbox[2] - arrow_bbox[0]
    d.text((size[0] - arrow_w - 30, size[1] - 160),
            "→", fill=(60, 60, 70), font=f_arrow)
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


def encode_video(src: Path, mp4: Path, seconds: float | None = None,
                  bg="white", loop: int = 1):
    """Re-encode to 1920x1080 white-letterbox. Loop input N times if needed."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if loop > 1:
        cmd += ["-stream_loop", str(loop - 1)]
    cmd += [
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


def get_duration(mp4: Path) -> float:
    """Return duration in seconds via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp4)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def prompt_with_drawing(prompt: str, draw_mp4: Path, out_mp4: Path,
                          hold_seconds: float = 1.5):
    """Side-by-side: prompt panel (left) + drawing video (right).

    The drawing plays once at native duration, then holds its last frame
    for `hold_seconds`. Explicit -t avoids ffmpeg hanging on infinite inputs.
    """
    panel_png = WORK / f"_panel_{out_mp4.stem}.png"
    prompt_panel(prompt, panel_png, size=(720, 1080))
    total = get_duration(draw_mp4) + hold_seconds
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-loop", "1", "-t", str(total), "-i", str(panel_png),
        "-stream_loop", "0", "-i", str(draw_mp4),
        "-filter_complex",
        f"[1:v]tpad=stop_mode=clone:stop_duration={hold_seconds},"
        f"scale=1200:1080:force_original_aspect_ratio=decrease,"
        f"pad=1200:1080:(ow-iw)/2:(oh-ih)/2:white,setpts=PTS-STARTPTS[draw];"
        f"[0:v]scale=720:1080,setpts=PTS-STARTPTS[panel];"
        f"[panel][draw]hstack[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-t", str(total),
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, timeout=60)


# Original cross_token MP4 layout: 2708 wide x 1016 high
# = panels (3 x 896 + 2 x 10 pad) x (896 panel + 50 label band + 70 caption band)
# Crop top 896 rows = pure drawings, then re-label larger.
CROSS_TOKEN_PROMPTS = {
    "paris_eiffel":  "The Eiffel Tower is in...",
    "capital_japan": "The capital of Japan is...",
    "ocean_color":   "The colour of the ocean is...",
    "dog_thinking":  "I am thinking about a dog. Specifically, a...",
    "storm_village": "When the storm hit, the village...",
    "sad_funeral":   "The funeral was somber and...",
    "face_smile":    "I am picturing a smiling face with...",
    "triangle_geom": "Imagine a triangle inscribed in a...",
    "once_upon":     "Once upon a time in a kingdom...",
    "rainbow":       "After the rain, a rainbow appeared in the...",
}


def parse_xtoken_log(log_path: Path) -> dict[str, list[str]]:
    """Parse ovl_xtoken.log into {prompt_slug: [step0_text, step1_text, ...]}."""
    import re
    if not log_path.exists():
        return {}
    out: dict[str, list[str]] = {}
    current: list[str] | None = None
    pattern_step = re.compile(r"\[xtoken\]\s+step\s+\d+\s+text='(.*)'$")
    # Map prompt-first-words to slug
    PREFIX_TO_SLUG = {
        "The Eiffel Tower": "paris_eiffel",
        "The capital of Japan": "capital_japan",
        "The colour of the ocean": "ocean_color",
        "I am thinking about a dog": "dog_thinking",
        "When the storm hit": "storm_village",
        "The funeral was somber": "sad_funeral",
        "I am picturing a smiling face": "face_smile",
        "Imagine a triangle inscribed": "triangle_geom",
        "Once upon a time in a kingdom": "once_upon",
        "After the rain, a rainbow": "rainbow",
    }
    for line in log_path.read_text().splitlines():
        m = pattern_step.match(line.strip())
        if not m:
            continue
        text = m.group(1).replace("\\n", "\n")
        # If step 0, start a new list
        if "step  0" in line or "step 0" in line:
            slug = None
            for prefix, s in PREFIX_TO_SLUG.items():
                if text.startswith(prefix):
                    slug = s
                    break
            if slug is None:
                current = None
                continue
            current = []
            out[slug] = current
        if current is not None:
            current.append(text)
    return out


XTOKEN_STEP_TEXT: dict[str, list[str]] | None = None


def get_xtoken_step_text(slug: str) -> list[str]:
    global XTOKEN_STEP_TEXT
    if XTOKEN_STEP_TEXT is None:
        XTOKEN_STEP_TEXT = parse_xtoken_log(Path("runs/ovl_xtoken.log"))
    return XTOKEN_STEP_TEXT.get(slug, [])


def render_xtoken_footer_frames(steps: list[str], out_dir: Path,
                                 size=(1920, 220)) -> list[Path]:
    """Render one PNG per step with the BIG growing prompt text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    f_big = font(64, bold=True)
    f_small = font(28)
    for i, txt in enumerate(steps):
        img = Image.new("RGB", size, color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((30, 12), f"the model wrote (token {i+1}):",
                fill=(140, 140, 150), font=f_small)
        # Wrap text into lines
        lines = []
        for ln in txt.split("\n"):
            # word-wrap to fit
            words = ln.split(" ")
            cur = ""
            for w in words:
                test = (cur + " " + w).strip()
                bbox = d.textbbox((0, 0), test, font=f_big)
                if (bbox[2] - bbox[0]) > size[0] - 60 and cur:
                    lines.append(cur)
                    cur = w
                else:
                    cur = test
            if cur:
                lines.append(cur)
        # Render up to 2 lines centred vertically below the label
        for j, line in enumerate(lines[:2]):
            bbox = d.textbbox((0, 0), line, font=f_big)
            x = (size[0] - (bbox[2] - bbox[0])) // 2
            d.text((x, 60 + j * 75), line, fill=(20, 20, 30), font=f_big)
        p = out_dir / f"step_{i:02d}.png"
        img.save(p)
        paths.append(p)
    return paths


def cross_token_with_big_labels(slug: str, src: Path, out_mp4: Path):
    """Crop tiny baked-in labels from a cross_token MP4 and add a dynamic
    footer that types out each generated token live."""
    # Header
    header = WORK / f"_xt_hdr_{slug}.png"
    h_img = Image.new("RGB", (1920, 90), color=(255, 255, 255))
    hd = ImageDraw.Draw(h_img)
    f_hdr = font(56, bold=True)
    labels = ["L10 (mid)", "L24 (deep)", "L29 (deepest)"]
    col_w = 1920 // 3
    for i, lab in enumerate(labels):
        bbox = hd.textbbox((0, 0), lab, font=f_hdr)
        x = i * col_w + (col_w - (bbox[2] - bbox[0])) // 2
        hd.text((x, 18), lab, fill=(30, 30, 40), font=f_hdr)
    h_img.save(header)

    # Dynamic footer: one PNG per step, concatenated into a footer mp4
    # synced to the source video's step cadence.
    total = get_duration(src)
    steps = get_xtoken_step_text(slug) or [CROSS_TOKEN_PROMPTS.get(slug, slug)]
    n_steps = len(steps)
    step_dur = total / max(n_steps, 1)
    footer_pngs = render_xtoken_footer_frames(
        steps, WORK / f"_xt_ftr_{slug}_frames", size=(1920, 220))
    # Build a sequenced frame folder where each step's PNG is duplicated to
    # cover step_dur seconds at FPS, then feed via image2 demuxer.
    seq_dir = WORK / f"_xt_ftr_{slug}_seq"
    seq_dir.mkdir(parents=True, exist_ok=True)
    for f in seq_dir.glob("*.png"):
        f.unlink()
    import shutil
    idx = 1
    frames_per_step = max(int(round(step_dur * FPS)), 1)
    for png in footer_pngs:
        for _ in range(frames_per_step):
            shutil.copy(png, seq_dir / f"f_{idx:06d}.png")
            idx += 1
    footer_mp4 = WORK / f"_xt_ftr_{slug}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(FPS), "-i", str(seq_dir / "f_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-t", str(total), str(footer_mp4),
    ], check=True, timeout=60)

    # Layout: header (90) + drawings (cropped 0..896, scaled to 1920 wide -> 635 h)
    # + 135 white spacer + footer (220) = 1080
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-loop", "1", "-t", str(total), "-i", str(header),
        "-i", str(src),
        "-i", str(footer_mp4),
        "-filter_complex",
        f"[1:v]crop=2708:896:0:0,scale=1920:-2,setpts=PTS-STARTPTS[draw];"
        f"color=white:1920x135:duration={total}[pad];"
        f"[0:v][draw][pad][2:v]vstack=inputs=4[stacked];"
        f"[stacked]crop=1920:1080:0:0[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-t", str(total),
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, timeout=120)


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


# Hero prompts mapping concept -> phrasing used in v1 captions
HERO_PROMPTS = {
    "cat":      "I am thinking about a cat",
    "elephant": "I am thinking about an elephant",
    "sun":      "The sun is shining",
    "flower":   "Imagine a flower",
    "dog":      "I am thinking about a dog",
    "horse":    "I am thinking about a horse",
    "fish":     "Imagine a fish",
    "bird":     "I am picturing a bird",
    "tree":     "I am picturing a tree",
    "airplane": "I am picturing an airplane",
    "car":      "I am thinking about a car",
    "mountain": "I am picturing a mountain",
}

OOD_PROMPTS = {
    "infinity":      "infinity.",
    "love":          "the feeling of love.",
    "consciousness": "consciousness.",
    "god":           "god.",
    "death":         "death.",
    "pi":            "the number pi.",
    "memory":        "an old memory.",
    "freedom":       "the feeling of freedom.",
}


def main():
    parts = []

    # 0:00 silent hook
    title_card("what does a language model\n'see' inside?",
                "the only input below is the activation vector out of block h.\nno text. no labels.",
                WORK / "00_hook.png", bg=(8, 8, 18), big_size=80, small_size=32)
    encode_image(WORK / "00_hook.png", WORK / "00_hook.mp4", 4.0)
    parts.append(WORK / "00_hook.mp4")

    # 0:04 prompt -> drawing reveals (5 concepts)
    for slug in ["cat", "elephant", "flower", "dog", "horse"]:
        draw = OUT_DIR / "anim/L29" / f"{slug}.mp4"
        if not draw.exists():
            continue
        out = WORK / f"01_p_{slug}.mp4"
        prompt_with_drawing(HERO_PROMPTS[slug], draw, out, hold_seconds=1.5)
        parts.append(out)

    # 0:24 architecture card
    title_card("prompt   →   Qwen 3.5-4B   →   activation h   →   stroke decoder   →   drawing",
                "the drawing is the same regardless of how you phrase the prompt.\nit decodes the vector, not the text.",
                WORK / "02_arch.png", bg=(20, 30, 50), big_size=42, small_size=28)
    encode_image(WORK / "02_arch.png", WORK / "02_arch.mp4", 5.5)
    parts.append(WORK / "02_arch.mp4")

    # 0:30 cross-layer reveal (3 concepts, all 7 layers at once)
    title_card("the same activation, decoded at each layer",
                "left = shallow (L3). right = deep (L29). watch the idea sharpen.",
                WORK / "03_xl_title.png", bg=(15, 25, 40), big_size=46, small_size=28)
    encode_image(WORK / "03_xl_title.png", WORK / "03_xl_title.mp4", 3.0)
    parts.append(WORK / "03_xl_title.mp4")
    for slug in ["cat", "elephant", "flower"]:
        src = OUT_DIR / "cross_layer_anim" / f"{slug}.mp4"
        if src.exists():
            out = WORK / f"03_xl_{slug}.mp4"
            encode_video(src, out, loop=1)
            parts.append(out)

    # 0:51 morphs (v2_2 era — interpolating in activation space)
    title_card("interpolating between concepts",
                "average two activation vectors. decode the average. the model morphs.",
                WORK / "04_morph_title.png", bg=(20, 40, 30), big_size=48, small_size=28)
    encode_image(WORK / "04_morph_title.png", WORK / "04_morph_title.mp4", 3.0)
    parts.append(WORK / "04_morph_title.mp4")
    for morph in ["cat_to_elephant", "apple_to_pizza", "dog_to_horse"]:
        src = Path(f"artefacts/v2_2/morph/{morph}.mp4")
        if src.exists():
            out = WORK / f"04_m_{morph}.mp4"
            encode_video(src, out, seconds=4.5, loop=1)
            parts.append(out)

    # 1:08 per-token trajectory
    title_card("what is the model thinking as it writes?",
                "the model writes \"The Eiffel Tower is in...\" one word at a time.\n"
                "after each word, we read its activation at three different depths and draw it.",
                WORK / "05_xt_title.png", bg=(25, 25, 55), big_size=44, small_size=28)
    encode_image(WORK / "05_xt_title.png", WORK / "05_xt_title.mp4", 4.5)
    parts.append(WORK / "05_xt_title.mp4")
    for slug in ["paris_eiffel", "ocean_color"]:
        src = OUT_DIR / "cross_token" / f"{slug}.mp4"
        if src.exists():
            out = WORK / f"05_xt_{slug}.mp4"
            cross_token_with_big_labels(slug, src, out)
            parts.append(out)

    # 1:27 OOD reveal
    title_card("now things it was never trained to draw",
                "abstract concepts. emotions. math.\nthe decoder has to map them to its visual codebook.",
                WORK / "06_ood_title.png", bg=(40, 20, 40), big_size=46, small_size=28)
    encode_image(WORK / "06_ood_title.png", WORK / "06_ood_title.mp4", 3.5)
    parts.append(WORK / "06_ood_title.mp4")
    for slug in ["infinity", "love", "consciousness", "god"]:
        # Use static png as the "drawing" panel (no anim mp4 for these in our set)
        png = OUT_DIR / "ood/L29" / f"{slug}.png"
        mp4 = OUT_DIR / "ood/L29" / f"{slug}.mp4"
        prompt = OOD_PROMPTS[slug]
        if mp4.exists():
            out = WORK / f"06_o_{slug}.mp4"
            prompt_with_drawing(prompt, mp4, out, hold_seconds=1.5)
            parts.append(out)
        elif png.exists():
            tmp_mp4 = WORK / f"_tmp_{slug}.mp4"
            encode_image(png, tmp_mp4, 3.5, bg="white")
            out = WORK / f"06_o_{slug}.mp4"
            prompt_with_drawing(prompt, tmp_mp4, out, hold_seconds=0.5)
            parts.append(out)

    # 1:41 honesty card
    title_card("the decoder snaps to its nearest known concept",
                "if 'maths' and 'pizza' produce the same drawing, that tells us\nthose concepts live close in the model's geometry.",
                WORK / "07_honesty.png", bg=(35, 35, 35), fg=(255, 230, 180),
                big_size=42, small_size=28)
    encode_image(WORK / "07_honesty.png", WORK / "07_honesty.mp4", 5.0)
    parts.append(WORK / "07_honesty.mp4")

    # 1:46 old vs new
    title_card("v1 (Gemma 4 base) vs v3 (Qwen 3.5-4B)",
                "same concept, two foundations. the base model is most of the signal.",
                WORK / "08_oldnew_title.png", bg=(15, 15, 30), big_size=44, small_size=28)
    encode_image(WORK / "08_oldnew_title.png", WORK / "08_oldnew_title.mp4", 3.0)
    parts.append(WORK / "08_oldnew_title.mp4")
    old = Path("artefacts/v1_5/demo.mp4")
    if old.exists():
        out = WORK / "08_old.mp4"
        encode_video(old, out, seconds=8.0)
        parts.append(out)

    # 1:57 outro — single-line URL only
    title_card("github.com/AryaaSk/residual_stream_visual_decoder",
                "",
                WORK / "09_outro.png", bg=(15, 20, 35), big_size=46, small_size=36)
    encode_image(WORK / "09_outro.png", WORK / "09_outro.mp4", 4.0)
    parts.append(WORK / "09_outro.mp4")

    out = OUT_DIR / "demo_v2.mp4"
    print(f"[v2] {len(parts)} parts -> {out}")
    concat(parts, out)
    import json
    info = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", str(out)], capture_output=True, text=True)
    dur = float(json.loads(info.stdout)["format"]["duration"])
    print(f"[v2] duration: {dur:.1f}s  size: {out.stat().st_size//1024} KB")


if __name__ == "__main__":
    main()
