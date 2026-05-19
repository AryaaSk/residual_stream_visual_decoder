"""Deterministic stroke renderer.

Takes a sequence of `Stroke` objects (see `stroke_tokenizer.py`) and produces
two artefacts in a single pass:

    (a) a final 224×224 grayscale PNG (the canvas after all strokes), used as
        the AR's image input;
    (b) optionally, an animated MP4/GIF showing the drawing emerging stroke
        by stroke, for human inspection.

Pen-state semantic (simpler than SketchRNN's stroke-5 to avoid ambiguity):

    PEN_DOWN  this stroke's movement is INKED (line from old position to new).
    PEN_UP    this stroke's movement is NOT inked (just a pen jump).
    PEN_END   stop decoding (this stroke's movement is treated as PEN_UP).

Convention:
    - canvas: 224×224, grayscale, white background (255), black ink (0).
    - pen starts at canvas centre.
    - line width: 2 px, anti-aliased.
    - pen position is clamped to canvas bounds so the model can't drift
      arbitrarily far off-screen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from stroke_tokenizer import PEN_DOWN, PEN_END, PEN_UP, Stroke

DEFAULT_CANVAS_SIZE = 224
DEFAULT_LINE_WIDTH = 2
INK_COLOR = 0       # black
BG_COLOR = 255      # white


def _new_canvas(size: int) -> Image.Image:
    return Image.new("L", (size, size), color=BG_COLOR)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def render(
    strokes: Sequence[Stroke],
    *,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    line_width: int = DEFAULT_LINE_WIDTH,
    save_animation_path: str | Path | None = None,
    fps: int = 24,
    display_scale: float = 1.0,
) -> Image.Image:
    """Render a stroke sequence to a PIL Image.

    If `save_animation_path` is provided, also write an animation showing the
    drawing emerging stroke by stroke. Use ``.mp4`` extension for video,
    ``.gif`` for GIF. One frame per stroke.

    `display_scale` re-renders the same stroke data at an upscaled resolution
    by multiplying canvas_size, all stroke (Δx, Δy) offsets, and line_width
    by the scale factor. Lossless vector upscaling (no interpolation). Useful
    for producing display-quality artefacts while AR still sees the 224×224
    native resolution.

    The returned PIL Image is the final canvas (post-last-stroke).
    """
    s_size = int(round(canvas_size * display_scale))
    s_line = max(1, int(round(line_width * display_scale)))
    canvas = _new_canvas(s_size)
    draw = ImageDraw.Draw(canvas)
    pen_x = s_size / 2.0
    pen_y = s_size / 2.0
    frames: list[Image.Image] = []

    for s in strokes:
        sdx = s.dx * display_scale
        sdy = s.dy * display_scale
        new_x = _clamp(pen_x + sdx, 0, s_size - 1)
        new_y = _clamp(pen_y + sdy, 0, s_size - 1)

        # Per the semantic above, the pen_state ON THIS STROKE determines
        # whether THIS movement leaves ink.
        if s.pen == PEN_DOWN:
            draw.line([(pen_x, pen_y), (new_x, new_y)], fill=INK_COLOR, width=s_line)

        pen_x, pen_y = new_x, new_y

        if save_animation_path is not None:
            frames.append(canvas.copy())

        if s.pen == PEN_END:
            break

    if save_animation_path is not None:
        _write_animation(frames, Path(save_animation_path), fps)

    return canvas


def render_to_array(strokes: Sequence[Stroke], **kwargs) -> np.ndarray:
    """Same as `render` but returns an (H, W) uint8 numpy array."""
    img = render(strokes, **kwargs)
    return np.asarray(img, dtype=np.uint8)


def _write_animation(frames: list[Image.Image], path: Path, fps: int) -> None:
    """Write frames to MP4 or GIF based on file extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if not frames:
        # write a single blank frame so the file exists
        blank = _new_canvas(frames[0].size[0] if frames else DEFAULT_CANVAS_SIZE)
        frames = [blank]
    if ext == ".gif":
        duration_ms = int(1000 / fps)
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        return
    if ext in (".mp4", ".webm", ".mkv"):
        # Use imageio with the ffmpeg plugin
        import imageio.v3 as iio
        arr = np.stack([np.asarray(f, dtype=np.uint8) for f in frames], axis=0)
        # imageio mp4 writer wants RGB; replicate grayscale into 3 channels
        if arr.ndim == 3:
            arr = np.stack([arr, arr, arr], axis=-1)
        iio.imwrite(path, arr, fps=fps, codec="libx264", macro_block_size=1)
        return
    raise ValueError(f"Unsupported animation extension: {ext}")


# Convenience: render directly from a token-id list given a StrokeVocab
def render_token_ids(
    token_ids: Iterable[int],
    vocab,  # StrokeVocab (imported lazily to avoid cycles)
    **render_kwargs,
) -> Image.Image:
    """Decode token ids then render."""
    strokes = vocab.decode_tokens(token_ids)
    return render(strokes, **render_kwargs)
