"""Tests for the deterministic stroke renderer."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render import (  # noqa: E402
    BG_COLOR,
    DEFAULT_CANVAS_SIZE,
    INK_COLOR,
    render,
    render_to_array,
)
from stroke_tokenizer import (  # noqa: E402
    PEN_DOWN,
    PEN_END,
    PEN_UP,
    Stroke,
)


def test_empty_strokes_returns_blank_canvas():
    img = render([])
    arr = np.asarray(img)
    assert arr.shape == (DEFAULT_CANVAS_SIZE, DEFAULT_CANVAS_SIZE)
    assert (arr == BG_COLOR).all()


def test_render_is_deterministic():
    strokes = [
        Stroke(dx=10, dy=0, pen=PEN_DOWN),
        Stroke(dx=10, dy=0, pen=PEN_DOWN),
        Stroke(dx=0, dy=10, pen=PEN_DOWN),
        Stroke(dx=0, dy=0, pen=PEN_END),
    ]
    a = render_to_array(strokes)
    b = render_to_array(strokes)
    assert (a == b).all()


def test_pen_down_draws_ink():
    strokes = [
        Stroke(dx=20, dy=0, pen=PEN_DOWN),
        Stroke(dx=0, dy=0, pen=PEN_END),
    ]
    arr = render_to_array(strokes)
    # At least one pixel should be ink
    assert (arr == INK_COLOR).any()


def test_pen_up_at_start_no_ink():
    """First stroke with no preceding PEN_DOWN should NOT leave ink (pen_down starts False)."""
    strokes = [
        Stroke(dx=20, dy=0, pen=PEN_UP),
        Stroke(dx=0, dy=0, pen=PEN_END),
    ]
    arr = render_to_array(strokes)
    assert (arr == BG_COLOR).all()


def test_pen_state_transitions():
    """PEN_DOWN inks this stroke; PEN_UP does not. Verify both behaviours."""
    strokes = [
        Stroke(dx=30, dy=0, pen=PEN_DOWN),     # inked: centre to centre+30
        Stroke(dx=30, dy=0, pen=PEN_UP),       # NOT inked: pen jumps to centre+60
        Stroke(dx=30, dy=0, pen=PEN_DOWN),     # inked: centre+60 to centre+90
        Stroke(dx=0, dy=0, pen=PEN_END),
    ]
    arr = render_to_array(strokes)
    centre = DEFAULT_CANVAS_SIZE // 2
    # ink exists in the first inked segment (around column centre+15)
    assert (arr[centre - 2: centre + 3, centre + 15] == INK_COLOR).any()
    # ink does NOT exist in the pen-up segment (around column centre+45)
    assert (arr[centre - 2: centre + 3, centre + 45] == BG_COLOR).all()
    # ink EXISTS in the second inked segment (around column centre+75)
    assert (arr[centre - 2: centre + 3, centre + 75] == INK_COLOR).any()


def test_pen_clamps_to_canvas():
    """A huge move shouldn't crash; pen position is clamped."""
    strokes = [Stroke(dx=10000, dy=10000, pen=PEN_DOWN), Stroke(dx=0, dy=0, pen=PEN_END)]
    arr = render_to_array(strokes)
    assert arr.shape == (DEFAULT_CANVAS_SIZE, DEFAULT_CANVAS_SIZE)


def test_render_produces_png(tmp_path):
    """Sanity: render output is a valid PIL Image."""
    strokes = [Stroke(dx=20, dy=20, pen=PEN_DOWN), Stroke(dx=0, dy=0, pen=PEN_END)]
    img = render(strokes)
    out_path = tmp_path / "test.png"
    img.save(out_path)
    assert out_path.exists() and out_path.stat().st_size > 0


def test_render_with_animation_gif(tmp_path):
    """Animation path should produce a GIF file."""
    strokes = [
        Stroke(dx=10, dy=0, pen=PEN_DOWN),
        Stroke(dx=10, dy=0, pen=PEN_DOWN),
        Stroke(dx=0, dy=10, pen=PEN_DOWN),
        Stroke(dx=0, dy=0, pen=PEN_END),
    ]
    gif_path = tmp_path / "anim.gif"
    img = render(strokes, save_animation_path=str(gif_path))
    assert gif_path.exists() and gif_path.stat().st_size > 0
    # final canvas should equal the rendering without animation
    final = render_to_array(strokes)
    assert (np.asarray(img) == final).all()
