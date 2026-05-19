"""Tests for the Cartesian stroke-5 tokenizer.

Run with:
    cd ~/Desktop/residual_stream_visual_decoder && PYTHONPATH=code python -m pytest code/tests -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stroke_tokenizer import (  # noqa: E402
    ACT_TOKEN,
    DRAW_CLOSE,
    DRAW_OPEN,
    N_DX_BINS,
    N_DY_BINS,
    PEN_DOWN,
    PEN_END,
    PEN_UP,
    Stroke,
    StrokeVocab,
    all_stroke_token_names,
    dequantise_dx,
    dequantise_dy,
    dx_token_name,
    dy_token_name,
    quantise_dx,
    quantise_dy,
    stroke_to_token_names,
)


def make_fake_vocab() -> StrokeVocab:
    """Build a StrokeVocab as if a tokenizer had assigned IDs 1000..1261."""
    names = all_stroke_token_names()
    name_to_id = {n: 1000 + i for i, n in enumerate(names)}
    return StrokeVocab.from_name_to_id(name_to_id)


def test_vocab_size():
    assert len(all_stroke_token_names()) == N_DX_BINS + N_DY_BINS + 3 + 2 + 1 == 262


def test_no_duplicate_token_names():
    names = all_stroke_token_names()
    assert len(set(names)) == len(names)


def test_quantise_dequantise_roundtrip_centre():
    assert quantise_dx(0) == 64
    assert dequantise_dx(64) == 0


def test_quantise_clamps_extremes():
    assert quantise_dx(1000) == 127  # clamped to max
    assert quantise_dx(-1000) == 0   # clamped to min


def test_quantise_dy_symmetric():
    for px in [-30, -10, 0, 10, 30]:
        bin_ = quantise_dy(px)
        assert dequantise_dy(bin_) == px


def test_stroke_to_token_names():
    s = Stroke(dx=10, dy=-5, pen=PEN_DOWN)
    names = stroke_to_token_names(s)
    assert names[0] == dx_token_name(quantise_dx(10))
    assert names[1] == dy_token_name(quantise_dy(-5))
    assert names[2] == PEN_DOWN


def test_encode_decode_roundtrip():
    vocab = make_fake_vocab()
    strokes_in = [
        Stroke(dx=10, dy=10, pen=PEN_DOWN),
        Stroke(dx=5, dy=-3, pen=PEN_DOWN),
        Stroke(dx=-8, dy=2, pen=PEN_UP),
        Stroke(dx=15, dy=0, pen=PEN_DOWN),
        Stroke(dx=0, dy=0, pen=PEN_END),
    ]
    ids = vocab.encode_drawing(strokes_in)
    # Expect: <DRAW>, (dx, dy, pen) × 5, </DRAW>
    assert len(ids) == 1 + 3 * 5 + 1
    assert ids[0] == vocab.name_to_id[DRAW_OPEN]
    assert ids[-1] == vocab.name_to_id[DRAW_CLOSE]

    strokes_out = vocab.decode_tokens(ids)
    # PEN_END stops decoding, so we drop the final stroke's continuation
    # but we should still get all 5 strokes
    assert len(strokes_out) == 5
    for s_in, s_out in zip(strokes_in, strokes_out):
        assert s_in.dx == s_out.dx
        assert s_in.dy == s_out.dy
        assert s_in.pen == s_out.pen


def test_decode_tolerates_foreign_tokens():
    vocab = make_fake_vocab()
    s = Stroke(dx=5, dy=5, pen=PEN_DOWN)
    ids = vocab.encode_drawing([s])
    # Insert a foreign token id in the middle
    polluted = ids[:1] + [9999, 9998] + ids[1:]
    out = vocab.decode_tokens(polluted)
    assert len(out) == 1
    assert out[0] == s


def test_decode_tolerates_partial_triple_at_end():
    vocab = make_fake_vocab()
    # Two complete strokes then a dangling Δx + Δy with no pen state
    ids = (
        [vocab.name_to_id[DRAW_OPEN]]
        + list(vocab.encode_stroke(Stroke(dx=1, dy=1, pen=PEN_DOWN)))
        + list(vocab.encode_stroke(Stroke(dx=2, dy=2, pen=PEN_DOWN)))
        + [vocab.name_to_id[dx_token_name(0)], vocab.name_to_id[dy_token_name(0)]]
    )
    out = vocab.decode_tokens(ids)
    assert len(out) == 2  # the dangling Δx, Δy without pen are dropped


def test_decode_stops_at_pen_end():
    vocab = make_fake_vocab()
    s1 = Stroke(dx=1, dy=1, pen=PEN_DOWN)
    s2 = Stroke(dx=2, dy=2, pen=PEN_END)
    s3 = Stroke(dx=3, dy=3, pen=PEN_DOWN)  # should be ignored, comes after PEN_END
    ids = (
        [vocab.name_to_id[DRAW_OPEN]]
        + list(vocab.encode_stroke(s1))
        + list(vocab.encode_stroke(s2))
        + list(vocab.encode_stroke(s3))
    )
    out = vocab.decode_tokens(ids)
    assert len(out) == 2
    assert out[1].pen == PEN_END


def test_decode_tolerates_misordered_triple():
    """A pen token where a Δx is expected should be skipped, not crash."""
    vocab = make_fake_vocab()
    bad = [
        vocab.name_to_id[DRAW_OPEN],
        vocab.name_to_id[PEN_DOWN],  # out-of-place pen at start of triple
        vocab.name_to_id[dx_token_name(64)],
        vocab.name_to_id[dy_token_name(64)],
        vocab.name_to_id[PEN_DOWN],
        vocab.name_to_id[DRAW_CLOSE],
    ]
    out = vocab.decode_tokens(bad)
    # The misordered first triple is discarded; the second forms a valid stroke
    assert len(out) == 1


def test_all_token_names_distinct_kinds():
    # Δx tokens follow the <DX_NNN> pattern
    assert dx_token_name(0) == "<DX_000>"
    assert dx_token_name(127) == "<DX_127>"
    assert dy_token_name(0) == "<DY_000>"
    # Special tokens are present
    names = all_stroke_token_names()
    for special in (PEN_DOWN, PEN_UP, PEN_END, DRAW_OPEN, DRAW_CLOSE, ACT_TOKEN):
        assert special in names
