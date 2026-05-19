"""Cartesian stroke-5 tokenizer with 128-bin quantisation.

A drawing is represented as a sequence of (Δx, Δy, pen_state) triples,
quantised into integer tokens. Total vocabulary additions = 262 tokens:

    128 Δx_bin tokens     (horizontal offset, -64 to +63 px)
    128 Δy_bin tokens     (vertical offset, -64 to +63 px)
      3 pen_state tokens  (DOWN, UP, END)
      2 bracket tokens    (<DRAW>, </DRAW>)
      1 injection token   (<ACT_TOKEN>)
    ───
    262 new tokens

Each stroke is exactly 3 tokens: [Δx_bin, Δy_bin, pen_state].

Token IDs are assigned by add_to_tokenizer(), returning a STROKE_VOCAB dict
mapping symbolic names to integer IDs. All other code should reference
tokens via this dict, never by hardcoded ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, NamedTuple

# Canonical token names (string form). IDs assigned at registration time.
N_DX_BINS = 128
N_DY_BINS = 128
DX_RANGE = (-64, 64)  # half-open: [-64, 64) → 128 bins of width 1
DY_RANGE = (-64, 64)

PEN_DOWN = "<PEN_DOWN>"
PEN_UP = "<PEN_UP>"
PEN_END = "<PEN_END>"
DRAW_OPEN = "<DRAW>"
DRAW_CLOSE = "</DRAW>"
ACT_TOKEN = "<ACT_TOKEN>"


def dx_token_name(bin_idx: int) -> str:
    """Symbolic name of the Δx-bin token at index `bin_idx` (0..127)."""
    assert 0 <= bin_idx < N_DX_BINS, f"dx bin {bin_idx} out of range [0, {N_DX_BINS})"
    return f"<DX_{bin_idx:03d}>"


def dy_token_name(bin_idx: int) -> str:
    """Symbolic name of the Δy-bin token at index `bin_idx` (0..127)."""
    assert 0 <= bin_idx < N_DY_BINS, f"dy bin {bin_idx} out of range [0, {N_DY_BINS})"
    return f"<DY_{bin_idx:03d}>"


def all_stroke_token_names() -> list[str]:
    """Full list of 262 new tokens to add to a Gemma tokenizer."""
    names: list[str] = []
    names.extend(dx_token_name(i) for i in range(N_DX_BINS))
    names.extend(dy_token_name(i) for i in range(N_DY_BINS))
    names.extend([PEN_DOWN, PEN_UP, PEN_END, DRAW_OPEN, DRAW_CLOSE, ACT_TOKEN])
    return names


# Quantisation helpers

def quantise_dx(dx: float) -> int:
    """Quantise a signed horizontal offset (pixels) into a bin index 0..127.

    Clamps to range [-64, 63]. Bin 64 represents Δx=0.
    """
    clamped = max(DX_RANGE[0], min(DX_RANGE[1] - 1, int(round(dx))))
    return clamped - DX_RANGE[0]  # shift so range starts at 0


def dequantise_dx(bin_idx: int) -> int:
    """Inverse of `quantise_dx`. Returns the centre of the bin in pixels."""
    return bin_idx + DX_RANGE[0]


def quantise_dy(dy: float) -> int:
    clamped = max(DY_RANGE[0], min(DY_RANGE[1] - 1, int(round(dy))))
    return clamped - DY_RANGE[0]


def dequantise_dy(bin_idx: int) -> int:
    return bin_idx + DY_RANGE[0]


# Stroke datatype

class Stroke(NamedTuple):
    """A single pen movement: relative offset + pen state at the end."""
    dx: int  # already quantised pixels (i.e., dequantised bin centre)
    dy: int
    pen: str  # one of PEN_DOWN, PEN_UP, PEN_END


def stroke_to_token_names(s: Stroke) -> tuple[str, str, str]:
    """Convert a Stroke into a triple of symbolic token names."""
    return (
        dx_token_name(quantise_dx(s.dx)),
        dy_token_name(quantise_dy(s.dy)),
        s.pen,
    )


@dataclass
class StrokeVocab:
    """Mapping between symbolic stroke-token names and integer IDs.

    Built by `add_stroke_tokens_to_tokenizer` after `tokenizer.add_special_tokens`
    or `tokenizer.add_tokens` returns new IDs.
    """
    name_to_id: dict[str, int]
    id_to_name: dict[int, str]

    @classmethod
    def from_name_to_id(cls, name_to_id: dict[str, int]) -> "StrokeVocab":
        return cls(name_to_id=name_to_id, id_to_name={v: k for k, v in name_to_id.items()})

    def encode_stroke(self, s: Stroke) -> tuple[int, int, int]:
        names = stroke_to_token_names(s)
        return (self.name_to_id[names[0]], self.name_to_id[names[1]], self.name_to_id[names[2]])

    def encode_drawing(self, strokes: Iterable[Stroke]) -> list[int]:
        """Encode a sequence of strokes into a token-id list, wrapped in <DRAW>...</DRAW>."""
        ids: list[int] = [self.name_to_id[DRAW_OPEN]]
        for s in strokes:
            ids.extend(self.encode_stroke(s))
        ids.append(self.name_to_id[DRAW_CLOSE])
        return ids

    def decode_tokens(self, token_ids: Iterable[int]) -> list[Stroke]:
        """Decode a stroke-token id list into Strokes.

        Malformed sequences (out-of-order tokens, partial triples, stray
        non-stroke tokens) are tolerated: the decoder skips invalid runs
        and returns the strokes it successfully recovered.

        For the malformation count too, call ``decode_tokens_with_stats``.

        The decoder terminates at PEN_END (consumed as the last stroke) or
        </DRAW> (dropped, not a stroke), whichever comes first.
        """
        strokes, _ = self.decode_tokens_with_stats(token_ids)
        return strokes

    def decode_tokens_with_stats(self, token_ids: Iterable[int]) -> tuple[list[Stroke], int]:
        """Like ``decode_tokens`` but also returns the malformation count.

        Strategy: maintain a triple buffer with expected slot kinds in order
        ['dx', 'dy', 'pen']. A token that doesn't match the expected slot is
        skipped (with malformed++) and the buffer is NOT advanced. This
        resyncs gracefully if the model emits an out-of-order token: the next
        properly-typed token completes the triple.
        """
        EXPECTED = ("dx", "dy", "pen")
        strokes: list[Stroke] = []
        triple_buffer: list[tuple[str, str]] = []
        malformed = 0

        for tid in token_ids:
            name = self.id_to_name.get(tid)
            if name is None:
                continue
            if name == DRAW_OPEN:
                triple_buffer.clear()
                continue
            if name == DRAW_CLOSE:
                if triple_buffer:
                    malformed += 1
                triple_buffer.clear()
                break

            kind = _classify_token(name)
            if kind is None:
                malformed += 1
                continue

            expected = EXPECTED[len(triple_buffer)]
            if kind != expected:
                # Wrong kind for this slot. Skip without advancing.
                malformed += 1
                continue

            triple_buffer.append((kind, name))

            if len(triple_buffer) == 3:
                ok = _consume_triple(triple_buffer, strokes)
                if not ok:
                    malformed += 1
                if strokes and strokes[-1].pen == PEN_END:
                    break

        return strokes, malformed


def _classify_token(name: str) -> str | None:
    if name.startswith("<DX_"):
        return "dx"
    if name.startswith("<DY_"):
        return "dy"
    if name in (PEN_DOWN, PEN_UP, PEN_END):
        return "pen"
    return None


def _consume_triple(buf: list[tuple[str, str]], out: list[Stroke]) -> bool:
    """If `buf` is exactly [dx, dy, pen], append a Stroke to `out` and clear `buf`.

    Returns True if a valid stroke was consumed, False if the triple was malformed
    (in which case `buf` is cleared anyway).
    """
    if len(buf) != 3:
        return False
    kinds = tuple(k for k, _ in buf)
    if kinds != ("dx", "dy", "pen"):
        buf.clear()
        return False
    dx_name, dy_name, pen_name = (name for _, name in buf)
    dx_bin = int(dx_name[len("<DX_"):-1])
    dy_bin = int(dy_name[len("<DY_"):-1])
    out.append(Stroke(dx=dequantise_dx(dx_bin), dy=dequantise_dy(dy_bin), pen=pen_name))
    buf.clear()
    return True


def add_stroke_tokens_to_tokenizer(tokenizer) -> StrokeVocab:
    """Register the 262 stroke tokens with a HuggingFace tokenizer.

    Returns a StrokeVocab built from the freshly-assigned IDs.
    The caller is responsible for calling
    ``model.resize_token_embeddings(len(tokenizer))`` after this.
    """
    names = all_stroke_token_names()
    added = tokenizer.add_tokens(names, special_tokens=True)
    # add_tokens returns count added; some may have collided. Rebuild mapping by id lookup.
    name_to_id: dict[str, int] = {}
    for n in names:
        tid = tokenizer.convert_tokens_to_ids(n)
        if tid is None or (isinstance(tid, int) and tid < 0):
            raise RuntimeError(f"Failed to register stroke token {n} in tokenizer")
        name_to_id[n] = tid
    return StrokeVocab.from_name_to_id(name_to_id)
