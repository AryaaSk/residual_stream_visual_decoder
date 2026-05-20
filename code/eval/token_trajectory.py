"""Per-token trajectory: animate the drawing as Gemma generates a sequence.

For each hero prompt:
  step 0: feed `prompt` to Gemma 4, extract h at layer ℓ, draw via AV
  step 1: append Gemma's predicted next token to prompt, extract h again, draw
  step 2: ...
  step N: stop

Render the N+1 drawings into a single MP4 with each frame captioned by the
partial generated text. The MP4 shows the model's "thought" morphing as it
commits to each token.

Cross-fade between adjacent frames is added so the animation is smooth.

Usage:
    python code/eval/token_trajectory.py \
        --av-ckpt checkpoints/v1/L12/final \
        --ar-ckpt checkpoints/v1/L12/final \
        --layer 12 --max-gen-tokens 18 \
        --out-dir artefacts/trajectory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render import render as stroke_render  # noqa: E402
from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402


HERO_PROMPTS = [
    ("paris_eiffel",  "The Eiffel Tower is in"),
    ("capital_japan", "The capital of Japan is"),
    ("ocean_color",   "The colour of the ocean is"),
    ("dog_thinking",  "I am thinking about a dog. Specifically, a"),
    ("storm_village", "When the storm hit, the village"),
    ("sad_funeral",   "The funeral was somber and"),
    ("math_simple",   "What is 47 + 38? The answer is"),
    ("code_fib",      "def fibonacci(n):"),
    ("face_smile",    "I am picturing a smiling face with"),
    ("triangle_geom", "Imagine a triangle inscribed in a"),
]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max-gen-tokens", type=int, default=18)
    p.add_argument("--av-max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--display-scale", type=float, default=4.0)
    p.add_argument("--out-dir", type=Path, default=Path("artefacts/trajectory"))
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--crossfade-frames", type=int, default=3)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[traj] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    device = av.device()

    for slug, prompt in HERO_PROMPTS:
        print(f"\n[traj] === {slug} === : {prompt!r}", flush=True)
        partial_tokens: list[int] = []
        partial_text = prompt
        frames: list[Image.Image] = []
        captions: list[str] = []

        # Initial activation: prompt only
        enc = av.tokenizer(partial_text, return_tensors="pt", add_special_tokens=True).to(device)
        for step in range(args.max_gen_tokens):
            # Forward to get current h_ℓ
            out = av.model(**enc, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[args.layer][0, -1, :].detach()

            # AV draws from this h
            ids = av.generate_from_activation(
                h, layer_ell=args.layer, alpha=args.alpha,
                max_new_tokens=args.av_max_tokens, temperature=args.temperature,
            )
            strokes = av.vocab.decode_tokens(ids.tolist())
            img = stroke_render(strokes, display_scale=args.display_scale)
            frames.append(img)
            captions.append(partial_text)
            print(f"  step {step:2d}: tokens so far ({len(partial_tokens)}): {partial_text[-80:]!r}  strokes={len(strokes)}",
                  flush=True)

            # Take Gemma's next predicted token (greedy).
            # CRITICAL: mask out the 262 stroke-vocab IDs so the caption doesn't
            # leak "<DX_092><PEN_UP>..." tokens (v1.1 bug fix). The AV's
            # extended vocab includes our stroke tokens; the TARGET model should
            # only ever emit normal Gemma text tokens.
            next_logits = out.logits[0, -1, :].clone()
            stroke_ids = list(av.vocab.name_to_id.values())
            next_logits[stroke_ids] = float("-inf")
            next_id = int(next_logits.argmax().item())
            partial_tokens.append(next_id)
            decoded = av.tokenizer.decode([next_id])
            partial_text = partial_text + decoded
            # Stop if EOS
            if next_id == av.tokenizer.eos_token_id:
                break
            # Re-tokenise the full string (cheap; avoids cache issues)
            enc = av.tokenizer(partial_text, return_tensors="pt", add_special_tokens=True).to(device)

        out_path = args.out_dir / f"{slug}.mp4"
        _write_captioned_mp4(frames, captions, out_path, fps=args.fps,
                             crossfade=args.crossfade_frames)
        print(f"[traj] → {out_path}", flush=True)

    # HTML index
    html = ["<!doctype html><html><body><h1>Per-token trajectory videos</h1>",
            "<p>Each MP4 shows the AV's drawing morphing as Gemma generates each next token of the prompt.</p>",
            "<div style='display:flex;flex-wrap:wrap;gap:1em'>"]
    for slug, prompt in HERO_PROMPTS:
        html.append(
            f"<div style='border:1px solid #ddd;padding:.6em;width:380px'>"
            f"<video src='{slug}.mp4' controls muted loop autoplay style='width:100%'></video><br>"
            f"<small><b>{slug}</b><br>{prompt}</small></div>"
        )
    html.append("</div></body></html>")
    (args.out_dir / "index.html").write_text("".join(html))
    print(f"[traj] index: {args.out_dir / 'index.html'}", flush=True)


def _write_captioned_mp4(frames, captions, out_path, *, fps: int, crossfade: int):
    """Write the frames as MP4 with the partial-text caption overlaid on each frame.

    Each frame is held for `1/fps` seconds, and between adjacent frames we add
    `crossfade` interpolated frames for smooth transitions.
    """
    import imageio.v3 as iio
    if not frames:
        return
    W, H = frames[0].size
    PAD = 60  # caption strip height
    canvas_size = (W, H + PAD)

    def caption_overlay(img: Image.Image, text: str) -> np.ndarray:
        canvas = Image.new("L", canvas_size, color=255)
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except Exception:
            font = ImageFont.load_default()
        # wrap text if too long
        wrapped = text
        max_chars = max(1, W // 11)
        if len(wrapped) > max_chars:
            wrapped = wrapped[: max_chars - 1] + "…"
        draw.text((10, H + 15), wrapped, fill=0, font=font)
        rgb = np.stack([np.asarray(canvas, dtype=np.uint8)] * 3, axis=-1)
        return rgb

    out_frames: list[np.ndarray] = []
    for i, (img, cap) in enumerate(zip(frames, captions)):
        arr = caption_overlay(img, cap)
        out_frames.append(arr)
        # Crossfade to next
        if i < len(frames) - 1 and crossfade > 0:
            next_arr = caption_overlay(frames[i + 1], captions[i + 1])
            for j in range(1, crossfade + 1):
                alpha = j / (crossfade + 1)
                blended = (arr * (1 - alpha) + next_arr * alpha).astype(np.uint8)
                out_frames.append(blended)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(out_frames, axis=0), fps=fps,
                codec="libx264", macro_block_size=1)


if __name__ == "__main__":
    main()
