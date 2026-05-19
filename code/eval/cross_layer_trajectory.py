"""Cross-layer trajectory: for a single prompt, render drawings at all anchor layers
and assemble into one MP4 sequence showing thought crystallise across depth.

For each hero prompt:
  layer 3  → drawing  (early features)
  layer 12 → drawing  (composition)
  layer 24 → drawing  (commitment)
  ...

Output: one MP4 per prompt with layer captions.

Requires per-layer AV checkpoints in a single directory tree like:
    checkpoints/v1/L03/final/av_ckpt.pt
    checkpoints/v1/L12/final/av_ckpt.pt
    checkpoints/v1/L24/final/av_ckpt.pt

Usage:
    python code/eval/cross_layer_trajectory.py \
        --ckpts-root checkpoints/v1 \
        --layers 3 12 24 \
        --out-dir artefacts/cross_layer
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render import render as stroke_render  # noqa: E402
from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402


HERO_PROMPTS = [
    ("capital_france",   "The capital of France is"),
    ("eiffel_tower",     "The Eiffel Tower is in Paris."),
    ("cat",              "I am thinking about a cat."),
    ("triangle",         "Imagine a triangle inscribed in a circle."),
    ("storm",            "When the storm hit, the village"),
    ("paris_lights",     "Paris, the city of lights, is famous for the Eiffel"),
    ("math",             "What is 47 + 38?"),
    ("sad",              "I am thinking about deep sadness."),
    ("code",             "def fibonacci(n):"),
    ("face",             "I am picturing a smiling face."),
]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts-root", type=Path, required=True,
                   help="Dir containing per-layer subdirs like LNN/final/av_ckpt.pt")
    p.add_argument("--layers", type=int, nargs="+", default=[3, 12, 24])
    p.add_argument("--model-id", default="google/gemma-4-e2b-it")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--av-max-tokens", type=int, default=200)
    p.add_argument("--display-scale", type=float, default=4.0)
    p.add_argument("--out-dir", type=Path, default=Path("artefacts/cross_layer"))
    p.add_argument("--per-layer-frames", type=int, default=24,
                   help="Frames to hold each layer's drawing (at fps=24, this is 1s)")
    p.add_argument("--crossfade-frames", type=int, default=12)
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # For each layer, sample drawings for every prompt, then release that AV.
    per_layer_drawings: dict[int, dict[str, Image.Image]] = {layer: {} for layer in args.layers}
    for layer in args.layers:
        ck = args.ckpts_root / f"L{layer:02d}" / "final"
        if not (ck / "av_ckpt.pt").exists():
            print(f"[crosslayer] WARN: {ck/'av_ckpt.pt'} missing, skipping layer {layer}", flush=True)
            continue
        print(f"\n[crosslayer] loading L{layer:02d} AV from {ck}", flush=True)
        av = StrokeDecoder.from_ckpt(ck, model_id=args.model_id)
        av.model.eval()
        device = av.device()
        for slug, prompt in HERO_PROMPTS:
            enc = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
            out = av.model(**enc, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[layer][0, -1, :].detach()
            ids = av.generate_from_activation(h, layer_ell=layer, alpha=args.alpha,
                                              max_new_tokens=args.av_max_tokens, temperature=1.0)
            strokes = av.vocab.decode_tokens(ids.tolist())
            img = stroke_render(strokes, display_scale=args.display_scale)
            per_layer_drawings[layer][slug] = img
            print(f"  L{layer:02d} | {slug:18s} strokes={len(strokes)}", flush=True)
        del av; gc.collect(); torch.cuda.empty_cache()

    # Assemble cross-layer MP4 per prompt
    for slug, prompt in HERO_PROMPTS:
        frames_pil: list[Image.Image] = []
        captions: list[str] = []
        for layer in args.layers:
            if slug not in per_layer_drawings[layer]:
                continue
            frames_pil.append(per_layer_drawings[layer][slug])
            captions.append(f"layer {layer}")
        if not frames_pil:
            continue
        out_path = args.out_dir / f"{slug}.mp4"
        _write_layer_trajectory_mp4(
            frames_pil, captions, prompt, out_path,
            per_layer_frames=args.per_layer_frames,
            crossfade_frames=args.crossfade_frames, fps=args.fps,
        )
        print(f"[crosslayer] → {out_path}", flush=True)

    # HTML index
    html = ["<!doctype html><html><body><h1>Cross-layer trajectory videos</h1>",
            f"<p>For each prompt, the drawing rendered at layers {args.layers} are shown in sequence, with crossfades.</p>",
            "<div style='display:flex;flex-wrap:wrap;gap:1em'>"]
    for slug, prompt in HERO_PROMPTS:
        html.append(
            f"<div style='border:1px solid #ddd;padding:.6em;width:420px'>"
            f"<video src='{slug}.mp4' controls muted loop autoplay style='width:100%'></video><br>"
            f"<small><b>{slug}</b><br>{prompt}</small></div>"
        )
    html.append("</div></body></html>")
    (args.out_dir / "index.html").write_text("".join(html))
    print(f"[crosslayer] index: {args.out_dir / 'index.html'}", flush=True)


def _write_layer_trajectory_mp4(frames, layer_captions, prompt, out_path, *,
                                 per_layer_frames: int, crossfade_frames: int, fps: int):
    import imageio.v3 as iio
    if not frames:
        return
    W, H = frames[0].size
    PAD = 80  # caption strip
    canvas_size = (W, H + PAD)

    def overlay(img: Image.Image, layer_caption: str) -> np.ndarray:
        canvas = Image.new("L", canvas_size, color=255)
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        except Exception:
            font_small = ImageFont.load_default()
            font_big = font_small
        # prompt line + layer line
        max_chars = max(1, W // 9)
        wrapped_prompt = prompt if len(prompt) <= max_chars else prompt[:max_chars - 1] + "…"
        draw.text((10, H + 10), wrapped_prompt, fill=0, font=font_small)
        draw.text((10, H + 35), layer_caption, fill=0, font=font_big)
        rgb = np.stack([np.asarray(canvas, dtype=np.uint8)] * 3, axis=-1)
        return rgb

    out_frames: list[np.ndarray] = []
    for i, (img, cap) in enumerate(zip(frames, layer_captions)):
        arr = overlay(img, cap)
        for _ in range(per_layer_frames):
            out_frames.append(arr)
        if i < len(frames) - 1 and crossfade_frames > 0:
            next_arr = overlay(frames[i + 1], layer_captions[i + 1])
            for j in range(1, crossfade_frames + 1):
                alpha = j / (crossfade_frames + 1)
                blended = (arr * (1 - alpha) + next_arr * alpha).astype(np.uint8)
                out_frames.append(blended)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(out_frames, axis=0), fps=fps,
                codec="libx264", macro_block_size=1)


if __name__ == "__main__":
    main()
