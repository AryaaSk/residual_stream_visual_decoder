"""For each generated token in a prompt continuation, render drawings at MULTIPLE
layers (L10, L24, L29). Side-by-side panels morph as the model reads each word.

Single GPU. Loads all per-layer AVs (each ~28 MB ckpt + base Qwen).
But Qwen is shared as the BASE — we just swap the per-layer LoRA + projector + vocab.

Actually it's easier to load one AV per call, generate token, extract h at all
layers via output_hidden_states, then render at each layer with the matching AV.
Memory will be ~3x AV ckpts. Let's see.

Simpler: for each token position, use the GENERATIVE model to get the partial
text + tokens. Then for each layer L, use the matching AV to draw from h at that
position.

Output: artefacts/v3/viral/cross_token/{slug}.mp4 (3-panel MP4 morphing)
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

from verbalizer.stroke_decoder import StrokeDecoder
from render import render as stroke_render


PROMPTS = [
    ("paris_eiffel",  "The Eiffel Tower is in"),
    ("capital_japan", "The capital of Japan is"),
    ("ocean_color",   "The colour of the ocean is"),
    ("dog_thinking",  "I am thinking about a dog. Specifically, a"),
    ("storm_village", "When the storm hit, the village"),
    ("sad_funeral",   "The funeral was somber and"),
    ("face_smile",    "I am picturing a smiling face with"),
    ("triangle_geom", "Imagine a triangle inscribed in a"),
    ("once_upon",     "Once upon a time in a kingdom"),
    ("rainbow",       "After the rain, a rainbow appeared in the"),
]

# AVs to load (layer, ckpt dir, label)
AVS = [
    (10, "checkpoints/overnight/L10_filtered/final", "L10 filtered"),
    (24, "checkpoints/overnight/L24/final",          "L24"),
    (29, "checkpoints/v2_0/L29/final",                "L29"),
]


def label_panel(img, label, sub=""):
    W, H = img.size
    PAD = 50
    out = Image.new("RGB", (W, H + PAD), color=(255, 255, 255))
    out.paste(img.convert("RGB"), (0, 0))
    d = ImageDraw.Draw(out)
    try:
        f_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        f_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        f_big = ImageFont.load_default()
        f_small = f_big
    d.text((10, H + 6), label, fill=(0, 0, 0), font=f_big)
    if sub:
        d.text((10, H + 32), sub, fill=(100, 100, 100), font=f_small)
    return out


def hstack(panels, pad=8):
    W = panels[0].size[0]
    H = panels[0].size[1]
    out = Image.new("RGB", (W * len(panels) + pad * (len(panels) - 1), H), color=(255, 255, 255))
    for i, p in enumerate(panels):
        out.paste(p, (i * (W + pad), 0))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--max-gen-tokens", type=int, default=15)
    p.add_argument("--av-max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--display-scale", type=float, default=4.0)
    p.add_argument("--out-dir", type=Path, default=Path("artefacts/v3/viral/cross_token"))
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--crossfade", type=int, default=2)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load AVs (and one Qwen for next-token generation; we'll use the first AV's
    # Qwen which has the stroke vocab — masks needed)
    print(f"[xtoken] loading {len(AVS)} per-layer AVs ...", flush=True)
    avs = []
    for layer, ckpt, label in AVS:
        print(f"[xtoken]   loading L{layer} from {ckpt}", flush=True)
        av = StrokeDecoder.from_ckpt(Path(ckpt), model_id=args.model_id)
        av.model.eval()
        avs.append({"layer": layer, "av": av, "label": label})

    # We'll use the FIRST AV's model for next-token generation (it has the full
    # original vocab + stroke vocab; we mask stroke ids during generation).
    gen_av = avs[0]["av"]
    device = gen_av.device()
    stroke_ids = list(gen_av.vocab.name_to_id.values())

    for slug, prompt in PROMPTS:
        print(f"\n[xtoken] === {slug}: {prompt!r} ===", flush=True)
        partial_text = prompt
        frames = []
        captions = []
        with torch.no_grad():
            for step in range(args.max_gen_tokens):
                # Forward through gen_av (which is one AV with full hidden_states)
                enc = gen_av.tokenizer(partial_text, return_tensors="pt", add_special_tokens=True).to(device)
                out = gen_av.model(**enc, output_hidden_states=True, use_cache=False)
                # Per-layer h at last token position
                last_pos = enc["input_ids"].shape[1] - 1
                panels = []
                for av_info in avs:
                    L = av_info["layer"]
                    h = out.hidden_states[L][0, last_pos, :].detach()
                    # Generate drawing from this h using the matching layer's AV
                    ids = av_info["av"].generate_from_activation(
                        h, layer_ell=L,
                        max_new_tokens=args.av_max_tokens,
                        temperature=args.temperature, top_k=args.top_k,
                    )
                    strokes = av_info["av"].vocab.decode_tokens(ids.tolist())
                    img = stroke_render(strokes, display_scale=args.display_scale)
                    panels.append(label_panel(img, av_info["label"], f"{len(strokes)} strokes"))
                # Stack the 3 panels horizontally
                strip = hstack(panels, pad=10)
                frames.append(strip)
                captions.append(partial_text)
                print(f"[xtoken]   step {step:2d}  text={partial_text[-60:]!r}", flush=True)
                # Generate next token (masking stroke ids)
                next_logits = out.logits[0, -1, :].clone()
                next_logits[stroke_ids] = float("-inf")
                next_id = int(next_logits.argmax().item())
                if next_id == gen_av.tokenizer.eos_token_id:
                    break
                partial_text = partial_text + gen_av.tokenizer.decode([next_id])

        # Write MP4
        out_path = args.out_dir / f"{slug}.mp4"
        import imageio.v3 as iio
        W, H = frames[0].size
        PAD = 70
        canvas_size = (W, H + PAD)
        out_frames = []

        def overlay(img, cap):
            canvas = Image.new("RGB", canvas_size, color=(255, 255, 255))
            canvas.paste(img.convert("RGB"), (0, 0))
            draw = ImageDraw.Draw(canvas)
            try:
                f_cap = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            except Exception:
                f_cap = ImageFont.load_default()
            wrapped = cap if len(cap) <= 90 else cap[:89] + "…"
            draw.text((12, H + 22), wrapped, fill=(0, 0, 0), font=f_cap)
            return np.asarray(canvas, dtype=np.uint8)

        for i, (img, cap) in enumerate(zip(frames, captions)):
            arr = overlay(img, cap)
            out_frames.append(arr)
            # Hold each frame more
            for _ in range(args.fps - 1):
                out_frames.append(arr)
            if i < len(frames) - 1 and args.crossfade > 0:
                next_arr = overlay(frames[i + 1], captions[i + 1])
                for j in range(1, args.crossfade + 1):
                    a = j / (args.crossfade + 1)
                    out_frames.append((arr * (1 - a) + next_arr * a).astype(np.uint8))

        iio.imwrite(out_path, np.stack(out_frames, axis=0), fps=args.fps,
                    codec="libx264", macro_block_size=1)
        print(f"[xtoken] → {out_path}", flush=True)

    # Clean up
    for av_info in avs:
        del av_info["av"]
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
