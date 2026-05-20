"""Cross-layer 4-panel video (the v2.2 centerpiece).

For each hero prompt, render best-of-N CLIP-ranked drawings at L3 / L10 / L20 / L29,
then assemble a 4-panel side-by-side MP4: viewer sees the concept "crystallise
across depth" — L3 abstract, L29 crisp (hopefully).

Requires per-layer AV checkpoints in:
    <ckpts-root>/L<NN>/final/av_ckpt.pt   for every NN in --layers

Output:
    <out-dir>/{slug}.mp4              # 4-panel sequence
    <out-dir>/{slug}_strip.png        # static 4-panel strip
    <out-dir>/grid.png                # all prompts × all layers, one big grid
    <out-dir>/index.html              # HTML preview index
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder
from render import render as stroke_render


DEFAULT_PROMPTS = [
    ("cat",      "I am thinking about a cat.",      "a drawing of a cat"),
    ("dog",      "I am thinking about a dog.",      "a drawing of a dog"),
    ("elephant", "I am thinking about an elephant.","a drawing of an elephant"),
    ("flower",   "Imagine a flower in bloom.",      "a drawing of a flower"),
    ("sun",      "The sun is shining.",             "a drawing of a sun"),
    ("fish",     "Imagine a fish.",                 "a drawing of a fish"),
    ("airplane", "I am picturing an airplane.",     "a drawing of an airplane"),
    ("tree",     "I am picturing a tree.",          "a drawing of a tree"),
]


def label_panel(img: Image.Image, label: str, sub: str = "") -> Image.Image:
    W, H = img.size
    out = Image.new("RGB", (W, H + 60), color=(255, 255, 255))
    out.paste(img.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = font_big
    draw.text((12, H + 8), label, fill=(0, 0, 0), font=font_big)
    if sub:
        draw.text((12, H + 38), sub, fill=(110, 110, 110), font=font_small)
    return out


def hstack(panels: list[Image.Image], pad: int = 6) -> Image.Image:
    W = panels[0].size[0]
    H = panels[0].size[1]
    out = Image.new("RGB", (W * len(panels) + pad * (len(panels) - 1), H),
                    color=(255, 255, 255))
    for i, p in enumerate(panels):
        out.paste(p, (i * (W + pad), 0))
    return out


def vstack(rows: list[Image.Image]) -> Image.Image:
    W = max(r.size[0] for r in rows)
    H = sum(r.size[1] for r in rows)
    out = Image.new("RGB", (W, H), color=(255, 255, 255))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.size[1]
    return out


def write_mp4(strip: Image.Image, prompt: str, out_path: Path, fps: int = 24,
              seconds: float = 4.0):
    """Static 4-panel strip held for `seconds` (the panels themselves don't
    animate — variation across depth is the story).
    """
    import imageio.v3 as iio
    PAD = 60
    W, H = strip.size
    full = Image.new("RGB", (W, H + PAD), color=(255, 255, 255))
    full.paste(strip, (0, 0))
    draw = ImageDraw.Draw(full)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = ImageFont.load_default()
    draw.text((12, H + 20), prompt, fill=(0, 0, 0), font=font)
    arr = np.asarray(full, dtype=np.uint8)
    n = int(round(fps * seconds))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack([arr] * n, axis=0), fps=fps,
                codec="libx264", macro_block_size=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts-root", type=Path, required=True,
                   help="Directory containing LNN/final/av_ckpt.pt subdirs")
    p.add_argument("--layers", type=int, nargs="+", required=True,
                   help="Layers to render (e.g. 3 10 20 29)")
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--max-tokens", type=int, default=240)
    p.add_argument("--display-scale", type=float, default=2.0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prompts-jsonl", type=Path, default=None,
                   help="Optional override; default uses DEFAULT_PROMPTS")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.prompts_jsonl and args.prompts_jsonl.exists():
        prompts = []
        with open(args.prompts_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("group") == "hero":
                    prompts.append((row["slug"], row["prompt"], row["clip_text"]))
    else:
        prompts = DEFAULT_PROMPTS

    print(f"[xlayer] {len(prompts)} hero prompts × {len(args.layers)} layers", flush=True)

    from transformers import CLIPModel, CLIPProcessor
    print("[xlayer] loading CLIP ...", flush=True)
    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to("cuda").eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_name)

    # Storage: per layer, per slug → PIL image
    per_layer: dict[int, dict[str, Image.Image]] = {}
    per_layer_meta: dict[int, dict[str, dict]] = {}

    for layer in args.layers:
        ck = args.ckpts_root / f"L{layer:02d}" / "final"
        if not (ck / "av_ckpt.pt").exists():
            print(f"[xlayer] WARN: {ck/'av_ckpt.pt'} missing, skipping layer {layer}", flush=True)
            continue
        print(f"\n[xlayer] === L{layer:02d} from {ck} ===", flush=True)
        av = StrokeDecoder.from_ckpt(ck, model_id=args.model_id)
        av.model.eval()
        device = av.device()
        per_layer[layer] = {}
        per_layer_meta[layer] = {}
        for slug, prompt, clip_text in prompts:
            t0 = time.time()
            enc = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
            with torch.no_grad():
                out = av.model(**enc, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[layer][0, -1, :].detach()
            ids_list = av.generate_from_activation_batched(
                h, layer_ell=layer, n_samples=args.n_samples,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature, top_k=args.top_k,
            )
            candidates = []
            for ids in ids_list:
                strokes, _ = av.vocab.decode_tokens_with_stats(ids.tolist())
                if len(strokes) < 8:
                    continue
                img = stroke_render(strokes, display_scale=2.0).convert("RGB")
                candidates.append({"strokes": strokes, "img": img,
                                   "n_strokes": len(strokes)})
            if not candidates:
                print(f"  L{layer:02d} {slug}: all degenerate", flush=True)
                per_layer[layer][slug] = Image.new("RGB", (448, 448), color=(245, 245, 245))
                per_layer_meta[layer][slug] = {"degenerate": True}
                continue
            imgs = [c["img"] for c in candidates]
            with torch.no_grad():
                inputs = clip_proc(text=[clip_text], images=imgs, return_tensors="pt",
                                   padding=True).to("cuda")
                scores = clip_model(**inputs).logits_per_image.squeeze(-1).tolist()
            winner_idx = int(np.argmax(scores))
            winner = candidates[winner_idx]
            full_img = stroke_render(winner["strokes"], display_scale=args.display_scale).convert("RGB")
            per_layer[layer][slug] = full_img
            per_layer_meta[layer][slug] = {
                "clip_score": float(scores[winner_idx]),
                "n_strokes": winner["n_strokes"],
            }
            elapsed = time.time() - t0
            print(f"  L{layer:02d} {slug:10s}  CLIP={scores[winner_idx]:5.2f}  n_strokes={winner['n_strokes']:3d}  ({elapsed:.1f}s)", flush=True)
        del av
        gc.collect()
        torch.cuda.empty_cache()

    # 4-panel per prompt
    rows_for_grid: list[Image.Image] = []
    summary = {"layers": args.layers, "prompts": []}
    for slug, prompt, clip_text in prompts:
        panels = []
        for layer in args.layers:
            if layer not in per_layer or slug not in per_layer[layer]:
                continue
            img = per_layer[layer][slug]
            meta = per_layer_meta[layer][slug]
            sub = f"CLIP {meta.get('clip_score', float('nan')):.1f}" if "clip_score" in meta else ""
            panels.append(label_panel(img, f"L{layer:02d}", sub))
        if not panels:
            continue
        strip = hstack(panels, pad=8)
        strip_path = args.out_dir / f"{slug}_strip.png"
        strip.save(strip_path)
        write_mp4(strip, prompt, args.out_dir / f"{slug}.mp4", fps=24, seconds=4.0)
        # also a row in the big grid: label the leftmost panel with the prompt
        label_row = Image.new("RGB", (strip.size[0], 42), color=(255, 255, 255))
        d = ImageDraw.Draw(label_row)
        try:
            f_ = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except Exception:
            f_ = ImageFont.load_default()
        d.text((10, 10), prompt, fill=(0, 0, 0), font=f_)
        rows_for_grid.append(vstack([label_row, strip]))
        summary["prompts"].append({"slug": slug, "prompt": prompt,
                                    "clip_text": clip_text,
                                    "per_layer": {str(layer): per_layer_meta.get(layer, {}).get(slug)
                                                  for layer in args.layers}})
        print(f"[xlayer] → {strip_path.name} + {slug}.mp4", flush=True)

    if rows_for_grid:
        grid = vstack(rows_for_grid)
        grid.save(args.out_dir / "grid.png")
        print(f"[xlayer] grid → {args.out_dir / 'grid.png'}  ({grid.size[0]}x{grid.size[1]})", flush=True)

    # HTML index
    html = ["<!doctype html><html><body>",
            "<h1>v2.2 cross-layer trajectory</h1>",
            f"<p>Layers: {args.layers}. For each prompt: same prompt, drawing emitted at each layer, best-of-{args.n_samples} CLIP-ranked.</p>",
            "<div style='display:flex;flex-direction:column;gap:1.5em'>"]
    for slug, prompt, _ in prompts:
        html.append(
            f"<div style='border:1px solid #ddd;padding:.6em'>"
            f"<b>{slug}</b> — {prompt}<br>"
            f"<img src='{slug}_strip.png' style='max-width:100%'></div>"
        )
    html.append("</div></body></html>")
    (args.out_dir / "index.html").write_text("".join(html))

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[xlayer] DONE → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
