"""v3 cross-depth video — the viral centerpiece.

For each hero prompt, render best-of-N drawings at L3 / L10 / L20 / L29.
RANK BY GEN-LIKELIHOOD (under frozen Qwen) instead of CLIP. Generate side-by-side
strips + a big grid + an HTML index.

This is the FOUNDATIONAL interpretability visual: the v3 metric (which Qwen
itself uses to judge whether a drawing represents a concept) drives the choice,
across multiple depths showing how concept-decodability changes with layer.

Inputs:
    --ckpts-root <dir>          # contains LNN/final/av_ckpt.pt subdirs
    --layers <L1> <L2> ...
Outputs:
    artefacts/v3/cross_layer/{slug}_strip.png    (one row per prompt)
    artefacts/v3/cross_layer/grid.png            (all prompts × all layers)
    artefacts/v3/cross_layer/index.html
    artefacts/v3/cross_layer/per_layer_scores.json
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
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder
from render import render as stroke_render


# Hero prompts: known to have strong gen-likelihood signal from v2.0 baseline eval
HERO = [
    ("cat",       "I am thinking about a cat.",       "cat"),
    ("dog",       "I am thinking about a dog.",       "dog"),
    ("elephant",  "I am thinking about an elephant.", "elephant"),
    ("fish",      "Imagine a fish.",                  "fish"),
    ("horse",     "I am thinking about a horse.",     "horse"),
    ("flower",    "Imagine a flower in bloom.",       "flower"),
    ("sun",       "The sun is shining.",              "sun"),
    ("tree",      "I am picturing a tree.",           "tree"),
    ("airplane",  "I am picturing an airplane.",      "airplane"),
    ("car",       "I am thinking about a car.",       "car"),
    ("pizza",     "I am thinking about a pizza.",     "pizza"),
    ("mountain",  "I am picturing a mountain.",       "mountain"),
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
    draw.text((12, H + 6), label, fill=(0, 0, 0), font=font_big)
    if sub:
        draw.text((12, H + 35), sub, fill=(110, 110, 110), font=font_small)
    return out


def hstack(panels, pad=8):
    W = panels[0].size[0]
    H = panels[0].size[1]
    out = Image.new("RGB", (W * len(panels) + pad * (len(panels) - 1), H), color=(255, 255, 255))
    for i, p in enumerate(panels):
        out.paste(p, (i * (W + pad), 0))
    return out


def vstack(rows):
    W = max(r.size[0] for r in rows)
    H = sum(r.size[1] for r in rows)
    out = Image.new("RGB", (W, H), color=(255, 255, 255))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.size[1]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts-root", type=Path, required=True,
                   help="Dir containing LNN/final/av_ckpt.pt subdirs")
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--max-tokens", type=int, default=240)
    p.add_argument("--display-scale", type=float, default=1.0,
                   help="1.0=224x224 (Qwen native, fast). Use 2.0+ only for the saved PNGs.")
    p.add_argument("--save-display-scale", type=float, default=4.0,
                   help="Scale at which saved PNGs render (only affects visual sharpness, not Qwen scoring)")
    p.add_argument("--question", default="What is this a drawing of?")
    p.add_argument("--continuation-prefix", default="A drawing of a")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[xv3] loading frozen Qwen evaluator (ImageTextToText) ...", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    qwen = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    qproc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    for p_ in qwen.parameters():
        p_.requires_grad = False
    tok = qproc.tokenizer

    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.question}]}]
    PREFIX_TEXT = qproc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + args.continuation_prefix
    concept_ids = {c: torch.tensor([tok.encode(" " + c, add_special_tokens=False)],
                                    dtype=torch.long, device=qwen.device)
                   for _, _, c in HERO}

    @torch.no_grad()
    def gen_logp(image, concept):
        cand = concept_ids.get(concept)
        if cand is None or cand.shape[1] == 0:
            return -1e9
        try:
            inp = qproc(text=[PREFIX_TEXT], images=[image], return_tensors="pt").to(qwen.device)
        except Exception:
            return -1e9
        prefix_len = inp["input_ids"].shape[1]
        full = {k: v for k, v in inp.items()}
        full["input_ids"] = torch.cat([inp["input_ids"], cand], dim=1)
        if "attention_mask" in full:
            full["attention_mask"] = torch.cat([inp["attention_mask"], torch.ones_like(cand)], dim=1)
        if "mm_token_type_ids" in full:
            full["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(cand)], dim=1)
        try:
            out = qwen(**full, use_cache=False)
        except Exception:
            return -1e9
        T = full["input_ids"].shape[1]
        logits = out.logits[0, prefix_len - 1: T - 1, :]
        target = full["input_ids"][0, prefix_len:]
        logp = F.log_softmax(logits.float(), dim=-1)
        chosen = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return float(chosen.sum().item())

    # Store per-layer per-prompt best image (PIL) + its score
    per_layer: dict[int, dict[str, Image.Image]] = {}
    per_layer_meta: dict[int, dict[str, dict]] = {}

    for layer in args.layers:
        ck = args.ckpts_root / f"L{layer:02d}" / "final"
        # Fallback to L{N}/final (without zero-pad) if zero-padded missing
        if not (ck / "av_ckpt.pt").exists():
            ck2 = args.ckpts_root / f"L{layer}" / "final"
            if (ck2 / "av_ckpt.pt").exists():
                ck = ck2
        if not (ck / "av_ckpt.pt").exists():
            print(f"[xv3] WARN: L{layer} ckpt missing at {ck}, skipping", flush=True)
            continue
        print(f"\n[xv3] === L{layer} from {ck} ===", flush=True)
        av = StrokeDecoder.from_ckpt(ck, model_id=args.model_id)
        av.model.eval()
        device = av.device()
        per_layer[layer] = {}
        per_layer_meta[layer] = {}
        for slug, prompt, concept in HERO:
            t0 = time.time()
            enc = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
            with torch.no_grad():
                out_h = av.model(**enc, output_hidden_states=True, use_cache=False)
            h = out_h.hidden_states[layer][0, -1, :].detach()
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
                img_low = stroke_render(strokes, display_scale=args.display_scale).convert("RGB")
                lp = gen_logp(img_low, concept)
                candidates.append({"strokes": strokes, "img_low": img_low,
                                   "logp": lp, "n_strokes": len(strokes)})
            if not candidates:
                per_layer[layer][slug] = Image.new("RGB", (448, 448), color=(240, 240, 240))
                per_layer_meta[layer][slug] = {"degenerate": True}
                print(f"  L{layer:02d} {slug:10s} all degenerate", flush=True)
                continue
            candidates.sort(key=lambda c: c["logp"], reverse=True)
            best = candidates[0]
            # Re-render at save-display-scale for the saved PNG
            full = stroke_render(best["strokes"], display_scale=args.save_display_scale).convert("RGB")
            per_layer[layer][slug] = full
            per_layer_meta[layer][slug] = {
                "logp": best["logp"], "n_strokes": best["n_strokes"],
                "n_candidates": len(candidates),
            }
            print(f"  L{layer:02d} {slug:10s}  logp={best['logp']:+.2f}  n_strokes={best['n_strokes']}  ({time.time()-t0:.1f}s)", flush=True)
        del av
        gc.collect()
        torch.cuda.empty_cache()

    # Build per-prompt strips
    rows_for_grid = []
    summary = {"layers": args.layers, "prompts": []}
    for slug, prompt, concept in HERO:
        panels = []
        for layer in args.layers:
            if layer not in per_layer or slug not in per_layer[layer]:
                continue
            img = per_layer[layer][slug]
            meta = per_layer_meta[layer][slug]
            sub = f"logp {meta.get('logp', float('nan')):+.1f}" if "logp" in meta else "(degenerate)"
            panels.append(label_panel(img, f"L{layer:02d}", sub))
        if not panels:
            continue
        strip = hstack(panels, pad=10)
        strip_path = args.out_dir / f"{slug}_strip.png"
        strip.save(strip_path)
        # Prompt label row
        prompt_row = Image.new("RGB", (strip.size[0], 44), color=(255, 255, 255))
        d = ImageDraw.Draw(prompt_row)
        try:
            f_ = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except Exception:
            f_ = ImageFont.load_default()
        d.text((12, 12), prompt, fill=(20, 20, 20), font=f_)
        rows_for_grid.append(vstack([prompt_row, strip]))
        summary["prompts"].append({
            "slug": slug, "prompt": prompt, "concept": concept,
            "per_layer": {str(layer): per_layer_meta.get(layer, {}).get(slug) for layer in args.layers},
        })
        print(f"[xv3] strip → {strip_path.name}", flush=True)

    if rows_for_grid:
        grid = vstack(rows_for_grid)
        grid.save(args.out_dir / "grid.png")
        print(f"[xv3] grid → {args.out_dir / 'grid.png'} ({grid.size[0]}x{grid.size[1]})", flush=True)

    # HTML index
    html = ["<!doctype html><html><body>",
            "<h1>v3 cross-depth trajectory</h1>",
            f"<p>Per-layer drawings ranked by gen-likelihood (log P(concept | image, prompt) under frozen Qwen 3.5-4B). Layers: {args.layers}. {args.n_samples} samples per layer per prompt.</p>",
            "<div style='display:flex;flex-direction:column;gap:1.5em'>"]
    for slug, prompt, concept in HERO:
        html.append(
            f"<div style='border:1px solid #ddd;padding:.6em'>"
            f"<b>{slug}</b> — {prompt}<br>"
            f"<img src='{slug}_strip.png' style='max-width:100%'></div>"
        )
    html.append("</div></body></html>")
    (args.out_dir / "index.html").write_text("".join(html))
    (args.out_dir / "per_layer_scores.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[xv3] DONE → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
