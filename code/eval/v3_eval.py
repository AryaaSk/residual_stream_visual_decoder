"""v3 evaluation — pure NLA cosine + discriminability falsifier.

Phase 2 (cosine on held-out): for each held-out prompt, sample N drawings from
the trained AV, render, feed the image alone back through frozen Qwen, read
activation at L*, compare to h_text(prompt). Report best, mean, std cosine.

Phase 3 (discriminability falsifier): build a pairwise cosine matrix:
    M[i, j] = cosine(h_text(prompt_i), h_image(drawing_from_prompt_j))
The diagonal should be the row-max. Mean diagonal vs mean off-diagonal is the
discriminability — does the AV produce drawings whose latent representation
encodes its specific concept, or just an average concept-like vector?

NO CLIP. NO image-feature heuristics. The model is the only oracle.

Output:
    findings/v3/eval/per_prompt.json        # cosine per prompt (Phase 2)
    findings/v3/eval/discriminability.json  # pairwise matrix + stats (Phase 3)
    findings/v3/eval/heldout_grid.png       # 5×4 grid of best-cosine drawings
    findings/v3/eval/discriminability.png   # heatmap of cosine matrix
"""

from __future__ import annotations

import argparse
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


# Held-out prompts (do NOT overlap with stage3 training prompts heavily)
HELDOUT = [
    ("cat",       "I am thinking about a cat."),
    ("dog",       "I am thinking about a dog."),
    ("elephant",  "I am thinking about an elephant."),
    ("fish",      "Imagine a fish."),
    ("bird",      "I am picturing a bird flying across the sky."),
    ("horse",     "I am thinking about a horse."),
    ("flower",    "Imagine a flower in bloom."),
    ("tree",      "I am picturing a tree."),
    ("sun",       "The sun is shining."),
    ("cloud",     "I am picturing a cloud in the sky."),
    ("house",     "I am picturing a small house with a red roof."),
    ("car",       "I am thinking about a car."),
    ("airplane",  "I am picturing an airplane."),
    ("apple",     "I am thinking about an apple."),
    ("pizza",     "I am thinking about a pizza."),
    ("mountain",  "I am picturing a mountain."),
    ("eiffel",    "Paris, the city of lights, is famous for the Eiffel"),
    ("smile",     "I am picturing a smiling face."),
    ("triangle",  "Imagine a triangle inscribed in a circle."),
    ("storm",     "When the storm hit the village."),
]


def grid_image(images: list[Image.Image], labels: list[str], cols: int = 4) -> Image.Image:
    if not images:
        raise ValueError("no images")
    cell = images[0].size[0]
    rows = (len(images) + cols - 1) // cols
    PAD = 50
    cell_h = images[0].size[1] + PAD
    grid = Image.new("RGB", (cols * cell, rows * cell_h), color=(255, 255, 255))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = ImageFont.load_default()
    for i, (img, lab) in enumerate(zip(images, labels)):
        x = (i % cols) * cell
        y = (i // cols) * cell_h
        grid.paste(img, (x, y))
        d = ImageDraw.Draw(grid)
        d.text((x + 8, y + images[0].size[1] + 8), lab, fill=(0, 0, 0), font=font)
    return grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--n-samples", type=int, default=8,
                   help="Drawings sampled per prompt for best-cosine pick")
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--max-tokens", type=int, default=240)
    p.add_argument("--display-scale", type=float, default=2.0)
    p.add_argument("--out-dir", type=Path, default=Path("findings/v3/eval"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v3eval] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    device = av.device()

    print("[v3eval] loading frozen evaluator Qwen (ImageTextToText) ...", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    qwen_eval = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    qwen_tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    qwen_proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    @torch.no_grad()
    def h_text(text: str) -> torch.Tensor:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        wrapped = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = qwen_proc(text=[wrapped], images=None, return_tensors="pt").to(device)
        out = qwen_eval(**inputs, output_hidden_states=True, use_cache=False)
        return out.hidden_states[args.layer][0, -1, :].detach().to(torch.float32)

    @torch.no_grad()
    def h_image_only(image: Image.Image) -> torch.Tensor | None:
        messages = [{"role": "user", "content": [{"type": "image"}]}]
        wrapped = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        try:
            inputs = qwen_proc(text=[wrapped], images=[image], return_tensors="pt").to(device)
            out = qwen_eval(**inputs, output_hidden_states=True, use_cache=False)
            return out.hidden_states[args.layer][0, -1, :].detach().to(torch.float32)
        except Exception as e:
            print(f"[v3eval]   image-only forward failed: {e}", flush=True)
            return None

    # Phase 2: best-of-N per prompt
    print(f"\n[v3eval] === Phase 2: cosine per held-out prompt ===", flush=True)
    per_prompt = []
    best_images: list[Image.Image] = []
    best_labels: list[str] = []
    best_h_image: dict[str, torch.Tensor] = {}  # for Phase 3
    h_text_cache: dict[str, torch.Tensor] = {}
    for slug, prompt in HELDOUT:
        t0 = time.time()
        h_t = h_text(prompt)
        h_text_cache[slug] = h_t.cpu()
        ids_list = av.generate_from_activation_batched(
            h_t, layer_ell=args.layer,
            n_samples=args.n_samples,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
        candidates = []
        for sample_idx, ids in enumerate(ids_list):
            strokes, _ = av.vocab.decode_tokens_with_stats(ids.tolist())
            if len(strokes) < 2:
                continue
            img = stroke_render(strokes, display_scale=args.display_scale).convert("RGB")
            h_i = h_image_only(img)
            if h_i is None:
                continue
            cos = float(F.cosine_similarity(h_t, h_i, dim=0).item())
            candidates.append({"sample": sample_idx, "n_strokes": len(strokes),
                               "cosine": cos, "img": img, "strokes": strokes, "h_image": h_i.cpu()})
        if not candidates:
            print(f"[v3eval] {slug:10s} all degenerate", flush=True)
            per_prompt.append({"slug": slug, "prompt": prompt, "best_cosine": None,
                               "mean_cosine": None, "all_degenerate": True})
            continue
        cosines = [c["cosine"] for c in candidates]
        candidates.sort(key=lambda c: c["cosine"], reverse=True)
        best = candidates[0]
        # Save best drawing
        best_path = args.out_dir / f"{slug}.png"
        full = stroke_render(best["strokes"], display_scale=4.0).convert("RGB")
        full.save(best_path)
        best_images.append(best["img"].resize((448, 448)))
        best_labels.append(f"{slug}  cos={best['cosine']:+.3f}")
        best_h_image[slug] = best["h_image"]
        elapsed = time.time() - t0
        per_prompt.append({
            "slug": slug, "prompt": prompt,
            "best_cosine": round(best["cosine"], 4),
            "mean_cosine": round(float(np.mean(cosines)), 4),
            "std_cosine": round(float(np.std(cosines)), 4),
            "n_candidates": len(candidates),
            "wallclock_s": round(elapsed, 1),
        })
        print(f"[v3eval] {slug:10s}  best={best['cosine']:+.3f}  mean={float(np.mean(cosines)):+.3f}  std={float(np.std(cosines)):.3f}  n_strokes={best['n_strokes']:3d}  ({elapsed:.1f}s)", flush=True)

    # Aggregate Phase 2
    valid = [r for r in per_prompt if r.get("best_cosine") is not None]
    summary_p2 = {
        "n_prompts": len(per_prompt),
        "n_valid": len(valid),
        "best_cosine_mean": float(np.mean([r["best_cosine"] for r in valid])) if valid else None,
        "best_cosine_std":  float(np.std([r["best_cosine"] for r in valid])) if valid else None,
        "mean_cosine_mean": float(np.mean([r["mean_cosine"] for r in valid])) if valid else None,
        "per_prompt": per_prompt,
    }
    (args.out_dir / "per_prompt.json").write_text(json.dumps(summary_p2, indent=2))

    print(f"\n[v3eval] Phase 2 aggregate:", flush=True)
    print(f"  mean best cosine across {len(valid)} prompts: {summary_p2['best_cosine_mean']:+.4f} ± {summary_p2['best_cosine_std']:.4f}", flush=True)

    # Save grid
    if best_images:
        grid = grid_image(best_images, best_labels, cols=5)
        grid.save(args.out_dir / "heldout_grid.png")
        print(f"[v3eval] saved {args.out_dir / 'heldout_grid.png'}", flush=True)

    # Phase 3: pairwise discriminability
    print(f"\n[v3eval] === Phase 3: discriminability (pairwise cosine matrix) ===", flush=True)
    valid_slugs = [r["slug"] for r in per_prompt if r.get("best_cosine") is not None]
    n = len(valid_slugs)
    if n < 2:
        print("[v3eval] not enough valid drawings for discriminability", flush=True)
        return
    M = np.zeros((n, n), dtype=np.float32)
    for i, slug_i in enumerate(valid_slugs):
        h_t_i = h_text_cache[slug_i]
        for j, slug_j in enumerate(valid_slugs):
            h_i_j = best_h_image[slug_j]
            M[i, j] = F.cosine_similarity(h_t_i, h_i_j, dim=0).item()
    # diag = self-recon; off-diag = average other
    diag = np.diag(M)
    off = (M.sum(axis=1) - diag) / (n - 1)
    discriminability = diag - off
    summary_p3 = {
        "n_prompts": n,
        "slugs": valid_slugs,
        "matrix": M.tolist(),
        "diag_mean": float(diag.mean()),
        "off_diag_mean": float(off.mean()),
        "discriminability_mean": float(discriminability.mean()),
        "discriminability_std": float(discriminability.std()),
        "per_prompt": [{"slug": s, "self": float(diag[i]), "other_avg": float(off[i]),
                        "delta": float(discriminability[i])}
                       for i, s in enumerate(valid_slugs)],
        # Top-1 retrieval: how often is the correct drawing's h_image the
        # closest match to a prompt's h_text?
        "top1_retrieval": float(np.mean([np.argmax(M[i]) == i for i in range(n)])),
    }
    (args.out_dir / "discriminability.json").write_text(json.dumps(summary_p3, indent=2))
    print(f"[v3eval] diag mean         = {summary_p3['diag_mean']:+.4f}", flush=True)
    print(f"[v3eval] off-diag mean     = {summary_p3['off_diag_mean']:+.4f}", flush=True)
    print(f"[v3eval] discriminability  = {summary_p3['discriminability_mean']:+.4f} ± {summary_p3['discriminability_std']:.4f}", flush=True)
    print(f"[v3eval] top-1 retrieval   = {summary_p3['top1_retrieval']*100:.1f}%  (chance {100/n:.1f}%)", flush=True)

    # Heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([s for s in valid_slugs], rotation=60, fontsize=8, ha="right")
        ax.set_yticklabels([s for s in valid_slugs], fontsize=8)
        ax.set_xlabel("drawing source prompt")
        ax.set_ylabel("h_text prompt")
        ax.set_title(f"cosine(h_text_i, h_image_j) — diag-off = {summary_p3['discriminability_mean']:+.3f}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(args.out_dir / "discriminability.png", dpi=120)
        print(f"[v3eval] saved {args.out_dir / 'discriminability.png'}", flush=True)
    except Exception as e:
        print(f"[v3eval] WARN: plot failed: {e}", flush=True)


if __name__ == "__main__":
    main()
