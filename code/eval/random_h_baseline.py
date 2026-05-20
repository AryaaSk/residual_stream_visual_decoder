"""Random-h baseline — the gating experiment for v2.2.

If the AV is genuinely decoding the activation, feeding random Gaussian h
vectors (matched in mean and covariance to real activations) should yield
unrecognisable / abstract drawings — and CLIP scores against the 20 hero
concepts should be uniformly low.

If the AV ignores h and produces memorised templates, the random-h drawings
will look like recognisable concepts (some concept, doesn't matter which).
That outcome would gut the v2.2 interpretability claim — better to know now.

Output:
  findings/v2_2/random_h_baseline/
    grid_random_matched.png    (4x4 grid)
    grid_random_iso.png        (4x4 grid)
    grid_real_control.png      (4x4 grid)
    individual/{type}/{i}.png  (individual renders)
    summary.json               (per-sample CLIP scores vs 20 hero concepts)
    verdict.txt                (one-paragraph interpretation)

Three conditions, 16 samples each:
  - random_matched: h ~ N(mean_real, diag(std_real^2)) — best-case "looks like real h"
  - random_iso:     h ~ N(0, sigma I) with sigma chosen so ||h|| matches real ||h||
  - real_control:   h extracted from 16 hero prompts (positive control)

For each condition, CLIP-rank the resulting drawing against 20 hero concept
texts. If random-h gets uniformly low best-CLIP, the AV depends on h.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder
from render import render as stroke_render


# Reference prompts used to (a) seed real-h moment estimation and
# (b) serve as the real-control condition. 16 prompts so the control grid
# matches the random grids in size.
REF_PROMPTS = [
    ("cat",      "I am thinking about a cat."),
    ("dog",      "I am thinking about a dog."),
    ("fish",     "Imagine a fish."),
    ("bird",     "I am picturing a bird flying across the sky."),
    ("horse",    "I am thinking about a horse."),
    ("elephant", "I am thinking about an elephant."),
    ("flower",   "Imagine a flower in bloom."),
    ("tree",     "I am picturing a tree."),
    ("cactus",   "I am picturing a cactus in the desert."),
    ("mountain", "I am picturing a mountain."),
    ("sun",      "The sun is shining."),
    ("cloud",    "I am picturing a cloud in the sky."),
    ("star",     "I am picturing a star in the night sky."),
    ("house",    "I am picturing a small house with a red roof."),
    ("car",      "I am thinking about a car."),
    ("airplane", "I am picturing an airplane."),
]

# CLIP texts: 20 concepts the AV was trained on. We score every drawing
# against every concept text; the maximum is the "best guess" concept.
CLIP_CONCEPTS = [
    "cat", "dog", "fish", "bird", "horse", "elephant",
    "flower", "tree", "cactus", "mountain",
    "sun", "cloud", "star", "house",
    "car", "airplane", "apple", "pizza", "clock", "umbrella",
]
CLIP_TEXTS = [f"a drawing of a {c}" for c in CLIP_CONCEPTS]


def grid_image(images: list[Image.Image], cols: int = 4) -> Image.Image:
    if not images:
        raise ValueError("no images")
    cell = images[0].size[0]
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell, rows * cell), color=(255, 255, 255))
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * cell, (i // cols) * cell))
    return grid


def extract_h_for_prompts(av, prompts: list[str], layer: int) -> torch.Tensor:
    device = av.device()
    hs = []
    for p in prompts:
        enc = av.tokenizer(p, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer][0, -1, :].detach().cpu()
        hs.append(h)
    return torch.stack(hs, dim=0)  # (N, d)


def best_candidate_clip(av, h: torch.Tensor, *, layer: int, n_samples: int,
                        max_tokens: int, temperature: float, top_k: int,
                        clip_model, clip_proc) -> tuple[Image.Image, list, dict]:
    """Generate N drawings from one h; CLIP-score each against every CLIP_TEXT;
    return the highest-overall image and its per-concept scores."""
    gen_ids_list = av.generate_from_activation_batched(
        h.to(av.device()), layer_ell=layer, n_samples=n_samples,
        max_new_tokens=max_tokens, temperature=temperature, top_k=top_k,
    )
    candidates = []
    for s, ids in enumerate(gen_ids_list):
        strokes, malformed = av.vocab.decode_tokens_with_stats(ids.tolist())
        if len(strokes) < 8:
            continue
        img = stroke_render(strokes, display_scale=2.0).convert("RGB")
        candidates.append({"strokes": strokes, "img": img,
                           "n_strokes": len(strokes), "malformed": malformed})
    if not candidates:
        return None, None, {"all_degenerate": True}

    imgs = [c["img"] for c in candidates]
    with torch.no_grad():
        inputs = clip_proc(text=CLIP_TEXTS, images=imgs, return_tensors="pt",
                           padding=True).to("cuda")
        out_clip = clip_model(**inputs)
        # logits_per_image: (N_images, N_texts)
        logits = out_clip.logits_per_image.detach().cpu()

    # For each candidate: best concept text + score
    best_per_cand = []
    for i, c in enumerate(candidates):
        scores = logits[i].tolist()
        best_idx = int(np.argmax(scores))
        best_per_cand.append({
            "best_concept": CLIP_CONCEPTS[best_idx],
            "best_score": float(scores[best_idx]),
            "n_strokes": c["n_strokes"],
            "scores": [round(s, 2) for s in scores],
        })

    # Overall winner across candidates: pick max best_score
    winner_idx = int(np.argmax([bp["best_score"] for bp in best_per_cand]))
    winner = candidates[winner_idx]
    winner_info = best_per_cand[winner_idx]
    return winner["img"], winner["strokes"], {
        "winner": winner_info,
        "all_candidates": best_per_cand,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--n-samples", type=int, default=16,
                   help="best-of-N CLIP per single h vector")
    p.add_argument("--n-random", type=int, default=16,
                   help="number of distinct random-h vectors per condition")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=240)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--display-scale", type=float, default=2.0,
                   help="display upscale for individual saved PNGs")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "individual" / "random_matched").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "individual" / "random_iso").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "individual" / "real_control").mkdir(parents=True, exist_ok=True)

    print(f"[random-h] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    device = av.device()

    print(f"[random-h] extracting real-h moments at L{args.layer} from {len(REF_PROMPTS)} prompts ...", flush=True)
    ref_h = extract_h_for_prompts(av, [p for _, p in REF_PROMPTS], layer=args.layer)
    h_mean = ref_h.mean(dim=0)
    h_std = ref_h.std(dim=0)
    h_norms = ref_h.norm(dim=1)
    mean_norm = float(h_norms.mean().item())
    d = ref_h.shape[1]
    print(f"[random-h] real h stats: shape={tuple(ref_h.shape)} ||h||~{mean_norm:.2f}  per-dim std mean={float(h_std.mean()):.4f}", flush=True)

    # Build random h vectors
    rng = np.random.default_rng(args.seed)
    # Matched: per-coord N(mean, std)
    random_matched = h_mean.unsqueeze(0) + torch.from_numpy(
        rng.standard_normal((args.n_random, d)).astype(np.float32)
    ) * h_std.unsqueeze(0)
    # Iso: N(0, I), then rescale so ||h|| ≈ mean_norm
    iso = torch.from_numpy(rng.standard_normal((args.n_random, d)).astype(np.float32))
    iso = iso / iso.norm(dim=1, keepdim=True) * mean_norm
    random_iso = iso

    print("[random-h] loading CLIP ranker ...", flush=True)
    from transformers import CLIPModel, CLIPProcessor
    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to("cuda").eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_name)

    summary = {
        "av_ckpt": str(args.av_ckpt),
        "layer": args.layer,
        "n_samples_per_h": args.n_samples,
        "n_random": args.n_random,
        "real_h_norm_mean": mean_norm,
        "real_h_norm_min": float(h_norms.min()),
        "real_h_norm_max": float(h_norms.max()),
        "real_h_dim": d,
        "conditions": {},
    }

    def run_condition(name: str, hs: torch.Tensor, labels: list[str]):
        print(f"\n[random-h] === condition: {name} ===", flush=True)
        out_imgs = []
        per_sample = []
        for i, (h, lab) in enumerate(zip(hs, labels)):
            t0 = time.time()
            img, strokes, meta = best_candidate_clip(
                av, h, layer=args.layer, n_samples=args.n_samples,
                max_tokens=args.max_tokens, temperature=args.temperature,
                top_k=args.top_k, clip_model=clip_model, clip_proc=clip_proc,
            )
            if img is None:
                print(f"[random-h] {name} {i:02d} {lab}: ALL DEGENERATE", flush=True)
                per_sample.append({"i": i, "label": lab, "all_degenerate": True})
                # render a blank placeholder
                ph = Image.new("RGB", (448, 448), color=(245, 245, 245))
                out_imgs.append(ph)
                continue
            ind_path = args.out_dir / "individual" / name / f"{i:02d}_{lab}.png"
            full_img = stroke_render(strokes, display_scale=args.display_scale).convert("RGB")
            full_img.save(ind_path)
            out_imgs.append(img.resize((448, 448)))
            elapsed = time.time() - t0
            w = meta["winner"]
            print(f"[random-h] {name} {i:02d} {lab:12s}  best={w['best_concept']:9s} score={w['best_score']:5.2f} n_strokes={w['n_strokes']:3d}  ({elapsed:.1f}s)", flush=True)
            per_sample.append({"i": i, "label": lab, **meta})
        grid_image(out_imgs, cols=4).save(args.out_dir / f"grid_{name}.png")
        summary["conditions"][name] = {"per_sample": per_sample}

    # condition 1: random matched
    matched_labels = [f"r{i:02d}" for i in range(args.n_random)]
    run_condition("random_matched", random_matched, matched_labels)
    # condition 2: random iso
    run_condition("random_iso", random_iso, matched_labels)
    # condition 3: real control
    real_labels = [slug for slug, _ in REF_PROMPTS]
    run_condition("real_control", ref_h, real_labels)

    # Verdict statistics
    def best_scores(name: str) -> list[float]:
        ss = []
        for s in summary["conditions"][name]["per_sample"]:
            if "winner" in s:
                ss.append(s["winner"]["best_score"])
        return ss

    matched_scores = best_scores("random_matched")
    iso_scores = best_scores("random_iso")
    real_scores = best_scores("real_control")
    summary["stats"] = {
        "random_matched_mean": float(np.mean(matched_scores)) if matched_scores else None,
        "random_matched_max":  float(np.max(matched_scores)) if matched_scores else None,
        "random_iso_mean":     float(np.mean(iso_scores)) if iso_scores else None,
        "random_iso_max":      float(np.max(iso_scores)) if iso_scores else None,
        "real_control_mean":   float(np.mean(real_scores)) if real_scores else None,
        "real_control_max":    float(np.max(real_scores)) if real_scores else None,
    }

    # Mode-collapse check: how many distinct best_concepts in each condition?
    def concept_histogram(name: str) -> dict:
        hist = {}
        for s in summary["conditions"][name]["per_sample"]:
            if "winner" in s:
                c = s["winner"]["best_concept"]
                hist[c] = hist.get(c, 0) + 1
        return hist

    summary["histograms"] = {
        "random_matched": concept_histogram("random_matched"),
        "random_iso":     concept_histogram("random_iso"),
        "real_control":   concept_histogram("real_control"),
    }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Verdict
    if matched_scores and real_scores:
        gap = summary["stats"]["real_control_mean"] - summary["stats"]["random_matched_mean"]
        verdict_lines = [
            f"REAL h mean best-CLIP: {summary['stats']['real_control_mean']:.2f}",
            f"RANDOM-MATCHED h mean best-CLIP: {summary['stats']['random_matched_mean']:.2f}",
            f"RANDOM-ISO h mean best-CLIP: {summary['stats']['random_iso_mean']:.2f}",
            f"Gap (real - matched): {gap:+.2f}",
            "",
            f"Real-control concept histogram: {summary['histograms']['real_control']}",
            f"Random-matched concept histogram: {summary['histograms']['random_matched']}",
            f"Random-iso concept histogram: {summary['histograms']['random_iso']}",
            "",
        ]
        if gap > 2.0:
            verdict_lines.append("VERDICT: AV is h-sensitive (gap > 2.0). Proceed with v2.2.")
        elif gap > 0.5:
            verdict_lines.append("VERDICT: Weak h-sensitivity (gap 0.5 - 2.0). Proceed with caveat.")
        else:
            verdict_lines.append("VERDICT: AV largely ignores h (gap <= 0.5). RE-PLAN v2.2 framing.")
        (args.out_dir / "verdict.txt").write_text("\n".join(verdict_lines))
        print("\n=== VERDICT ===\n" + "\n".join(verdict_lines))

    print(f"\n[random-h] DONE -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
