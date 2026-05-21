"""v3 generation-likelihood evaluation — Phase 2 + Phase 3.

For an AV checkpoint (typically a v3 trained one, or v2.0 SFT as baseline):

Phase 2 (held-out eval):
  - For each of 20 held-out concept prompts:
    - Sample N drawings from the AV
    - For each drawing, compute:
      a) log P(correct_concept | image, "A drawing of a ") under frozen Qwen
      b) argmax over all 44 SFT-known concept words ("top-1 retrieval")
      c) margin = log P(correct) - log P(best_wrong)
  - Report mean / best-of-N per prompt + overall stats.

Phase 3 (discriminability):
  - For each held-out prompt's BEST drawing, score it against all 20 prompt's
    concepts. Pairwise log-prob matrix.
  - diag = correct match; off = wrong match. Top-1 retrieval over 20 candidates
    (chance = 5%).

NO CLIP. NO cosine. The frozen Qwen's caption log-prob is the ONLY oracle.

Output:
    findings/v3/eval_<run-tag>/
        per_prompt.json
        retrieval_<top1>.json
        discriminability.json
        heldout_grid.png
        retrieval_table.png
        discriminability.png
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


# 20 held-out prompts (overlap with v3 training prompts is fine — same prompt
# templates as data/expanded_captions.jsonl). Each has a known target concept.
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
    ("star",      "I am picturing a star in the night sky."),
    ("umbrella",  "I am picturing an umbrella."),
    ("clock",     "I am picturing a clock on the wall."),
    ("banana",    "I am picturing a banana."),
]

# 44 concepts from the SFT-trained set — used for "top-1 retrieval" eval
ALL_CONCEPTS = [
    "airplane", "apple", "banana", "bed", "bicycle", "bird", "book", "bread",
    "bridge", "cactus", "car", "carrot", "cat", "chair", "clock", "cloud",
    "cookie", "dog", "donut", "door", "elephant", "fish", "flower", "horse",
    "house", "key", "leaf", "moon", "mountain", "mushroom", "pencil", "pizza",
    "rainbow", "scissors", "snake", "spider", "star", "sun", "table", "tent",
    "train", "tree", "truck", "umbrella",
]


def grid_image(images: list[Image.Image], labels: list[str], cols: int = 5) -> Image.Image:
    cell = images[0].size[0]
    PAD = 50
    cell_h = images[0].size[1] + PAD
    rows = (len(images) + cols - 1) // cols
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
        d.text((x + 6, y + images[0].size[1] + 6), lab, fill=(0, 0, 0), font=font)
    return grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--max-tokens", type=int, default=240)
    p.add_argument("--display-scale", type=float, default=1.0,
                   help="Render scale for images fed to Qwen. 1.0=224x224 (Qwen native; fast). Higher scales are slower without quality benefit at this resolution.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--question", default="What is this a drawing of?")
    p.add_argument("--continuation-prefix", default="A drawing of a")
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
    from transformers import AutoModelForImageTextToText, AutoProcessor
    qwen_eval = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    qwen_proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    for p_ in qwen_eval.parameters():
        p_.requires_grad = False

    # Pre-tokenize all candidate concepts
    candidate_token_ids: dict[str, list[int]] = {
        c: qwen_proc.tokenizer.encode(" " + c, add_special_tokens=False)
        for c in ALL_CONCEPTS
    }

    @torch.no_grad()
    def h_text(text: str) -> torch.Tensor:
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        wrap = qwen_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        inp = qwen_proc(text=[wrap], images=None, return_tensors="pt").to(device)
        out = qwen_eval(**inp, output_hidden_states=True, use_cache=False)
        return out.hidden_states[args.layer][0, -1, :].detach().to(torch.float32)

    # Build the prefix text once
    PROMPT_MSGS = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": args.question}],
    }]
    PREFIX_TEXT = qwen_proc.apply_chat_template(PROMPT_MSGS, tokenize=False, add_generation_prompt=True) + args.continuation_prefix
    eval_tok = qwen_proc.tokenizer

    # Pre-build concept target tensors (CPU; copy to device as needed)
    concept_target_t: dict[str, torch.Tensor] = {
        c: torch.tensor([eval_tok.encode(" " + c, add_special_tokens=False)],
                        dtype=torch.long, device=device)
        for c in ALL_CONCEPTS
    }

    @torch.no_grad()
    def _process_prefix(image: Image.Image):
        try:
            return qwen_proc(text=[PREFIX_TEXT], images=[image], return_tensors="pt").to(device)
        except Exception:
            return None

    @torch.no_grad()
    def _score_prepared(prefix_inp, prefix_len: int, concept: str) -> float | None:
        cand_t = concept_target_t.get(concept)
        if cand_t is None or cand_t.shape[1] == 0:
            return None
        full = {k: v for k, v in prefix_inp.items()}
        full["input_ids"] = torch.cat([prefix_inp["input_ids"], cand_t], dim=1)
        if "attention_mask" in full:
            full["attention_mask"] = torch.cat(
                [prefix_inp["attention_mask"], torch.ones_like(cand_t)], dim=1
            )
        if "mm_token_type_ids" in full:
            full["mm_token_type_ids"] = torch.cat(
                [prefix_inp["mm_token_type_ids"], torch.zeros_like(cand_t)], dim=1
            )
        try:
            out = qwen_eval(**full, use_cache=False)
        except Exception:
            return None
        T = full["input_ids"].shape[1]
        logits = out.logits[0, prefix_len - 1: T - 1, :]
        target = full["input_ids"][0, prefix_len:]
        logp = F.log_softmax(logits.float(), dim=-1)
        chosen = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return float(chosen.sum().item())

    @torch.no_grad()
    def all_concept_logprobs(image: Image.Image) -> dict[str, float]:
        """For one image, compute log P(concept | image, prefix) for ALL 44 concepts.
        Uses ONE processor call (prefix processing) + 44 model forwards."""
        prefix_inp = _process_prefix(image)
        if prefix_inp is None:
            return {c: -1e9 for c in candidate_token_ids}
        prefix_len = prefix_inp["input_ids"].shape[1]
        out_dict = {}
        for concept in candidate_token_ids:
            lp = _score_prepared(prefix_inp, prefix_len, concept)
            out_dict[concept] = lp if lp is not None else -1e9
        return out_dict

    @torch.no_grad()
    def concept_logprob(image: Image.Image, concept: str) -> float | None:
        """Single-concept log P for one image."""
        if concept not in candidate_token_ids:
            return None
        prefix_inp = _process_prefix(image)
        if prefix_inp is None:
            return None
        prefix_len = prefix_inp["input_ids"].shape[1]
        return _score_prepared(prefix_inp, prefix_len, concept)

    # -------- Phase 2: per-prompt eval, with full 44-way retrieval on best drawing --------
    print(f"\n[v3eval] === Phase 2: per-prompt eval (N={args.n_samples} samples/prompt) ===", flush=True)
    per_prompt = []
    best_images: list[Image.Image] = []
    best_labels: list[str] = []
    best_strokes_by_slug: dict[str, list] = {}
    best_image_by_slug: dict[str, Image.Image] = {}
    for slug, prompt in HELDOUT:
        t0 = time.time()
        h_t = h_text(prompt)
        ids_list = av.generate_from_activation_batched(
            h_t, layer_ell=args.layer,
            n_samples=args.n_samples,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
        candidates = []
        for s_idx, ids in enumerate(ids_list):
            strokes, _ = av.vocab.decode_tokens_with_stats(ids.tolist())
            if len(strokes) < 2:
                continue
            img = stroke_render(strokes, display_scale=args.display_scale).convert("RGB")
            rew = concept_logprob(img, slug)
            if rew is None:
                continue
            candidates.append({"sample": s_idx, "n_strokes": len(strokes),
                               "reward": rew, "img": img, "strokes": strokes})
        if not candidates:
            print(f"[v3eval] {slug:10s}  all degenerate", flush=True)
            per_prompt.append({"slug": slug, "prompt": prompt, "all_degenerate": True})
            continue
        # Best-of-N by reward
        candidates.sort(key=lambda c: c["reward"], reverse=True)
        best = candidates[0]
        # 44-way retrieval on the best drawing
        all_lp = all_concept_logprobs(best["img"])
        # Margin
        sorted_lp = sorted(all_lp.items(), key=lambda kv: kv[1], reverse=True)
        correct_lp = all_lp.get(slug, float("-inf"))
        best_wrong = next((lp for c, lp in sorted_lp if c != slug), float("-inf"))
        margin = correct_lp - best_wrong
        top1 = sorted_lp[0][0]
        # Save
        full = stroke_render(best["strokes"], display_scale=4.0).convert("RGB")
        full.save(args.out_dir / f"{slug}.png")
        best_images.append(best["img"].resize((448, 448)))
        best_labels.append(f"{slug}  lp={best['reward']:+.2f}  top1={top1}")
        best_strokes_by_slug[slug] = best["strokes"]
        best_image_by_slug[slug] = best["img"]
        per_prompt.append({
            "slug": slug, "prompt": prompt,
            "best_reward": round(best["reward"], 4),
            "mean_reward": round(float(np.mean([c["reward"] for c in candidates])), 4),
            "std_reward": round(float(np.std([c["reward"] for c in candidates])), 4),
            "n_candidates": len(candidates),
            "best_n_strokes": best["n_strokes"],
            "all_44_logprobs": {c: round(v, 4) for c, v in all_lp.items()},
            "top1_concept": top1,
            "top1_correct": (top1 == slug),
            "rank_of_correct": next((i for i, (c, _) in enumerate(sorted_lp) if c == slug), -1),
            "margin": round(margin, 4),
            "wallclock_s": round(time.time() - t0, 1),
        })
        print(f"[v3eval] {slug:10s}  best_lp={best['reward']:+.2f}  top1={top1:10s} {'✓' if top1 == slug else 'X'}  margin={margin:+.3f}  ({time.time()-t0:.1f}s)", flush=True)

    valid = [r for r in per_prompt if not r.get("all_degenerate")]
    summary_p2 = {
        "n_prompts": len(per_prompt),
        "n_valid": len(valid),
        "best_reward_mean": float(np.mean([r["best_reward"] for r in valid])) if valid else None,
        "best_reward_std":  float(np.std([r["best_reward"] for r in valid])) if valid else None,
        "mean_reward_mean": float(np.mean([r["mean_reward"] for r in valid])) if valid else None,
        "top1_accuracy_44way": float(np.mean([r["top1_correct"] for r in valid])) if valid else None,
        "top1_chance_44way":   1.0 / len(ALL_CONCEPTS),
        "margin_mean": float(np.mean([r["margin"] for r in valid])) if valid else None,
        "per_prompt": per_prompt,
    }
    (args.out_dir / "per_prompt.json").write_text(json.dumps(summary_p2, indent=2))

    print(f"\n[v3eval] Phase 2 aggregate:", flush=True)
    print(f"  best_reward mean across {len(valid)} prompts: {summary_p2['best_reward_mean']:+.4f}", flush=True)
    print(f"  top-1 over 44 concepts: {summary_p2['top1_accuracy_44way']*100:.1f}%  (chance {summary_p2['top1_chance_44way']*100:.1f}%)", flush=True)
    print(f"  margin mean: {summary_p2['margin_mean']:+.4f}", flush=True)

    if best_images:
        grid_image(best_images, best_labels, cols=5).save(args.out_dir / "heldout_grid.png")

    # -------- Phase 3: pairwise discriminability across the 20 held-out drawings --------
    print(f"\n[v3eval] === Phase 3: pairwise discriminability (20×20) ===", flush=True)
    valid_slugs = [r["slug"] for r in per_prompt if not r.get("all_degenerate")]
    n = len(valid_slugs)
    if n < 2:
        print("[v3eval] not enough valid drawings for discriminability", flush=True)
        return
    M = np.zeros((n, n), dtype=np.float64)
    for i, slug_i in enumerate(valid_slugs):
        img_i = best_image_by_slug[slug_i]
        for j, slug_j in enumerate(valid_slugs):
            lp = concept_logprob(img_i, slug_j)
            M[i, j] = lp if lp is not None else -1e9
    diag = np.diag(M)
    off = (M.sum(axis=1) - diag) / (n - 1)
    disc = diag - off
    top1 = float(np.mean(np.argmax(M, axis=1) == np.arange(n)))
    summary_p3 = {
        "n_prompts": n,
        "slugs": valid_slugs,
        "matrix": M.tolist(),
        "diag_mean": float(diag.mean()),
        "off_diag_mean": float(off.mean()),
        "discriminability_mean": float(disc.mean()),
        "discriminability_std": float(disc.std()),
        "top1_retrieval_20way": top1,
        "top1_chance_20way": 1.0 / n,
        "per_prompt": [{"slug": s, "self": float(diag[i]),
                        "other_avg": float(off[i]), "delta": float(disc[i])}
                       for i, s in enumerate(valid_slugs)],
    }
    (args.out_dir / "discriminability.json").write_text(json.dumps(summary_p3, indent=2))
    print(f"[v3eval] diag mean       = {summary_p3['diag_mean']:+.3f}", flush=True)
    print(f"[v3eval] off-diag mean   = {summary_p3['off_diag_mean']:+.3f}", flush=True)
    print(f"[v3eval] discriminability= {summary_p3['discriminability_mean']:+.4f} ± {summary_p3['discriminability_std']:.3f}", flush=True)
    print(f"[v3eval] top-1 retrieval = {top1*100:.1f}%  (chance {100/n:.1f}%)", flush=True)

    # Heatmap (row-centered for visibility)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        M_show = M - M.mean(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(8, 7))
        vmax = float(np.percentile(np.abs(M_show), 95))
        im = ax.imshow(M_show, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(valid_slugs, rotation=60, fontsize=8, ha="right")
        ax.set_yticklabels(valid_slugs, fontsize=8)
        ax.set_xlabel("scoring candidate concept")
        ax.set_ylabel("drawing source prompt")
        ax.set_title(f"log P(concept_j | drawing_i) — disc {summary_p3['discriminability_mean']:+.3f}  top1 {top1*100:.0f}%")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(args.out_dir / "discriminability.png", dpi=120)
    except Exception as e:
        print(f"[v3eval] discriminability plot failed: {e}", flush=True)

    # Retrieval table: per-prompt top-3 (by lp) over 44 concepts
    rows = []
    for r in per_prompt:
        if "all_44_logprobs" not in r:
            continue
        sorted_lp = sorted(r["all_44_logprobs"].items(), key=lambda kv: kv[1], reverse=True)
        rows.append({"slug": r["slug"], "top3": sorted_lp[:3], "rank_of_correct": r["rank_of_correct"]})
    (args.out_dir / "retrieval_top3.json").write_text(json.dumps(rows, indent=2))

    print(f"\n[v3eval] DONE → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
