"""Phase 0 validation: is `log P(concept_word | image, prompt)` a usable v3 reward signal?

For each (concept, real canonical drawing) pair in `data/canonical_drawings_top5.jsonl`:
  - render the drawing
  - feed it to frozen Qwen 3.5-4B (loaded via AutoModelForImageTextToText — the loader
    that actually consumes pixel_values; see RESEARCH_NOTES.md "loader bug" section)
  - prompt-prefix: `<|im_start|>user\\n<image>What is this a drawing of?<|im_end|>\\n<|im_start|>assistant\\nA drawing of a `
  - score every CANDIDATE concept word as the continuation (sum log-prob of its tokens)

Output a full pairwise matrix `M[i, j] = log P(concept_j | drawing_i, prefix)` and report:
  - diag_mean   = log P(correct | own drawing)
  - off_mean    = log P(wrong | drawing)
  - disc        = diag - off (mean across pairs)
  - top1        = fraction of drawings where argmax_j M[i, j] == i
  - margin      = log P(correct) - log P(best_wrong)

Decision gate (see plan):
  top1 > 60%, disc > 1.0:    STRONG signal → Phase 1
  top1 20-60%, disc > 0.3:   moderate signal → Phase 1, slow REINFORCE expected
  top1 ≈ chance, disc ≈ 0:   foundational architecture broken → ship negative finding

Optimisation: for each image, compute the prompt-prefix forward ONCE (storing
past_key_values), then for each candidate just forward the target tokens with the
cached prefix. ~30× speedup vs naive.

Output:
    findings/v3/gen_likelihood_validation/pairwise.json
    findings/v3/gen_likelihood_validation/pairwise.png
    findings/v3/gen_likelihood_validation/verdict.md
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render import render as stroke_render
from stroke_tokenizer import Stroke


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def one_drawing_per_concept(rows: list[dict]) -> list[dict]:
    by_concept: dict[str, dict] = {}
    for r in rows:
        cap = r.get("caption", "")
        concept = (cap.replace("a drawing of an ", "")
                       .replace("a drawing of a ", "")
                       .replace("a drawing of ", "")
                       .strip().rstrip("s"))
        if concept and concept not in by_concept:
            by_concept[concept] = {"concept": concept, "caption": cap, "strokes": r["strokes"]}
    return list(by_concept.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--data", type=Path, default=Path("data/canonical_drawings_top5.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("findings/v3/gen_likelihood_validation"))
    p.add_argument("--max-concepts", type=int, default=30)
    p.add_argument("--display-scale", type=float, default=2.0)
    p.add_argument("--question", default="What is this a drawing of?")
    p.add_argument("--continuation-prefix", default="A drawing of a")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[genv] loading Qwen 3.5-4B (ImageTextToText) ...", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    device = next(model.parameters()).device

    raw = load_jsonl(args.data)
    rows = one_drawing_per_concept(raw)
    if len(rows) > args.max_concepts:
        rows = rows[: args.max_concepts]
    n = len(rows)
    print(f"[genv] {n} distinct concepts", flush=True)

    # Pre-tokenise each candidate concept as " {concept}" (with leading space) —
    # the leading space is critical because byte-pair tokenisers treat "cat"
    # mid-string differently from " cat" at word boundary. The continuation
    # prefix ends without a trailing space.
    target_token_ids: list[list[int]] = []
    for r in rows:
        # Add a leading space so the token boundary is correct
        toks = proc.tokenizer.encode(" " + r["concept"], add_special_tokens=False)
        target_token_ids.append(toks)
        print(f"  concept {r['concept']:10s} → {len(toks)} tokens = {toks}", flush=True)

    # Build prompt-prefix text once
    msgs = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": args.question}],
    }]
    PREFIX_TEXT = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + args.continuation_prefix
    tok = proc.tokenizer

    @torch.no_grad()
    def process_prefix(image):
        """Process image+prefix once per image. Returns the prefix tensor dict (on device)
        + the prefix_len. Reused for all candidates."""
        inp = proc(text=[PREFIX_TEXT], images=[image], return_tensors="pt").to(device)
        prefix_len = inp["input_ids"].shape[1]
        return inp, prefix_len

    @torch.no_grad()
    def score_candidate_fast(prefix_inp, prefix_len: int, candidate_concept: str) -> float:
        """Score: sum log P(target_tokens | image + prefix).

        Reuses the prefix processor output. Tokenizes candidate text-only, extends every
        sequence-shaped multimodal tensor (input_ids / attention_mask / mm_token_type_ids
        — text type 0 for the appended tokens), then does one model forward.
        """
        cand_ids = tok.encode(" " + candidate_concept, add_special_tokens=False)
        if not cand_ids:
            return -1e9
        cand_t = torch.tensor([cand_ids], dtype=torch.long, device=device)
        full = {k: v for k, v in prefix_inp.items()}
        full["input_ids"] = torch.cat([prefix_inp["input_ids"], cand_t], dim=1)
        if "attention_mask" in full:
            full["attention_mask"] = torch.cat(
                [prefix_inp["attention_mask"], torch.ones_like(cand_t)], dim=1
            )
        if "mm_token_type_ids" in full:
            # Appended tokens are text (type 0), not image
            full["mm_token_type_ids"] = torch.cat(
                [prefix_inp["mm_token_type_ids"], torch.zeros_like(cand_t)], dim=1
            )
        out = model(**full, use_cache=False)
        T = full["input_ids"].shape[1]
        logits = out.logits[0, prefix_len - 1: T - 1, :]
        target = full["input_ids"][0, prefix_len:]
        logp = F.log_softmax(logits.float(), dim=-1)
        chosen = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return float(chosen.sum().item())

    # Pre-render images
    images = []
    for r in rows:
        stroke_objs = [Stroke(dx=s["dx"], dy=s["dy"], pen=s["pen"]) for s in r["strokes"]]
        images.append(stroke_render(stroke_objs, display_scale=args.display_scale).convert("RGB"))

    # Build pairwise matrix M[i, j] = log P(concept_j | drawing_i)
    M = np.full((n, n), -1e9, dtype=np.float64)
    t0 = time.time()
    for i in range(n):
        # Process image+prefix ONCE per image
        prefix_inp, prefix_len = process_prefix(images[i])
        for j in range(n):
            try:
                lp = score_candidate_fast(prefix_inp, prefix_len, rows[j]["concept"])
            except Exception as e:
                print(f"[genv]   score failed (i={i}, j={j}): {e}", flush=True)
                lp = -1e9
            M[i, j] = lp
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n - i - 1)
        print(f"[genv] image {i+1}/{n}  {rows[i]['concept']:10s}  done in {elapsed:.1f}s (eta {eta:.0f}s)", flush=True)

    # Stats
    diag = np.diag(M)
    off_mask = ~np.eye(n, dtype=bool)
    off = M[off_mask].reshape(n, n - 1).mean(axis=1)
    disc = diag - off
    top1 = np.mean(np.argmax(M, axis=1) == np.arange(n))
    # Per-row top-1 retrieval: did the correct concept have the highest log-prob?
    correct_rank = np.array([
        int((M[i] > M[i, i]).sum())  # 0 = top-1; rank-out-of-n
        for i in range(n)
    ])
    # Margin: log P(correct) - log P(best_wrong)
    best_wrong = np.array([
        M[i, [j for j in range(n) if j != i]].max()
        for i in range(n)
    ])
    margin = diag - best_wrong

    summary = {
        "n_concepts": n,
        "concepts": [r["concept"] for r in rows],
        "matrix": M.tolist(),
        "diag_mean": float(diag.mean()),
        "diag_std": float(diag.std()),
        "off_diag_mean": float(off.mean()),
        "discriminability_mean": float(disc.mean()),
        "discriminability_std": float(disc.std()),
        "top1_retrieval": float(top1),
        "top1_chance": 1.0 / n,
        "margin_mean": float(margin.mean()),
        "margin_std": float(margin.std()),
        "per_concept": [
            {"concept": rows[i]["concept"],
             "log_p_correct": float(diag[i]),
             "log_p_best_wrong": float(best_wrong[i]),
             "margin": float(margin[i]),
             "rank_of_correct": int(correct_rank[i]),  # 0 = correct is top-1
            }
            for i in range(n)
        ],
    }
    (args.out_dir / "pairwise.json").write_text(json.dumps(summary, indent=2))
    print(f"[genv] saved {args.out_dir / 'pairwise.json'}", flush=True)

    print(f"\n=== Phase 0 verdict ===", flush=True)
    print(f"  diag (correct)      = {diag.mean():+.3f} ± {diag.std():.3f}", flush=True)
    print(f"  off-diag (wrong)    = {off.mean():+.3f}", flush=True)
    print(f"  discriminability    = {disc.mean():+.4f} ± {disc.std():.3f}  (log-prob units)", flush=True)
    print(f"  top-1 retrieval     = {top1*100:.1f}%  (chance {100/n:.1f}%)", flush=True)
    print(f"  margin (correct - best_wrong) = {margin.mean():+.3f} ± {margin.std():.3f}", flush=True)
    print()
    if top1 > 0.60 and disc.mean() > 1.0:
        verdict = "STRONG signal — proceed to Phase 1 (training)"
    elif top1 > 0.20 and disc.mean() > 0.3:
        verdict = "MODERATE signal — proceed to Phase 1 (REINFORCE may be slow)"
    else:
        verdict = "WEAK / NO signal — even real drawings can't be decoded; ship negative finding"
    print(f"VERDICT: {verdict}\n", flush=True)

    verdict_md = f"""# Phase 0 — generation-likelihood validation

Goal: verify that frozen Qwen 3.5-4B assigns higher log P(correct_concept | image, prompt) than for wrong concepts, on REAL canonical drawings. This is the foundational signal v3 trains on.

## Setup

- Loader: `AutoModelForImageTextToText` (the correct one for Qwen3-VL; `AutoModelForCausalLM` silently drops pixel_values).
- Prompt: chat-template-wrapped `<image>What is this a drawing of?<|im_end|>\\n<|im_start|>assistant\\nA drawing of a `
- For each (concept, canonical drawing), score every other concept's log-prob as the continuation. Pairwise {n}×{n} matrix.

## Results

| metric                 | value                                          |
|-----------------------:|:----------------------------------------------:|
| diag mean (correct)    | {diag.mean():+.3f} ± {diag.std():.3f}          |
| off-diag mean (wrong)  | {off.mean():+.3f}                              |
| discriminability       | **{disc.mean():+.4f}** ± {disc.std():.3f}      |
| top-1 retrieval        | **{top1*100:.1f}%**  (chance {100/n:.1f}%)     |
| margin (correct − wrong) | {margin.mean():+.3f} ± {margin.std():.3f}     |

## Verdict

{verdict}
"""
    (args.out_dir / "verdict.md").write_text(verdict_md)
    print(f"[genv] saved {args.out_dir / 'verdict.md'}", flush=True)

    # Heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Normalize by row for visualization
        M_show = M - M.mean(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(9, 8))
        vmax = float(np.percentile(np.abs(M_show), 95))
        im = ax.imshow(M_show, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([r["concept"] for r in rows], rotation=70, fontsize=8, ha="right")
        ax.set_yticklabels([r["concept"] for r in rows], fontsize=8)
        ax.set_xlabel("candidate concept")
        ax.set_ylabel("real image's concept")
        ax.set_title(f"log P(concept_j | drawing_i) [row-centered] — disc {disc.mean():+.3f}, top1 {top1*100:.0f}%")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(args.out_dir / "pairwise.png", dpi=110)
        print(f"[genv] saved {args.out_dir / 'pairwise.png'}", flush=True)
    except Exception as e:
        print(f"[genv] plot failed: {e}", flush=True)


if __name__ == "__main__":
    main()
