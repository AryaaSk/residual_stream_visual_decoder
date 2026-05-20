"""Analyse the geometry of Gemma 4 activations at a target layer across diverse prompts.

Questions answered:
    - What's the mean L2 norm of activations?
    - What's the within-prompt-cluster cosine (across diverse prompts)?
    - How much per-prompt variance survives mean-centering?
    - At which layer ℓ are activations MOST discriminative across prompts?

Run as:
    python code/eval/activation_geometry.py --model-id Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

DIVERSE_PROMPTS = [
    "The capital of France is",
    "I am thinking about a dog.",
    "I am thinking about a cat.",
    "I am picturing a small house with a red roof.",
    "Imagine a triangle inscribed in a circle.",
    "When the storm hit, the village",
    "Paris, the city of lights, is famous for the Eiffel",
    "What is 47 + 38? The answer is",
    "She received the news and felt deeply",
    "def fibonacci(n):",
    "The three primary colours are",
    "Once upon a time, in a kingdom far away,",
    "The funeral was somber and",
    "Her face lit up with joy",
    "An adult elephant is taller than a",
    "The smallest prime number is",
    "I am picturing a smiling face.",
    "Antarctica is colder than",
    "I am thinking about deep sadness.",
    "Right now I am thinking about",
    "The chemical formula for water is",
    "SELECT * FROM users WHERE",
    "if __name__ == '__main__':",
    "The mother of Barack Obama's wife is named",
    "Paris is not the capital of",
    "A short story: Once upon a time,",
    "The colour of the sky at noon is",
    "When she opened the box, she found",
    "I saw the man with the telescope",
    "The four seasons of the year are",
]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="If given, only analyse these layers. Else all.")
    parser.add_argument("--out", type=Path, default=Path("findings/activation_geometry.json"))
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[geom] loading {args.model_id}", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    tok = AutoTokenizer.from_pretrained(args.model_id)

    n_layers = m.config.text_config.num_hidden_layers
    hidden_size = m.config.text_config.hidden_size
    layers = args.layers if args.layers else list(range(n_layers + 1))
    print(f"[geom] {len(DIVERSE_PROMPTS)} prompts, {len(layers)} layer positions of {n_layers + 1}", flush=True)

    # Collect (T, hidden) per layer for the final-token position
    h_by_layer: dict[int, torch.Tensor] = {ell: [] for ell in layers}
    for text in DIVERSE_PROMPTS:
        enc = tok(text, return_tensors="pt").to("cuda")
        out = m(**enc, output_hidden_states=True, use_cache=False)
        for ell in layers:
            h_by_layer[ell].append(out.hidden_states[ell][0, -1, :].float().cpu())

    rows = []
    for ell in layers:
        h = torch.stack(h_by_layer[ell], dim=0)  # (N, hidden)
        norms = h.norm(dim=1)
        # Pairwise cosine across all distinct pairs
        h_norm = h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)
        cos_mat = h_norm @ h_norm.T
        n = h.shape[0]
        # exclude diagonal
        off_diag = cos_mat[torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)]
        # Mean-centered variance
        mu = h.mean(dim=0, keepdim=True)
        h_centered = h - mu
        var_centered = h_centered.var(dim=0).sum()
        var_raw = h.var(dim=0).sum()
        ratio_centered_to_raw = (var_centered / var_raw.clamp_min(1e-12)).item()
        # Effective rank (PCA)
        u, s, v = torch.linalg.svd(h_centered, full_matrices=False)
        s_sum = s.sum().item()
        s_norm = (s / s.sum().clamp_min(1e-12)).clamp_min(1e-12)
        eff_rank = float(torch.exp(-(s_norm * s_norm.log()).sum()).item())

        row = {
            "layer": ell,
            "mean_norm": float(norms.mean().item()),
            "std_norm": float(norms.std().item()),
            "mean_pairwise_cosine": float(off_diag.mean().item()),
            "std_pairwise_cosine": float(off_diag.std().item()),
            "min_pairwise_cosine": float(off_diag.min().item()),
            "max_pairwise_cosine": float(off_diag.max().item()),
            "var_total": float(var_raw.item()),
            "var_centered_over_raw": float(ratio_centered_to_raw),
            "effective_rank": eff_rank,
        }
        rows.append(row)
        print(
            f"  L{ell:02d}  ||h||={row['mean_norm']:6.2f}±{row['std_norm']:5.2f}  "
            f"pair_cos={row['mean_pairwise_cosine']:.3f}±{row['std_pairwise_cosine']:.3f} "
            f"var_centered/raw={row['var_centered_over_raw']:.3f} "
            f"eff_rank={row['effective_rank']:.2f}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"prompts": DIVERSE_PROMPTS, "n_layers": n_layers + 1, "rows": rows}, indent=2))
    print(f"\n[geom] wrote {args.out}")


if __name__ == "__main__":
    main()
