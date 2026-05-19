"""Fraction-of-Variance-Explained (FVE) metric.

For a set of activation pairs (h, ĥ):

    FVE = 1 - Var(h - ĥ) / Var(h)

Higher is better. NLA on Claude reaches 0.6-0.8.

Also computes mean cosine similarity and L2-MSE for additional reporting.
"""

from __future__ import annotations

import numpy as np
import torch


def fve(h: torch.Tensor, h_hat: torch.Tensor) -> float:
    """Fraction of variance in h explained by h_hat."""
    assert h.shape == h_hat.shape, f"shape mismatch {h.shape} vs {h_hat.shape}"
    h = h.float()
    h_hat = h_hat.float()
    residual = h - h_hat
    var_res = residual.var(dim=0, unbiased=False).sum()
    var_h = h.var(dim=0, unbiased=False).sum()
    return float(1.0 - (var_res / var_h.clamp_min(1e-12)).item())


def mean_cosine(h: torch.Tensor, h_hat: torch.Tensor) -> float:
    cos = torch.nn.functional.cosine_similarity(h.float(), h_hat.float(), dim=-1)
    return float(cos.mean().item())


def mse(h: torch.Tensor, h_hat: torch.Tensor) -> float:
    return float(torch.nn.functional.mse_loss(h.float(), h_hat.float()).item())


def metric_summary(h: torch.Tensor, h_hat: torch.Tensor) -> dict[str, float]:
    return {
        "fve": fve(h, h_hat),
        "cosine": mean_cosine(h, h_hat),
        "mse": mse(h, h_hat),
        "round_trip_loss_2_1_minus_cos": float(2.0 * (1.0 - mean_cosine(h, h_hat))),
    }
