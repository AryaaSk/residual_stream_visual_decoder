"""Activation Projector for the AV.

A learnable mapping from residual-stream space (where activations live) to
embedding space (where the AV's backbone expects inputs).

v1.0/v1.1 injected α·h directly into the AV's embedding slot. That assumes
activation-space ≈ embedding-space, which isn't true: h at L24 has norm ~70
and lives in a coordinate frame produced by 24 layers of transformer
computation, while embeddings have norm ~10 and live at the pre-transformer
input.

This projector is the missing learnable surface. Init as α·I so the v1.1
behaviour is exactly recovered at step 0; training pulls it away.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActProjector(nn.Module):
    """Linear projection h ∈ ℝᵈ → embedding ∈ ℝᵈ.

    Init: weight = α·I, bias = 0. So at step 0, projector(h) == α·h, matching
    the v1.1 scaled-injection behaviour exactly.
    """

    def __init__(self, d: int, alpha_init: float = 0.5,
                 dtype: torch.dtype = torch.bfloat16,
                 device: torch.device | str = "cuda"):
        super().__init__()
        self.d = d
        self.alpha_init = alpha_init
        self.linear = nn.Linear(d, d, bias=True, dtype=dtype, device=device)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(d, dtype=dtype, device=device) * alpha_init)
            self.linear.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h shape: (..., d) — the linear handles any leading dims
        return self.linear(h)

    def state(self) -> dict:
        return {
            "d": self.d,
            "alpha_init": self.alpha_init,
            "weight": self.linear.weight.detach().cpu().clone(),
            "bias": self.linear.bias.detach().cpu().clone(),
        }

    @classmethod
    def from_state(cls, state: dict,
                   dtype: torch.dtype = torch.bfloat16,
                   device: torch.device | str = "cuda") -> "ActProjector":
        proj = cls(d=int(state["d"]), alpha_init=float(state["alpha_init"]),
                   dtype=dtype, device=device)
        with torch.no_grad():
            proj.linear.weight.copy_(state["weight"].to(dtype=dtype, device=device))
            proj.linear.bias.copy_(state["bias"].to(dtype=dtype, device=device))
        return proj

    def extra_repr(self) -> str:
        return f"d={self.d}, alpha_init={self.alpha_init}"
