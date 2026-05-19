"""Custom LoRA for Gemma4ClippableLinear.

PEFT 0.13 only auto-recognises `torch.nn.Linear` and friends; Gemma 4 wraps
its attention projections in a custom `Gemma4ClippableLinear` class
(which contains `self.linear: nn.Linear` plus runtime clipping logic).
PEFT therefore refuses to attach LoRA to those modules.

This module adds LoRA via a parallel branch that we splice into the wrapped
module's forward pass. The wrapped module's original behaviour is preserved
exactly; we just *add* a `α/r · B · A · x` delta on top.

Usage
-----
    from ar.lora_gemma4 import attach_lora_to_ar
    ar = TruncatedGemmaAR.from_pretrained(...)
    lora_params = attach_lora_to_ar(ar, layer_ell=12, rank=16, alpha=32)
    # lora_params is the list of LoRADelta modules; everything else stays frozen
    optim = AdamW(list(ar.linear.parameters()) + [p for m in lora_params for p in m.parameters()], lr=...)

Saving/restoring LoRA state:
    state = lora_state_dict(ar)               # → dict of LoRA A/B tensors keyed by qualified module name
    torch.save({"lora": state, "linear": ar.linear.state_dict(), ...}, "head.pt")

    # later
    ar = TruncatedGemmaAR.from_pretrained(..., use_backbone_lora=True, lora_rank=16, lora_alpha=32)
    load_lora_state(ar, torch.load("head.pt")["lora"])

Design choices and trade-offs
-----------------------------
- We patch by REPLACING `module.forward` with a closure that calls the original
  forward and adds the LoRA delta. We KEEP a reference to the original forward
  on the module itself (`module._orig_forward`) so the patch is idempotent and
  can be safely re-applied (it detects the prior patch and short-circuits).
- LoRA A is initialised with Kaiming-uniform (matches PEFT). B is zero-init so
  the LoRA delta is exactly 0 at attach time — the model behaves identically
  to a non-LoRA-augmented one until training starts.
- LoRA params live in the same dtype as the wrapped linear's weight (bf16 by
  default for Gemma 4). For training stability we still keep MSE losses in
  fp32 (cast at the loss boundary).
- We target both the vision_tower's self-attention AND the language_model's
  first-ℓ self-attention. Vision LoRA adapts Gemma's vision encoder to our
  line-art distribution; language LoRA adapts the projection from image
  features into the residual stream's coordinate frame.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRADelta(nn.Module):
    """Parallel LoRA branch with zero-init B so the initial delta is identically 0.

    delta(x) = (alpha / rank) * x @ A.T @ B.T  (equivalent to B @ A @ x in matrix form)

    A: (rank, in_dim), Kaiming init
    B: (out_dim, rank), zero init
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int = 16, alpha: int = 32,
                 dtype: torch.dtype = torch.bfloat16, device: torch.device | str = "cuda"):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        # Match PEFT's convention: A (rank, in), B (out, rank)
        self.A = nn.Parameter(torch.zeros(rank, in_dim, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.zeros(out_dim, rank, dtype=dtype, device=device))
        # Kaiming-uniform init for A (the "down-projection")
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        # B stays zero so the entire LoRA delta is zero at init — no behaviour change.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (..., in_dim); F.linear treats trailing dim as the input dim.
        return F.linear(F.linear(x, self.A), self.B) * self.scale

    def extra_repr(self) -> str:
        return f"in={self.in_dim}, out={self.out_dim}, rank={self.rank}, alpha={self.alpha}, scale={self.scale:.4f}"


def _layer_index_from_name(name: str) -> int | None:
    """Parse a fully-qualified module name like
    `model.language_model.layers.12.self_attn.q_proj` and return 12.

    Returns None if no `.layers.<int>.` segment is found.
    """
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def _module_kind_from_name(name: str) -> str:
    """Return one of 'vision' or 'language' or 'other' based on the module path."""
    if "vision_tower" in name:
        return "vision"
    if "language_model" in name:
        return "language"
    return "other"


def attach_lora_to_ar(
    ar,
    layer_ell: int,
    rank: int = 16,
    alpha: int = 32,
    targets: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    include_vision_tower: bool = True,
    include_language_first_ell: bool = True,
    verbose: bool = True,
) -> list[LoRADelta]:
    """Walk `ar.backbone` and attach a parallel LoRA branch to every matching Gemma4ClippableLinear.

    Modules matched:
      * if `include_vision_tower`: all `vision_tower.encoder.layers.{i}.self_attn.{target}` modules
      * if `include_language_first_ell`: `language_model.layers.{i}.self_attn.{target}` for i in [0, ℓ)
        (the language layers whose residual stream the AR reads)

    Returns the list of attached `LoRADelta` modules in deterministic insertion order.
    Their parameters can be passed straight to an optimiser.
    """
    attached: list[LoRADelta] = []
    skipped = 0
    seen: set[str] = set()

    target_set = set(targets)

    # Choose a representative device/dtype from the backbone (first parameter we find)
    p0 = next(ar.backbone.parameters())
    target_dtype = p0.dtype
    target_device = p0.device

    # Walk modules in deterministic (sorted) order
    named_modules = sorted(ar.backbone.named_modules(), key=lambda kv: kv[0])

    for name, module in named_modules:
        if module.__class__.__name__ != "Gemma4ClippableLinear":
            continue
        proj_name = name.rsplit(".", 1)[-1]
        if proj_name not in target_set:
            skipped += 1
            continue
        kind = _module_kind_from_name(name)
        layer_idx = _layer_index_from_name(name)

        keep = False
        if kind == "vision" and include_vision_tower:
            keep = True
        elif kind == "language" and include_language_first_ell:
            if layer_idx is not None and layer_idx < layer_ell:
                keep = True
        if not keep:
            skipped += 1
            continue
        if name in seen:
            continue
        seen.add(name)

        inner: nn.Linear = module.linear
        in_dim = inner.in_features
        out_dim = inner.out_features

        # Build LoRA branch
        lora = LoRADelta(
            in_dim=in_dim, out_dim=out_dim, rank=rank, alpha=alpha,
            dtype=target_dtype, device=target_device,
        )
        # Register as a child module so torch tracks parameters
        module.add_module("_lora", lora)

        # Patch forward idempotently
        if not getattr(module, "_lora_patched", False):
            orig_forward = module.forward
            module._orig_forward = orig_forward

            def make_patched(orig, mod):
                def patched(x, *args, **kwargs):
                    base = orig(x, *args, **kwargs)
                    delta = mod._lora(x)
                    return base + delta
                return patched

            module.forward = make_patched(orig_forward, module)
            module._lora_patched = True

        attached.append(lora)

    if verbose:
        n_params = sum(p.numel() for m in attached for p in m.parameters())
        print(
            f"[lora] attached LoRA(r={rank}, α={alpha}) to {len(attached)} Gemma4ClippableLinear modules; "
            f"skipped {skipped}; total trainable LoRA params: {n_params / 1e6:.2f}M",
            flush=True,
        )
    return attached


def lora_state_dict(ar) -> dict[str, torch.Tensor]:
    """Collect a flat state-dict of all LoRA A and B tensors in `ar.backbone`.

    Keys are `<qualified_module_name>.A` and `<qualified_module_name>.B`.
    Suitable for `torch.save({"lora": lora_state_dict(ar), ...}, path)`.
    """
    state = {}
    for name, module in ar.backbone.named_modules():
        if not hasattr(module, "_lora"):
            continue
        state[f"{name}.A"] = module._lora.A.detach().clone()
        state[f"{name}.B"] = module._lora.B.detach().clone()
    return state


def load_lora_state(ar, state: dict[str, torch.Tensor], strict: bool = True) -> None:
    """Restore LoRA A/B weights into a previously-attached AR.

    `attach_lora_to_ar` must have been called first to create the LoRA modules.
    """
    loaded = 0
    expected = 0
    for name, module in ar.backbone.named_modules():
        if not hasattr(module, "_lora"):
            continue
        expected += 1
        a_key = f"{name}.A"
        b_key = f"{name}.B"
        if a_key not in state or b_key not in state:
            if strict:
                raise KeyError(f"missing LoRA tensors for {name} in state dict")
            continue
        with torch.no_grad():
            module._lora.A.copy_(state[a_key].to(module._lora.A.dtype).to(module._lora.A.device))
            module._lora.B.copy_(state[b_key].to(module._lora.B.dtype).to(module._lora.B.device))
        loaded += 1
    if strict and loaded != expected:
        raise RuntimeError(f"loaded {loaded} LoRA modules, expected {expected}")


def freeze_all_but_lora_and_linear(ar) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Set requires_grad correctly for v1.0 training: only LoRA A/B and the
    final `Linear(d, d)` head are trainable; everything else frozen.

    Returns (lora_params, head_params) for the optimiser.
    """
    lora_params: list[nn.Parameter] = []
    for p in ar.backbone.parameters():
        p.requires_grad = False
    for name, module in ar.backbone.named_modules():
        if hasattr(module, "_lora"):
            for p in module._lora.parameters():
                p.requires_grad = True
                lora_params.append(p)
    head_params = list(ar.linear.parameters())
    for p in head_params:
        p.requires_grad = True
    return lora_params, head_params


def lora_summary(ar) -> str:
    """Human-readable summary of attached LoRA modules — useful for debug prints."""
    rows = []
    for name, module in ar.backbone.named_modules():
        if not hasattr(module, "_lora"):
            continue
        lora = module._lora
        rows.append(f"  {name}: in={lora.in_dim} out={lora.out_dim} r={lora.rank}")
    if not rows:
        return "no LoRA attached"
    return f"LoRA attached to {len(rows)} modules:\n" + "\n".join(rows)
