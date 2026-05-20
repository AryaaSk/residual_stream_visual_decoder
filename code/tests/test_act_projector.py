"""Sanity tests for ActProjector + AV LoRA integration.

Tests run without loading Gemma 4 (CPU-only, minimal nn.Modules used as stand-ins
where the full model is not needed).
"""

from __future__ import annotations

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.projector import ActProjector


def test_projector_init_is_scaled_identity():
    """At init, projector(h) == α·h (no bias). This guarantees v1.1 behaviour
    at step 0 so we can isolate the change."""
    d = 64
    alpha = 0.5
    proj = ActProjector(d=d, alpha_init=alpha, dtype=torch.float32, device="cpu")
    h = torch.randn(4, d)
    out = proj(h)
    expected = h * alpha
    assert torch.allclose(out, expected, atol=1e-5), f"projector init not α·I: max diff {(out - expected).abs().max().item()}"
    print("[test] projector_init_is_scaled_identity OK")


def test_projector_output_shape():
    """Projector preserves shape; works on (B, d) and (d,)."""
    d = 64
    proj = ActProjector(d=d, dtype=torch.float32, device="cpu")
    assert proj(torch.randn(d)).shape == (d,)
    assert proj(torch.randn(3, d)).shape == (3, d)
    assert proj(torch.randn(2, 5, d)).shape == (2, 5, d)
    print("[test] projector_output_shape OK")


def test_projector_gradients_flow():
    """Confirm gradients flow back to the projector weight."""
    d = 32
    proj = ActProjector(d=d, dtype=torch.float32, device="cpu")
    h = torch.randn(2, d, requires_grad=False)
    out = proj(h)
    out.sum().backward()
    assert proj.linear.weight.grad is not None
    assert proj.linear.weight.grad.abs().sum().item() > 0
    assert proj.linear.bias.grad is not None
    print("[test] projector_gradients_flow OK")


def test_projector_state_round_trip():
    """save → from_state recovers exactly."""
    d = 32
    proj = ActProjector(d=d, alpha_init=0.5, dtype=torch.float32, device="cpu")
    # Train a step to perturb away from init
    h = torch.randn(2, d)
    out = proj(h)
    out.sum().backward()
    with torch.no_grad():
        proj.linear.weight -= 0.01 * proj.linear.weight.grad
        proj.linear.bias -= 0.01 * proj.linear.bias.grad
    state = proj.state()
    proj2 = ActProjector.from_state(state, dtype=torch.float32, device="cpu")
    h_test = torch.randn(3, d)
    assert torch.allclose(proj(h_test), proj2(h_test), atol=1e-6)
    print("[test] projector_state_round_trip OK")


def test_lora_walker_works_on_plain_linear():
    """Verify that _attach_lora_walk attaches LoRA to plain nn.Linear modules
    (the case for AV's language layers, which don't use Gemma4ClippableLinear)."""
    from ar.lora_gemma4 import _attach_lora_walk

    # Build a tiny model whose language_model layers contain plain Linears
    class FakeSelfAttn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.q_proj = nn.Linear(d, d, bias=False)
            self.k_proj = nn.Linear(d, d, bias=False)
            self.v_proj = nn.Linear(d, d, bias=False)
            self.o_proj = nn.Linear(d, d, bias=False)

    class FakeLayer(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.self_attn = FakeSelfAttn(d)

    class FakeLM(nn.Module):
        def __init__(self, d, n_layers):
            super().__init__()
            self.language_model = nn.ModuleDict({
                "layers": nn.ModuleList([FakeLayer(d) for _ in range(n_layers)]),
            })

    d, n = 16, 4
    fake = FakeLM(d, n)
    attached = _attach_lora_walk(
        fake,
        rank=4, alpha=8,
        targets=("q_proj", "k_proj", "v_proj", "o_proj"),
        include_vision_tower=False,
        include_language_first_ell=True,
        language_layer_limit=2,
        accept_classes=("Linear",),
        verbose=False,
    )
    # 4 projections × 2 layers = 8
    assert len(attached) == 8, f"expected 8 LoRA modules, got {len(attached)}"
    # Verify LoRA delta is zero at init (B = 0)
    x = torch.randn(1, 4, d)
    q = fake.language_model["layers"][0].self_attn.q_proj
    base = nn.functional.linear(x, q.weight, q.bias)
    patched = q(x)
    assert torch.allclose(base, patched, atol=1e-6), "LoRA delta should be zero at init"
    print("[test] lora_walker_works_on_plain_linear OK")


def test_lora_state_round_trip():
    """attach → train one step → save state → fresh attach → load → outputs match."""
    from ar.lora_gemma4 import _attach_lora_walk, lora_state_dict, load_lora_state

    class FakeAtt(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.q_proj = nn.Linear(d, d, bias=False)
            self.k_proj = nn.Linear(d, d, bias=False)
            self.v_proj = nn.Linear(d, d, bias=False)
            self.o_proj = nn.Linear(d, d, bias=False)

    class FakeLayer(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.self_attn = FakeAtt(d)

    class FakeLM(nn.Module):
        def __init__(self, d, n):
            super().__init__()
            self.language_model = nn.ModuleDict({"layers": nn.ModuleList([FakeLayer(d) for _ in range(n)])})

    d, n = 8, 2
    m1 = FakeLM(d, n)
    attached = _attach_lora_walk(
        m1, rank=2, alpha=4,
        targets=("q_proj", "k_proj", "v_proj", "o_proj"),
        include_vision_tower=False, include_language_first_ell=True,
        language_layer_limit=n, accept_classes=("Linear",), verbose=False,
    )
    # Perturb LoRA B (originally zero) to make state non-trivial
    with torch.no_grad():
        for lora in attached:
            lora.B.add_(torch.randn_like(lora.B) * 0.01)
    state = lora_state_dict(m1)

    # Fresh model + LoRA, load
    m2 = FakeLM(d, n)
    _attach_lora_walk(
        m2, rank=2, alpha=4,
        targets=("q_proj", "k_proj", "v_proj", "o_proj"),
        include_vision_tower=False, include_language_first_ell=True,
        language_layer_limit=n, accept_classes=("Linear",), verbose=False,
    )
    # Match base weights too so we can compare outputs
    m2.load_state_dict(m1.state_dict(), strict=False)
    load_lora_state(m2, state)

    x = torch.randn(1, 3, d)
    q1 = m1.language_model["layers"][0].self_attn.q_proj(x)
    q2 = m2.language_model["layers"][0].self_attn.q_proj(x)
    assert torch.allclose(q1, q2, atol=1e-6), f"LoRA round-trip mismatch: {(q1 - q2).abs().max()}"
    print("[test] lora_state_round_trip OK")


if __name__ == "__main__":
    test_projector_init_is_scaled_identity()
    test_projector_output_shape()
    test_projector_gradients_flow()
    test_projector_state_round_trip()
    test_lora_walker_works_on_plain_linear()
    test_lora_state_round_trip()
    print("\nALL OK")
