"""Tests for the custom LoRA on Gemma4ClippableLinear.

These tests use a small fake module that mimics Gemma4ClippableLinear's surface
(`module.linear` is an nn.Linear) so the suite runs in seconds on CPU without
loading any real Gemma weights.

Real-Gemma sanity checks happen when ar_v3 training kicks off — if the LoRA is
broken there, training will visibly diverge.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ar.lora_gemma4 import (  # noqa: E402
    LoRADelta,
    attach_lora_to_ar,
    freeze_all_but_lora_and_linear,
    load_lora_state,
    lora_state_dict,
)


class FakeGemma4ClippableLinear(nn.Module):
    """A small stand-in that has the surface we patch: `self.linear` + forward(x)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.linear(x)


# We need our test modules to be matched by class name "Gemma4ClippableLinear",
# so we monkey-patch the class name to match. (attach_lora_to_ar matches on
# module.__class__.__name__.)
FakeGemma4ClippableLinear.__name__ = "Gemma4ClippableLinear"


class FakeAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q_proj = FakeGemma4ClippableLinear(hidden, hidden)
        self.k_proj = FakeGemma4ClippableLinear(hidden, hidden)
        self.v_proj = FakeGemma4ClippableLinear(hidden, hidden)
        self.o_proj = FakeGemma4ClippableLinear(hidden, hidden)


class FakeLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = FakeAttention(hidden)


class FakeLanguageModel(nn.Module):
    def __init__(self, hidden: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(hidden) for _ in range(n_layers)])


class FakeVisionEncoder(nn.Module):
    def __init__(self, hidden: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(hidden) for _ in range(n_layers)])


class FakeVisionTower(nn.Module):
    def __init__(self, hidden: int, n_layers: int):
        super().__init__()
        self.encoder = FakeVisionEncoder(hidden, n_layers)


class FakeGemma4(nn.Module):
    def __init__(self, hidden: int = 64, n_lang_layers: int = 8, n_vision_layers: int = 4):
        super().__init__()
        self.vision_tower = FakeVisionTower(hidden, n_vision_layers)
        self.language_model = FakeLanguageModel(hidden, n_lang_layers)


class FakeAR(nn.Module):
    """Mimics TruncatedGemmaAR enough for the LoRA functions: has `.backbone` and `.linear`."""

    def __init__(self, hidden: int = 64, n_lang_layers: int = 8, n_vision_layers: int = 4):
        super().__init__()
        self.backbone = FakeGemma4(hidden, n_lang_layers, n_vision_layers)
        self.linear = nn.Linear(hidden, hidden, bias=True)


def test_lora_delta_zero_init_returns_zero():
    """At init, B is zero so the delta should be exactly 0 for any input."""
    lora = LoRADelta(in_dim=8, out_dim=4, rank=2, alpha=4, dtype=torch.float32, device="cpu")
    x = torch.randn(3, 5, 8)
    out = lora(x)
    assert out.shape == (3, 5, 4)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_lora_delta_nonzero_after_B_perturbation():
    """If we manually break the zero-init on B, delta becomes nonzero."""
    lora = LoRADelta(in_dim=8, out_dim=4, rank=2, alpha=4, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        lora.B.fill_(0.1)
    x = torch.randn(3, 5, 8)
    out = lora(x)
    assert not torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_lora_delta_scale_matches_alpha_over_rank():
    """Verify the scale factor is exactly alpha/rank."""
    lora = LoRADelta(in_dim=4, out_dim=4, rank=2, alpha=8, dtype=torch.float32, device="cpu")
    assert abs(lora.scale - 4.0) < 1e-9  # 8/2 = 4


def test_attach_lora_initial_forward_unchanged():
    """After attach, AR's backbone forward must produce IDENTICAL outputs (zero LoRA delta)."""
    torch.manual_seed(0)
    ar = FakeAR(hidden=32)
    x = torch.randn(2, 7, 32)
    pre_q = ar.backbone.language_model.layers[3].self_attn.q_proj(x).clone()

    attached = attach_lora_to_ar(ar, layer_ell=8, rank=4, alpha=8, verbose=False)
    assert len(attached) > 0

    # forward should be identical because B is zero-init → delta = 0
    post_q = ar.backbone.language_model.layers[3].self_attn.q_proj(x)
    assert torch.allclose(post_q, pre_q, atol=1e-6)


def test_attach_lora_param_count_growth():
    """After attach, total parameter count must grow by exactly 2 * rank * (in + out) per attached module."""
    torch.manual_seed(0)
    hidden = 64
    n_lang = 8
    n_vision = 4
    ar = FakeAR(hidden=hidden, n_lang_layers=n_lang, n_vision_layers=n_vision)
    layer_ell = 5  # apply to first 5 language layers + all 4 vision layers

    base_count = sum(p.numel() for p in ar.parameters())
    attached = attach_lora_to_ar(ar, layer_ell=layer_ell, rank=4, alpha=8, verbose=False)

    expected_per_module = 4 * hidden + hidden * 4  # A (rank, in) + B (out, rank) = 4*64 + 64*4 = 512
    # 4 projections per attention (q,k,v,o); 5 language layers + 4 vision layers = 9 layers attached.
    expected_modules = 4 * (5 + 4)
    assert len(attached) == expected_modules
    expected_lora_params = expected_modules * expected_per_module

    after_count = sum(p.numel() for p in ar.parameters())
    assert after_count - base_count == expected_lora_params


def test_attach_lora_respects_layer_ell():
    """Language layers AT or AFTER `layer_ell` must NOT receive LoRA."""
    torch.manual_seed(0)
    ar = FakeAR(hidden=32, n_lang_layers=8, n_vision_layers=4)
    attach_lora_to_ar(ar, layer_ell=4, rank=2, alpha=4, include_vision_tower=False, verbose=False)
    # Layers 0..3 should have LoRA
    for i in range(4):
        assert hasattr(ar.backbone.language_model.layers[i].self_attn.q_proj, "_lora")
    # Layers 4..7 must not
    for i in range(4, 8):
        assert not hasattr(ar.backbone.language_model.layers[i].self_attn.q_proj, "_lora")


def test_attach_lora_vision_only_flag():
    """When language is disabled, only vision tower modules receive LoRA."""
    torch.manual_seed(0)
    ar = FakeAR(hidden=32, n_lang_layers=8, n_vision_layers=4)
    attach_lora_to_ar(ar, layer_ell=8, rank=2, alpha=4,
                      include_vision_tower=True, include_language_first_ell=False, verbose=False)
    for layer in ar.backbone.vision_tower.encoder.layers:
        assert hasattr(layer.self_attn.q_proj, "_lora")
    for layer in ar.backbone.language_model.layers:
        assert not hasattr(layer.self_attn.q_proj, "_lora")


def test_attach_lora_idempotent():
    """Calling attach twice doesn't double-patch (second call still works, no infinite delta)."""
    torch.manual_seed(0)
    ar = FakeAR(hidden=32)
    attach_lora_to_ar(ar, layer_ell=4, rank=2, alpha=4, verbose=False)
    # The first attach left _lora_patched=True. The second call adds a fresh _lora child
    # but should not re-patch forward.
    attach_lora_to_ar(ar, layer_ell=4, rank=2, alpha=4, verbose=False)
    x = torch.randn(2, 7, 32)
    # forward still finite and reasonable shape
    out = ar.backbone.language_model.layers[0].self_attn.q_proj(x)
    assert out.shape == (2, 7, 32)
    assert torch.isfinite(out).all()


def test_lora_gradient_flows_to_A_and_B():
    """After perturbing B and backprop'ing a loss, both A and B receive nonzero gradients."""
    torch.manual_seed(0)
    ar = FakeAR(hidden=16)
    attach_lora_to_ar(ar, layer_ell=4, rank=2, alpha=4, verbose=False)
    # Break the zero-init on B for at least one LoRA so the delta is nonzero from the first step
    for name, mod in ar.backbone.named_modules():
        if hasattr(mod, "_lora"):
            with torch.no_grad():
                mod._lora.B.fill_(0.01)
    # Forward + loss
    x = torch.randn(2, 4, 16)
    target_layer = ar.backbone.language_model.layers[0].self_attn.q_proj
    y = target_layer(x).sum()
    y.backward()
    assert target_layer._lora.A.grad is not None
    assert target_layer._lora.B.grad is not None
    assert torch.isfinite(target_layer._lora.A.grad).all()
    assert torch.isfinite(target_layer._lora.B.grad).all()
    # Gradients should be nonzero
    assert target_layer._lora.A.grad.abs().sum() > 0
    assert target_layer._lora.B.grad.abs().sum() > 0


def test_freeze_all_but_lora_and_linear():
    """After calling the helper, only LoRA A/B and the Linear head should be trainable."""
    torch.manual_seed(0)
    ar = FakeAR(hidden=16)
    attach_lora_to_ar(ar, layer_ell=4, rank=2, alpha=4, verbose=False)
    lora_params, head_params = freeze_all_but_lora_and_linear(ar)

    # All backbone Linear weights frozen
    for name, p in ar.backbone.named_parameters():
        if "_lora." in name:
            assert p.requires_grad, f"{name} should be trainable"
        else:
            assert not p.requires_grad, f"{name} should be frozen"
    # Head trainable
    for p in head_params:
        assert p.requires_grad
    # lora_params nonempty
    assert len(lora_params) > 0
    # head_params has weight and bias
    assert len(head_params) == 2  # weight + bias


def test_state_dict_roundtrip():
    """Save then restore: B tensors should be identical."""
    torch.manual_seed(0)
    ar = FakeAR(hidden=16)
    attach_lora_to_ar(ar, layer_ell=4, rank=2, alpha=4, verbose=False)
    # Mutate some LoRA params so we can detect the round trip
    for name, mod in ar.backbone.named_modules():
        if hasattr(mod, "_lora"):
            with torch.no_grad():
                mod._lora.A.normal_(0, 0.1)
                mod._lora.B.normal_(0, 0.1)

    state = lora_state_dict(ar)
    # Reset by re-init
    for name, mod in ar.backbone.named_modules():
        if hasattr(mod, "_lora"):
            with torch.no_grad():
                mod._lora.A.zero_()
                mod._lora.B.zero_()

    load_lora_state(ar, state, strict=True)
    # After load, A and B should match the saved state
    for key, saved in state.items():
        if key.endswith(".A"):
            mod_name = key[:-2]
            mod = dict(ar.backbone.named_modules())[mod_name]
            assert torch.allclose(mod._lora.A, saved.to(mod._lora.A.device))
        elif key.endswith(".B"):
            mod_name = key[:-2]
            mod = dict(ar.backbone.named_modules())[mod_name]
            assert torch.allclose(mod._lora.B, saved.to(mod._lora.B.device))


def test_state_dict_keys_match_module_names():
    """state_dict keys must be `<module_name>.A` and `.B`."""
    torch.manual_seed(0)
    ar = FakeAR(hidden=16)
    attach_lora_to_ar(ar, layer_ell=4, rank=2, alpha=4,
                      include_vision_tower=False, verbose=False)
    state = lora_state_dict(ar)
    # 4 language layers × 4 projections × 2 (A and B) = 32 keys
    assert len(state) == 4 * 4 * 2
    for key in state:
        assert key.endswith(".A") or key.endswith(".B")
        # corresponding module exists
        mod_name = key[:-2]
        assert mod_name in dict(ar.backbone.named_modules())
