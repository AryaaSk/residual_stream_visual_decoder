"""Activation Reconstructor (AR): truncated Gemma 4 + Linear(d, d) + optional backbone LoRA.

Reads a rendered PNG of strokes through Gemma 4's existing vision encoder
and the first ℓ transformer layers, then projects through a learned
Linear(d, d) head to produce a reconstructed activation in the same
coordinate frame as the target residual at layer ℓ.

The "truncation" is implemented by EARLY-EXITING the forward pass after
layer ℓ rather than physically deleting layers. This makes it trivial to
share a single base model across multiple AR instances (one per ℓ) at
training time and at inference time.

v1.0 addition: optional backbone LoRA on attention projections. When
``use_backbone_lora=True`` is passed to ``from_pretrained``, we attach
parallel ``LoRADelta`` branches to every ``Gemma4ClippableLinear`` matching
q/k/v/o_proj in the vision tower and in the first ``layer_ell`` language
layers. See ``code/ar/lora_gemma4.py``.

Gradient policy: the backbone's BASE weights are always frozen
(``param.requires_grad = False``); only LoRA A/B parameters and the final
``Linear(d, d)`` head receive gradients. ``forward()`` no longer wraps the
backbone in ``torch.no_grad()``: gradients flow back through the LoRA
branches when LoRA is attached, otherwise the frozen-param graph is small
enough that the overhead is negligible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


class TruncatedGemmaAR(nn.Module):
    """Wraps a vision-language Gemma 4 model and a `Linear(d, d)` reconstruction head.

    For a forward pass:
        1. encode the image via Gemma's vision encoder (with LoRA if enabled)
        2. wrap with a fixed text prompt
        3. run through the first ℓ transformer layers (with LoRA if enabled)
        4. take the activation at the LAST position
        5. apply Linear(d, d) to get ĥ

    Multiple ARs can share the same backbone; we don't physically truncate.
    """

    PROMPT_PREFIX = "This drawing depicts the model's thought:"
    PROMPT_SUFFIX = "."

    def __init__(
        self,
        backbone,                # PreTrainedModel: Gemma4 multimodal
        processor,
        layer_ell: int,
        hidden_size: int,
        linear_init_std: float = 0.02,
    ):
        super().__init__()
        self.backbone = backbone
        self.processor = processor
        self.layer_ell = layer_ell
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        with torch.no_grad():
            self.linear.weight.normal_(mean=0.0, std=linear_init_std)
            self.linear.bias.zero_()
        self.has_backbone_lora = False  # set by from_pretrained when LoRA attached

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        layer_ell: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_backbone_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        include_vision_tower: bool = True,
    ) -> "TruncatedGemmaAR":
        from transformers import AutoModelForCausalLM, AutoProcessor
        backbone = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="sdpa"
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        hidden = backbone.config.text_config.hidden_size
        ar = cls(backbone, processor, layer_ell=layer_ell, hidden_size=hidden)
        ar.linear.to(device=device, dtype=dtype)
        if use_backbone_lora:
            # Freeze the backbone before attaching LoRA, then unfreeze only LoRA params.
            for p in ar.backbone.parameters():
                p.requires_grad = False
            import sys
            from pathlib import Path as _P
            sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
            from ar.lora_gemma4 import attach_lora_to_ar
            attach_lora_to_ar(
                ar, layer_ell=layer_ell, rank=lora_rank, alpha=lora_alpha,
                include_vision_tower=include_vision_tower,
                include_language_first_ell=True, verbose=True,
            )
            ar.has_backbone_lora = True
        return ar

    def _process_image(self, image) -> dict:
        """Build the multimodal input dict (matches Day-0 alignment script)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What does this drawing depict?"},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        return inputs

    def forward(self, images, return_aux: bool = False) -> torch.Tensor | tuple:
        """Run a batch of PIL images through the AR. Returns reconstructed activations.

        images: list of PIL Images (length B)
        """
        device = next(self.parameters()).device
        outs: list[torch.Tensor] = []
        last_positions: list[int] = []
        # Gradients must flow when LoRA is attached and we're in training mode.
        grad_enabled = self.training and (self.has_backbone_lora or any(p.requires_grad for p in self.linear.parameters()))
        for img in images:
            inputs = self._process_image(img)
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            with torch.set_grad_enabled(grad_enabled):
                out = self.backbone(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
            h = out.hidden_states[self.layer_ell][0, -1, :]
            outs.append(h)
            last_positions.append(int(inputs["input_ids"].shape[1] - 1))

        stacked = torch.stack(outs, dim=0)  # (B, hidden)
        h_hat = self.linear(stacked)
        if return_aux:
            return h_hat, {"last_positions": last_positions}
        return h_hat

    def loss(self, images, h_target: torch.Tensor) -> torch.Tensor:
        """MSE between reconstructed and target activations."""
        h_hat = self.forward(images)
        return torch.nn.functional.mse_loss(h_hat.float(), h_target.float())
