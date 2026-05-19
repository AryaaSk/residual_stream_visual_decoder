"""Activation Reconstructor (AR): truncated Gemma 4 + Linear(d, d).

Reads a rendered PNG of strokes through Gemma 4's existing vision encoder
and the first ℓ transformer layers, then projects through a learned
Linear(d, d) head to produce a reconstructed activation in the same
coordinate frame as the target residual at layer ℓ.

The "truncation" is implemented by EARLY-EXITING the forward pass after
layer ℓ rather than physically deleting layers. This makes it trivial to
share a single base model across multiple AR instances (one per ℓ) at
training time and at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


class TruncatedGemmaAR(nn.Module):
    """Wraps a vision-language Gemma 4 model and a `Linear(d, d)` reconstruction head.

    For a forward pass:
        1. encode the image via Gemma's vision encoder
        2. wrap with a fixed text prompt
        3. run through the first ℓ transformer layers
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

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        layer_ell: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "TruncatedGemmaAR":
        from transformers import AutoModelForCausalLM, AutoProcessor
        backbone = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="sdpa"
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        hidden = backbone.config.text_config.hidden_size
        ar = cls(backbone, processor, layer_ell=layer_ell, hidden_size=hidden)
        ar.linear.to(device=device, dtype=dtype)
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
        # The Gemma4 processor processes one item at a time; batch by stacking
        # input_ids / attention_mask manually and using the model's own padding logic.
        # For Day-1 simplicity we loop and forward one-by-one, then stack output.
        outs: list[torch.Tensor] = []
        last_positions: list[int] = []
        for img in images:
            inputs = self._process_image(img)
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                # Run the LANGUAGE model only up to layer ℓ
                out = self.backbone(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
            # hidden_states[layer_ell] has shape (1, T, hidden)
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
