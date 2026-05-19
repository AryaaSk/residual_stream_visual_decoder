"""Activation injection: replace the embedding of <ACT_TOKEN> with a target activation.

This is the surgical core of the NLA pattern. Given a Gemma 4 model with
the extended vocabulary, a tokenised prompt containing <ACT_TOKEN>, and an
activation tensor `h_ℓ`, we override the input-embeddings tensor so that
the position of <ACT_TOKEN> carries `alpha * h_ℓ` instead of the learned
embedding row.

Usage
-----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from av.vocab_extend import extend_model_and_tokenizer
    from av.activation_injection import build_prompt_with_activation

    model = AutoModelForCausalLM.from_pretrained(...)
    tok = AutoTokenizer.from_pretrained(...)
    vocab = extend_model_and_tokenizer(model, tok)

    h = ...  # an activation, shape (hidden_size,) in the same dtype as model
    input_ids, inputs_embeds = build_prompt_with_activation(
        model, tok, vocab, layer_ell=16, activation=h, alpha=1.0,
    )
    # Generate strokes constrained to the stroke-token subset:
    out = model.generate(inputs_embeds=inputs_embeds, ...)
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class PromptParts(NamedTuple):
    input_ids: torch.Tensor          # (1, T) on the model's device
    inputs_embeds: torch.Tensor      # (1, T, hidden) on the model's device
    act_position: int                # the index of <ACT_TOKEN> inside the prompt
    layer_ell: int                   # the layer the activation came from (just for bookkeeping)


def build_prompt_with_activation(
    model,
    tokenizer,
    vocab,           # StrokeVocab
    layer_ell: int,
    activation: torch.Tensor,
    *,
    prompt_template: str = "Visualize the following thought from layer {layer}: {act} <DRAW>",
    alpha: float = 1.0,
    device: str | torch.device | None = None,
) -> PromptParts:
    """Tokenise the prompt and surgically inject `alpha * activation` at <ACT_TOKEN>.

    The prompt template MUST contain literal "{act}" where the activation token
    goes. The "{layer}" placeholder is filled with the layer index (useful for
    layer-conditioned variants; for per-layer models it's still emitted so the
    prompt is identical at inference and training).
    """
    if device is None:
        device = next(model.parameters()).device

    if activation.ndim != 1:
        raise ValueError(f"activation must be 1-D (hidden_size,), got shape {tuple(activation.shape)}")

    act_token_str = "<ACT_TOKEN>"
    if "{act}" not in prompt_template:
        raise ValueError("prompt_template must contain literal '{act}' placeholder")

    text = prompt_template.format(layer=layer_ell, act=act_token_str)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
    input_ids: torch.Tensor = enc["input_ids"]
    if input_ids.shape[0] != 1:
        raise RuntimeError(f"expected batch size 1, got {input_ids.shape[0]}")

    act_token_id = vocab.name_to_id[act_token_str]
    # find position of <ACT_TOKEN> in the prompt
    pos_mask = input_ids[0] == act_token_id
    if not pos_mask.any():
        raise RuntimeError("<ACT_TOKEN> not found after tokenisation. Check tokenizer config.")
    act_position = int(pos_mask.nonzero(as_tuple=False).item())

    # Build inputs_embeds and overwrite that position
    inputs_embeds: torch.Tensor = model.get_input_embeddings()(input_ids)
    target_dtype = inputs_embeds.dtype
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[0, act_position, :] = activation.to(device=device, dtype=target_dtype) * alpha

    return PromptParts(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        act_position=act_position,
        layer_ell=layer_ell,
    )


def stroke_token_ids(vocab) -> list[int]:
    """All stroke-token IDs (the 262 we added). Used to constrain generation
    to only emit stroke tokens between <DRAW> and </DRAW>."""
    return list(vocab.id_to_name.keys())
