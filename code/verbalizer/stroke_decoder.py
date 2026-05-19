"""Activation Verbalizer (AV) wrapper.

Wraps a vocab-extended Gemma 4 E2B model and exposes the two operations we need:

    sft_loss(prompt, target_stroke_ids)
        Cross-entropy loss for Stage-1 SFT (text → strokes, no activation injection).

    generate_from_activation(activation, layer_ell, **gen_kwargs)
        Stage-3/eval inference: inject activation at <ACT_TOKEN>, sample stroke tokens.

Light LoRA can be applied externally via peft; this module is the pure
forward/sample interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.activation_injection import build_prompt_with_activation, stroke_token_ids  # noqa: E402
from stroke_tokenizer import ACT_TOKEN, DRAW_CLOSE, DRAW_OPEN, PEN_END, StrokeVocab  # noqa: E402


SFT_PROMPT_TEMPLATE = "Draw: {caption} <DRAW>"
INJECT_PROMPT_TEMPLATE = "Visualize the following thought from layer {layer}: {act} <DRAW>"


@dataclass
class StrokeDecoder:
    """Container holding the model, tokenizer, and vocab.

    Use ``StrokeDecoder.from_pretrained_and_extend(...)`` to set everything up.
    """
    model: object  # PreTrainedModel
    tokenizer: object
    vocab: StrokeVocab

    @classmethod
    def from_pretrained_and_extend(
        cls,
        model_id: str = "google/gemma-4-e2b-it",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "StrokeDecoder":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from verbalizer.vocab_extend import extend_model_and_tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="sdpa").to(device)
        tok = AutoTokenizer.from_pretrained(model_id)
        vocab = extend_model_and_tokenizer(model, tok)
        return cls(model=model, tokenizer=tok, vocab=vocab)

    @classmethod
    def from_ckpt(
        cls,
        av_ckpt_dir,
        model_id: str = "google/gemma-4-e2b-it",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "StrokeDecoder":
        """Load Stage-1 checkpoint: backbone fresh, only the new-vocab embedding rows are loaded.

        The Stage-1 SFT script saves only the new embedding rows + the vocab name->id
        mapping in `av_ckpt.pt`. We rebuild the full model and overwrite those rows.
        """
        from pathlib import Path as _Path
        instance = cls.from_pretrained_and_extend(model_id, device=device, dtype=dtype)
        ckpt_file = _Path(av_ckpt_dir) / "av_ckpt.pt"
        ckpt = torch.load(ckpt_file, weights_only=False)
        instance.vocab = StrokeVocab.from_name_to_id(ckpt["vocab_name_to_id"])
        old_vocab = int(ckpt["old_vocab_size"])
        new_rows = ckpt["new_embed_rows"]
        embed = instance.model.get_input_embeddings()
        with torch.no_grad():
            embed.weight.data[old_vocab : old_vocab + new_rows.shape[0]] = new_rows.to(
                device=embed.weight.device, dtype=embed.weight.dtype
            )
        return instance

    def device(self):
        return next(self.model.parameters()).device

    # ------- Stage 1: text → strokes SFT -------

    def sft_loss(self, caption: str, target_stroke_ids: Sequence[int]) -> torch.Tensor:
        """Cross-entropy loss for a single (caption, strokes) example.

        Masks loss over the prompt portion (we don't backprop CE on the prompt tokens).
        """
        device = self.device()
        prompt = SFT_PROMPT_TEMPLATE.format(caption=caption)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)
        full_ids = prompt_ids + list(target_stroke_ids)
        input_ids = torch.tensor([full_ids], device=device)
        labels = input_ids.clone()
        # mask prompt tokens
        labels[0, :prompt_len] = -100
        out = self.model(input_ids=input_ids, labels=labels, use_cache=False)
        return out.loss

    def sft_loss_batched(self, captions: list[str], target_stroke_ids_list: list[list[int]], pad_id: int | None = None) -> torch.Tensor:
        """Batched SFT loss with proper masking for variable-length stroke targets."""
        device = self.device()
        if pad_id is None:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # Tokenise prompts to get their lengths
        prompts = [SFT_PROMPT_TEMPLATE.format(caption=c) for c in captions]
        prompt_id_lists = [self.tokenizer(p, add_special_tokens=True)["input_ids"] for p in prompts]

        full_seqs = [p + t for p, t in zip(prompt_id_lists, target_stroke_ids_list)]
        max_len = max(len(s) for s in full_seqs)

        input_ids = torch.full((len(full_seqs), max_len), pad_id, dtype=torch.long, device=device)
        labels = torch.full((len(full_seqs), max_len), -100, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(full_seqs), max_len), dtype=torch.long, device=device)

        for i, (p, t) in enumerate(zip(prompt_id_lists, target_stroke_ids_list)):
            seq = p + t
            input_ids[i, :len(seq)] = torch.tensor(seq, device=device)
            attention_mask[i, :len(seq)] = 1
            # only train on the stroke tokens
            labels[i, len(p):len(seq)] = torch.tensor(t, device=device)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        return out.loss

    # ------- Stage 3 / inference: activation → strokes -------

    @torch.no_grad()
    def generate_from_activation(
        self,
        activation: torch.Tensor,
        layer_ell: int,
        *,
        alpha: float = 1.0,
        max_new_tokens: int = 600,
        temperature: float = 1.0,
        top_k: int = 0,
        constrain_to_stroke_vocab: bool = True,
    ) -> torch.Tensor:
        """Inject the activation, autoregressively sample stroke tokens.

        Returns a 1-D LongTensor of token ids (not including the prompt).
        Generation stops on </DRAW> or after `max_new_tokens`.
        """
        parts = build_prompt_with_activation(
            self.model, self.tokenizer, self.vocab,
            layer_ell=layer_ell, activation=activation, alpha=alpha,
            prompt_template=INJECT_PROMPT_TEMPLATE,
        )
        # We sample directly rather than using model.generate(), so we have full
        # control over how the activation-injected embeds are fed and how we
        # constrain to the stroke vocabulary.
        draw_close_id = self.vocab.name_to_id[DRAW_CLOSE]
        pen_end_id = self.vocab.name_to_id[self.vocab.id_to_name[self.vocab.name_to_id[DRAW_OPEN]]]  # noqa: unused — readability
        allowed = set(stroke_token_ids(self.vocab)) if constrain_to_stroke_vocab else None

        device = self.device()
        inputs_embeds = parts.inputs_embeds
        input_ids = parts.input_ids
        past_key_values = None
        generated: list[int] = []

        for step in range(max_new_tokens):
            if past_key_values is None:
                out = self.model(inputs_embeds=inputs_embeds, use_cache=True)
            else:
                # feed only the latest token's embed
                last_id = generated[-1]
                last_embed = self.model.get_input_embeddings()(
                    torch.tensor([[last_id]], device=device)
                )
                out = self.model(inputs_embeds=last_embed, past_key_values=past_key_values, use_cache=True)

            logits = out.logits[:, -1, :]  # (1, vocab)
            past_key_values = out.past_key_values

            if allowed is not None:
                mask = torch.full_like(logits, float("-inf"))
                allowed_ids = torch.tensor(list(allowed), device=device)
                mask[0, allowed_ids] = 0.0
                logits = logits + mask

            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                kth_value = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = torch.where(logits < kth_value, torch.full_like(logits, float("-inf")), logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_id)

            if next_id == draw_close_id:
                break

        return torch.tensor(generated, dtype=torch.long)


def build_target_stroke_ids(vocab: StrokeVocab, stroke_dicts: list[dict]) -> list[int]:
    """Convert a list of {dx,dy,pen} dicts (from QuickDraw loader) into a token-id sequence
    wrapped in <DRAW>...</DRAW>.
    """
    from stroke_tokenizer import Stroke
    strokes = [Stroke(dx=s["dx"], dy=s["dy"], pen=s["pen"]) for s in stroke_dicts]
    return vocab.encode_drawing(strokes)
