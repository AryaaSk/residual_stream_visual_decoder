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

# Default injection scale (alpha). Tuned via alpha_sweep on Stage-1 AV at L16:
# at alpha=0.5 the malformation rate dropped from 59% (alpha=1.0) to 20%, and
# drawings became visibly richer and more structured. Activation norm at L16 is
# ~70; alpha=0.5 → injected magnitude ~35, much closer to Gemma 4's typical
# embedding norm (~10-20). See findings/alpha_sweep/.
DEFAULT_ALPHA = 0.5


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
        alpha: float = DEFAULT_ALPHA,
        max_new_tokens: int = 600,
        temperature: float = 1.0,
        top_k: int = 0,
        constrain_to_stroke_vocab: bool = True,
    ) -> torch.Tensor:
        """Inject the activation via embedding-layer forward hook, then sample stroke tokens.

        Gemma 4 doesn't allow passing both input_ids and inputs_embeds, and it
        verifies that any inputs_embeds you pass matches the embedding table (so
        it can recover input_ids). Our trick: pass input_ids normally, and use a
        forward hook on the embedding layer to OVERWRITE the embedding row at
        the <ACT_TOKEN> position with alpha * activation.

        Returns a 1-D LongTensor of token ids (not including the prompt).
        """
        device = self.device()
        if activation.ndim != 1:
            raise ValueError(f"activation must be 1-D, got {tuple(activation.shape)}")

        # Build prompt as input_ids (no embedding surgery here)
        act_token_id = self.vocab.name_to_id[ACT_TOKEN]
        text = INJECT_PROMPT_TEMPLATE.format(layer=layer_ell, act=ACT_TOKEN)
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        input_ids = enc["input_ids"]
        pos_mask = input_ids[0] == act_token_id
        if not pos_mask.any():
            raise RuntimeError("<ACT_TOKEN> not found in tokenised prompt")
        act_position = int(pos_mask.nonzero(as_tuple=False).item())

        # Register a forward hook on the embedding layer that overwrites the
        # row at act_position on the FIRST forward pass (when sequence length
        # includes act_position). On subsequent (single-token) passes the hook
        # is a no-op.
        embed_layer = self.model.get_input_embeddings()
        scaled_activation = (activation.to(device=device, dtype=embed_layer.weight.dtype) * alpha)
        first_pass_done = {"done": False}

        def embed_hook(module, inputs, output):
            seq_len = output.shape[1]
            if not first_pass_done["done"] and seq_len > act_position:
                output = output.clone()
                output[0, act_position, :] = scaled_activation
                first_pass_done["done"] = True
            return output

        hook_handle = embed_layer.register_forward_hook(embed_hook)

        draw_close_id = self.vocab.name_to_id[DRAW_CLOSE]
        allowed = set(stroke_token_ids(self.vocab)) if constrain_to_stroke_vocab else None
        allowed_ids = torch.tensor(list(allowed), device=device) if allowed is not None else None

        past_key_values = None
        cur_input_ids = input_ids
        generated: list[int] = []

        try:
            for step in range(max_new_tokens):
                if past_key_values is None:
                    out = self.model(input_ids=cur_input_ids, use_cache=True)
                else:
                    out = self.model(
                        input_ids=cur_input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                logits = out.logits[:, -1, :]
                past_key_values = out.past_key_values

                if allowed is not None:
                    mask = torch.full_like(logits, float("-inf"))
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

                cur_input_ids = torch.tensor([[next_id]], device=device)
        finally:
            hook_handle.remove()

        return torch.tensor(generated, dtype=torch.long)


def build_target_stroke_ids(vocab: StrokeVocab, stroke_dicts: list[dict]) -> list[int]:
    """Convert a list of {dx,dy,pen} dicts (from QuickDraw loader) into a token-id sequence
    wrapped in <DRAW>...</DRAW>.
    """
    from stroke_tokenizer import Stroke
    strokes = [Stroke(dx=s["dx"], dy=s["dy"], pen=s["pen"]) for s in stroke_dicts]
    return vocab.encode_drawing(strokes)
