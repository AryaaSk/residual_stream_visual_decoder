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
from verbalizer.projector import ActProjector  # noqa: E402
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
    """Container holding the model, tokenizer, vocab, and optional activation projector.

    Use ``StrokeDecoder.from_pretrained_and_extend(...)`` to set everything up.

    ``act_projector`` (optional ``ActProjector``): if present, projects the injected
    activation through a learnable Linear before overwriting the <ACT_TOKEN>
    embedding row. Init as α·I so v1.1 behaviour is recovered when untrained.
    Set ``act_projector=None`` to fall back to raw α·h injection (v1.1 behaviour).
    """
    model: object  # PreTrainedModel
    tokenizer: object
    vocab: StrokeVocab
    act_projector: ActProjector | None = None

    @classmethod
    def from_pretrained_and_extend(
        cls,
        model_id: str = "google/gemma-4-e2b-it",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_projector: bool = False,
        projector_alpha_init: float = 0.5,
    ) -> "StrokeDecoder":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from verbalizer.vocab_extend import extend_model_and_tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="sdpa").to(device)
        tok = AutoTokenizer.from_pretrained(model_id)
        vocab = extend_model_and_tokenizer(model, tok)
        d = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
        proj = None
        if use_projector:
            proj = ActProjector(d=d, alpha_init=projector_alpha_init, dtype=dtype, device=device)
        return cls(model=model, tokenizer=tok, vocab=vocab, act_projector=proj)

    @classmethod
    def from_ckpt(
        cls,
        av_ckpt_dir,
        model_id: str = "google/gemma-4-e2b-it",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "StrokeDecoder":
        """Load AV checkpoint (Stage 1, Stage 1.5, or Stage 4 format).

        Backward-compatible with Stage 1 ckpts that only contain new-vocab rows
        plus the vocab mapping. Stage 1.5+ checkpoints may also include:
          - ``projector_state``: state dict for an ActProjector (loaded → ``self.act_projector``)
          - ``lora_state``: AV LoRA A/B tensors (requires LoRA to have been attached;
            call ``attach_lora_to_av`` before loading if your ckpt has this)
        """
        from pathlib import Path as _Path
        ckpt_file = _Path(av_ckpt_dir) / "av_ckpt.pt"
        ckpt = torch.load(ckpt_file, weights_only=False)
        has_proj = "projector_state" in ckpt
        instance = cls.from_pretrained_and_extend(
            model_id, device=device, dtype=dtype,
            use_projector=has_proj,
        )
        instance.vocab = StrokeVocab.from_name_to_id(ckpt["vocab_name_to_id"])
        old_vocab = int(ckpt["old_vocab_size"])
        new_rows = ckpt["new_embed_rows"]
        embed = instance.model.get_input_embeddings()
        with torch.no_grad():
            embed.weight.data[old_vocab : old_vocab + new_rows.shape[0]] = new_rows.to(
                device=embed.weight.device, dtype=embed.weight.dtype
            )
        if has_proj:
            instance.act_projector = ActProjector.from_state(
                ckpt["projector_state"], dtype=dtype, device=device,
            )
        if "lora_state" in ckpt:
            from ar.lora_gemma4 import attach_lora_to_av, load_lora_state
            lora_meta = ckpt.get("lora_meta", {"first_n_layers": 8, "rank": 16, "alpha": 32})
            # Infer ACTUAL number of LoRA layers from the saved state dict:
            # if state dict only covers layers 0-7, attach only on those layers
            # even if lora_meta claims more (this handles the
            # skip-attach-on-resume case where meta and actual diverge).
            import re
            layer_idxs = set()
            for k in ckpt["lora_state"].keys():
                m = re.search(r"\.layers\.(\d+)\.", k)
                if m:
                    layer_idxs.add(int(m.group(1)))
            actual_layers = max(layer_idxs) + 1 if layer_idxs else int(lora_meta.get("first_n_layers", 8))
            attach_lora_to_av(
                instance,
                first_n_layers=actual_layers,
                rank=int(lora_meta.get("rank", 16)),
                alpha=int(lora_meta.get("alpha", 32)),
                verbose=False,
            )
            load_lora_state(instance, ckpt["lora_state"], strict=False)
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

    # ------- Stage 1.5: activation-conditioned SFT (supervised) -------

    def act_sft_loss_batched(
        self,
        activations: torch.Tensor,                # (B, d) — caller pre-extracts h_ℓ
        target_stroke_ids_list: list[list[int]],  # per-example drawing token ids
        layer_ell: int,
        pad_id: int | None = None,
    ) -> torch.Tensor:
        """Activation-conditioned teacher-forced cross-entropy.

        Builds prompt `"Visualize the following thought from layer ℓ: <ACT_TOKEN> <DRAW>" + drawing_tokens`
        for each example, hooks the embedding layer to overwrite the <ACT_TOKEN>
        row with `projector(h)` (or `alpha·h` if no projector), and computes CE
        loss on drawing tokens only (prompt masked with -100).

        Unlike `generate_from_activation`, this runs WITH GRAD so the projector,
        AV LoRA, and new-vocab rows can be optimised. Caller is responsible
        for ensuring requires_grad is set correctly.
        """
        device = self.device()
        if pad_id is None:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        B, d = activations.shape
        if B != len(target_stroke_ids_list):
            raise ValueError(f"activations B={B} != targets B={len(target_stroke_ids_list)}")

        act_token_id = self.vocab.name_to_id[ACT_TOKEN]
        prompt_text = INJECT_PROMPT_TEMPLATE.format(layer=layer_ell, act=ACT_TOKEN)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        # Find act position in the prompt (same for all examples since prompt is fixed)
        act_pos = next(i for i, tid in enumerate(prompt_ids) if tid == act_token_id)
        prompt_len = len(prompt_ids)

        full_seqs = [prompt_ids + t for t in target_stroke_ids_list]
        max_len = max(len(s) for s in full_seqs)

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for i, (p, t) in enumerate(zip([prompt_ids] * B, target_stroke_ids_list)):
            seq = p + t
            input_ids[i, :len(seq)] = torch.tensor(seq, device=device)
            attention_mask[i, :len(seq)] = 1
            labels[i, prompt_len:len(seq)] = torch.tensor(t, device=device)

        # Project (or scale) all activations
        embed_layer = self.model.get_input_embeddings()
        h_dev = activations.to(device=device, dtype=embed_layer.weight.dtype)
        if self.act_projector is not None:
            injected = self.act_projector(h_dev)  # (B, d) — grad flows
        else:
            injected = h_dev * DEFAULT_ALPHA

        # Forward-hook: overwrite act_pos column for every batch row, first pass only.
        first_pass_done = {"done": False}

        def embed_hook(module, inputs, output):
            seq_len = output.shape[1]
            if not first_pass_done["done"] and seq_len > act_pos:
                output = output.clone()
                output[:, act_pos, :] = injected
                first_pass_done["done"] = True
            return output

        hook_handle = embed_layer.register_forward_hook(embed_hook)
        try:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
        finally:
            hook_handle.remove()
        return out.loss

    # ------- Stage 1.5+: save/load with projector + LoRA -------

    def save_ckpt(self, out_dir, *, include_lora_meta: dict | None = None) -> None:
        """Save AV state to `<out_dir>/av_ckpt.pt`.

        Always includes: new_embed_rows, old_vocab_size, vocab_name_to_id.
        Includes if present: projector_state, lora_state.
        """
        from pathlib import Path as _Path
        out_dir = _Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        embed = self.model.get_input_embeddings()
        old_vocab = embed.weight.shape[0] - len(self.vocab.name_to_id)
        ckpt = {
            "new_embed_rows": embed.weight.detach()[old_vocab:].cpu().clone(),
            "old_vocab_size": old_vocab,
            "vocab_name_to_id": self.vocab.name_to_id,
        }
        if self.act_projector is not None:
            ckpt["projector_state"] = self.act_projector.state()
        # LoRA state (if any LoRA modules are attached to the model)
        from ar.lora_gemma4 import lora_state_dict
        ls = lora_state_dict(self)
        if ls:
            ckpt["lora_state"] = ls
            if include_lora_meta is not None:
                ckpt["lora_meta"] = include_lora_meta
        torch.save(ckpt, out_dir / "av_ckpt.pt")

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
        #
        # When self.act_projector is set (Stage 1.5+), the projector handles
        # both the scale and any learned activation→embedding mapping; alpha is
        # then ignored. When act_projector is None (v1.1 behaviour), inject
        # alpha·h directly.
        embed_layer = self.model.get_input_embeddings()
        h_dev = activation.to(device=device, dtype=embed_layer.weight.dtype)
        if self.act_projector is not None:
            injected = self.act_projector(h_dev)
        else:
            injected = h_dev * alpha
        first_pass_done = {"done": False}

        def embed_hook(module, inputs, output):
            seq_len = output.shape[1]
            if not first_pass_done["done"] and seq_len > act_position:
                output = output.clone()
                output[0, act_position, :] = injected
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


    @torch.no_grad()
    def generate_from_activation_batched(
        self,
        activation: torch.Tensor,
        layer_ell: int,
        *,
        n_samples: int = 8,
        alpha: float = DEFAULT_ALPHA,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_k: int = 20,
        constrain_to_stroke_vocab: bool = True,
    ) -> list[torch.Tensor]:
        """Batched activation-conditioned sampling: produce N samples in parallel.

        Same activation injected into all N batch rows. KV-cache is shared so
        the prefill cost (~24 layers of attention over the prompt) is paid
        ONCE; only the per-token sampling step is N-wide. On H200 with N=16,
        this is ~10x faster than calling generate_from_activation N times.

        Returns a list of N LongTensor token sequences (each truncated at
        DRAW_CLOSE; lengths vary).
        """
        device = self.device()
        if activation.ndim != 1:
            raise ValueError(f"activation must be 1-D, got {tuple(activation.shape)}")

        act_token_id = self.vocab.name_to_id[ACT_TOKEN]
        text = INJECT_PROMPT_TEMPLATE.format(layer=layer_ell, act=ACT_TOKEN)
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        input_ids = enc["input_ids"]  # (1, prompt_len)
        pos_mask = input_ids[0] == act_token_id
        if not pos_mask.any():
            raise RuntimeError("<ACT_TOKEN> not found in tokenised prompt")
        act_position = int(pos_mask.nonzero(as_tuple=False).item())

        # Expand prompt to batch dim
        input_ids = input_ids.expand(n_samples, -1).contiguous()

        embed_layer = self.model.get_input_embeddings()
        h_dev = activation.to(device=device, dtype=embed_layer.weight.dtype)
        if self.act_projector is not None:
            injected = self.act_projector(h_dev)
        else:
            injected = h_dev * alpha
        first_pass_done = {"done": False}

        def embed_hook(module, inputs, output):
            seq_len = output.shape[1]
            if not first_pass_done["done"] and seq_len > act_position:
                output = output.clone()
                output[:, act_position, :] = injected
                first_pass_done["done"] = True
            return output

        hook_handle = embed_layer.register_forward_hook(embed_hook)

        draw_close_id = self.vocab.name_to_id[DRAW_CLOSE]
        allowed = sorted(set(stroke_token_ids(self.vocab))) if constrain_to_stroke_vocab else None
        allowed_ids = torch.tensor(allowed, device=device) if allowed is not None else None

        past_key_values = None
        cur_input_ids = input_ids
        active = torch.ones(n_samples, dtype=torch.bool, device=device)
        # Pre-allocate token buffer per sample
        out_tokens: list[list[int]] = [[] for _ in range(n_samples)]

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
                logits = out.logits[:, -1, :]  # (N, V)
                past_key_values = out.past_key_values

                if allowed_ids is not None:
                    mask = torch.full_like(logits, float("-inf"))
                    mask[:, allowed_ids] = 0.0
                    logits = logits + mask
                if temperature != 1.0:
                    logits = logits / max(temperature, 1e-5)
                if top_k > 0:
                    kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
                probs = torch.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (N,)

                # Record tokens only for active samples
                for i in range(n_samples):
                    if active[i]:
                        nid = int(next_ids[i].item())
                        out_tokens[i].append(nid)
                        if nid == draw_close_id:
                            active[i] = False
                if not active.any():
                    break
                cur_input_ids = next_ids.unsqueeze(-1)  # (N, 1)
        finally:
            hook_handle.remove()

        return [torch.tensor(tokens, dtype=torch.long) for tokens in out_tokens]


def build_target_stroke_ids(vocab: StrokeVocab, stroke_dicts: list[dict]) -> list[int]:
    """Convert a list of {dx,dy,pen} dicts (from QuickDraw loader) into a token-id sequence
    wrapped in <DRAW>...</DRAW>.
    """
    from stroke_tokenizer import Stroke
    strokes = [Stroke(dx=s["dx"], dy=s["dy"], pen=s["pen"]) for s in stroke_dicts]
    return vocab.encode_drawing(strokes)
