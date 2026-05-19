"""Tuned lens: per-layer Linear(d, d) adapter to align intermediate residuals
with the final-layer's coordinate frame.

Belrose et al. 2023 (arxiv 2303.08112). Trains a small `A_ℓ` so that
applying `final_LN · lm_head · A_ℓ` to layer-ℓ residual gives a good
prediction of the final-layer's logits.

Cheap: one Linear per layer, trained for ~1k steps each on a text corpus
without modifying the backbone. Outputs a single .pt with the per-layer
matrices.

Usage
-----
    python code/lenses/tuned_lens.py --out checkpoints/tuned_lens.pt --steps 1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_text_corpus(n: int) -> list[str]:
    """Reuse the same built-in corpus as activation_extractor.py for consistency."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.activation_extractor import load_text_corpus as _load
    return _load(None, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/tuned_lens.pt"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--n-text", type=int, default=2000)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[tuned_lens] loading {args.model_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    for p in model.parameters():
        p.requires_grad = False
    tok = AutoTokenizer.from_pretrained(args.model_id)

    hidden = model.config.text_config.hidden_size
    n_layers = model.config.text_config.num_hidden_layers
    print(f"[tuned_lens] hidden={hidden} n_layers={n_layers}", flush=True)

    # One Linear(d, d) per layer; layer 0 = embedding, 1..n_layers = transformer blocks
    n_positions = n_layers + 1
    adapters = nn.ModuleList([nn.Linear(hidden, hidden, bias=True).cuda() for _ in range(n_positions)])
    # init close to identity
    with torch.no_grad():
        for a in adapters:
            a.weight.copy_(torch.eye(hidden))
            a.bias.zero_()
    optim = torch.optim.AdamW(adapters.parameters(), lr=args.lr)

    # Gemma 4 module layout: model.model.language_model.{embed_tokens, layers, norm, ...}
    inner = model.model if hasattr(model, "model") else model
    lang = inner.language_model if hasattr(inner, "language_model") else inner
    final_norm = lang.norm if hasattr(lang, "norm") else lang.final_layer_norm
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()

    texts = load_text_corpus(args.n_text)
    print(f"[tuned_lens] {len(texts)} texts", flush=True)

    t_start = time.time()
    for step in range(args.steps):
        # Sample a batch of texts
        import random
        batch = random.sample(texts, args.batch_size)
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
            final_logits = out.logits  # (B, T, V), target distribution
            target_log_probs = F.log_softmax(final_logits, dim=-1)

        total_loss = 0.0
        n_valid = 0
        # For each layer, compute KL(adapter @ residual → final logits)
        for ell, h in enumerate(out.hidden_states):
            h_typed = h.float()
            with torch.no_grad():
                pass
            # Adapter forward (we want gradient through adapter)
            adapter = adapters[ell]
            h_proj = adapter(h_typed.to(adapter.weight.dtype))
            # Decode through final_norm + lm_head (frozen)
            with torch.set_grad_enabled(True):
                h_normed = final_norm(h_proj.to(dtype=h.dtype))
                logits = lm_head(h_normed).float()
            # KL(p_ℓ || p_final), per token, averaged
            log_p_ell = F.log_softmax(logits, dim=-1)
            kl = (target_log_probs.exp() * (target_log_probs - log_p_ell)).sum(dim=-1).mean()
            total_loss = total_loss + kl
            n_valid += 1
        loss = total_loss / n_valid
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            print(f"[tuned_lens] step {step:4d} loss={float(loss.item()):.4f} ({time.time()-t_start:.0f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    state = {f"adapter_L{i:02d}": adapters[i].state_dict() for i in range(n_positions)}
    state["meta"] = {"n_positions": n_positions, "hidden_size": hidden, "model_id": args.model_id}
    torch.save(state, args.out)
    print(f"[tuned_lens] saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
