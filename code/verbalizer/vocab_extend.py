"""Add 262 stroke tokens to Gemma 4's tokenizer and resize model embeddings.

Anole pattern: only the new vocabulary's embedding rows are freshly initialised.
Because Gemma 4 has tied word embeddings (embed and lm_head share weights),
initialising the new embedding rows automatically initialises the new lm_head rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stroke_tokenizer import add_stroke_tokens_to_tokenizer  # noqa: E402


def extend_model_and_tokenizer(
    model,
    tokenizer,
    init_scale: float = 0.02,
    pad_to_multiple_of: int = 8,
):
    """Add stroke tokens, resize embeddings, return the StrokeVocab.

    The new embedding rows are initialised with small Gaussian noise (std=`init_scale`).
    This matches Anole's recipe; the new rows learn fast in SFT.
    """
    vocab = add_stroke_tokens_to_tokenizer(tokenizer)
    old_size = model.get_input_embeddings().weight.shape[0]
    new_size = len(tokenizer)
    model.resize_token_embeddings(new_size, pad_to_multiple_of=pad_to_multiple_of)
    print(f"[vocab_extend] tokenizer: {old_size} → {new_size} (target {old_size + len(vocab.name_to_id)})")

    # Re-initialise the rows we just added (above old_size). HF's resize copies
    # the embedding mean by default, which is fine but we get sharper signal with
    # a small Gaussian init.
    with torch.no_grad():
        embed = model.get_input_embeddings()
        d = embed.weight.shape[1]
        new_rows = torch.zeros(new_size - old_size, d, dtype=embed.weight.dtype, device=embed.weight.device)
        new_rows.normal_(mean=0.0, std=init_scale)
        embed.weight.data[old_size:new_size] = new_rows
        # tied: lm_head shares weights via tied_weights, so no separate init needed
        if not model.config.tie_word_embeddings:
            out = model.get_output_embeddings()
            out_new = torch.zeros(new_size - old_size, d, dtype=out.weight.dtype, device=out.weight.device)
            out_new.normal_(mean=0.0, std=init_scale)
            out.weight.data[old_size:new_size] = out_new

    return vocab


def main():
    """Smoke test: load Gemma 4 E2B, extend vocab, verify shapes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[vocab_extend] loading {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(args.model_id)

    old_vocab = len(tok)
    print(f"[vocab_extend] before: vocab={old_vocab}, embed shape={model.get_input_embeddings().weight.shape}")
    vocab = extend_model_and_tokenizer(model, tok)
    print(f"[vocab_extend] after:  vocab={len(tok)}, embed shape={model.get_input_embeddings().weight.shape}")
    print(f"[vocab_extend] added {len(vocab.name_to_id)} stroke tokens")
    print(f"[vocab_extend] sample IDs: <DRAW>={vocab.name_to_id['<DRAW>']}, <ACT_TOKEN>={vocab.name_to_id['<ACT_TOKEN>']}")

    # Verify token round-trip
    ids = tok(f"hello <DRAW> world", return_tensors="pt")["input_ids"][0].tolist()
    print(f"[vocab_extend] 'hello <DRAW> world' encoded as: {ids}")
    print(f"[vocab_extend] decoded: {tok.decode(ids)}")


if __name__ == "__main__":
    main()
