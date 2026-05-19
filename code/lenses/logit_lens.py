"""Vanilla logit lens: apply final-LN + lm_head to intermediate residuals.

For each layer ℓ in the target model, decode the model's "prediction" of the
next token if the residual at layer ℓ were the final residual. This shows
how predictions sharpen across depth.

Usage
-----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lenses.logit_lens import LogitLens

    model = AutoModelForCausalLM.from_pretrained("google/gemma-4-e2b-it", ...).cuda()
    tok = AutoTokenizer.from_pretrained("google/gemma-4-e2b-it")
    lens = LogitLens(model, tok)

    trajectory = lens.trajectory("The capital of France is")
    # trajectory[ℓ] = list[(token_str, prob)] top-5 per layer

Cheap and parameter-free: no training, pure inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class LogitLensResult:
    layer: int
    top_tokens: list[tuple[str, float]]  # (token_string, probability)
    argmax_id: int
    entropy: float

    def __repr__(self) -> str:
        return f"L{self.layer:02d}: {self.top_tokens[0][0]!r} (p={self.top_tokens[0][1]:.3f}, H={self.entropy:.2f})"


class LogitLens:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def trajectory(
        self,
        text: str,
        top_k: int = 5,
        position: int = -1,
    ) -> list[LogitLensResult]:
        """Decode each layer's residual via the final-LN + lm_head pathway.

        Returns one LogitLensResult per layer (including embedding output, layer 0).
        `position` selects which token's residual we decode (default: last).
        """
        device = next(self.model.parameters()).device
        enc = self.tokenizer(text, return_tensors="pt").to(device)
        out = self.model(**enc, output_hidden_states=True, use_cache=False)

        results: list[LogitLensResult] = []
        # Gemma 4 module layout: model.model.language_model.norm + model.lm_head
        inner = self.model.model if hasattr(self.model, "model") else self.model
        lang = inner.language_model if hasattr(inner, "language_model") else inner
        final_norm = lang.norm if hasattr(lang, "norm") else lang.final_layer_norm
        lm_head = self.model.lm_head if hasattr(self.model, "lm_head") else self.model.get_output_embeddings()

        for ell, h in enumerate(out.hidden_states):
            # h shape: (1, T, hidden)
            h_pos = h[:, position, :]
            h_normed = final_norm(h_pos.to(dtype=h.dtype))
            logits = lm_head(h_normed).float()[0]
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, k=top_k)
            top_tokens = [(self.tokenizer.decode([int(i)]), float(p)) for p, i in zip(top.values, top.indices)]
            argmax_id = int(probs.argmax().item())
            ent = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
            results.append(LogitLensResult(layer=ell, top_tokens=top_tokens, argmax_id=argmax_id, entropy=ent))

        return results


def main():
    """Smoke test: run logit lens on a single prompt."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    lens = LogitLens(model, tok)
    traj = lens.trajectory(args.prompt, top_k=args.top_k)
    print(f"\nLogit-lens trajectory for: {args.prompt!r}\n")
    for r in traj:
        head = " | ".join(f"{t!r}={p:.2f}" for t, p in r.top_tokens)
        print(f"  L{r.layer:02d}  H={r.entropy:.2f}  top: {head}")


if __name__ == "__main__":
    main()
