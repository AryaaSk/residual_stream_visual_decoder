"""v2.0 Phase 0 — Qwen 3.5-4B sanity check + activation-geometry layer pick.

What it does:
1. Load Qwen 3.5-4B (or fallback if unavailable).
2. Print architecture: hidden_size, num_hidden_layers, vocab_size, tokenizer info.
3. Verify the model accepts `inputs_embeds` (needed for our injection path).
4. Run a probe forward on "I am thinking about a cat".
5. Activation-geometry probe: feed 30 diverse prompts, compute pair-cosine of
   last-token activations at every layer, pick the discriminative band.
6. Print recommended L_primary and L_late, save geometry plot.

Outputs: findings/v2_0/qwen_arch.json (printed stats),
         findings/v2_0/qwen_layer_geometry.png (curve).

Usage:
    python code/eval/qwen_sanity_check.py --model-id Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROBES = [
    "I am thinking about a cat.",
    "Imagine a dog running through grass.",
    "The Eiffel Tower stands tall in Paris.",
    "She drew a flower on the canvas.",
    "A fish swimming in the deep ocean.",
    "The capital of France is Paris.",
    "Imagine a triangle inscribed in a circle.",
    "I am picturing a smiling face.",
    "The sun is shining brightly today.",
    "What is 47 + 38?",
    "Once upon a time in a kingdom far away.",
    "def fibonacci(n):",
    "She received the news and felt deeply sad.",
    "The mountain peak was covered in snow.",
    "A small house with a red roof and chimney.",
    "I am thinking about an elephant.",
    "She laughed until her sides hurt.",
    "The river bank was muddy after the rain.",
    "An apple a day keeps the doctor away.",
    "The crowd erupted in cheers when the team won.",
    "I am picturing a bird flying across the sky.",
    "When the storm hit the village.",
    "A cactus in the desert under the hot sun.",
    "The capital of Japan is Tokyo.",
    "Print all numbers from one to ten.",
    "The lake was calm and still in the morning.",
    "I am picturing a clock on the wall.",
    "Mount Everest is in Nepal.",
    "She drew a triangle and shaded one side.",
    "An old man walked slowly toward the bridge.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--fallback-ids", nargs="*",
                   default=["Qwen/Qwen3.5-3B", "Qwen/Qwen3.5-1.7B", "Qwen/Qwen3-4B"],
                   help="Try in order if --model-id fails")
    p.add_argument("--out-dir", type=Path, default=Path("findings/v2_0"))
    p.add_argument("--layer-step", type=int, default=2)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    # ----- Load model with fallback -----
    candidates = [args.model_id] + list(args.fallback_ids)
    model = None
    tok = None
    chosen = None
    for mid in candidates:
        try:
            print(f"[sanity] trying to load {mid} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                mid, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
            ).to("cuda").eval()
            chosen = mid
            print(f"[sanity] OK loaded {mid}", flush=True)
            break
        except Exception as e:
            print(f"[sanity] FAILED loading {mid}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    if model is None:
        print("[sanity] no candidate loaded; aborting", file=sys.stderr)
        sys.exit(1)

    cfg = model.config
    # Some configs put text dims under .text_config; others have them top-level.
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden_size = getattr(text_cfg, "hidden_size", None) or getattr(cfg, "hidden_size", None)
    num_layers = getattr(text_cfg, "num_hidden_layers", None) or getattr(cfg, "num_hidden_layers", None)
    vocab_size = getattr(text_cfg, "vocab_size", None) or getattr(cfg, "vocab_size", None)
    arch_summary = {
        "chosen_model_id": chosen,
        "architectures": getattr(cfg, "architectures", []),
        "hidden_size": int(hidden_size) if hidden_size else None,
        "num_hidden_layers": int(num_layers) if num_layers else None,
        "vocab_size": int(vocab_size) if vocab_size else None,
        "tokenizer_vocab_size": len(tok),
        "model_total_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "has_text_config": hasattr(cfg, "text_config"),
        "has_vision_config": hasattr(cfg, "vision_config"),
    }
    print(f"[sanity] arch: {json.dumps(arch_summary, indent=2)}", flush=True)

    # ----- inputs_embeds support check -----
    try:
        with torch.no_grad():
            enc = tok("hello", return_tensors="pt").to("cuda")
            embed = model.get_input_embeddings()
            inputs_embeds = embed(enc["input_ids"])
            out = model(inputs_embeds=inputs_embeds)
            assert out.logits.shape[-1] > 0
        arch_summary["accepts_inputs_embeds"] = True
        print(f"[sanity] model accepts inputs_embeds ✓", flush=True)
    except Exception as e:
        arch_summary["accepts_inputs_embeds"] = False
        arch_summary["inputs_embeds_error"] = str(e)[:200]
        print(f"[sanity] inputs_embeds FAILED: {e}", flush=True)
        print(f"[sanity] (will use embedding-hook workaround like Gemma)", flush=True)

    # ----- Forward probe -----
    text = "I am thinking about a cat."
    with torch.no_grad():
        enc = tok(text, return_tensors="pt").to("cuda")
        out = model(**enc, output_hidden_states=True, use_cache=False)
    n_hidden = len(out.hidden_states)
    print(f"[sanity] forward on {text!r}: hidden_states has {n_hidden} entries (= num_layers + 1 = embedding + each layer output)", flush=True)
    print(f"[sanity]   final h shape: {tuple(out.hidden_states[-1].shape)}", flush=True)
    arch_summary["num_hidden_states_entries"] = n_hidden

    # ----- Activation-geometry probe -----
    print(f"[sanity] extracting activations across {len(PROBES)} prompts at every {args.layer_step} layers ...", flush=True)
    activations_per_layer: dict[int, list[torch.Tensor]] = {}
    with torch.no_grad():
        for prompt in PROBES:
            enc = tok(prompt, return_tensors="pt").to("cuda")
            out = model(**enc, output_hidden_states=True, use_cache=False)
            for L in range(0, n_hidden, args.layer_step):
                h = out.hidden_states[L][0, -1, :].detach().cpu().to(torch.float32)
                activations_per_layer.setdefault(L, []).append(h)

    # pair-cosine per layer
    def pair_cosine(vecs: list[torch.Tensor]) -> float:
        V = torch.stack(vecs)
        V = V / (V.norm(dim=-1, keepdim=True) + 1e-8)
        cos = V @ V.T
        n = cos.shape[0]
        # exclude diagonal
        mask = ~torch.eye(n, dtype=torch.bool)
        return float(cos[mask].mean().item())

    layer_geometry = {L: pair_cosine(vs) for L, vs in activations_per_layer.items()}
    arch_summary["layer_pair_cosine"] = {str(k): round(v, 4) for k, v in layer_geometry.items()}
    print(f"[sanity] pair-cosine across layers:", flush=True)
    for L in sorted(layer_geometry.keys()):
        print(f"  L{L:02d}: pair-cosine = {layer_geometry[L]:.4f}", flush=True)

    # Recommend L_primary: the layer in mid-band with lowest pair-cosine
    n_layers = num_layers if num_layers else (n_hidden - 1)
    mid_start = int(0.30 * n_layers)
    mid_end = int(0.65 * n_layers)
    mid_band = {L: v for L, v in layer_geometry.items() if mid_start <= L <= mid_end}
    if mid_band:
        L_primary = min(mid_band, key=mid_band.get)
    else:
        L_primary = round(0.46 * n_layers)
    L_late = round(0.92 * n_layers)
    arch_summary["L_primary"] = int(L_primary)
    arch_summary["L_late"] = int(L_late)
    print(f"\n[sanity] RECOMMENDED LAYERS:", flush=True)
    print(f"  L_primary = {L_primary}   (mid-band, lowest pair-cosine in [L{mid_start}, L{mid_end}])", flush=True)
    print(f"  L_late    = {L_late}   (fractional depth 0.92, analogue of Gemma L24)", flush=True)

    # Save
    (args.out_dir / "qwen_arch.json").write_text(json.dumps(arch_summary, indent=2))
    print(f"[sanity] wrote {args.out_dir / 'qwen_arch.json'}", flush=True)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ls = sorted(layer_geometry.keys())
        vs = [layer_geometry[L] for L in Ls]
        plt.figure(figsize=(10, 4))
        plt.plot(Ls, vs, "o-")
        plt.axvline(L_primary, color="green", linestyle="--", label=f"L_primary={L_primary}")
        plt.axvline(L_late, color="red", linestyle="--", label=f"L_late={L_late}")
        plt.xlabel("layer index"); plt.ylabel("pair-cosine (lower = more discriminative)")
        plt.title(f"Activation geometry — {chosen} (last-token, {len(PROBES)} prompts)")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_dir / "qwen_layer_geometry.png", dpi=120)
        print(f"[sanity] wrote {args.out_dir / 'qwen_layer_geometry.png'}", flush=True)
    except Exception as e:
        print(f"[sanity] plot skipped: {e}")

    print("\n[sanity] DONE", flush=True)


if __name__ == "__main__":
    main()
