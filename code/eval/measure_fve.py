"""Measure FVE on a held-out set: how well does AR reconstruct h from AV-generated drawings?

Pipeline per probe:
    1. text → Gemma 4 → h_target at layer ℓ
    2. h_target → AV (with activation injection) → strokes
    3. strokes → renderer → PNG
    4. PNG → AR (truncated Gemma 4 + Linear/MLP head) → ĥ
    5. accumulate (h_target, ĥ) pairs
Then compute FVE, mean cosine, MSE.

Run as:
    python code/eval/measure_fve.py --av-ckpt checkpoints/av_grpo_v2/L16/final \
                                     --ar-ckpt checkpoints/ar_v2/L16/final \
                                     --layer 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ar.reconstructor import TruncatedGemmaAR  # noqa: E402
from eval.fve_metric import metric_summary  # noqa: E402
from render import render as stroke_render  # noqa: E402
from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402

DEFAULT_PROBES = [
    "The capital of France is",
    "I am thinking about a dog.",
    "I am thinking about a cat.",
    "I am picturing a small house with a red roof.",
    "Imagine a triangle inscribed in a circle.",
    "When the storm hit, the village",
    "Paris, the city of lights, is famous for the Eiffel",
    "What is 47 + 38? The answer is",
    "She received the news and felt deeply",
    "def fibonacci(n):",
    "The three primary colours are",
    "Once upon a time, in a kingdom far away,",
    "The funeral was somber and",
    "Her face lit up with joy",
    "An adult elephant is taller than a",
    "The smallest prime number is",
    "I am picturing a smiling face.",
    "Antarctica is colder than",
    "I am thinking about deep sadness.",
    "Right now I am thinking about",
]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it")
    parser.add_argument("--av-ckpt", type=Path, required=True)
    parser.add_argument("--ar-ckpt", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-samples-per-probe", type=int, default=4,
                        help="Average over N samples per probe to reduce per-sample noise")
    parser.add_argument("--out", type=Path, default=Path("findings/fve_measure.json"))
    args = parser.parse_args()

    # Load AV
    print(f"[fve] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    device = av.device()

    # Load AR
    print(f"[fve] loading AR from {args.ar_ckpt}", flush=True)
    ar = TruncatedGemmaAR.from_pretrained(args.model_id, layer_ell=args.layer, device="cuda")
    head_v2 = args.ar_ckpt / "head.pt"
    head_v1 = args.ar_ckpt / "linear.pt"
    if head_v2.exists():
        ckpt = torch.load(head_v2, map_location="cuda", weights_only=False)
        head_type = ckpt.get("head_type", "linear")
        if head_type == "mlp":
            from train.stage2_v2_ar_supervised import MLP2Head
            ar.linear = MLP2Head(ar.hidden_size).cuda().to(next(ar.backbone.parameters()).dtype)
        ar.linear.load_state_dict(ckpt["linear"])
        print(f"[fve] loaded AR v2 head_type={head_type}", flush=True)
    elif head_v1.exists():
        ar.linear.load_state_dict(torch.load(head_v1, map_location="cuda"))
        print(f"[fve] loaded AR v1 (Linear only)", flush=True)
    else:
        raise FileNotFoundError(f"No AR head found at {args.ar_ckpt}")
    ar.eval()

    all_h: list[torch.Tensor] = []
    all_hat: list[torch.Tensor] = []

    for probe_idx, text in enumerate(DEFAULT_PROBES):
        # Extract h_target
        enc = av.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h_target = out.hidden_states[args.layer][0, -1, :].detach()
        print(f"[fve] {probe_idx+1:2d}/{len(DEFAULT_PROBES)}: {text[:60]!r}  ||h||={float(h_target.norm()):.2f}", flush=True)

        # Sample N drawings, average ĥ across them
        h_hats: list[torch.Tensor] = []
        for s in range(args.n_samples_per_probe):
            ids = av.generate_from_activation(
                h_target, layer_ell=args.layer, alpha=args.alpha,
                max_new_tokens=args.max_tokens, temperature=args.temperature,
            )
            strokes = av.vocab.decode_tokens(ids.tolist())
            img = stroke_render(strokes)
            h_hat = ar.forward([img]).squeeze(0)
            h_hats.append(h_hat.detach())
        h_hat_avg = torch.stack(h_hats, dim=0).mean(dim=0)

        all_h.append(h_target.cpu())
        all_hat.append(h_hat_avg.cpu())

    h_stack = torch.stack(all_h, dim=0)
    hat_stack = torch.stack(all_hat, dim=0)
    summary = metric_summary(h_stack, hat_stack)
    print(f"\n[fve] SUMMARY:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    out = {
        "model_id": args.model_id,
        "layer": args.layer,
        "alpha": args.alpha,
        "av_ckpt": str(args.av_ckpt),
        "ar_ckpt": str(args.ar_ckpt),
        "n_probes": len(DEFAULT_PROBES),
        "n_samples_per_probe": args.n_samples_per_probe,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[fve] wrote {args.out}")


if __name__ == "__main__":
    main()
