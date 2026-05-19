"""Diagnostic for the v1.0 training: measure FVE using TRAINING-distribution prompts
("a drawing of a cat", "a drawing of a dog", ...) instead of the standard held-out
"thinking" prompts.

If FVE is HIGH here but ~0 on the held-out eval, then the model HAS learned but
the held-out test is measuring distribution shift between training captions and
"thinking" prompts. That's a fixable measurement issue (broaden eval distribution).

If FVE is STILL ~0 with training-style probes, then the model is overfitting per-pair
within the buffer and not learning a general image→activation mapping. That's a deeper
problem that needs a recipe change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ar.lora_gemma4 import load_lora_state  # noqa: E402
from ar.reconstructor import TruncatedGemmaAR  # noqa: E402
from eval.fve_metric import metric_summary  # noqa: E402
from render import render as stroke_render  # noqa: E402
from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402

# QuickDraw-style training-distribution probes — match the SFT corpus structure
TRAIN_STYLE_PROBES = [
    "a drawing of a cat",
    "a drawing of a dog",
    "a drawing of a horse",
    "a drawing of an elephant",
    "a drawing of a fish",
    "a drawing of a bird",
    "a drawing of an apple",
    "a drawing of a banana",
    "a drawing of a tree",
    "a drawing of a flower",
    "a drawing of a mountain",
    "a drawing of a sun",
    "a drawing of a house",
    "a drawing of a car",
    "a drawing of an airplane",
    "a drawing of a chair",
    "a drawing of a book",
    "a drawing of a phone",
    "a drawing of a key",
    "a drawing of a clock",
]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--ar-ckpt", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--model-id", default="google/gemma-4-e2b-it")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=150)
    p.add_argument("--n-samples-per-probe", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("findings/v1/fve_train_dist.json"))
    args = p.parse_args()

    print(f"[fve-traindist] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    device = av.device()

    print(f"[fve-traindist] loading AR from {args.ar_ckpt}", flush=True)
    head_v2 = args.ar_ckpt / "head.pt"
    if not head_v2.exists():
        raise FileNotFoundError(f"missing head.pt at {args.ar_ckpt}")
    ckpt = torch.load(head_v2, map_location="cuda", weights_only=False)
    has_lora = "lora" in ckpt and ckpt["lora"]
    ar = TruncatedGemmaAR.from_pretrained(args.model_id, layer_ell=args.layer, device="cuda",
                                          use_backbone_lora=has_lora)
    ar.linear.load_state_dict(ckpt["linear"])
    if has_lora:
        load_lora_state(ar, ckpt["lora"], strict=True)
        print(f"[fve-traindist] AR loaded with LoRA ({len(ckpt['lora']) // 2} modules)", flush=True)
    ar.eval()

    all_h = []
    all_hat = []
    for text in TRAIN_STYLE_PROBES:
        enc = av.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach()
        hats = []
        for _ in range(args.n_samples_per_probe):
            ids = av.generate_from_activation(h, layer_ell=args.layer, alpha=args.alpha,
                                              max_new_tokens=args.max_tokens, temperature=1.0)
            strokes = av.vocab.decode_tokens(ids.tolist())
            img = stroke_render(strokes)
            h_hat = ar.forward([img]).squeeze(0)
            hats.append(h_hat.detach())
        all_h.append(h.cpu())
        all_hat.append(torch.stack(hats, dim=0).mean(dim=0).cpu())
        print(f"[fve-traindist] {text}: ||h||={float(h.norm()):.2f}", flush=True)

    summary = metric_summary(torch.stack(all_h, dim=0), torch.stack(all_hat, dim=0))
    print(f"\n[fve-traindist] SUMMARY (training-distribution probes):")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "av_ckpt": str(args.av_ckpt),
        "ar_ckpt": str(args.ar_ckpt),
        "layer": args.layer,
        "alpha": args.alpha,
        "probes": TRAIN_STYLE_PROBES,
        "summary": summary,
    }, indent=2))
    print(f"[fve-traindist] wrote {args.out}")


if __name__ == "__main__":
    main()
