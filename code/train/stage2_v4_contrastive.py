"""Phase A4 final: Stage 2 v4 — contrastive AR training (InfoNCE).

Switches the AR's training objective from MSE regression to a contrastive loss.
For each batch of B (image, h) pairs:
  - Forward all images through AR → ĥ_1, ..., ĥ_B
  - For each i: loss = -log(exp(sim(ĥ_i, h_i)/τ) / Σ_j exp(sim(ĥ_i, h_j)/τ))
  - sim = cosine (after L2 normalisation)

This explicitly pushes AR to discriminate per prompt rather than to minimise
average reconstruction error. The "predict the mean" shortcut becomes useless
because all predictions equal to the mean have identical similarity to every
target → zero discriminative signal under contrastive loss.

If FVE on the trained AR is still ~0 with this loss too, the bottleneck is
unrecoverable with frozen-backbone + linear head. If FVE > 0.1, AR's image
representation does carry per-prompt info that MSE couldn't extract.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ar.reconstructor import TruncatedGemmaAR  # noqa: E402
from render import render as stroke_render  # noqa: E402
from train.stage2_v2_ar_supervised import (  # noqa: E402
    MLP2Head,
    extract_caption_activations,
    load_sft_corpus,
    strokes_from_dicts,
)


def info_nce_loss(h_hat: torch.Tensor, h_target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE on a batch (B, d) of predicted and target activations."""
    h_hat_n = F.normalize(h_hat.float(), dim=-1)
    h_target_n = F.normalize(h_target.float(), dim=-1)
    logits = h_hat_n @ h_target_n.T / temperature  # (B, B)
    labels = torch.arange(h_hat_n.shape[0], device=h_hat_n.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sft-corpus", type=Path, default=Path("data/sft_quickdraw.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/ar_v4"))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="bigger is better for contrastive: more negatives per anchor")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--canvas-size", type=int, default=224)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"L{args.layer:02d}_train.jsonl"

    print(f"[ar4] loading SFT corpus {args.sft_corpus}", flush=True)
    rows = load_sft_corpus(args.sft_corpus)[: args.n_train]
    print(f"[ar4] {len(rows)} rows kept", flush=True)

    print(f"[ar4] building AR at layer {args.layer}", flush=True)
    ar = TruncatedGemmaAR.from_pretrained(args.model_id, layer_ell=args.layer, device="cuda")
    for p in ar.backbone.parameters():
        p.requires_grad = False
    hidden = ar.hidden_size
    if args.head_type == "linear":
        for p in ar.linear.parameters():
            p.requires_grad = True
        head_params = list(ar.linear.parameters())
    else:
        ar.linear = MLP2Head(hidden).cuda().to(next(ar.backbone.parameters()).dtype)
        head_params = list(ar.linear.parameters())
    print(f"[ar4] head={args.head_type}, trainable={sum(p.numel() for p in head_params)/1e6:.2f}M, τ={args.temperature}", flush=True)
    optim = torch.optim.AdamW(head_params, lr=args.lr)

    captions = [r["caption"] for r in rows]
    print(f"[ar4] extracting h_ℓ for {len(captions)} captions", flush=True)
    from transformers import AutoTokenizer
    target = ar.backbone.cuda()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    h_all = extract_caption_activations(target, tok, captions, args.layer)
    print(f"[ar4] activations shape={tuple(h_all.shape)} mean_norm={float(h_all.norm(dim=1).mean()):.2f}", flush=True)

    print(f"[ar4] pre-rendering {len(rows)} drawings", flush=True)
    images = []
    for i, r in enumerate(rows):
        strokes = strokes_from_dicts(r["strokes"])
        img = stroke_render(strokes, canvas_size=args.canvas_size)
        images.append(img)
        if i % 1000 == 0:
            print(f"  [render] {i}/{len(rows)}", flush=True)

    n = len(rows)
    perm = torch.randperm(n).tolist()
    cursor = 0
    t_start = time.time()

    with open(log_path, "a") as log_f:
        for step in range(args.steps):
            if cursor + args.batch_size > n:
                perm = torch.randperm(n).tolist()
                cursor = 0
            batch_idx = perm[cursor : cursor + args.batch_size]
            cursor += args.batch_size

            batch_imgs = [images[i] for i in batch_idx]
            h_target = h_all[batch_idx].cuda()
            h_hat = ar.forward(batch_imgs)
            loss = info_nce_loss(h_hat, h_target, args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                # diagnostic: mean cosine of (ĥ_i, h_i) vs mean cos of (ĥ_i, h_j≠i)
                with torch.no_grad():
                    hhn = F.normalize(h_hat.float(), dim=-1)
                    htn = F.normalize(h_target.float(), dim=-1)
                    cos_mat = hhn @ htn.T
                    pos = cos_mat.diag().mean().item()
                    neg = (cos_mat.sum() - cos_mat.diag().sum()) / (args.batch_size * (args.batch_size - 1))
                    neg = float(neg.item())
                msg = {"step": step, "loss": float(loss.item()), "pos_cos": pos, "neg_cos": neg,
                       "elapsed_sec": round(time.time() - t_start, 1)}
                log_f.write(json.dumps(msg) + "\n")
                log_f.flush()
                print(f"[ar4] step {step:5d} loss={msg['loss']:.4f} pos_cos={pos:.3f} neg_cos={neg:.3f} ({msg['elapsed_sec']:.0f}s)", flush=True)

            if step % args.save_every == 0 and step > 0:
                save_dir = args.out_dir / f"L{args.layer:02d}" / f"step_{step:06d}"
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"linear": ar.linear.state_dict(), "head_type": args.head_type},
                           save_dir / "head.pt")
                print(f"[ar4] saved → {save_dir}", flush=True)

    final = args.out_dir / f"L{args.layer:02d}" / "final"
    final.mkdir(parents=True, exist_ok=True)
    torch.save({"linear": ar.linear.state_dict(), "head_type": args.head_type}, final / "head.pt")
    print(f"[ar4] DONE → {final}", flush=True)


if __name__ == "__main__":
    main()
