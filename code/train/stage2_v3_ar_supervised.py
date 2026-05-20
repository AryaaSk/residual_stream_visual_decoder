"""Phase A4 final attempt: Stage 2 v3 — mean-centered AR training.

Diagnosis from FVE measurements: Linear/MLP AR with frozen backbone learns to
predict the *mean* activation across the training distribution, not the per-prompt
deviation from the mean. This gives FVE ~0 (no variance explained) but cosine
close to the within-cluster cosine.

Fix: subtract the training-set mean from h_target before computing the loss.
This forces AR to learn the PROMPT-SPECIFIC deviation from the mean, not the
mean itself. At inference time, add the mean back to AR's output.

If FVE on mean-centered targets is also ~0, the bottleneck is fundamental
(AR's Linear/MLP head over a frozen image embedding doesn't have enough
representational capacity to discriminate prompts at all, and we need a
different architecture).

If FVE > 0.1, the original setup was just hitting the mean-prediction
shortcut and centering unlocks real signal.

Usage:
    python code/train/stage2_v3_ar_supervised.py --layer 12 --steps 600 \
        --batch-size 8 --out-dir checkpoints/ar_v3
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
from stroke_tokenizer import Stroke  # noqa: E402
from train.stage2_v2_ar_supervised import (  # noqa: E402
    MLP2Head,
    extract_caption_activations,
    load_sft_corpus,
    strokes_from_dicts,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sft-corpus", type=Path, default=Path("data/sft_quickdraw.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/ar_v3"))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--canvas-size", type=int, default=224)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"L{args.layer:02d}_train.jsonl"

    print(f"[ar3] loading SFT corpus {args.sft_corpus}", flush=True)
    rows = load_sft_corpus(args.sft_corpus)[: args.n_train]
    print(f"[ar3] {len(rows)} rows kept", flush=True)

    print(f"[ar3] building AR at layer {args.layer}", flush=True)
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
    print(f"[ar3] head type={args.head_type}, trainable params={sum(p.numel() for p in head_params)/1e6:.2f}M", flush=True)
    optim = torch.optim.AdamW(head_params, lr=args.lr)

    # Extract activations + compute mean
    captions = [r["caption"] for r in rows]
    print(f"[ar3] extracting h_ℓ for {len(captions)} captions", flush=True)
    from transformers import AutoTokenizer
    target = ar.backbone.cuda()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    h_all = extract_caption_activations(target, tok, captions, args.layer)
    h_mean = h_all.mean(dim=0)
    h_centered = h_all - h_mean
    print(f"[ar3] activations shape={tuple(h_all.shape)} mean_norm={float(h_mean.norm()):.2f} "
          f"centered_norm_mean={float(h_centered.norm(dim=1).mean()):.2f}", flush=True)

    # Pre-render images
    print(f"[ar3] pre-rendering {len(rows)} drawings", flush=True)
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
            h_target_centered = h_centered[batch_idx].cuda()
            h_hat = ar.forward(batch_imgs)  # AR predicts in centered space now
            loss = F.mse_loss(h_hat.float(), h_target_centered.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                msg = {"step": step, "loss": float(loss.item()), "elapsed_sec": round(time.time() - t_start, 1)}
                log_f.write(json.dumps(msg) + "\n")
                log_f.flush()
                print(f"[ar3] step {step:5d} loss={msg['loss']:.4f}  ({msg['elapsed_sec']:.0f}s)", flush=True)

            if step % args.save_every == 0 and step > 0:
                save_dir = args.out_dir / f"L{args.layer:02d}" / f"step_{step:06d}"
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"linear": ar.linear.state_dict(), "head_type": args.head_type, "h_mean": h_mean.cpu()},
                           save_dir / "head.pt")
                print(f"[ar3] saved → {save_dir}", flush=True)

    final = args.out_dir / f"L{args.layer:02d}" / "final"
    final.mkdir(parents=True, exist_ok=True)
    torch.save({"linear": ar.linear.state_dict(), "head_type": args.head_type, "h_mean": h_mean.cpu()},
               final / "head.pt")
    print(f"[ar3] DONE → {final}", flush=True)


if __name__ == "__main__":
    main()
