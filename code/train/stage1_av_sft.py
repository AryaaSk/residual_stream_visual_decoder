"""Stage 1 — Activation Verbalizer SFT.

Train the AV on (caption → stroke-token-sequence) pairs from QuickDraw,
WITHOUT activation injection. Goal: AV learns to draw concepts from text.

This bias-trains the AV toward emitting valid stroke sequences and learning
the basic "drawing" behaviour, so that Stage 3 RL (which uses activation
injection) starts from a reasonable initialisation.

Trainable params (Anole pattern):
    * embedding rows for the 262 new stroke tokens (~0.4M)
    * lm_head rows for the 262 new stroke tokens (tied with embedding in Gemma 4,
      so it's the same tensor)
    * a thin LoRA on attention projections (rank 16) over the backbone

Logged via simple JSON-lines and tensorboard-style scalar prints.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder, build_target_stroke_ids, SFT_PROMPT_TEMPLATE  # noqa: E402
from stroke_tokenizer import DRAW_OPEN, DRAW_CLOSE  # noqa: E402


@dataclass
class SFTConfig:
    model_id: str = "Qwen/Qwen3.5-4B"
    corpus_path: Path = Path("data/sft_quickdraw.jsonl")
    out_dir: Path = Path("checkpoints/av_sft")
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 1e-4
    lora_lr: float = 1e-4
    new_vocab_lr: float = 5e-4
    epochs: int = 1
    max_steps: int | None = None
    lora_rank: int = 16
    lora_alpha: int = 32
    eval_every: int = 200
    log_every: int = 20
    save_every: int = 500
    seed: int = 0
    max_seq_len: int = 800


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def make_lora_config(rank: int, alpha: int):
    from peft import LoraConfig, TaskType
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=SFTConfig.model_id)
    parser.add_argument("--corpus", type=Path, default=SFTConfig.corpus_path)
    parser.add_argument("--out-dir", type=Path, default=SFTConfig.out_dir)
    parser.add_argument("--batch-size", type=int, default=SFTConfig.batch_size)
    parser.add_argument("--grad-accum", type=int, default=SFTConfig.grad_accum)
    parser.add_argument("--lr", type=float, default=SFTConfig.lr)
    parser.add_argument("--epochs", type=int, default=SFTConfig.epochs)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=SFTConfig.lora_rank)
    parser.add_argument("--eval-every", type=int, default=SFTConfig.eval_every)
    parser.add_argument("--log-every", type=int, default=SFTConfig.log_every)
    parser.add_argument("--save-every", type=int, default=SFTConfig.save_every)
    parser.add_argument("--seed", type=int, default=SFTConfig.seed)
    parser.add_argument("--max-seq-len", type=int, default=SFTConfig.max_seq_len)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.jsonl"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load corpus
    print(f"[sft] loading corpus {args.corpus}", flush=True)
    corpus = load_jsonl(args.corpus)
    print(f"[sft] {len(corpus)} examples loaded", flush=True)
    random.shuffle(corpus)
    val_size = max(1, int(0.02 * len(corpus)))
    val_corpus = corpus[:val_size]
    train_corpus = corpus[val_size:]
    print(f"[sft] train={len(train_corpus)} val={len(val_corpus)}", flush=True)

    # Model + vocab extension
    print(f"[sft] loading model {args.model_id}", flush=True)
    decoder = StrokeDecoder.from_pretrained_and_extend(args.model_id, device="cuda", dtype=torch.bfloat16)
    print(f"[sft] vocab size after extension: {len(decoder.tokenizer)}", flush=True)
    print(f"[sft] new tokens: {len(decoder.vocab.name_to_id)} (e.g. <DRAW>={decoder.vocab.name_to_id[DRAW_OPEN]})", flush=True)

    # Pre-tokenise stroke targets once
    print("[sft] pre-tokenising targets...", flush=True)
    train_data: list[tuple[str, list[int]]] = []
    for ex in train_corpus:
        ids = build_target_stroke_ids(decoder.vocab, ex["strokes"])
        if len(ids) > args.max_seq_len:
            continue
        train_data.append((ex["caption"], ids))
    val_data: list[tuple[str, list[int]]] = []
    for ex in val_corpus:
        ids = build_target_stroke_ids(decoder.vocab, ex["strokes"])
        if len(ids) > args.max_seq_len:
            continue
        val_data.append((ex["caption"], ids))
    print(f"[sft] retained train={len(train_data)} val={len(val_data)}", flush=True)

    # Anole-minimal: freeze backbone, train only the new-vocab embedding rows.
    # (Gemma 4 uses Gemma4ClippableLinear which PEFT 0.13 doesn't auto-recognise;
    # skipping LoRA simplifies things, matches Anole's smallest recipe, and is the
    # right starting point for "do the new tokens learn to draw at all?")
    for p in decoder.model.parameters():
        p.requires_grad = False
    embed = decoder.model.get_input_embeddings()
    old_vocab = embed.weight.shape[0] - len(decoder.vocab.name_to_id)
    print(f"[sft] embedding shape: {tuple(embed.weight.shape)}, new tokens start at {old_vocab}", flush=True)
    embed.weight.requires_grad = True

    optim = torch.optim.AdamW(
        [{"params": [embed.weight], "lr": SFTConfig.new_vocab_lr}],
        weight_decay=0.0,
    )

    # Mask gradient for the OLD embedding rows so we only update the new ones
    def mask_old_embed_grad(grad):
        out = grad.clone()
        out[:old_vocab] = 0
        return out
    embed.weight.register_hook(mask_old_embed_grad)

    # Training loop
    decoder.model.train()
    step = 0
    epoch_iter = range(args.epochs) if args.max_steps is None else range(10**9)
    t_start = time.time()
    with open(log_path, "a") as log_f:
        for epoch in epoch_iter:
            random.shuffle(train_data)
            for i in range(0, len(train_data), args.batch_size):
                batch = train_data[i : i + args.batch_size]
                captions = [c for c, _ in batch]
                targets = [t for _, t in batch]
                try:
                    loss = decoder.sft_loss_batched(captions, targets) / args.grad_accum
                    loss.backward()
                except torch.cuda.OutOfMemoryError:
                    print(f"[sft] OOM at step {step}, skipping batch", flush=True)
                    optim.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue

                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in decoder.model.parameters() if p.requires_grad], max_norm=1.0
                    )
                    optim.step()
                    optim.zero_grad(set_to_none=True)

                if step % args.log_every == 0:
                    msg = {
                        "step": step,
                        "epoch": epoch,
                        "loss": float(loss.item()) * args.grad_accum,
                        "elapsed_sec": round(time.time() - t_start, 1),
                    }
                    log_f.write(json.dumps(msg) + "\n")
                    log_f.flush()
                    print(f"[sft] step {step:6d}  loss={msg['loss']:.4f}  ({msg['elapsed_sec']:.0f}s)", flush=True)

                if step % args.save_every == 0 and step > 0:
                    save_dir = out_dir / f"step_{step:06d}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    # Save just the new-vocab rows + the vocab dict (Anole-minimal)
                    new_rows = embed.weight.detach()[old_vocab:].cpu().clone()
                    torch.save({"new_embed_rows": new_rows, "old_vocab_size": old_vocab,
                                "vocab_name_to_id": decoder.vocab.name_to_id}, save_dir / "av_ckpt.pt")
                    print(f"[sft] saved → {save_dir}", flush=True)

                step += 1
                if args.max_steps is not None and step >= args.max_steps:
                    break
            if args.max_steps is not None and step >= args.max_steps:
                break

    # Final save
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    new_rows = embed.weight.detach()[old_vocab:].cpu().clone()
    torch.save({"new_embed_rows": new_rows, "old_vocab_size": old_vocab,
                "vocab_name_to_id": decoder.vocab.name_to_id}, final_dir / "av_ckpt.pt")
    print(f"[sft] DONE. final at {final_dir}", flush=True)


if __name__ == "__main__":
    main()
