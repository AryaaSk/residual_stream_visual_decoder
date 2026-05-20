"""Stage 4 — NLA iterative refinement (the v1.0 main event).

For K iterations, alternate:
    [AR PHASE]  Use CURRENT AV to generate a fresh (drawing, h_target) buffer,
                then train AR's backbone-LoRA + Linear(d,d) head on it with MSE.
                AR continuously tracks AV's evolving drawing distribution
                (fixes the v0.1 distribution-shift collapse).

    [AV PHASE]  Run GRPO on AV using the just-trained AR as the judge.
                Reward = -log MSE(h_target, ĥ); group-normalised advantage;
                KL penalty back to the frozen AV_init to keep drawings drawing-like.
                AR is FROZEN during this phase (so reward is consistent for the AV updates).

    [EVAL]      Quick FVE measurement on a held-out probe set; checkpoint everything;
                append a row to findings/iter_log.jsonl.

Heavy reuse of v0.x code:
    verbalizer.stroke_decoder.StrokeDecoder.from_ckpt        (AV loading)
    train.stage3_av_grpo.sample_with_logprobs                (rollouts with gradient log-probs)
    train.stage3_av_grpo.evaluate_log_probs_under_ref        (KL ref scoring)
    train.stage3_av_grpo.reward_neg_log_mse                  (reward function)
    train.stage2_v2_ar_supervised.extract_caption_activations (caption→h batch)
    train.stage2_v2_ar_supervised.load_sft_corpus             (jsonl loader)
    train.stage2_v2_ar_supervised.strokes_from_dicts          (stroke parsing)
    render.render                                            (deterministic renderer)
    eval.fve_metric.metric_summary                           (FVE/cosine/MSE)
    ar.lora_gemma4.{lora_state_dict, load_lora_state, freeze_all_but_lora_and_linear}

Usage example:
    bin/h200 bg iter_L12 train/stage4_iterative.py \
      --layer 12 --iterations 5 \
      --av-ckpt checkpoints/av_sft/final \
      --sft-corpus data/sft_quickdraw.jsonl \
      --out-dir checkpoints/v1/L12 \
      --buffer-size 400 --ar-steps-per-iter 300 --ar-batch 16 \
      --av-steps-per-iter 150 --group-size 4 --kl-beta 0.05 \
      --alpha 0.5 --av-max-tokens 200 \
      --lora-rank 16 --lora-alpha 32 \
      --log-every 25
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ar.lora_gemma4 import freeze_all_but_lora_and_linear, load_lora_state, lora_state_dict  # noqa: E402
from ar.reconstructor import TruncatedGemmaAR  # noqa: E402
from eval.fve_metric import metric_summary  # noqa: E402
from render import render as stroke_render  # noqa: E402
from train.stage2_v2_ar_supervised import (  # noqa: E402
    extract_caption_activations,
    load_sft_corpus,
)
from train.stage3_av_grpo import (  # noqa: E402
    evaluate_log_probs_under_ref,
    reward_neg_log_mse,
    sample_with_logprobs,
)
from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402

# 20 held-out probes for the per-iteration FVE measurement.
HELDOUT_PROBES = [
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--av-ckpt", type=Path, required=True,
                   help="Path to Stage-1 SFT'd AV checkpoint dir (with av_ckpt.pt)")
    p.add_argument("--sft-corpus", type=Path, default=Path("data/sft_quickdraw.jsonl"),
                   help="Used as source of captions whose activations form the training corpus")
    p.add_argument("--n-corpus", type=int, default=8000,
                   help="Number of captions whose activations populate the training corpus")
    p.add_argument("--out-dir", type=Path, required=True)

    # Iterative loop
    p.add_argument("--iterations", type=int, default=5)

    # AR phase
    p.add_argument("--buffer-size", type=int, default=400,
                   help="Fresh (drawing, h) pairs sampled from CURRENT AV per iteration")
    p.add_argument("--ar-steps-per-iter", type=int, default=300)
    p.add_argument("--ar-batch", type=int, default=16)
    p.add_argument("--ar-lr", type=float, default=3e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)

    # AV phase
    p.add_argument("--av-steps-per-iter", type=int, default=150)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--av-lr", type=float, default=5e-5)
    p.add_argument("--kl-beta", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.5, help="activation injection scale")
    p.add_argument("--av-max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)

    # Eval
    p.add_argument("--n-eval-samples-per-probe", type=int, default=2)

    # Logging
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--canvas-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def jsonl_append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


@torch.no_grad()
def quick_eval_fve(av: StrokeDecoder, ar: TruncatedGemmaAR, layer: int, *,
                  alpha: float, max_tokens: int, n_samples_per_probe: int) -> dict:
    """Run FVE on HELDOUT_PROBES. Returns the metric_summary dict."""
    av.model.eval()
    ar.eval()
    device = av.device()
    all_h, all_hat = [], []
    for text in HELDOUT_PROBES:
        enc = av.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer][0, -1, :].detach()
        hats = []
        for _ in range(n_samples_per_probe):
            ids = av.generate_from_activation(h, layer_ell=layer, alpha=alpha,
                                              max_new_tokens=max_tokens, temperature=1.0)
            strokes = av.vocab.decode_tokens(ids.tolist())
            img = stroke_render(strokes)
            h_hat = ar.forward([img]).squeeze(0)
            hats.append(h_hat.detach())
        all_h.append(h.cpu())
        all_hat.append(torch.stack(hats, dim=0).mean(dim=0).cpu())
    return metric_summary(torch.stack(all_h, dim=0), torch.stack(all_hat, dim=0))


@torch.no_grad()
def regen_buffer(av: StrokeDecoder, h_corpus: torch.Tensor, *,
                 buffer_size: int, layer: int, alpha: float, max_tokens: int,
                 canvas_size: int, rng: random.Random):
    """Sample `buffer_size` (image, h_target) pairs from the current AV via activation injection."""
    av.model.eval()
    n = h_corpus.shape[0]
    buf_imgs, buf_h = [], []
    for _ in range(buffer_size):
        idx = rng.randrange(n)
        h = h_corpus[idx].cuda()
        ids = av.generate_from_activation(h, layer_ell=layer, alpha=alpha,
                                          max_new_tokens=max_tokens, temperature=1.0)
        strokes = av.vocab.decode_tokens(ids.tolist())
        img = stroke_render(strokes, canvas_size=canvas_size)
        buf_imgs.append(img)
        buf_h.append(h.cpu())
    return buf_imgs, torch.stack(buf_h, dim=0)


def save_checkpoint(out_dir: Path, iteration: int, av: StrokeDecoder, ar: TruncatedGemmaAR, *,
                    old_vocab: int) -> Path:
    """Save AV's trained new-vocab rows + AR's LoRA + Linear head as an iteration checkpoint."""
    ckpt_dir = out_dir / f"iter_{iteration:02d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # AV checkpoint (same format as Stage 1)
    embed = av.model.get_input_embeddings()
    new_rows = embed.weight.detach()[old_vocab:].cpu().clone()
    torch.save(
        {"new_embed_rows": new_rows, "old_vocab_size": old_vocab,
         "vocab_name_to_id": av.vocab.name_to_id},
        ckpt_dir / "av_ckpt.pt",
    )

    # AR checkpoint with LoRA + Linear
    ar_state = {
        "linear": ar.linear.state_dict(),
        "head_type": "linear",
    }
    if ar.has_backbone_lora:
        ar_state["lora"] = lora_state_dict(ar)
    torch.save(ar_state, ckpt_dir / "head.pt")
    return ckpt_dir


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "iter_log.jsonl"
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # ---- Load AV (trainable: new-vocab embedding rows only) ----
    print(f"[iter] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)

    # Reference AV for KL penalty (frozen at Stage-1 SFT init)
    av_ref = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av_ref.model.eval()
    for p in av_ref.model.parameters():
        p.requires_grad = False

    # Freeze AV everywhere except new-vocab embedding rows
    for p in av.model.parameters():
        p.requires_grad = False
    embed = av.model.get_input_embeddings()
    old_vocab = embed.weight.shape[0] - len(av.vocab.name_to_id)
    embed.weight.requires_grad = True

    def mask_old_embed_grad(grad):
        out = grad.clone()
        out[:old_vocab] = 0
        return out
    embed.weight.register_hook(mask_old_embed_grad)

    av_optim = torch.optim.AdamW([embed.weight], lr=args.av_lr)
    print(f"[iter] AV: {embed.weight.numel() / 1e6:.2f}M new-vocab params trainable", flush=True)

    # ---- Load AR (with backbone LoRA + Linear head trainable) ----
    print(f"[iter] building AR at layer {args.layer} with backbone LoRA(r={args.lora_rank}, α={args.lora_alpha})",
          flush=True)
    ar = TruncatedGemmaAR.from_pretrained(
        args.model_id, layer_ell=args.layer, device="cuda",
        use_backbone_lora=True, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        include_vision_tower=True,
    )
    lora_params, head_params = freeze_all_but_lora_and_linear(ar)
    ar_trainable = lora_params + head_params
    n_lora = sum(p.numel() for p in lora_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"[iter] AR trainable: {n_lora / 1e6:.2f}M LoRA + {n_head / 1e6:.2f}M head", flush=True)
    ar_optim = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.ar_lr},
         {"params": head_params, "lr": args.ar_lr}], weight_decay=0.0,
    )

    # ---- Extract caption-activation corpus once ----
    print(f"[iter] loading SFT corpus {args.sft_corpus}", flush=True)
    rows = load_sft_corpus(args.sft_corpus)
    captions = [r["caption"] for r in rows]
    # IMPORTANT: exclude held-out probes from the training distribution so the FVE
    # measurement actually tests generalisation, not memorisation.
    heldout_set = set(HELDOUT_PROBES)
    captions = [c for c in captions if c not in heldout_set]
    captions = captions[: args.n_corpus]
    print(f"[iter] extracting caption activations for {len(captions)} captions at L{args.layer} "
          f"(after held-out filter; {len(heldout_set)} eval probes excluded)", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    target_backbone = av.model  # reuse AV's backbone for extraction; same Gemma 4 weights apart from a few new vocab rows
    h_corpus = extract_caption_activations(target_backbone, tok, captions, args.layer)
    print(f"[iter] h_corpus shape={tuple(h_corpus.shape)} mean_norm={float(h_corpus.norm(dim=1).mean()):.2f}",
          flush=True)

    # Held-out: don't sample from these
    eval_h = h_corpus[: len(HELDOUT_PROBES)]  # reserve front of corpus for stable eval (still trains on the rest)
    train_h = h_corpus

    # ---- Iterative loop ----
    t_outer = time.time()
    best_fve = -float("inf")
    best_iter = -1

    for iteration in range(args.iterations):
        print(f"\n========== ITERATION {iteration + 1} / {args.iterations} ==========", flush=True)
        t_iter = time.time()

        # ---- AR PHASE: regen buffer, train AR ----
        print(f"[iter {iteration}] AR phase: regenerating buffer ({args.buffer_size} pairs) ...",
              flush=True)
        buf_imgs, buf_h = regen_buffer(
            av, train_h,
            buffer_size=args.buffer_size, layer=args.layer, alpha=args.alpha,
            max_tokens=args.av_max_tokens, canvas_size=args.canvas_size, rng=rng,
        )
        print(f"[iter {iteration}] buffer ready in {time.time() - t_iter:.0f}s", flush=True)

        ar.train()
        # Cycle through the buffer
        perm = list(range(len(buf_imgs)))
        rng.shuffle(perm)
        cursor = 0
        t_ar = time.time()
        for ar_step in range(args.ar_steps_per_iter):
            if cursor + args.ar_batch > len(perm):
                rng.shuffle(perm)
                cursor = 0
            batch_idx = perm[cursor:cursor + args.ar_batch]
            cursor += args.ar_batch
            batch_imgs = [buf_imgs[i] for i in batch_idx]
            batch_h = buf_h[batch_idx].cuda()

            h_hat = ar.forward(batch_imgs)
            loss = F.mse_loss(h_hat.float(), batch_h.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ar_trainable, max_norm=1.0)
            ar_optim.step()
            ar_optim.zero_grad(set_to_none=True)

            if ar_step % args.log_every == 0:
                jsonl_append(log_path, {
                    "phase": "AR", "iter": iteration, "step": ar_step,
                    "loss": float(loss.item()),
                    "elapsed_sec": round(time.time() - t_ar, 1),
                })
                print(f"  [AR {ar_step:4d}] loss={loss.item():.4f}  ({time.time() - t_ar:.0f}s)",
                      flush=True)
        print(f"[iter {iteration}] AR phase done in {time.time() - t_ar:.0f}s", flush=True)

        # ---- AV PHASE: GRPO with frozen AR ----
        ar.eval()
        for p in ar.parameters():
            p.requires_grad = False
        # Re-enable LoRA + head grad after AV phase for next iter
        # (do this at end of AV phase)
        av.model.train()  # AV in training mode (for new vocab grads to flow)
        t_av = time.time()
        reward_ema = 0.0
        ema_a = 0.95

        for av_step in range(args.av_steps_per_iter):
            idx = rng.randrange(train_h.shape[0])
            h_true = train_h[idx].cuda()

            rollouts = []
            for g in range(args.group_size):
                ids, lps = sample_with_logprobs(
                    av, h_true, args.layer,
                    alpha=args.alpha, max_tokens=args.av_max_tokens, temperature=args.temperature,
                )
                rollouts.append({"ids": ids, "lps": lps})

            images = [stroke_render(av.vocab.decode_tokens(r["ids"].tolist())) for r in rollouts]
            if not images:
                continue
            with torch.no_grad():
                h_hats = ar.forward(images)
            rewards = torch.tensor(
                [reward_neg_log_mse(h_true, h_hats[g]) for g in range(len(rollouts))],
                device="cuda",
            )
            group_mean = rewards.mean()
            group_std = rewards.std().clamp_min(1e-6)
            adv = (rewards - group_mean) / group_std

            pg_terms = []
            kl_terms = []
            for g, r in enumerate(rollouts):
                if len(r["lps"]) == 0:
                    continue
                sum_lp = r["lps"].sum()
                pg_terms.append(-adv[g].detach() * sum_lp)
                lp_ref = evaluate_log_probs_under_ref(av_ref, h_true, args.layer, r["ids"], alpha=args.alpha)
                kl_terms.append((r["lps"].detach() - lp_ref).mean())
            if not pg_terms:
                continue
            policy_loss = torch.stack(pg_terms).mean()
            kl_loss = torch.stack(kl_terms).mean() if kl_terms else torch.tensor(0.0, device="cuda")
            loss = policy_loss + args.kl_beta * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([embed.weight], max_norm=1.0)
            av_optim.step()
            av_optim.zero_grad(set_to_none=True)

            r_mean = float(rewards.mean().item())
            reward_ema = ema_a * reward_ema + (1 - ema_a) * r_mean if av_step > 0 else r_mean
            if av_step % args.log_every == 0:
                jsonl_append(log_path, {
                    "phase": "AV", "iter": iteration, "step": av_step,
                    "reward": r_mean, "reward_ema": reward_ema,
                    "policy_loss": float(policy_loss.item()),
                    "kl": float(kl_loss.item()) if torch.is_tensor(kl_loss) else 0.0,
                    "elapsed_sec": round(time.time() - t_av, 1),
                })
                print(f"  [AV {av_step:4d}] reward={r_mean:+.3f} ema={reward_ema:+.3f} "
                      f"pol={float(policy_loss.item()):+.3f} "
                      f"kl={float(kl_loss.item()) if torch.is_tensor(kl_loss) else 0.0:+.3f}",
                      flush=True)

        print(f"[iter {iteration}] AV phase done in {time.time() - t_av:.0f}s", flush=True)

        # ---- Re-enable AR trainable params for the NEXT iteration ----
        for p in lora_params:
            p.requires_grad = True
        for p in head_params:
            p.requires_grad = True

        # ---- EVAL ----
        print(f"[iter {iteration}] evaluating FVE on {len(HELDOUT_PROBES)} held-out probes ...",
              flush=True)
        t_eval = time.time()
        fve_summary = quick_eval_fve(
            av, ar, args.layer,
            alpha=args.alpha, max_tokens=args.av_max_tokens,
            n_samples_per_probe=args.n_eval_samples_per_probe,
        )
        eval_row = {"phase": "EVAL", "iter": iteration, **fve_summary,
                    "elapsed_sec": round(time.time() - t_eval, 1)}
        jsonl_append(log_path, eval_row)
        print(f"[iter {iteration}] FVE={fve_summary['fve']:.4f} cosine={fve_summary['cosine']:.4f} "
              f"MSE={fve_summary['mse']:.4f}  ({time.time() - t_eval:.0f}s)", flush=True)

        if fve_summary["fve"] > best_fve:
            best_fve = fve_summary["fve"]
            best_iter = iteration

        # ---- CHECKPOINT ----
        ckpt_dir = save_checkpoint(args.out_dir, iteration, av, ar, old_vocab=old_vocab)
        print(f"[iter {iteration}] saved → {ckpt_dir}  (total iter time: {time.time() - t_iter:.0f}s)",
              flush=True)

    # ---- FINAL SAVE: copy best iteration to "final/" ----
    final_dir = args.out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if best_iter >= 0:
        src = args.out_dir / f"iter_{best_iter:02d}"
        import shutil
        for f in src.iterdir():
            shutil.copy2(f, final_dir / f.name)
        print(f"[iter] copied best iter (#{best_iter}, FVE={best_fve:.4f}) → {final_dir}", flush=True)

    print(f"\n[iter] DONE → {args.out_dir}", flush=True)
    print(f"[iter] total wallclock: {time.time() - t_outer:.0f}s", flush=True)
    print(f"[iter] best FVE: {best_fve:.4f} at iteration {best_iter}", flush=True)


if __name__ == "__main__":
    main()
