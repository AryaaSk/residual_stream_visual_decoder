"""Stage 3 — AV reinforcement learning with reconstruction reward (GRPO).

For each step:
    1. Pick a batch of activations (h, text) from the activation corpus.
    2. For each h, sample G drawings via AV with activation injection.
       Record per-token log-probs during sampling.
    3. Render each drawing to PNG.
    4. Run AR (frozen) on each PNG → ĥ.
    5. Reward = -log ||h - ĥ||²  (log-transformed for stability, NLA paper).
    6. Advantage = (reward - group_mean) / max(group_std, 1e-6)
    7. Policy-gradient loss: -mean(advantage * sum_t log_pi(a_t)).
    8. KL penalty: β * KL(AV ‖ AV_ref) where AV_ref is the frozen Stage-1 init.
    9. Backprop into AV's LoRA + new-vocab rows.

For Day-1 budget, this is a SCALED-DOWN GRPO: smaller group (G=4), shorter
drawings, fewer steps. Full-scale would use 16 H100s; we have 1-2 H200s.

The AR is frozen at the Stage-2 checkpoint. NLA continues training AR in parallel;
we skip that for Day-1 simplicity.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ar.reconstructor import TruncatedGemmaAR  # noqa: E402
from verbalizer.activation_injection import build_prompt_with_activation, stroke_token_ids  # noqa: E402
from verbalizer.stroke_decoder import INJECT_PROMPT_TEMPLATE, StrokeDecoder  # noqa: E402
from render import render as stroke_render  # noqa: E402
from stroke_tokenizer import DRAW_CLOSE, StrokeVocab  # noqa: E402


@torch.no_grad()
def load_ar_from_ckpt(ar_ckpt_dir: Path, model_id: str, layer_ell: int) -> TruncatedGemmaAR:
    """Load AR: full Gemma 4 + Linear(d,d) head with weights from `linear.pt`."""
    ar = TruncatedGemmaAR.from_pretrained(model_id, layer_ell=layer_ell, device="cuda")
    linear_sd = torch.load(ar_ckpt_dir / "linear.pt", map_location="cuda")
    ar.linear.load_state_dict(linear_sd)
    ar.eval()
    for p in ar.parameters():
        p.requires_grad = False
    return ar


def load_av_from_ckpt(av_ckpt_dir: Path, model_id: str) -> StrokeDecoder:
    """Load AV from Stage-1 av_ckpt.pt (Anole-minimal: just new embedding rows)."""
    return StrokeDecoder.from_ckpt(av_ckpt_dir, model_id=model_id, device="cuda", dtype=torch.bfloat16)


def sample_drawing_with_logprobs(
    av: StrokeDecoder,
    activation: torch.Tensor,
    layer_ell: int,
    *,
    alpha: float = 1.0,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
):
    """Sample one drawing from AV with activation injection.

    Returns (token_ids_tensor, log_probs_tensor) of shape (T,) each.
    Stops at </DRAW> or max_new_tokens.
    """
    parts = build_prompt_with_activation(
        av.model, av.tokenizer, av.vocab,
        layer_ell=layer_ell, activation=activation, alpha=alpha,
        prompt_template=INJECT_PROMPT_TEMPLATE,
    )
    device = av.device()
    allowed_ids = torch.tensor(list(stroke_token_ids(av.vocab)), device=device)
    allowed_mask = torch.full((av.model.get_input_embeddings().weight.shape[0],), float("-inf"), device=device)
    allowed_mask[allowed_ids] = 0.0
    draw_close_id = av.vocab.name_to_id[DRAW_CLOSE]

    generated_ids: list[int] = []
    log_probs: list[torch.Tensor] = []
    past = None
    inputs_embeds = parts.inputs_embeds

    for step in range(max_new_tokens):
        if past is None:
            out = av.model(inputs_embeds=inputs_embeds, use_cache=True)
        else:
            last_embed = av.model.get_input_embeddings()(torch.tensor([[generated_ids[-1]]], device=device))
            out = av.model(inputs_embeds=last_embed, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :] + allowed_mask  # (1, V)
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        lp = torch.log(probs[0, nxt].clamp_min(1e-12))
        generated_ids.append(nxt)
        log_probs.append(lp)
        if nxt == draw_close_id:
            break

    if not generated_ids:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, device=device)
    return torch.tensor(generated_ids, device=device, dtype=torch.long), torch.stack(log_probs, dim=0)


def reward_from_activations_robust(h_true: torch.Tensor, h_hat: torch.Tensor) -> float:
    """Reward = -log(MSE + eps)."""
    mse = F.mse_loss(h_true.float(), h_hat.float()).item()
    return float(-torch.log(torch.tensor(mse + 1e-6)).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--av-ckpt", type=Path, required=True)
    parser.add_argument("--ar-ckpt", type=Path, required=True, help="Stage-2 AR checkpoint dir")
    parser.add_argument("--activations-dir", type=Path, default=Path("data/activations"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/av_grpo"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=2,
                        help="How many activations per gradient step (effective batch = micro_batch * group_size)")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"L{args.layer:02d}_train.jsonl"

    # Load activation corpus
    from safetensors.torch import load_file
    layer_dir = args.activations_dir / f"L{args.layer:02d}"
    h_all = load_file(layer_dir / "activations.safetensors")["h"].float()
    texts: list[str] = []
    with open(layer_dir / "texts.jsonl") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    print(f"[grpo] {h_all.shape[0]} activations loaded; hidden={h_all.shape[1]}", flush=True)

    # Load AV (trainable) and a frozen AV reference for KL
    print(f"[grpo] loading AV from {args.av_ckpt}", flush=True)
    av = load_av_from_ckpt(args.av_ckpt, args.model_id)
    av_ref = load_av_from_ckpt(args.av_ckpt, args.model_id)
    av_ref.model.eval()
    for p in av_ref.model.parameters():
        p.requires_grad = False

    # Unfreeze AV's LoRA + new-vocab rows
    for p in av.model.parameters():
        p.requires_grad = False
    for name, p in av.model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
    embed = av.model.get_input_embeddings()
    old_vocab = embed.weight.shape[0] - len(av.vocab.name_to_id)
    embed.weight.requires_grad = True

    def mask_old_embed_grad(grad):
        out = grad.clone()
        out[:old_vocab] = 0
        return out
    embed.weight.register_hook(mask_old_embed_grad)

    trainable = [p for p in av.model.parameters() if p.requires_grad]
    print(f"[grpo] trainable AV params: {sum(p.numel() for p in trainable)/1e6:.2f}M", flush=True)
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    # Load AR (frozen)
    print(f"[grpo] loading AR from {args.ar_ckpt}", flush=True)
    ar = load_ar_from_ckpt(args.ar_ckpt, args.model_id, args.layer)

    n = h_all.shape[0]
    rng = torch.Generator()
    rng.manual_seed(0)
    perm = torch.randperm(n, generator=rng).tolist()
    cursor = 0
    t_start = time.time()

    with open(log_path, "a") as log_f:
        for step in range(args.steps):
            # Pick micro-batch of activations
            if cursor + args.micro_batch > n:
                perm = torch.randperm(n, generator=rng).tolist()
                cursor = 0
            batch_idx = perm[cursor : cursor + args.micro_batch]
            cursor += args.micro_batch

            h_batch = h_all[batch_idx].cuda()

            total_policy_loss = 0.0
            total_kl = 0.0
            total_reward = 0.0
            total_samples = 0

            for b in range(args.micro_batch):
                h_true = h_batch[b]  # (hidden,)
                # Sample G drawings
                samples = []
                for g in range(args.group_size):
                    ids, lps = sample_drawing_with_logprobs(
                        av, h_true, args.layer, alpha=args.alpha,
                        max_new_tokens=args.max_tokens, temperature=args.temperature,
                    )
                    samples.append({"ids": ids, "lps": lps})

                # Render + reconstruct
                rewards = []
                images = []
                for sm in samples:
                    strokes = av.vocab.decode_tokens(sm["ids"].tolist())
                    img = stroke_render(strokes)
                    images.append(img)
                if not images:
                    continue
                with torch.no_grad():
                    h_hats = ar.forward(images)  # (G, hidden)
                for g in range(len(samples)):
                    r = reward_from_activations_robust(h_true, h_hats[g])
                    rewards.append(r)
                rewards_t = torch.tensor(rewards, device="cuda")
                advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std().clamp_min(1e-6))

                # Policy gradient
                pg_loss_terms = []
                kl_terms = []
                for g, sm in enumerate(samples):
                    if len(sm["lps"]) == 0:
                        continue
                    sum_lp = sm["lps"].sum()
                    pg_loss_terms.append(-advantages[g].detach() * sum_lp)
                    # KL approximation: log_pi - log_pi_ref summed over the same tokens
                    # Recompute log_probs under av_ref for the same sampled ids
                    with torch.no_grad():
                        # cheap KL proxy: skip if too expensive. Here we re-forward av_ref.
                        parts = build_prompt_with_activation(
                            av_ref.model, av_ref.tokenizer, av_ref.vocab,
                            layer_ell=args.layer, activation=h_true, alpha=args.alpha,
                            prompt_template=INJECT_PROMPT_TEMPLATE,
                        )
                        # Run av_ref over (prompt + sampled_ids), collect log-probs of sampled tokens
                        full_ids = torch.cat([parts.input_ids[0], sm["ids"]], dim=0).unsqueeze(0)
                        # Embedding override for <ACT_TOKEN> in this combined sequence
                        full_embeds = av_ref.model.get_input_embeddings()(full_ids)
                        full_embeds[0, parts.act_position, :] = (h_true.to(full_embeds.dtype) * args.alpha)
                        out_ref = av_ref.model(inputs_embeds=full_embeds, use_cache=False)
                        log_probs_ref = F.log_softmax(out_ref.logits, dim=-1)
                        prompt_len = parts.input_ids.shape[1]
                        # log_prob_ref at positions prompt_len..prompt_len+T-1 for tokens sm["ids"]
                        targets = sm["ids"]
                        positions = torch.arange(prompt_len - 1, prompt_len - 1 + len(targets), device="cuda")
                        # log_probs_ref[0, position, target]
                        lp_ref = log_probs_ref[0, positions, targets]
                    kl = (sm["lps"].detach() - lp_ref).mean()
                    kl_terms.append(kl)

                if not pg_loss_terms:
                    continue
                policy_loss = torch.stack(pg_loss_terms).mean()
                kl_loss = torch.stack(kl_terms).mean() if kl_terms else torch.tensor(0.0, device="cuda")
                loss = policy_loss + args.kl_beta * kl_loss
                loss.backward()

                total_policy_loss += float(policy_loss.item())
                total_kl += float(kl_loss.item()) if torch.is_tensor(kl_loss) else 0.0
                total_reward += float(rewards_t.mean().item())
                total_samples += 1

            if total_samples == 0:
                optim.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                msg = {
                    "step": step,
                    "policy_loss": total_policy_loss / total_samples,
                    "kl": total_kl / total_samples,
                    "reward": total_reward / total_samples,
                    "elapsed_sec": round(time.time() - t_start, 1),
                }
                log_f.write(json.dumps(msg) + "\n")
                log_f.flush()
                print(f"[grpo] step {step:5d}  reward={msg['reward']:.3f}  pol={msg['policy_loss']:.3f}  kl={msg['kl']:.3f}  ({msg['elapsed_sec']:.0f}s)", flush=True)

            if step % args.save_every == 0 and step > 0:
                save_dir = args.out_dir / f"L{args.layer:02d}" / f"step_{step:06d}"
                save_dir.mkdir(parents=True, exist_ok=True)
                av.model.save_pretrained(save_dir)
                torch.save({"vocab_name_to_id": av.vocab.name_to_id}, save_dir / "stroke_vocab.pt")
                print(f"[grpo] saved → {save_dir}", flush=True)

    final = args.out_dir / f"L{args.layer:02d}" / "final"
    final.mkdir(parents=True, exist_ok=True)
    av.model.save_pretrained(final)
    torch.save({"vocab_name_to_id": av.vocab.name_to_id}, final / "stroke_vocab.pt")
    print(f"[grpo] DONE → {final}", flush=True)


if __name__ == "__main__":
    main()
