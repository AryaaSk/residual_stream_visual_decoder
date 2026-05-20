"""Stage 3 — AV reinforcement learning with reconstruction reward.

Per step:
    1. Pick K activations from the corpus.
    2. For each h, sample G drawings via AV with activation injection (hook-based).
       Record per-token log-probs during sampling.
    3. Render each drawing to PNG.
    4. Run AR (frozen) on each PNG → ĥ.
    5. Reward = -log ||h - ĥ||²  (log-transformed for numerical stability).
    6. Within each group: advantage = (reward - group_mean) / max(group_std, 1e-6).
       Across the whole batch: also subtract a moving-average baseline to reduce variance.
    7. Policy-gradient loss: -mean(advantage * sum_t log_pi(a_t)).
    8. KL penalty against the frozen AV_ref initial: β · KL(AV‖AV_ref).
    9. Backprop into AV's new-vocab embedding rows only (Anole-minimal).

Notes vs the original Anthropic NLA recipe:
    * No LoRA on backbone (PEFT 0.13 doesn't support Gemma4ClippableLinear).
    * Activation injection uses an embedding-layer FORWARD HOOK rather than
      passing inputs_embeds (Gemma 4 forbids the latter).
    * Group size and step count are scaled down for the Day-2 compute window.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ar.reconstructor import TruncatedGemmaAR  # noqa: E402
from render import render as stroke_render  # noqa: E402
from stroke_tokenizer import ACT_TOKEN, DRAW_CLOSE  # noqa: E402
from verbalizer.activation_injection import stroke_token_ids  # noqa: E402
from verbalizer.stroke_decoder import INJECT_PROMPT_TEMPLATE, StrokeDecoder  # noqa: E402


def install_act_hook(model, embed_layer, act_position: int, scaled_activation: torch.Tensor):
    """Register a forward hook that overrides embedding[0, act_position, :] on the FIRST call."""
    state = {"done": False}

    def hook(_module, _inputs, output):
        if not state["done"] and output.shape[1] > act_position:
            output = output.clone()
            output[0, act_position, :] = scaled_activation
            state["done"] = True
        return output

    return embed_layer.register_forward_hook(hook)


def sample_with_logprobs(
    av: StrokeDecoder,
    activation: torch.Tensor,
    layer_ell: int,
    *,
    alpha: float = 1.0,
    max_tokens: int = 200,
    temperature: float = 1.0,
    grad_through_logits: bool = True,
):
    """Sample one drawing from AV with activation injection. Returns (ids, log_probs).

    log_probs is computed with gradient enabled so we can backprop the policy
    gradient through it. (The sampled token IDs are detached — only the
    log-probability tensor is connected to the trainable embedding rows.)
    """
    device = av.device()
    act_token_id = av.vocab.name_to_id[ACT_TOKEN]
    text = INJECT_PROMPT_TEMPLATE.format(layer=layer_ell, act=ACT_TOKEN)
    enc = av.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
    input_ids = enc["input_ids"]
    pos_mask = input_ids[0] == act_token_id
    act_position = int(pos_mask.nonzero(as_tuple=False).item())

    embed_layer = av.model.get_input_embeddings()
    scaled_activation = activation.to(device=device, dtype=embed_layer.weight.dtype) * alpha
    hook_handle = install_act_hook(av.model, embed_layer, act_position, scaled_activation)

    draw_close_id = av.vocab.name_to_id[DRAW_CLOSE]
    allowed_ids = torch.tensor(list(stroke_token_ids(av.vocab)), device=device)
    allowed_mask = torch.full((embed_layer.weight.shape[0],), float("-inf"), device=device)
    allowed_mask[allowed_ids] = 0.0

    generated_ids: list[int] = []
    log_probs: list[torch.Tensor] = []
    past = None
    cur_input_ids = input_ids

    try:
        for step in range(max_tokens):
            ctx = torch.enable_grad() if grad_through_logits else torch.no_grad()
            with ctx:
                if past is None:
                    out = av.model(input_ids=cur_input_ids, use_cache=True)
                else:
                    out = av.model(input_ids=cur_input_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :] + allowed_mask
            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            with torch.no_grad():
                next_id = int(torch.multinomial(probs, 1).item())
            # log_prob of the *sampled* id with gradient
            lp = torch.log(probs[0, next_id].clamp_min(1e-12))
            generated_ids.append(next_id)
            log_probs.append(lp)

            if next_id == draw_close_id:
                break
            cur_input_ids = torch.tensor([[next_id]], device=device)
    finally:
        hook_handle.remove()

    if not generated_ids:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, device=device)
    return torch.tensor(generated_ids, device=device, dtype=torch.long), torch.stack(log_probs, dim=0)


@torch.no_grad()
def evaluate_log_probs_under_ref(
    av_ref: StrokeDecoder,
    activation: torch.Tensor,
    layer_ell: int,
    sampled_ids: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Recompute per-token log-probs of an existing sample under av_ref.

    Used for KL(AV ‖ AV_ref) where AV_ref is the frozen Stage-1 init.
    """
    device = av_ref.device()
    act_token_id = av_ref.vocab.name_to_id[ACT_TOKEN]
    text = INJECT_PROMPT_TEMPLATE.format(layer=layer_ell, act=ACT_TOKEN)
    enc = av_ref.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
    input_ids = enc["input_ids"]
    pos_mask = input_ids[0] == act_token_id
    act_position = int(pos_mask.nonzero(as_tuple=False).item())

    embed_layer = av_ref.model.get_input_embeddings()
    scaled_activation = activation.to(device=device, dtype=embed_layer.weight.dtype) * alpha
    hook_handle = install_act_hook(av_ref.model, embed_layer, act_position, scaled_activation)

    full_ids = torch.cat([input_ids[0], sampled_ids], dim=0).unsqueeze(0)
    try:
        out = av_ref.model(input_ids=full_ids, use_cache=False)
    finally:
        hook_handle.remove()
    # log probs over the full sequence
    log_probs_all = F.log_softmax(out.logits, dim=-1)
    prompt_len = input_ids.shape[1]
    # positions (prompt_len - 1) .. (prompt_len + len(sampled) - 2) generate tokens at
    # positions prompt_len .. prompt_len + len(sampled) - 1
    positions = torch.arange(prompt_len - 1, prompt_len - 1 + len(sampled_ids), device=device)
    return log_probs_all[0, positions, sampled_ids]


def reward_neg_log_mse(h_true: torch.Tensor, h_hat: torch.Tensor) -> float:
    mse = F.mse_loss(h_true.float(), h_hat.float()).item()
    return float(-torch.log(torch.tensor(mse + 1e-6)).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--av-ckpt", type=Path, required=True)
    parser.add_argument("--ar-ckpt", type=Path, required=True)
    parser.add_argument("--activations-dir", type=Path, default=Path("data/activations"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/av_grpo"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=1,
                        help="Number of distinct activations per step (effective rollouts = micro_batch * group_size)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.5)  # tuned via alpha_sweep on Stage-1 AV at L16
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"L{args.layer:02d}_train.jsonl"

    # Activations
    from safetensors.torch import load_file
    layer_dir = args.activations_dir / f"L{args.layer:02d}"
    h_all = load_file(layer_dir / "activations.safetensors")["h"].float()
    print(f"[grpo] {h_all.shape[0]} activations loaded; hidden={h_all.shape[1]}", flush=True)

    # AV (trainable: only new-vocab embedding rows)
    print(f"[grpo] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av_ref = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av_ref.model.eval()
    for p in av_ref.model.parameters():
        p.requires_grad = False

    # Freeze AV's backbone; train only the new-vocab embedding rows.
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

    print(f"[grpo] trainable params: {embed.weight.numel() / 1e6:.2f}M (new-vocab embeddings only)", flush=True)
    optim = torch.optim.AdamW([embed.weight], lr=args.lr)

    # AR (frozen). Supports two on-disk formats:
    #   stage2 v1: linear.pt  (Linear(d,d) state_dict)
    #   stage2 v2: head.pt    (dict with 'linear' state_dict + 'head_type')
    print(f"[grpo] loading AR from {args.ar_ckpt}", flush=True)
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
        print(f"[grpo] loaded AR v2 head_type={head_type}", flush=True)
    elif head_v1.exists():
        linear_sd = torch.load(head_v1, map_location="cuda")
        ar.linear.load_state_dict(linear_sd)
        print(f"[grpo] loaded AR v1 (Linear only)", flush=True)
    else:
        raise FileNotFoundError(f"No AR head found at {args.ar_ckpt} (expected head.pt or linear.pt)")
    ar.eval()
    for p in ar.parameters():
        p.requires_grad = False

    n = h_all.shape[0]
    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=rng).tolist()
    cursor = 0
    t_start = time.time()
    reward_ema = 0.0
    ema_alpha = 0.95

    with open(log_path, "a") as log_f:
        for step in range(args.steps):
            if cursor + args.micro_batch > n:
                perm = torch.randperm(n, generator=rng).tolist()
                cursor = 0
            batch_idx = perm[cursor : cursor + args.micro_batch]
            cursor += args.micro_batch

            step_loss = 0.0
            step_reward = 0.0
            step_kl = 0.0
            samples_seen = 0

            for h_true in (h_all[i].cuda() for i in batch_idx):
                rollouts = []
                for g in range(args.group_size):
                    ids, lps = sample_with_logprobs(
                        av, h_true, args.layer,
                        alpha=args.alpha, max_tokens=args.max_tokens, temperature=args.temperature,
                    )
                    rollouts.append({"ids": ids, "lps": lps})

                # Render + reconstruct
                images = []
                for r in rollouts:
                    strokes = av.vocab.decode_tokens(r["ids"].tolist())
                    images.append(stroke_render(strokes))
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
                advantage = (rewards - group_mean) / group_std

                pg_terms = []
                kl_terms = []
                for g, r in enumerate(rollouts):
                    if len(r["lps"]) == 0:
                        continue
                    sum_lp = r["lps"].sum()
                    pg_terms.append(-advantage[g].detach() * sum_lp)
                    # KL: log_pi - log_pi_ref summed over the same sampled tokens
                    lp_ref = evaluate_log_probs_under_ref(av_ref, h_true, args.layer, r["ids"], alpha=args.alpha)
                    kl_terms.append((r["lps"].detach() - lp_ref).mean())

                if not pg_terms:
                    continue
                policy_loss = torch.stack(pg_terms).mean()
                kl_loss = torch.stack(kl_terms).mean() if kl_terms else torch.tensor(0.0, device="cuda")
                loss = policy_loss + args.kl_beta * kl_loss
                loss.backward()

                step_loss += float(policy_loss.item())
                step_kl += float(kl_loss.item())
                step_reward += float(rewards.mean().item())
                samples_seen += 1

            if samples_seen == 0:
                optim.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_([embed.weight], max_norm=1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)

            step_reward_avg = step_reward / samples_seen
            reward_ema = ema_alpha * reward_ema + (1 - ema_alpha) * step_reward_avg if step > 0 else step_reward_avg

            if step % args.log_every == 0:
                msg = {
                    "step": step,
                    "policy_loss": step_loss / samples_seen,
                    "kl": step_kl / samples_seen,
                    "reward": step_reward_avg,
                    "reward_ema": reward_ema,
                    "elapsed_sec": round(time.time() - t_start, 1),
                }
                log_f.write(json.dumps(msg) + "\n")
                log_f.flush()
                print(
                    f"[grpo] step {step:4d} reward={msg['reward']:.3f} ema={msg['reward_ema']:.3f} "
                    f"pol={msg['policy_loss']:.3f} kl={msg['kl']:.3f} ({msg['elapsed_sec']:.0f}s)",
                    flush=True,
                )

            if step % args.save_every == 0 and step > 0:
                save_dir = args.out_dir / f"L{args.layer:02d}" / f"step_{step:06d}"
                save_dir.mkdir(parents=True, exist_ok=True)
                new_rows = embed.weight.detach()[old_vocab:].cpu().clone()
                torch.save(
                    {"new_embed_rows": new_rows, "old_vocab_size": old_vocab,
                     "vocab_name_to_id": av.vocab.name_to_id},
                    save_dir / "av_ckpt.pt",
                )
                print(f"[grpo] saved → {save_dir}", flush=True)

    final = args.out_dir / f"L{args.layer:02d}" / "final"
    final.mkdir(parents=True, exist_ok=True)
    new_rows = embed.weight.detach()[old_vocab:].cpu().clone()
    torch.save(
        {"new_embed_rows": new_rows, "old_vocab_size": old_vocab,
         "vocab_name_to_id": av.vocab.name_to_id},
        final / "av_ckpt.pt",
    )
    print(f"[grpo] DONE → {final}", flush=True)


if __name__ == "__main__":
    main()
