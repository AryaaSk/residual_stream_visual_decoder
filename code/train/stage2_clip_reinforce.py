"""Stage 2 v2.1 (variant B) — CLIP-direct REINFORCE.

After both v2.0's Qwen-self-consistency and v2.1's Qwen-contrastive rewards
saturated at near-zero gradient (the user's interpretability critique stands:
"we're training a text-to-image model with h as a lossy conditioning channel"),
the simplest reward that actually moves is the SAME oracle we use at inference:
CLIP image-text similarity.

For each step:
    P = sample prompt
    h_text = Qwen(P)[L][last]                 (frozen target; AV's only input)
    drawings = AV.sample(h_text, n=G)
    for d:
        image = render(d)
        reward[d] = CLIP_cosine(image, P) - min_stroke_penalty(d)
    GRPO advantage + REINFORCE on AV (projector + LoRA + vocab)
    KL anchor to v2.0 SFT init

This IS the "drift toward text-to-image" mode the user warned about. The
interpretability story now lives in the *layer choice* — the AV must still
recover the concept from h, but the reward grades on how visually
recognisable the output is. h's content gates how recognisable we can get.

Practically: CLIP gives a dense, well-calibrated signal that scales with
visual recognisability. Training on it should produce visible improvements.

Differs from the inference-time best-of-32 CLIP-rank: that's a one-shot
selection over already-trained AV samples. This is *training* — REINFORCE
on the same metric, so the AV's whole distribution shifts toward
higher-CLIP outputs over many steps. The two compose: train via CLIP-direct,
then still CLIP-rank best-of-32 at inference for cleanup.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402
from ar.lora_gemma4 import lora_param_iter  # noqa: E402
from render import render as stroke_render  # noqa: E402


FALLBACK_PROMPTS = [
    "I am thinking about a cat.",
    "I am thinking about a dog.",
    "Imagine a fish.",
    "I am picturing a bird flying across the sky.",
    "I am thinking about a horse.",
    "I am thinking about an elephant.",
    "Imagine a flower in bloom.",
    "I am picturing a tree.",
    "I am picturing a cactus in the desert.",
    "I am picturing a mountain.",
    "The sun is shining.",
    "I am picturing a cloud in the sky.",
    "I am picturing a star in the night sky.",
    "I am picturing a small house with a red roof.",
    "I am thinking about a car.",
    "I am picturing an airplane.",
    "I am thinking about an apple.",
    "I am thinking about a pizza.",
    "I am picturing a clock on the wall.",
    "I am picturing an umbrella.",
]


def load_prompts(path: Path, fallback: list[str]) -> list[str]:
    if not path.exists():
        return fallback
    rows = [json.loads(l) for l in open(path)]
    out = []
    for r in rows:
        if isinstance(r, dict) and "caption" in r:
            out.append(r["caption"])
        elif isinstance(r, dict) and "prompt" in r:
            out.append(r["prompt"])
        else:
            out.append(str(r))
    return out


def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    return CLIPModel.from_pretrained(name).to("cuda").eval(), CLIPProcessor.from_pretrained(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--prompts", type=Path, default=Path("data/expanded_captions.jsonl"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--kl-beta", type=float, default=0.05)
    p.add_argument("--min-strokes", type=int, default=15)
    p.add_argument("--min-stroke-penalty", type=float, default=0.05)
    p.add_argument("--projector-lr", type=float, default=2e-5)
    p.add_argument("--lora-lr", type=float, default=5e-5)
    p.add_argument("--vocab-lr", type=float, default=5e-5)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(args.out_dir / "train.jsonl", "a")

    print(f"[s2clip] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.train()
    device = av.device()

    print(f"[s2clip] loading frozen Qwen for h extraction ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    qwen_eval = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    qwen_tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    for p_ in qwen_eval.parameters():
        p_.requires_grad = False

    print(f"[s2clip] loading CLIP (reward source) ...", flush=True)
    clip_model, clip_proc = load_clip()

    # AV trainable surface
    for p_ in av.model.parameters():
        p_.requires_grad = False
    embed = av.model.get_input_embeddings()
    old_vocab = embed.weight.shape[0] - len(av.vocab.name_to_id)
    embed.weight.requires_grad = True

    def mask_old_embed_grad(grad):
        out = grad.clone()
        out[:old_vocab] = 0
        return out
    embed.weight.register_hook(mask_old_embed_grad)

    for p_ in av.act_projector.parameters():
        p_.requires_grad = True
    lora_params = list(lora_param_iter(av))
    for p_ in lora_params:
        p_.requires_grad = True

    init_new_vocab = embed.weight.detach()[old_vocab:].clone()

    param_groups = [
        {"params": [embed.weight], "lr": args.vocab_lr, "name": "vocab"},
        {"params": list(av.act_projector.parameters()), "lr": args.projector_lr, "name": "projector"},
        {"params": lora_params, "lr": args.lora_lr, "name": "lora"},
    ]
    optim = torch.optim.AdamW(param_groups, weight_decay=0.0, betas=(0.9, 0.95))

    prompts = load_prompts(args.prompts, FALLBACK_PROMPTS)
    if not prompts:
        prompts = FALLBACK_PROMPTS
    print(f"[s2clip] {len(prompts)} prompts loaded", flush=True)

    @torch.no_grad()
    def extract_h(text: str) -> torch.Tensor:
        enc = qwen_tok(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = qwen_eval(**enc, output_hidden_states=True, use_cache=False)
        return out.hidden_states[args.layer][0, -1, :].detach().to(torch.float32)

    @torch.no_grad()
    def clip_batch_scores(images, text: str) -> list[float]:
        inputs = clip_proc(text=[text], images=images, return_tensors="pt", padding=True).to("cuda")
        return clip_model(**inputs).logits_per_image.squeeze(-1).tolist()

    def sample_drawings(h: torch.Tensor, n: int):
        return av.generate_from_activation_batched(
            h, layer_ell=args.layer, n_samples=n,
            alpha=0.5, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            constrain_to_stroke_vocab=True,
        )

    def compute_logprobs(h: torch.Tensor, ids_list):
        from verbalizer.stroke_decoder import INJECT_PROMPT_TEMPLATE
        from stroke_tokenizer import ACT_TOKEN
        act_token_id = av.vocab.name_to_id[ACT_TOKEN]
        prompt_text = INJECT_PROMPT_TEMPLATE.format(layer=args.layer, act=ACT_TOKEN)
        prompt_ids = av.tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        act_pos = next(i for i, tid in enumerate(prompt_ids) if tid == act_token_id)
        prompt_len = len(prompt_ids)
        embed_layer = av.model.get_input_embeddings()
        h_dev = h.to(device=device, dtype=embed_layer.weight.dtype)
        injected = av.act_projector(h_dev)
        first = {"done": False}

        def hook(module, inputs, output):
            seq_len = output.shape[1]
            if not first["done"] and seq_len > act_pos:
                output = output.clone()
                output[:, act_pos, :] = injected
                first["done"] = True
            return output

        out_lps = []
        for ids in ids_list:
            tl = ids.tolist()
            if len(tl) == 0:
                out_lps.append(torch.tensor(0.0, device=device))
                continue
            full_ids = prompt_ids + tl
            input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
            first["done"] = False
            handle = embed_layer.register_forward_hook(hook)
            try:
                out = av.model(input_ids=input_ids, use_cache=False)
            finally:
                handle.remove()
            logits = out.logits[0, prompt_len - 1:-1]
            target = torch.tensor(tl, device=device, dtype=torch.long)
            logp = F.log_softmax(logits.float(), dim=-1)
            chosen = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            out_lps.append(chosen.sum())
        return out_lps

    t_start = time.time()
    reward_ema = None
    for step in range(args.steps):
        P = random.choice(prompts)
        try:
            h_text = extract_h(P)
        except Exception as e:
            continue

        ids_list = sample_drawings(h_text, args.group_size)
        candidates = []
        for ids in ids_list:
            tl = ids.tolist()
            strokes, malformed = av.vocab.decode_tokens_with_stats(tl)
            if len(strokes) < 2:
                candidates.append({"strokes": strokes, "n_strokes": 0, "image": None})
            else:
                img = stroke_render(strokes, display_scale=2.0).convert("RGB")
                candidates.append({"strokes": strokes, "n_strokes": len(strokes), "image": img})

        # Batch CLIP score
        valid_imgs = [c["image"] for c in candidates if c["image"] is not None]
        valid_idx = [i for i, c in enumerate(candidates) if c["image"] is not None]
        rewards = [-1.0] * len(candidates)
        if valid_imgs:
            scores = clip_batch_scores(valid_imgs, P)
            for i, s in zip(valid_idx, scores):
                penalty = max(0, args.min_strokes - candidates[i]["n_strokes"]) * args.min_stroke_penalty
                rewards[i] = (s / 30.0) - penalty   # normalised so reward is roughly in [0, 1.5]

        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)
        lps = compute_logprobs(h_text, ids_list)
        pg = -torch.stack([adv[g].detach() * lps[g] for g in range(args.group_size)]).mean()

        current_new = embed.weight[old_vocab:]
        kl = ((current_new - init_new_vocab.to(current_new.dtype)) ** 2).mean()
        loss = pg + args.kl_beta * kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in param_groups for p in g["params"] if p.requires_grad and p.grad is not None],
            max_norm=1.0,
        )
        optim.step()
        optim.zero_grad(set_to_none=True)

        mean_r = float(rewards_t.mean().item())
        reward_ema = mean_r if reward_ema is None else 0.9 * reward_ema + 0.1 * mean_r

        if step % args.log_every == 0:
            msg = {
                "step": step, "prompt": P[:60],
                "mean_reward": round(mean_r, 4),
                "reward_ema": round(reward_ema, 4),
                "kl": round(float(kl.item()), 4),
                "pg": round(float(pg.item()), 4),
                "n_strokes_mean": round(sum(c["n_strokes"] for c in candidates) / len(candidates), 1),
                "elapsed": round(time.time() - t_start, 1),
            }
            log_f.write(json.dumps(msg) + "\n"); log_f.flush()
            print(f"[s2clip] step {step:4d}  r_ema={msg['reward_ema']:+.3f}  pg={msg['pg']:+.3f}  kl={msg['kl']:.3f}  strokes={msg['n_strokes_mean']:.1f}  ({msg['elapsed']:.0f}s)", flush=True)

        if step > 0 and step % args.save_every == 0:
            save_dir = args.out_dir / f"step_{step:06d}"
            lora_meta = {"first_n_layers": 24, "rank": 16, "alpha": 32}
            av.save_ckpt(save_dir, include_lora_meta=lora_meta)
            print(f"[s2clip] saved → {save_dir}", flush=True)

    final_dir = args.out_dir / "final"
    lora_meta = {"first_n_layers": 24, "rank": 16, "alpha": 32}
    av.save_ckpt(final_dir, include_lora_meta=lora_meta)
    print(f"[s2clip] DONE → {final_dir}", flush=True)
    log_f.close()


if __name__ == "__main__":
    main()
