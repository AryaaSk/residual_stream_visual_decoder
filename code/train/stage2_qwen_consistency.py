"""Stage 2 — Qwen-self-consistency REINFORCE.

The principled fix for the v1.0/v1.1 reward-hacking problem: use the target
model itself (Qwen 3.5-4B, frozen, unified vision+text stream) as the
evaluator. No trained AR head. No co-evolution.

Loop per step:
    1. Sample prompt P from data/v2_prompts.jsonl (any arbitrary text).
    2. h_text = Qwen(P).hidden_states[L][last_text_token]    # frozen target
    3. AV samples G drawings from h_text (batched).
    4. For each drawing g:
         image[g] = render(strokes[g])
         h_image_then_text[g] = Qwen(image[g] + P).hidden_states[L][last_text_token]
         reward[g] = cosine(h_text, h_image_then_text[g])
         reward[g] -= max(0, 15 - n_strokes[g]) * 0.05  # min-stroke penalty
         reward[g] += 0.1 * CLIP_sim(image[g], P)        # small sanity regulariser
    5. advantage[g] = (reward[g] - mean) / (std + 1e-6)
    6. pg_loss = - Σ_g advantage[g].detach() * Σ_t log_prob(drawing[g][t] | h_text)
    7. KL anchor: β * KL(av_current_new_vocab || av_init_new_vocab)
    8. Backprop on AV's projector + AV-LoRA + new-vocab rows. Qwen frozen.

Qwen 3.5-4B is a unified-stream multimodal model, so feeding the rendered
image alongside text and reading the residual stream at layer L is the
literal "same model, same layer, see what it thinks now" check the canonical
NLA formulation always wanted. v1.x couldn't have this because Gemma's
vision-tower-then-projection path produced activations in a different
distribution than text-derived activations at the same layer.

Usage:
    python code/train/stage2_qwen_consistency.py \
        --av-ckpt checkpoints/v2_0/L10/final \
        --layer 10 --steps 3000 --batch 4 --group-size 4 \
        --out-dir checkpoints/v2_0_stage2/L10
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402
from ar.lora_gemma4 import lora_param_iter  # noqa: E402
from render import render as stroke_render  # noqa: E402


def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(name)
    return model, proc


def load_prompts(path: Path, fallback: list[str]) -> list[str]:
    if not path.exists():
        print(f"[s2qsc] prompts file not found: {path}; using fallback ({len(fallback)} prompts)", flush=True)
        return fallback
    rows = [json.loads(l) for l in open(path)]
    prompts = [r["caption"] if isinstance(r, dict) and "caption" in r else r["prompt"] if isinstance(r, dict) and "prompt" in r else str(r) for r in rows]
    print(f"[s2qsc] loaded {len(prompts)} prompts from {path}", flush=True)
    return prompts


# A fallback prompt set if no file is present. Spans concrete + abstract.
FALLBACK_PROMPTS = [
    "I am thinking about a cat.",
    "Imagine a dog running through grass.",
    "Paris, the city of lights, is famous for the Eiffel",
    "She drew a flower on the canvas.",
    "A fish swimming in the deep ocean.",
    "The capital of France is",
    "Imagine a triangle inscribed in a circle.",
    "I am picturing a smiling face.",
    "The sun is shining brightly today.",
    "What is 47 + 38?",
    "Once upon a time in a kingdom far away.",
    "She received the news and felt deeply sad.",
    "The mountain peak was covered in snow.",
    "A small house with a red roof and chimney.",
    "I am thinking about an elephant.",
    "She laughed until her sides hurt.",
    "An apple a day keeps the doctor away.",
    "The crowd erupted in cheers when the team won.",
    "I am picturing a bird flying across the sky.",
    "When the storm hit the village.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, required=True, help="Activation layer L")
    p.add_argument("--prompts", type=Path, default=Path("data/v2_prompts.jsonl"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=1, help="How many prompts per step")
    p.add_argument("--group-size", type=int, default=4, help="Samples per prompt for GRPO-style advantage")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--kl-beta", type=float, default=0.02)
    p.add_argument("--min-strokes", type=int, default=15)
    p.add_argument("--min-stroke-penalty", type=float, default=0.05)
    p.add_argument("--clip-bonus-weight", type=float, default=0.1)
    p.add_argument("--projector-lr", type=float, default=2e-5)
    p.add_argument("--lora-lr", type=float, default=5e-5)
    p.add_argument("--vocab-lr", type=float, default=5e-5)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--probe-at", type=int, nargs="*", default=[200, 1000, 2000])
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train.jsonl"
    log_f = open(log_path, "a")

    # ----- Load AV (from v2.0 SFT ckpt) -----
    print(f"[s2qsc] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.train()  # we'll selectively unfreeze
    device = av.device()

    # ----- Load a SECOND frozen Qwen instance for evaluation -----
    # We need to feed image+text through Qwen WITHOUT touching av.model's grads
    # or KV cache state. Using a separate eval model keeps the two graphs clean.
    print(f"[s2qsc] loading frozen evaluator Qwen ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    qwen_eval = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    qwen_proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    qwen_tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    for p_ in qwen_eval.parameters():
        p_.requires_grad = False

    # ----- CLIP for the small regulariser -----
    print(f"[s2qsc] loading CLIP regulariser ...", flush=True)
    clip_model, clip_proc = load_clip()

    # ----- Set AV trainable surface -----
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

    # Capture init-vocab embed for KL anchor
    init_new_vocab_rows = embed.weight.detach()[old_vocab:].clone()

    param_groups = [
        {"params": [embed.weight], "lr": args.vocab_lr, "name": "vocab"},
        {"params": list(av.act_projector.parameters()), "lr": args.projector_lr, "name": "projector"},
        {"params": lora_params, "lr": args.lora_lr, "name": "lora"},
    ]
    optim = torch.optim.AdamW(param_groups, weight_decay=0.0, betas=(0.9, 0.95))

    # ----- Load prompts -----
    prompts = load_prompts(args.prompts, FALLBACK_PROMPTS)
    if not prompts:
        prompts = FALLBACK_PROMPTS

    # ----- Helpers -----
    @torch.no_grad()
    def extract_h(model, text: str) -> torch.Tensor:
        enc = qwen_tok(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        return out.hidden_states[args.layer][0, -1, :].detach().to(torch.float32)

    @torch.no_grad()
    def extract_h_with_image(model, image: Image.Image, text: str) -> torch.Tensor:
        """Run Qwen on image+text and extract activation at layer L, last token."""
        try:
            # Multimodal processor (AutoProcessor) handles image + text together
            inputs = qwen_proc(text=text, images=image, return_tensors="pt").to(device)
            out = model(**inputs, output_hidden_states=True, use_cache=False)
            return out.hidden_states[args.layer][0, -1, :].detach().to(torch.float32)
        except Exception as e:
            # Fallback: text-only forward (cosine with itself == 1, gives no signal)
            # This shouldn't happen if Qwen 3.5-4B is genuinely multimodal.
            print(f"[s2qsc] WARN: multimodal forward failed ({e}); falling back to text-only", flush=True)
            return extract_h(model, text)

    @torch.no_grad()
    def clip_score(image: Image.Image, text: str) -> float:
        inputs = clip_proc(text=[text], images=[image], return_tensors="pt", padding=True).to("cuda")
        out = clip_model(**inputs)
        return float(out.logits_per_image.squeeze().item())

    def sample_drawings_with_logprobs(h: torch.Tensor, n: int):
        """Sample n drawings, returning (list_of_token_ids, summed_logprob_tensor_grad).

        Uses generate_from_activation_batched for the tokens (no_grad inside) but
        then RE-RUNS a forward with grad to compute log_probs for REINFORCE.
        """
        ids_list = av.generate_from_activation_batched(
            h, layer_ell=args.layer,
            n_samples=n,
            alpha=0.5,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            constrain_to_stroke_vocab=True,
        )
        # ids_list: list[Tensor] of varying length.
        # Now we need to compute log_prob of each generated sequence under the
        # CURRENT model (with grad). Use act_sft_loss_batched but extract the
        # raw log-probs at each generated position.
        return ids_list

    def compute_logprobs_for_batch(h: torch.Tensor, ids_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Compute summed log-prob of each generated sequence with grad on."""
        # Use the same act_sft_loss_batched machinery: teacher-force the model
        # on (prompt + drawing_tokens) and read out per-token log-probs.
        # We rebuild this here because we need the raw log-probs, not the loss.
        from verbalizer.stroke_decoder import INJECT_PROMPT_TEMPLATE
        from stroke_tokenizer import ACT_TOKEN
        act_token_id = av.vocab.name_to_id[ACT_TOKEN]
        prompt_text = INJECT_PROMPT_TEMPLATE.format(layer=args.layer, act=ACT_TOKEN)
        prompt_ids = av.tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        act_pos = next(i for i, tid in enumerate(prompt_ids) if tid == act_token_id)
        prompt_len = len(prompt_ids)

        sum_logprobs = []
        embed_layer = av.model.get_input_embeddings()
        h_dev = h.to(device=device, dtype=embed_layer.weight.dtype)
        injected = av.act_projector(h_dev)

        first_pass = {"done": False}
        def hook(module, inputs, output):
            seq_len = output.shape[1]
            if not first_pass["done"] and seq_len > act_pos:
                output = output.clone()
                output[:, act_pos, :] = injected
                first_pass["done"] = True
            return output

        for ids in ids_list:
            tok_list = ids.tolist()
            if len(tok_list) == 0:
                sum_logprobs.append(torch.tensor(0.0, device=device))
                continue
            full_ids = prompt_ids + tok_list
            input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
            labels = torch.full_like(input_ids, -100)
            labels[0, prompt_len:] = torch.tensor(tok_list, device=device, dtype=torch.long)
            first_pass["done"] = False
            handle = embed_layer.register_forward_hook(hook)
            try:
                out = av.model(input_ids=input_ids, use_cache=False)
            finally:
                handle.remove()
            logits = out.logits[0, prompt_len - 1: -1]  # logits for predicting positions prompt_len..end
            target = torch.tensor(tok_list, device=device, dtype=torch.long)
            logp = F.log_softmax(logits.float(), dim=-1)
            chosen = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            sum_logprobs.append(chosen.sum())
        return sum_logprobs

    # ----- Training loop -----
    t_start = time.time()
    reward_ema = None
    for step in range(args.steps):
        # Sample a prompt
        P = random.choice(prompts)
        try:
            h_text = extract_h(qwen_eval, P)
        except Exception as e:
            print(f"[s2qsc] step {step}: h_text extraction failed: {e}", flush=True)
            continue

        # Sample G drawings
        ids_list = sample_drawings_with_logprobs(h_text, args.group_size)

        # Compute rewards (no grad)
        rewards = []
        n_strokes_list = []
        for ids in ids_list:
            tok_list = ids.tolist()
            strokes, malformed = av.vocab.decode_tokens_with_stats(tok_list)
            n_strokes_list.append(len(strokes))
            if len(strokes) < 1:
                rewards.append(-1.0)
                continue
            image = stroke_render(strokes, display_scale=2.0).convert("RGB")
            try:
                h_img = extract_h_with_image(qwen_eval, image, P)
                consistency = float(F.cosine_similarity(h_text, h_img, dim=0).item())
            except Exception as e:
                consistency = 0.0
            penalty = max(0, args.min_strokes - len(strokes)) * args.min_stroke_penalty
            try:
                cs = clip_score(image, P)
            except Exception:
                cs = 0.0
            reward = consistency - penalty + args.clip_bonus_weight * (cs / 30.0)
            rewards.append(reward)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

        # Compute log probs WITH grad
        sum_logprobs = compute_logprobs_for_batch(h_text, ids_list)

        # Policy gradient loss
        pg = -torch.stack([adv[g].detach() * sum_logprobs[g] for g in range(args.group_size)]).mean()

        # KL anchor on new-vocab rows
        current_new_vocab = embed.weight[old_vocab:]
        kl = ((current_new_vocab - init_new_vocab_rows.to(current_new_vocab.dtype)) ** 2).mean()
        loss = pg + args.kl_beta * kl

        (loss / args.grad_accum).backward()
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"] if p.requires_grad and p.grad is not None],
                max_norm=1.0,
            )
            optim.step()
            optim.zero_grad(set_to_none=True)

        # Logging
        mean_r = float(rewards_t.mean().item())
        if reward_ema is None:
            reward_ema = mean_r
        else:
            reward_ema = 0.95 * reward_ema + 0.05 * mean_r

        if step % args.log_every == 0:
            msg = {
                "step": step,
                "prompt": P[:60],
                "mean_reward": round(mean_r, 4),
                "reward_ema": round(reward_ema, 4),
                "kl": round(float(kl.item()), 4),
                "pg": round(float(pg.item()), 4),
                "n_strokes_mean": round(sum(n_strokes_list) / len(n_strokes_list), 1),
                "elapsed_sec": round(time.time() - t_start, 1),
            }
            log_f.write(json.dumps(msg) + "\n")
            log_f.flush()
            print(f"[s2qsc] step {step:5d}  reward_ema={msg['reward_ema']:+.3f}  pg={msg['pg']:+.3f}  kl={msg['kl']:.3f}  strokes={msg['n_strokes_mean']:.1f}  ({msg['elapsed_sec']:.0f}s)", flush=True)

        if step > 0 and step % args.save_every == 0:
            save_dir = args.out_dir / f"step_{step:06d}"
            lora_meta = {"first_n_layers": 24, "rank": 16, "alpha": 32}
            av.save_ckpt(save_dir, include_lora_meta=lora_meta)
            print(f"[s2qsc] saved → {save_dir}", flush=True)

    # Final save
    final_dir = args.out_dir / "final"
    lora_meta = {"first_n_layers": 24, "rank": 16, "alpha": 32}
    av.save_ckpt(final_dir, include_lora_meta=lora_meta)
    print(f"[s2qsc] DONE → {final_dir}", flush=True)
    log_f.close()


if __name__ == "__main__":
    main()
