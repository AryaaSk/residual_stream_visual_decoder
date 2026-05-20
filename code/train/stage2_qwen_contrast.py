"""Stage 2 v2.1 — Contrastive Qwen-self-consistency REINFORCE.

v2.0's Stage 2 reward (Qwen-self-consistency) saturated immediately because
v2.0 Stage 1.5 SFT already produces drawings that pass the cosine check at
near-ceiling. The reward had no headroom.

v2.1 fix: MAKE THE REWARD HARDER by adding a negative-prompt contrast term.
A drawing must produce activations that match its OWN concept's prompt
MORE than a randomly-chosen wrong concept's prompt.

Reward per drawing d sampled from h_text(P_correct):
    image = render(d)
    h_img_pos = Qwen(image + P_correct)[L][last]
    h_img_neg = Qwen(image + P_wrong)[L][last]
    sim_pos = cosine(h_text(P_correct), h_img_pos)
    sim_neg = cosine(h_text(P_wrong),   h_img_neg)
    reward = sim_pos - sim_neg                  # MARGIN: how much more does the
                                                # drawing fit the right prompt than wrong?
              + 0.1 * CLIP(image, P_correct)
              - min_stroke_penalty(d)

The reward is now ~0 for generic-plausible drawings (sim_pos ≈ sim_neg) and
positive for concept-specific drawings (sim_pos > sim_neg). REINFORCE
pushes AV toward specificity, which is what we actually want.

For training stability, P_wrong is sampled from the same prompt set but
filtered to be a DIFFERENT concept (uses a simple keyword heuristic on
the prompt strings).

Otherwise identical to stage2_qwen_consistency.py: KL anchor, batched
sampling, AV trainable surface = projector + LoRA + vocab rows.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402
from ar.lora_gemma4 import lora_param_iter  # noqa: E402
from render import render as stroke_render  # noqa: E402


# Same fallback set as v2.0 Stage 2.
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
    "The Eiffel Tower stands tall in Paris.",
    "The capital of France is Paris.",
    "She received the news and felt deeply sad.",
    "When the storm hit the village.",
]

# Very rough heuristic: pull the noun-ish key word out of a prompt to detect
# when two prompts are "about the same thing." If two prompts share their
# main noun (case-insensitive), we treat them as same-concept and avoid
# pairing them as positive/negative.
NOUN_KEYWORDS = [
    "cat", "dog", "fish", "bird", "horse", "elephant", "spider", "snake",
    "apple", "banana", "pizza", "donut", "tree", "flower", "leaf", "mushroom",
    "cactus", "mountain", "cloud", "sun", "moon", "star", "rainbow",
    "house", "bridge", "tent", "car", "bicycle", "train", "truck", "airplane",
    "book", "pencil", "scissors", "key", "clock", "umbrella", "chair", "table",
    "bed", "lamp", "door", "eiffel", "paris", "smile", "storm", "village",
    "ocean", "river", "lake", "face",
]


def keyword_of(prompt: str) -> str:
    """Return the first matching noun keyword or empty string."""
    s = prompt.lower()
    for kw in NOUN_KEYWORDS:
        if kw in s:
            return kw
    return ""


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
    p.add_argument("--prompts", type=Path, default=Path("data/v2_prompts.jsonl"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=240)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--kl-beta", type=float, default=0.02)
    p.add_argument("--min-strokes", type=int, default=15)
    p.add_argument("--min-stroke-penalty", type=float, default=0.05)
    p.add_argument("--clip-bonus-weight", type=float, default=0.1)
    p.add_argument("--projector-lr", type=float, default=3e-5)
    p.add_argument("--lora-lr", type=float, default=7e-5)
    p.add_argument("--vocab-lr", type=float, default=7e-5)
    p.add_argument("--probe-at", type=int, nargs="*", default=[50, 200, 400])
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train.jsonl"
    log_f = open(log_path, "a")

    print(f"[s2c] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.train()
    device = av.device()

    print(f"[s2c] loading frozen Qwen evaluator ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    qwen_eval = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    qwen_proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    qwen_tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    for p_ in qwen_eval.parameters():
        p_.requires_grad = False

    print(f"[s2c] loading CLIP regulariser ...", flush=True)
    clip_model, clip_proc = load_clip()

    # ----- AV trainable surface -----
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
    print(f"[s2c] {len(prompts)} prompts loaded", flush=True)

    @torch.no_grad()
    def extract_h(text: str) -> torch.Tensor:
        enc = qwen_tok(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = qwen_eval(**enc, output_hidden_states=True, use_cache=False)
        return out.hidden_states[args.layer][0, -1, :].detach().to(torch.float32)

    @torch.no_grad()
    def extract_h_with_image(image, text: str) -> torch.Tensor:
        try:
            inputs = qwen_proc(text=text, images=image, return_tensors="pt").to(device)
            out = qwen_eval(**inputs, output_hidden_states=True, use_cache=False)
            return out.hidden_states[args.layer][0, -1, :].detach().to(torch.float32)
        except Exception as e:
            print(f"[s2c] multimodal fwd failed: {e}; using text-only fallback", flush=True)
            return extract_h(text)

    @torch.no_grad()
    def clip_score(image, text: str) -> float:
        inputs = clip_proc(text=[text], images=[image], return_tensors="pt", padding=True).to("cuda")
        return float(clip_model(**inputs).logits_per_image.squeeze().item())

    def sample_drawings(h: torch.Tensor, n: int):
        return av.generate_from_activation_batched(
            h, layer_ell=args.layer, n_samples=n,
            alpha=0.5,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            constrain_to_stroke_vocab=True,
        )

    def compute_logprobs(h: torch.Tensor, ids_list):
        """Per-sequence sum of log-probs WITH grad on AV's trainable params."""
        from verbalizer.stroke_decoder import INJECT_PROMPT_TEMPLATE
        from stroke_tokenizer import ACT_TOKEN
        act_token_id = av.vocab.name_to_id[ACT_TOKEN]
        prompt_text = INJECT_PROMPT_TEMPLATE.format(layer=args.layer, act=ACT_TOKEN)
        prompt_ids = av.tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        act_pos = next(i for i, tid in enumerate(prompt_ids) if tid == act_token_id)
        prompt_len = len(prompt_ids)

        out_lps = []
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

    # ----- Training loop -----
    t_start = time.time()
    reward_ema = None
    margin_ema = None
    for step in range(args.steps):
        # Pick a positive prompt and a negative prompt (different concept)
        P_pos = random.choice(prompts)
        kw_pos = keyword_of(P_pos)
        # find a prompt with different keyword
        for _ in range(20):
            P_neg = random.choice(prompts)
            if keyword_of(P_neg) != kw_pos and P_neg != P_pos:
                break
        else:
            P_neg = random.choice([p for p in prompts if p != P_pos] or prompts)

        try:
            h_text_pos = extract_h(P_pos)
            h_text_neg = extract_h(P_neg)
        except Exception as e:
            print(f"[s2c] step {step}: target h extraction failed: {e}", flush=True)
            continue

        ids_list = sample_drawings(h_text_pos, args.group_size)

        rewards = []
        margins = []
        n_strokes_list = []
        for ids in ids_list:
            tl = ids.tolist()
            strokes, malformed = av.vocab.decode_tokens_with_stats(tl)
            n_strokes_list.append(len(strokes))
            if len(strokes) < 2:
                rewards.append(-1.0)
                margins.append(0.0)
                continue
            image = stroke_render(strokes, display_scale=2.0).convert("RGB")
            try:
                h_pos = extract_h_with_image(image, P_pos)
                h_neg = extract_h_with_image(image, P_neg)
                sim_pos = float(F.cosine_similarity(h_text_pos, h_pos, dim=0).item())
                sim_neg = float(F.cosine_similarity(h_text_neg, h_neg, dim=0).item())
                margin = sim_pos - sim_neg
            except Exception as e:
                print(f"[s2c]   evaluator fwd failed: {e}", flush=True)
                margin = 0.0
            penalty = max(0, args.min_strokes - len(strokes)) * args.min_stroke_penalty
            try:
                cs = clip_score(image, P_pos)
            except Exception:
                cs = 0.0
            reward = margin - penalty + args.clip_bonus_weight * (cs / 30.0)
            rewards.append(reward)
            margins.append(margin)

        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

        lps = compute_logprobs(h_text_pos, ids_list)
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
        mean_m = sum(margins) / max(1, len(margins))
        reward_ema = mean_r if reward_ema is None else 0.95 * reward_ema + 0.05 * mean_r
        margin_ema = mean_m if margin_ema is None else 0.95 * margin_ema + 0.05 * mean_m

        if step % args.log_every == 0:
            msg = {
                "step": step,
                "pos_prompt": P_pos[:50], "neg_prompt": P_neg[:50],
                "mean_reward": round(mean_r, 4),
                "reward_ema": round(reward_ema, 4),
                "margin_ema": round(margin_ema, 4),
                "kl": round(float(kl.item()), 4),
                "pg": round(float(pg.item()), 4),
                "n_strokes_mean": round(sum(n_strokes_list) / len(n_strokes_list), 1),
                "elapsed": round(time.time() - t_start, 1),
            }
            log_f.write(json.dumps(msg) + "\n")
            log_f.flush()
            print(f"[s2c] step {step:4d}  r_ema={msg['reward_ema']:+.3f}  margin_ema={msg['margin_ema']:+.3f}  pg={msg['pg']:+.3f}  strokes={msg['n_strokes_mean']:.1f}  ({msg['elapsed']:.0f}s)", flush=True)

        if step > 0 and step % args.save_every == 0:
            save_dir = args.out_dir / f"step_{step:06d}"
            lora_meta = {"first_n_layers": 24, "rank": 16, "alpha": 32}
            av.save_ckpt(save_dir, include_lora_meta=lora_meta)
            print(f"[s2c] saved → {save_dir}", flush=True)

    final_dir = args.out_dir / "final"
    lora_meta = {"first_n_layers": 24, "rank": 16, "alpha": 32}
    av.save_ckpt(final_dir, include_lora_meta=lora_meta)
    print(f"[s2c] DONE → {final_dir}", flush=True)
    log_f.close()


if __name__ == "__main__":
    main()
