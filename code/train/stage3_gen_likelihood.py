"""Stage 3 — Pure NLA via generation-likelihood reward (v3, the architecture Qwen 3.5-4B actually supports).

After empirically disproving the cosine-based v3 hypothesis (see RESEARCH_NOTES.md
v3 chapter and `code/train/stage3_pure_nla.py` which is now deprecated), the
pivot is to use the frozen Qwen as a *captioner*. The reward is the model's
own conditional probability of correctly naming the concept in the AV's drawing.

Reward loop:

    h_text  = Qwen_frozen(chat_wrap(caption))[L10][last_token]
    ids     = AV(h_text, sampled)
    image   = render(ids)
    prompt  = "<|im_start|>user\\n<image>What is this a drawing of?<|im_end|>\\n"
              "<|im_start|>assistant\\nA drawing of a "
    target  = " {concept_word}"     # e.g. " cat", " elephant"
    reward  = sum log P(target_tokens | image + prompt)    # under frozen Qwen
    # REINFORCE on the AV:
    #   advantage[g] = (reward[g] - reward.mean()) / (reward.std() + eps)
    #   pg_loss = - advantage.detach() * sum_log_probs(drawing | h_text)

That is the entire loss. NO CLIP. NO supervised CE. NO min-stroke penalty.
NO KL anchor.

Key fixes vs the v2.x Stage 2 attempts:
  1. AutoModelForImageTextToText (NOT AutoModelForCausalLM). The latter silently
     drops pixel_values for Qwen 3.5-4B.
  2. The original caption never appears in the reward computation. Only the
     image and the fixed `"A drawing of a "` prefix.
  3. The reward is a generation log-prob, not a cosine on raw activations
     (cosine doesn't work for this base model — see v3 cosine dead-end finding).

Usage:
    python code/train/stage3_gen_likelihood.py \\
        --layer 10 --av-init-ckpt checkpoints/v2_0/L10/final \\
        --steps 5000 --out-dir checkpoints/v3/gen_likelihood/L10
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
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402
from ar.lora_gemma4 import attach_lora_to_av, lora_param_iter  # noqa: E402
from render import render as stroke_render  # noqa: E402
from stroke_tokenizer import ACT_TOKEN  # noqa: E402


# Fallback prompt set (used if data/expanded_captions.jsonl isn't loadable)
FALLBACK_PROMPTS = [
    ("cat",      "I am thinking about a cat."),
    ("dog",      "I am thinking about a dog."),
    ("elephant", "I am thinking about an elephant."),
    ("flower",   "Imagine a flower in bloom."),
    ("fish",     "Imagine a fish."),
    ("bird",     "I am picturing a bird flying across the sky."),
    ("sun",      "The sun is shining."),
    ("tree",     "I am picturing a tree."),
    ("airplane", "I am picturing an airplane."),
    ("car",      "I am thinking about a car."),
]


# Probe set: 6 prompts rendered at every probe step
PROBE_PROMPTS = [
    ("cat",      "I am thinking about a cat."),
    ("dog",      "I am thinking about a dog."),
    ("elephant", "I am thinking about an elephant."),
    ("flower",   "Imagine a flower in bloom."),
    ("sun",      "The sun is shining."),
    ("tree",     "I am picturing a tree."),
]


def load_prompt_concepts(path: Path, fallback: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Load (concept, caption) pairs. Concept-template rows have an explicit
    concept field; others get parsed from the caption."""
    if path is None or not path.exists():
        print(f"[s3gl] prompts file missing; using fallback ({len(fallback)} prompts)", flush=True)
        return fallback
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cap = r.get("caption", "").strip()
            concept = r.get("concept")
            if not concept:
                m = re.search(r"\ba drawing of an? (\w+)", cap)
                if m:
                    concept = m.group(1).rstrip("s")
            if cap and concept:
                pairs.append((concept, cap))
    if not pairs:
        return fallback
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, default=10, help="Activation layer for AV injection")
    p.add_argument("--av-init-ckpt", type=Path, default=Path("checkpoints/v2_0/L10/final"))
    p.add_argument("--prompts", type=Path, default=Path("data/expanded_captions.jsonl"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=240)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--projector-lr", type=float, default=4e-6)
    p.add_argument("--lora-lr", type=float, default=1e-5)
    p.add_argument("--vocab-lr", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.05,
                   help="KL anchor strength against frozen init (snapshot of v2.0 SFT) — "
                        "prevents REINFORCE from destroying the v2.0 visual prior")
    p.add_argument("--lora-first-n-layers", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--projector-alpha-init", type=float, default=0.5)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--question", default="What is this a drawing of?")
    p.add_argument("--continuation-prefix", default="A drawing of a")
    p.add_argument("--probe-at", type=int, nargs="*", default=[50, 200, 500, 1000, 2500, 5000])
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--gate-step", type=int, default=500)
    p.add_argument("--gate-delta", type=float, default=0.2,
                   help="Required reward EMA delta (log-prob units) at gate-step to pass")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train.jsonl"
    log_f = open(log_path, "a")

    # ----- Build AV (option B: init from v2.0 SFT) -----
    if args.av_init_ckpt and args.av_init_ckpt.exists():
        print(f"[s3gl] AV from ckpt {args.av_init_ckpt} (option B)", flush=True)
        av = StrokeDecoder.from_ckpt(args.av_init_ckpt, model_id=args.model_id)
    else:
        print(f"[s3gl] AV from scratch (option A) — pure base + new vocab + projector α·I", flush=True)
        av = StrokeDecoder.from_pretrained_and_extend(
            args.model_id, device="cuda", dtype=torch.bfloat16,
            use_projector=True, projector_alpha_init=args.projector_alpha_init,
        )
    av.model.train()
    device = av.device()

    # ----- Attach LoRA if not already present -----
    existing_layer_idxs = set()
    for name, module in av.model.named_modules():
        if hasattr(module, "_lora"):
            m = re.search(r"\.layers\.(\d+)\.", name)
            if m:
                existing_layer_idxs.add(int(m.group(1)))
    if not existing_layer_idxs:
        print(f"[s3gl] attaching AV LoRA on first {args.lora_first_n_layers} layers ...", flush=True)
        attach_lora_to_av(av, first_n_layers=args.lora_first_n_layers,
                          rank=args.lora_rank, alpha=args.lora_alpha, verbose=False)

    # ----- Frozen evaluator Qwen via AutoModelForImageTextToText -----
    # CRITICAL: AutoModelForCausalLM silently drops pixel_values for Qwen 3.5-4B.
    # The bug that invalidated v2.0 Stage 2.
    print(f"[s3gl] loading frozen evaluator Qwen (ImageTextToText) ...", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    qwen_eval = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    qwen_proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    for p_ in qwen_eval.parameters():
        p_.requires_grad = False

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

    param_groups = [
        {"params": [embed.weight], "lr": args.vocab_lr, "name": "vocab"},
        {"params": list(av.act_projector.parameters()), "lr": args.projector_lr, "name": "projector"},
        {"params": lora_params, "lr": args.lora_lr, "name": "lora"},
    ]
    optim = torch.optim.AdamW(param_groups, weight_decay=0.0, betas=(0.9, 0.95))

    # Snapshot trainable parameters at init for KL anchor. We compare the AV's
    # current logits over the AV stroke vocab to its INITIAL logits on the same
    # prompts. This penalises divergence from the v2.0 SFT distribution.
    init_new_vocab = embed.weight[old_vocab:].detach().clone()
    # Snapshot projector params for L2 anchor as well
    init_projector = {
        k: v.detach().clone() for k, v in av.act_projector.state_dict().items()
    }
    init_lora = {id(p): p.detach().clone() for p in lora_params}

    # ----- Load prompts -----
    prompts = load_prompt_concepts(args.prompts, FALLBACK_PROMPTS)
    print(f"[s3gl] loaded {len(prompts)} (concept, caption) pairs", flush=True)

    # ----- Pre-tokenize all unique concept targets (with leading space) -----
    unique_concepts = sorted(set(c for c, _ in prompts))
    concept_target_ids: dict[str, list[int]] = {}
    for c in unique_concepts:
        # Leading space is critical: BPE tokenises " cat" vs "cat" differently
        toks = qwen_proc.tokenizer.encode(" " + c, add_special_tokens=False)
        concept_target_ids[c] = toks
    print(f"[s3gl] {len(unique_concepts)} unique concepts pre-tokenised (avg {sum(len(t) for t in concept_target_ids.values())/len(concept_target_ids):.1f} tokens each)", flush=True)

    # ----- Helpers -----
    @torch.no_grad()
    def extract_h_text(text: str) -> torch.Tensor:
        """h_text = Qwen(chat_wrap(text))[L][last_token] — input to the AV."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        wrap = qwen_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        inp = qwen_proc(text=[wrap], images=None, return_tensors="pt").to(device)
        out = qwen_eval(**inp, output_hidden_states=True, use_cache=False)
        return out.hidden_states[args.layer][0, -1, :].detach().to(torch.float32)

    # Build the prefix text once
    PROMPT_MSGS = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": args.question}],
    }]
    PREFIX_TEXT = qwen_proc.apply_chat_template(PROMPT_MSGS, tokenize=False, add_generation_prompt=True) + args.continuation_prefix
    eval_tok = qwen_proc.tokenizer

    # Pre-tokenize each unique concept ONCE; reused for every step
    concept_target_tensor: dict[str, torch.Tensor] = {
        c: torch.tensor([eval_tok.encode(" " + c, add_special_tokens=False)],
                        dtype=torch.long, device=device)
        for c in unique_concepts
    }

    @torch.no_grad()
    def reward_log_prob(image: Image.Image, concept: str) -> float | None:
        """Reward = log P(concept_tokens | image + prefix) under frozen Qwen.

        Process image+prefix ONCE per call (single processor invocation), then
        extend the multimodal tensors (input_ids, attention_mask, mm_token_type_ids)
        with the pre-tokenised concept target. One model forward.
        """
        cand_t = concept_target_tensor.get(concept)
        if cand_t is None or cand_t.shape[1] == 0:
            return None
        try:
            inp_pre = qwen_proc(text=[PREFIX_TEXT], images=[image], return_tensors="pt").to(device)
        except Exception as e:
            return None
        prefix_len = inp_pre["input_ids"].shape[1]
        full = {k: v for k, v in inp_pre.items()}
        full["input_ids"] = torch.cat([inp_pre["input_ids"], cand_t], dim=1)
        if "attention_mask" in full:
            full["attention_mask"] = torch.cat(
                [inp_pre["attention_mask"], torch.ones_like(cand_t)], dim=1
            )
        if "mm_token_type_ids" in full:
            full["mm_token_type_ids"] = torch.cat(
                [inp_pre["mm_token_type_ids"], torch.zeros_like(cand_t)], dim=1
            )
        try:
            out = qwen_eval(**full, use_cache=False)
        except Exception as e:
            return None
        T = full["input_ids"].shape[1]
        logits = out.logits[0, prefix_len - 1: T - 1, :]
        target = full["input_ids"][0, prefix_len:]
        logp = F.log_softmax(logits.float(), dim=-1)
        chosen = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return float(chosen.sum().item())

    def compute_logprobs_for_batch(h: torch.Tensor, ids_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Per-sample sum log-prob of generated stroke tokens, WITH grad through AV."""
        from verbalizer.stroke_decoder import INJECT_PROMPT_TEMPLATE
        act_token_id = av.vocab.name_to_id[ACT_TOKEN]
        prompt_text = INJECT_PROMPT_TEMPLATE.format(layer=args.layer, act=ACT_TOKEN)
        prompt_ids = av.tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        act_pos = next(i for i, tid in enumerate(prompt_ids) if tid == act_token_id)
        prompt_len = len(prompt_ids)

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

        sum_logprobs = []
        for ids in ids_list:
            tok_list = ids.tolist()
            if len(tok_list) == 0:
                sum_logprobs.append(torch.tensor(0.0, device=device))
                continue
            full_ids = prompt_ids + tok_list
            input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
            first_pass["done"] = False
            handle = embed_layer.register_forward_hook(hook)
            try:
                out = av.model(input_ids=input_ids, use_cache=False)
            finally:
                handle.remove()
            logits = out.logits[0, prompt_len - 1: -1]
            target = torch.tensor(tok_list, device=device, dtype=torch.long)
            logp = F.log_softmax(logits.float(), dim=-1)
            chosen = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            sum_logprobs.append(chosen.sum())
        return sum_logprobs

    @torch.no_grad()
    def render_probes(step: int):
        probe_dir = args.out_dir / f"probe_step_{step:06d}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        was_training = av.model.training
        av.model.eval()
        rows = []
        try:
            for slug, prompt in PROBE_PROMPTS:
                # Get the concept from the slug for reward measurement
                concept = slug
                h_text = extract_h_text(prompt)
                ids = av.generate_from_activation(
                    h_text, layer_ell=args.layer,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_k=args.top_k,
                )
                strokes, _ = av.vocab.decode_tokens_with_stats(ids.tolist())
                if len(strokes) < 2:
                    rows.append({"slug": slug, "n_strokes": len(strokes), "reward": None, "note": "degenerate"})
                    continue
                # Reward image at 224 for speed; save a 4× upscale for human-eye review
                img = stroke_render(strokes, display_scale=1.0).convert("RGB")
                img_4x = stroke_render(strokes, display_scale=4.0).convert("RGB")
                img_4x.save(probe_dir / f"{slug}.png")
                rew = reward_log_prob(img, concept)
                rows.append({"slug": slug, "n_strokes": len(strokes), "reward": rew})
                print(f"[probe step {step}] {slug:10s} n_strokes={len(strokes):3d} reward={rew if rew is None else f'{rew:+.3f}'}", flush=True)
        finally:
            if was_training:
                av.model.train()
        (probe_dir / "probes.json").write_text(json.dumps(rows, indent=2))

    # ---- Initial probe (baseline) ----
    print("[s3gl] initial probe ...", flush=True)
    render_probes(step=0)

    # ---- Training loop ----
    t_start = time.time()
    reward_ema = None
    initial_reward = None
    for step in range(1, args.steps + 1):
        concept, P = random.choice(prompts)
        try:
            h_text = extract_h_text(P)
        except Exception as e:
            print(f"[s3gl] step {step} h_text failed: {e}", flush=True)
            continue

        try:
            ids_list = av.generate_from_activation_batched(
                h_text, layer_ell=args.layer, n_samples=args.group_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k,
                constrain_to_stroke_vocab=True,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[s3gl] step {step}: OOM during sampling; skipping", flush=True)
            torch.cuda.empty_cache()
            continue

        # Compute rewards (image-only, via generation log-prob of concept word)
        rewards = []
        n_strokes_list = []
        for ids in ids_list:
            tok_list = ids.tolist()
            strokes, _ = av.vocab.decode_tokens_with_stats(tok_list)
            n_strokes_list.append(len(strokes))
            if len(strokes) < 2:
                rewards.append(-10.0)  # degenerate → strong negative
                continue
            try:
                # 224x224 (display_scale=1.0) matches Qwen3-VL's native patch size,
                # producing 256 image tokens instead of 1024 at 448x448 → 4x faster
                # reward forward. Visual fidelity of strokes is unchanged for Qwen at 224x224.
                image = stroke_render(strokes, display_scale=1.0).convert("RGB")
            except Exception:
                rewards.append(-10.0)
                continue
            rew = reward_log_prob(image, concept)
            rewards.append(rew if rew is not None else -10.0)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

        sum_logprobs = compute_logprobs_for_batch(h_text, ids_list)
        pg = -torch.stack([adv[g].detach() * sum_logprobs[g] for g in range(args.group_size)]).mean()

        # KL anchor: penalise drift of new-vocab embeddings + projector + LoRA
        # from their v2.0 SFT init. Prevents REINFORCE from destroying the
        # visual prior that made flower/sun/tree decodable at step 0.
        kl_terms = []
        kl_terms.append(((embed.weight[old_vocab:] - init_new_vocab.to(embed.weight.dtype)) ** 2).mean())
        for k, init_v in init_projector.items():
            cur = av.act_projector.state_dict()[k]
            kl_terms.append(((cur - init_v.to(cur.dtype)) ** 2).mean())
        for lp in lora_params:
            init_v = init_lora.get(id(lp))
            if init_v is not None:
                kl_terms.append(((lp - init_v.to(lp.dtype)) ** 2).mean())
        kl_loss = torch.stack(kl_terms).mean()

        loss = pg + args.kl_beta * kl_loss

        try:
            (loss / args.grad_accum).backward()
        except torch.cuda.OutOfMemoryError:
            print(f"[s3gl] step {step}: OOM during backward; skipping", flush=True)
            optim.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"] if p.requires_grad and p.grad is not None],
                max_norm=1.0,
            )
            optim.step()
            optim.zero_grad(set_to_none=True)

        mean_r = float(rewards_t.mean().item())
        if reward_ema is None:
            reward_ema = mean_r
            initial_reward = mean_r
        else:
            reward_ema = 0.95 * reward_ema + 0.05 * mean_r

        if step % args.log_every == 0:
            msg = {
                "step": step,
                "prompt": P[:60],
                "concept": concept,
                "mean_reward": round(mean_r, 4),
                "reward_ema": round(reward_ema, 4),
                "rewards": [round(r, 3) for r in rewards],
                "pg": round(float(pg.item()), 4),
                "kl": round(float(kl_loss.item()), 6),
                "n_strokes_mean": round(sum(n_strokes_list) / max(1, len(n_strokes_list)), 1),
                "elapsed_sec": round(time.time() - t_start, 1),
            }
            log_f.write(json.dumps(msg) + "\n")
            log_f.flush()
            print(f"[s3gl] step {step:5d}  r_ema={msg['reward_ema']:+.3f}  r_now={msg['mean_reward']:+.3f}  pg={msg['pg']:+.3f}  kl={msg['kl']:.5f}  strokes={msg['n_strokes_mean']:.1f}  ({concept:10s}) ({msg['elapsed_sec']:.0f}s)", flush=True)

        if step == args.gate_step:
            delta = reward_ema - initial_reward
            print(f"\n=== GATE @ step {args.gate_step} ===", flush=True)
            print(f"initial reward EMA = {initial_reward:+.4f}", flush=True)
            print(f"current reward EMA = {reward_ema:+.4f}", flush=True)
            print(f"delta              = {delta:+.4f}  (gate threshold: delta >= +{args.gate_delta:.2f})", flush=True)
            print(f"=== {'PASS' if delta >= args.gate_delta else 'WEAK (consider tuning)'} ===\n", flush=True)

        if step in args.probe_at:
            print(f"[s3gl] probe at step {step}", flush=True)
            render_probes(step=step)
            av.model.train()

        if step > 0 and step % args.save_every == 0:
            save_dir = args.out_dir / f"step_{step:06d}"
            lora_meta = {"first_n_layers": args.lora_first_n_layers,
                         "rank": args.lora_rank, "alpha": args.lora_alpha}
            av.save_ckpt(save_dir, include_lora_meta=lora_meta)
            print(f"[s3gl] saved → {save_dir}", flush=True)

    # ---- Final save + probe ----
    final_dir = args.out_dir / "final"
    lora_meta = {"first_n_layers": args.lora_first_n_layers,
                 "rank": args.lora_rank, "alpha": args.lora_alpha}
    av.save_ckpt(final_dir, include_lora_meta=lora_meta)
    render_probes(step=args.steps)
    log_f.close()
    print(f"[s3gl] DONE → {final_dir}", flush=True)


if __name__ == "__main__":
    main()
