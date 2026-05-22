"""Stage 3 — Pure NLA reconstruction (v3, the architecture we should have used from day 1).

The whole project's foundational claim, finally implemented without compromise:

    h_text  = Qwen_frozen(caption).hidden_states[L][last_text_token]
    drawing = AV(h_text)
    image   = render(drawing)
    h_image = Qwen_frozen(images=image).hidden_states[L][last_token]   # IMAGE ONLY
    reward  = cosine(h_text, h_image)

Train the AV (projector + LoRA + new-vocab rows) to maximise the reward via
REINFORCE with group-normalised advantage.

WHAT IS NOT IN THIS LOSS (deliberate):
    - No CLIP score. CLIP is the wrong oracle. The interpretability claim is
      "the model itself cannot distinguish text-vs-image"; CLIP doesn't enter.
    - No supervised CE on canonical drawings. That was the v1.x compromise
      that diluted the claim into "concept-selector over memorized templates".
    - No min-stroke penalty, no clip_bonus_weight regulariser. Single reward.
    - No KL anchor. Pure reward signal.

WHAT IS DIFFERENT FROM stage2_qwen_consistency.py (the bug we shipped):
    The old `extract_h_with_image(model, image, P)` passed the original caption
    `P` alongside the image. Qwen could recover h_text from the text alone;
    image was irrelevant; reward saturated near ceiling. That's why "the loop
    didn't work" — it was never actually tested. v3 image-only forwards
    eliminate the leak.

Two initialisation strategies:
    --av-init-ckpt /nonexistent       # option A — pure NLA from scratch
    --av-init-ckpt checkpoints/v2_0/L10/final   # option B — bootstrapped from v2.0 SFT

Run option A first (500 step gate). If reward EMA doesn't move, fall back to B
and explicitly note in shipped docs that drawings are SFT-prior-biased.

Usage:
    python code/train/stage3_pure_nla.py \\
        --layer 24 --steps 5000 --group-size 8 \\
        --out-dir checkpoints/v3/L24
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
from ar.lora_gemma4 import attach_lora_to_av, lora_param_iter  # noqa: E402
from render import render as stroke_render  # noqa: E402


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
    "Once upon a time in a kingdom far away.",
    "The mountain peak was covered in snow.",
    "A small house with a red roof and chimney.",
    "I am thinking about an elephant.",
    "An apple a day keeps the doctor away.",
    "I am picturing a bird flying across the sky.",
    "When the storm hit the village.",
]

# Probe set: 6 prompts we render at every probe step to see drawings emerge
PROBE_PROMPTS = [
    ("cat",      "I am thinking about a cat."),
    ("dog",      "I am thinking about a dog."),
    ("eiffel",   "Paris, the city of lights, is famous for the Eiffel"),
    ("smile",    "I am picturing a smiling face."),
    ("triangle", "Imagine a triangle inscribed in a circle."),
    ("storm",    "When the storm hit the village."),
]


def load_prompts(path: Path, fallback: list[str]) -> list[str]:
    if path is None or not path.exists():
        return fallback
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = []
    for r in rows:
        if isinstance(r, dict):
            if "caption" in r:
                out.append(r["caption"])
            elif "prompt" in r:
                out.append(r["prompt"])
        else:
            out.append(str(r))
    return out or fallback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, required=True, help="Activation layer L*")
    p.add_argument("--av-init-ckpt", type=Path, default=Path("nonexistent_pure_nla"),
                   help="If exists: option B (bootstrap from SFT). Otherwise: option A (from scratch).")
    p.add_argument("--prompts", type=Path,
                   default=Path("data/expanded_captions.jsonl"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=240)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--projector-lr", type=float, default=2e-5)
    p.add_argument("--lora-lr", type=float, default=5e-5)
    p.add_argument("--vocab-lr", type=float, default=5e-5)
    p.add_argument("--lora-first-n-layers", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--projector-alpha-init", type=float, default=0.5)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--probe-at", type=int, nargs="*", default=[100, 500, 1000, 2500, 5000])
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--gate-step", type=int, default=500,
                   help="At this step, print whether reward EMA moved by >=0.05 (gate decision)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train.jsonl"
    log_f = open(log_path, "a")

    # ---- Build AV (option A from scratch, or option B from SFT ckpt) ----
    if args.av_init_ckpt and args.av_init_ckpt.exists():
        print(f"[s3] OPTION B: loading AV from {args.av_init_ckpt}", flush=True)
        av = StrokeDecoder.from_ckpt(args.av_init_ckpt, model_id=args.model_id)
        init_mode = "option_B_from_sft"
    else:
        print(f"[s3] OPTION A: AV from pure base Qwen + new vocab + projector(α·I) + LoRA(B=0)", flush=True)
        av = StrokeDecoder.from_pretrained_and_extend(
            args.model_id, device="cuda", dtype=torch.bfloat16,
            use_projector=True, projector_alpha_init=args.projector_alpha_init,
        )
        init_mode = "option_A_from_scratch"
    av.model.train()
    device = av.device()

    # ---- Attach LoRA if not present ----
    existing_layer_idxs = set()
    for name, module in av.model.named_modules():
        if hasattr(module, "_lora"):
            import re
            m = re.search(r"\.layers\.(\d+)\.", name)
            if m:
                existing_layer_idxs.add(int(m.group(1)))
    if not existing_layer_idxs:
        print(f"[s3] attaching AV LoRA on first {args.lora_first_n_layers} language layers ...", flush=True)
        attach_lora_to_av(av, first_n_layers=args.lora_first_n_layers,
                          rank=args.lora_rank, alpha=args.lora_alpha, verbose=False)

    # ---- Frozen EVALUATOR Qwen (separate instance, no grads) ----
    # CRITICAL: AutoModelForImageTextToText loads the vision encoder. Using
    # AutoModelForCausalLM would silently drop pixel_values → image-only forwards
    # return identical activations for every input (huge bug we hit in v2.x).
    print(f"[s3] loading frozen evaluator Qwen (ImageTextToText) ...", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    qwen_eval = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    qwen_tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    try:
        qwen_proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        has_mm = True
    except Exception as e:
        qwen_proc = None
        has_mm = False
        print(f"[s3] WARN: AutoProcessor not available ({e}); image forward path may not work", flush=True)
    for p_ in qwen_eval.parameters():
        p_.requires_grad = False

    # ---- AV trainable surface ----
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

    n_train = sum(p.numel() for g in param_groups for p in g["params"] if p.requires_grad)
    print(f"[s3] trainable params: {n_train/1e6:.2f}M  (init={init_mode})", flush=True)

    # ---- Load prompts ----
    prompts = load_prompts(args.prompts, FALLBACK_PROMPTS)
    print(f"[s3] loaded {len(prompts)} prompts from {args.prompts}", flush=True)

    # ---- Helpers ----
    # Use Qwen-VL chat template for BOTH paths so the read position is
    # structurally identical: input ends with <|im_end|>\n, we read the last
    # token activation. The image path's wrapper contains NO caption — only
    # the conversation tokens + image markers.
    @torch.no_grad()
    def extract_h_text(text: str) -> torch.Tensor:
        """h_text = frozen Qwen(chat_wrap(text))[L][last_token]"""
        if has_mm:
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
            wrapped = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = qwen_proc(text=[wrapped], images=None, return_tensors="pt").to(device)
        else:
            inputs = qwen_tok(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = qwen_eval(**inputs, output_hidden_states=True, use_cache=False)
        return out.hidden_states[args.layer][0, -1, :].detach().to(torch.float32)

    @torch.no_grad()
    def extract_h_image_only(image: Image.Image) -> torch.Tensor | None:
        """h_image_only = frozen Qwen(chat_wrap(image_only))[L][last_token]

        The caption from the AV's prompt MUST NOT appear here. Only the
        Qwen-VL conversation wrapper + image patch tokens.
        """
        if not has_mm:
            return None
        messages = [{"role": "user", "content": [{"type": "image"}]}]
        wrapped = qwen_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        try:
            inputs = qwen_proc(text=[wrapped], images=[image], return_tensors="pt").to(device)
            out = qwen_eval(**inputs, output_hidden_states=True, use_cache=False)
            return out.hidden_states[args.layer][0, -1, :].detach().to(torch.float32)
        except Exception as e:
            print(f"[s3] image-only forward failed: {type(e).__name__}: {str(e)[:140]}", flush=True)
            return None

    def compute_logprobs_for_batch(h: torch.Tensor, ids_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Per-sample sum log-prob of generated stroke tokens, WITH grad through AV."""
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
        """Render each probe prompt and report (cosine, n_strokes, png) to out_dir/probe_step_N/."""
        probe_dir = args.out_dir / f"probe_step_{step:06d}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        was_training = av.model.training
        av.model.eval()
        try:
            for slug, prompt in PROBE_PROMPTS:
                h_text = extract_h_text(prompt)
                ids = av.generate_from_activation(
                    h_text, layer_ell=args.layer,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_k=args.top_k,
                )
                strokes, _ = av.vocab.decode_tokens_with_stats(ids.tolist())
                if len(strokes) < 2:
                    rows.append({"slug": slug, "prompt": prompt, "n_strokes": len(strokes),
                                 "cosine": None, "note": "degenerate"})
                    continue
                img = stroke_render(strokes, display_scale=2.0).convert("RGB")
                img_4x = stroke_render(strokes, display_scale=4.0).convert("RGB")
                img_4x.save(probe_dir / f"{slug}.png")
                h_img = extract_h_image_only(img)
                cos = float(F.cosine_similarity(h_text, h_img, dim=0).item()) if h_img is not None else None
                rows.append({"slug": slug, "prompt": prompt,
                             "n_strokes": len(strokes), "cosine": cos})
                print(f"[probe step {step}] {slug:8s}  n_strokes={len(strokes):3d}  cosine={cos:.4f}" if cos is not None else
                      f"[probe step {step}] {slug:8s}  (degenerate)", flush=True)
        finally:
            if was_training:
                av.model.train()
        (probe_dir / "probes.json").write_text(json.dumps(rows, indent=2))

    # ---- Optional initial probe (baseline measurement) ----
    print("[s3] initial probe ...", flush=True)
    render_probes(step=0)

    # ---- Training loop ----
    t_start = time.time()
    reward_ema = None
    initial_reward = None
    for step in range(1, args.steps + 1):
        P = random.choice(prompts)
        try:
            h_text = extract_h_text(P)
        except Exception as e:
            print(f"[s3] step {step}: h_text extraction failed: {e}", flush=True)
            continue

        # Sample G drawings (no grad through sampling)
        try:
            ids_list = av.generate_from_activation_batched(
                h_text, layer_ell=args.layer,
                n_samples=args.group_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k,
                constrain_to_stroke_vocab=True,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[s3] step {step}: OOM during sampling; skipping", flush=True)
            torch.cuda.empty_cache()
            continue

        # Compute rewards (image-only second forward — the FIX)
        rewards = []
        n_strokes_list = []
        for ids in ids_list:
            tok_list = ids.tolist()
            strokes, _malformed = av.vocab.decode_tokens_with_stats(tok_list)
            n_strokes_list.append(len(strokes))
            if len(strokes) < 2:
                rewards.append(0.0)  # degenerate → no signal
                continue
            try:
                image = stroke_render(strokes, display_scale=2.0).convert("RGB")
            except Exception:
                rewards.append(0.0)
                continue
            h_img = extract_h_image_only(image)   # ← IMAGE ONLY; caption never leaked
            if h_img is None:
                rewards.append(0.0)
                continue
            cos = float(F.cosine_similarity(h_text, h_img, dim=0).item())
            rewards.append(cos)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

        # Compute log-probs WITH grad and form policy-gradient loss
        sum_logprobs = compute_logprobs_for_batch(h_text, ids_list)
        pg = -torch.stack([adv[g].detach() * sum_logprobs[g] for g in range(args.group_size)]).mean()

        loss = pg

        try:
            (loss / args.grad_accum).backward()
        except torch.cuda.OutOfMemoryError:
            print(f"[s3] step {step}: OOM during backward; skipping", flush=True)
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

        # Stats
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
                "mean_reward": round(mean_r, 4),
                "reward_ema": round(reward_ema, 4),
                "rewards": [round(r, 3) for r in rewards],
                "pg": round(float(pg.item()), 4),
                "n_strokes_mean": round(sum(n_strokes_list) / max(1, len(n_strokes_list)), 1),
                "elapsed_sec": round(time.time() - t_start, 1),
            }
            log_f.write(json.dumps(msg) + "\n")
            log_f.flush()
            print(f"[s3] step {step:5d}  r_ema={msg['reward_ema']:+.3f}  r_now={msg['mean_reward']:+.3f}  pg={msg['pg']:+.3f}  strokes={msg['n_strokes_mean']:.1f}  ({msg['elapsed_sec']:.0f}s)", flush=True)

        if step == args.gate_step:
            delta = reward_ema - initial_reward
            print(f"\n=== GATE @ step {args.gate_step} ===", flush=True)
            print(f"initial reward EMA = {initial_reward:.4f}", flush=True)
            print(f"current reward EMA = {reward_ema:.4f}", flush=True)
            print(f"delta = {delta:+.4f}  (gate threshold: |delta| >= 0.05)", flush=True)
            print(f"=== {'PASS' if abs(delta) >= 0.05 else 'FAIL (reward not moving — consider option B)'} ===\n", flush=True)

        if step in args.probe_at:
            print(f"[s3] probe at step {step}", flush=True)
            render_probes(step=step)
            av.model.train()

        if step > 0 and step % args.save_every == 0:
            save_dir = args.out_dir / f"step_{step:06d}"
            lora_meta = {"first_n_layers": args.lora_first_n_layers,
                         "rank": args.lora_rank, "alpha": args.lora_alpha}
            av.save_ckpt(save_dir, include_lora_meta=lora_meta)
            print(f"[s3] saved → {save_dir}", flush=True)

    # Final save + probe
    final_dir = args.out_dir / "final"
    lora_meta = {"first_n_layers": args.lora_first_n_layers,
                 "rank": args.lora_rank, "alpha": args.lora_alpha}
    av.save_ckpt(final_dir, include_lora_meta=lora_meta)
    render_probes(step=args.steps)
    log_f.close()
    print(f"[s3] DONE → {final_dir}", flush=True)


if __name__ == "__main__":
    main()
