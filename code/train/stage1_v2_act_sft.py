"""Stage 1.5 — Activation-Conditioned Supervised SFT for the AV.

The missing training stage in v1.0/v1.1. Direct supervised signal:
    given activation h_ℓ(caption), produce the real QuickDraw drawing of
    the matching concept.

Training pipeline per step:
    1. Sample (concept, drawing) from data/sft_quickdraw.jsonl.
    2. Pick a caption template for the concept (from expanded_captions.jsonl
       templates), e.g. "I am thinking about a cat" — gives caption diversity.
    3. Look up cached h = TARGET_GEMMA(caption).hidden_states[layer_ell][0, -1, :].
    4. Inject h via AV's embedding hook (routed through learnable ActProjector).
    5. Teacher-force AV on prompt + drawing tokens; CE loss on drawing tokens only.

Trainable surface:
    - new-vocab embedding rows (262 stroke tokens × d ~= 0.4M)
    - ActProjector (d × d Linear ~= 2.4M)
    - AV LoRA on first N language layers (q/k/v/o_proj, r=16 ~= 1.5M)

Total ~5M params. Backbone is frozen. Init: projector = α·I, LoRA B = 0 →
behaviour at step 0 is exactly v1.1's α·h injection.

Verification gate at step 500: render 4 probe drawings (dog, cat, eiffel,
triangle) and write PNGs to `<out_dir>/probe_step_0500/`. The orchestrator
checks visual recognisability before continuing to 5000 steps.

Usage:
    python code/train/stage1_v2_act_sft.py \
        --layer 24 \
        --av-init-ckpt checkpoints/av_sft/final \
        --data data/sft_quickdraw.jsonl \
        --captions-overlay data/expanded_captions.jsonl \
        --out-dir checkpoints/v1_2/L24 \
        --steps 5000 --batch 8
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

from verbalizer.stroke_decoder import StrokeDecoder, build_target_stroke_ids  # noqa: E402
from verbalizer.projector import ActProjector  # noqa: E402
from stroke_tokenizer import DRAW_OPEN, DRAW_CLOSE  # noqa: E402
from ar.lora_gemma4 import attach_lora_to_av, lora_param_iter  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def build_concept_to_templates(overlay_path: Path | None) -> dict[str, list[str]]:
    """Read the expanded captions corpus and group templates by concept.

    Returns {concept: [caption_template_1, caption_template_2, ...]}. Falls back
    to a single "a drawing of a {concept}" template if the overlay isn't found.
    """
    out: dict[str, list[str]] = {}
    if overlay_path is None or not overlay_path.exists():
        return out
    for row in load_jsonl(overlay_path):
        if row.get("source") != "concept_template":
            continue
        c = row.get("concept")
        cap = row.get("caption")
        if c is None or cap is None:
            continue
        out.setdefault(c, []).append(cap)
    return out


def extract_caption_to_h(av_model, tokenizer, captions: list[str], layer_ell: int,
                        device: str = "cuda") -> dict[str, torch.Tensor]:
    """Pre-compute h_ℓ for each unique caption. One forward pass per caption.

    Uses the AV's model (or any Gemma 4) as the TARGET. We extract the
    last-token hidden state at the chosen layer.
    """
    cache: dict[str, torch.Tensor] = {}
    av_model.eval()
    for cap in captions:
        if cap in cache:
            continue
        with torch.no_grad():
            enc = tokenizer(cap, return_tensors="pt", add_special_tokens=True).to(device)
            out = av_model(**enc, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[layer_ell][0, -1, :].detach().to("cpu").to(torch.float32).clone()
            cache[cap] = h
    return cache


def render_probe(av: StrokeDecoder, captions: list[tuple[str, str]], layer_ell: int,
                 h_lookup: dict[str, torch.Tensor], out_dir: Path,
                 max_tokens: int = 300, alpha_for_no_projector: float = 0.5) -> None:
    """Render probe drawings to PNG (no MP4 to keep gate-check fast)."""
    from render import render as stroke_render
    out_dir.mkdir(parents=True, exist_ok=True)
    av.model.eval()
    for slug, caption in captions:
        h = h_lookup.get(caption)
        if h is None:
            print(f"[probe] WARN: no cached h for {caption!r}; extracting now")
            enc = av.tokenizer(caption, return_tensors="pt", add_special_tokens=True).to(av.device())
            with torch.no_grad():
                out = av.model(**enc, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[layer_ell][0, -1, :].detach().to("cpu").to(torch.float32).clone()
        gen_ids = av.generate_from_activation(
            h, layer_ell=layer_ell,
            alpha=alpha_for_no_projector,
            max_new_tokens=max_tokens, temperature=1.0,
        )
        strokes, _malformed = av.vocab.decode_tokens_with_stats(gen_ids.tolist())
        png_path = out_dir / f"{slug}.png"
        img = stroke_render(strokes)
        img.save(png_path)
        png_path_4x = out_dir / f"{slug}_4x.png"
        img_4x = stroke_render(strokes, display_scale=4.0)
        img_4x.save(png_path_4x)
        print(f"[probe] {slug}: strokes={len(strokes)} → {png_path.name}")
    av.model.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--av-init-ckpt", type=Path, default=Path("checkpoints/av_sft/final"))
    p.add_argument("--data", type=Path, default=Path("data/sft_quickdraw.jsonl"))
    p.add_argument("--captions-overlay", type=Path, default=Path("data/expanded_captions.jsonl"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--projector-lr", type=float, default=5e-5)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--vocab-lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-first-n-layers", type=int, default=8,
                   help="Number of language layers to attach LoRA to. v1.4 uses 24 (all).")
    p.add_argument("--cosine-decay", action="store_true",
                   help="Cosine LR decay from peak to peak*0.1 over training")
    p.add_argument("--projector-alpha-init", type=float, default=0.5)
    p.add_argument("--max-seq-len", type=int, default=800)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--probe-at", type=int, nargs="*", default=[500, 1500, 3000])
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--held-out-concepts", nargs="*",
                   default=[],  # caller can pass e.g. cat dog if we want to hold out training concepts
                   help="Concepts to EXCLUDE from training (e.g. for generalisation eval)")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train.jsonl"
    log_f = open(log_path, "a")

    # ----- Load corpus -----
    print(f"[s15] loading corpus {args.data}", flush=True)
    corpus = load_jsonl(args.data)
    # Group by concept (extracted from "a drawing of a {X}")
    def concept_of(cap: str) -> str:
        # Match the build_expanded_corpus pattern: "a drawing of a/an X" or "...{X}s"
        # We use the suffix as the concept identifier.
        words = cap.replace("a drawing of an ", "").replace("a drawing of a ", "").replace("a drawing of ", "").rstrip("s")
        return words.strip()
    by_concept: dict[str, list[list[int]]] = {}
    skipped_long = 0
    print("[s15] building AV (frozen backbone, projector + LoRA trainable) ...", flush=True)
    if args.av_init_ckpt and args.av_init_ckpt.exists():
        print(f"[s15]   from ckpt {args.av_init_ckpt} (vocab rows preserved); add projector + LoRA fresh", flush=True)
        # Load Stage-1 ckpt (has new-vocab rows, no projector). Then add projector + LoRA fresh.
        av = StrokeDecoder.from_ckpt(args.av_init_ckpt, model_id=args.model_id)
        # Add projector if not present
        if av.act_projector is None:
            d = av.model.config.text_config.hidden_size if hasattr(av.model.config, "text_config") else av.model.config.hidden_size
            av.act_projector = ActProjector(d=d, alpha_init=args.projector_alpha_init,
                                             dtype=torch.bfloat16, device=av.device())
    else:
        print(f"[s15]   ckpt not found at {args.av_init_ckpt}; from_pretrained_and_extend", flush=True)
        av = StrokeDecoder.from_pretrained_and_extend(
            args.model_id, device="cuda", dtype=torch.bfloat16,
            use_projector=True, projector_alpha_init=args.projector_alpha_init,
        )

    # Tokenise stroke targets (needs decoder.vocab to exist first)
    print("[s15] pre-tokenising stroke targets ...", flush=True)
    for ex in corpus:
        cap = ex["caption"]
        c = concept_of(cap)
        if c in args.held_out_concepts:
            continue
        ids = build_target_stroke_ids(av.vocab, ex["strokes"])
        if len(ids) > args.max_seq_len:
            skipped_long += 1
            continue
        by_concept.setdefault(c, []).append(ids)
    n_concepts = len(by_concept)
    n_total = sum(len(v) for v in by_concept.values())
    print(f"[s15] {n_concepts} concepts, {n_total} drawings (skipped {skipped_long} too-long); held out: {args.held_out_concepts}", flush=True)

    # ----- Caption templates (overlay) -----
    concept_to_templates = build_concept_to_templates(args.captions_overlay)
    print(f"[s15] caption templates loaded for {len(concept_to_templates)} concepts", flush=True)

    # Build the union set of captions we'll need activations for: for each concept
    # in by_concept, include the base "a drawing of a/an X" caption + all overlay templates.
    needed_captions: set[str] = set()
    for c in by_concept:
        # Base caption(s) from training corpus
        base_caps = set()
        # Find original base caption from the corpus
        for ex in corpus:
            if concept_of(ex["caption"]) == c:
                base_caps.add(ex["caption"])
                break  # one base caption per concept is enough
        needed_captions.update(base_caps)
        # Templates from overlay
        needed_captions.update(concept_to_templates.get(c, []))
    captions_list = sorted(needed_captions)
    print(f"[s15] will pre-compute h for {len(captions_list)} unique captions (concepts × templates)", flush=True)

    # ----- Pre-extract activations (frozen target = av.model itself) -----
    t_extract = time.time()
    h_lookup = extract_caption_to_h(av.model, av.tokenizer, captions_list, args.layer, device=str(av.device()))
    print(f"[s15] cached {len(h_lookup)} activations in {time.time() - t_extract:.1f}s; mean h-norm={sum(h.norm().item() for h in h_lookup.values()) / max(1, len(h_lookup)):.2f}", flush=True)

    # ----- Attach AV LoRA -----
    # Count existing LoRA modules (from ckpt). If user asked for MORE layers
    # than the ckpt has, attach additional LoRA on the new layers (the
    # walker's seen-set + add_module replace makes this idempotent: existing
    # modules won't get re-attached because of the seen check, BUT
    # add_module on a name that already exists would clobber. So we
    # explicitly skip layers that already have _lora attached.)
    from ar.lora_gemma4 import _attach_lora_walk
    existing_layer_idxs = set()
    for name, module in av.model.named_modules():
        if hasattr(module, "_lora"):
            import re
            m = re.search(r"\.layers\.(\d+)\.", name)
            if m:
                existing_layer_idxs.add(int(m.group(1)))
    print(f"[s15] {len(existing_layer_idxs)} layers have LoRA from ckpt: {sorted(existing_layer_idxs)}", flush=True)
    if max(existing_layer_idxs, default=-1) + 1 < args.lora_first_n_layers:
        # Attach LoRA on new (uncovered) language layers
        new_first_n = args.lora_first_n_layers
        print(f"[s15] expanding AV LoRA from {len(existing_layer_idxs)} → {new_first_n} language layers ...", flush=True)
        attach_lora_to_av(
            av, first_n_layers=new_first_n,
            rank=args.lora_rank, alpha=args.lora_alpha, verbose=True,
        )
    elif not existing_layer_idxs:
        print(f"[s15] attaching AV LoRA on first {args.lora_first_n_layers} language layers ...", flush=True)
        attach_lora_to_av(
            av, first_n_layers=args.lora_first_n_layers,
            rank=args.lora_rank, alpha=args.lora_alpha, verbose=True,
        )
    av_lora_modules = list(lora_param_iter(av))

    # ----- Set requires_grad -----
    for p_ in av.model.parameters():
        p_.requires_grad = False
    embed = av.model.get_input_embeddings()
    old_vocab = embed.weight.shape[0] - len(av.vocab.name_to_id)
    embed.weight.requires_grad = True
    # Mask grad for old vocab rows
    def mask_old_embed_grad(grad):
        out = grad.clone()
        out[:old_vocab] = 0
        return out
    embed.weight.register_hook(mask_old_embed_grad)

    # Projector params
    for p_ in av.act_projector.parameters():
        p_.requires_grad = True
    # LoRA params
    lora_params = list(lora_param_iter(av))
    for p_ in lora_params:
        p_.requires_grad = True

    # ----- Optimiser -----
    param_groups = [
        {"params": [embed.weight], "lr": args.vocab_lr, "name": "vocab"},
        {"params": list(av.act_projector.parameters()), "lr": args.projector_lr, "name": "projector"},
        {"params": lora_params, "lr": args.lora_lr, "name": "lora"},
    ]
    optim = torch.optim.AdamW(param_groups, weight_decay=0.0, betas=(0.9, 0.95))

    n_train = sum(p.numel() for g in param_groups for p in g["params"] if p.requires_grad)
    print(f"[s15] trainable params: {n_train / 1e6:.2f}M  (vocab + projector + LoRA)", flush=True)

    # ----- Probe captions (use HELD-OUT phrasings: prompts NOT in training overlay) -----
    PROBE_CAPTIONS = [
        ("dog", "I am thinking about a dog."),
        ("cat", "I am thinking about a cat."),
        ("eiffel", "Paris, the city of lights, is famous for the Eiffel"),
        ("triangle", "Imagine a triangle inscribed in a circle."),
        ("capital_france", "The capital of France is"),
        ("smile_face", "I am picturing a smiling face."),
    ]
    # Pre-extract activations for probes too (held-out test of generalisation)
    probe_caps = [c for _, c in PROBE_CAPTIONS]
    print(f"[s15] pre-computing {len(probe_caps)} probe activations ...", flush=True)
    probe_h_lookup = extract_caption_to_h(av.model, av.tokenizer, probe_caps, args.layer, device=str(av.device()))
    h_lookup.update(probe_h_lookup)  # merge so render_probe can find them

    # ----- Training loop -----
    av.model.train()
    av.act_projector.train()
    concepts_list = sorted(by_concept.keys())

    t_start = time.time()
    last_log = time.time()

    def sample_batch():
        """Sample B examples: random concept, random drawing, random caption template."""
        items = []
        for _ in range(args.batch):
            c = random.choice(concepts_list)
            drawings = by_concept[c]
            ids = random.choice(drawings)
            # Sample caption: 50/50 base caption vs overlay template
            templates = concept_to_templates.get(c, [])
            base_cap = f"a drawing of a {c}" if not c.startswith(("a", "e", "i", "o", "u")) else f"a drawing of an {c}"
            if templates and random.random() < 0.75:
                cap = random.choice(templates)
            else:
                cap = base_cap
            if cap not in h_lookup:
                # Fall back: skip if no h cached
                cap = next(iter(c for c in h_lookup.keys() if concept_of(c) == c), base_cap)
                if cap not in h_lookup:
                    continue
            items.append((h_lookup[cap], ids))
        return items

    for step in range(args.steps):
        items = sample_batch()
        if not items:
            continue
        hs = torch.stack([h for h, _ in items], dim=0).to(av.device())
        targets = [t for _, t in items]
        try:
            loss = av.act_sft_loss_batched(hs, targets, layer_ell=args.layer)
            (loss / args.grad_accum).backward()
        except torch.cuda.OutOfMemoryError:
            print(f"[s15] OOM at step {step}; skipping", flush=True)
            optim.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        if (step + 1) % args.grad_accum == 0:
            # LR schedule: warmup then optional cosine decay
            import math
            if step < args.warmup_steps:
                lr_mult = (step + 1) / max(1, args.warmup_steps)
            elif args.cosine_decay:
                progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
                lr_mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            else:
                lr_mult = 1.0
            for g in optim.param_groups:
                if "base_lr" not in g:
                    g["base_lr"] = g["lr"]  # capture original
                g["lr"] = g["base_lr"] * lr_mult
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"] if p.requires_grad],
                max_norm=1.0,
            )
            optim.step()
            optim.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            now = time.time()
            msg = {
                "step": step,
                "loss": float(loss.item()),
                "elapsed_sec": round(now - t_start, 1),
                "dt_per_step": round((now - last_log) / max(1, args.log_every), 2),
            }
            log_f.write(json.dumps(msg) + "\n")
            log_f.flush()
            print(f"[s15] step {step:6d}  loss={msg['loss']:.4f}  ({msg['elapsed_sec']:.0f}s, {msg['dt_per_step']:.2f}s/step)", flush=True)
            last_log = now

        if step in args.probe_at:
            probe_dir = args.out_dir / f"probe_step_{step:06d}"
            print(f"[s15] probe at step {step} → {probe_dir}", flush=True)
            render_probe(av, PROBE_CAPTIONS, args.layer, h_lookup, probe_dir,
                        alpha_for_no_projector=args.projector_alpha_init)

        if step > 0 and step % args.save_every == 0:
            save_dir = args.out_dir / f"step_{step:06d}"
            save_dir.mkdir(parents=True, exist_ok=True)
            lora_meta = {"first_n_layers": args.lora_first_n_layers,
                         "rank": args.lora_rank, "alpha": args.lora_alpha}
            av.save_ckpt(save_dir, include_lora_meta=lora_meta)
            print(f"[s15] saved → {save_dir}", flush=True)

    # Final save + probe
    final_dir = args.out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    lora_meta = {"first_n_layers": args.lora_first_n_layers,
                 "rank": args.lora_rank, "alpha": args.lora_alpha}
    av.save_ckpt(final_dir, include_lora_meta=lora_meta)
    final_probe = args.out_dir / "probe_final"
    render_probe(av, PROBE_CAPTIONS, args.layer, h_lookup, final_probe,
                alpha_for_no_projector=args.projector_alpha_init)
    log_f.close()
    print(f"[s15] DONE → {final_dir}", flush=True)


if __name__ == "__main__":
    main()
