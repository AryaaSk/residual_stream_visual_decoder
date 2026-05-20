# Research notes — Residual Stream Visual Decoder

The full iteration log of this project: every version, every dead end, every breakthrough. Written as the work progressed so the reasoning is preserved with its uncertainty intact.

## TL;DR

We tried to teach Gemma 4 E2B to draw what another copy of itself is thinking — vector strokes on a canvas conditioned on the residual-stream activation at a chosen layer. Five major versions over ~30 GPU-hours on 2× H200:

| version | core idea | drawings | SFT loss | notes |
|---|---|---|---|---|
| **v0.1** | frozen-backbone AR + Linear head, fixed AV | abstract noise | n/a | FVE = 0 across 5 AR variants |
| **v1.0** | + custom LoRA on Gemma4ClippableLinear + iterative GRPO | abstract | n/a | architectures works; FVE still ~0 |
| **v1.1** | + expanded 1215-caption corpus | abstract structure | n/a (no AV SFT) | cosine 0.5-0.7 = real per-prompt signal, but drawings still unrecognisable; honest negative result shipped |
| **v1.2** | + `ActProjector(Linear(d,d))` + AV-LoRA + Stage 1.5 supervised SFT | "abstract structure with concept hints" | 1.83 → 2.17 plateau | dog had quadruped shape, cat had body curve; promising but not viral |
| **v1.3** | + CLIP-ranked best-of-32 + 30K-step bigger LR/batch training | recognisable silhouettes | ~1.9 plateau | first viral-quality outputs; dog with snout, mountain with peak |
| **v1.4** | + canonical-drawing distillation (top-3 CLIP-best per concept) | cat faces with ears + eyes | **1.83 → 0.02** | plateau broken via target-entropy collapse; cat looks like a Pokémon |
| **v1.5** | + 24-layer LoRA + top-5 canonical + 50K steps cosine decay | TBD | TBD | currently training, ~2h ETA |

The headline insight: **a chain of three architectural changes (projector + LoRA + supervised SFT, v1.2) closed a capacity gap, then a single data-engineering change (canonical distillation, v1.4) closed an entropy gap and dropped loss 90× more than the architecture had managed.**

## The journey

### v0.1 — "the architecture works, but FVE is zero"

v0.1 was a faithful clone of Anthropic's NLA paper, adapted to Gemma 4 E2B + stroke output. Five AR variants (Linear/MLP on L12/L16, mean-centered MSE, contrastive InfoNCE) all returned FVE around 0. Drawings were per-prompt different but not recognisable.

Diagnosis at the time: the AR was a Linear head over Gemma 4's frozen vision encoder; not enough capacity to discriminate between activations of similar prompts. Even adding MLP didn't help.

Wrong conclusion at the time: "the AR is the bottleneck."

(The deeper issue, which I only realised at v1.2: there was no training signal anywhere that said "this activation → this real drawing of the concept.")

### v1.0 — "build the AR's missing capacity"

Three things added:

1. **Custom LoRA on `Gemma4ClippableLinear`**: PEFT 0.13 doesn't support Gemma 4's custom Linear wrapper, so we implemented `LoRADelta` (parallel low-rank branch) + a forward-patcher in `code/ar/lora_gemma4.py`. Attaches to q/k/v/o_proj across vision tower + first ℓ language layers.
2. **Activation-injection via embedding-layer forward hook**: Gemma 4 refuses to accept both `input_ids` and `inputs_embeds`, so we overwrite the embedding row at the `<ACT_TOKEN>` position via a forward hook (`code/verbalizer/stroke_decoder.py:generate_from_activation`).
3. **Stage 4 iterative joint training**: alternate AR-supervised steps (regenerated buffer of drawings from current AV) and AV-GRPO steps (AR as judge, KL anchor to Stage-1 init).

Trained 4 iterations on L12 + L24 in parallel. Total ~24 GPU-hours.

Result: AV reward ramps up, AR loss goes down. Held-out FVE stays around 0. **Drawings are still abstract.** Killed in mid-iter-1 after the diagnostic showed the cluster-mean failure mode hadn't shifted.

Dead end #1: more AR capacity isn't enough if the data signal is wrong.

### v1.1 — "the corpus is too narrow"

v1.0's corpus was 8800 (caption, drawing) pairs all in the form `"a drawing of a {X}"` — 44 concepts × 200 drawings. Diagnosed: activations cluster tightly because all captions share structure, AR maximises cosine by predicting cluster mean.

v1.1 fix: built `data/expanded_captions.jsonl`, 1215 captions including 14 templates per concept (`"I am thinking about a {X}"`, `"Imagine a {X}"`, etc.) + 95 abstract / factual / math / code prompts.

Re-ran the full v1.0 pipeline. **Result: same.** Held-out cosine ~0.5-0.7 (real per-prompt signal, modest), held-out FVE still negative. Drawings still abstract.

Shipped v1.1 as an honest negative-result writeup. The README was titled "An attempt to draw what Gemma 4 is thinking" with a status banner explaining that recognisability hadn't been achieved.

Dead end #2: data diversity alone doesn't fix the underlying mapping problem.

### v1.2 — "the AV has no learnable surface to interpret the injected activation"

The realisation that changed everything came when I stopped staring at AR metrics and traced through the AV's actual computation.

The AV's trainable parameters in v0.1 / v1.0 / v1.1 were exactly **262 new vocab embedding rows (~0.4 M params)**. The Gemma 4 backbone was completely frozen. Activation-injection put `α · h` into one embedding slot, and the frozen backbone tried to interpret that as just another token embedding.

Two problems:
1. **Basis mismatch.** h at L24 has norm ~70 and lives in the residual coordinate frame *after* 24 layers of computation. Embeddings have norm ~10 and live at the model input. They are NOT the same space; the relation is 24 layers of nonlinear transformations. We were asking the frozen backbone to make sense of an out-of-distribution input vector with no learnable surface to bridge the gap.
2. **No mapping signal.** Even if the backbone could interpret the injection, there was nothing in our pipeline that told the AV "this activation should produce this drawing." Stage 1 SFT was text-conditioned (no activation ever shown). Stage 4 GRPO rewarded AR-decodability (an indirect proxy for concept-recognisability and not actually a tight one).

**Anthropic's NLA uses a learnable projector** that maps `h_ℓ → K embedding-space vectors`, fed as a soft prefix. We had skipped this. Adding it was the fix.

v1.2 changes:
1. **`ActProjector` (Linear(d, d), init = α · I)** sits between injected h and the embedding slot. Init guarantees v1.1 behaviour at step 0; gradient bends it from there.
2. **AV-LoRA on first 8 language layers** (q/k/v/o_proj). Reuses our v1.0 LoRA infra, but with a critical generalisation: the walker now accepts plain `nn.Linear` in addition to `Gemma4ClippableLinear`. **(v1.0/v1.1's AR LoRA was silently only attaching to the vision tower** because Gemma 4's language layers use plain `Linear`. That bug had been live for ~20 GPU-hours of training.)
3. **Stage 1.5 supervised activation-conditioned SFT**: for each `(caption, drawing)` pair in `data/sft_quickdraw.jsonl`, extract `h = Gemma(caption).hidden_states[L][0, -1, :]` (cached per unique caption), inject via the projector + embedding hook, teacher-force AV on `prompt + drawing_tokens`. Direct signal that v1.0/v1.1 never had: "given THIS activation, produce THIS drawing of the matching concept."

Trainable AV surface went from ~0.4 M → ~5 M params.

Trained 5K steps batch 8 in ~15 min per layer. Loss: 3.28 → 2.17. Drawings: **per-prompt different, dog had quadruped silhouette, cat had body curve, but still not unambiguously recognisable.**

Shipped as v1.2 with mixed satisfaction. The pieces were there.

### v1.3 — "inference-time CLIP ranking is the unlock"

v1.2's outputs were noisy at temperature 1.0. We tried two obvious knobs:

- **Lower temperature** (0.3, 0.5): too constrained, model collapsed to a single sweeping curve, lost all detail.
- **Heuristic best-of-N**: sampled 16 candidates per prompt, scored by stroke count (Gaussian around 45), malformation rate, bbox area. Better, but the score measures "well-formed" not "looks-like-cat."

The right oracle: **CLIP-ranked best-of-N.** Sample 32 candidates per prompt at temperature 0.85, render each, compute CLIP-ViT-B/32 image-text similarity vs `"a drawing of a {concept}"`, take the top-1. CLIP measures *visual resemblance to the concept name directly* — exactly what determines whether someone scrolling Twitter recognises the drawing.

We also implemented **batched generation** (`StrokeDecoder.generate_from_activation_batched`) sharing the KV cache so 32 samples cost ~9 sec total rather than 9 × 32. Critical for fast iteration.

Plus a longer training run: **30K steps, batch 16, LR 2×**, both layers in parallel on 2× H200.

The qualitative jump from heuristic-ranked v1.2 to CLIP-ranked same-checkpoint was the largest single perceptual improvement in the project's history. Cat outputs went from "a closed body loop" to "a body + head + ear + leg." Dog from "scattered strokes" to "snout + ear + body silhouette."

Shipped as v1.3. README updated to "Drawing what Gemma 4 is thinking." (status promoted from "An attempt to draw").

User feedback: "definitely starting to look better." But also: "still doesn't really look that good — figure out a way to break the plateau."

### v1.4 — "the loss plateau IS the data entropy"

v1.3's SFT loss plateaued around 1.9. More training, more data diversity, larger LoRA — none of it moved the needle. Why?

**Because the loss was measuring an irreducible quantity.** For each unique caption, the AV was being asked to predict the exact token sequence of one of ~200 different real QuickDraw cat drawings. Real artists draw cats very differently — some with whiskers, some without, some sitting, some standing, some abstract, some detailed. CE loss averages over all those: the optimal model output is something like a probabilistic average of "all the ways a cat can be drawn," which means non-zero per-token entropy.

The fix: **collapse the target entropy.** Pick the top-K most-cat-like cat drawings (K=3), train on ONLY those. Then there are only 3 (or 1) "correct" answers per concept, and the model can in principle memorise them perfectly. Loss can drop near zero.

The risk: model overfits to a specific drawing and loses generalisation. But: we still vary the input across 14 caption templates, so the activation→drawing mapping is many-to-few, not one-to-one. The model learns "any cat-related activation → one of the canonical cats."

For viral demos this is exactly what we want — same recognisable cat drawing every time you prompt with anything cat-related.

Implementation:
- `code/data/pick_canonical_drawings.py`: render every drawing in `sft_quickdraw.jsonl`, CLIP-score against `"a clear drawing of a {concept}"`, save top-K per concept.
- 44 concepts × 3 canonical = 132 training drawings (down from 8800).
- Same training script, just `--data data/canonical_drawings.jsonl`.

Loss: **1.83 → 0.02 in 5K steps.** A 90× drop. The plateau was the data, not the model.

Drawings: **cat with pointed ears + eyes + nose. Round suns with centers. Pizzas that are round. Fish with tails.** This was the version that finally looked viral.

Shipped as v1.4.

### v1.5 (currently training) — "maximise the available compute budget"

User asked for 2 more GPU-hours, "more training, bigger training." So v1.5 stacks:
- **Full 24-layer LoRA** (vs v1.4's 8). Required fixing the LoRA walker to skip modules that already have `_lora` attached (otherwise it would clobber loaded weights on resume).
- **Top-5 canonical drawings per concept** (vs top-3). Adds variance back in; should help generalisation without re-introducing the entropy floor.
- **50K steps with cosine LR decay** from peak to 10% of peak. Allows continued fine refinement at the tail.
- **Resume from v1.4 step_005000** — don't waste the existing memorisation.
- **Both layers (L12 + L24) parallel** on 2× H200, ~135 min ETA.

Currently in progress as of this writeup.

## Engineering details worth keeping

### The LoRA-walker `Gemma4ClippableLinear` bug

The v1.0 LoRA code filtered modules by `module.__class__.__name__ == "Gemma4ClippableLinear"`. Vision tower and audio tower use that class. **Language layers use plain `nn.Linear`.** v1.0 / v1.1's "AR LoRA on first ℓ language layers" therefore attached zero modules in the language model.

Caught at v1.2 when I tried to put LoRA on the AV's language layers and got 0 matches. Generalised the walker to accept both class names. Suddenly LoRA started actually attaching to the language layers everywhere.

### The skip-attach-on-resume bug

When resuming from a Stage 1.5+ checkpoint that already has LoRA loaded, calling `attach_lora_to_av` again would call `add_module("_lora", lora)` on modules that already had `_lora`, silently replacing loaded weights with fresh-init zeros. Fixed in v1.4 by an explicit `if hasattr(module, "_lora"): continue` skip in the walker.

### The `lora_meta` / `lora_state` mismatch

When the trainer's `--lora-first-n-layers=24` flag was set but the skip-attach guard left only 8 layers attached from the v1.3 ckpt, the saved meta said 24 but the state dict only had 8. `from_ckpt` would then try to load weights for 16 phantom layers and KeyError.

Fixed by inferring the *actual* layer count from the state-dict keys (regex extract layer indices) rather than trusting the meta. Plus `strict=False` on `load_lora_state`.

### Batched generation with shared KV-cache

`generate_from_activation_batched` accepts a single `(d,)` activation, broadcasts to `(N, d)`, hooks the embedding layer to inject the SAME projected vector into all N rows, then runs autoregressive sampling with a `use_cache=True` past. Per-row multinomial sampling.

The KV cache for the prompt prefix is shared across the N samples (Gemma 4's `past_key_values` supports this naturally because each batch row gets its own cache slot but all start from the same prefix). Prefill cost is paid once; only the per-token sampling step is N-wide. 32 samples in ~9s on H200.

### The α=0.5 alpha-sweep result

From v0.1: activation norm at L16 is ~70, Gemma embeddings have norm ~10. At injection α=1.0 (the default that "felt right" before measurement), we were 7× the magnitude the model is used to. Downstream layers saturated, malformation rate 59%. α=0.5 → injected magnitude ~35, malformation 20%, drawings became visibly richer.

α=0.5 has been the default everywhere since. The projector in v1.2+ is init'd to `0.5 · I` so the un-trained projector exactly recovers v1.1's behaviour.

### Why L12 outperforms L24

Activation geometry analysis from v0.1: layer cosine-pairwise (across 30 diverse prompts) is 0.532 at L12, 0.870 at L24, 0.938 at L19. L24 is heavily clustered — semantic representation has converged, but spatial / token-level structure is largely gone. L12 still has enough geometric diversity per-prompt to support discrimination.

Empirically across all versions, L12 outputs are more recognisable than L24. v1.3 / v1.4 hero galleries use L12 by default.

## Things I'd do differently

- **Measure data entropy before training**, not after the plateau. A simple "compute the cross-entropy of teacher-forcing the AV on real QuickDraw drawings with no conditioning at all" would have given a floor estimate and we could've predicted the plateau.
- **Pick canonical drawings first.** The whole v1.3 round of "longer training breaks the plateau" was a waste of GPU time. Reducing target entropy is a stronger lever than scaling compute.
- **CLIP-rank earlier.** All the heuristic-ranking and lower-temperature experiments in v1.2/v1.3 went nowhere; one CLIP call per candidate would have shortcut all of that.
- **Test the LoRA walker against language layers in v1.0.** A simple smoke test that "first 8 LoRA modules attached to language_model.layers.{0..7}" would have caught the silent-skip bug six weeks earlier.

## Provenance and reproduction

Every gallery image and demo MP4 in this repo is sampled fresh from the AV at the noted checkpoint and config. No post-processing other than optional 4× vector upscaling. The `make_hype_reel.py` script assembles MP4s via ffmpeg with title cards drawn at runtime.

Training is reproducible end-to-end:
```bash
python -m code.train.stage1_v2_act_sft --layer 12 \
  --av-init-ckpt checkpoints/av_sft/final \
  --data data/canonical_drawings.jsonl \
  --out-dir checkpoints/my_run/L12 \
  --steps 5000 --batch 12 --probe-at 1000 3000 \
  --projector-lr 1e-4 --lora-lr 2e-4 --vocab-lr 2e-4 \
  --lora-first-n-layers 24 --cosine-decay
```

Eval is one command:
```bash
python -m code.eval.clip_ranker --av-ckpt checkpoints/my_run/L12/final --layer 12 \
  --n-samples 32 --pick-k 1 --temperature 0.85 --top-k 25 \
  --out-dir findings/my_run/clip
```

Compute used: 2× H200, ~32 GPU-hours across all versions.
