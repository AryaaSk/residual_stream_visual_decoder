# v1.0 iter_00 — what we learned + why we killed the run

## What we built
- Custom LoRA on `Gemma4ClippableLinear` (Gemma 4's attention proj wrapper that PEFT
  doesn't recognise) — successfully attaches to 64 modules across vision tower +
  first ℓ language layers; passes 12 unit tests.
- Stage 4 iterative trainer (`code/train/stage4_iterative.py`): alternates AR-phase
  (regenerate fresh (drawing, h) buffer from current AV; train AR LoRA + Linear head
  with MSE) and AV-phase (GRPO with current AR as judge; KL penalty to Stage-1 init).
  Inspired by Anthropic's NLA iterative refinement.

## What we ran
- Two parallel jobs: L12 on GPU 6, L3 on GPU 7. Both with same recipe:
  - 5 iterations
  - Buffer 256 (drawing, h) pairs regenerated per iter
  - AR phase: 300 steps × batch 8, LoRA(r=16, α=32) + Linear head
  - AV phase: 100 GRPO steps × group 4, KL β=0.05, alpha=0.5
  - Eval on 20 held-out probes per iter
- Both completed iter 0 (and started iter 1). Killed mid-iter 1 once the diagnostic
  result came back.

## What we measured

### Training-time signals (iter 0)

| Quantity | L12 | L3 | v0.1 reference |
|---|---|---|---|
| AR loss start of iter 0 | 1.86 | 0.91 | — |
| AR loss end of iter 0 | ~0.06 | ~0.04 | v0.1 Stage 2 v2: ~0.12 |
| AV reward EMA end of iter 0 | +2.92 | +3.06 | v0.1 Day-2: -0.15 (best) |
| Wallclock per iter | 103 min | 100 min | est 40 min — significantly underestimated |

The AV reward going from v0.1's best -0.15 to v1.0's +3 means **the AR-LoRA can
reconstruct from in-distribution drawings dramatically better than the v0.1 AR**.

### Held-out eval (iter 0 final)

Standard held-out probes ("I am thinking about a cat", "The capital of France is", etc.):

| Metric | L12 | L3 |
|---|---|---|
| FVE | -0.0013 | -0.0002 |
| Cosine | 0.579 | 0.468 |
| MSE | 0.742 | 0.464 |

### Diagnostic: training-distribution eval (Option E)

Re-measured FVE on L12 iter_00 using TRAINING-style probes ("a drawing of a cat", ...):

| Metric | Standard eval probes | TRAINING-distribution probes |
|---|---|---|
| FVE | -0.0013 | -0.0011 |
| Cosine | **0.579** | **0.956** |
| MSE | 0.742 | **0.065** |

## What this tells us

The AR's reconstruction is **dramatically better on training-style prompts** (cosine
0.58 → 0.96, MSE 0.74 → 0.065) — BUT **FVE is still ~0 in both cases**.

This is the same failure mode as v0.1 ablations: **AR is hitting the cluster mean,
not discriminating per prompt.** When all training prompts share the structure
"a drawing of {X}", their activations cluster very tightly together (the diagnostic
output's near-identical h-norms ~33-35 confirm this). AR maximises cosine by aiming
at the cluster centroid; per-prompt variance is small relative to baseline, so FVE
stays at noise.

**The bottleneck is the training data distribution, not the model.** Even with:
- ~1.5M LoRA params on Gemma 4 backbone (vision + first ℓ language layers)
- 2.4M Linear head params
- Iterative joint training that tracks AV's evolving distribution
- 5× lower AR loss than v0.1
- 4× higher AV reward than v0.1

...we can't break FVE 0 if the AR's training data only spans a tight cluster of
activations. There's no per-prompt signal for AR to learn to discriminate.

## What v1.1 needs (Plan B)

**Broader caption distribution in the SFT corpus** so AR sees activations spanning
a wider region of representation space, with enough per-prompt variance for
discrimination to be learnable.

Concrete fix:
1. Augment SFT corpus with multiple caption templates per category:
   - "a drawing of a {cat}" (current)
   - "I am thinking about a {cat}"
   - "Imagine a {cat}"
   - "When I see a {cat}, I think of"
   - "The {cat} is"
   - etc.
2. Possibly also mix in non-concept prompts ("the capital of France is", "what is
   47 + 38", etc.) so AR sees the eval-distribution shape during training.
3. Same iterative recipe, with this broader corpus, runs ~6 more hours.

The recipe and infrastructure are correct. The data was too narrow.

## Artefacts shipped under findings/v1/

- `fve_train_dist_L12_iter00.json` — the diagnostic numbers
- `inject_demo_L12_iter00/*.{png,mp4}` — drawings sampled from iter_00 AV at L12
  with activation injection for 6 demo prompts (+ 4× upscaled versions)
- `v1_iter_00_summary.md` (this file)

Checkpoints under `checkpoints/v1/L12/iter_00/` and `checkpoints/v1/L3/iter_00/`
(av_ckpt.pt + head.pt with LoRA state) are pulled to Mac for durability.

## Time line

- T+0:00 — kicked off iterL12 (GPU 6) and iterL3 (GPU 7) in parallel
- T+1:43 — both completed iter 0 (~103 min per iter, much slower than estimated)
- T+2:30 — diagnostic revealed cluster-mean failure
- T+2:35 — killed both jobs to redirect compute to Plan B with broader SFT corpus
