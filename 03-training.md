# 03 — Training pipeline

Four stages. Each stage produces an artefact useful in isolation, so we can stop after any stage with a meaningful result. This matters because compute is uncertain.

## Pipeline overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  STAGE 0: data preparation                              (CPU only, hours)│
│  ──────────────────────────                                              │
│  (a) Build stroke tokenizer                                              │
│  (b) Download IAM On-Line + QuickDraw → 150K (caption, strokes) pairs    │
│  (c) Run Gemma 4 E2B forward on ~1M text snippets, save                  │
│      (text, layer_ℓ_activation) for every layer ℓ in sweep set           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  STAGE 1: AV warm-start (SFT on text→drawing)            (1-3 GPU-hours) │
│  ─────────────────────────────────────────────                           │
│  Train AV with NO activation injection.                                  │
│  Loss: CE on stroke tokens given text caption prompt.                    │
│  Result: AV that can draw concepts when given text prompts               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  STAGE 2: AR supervised training                         (1-3 GPU-hours) │
│  ───────────────────────────────                                         │
│  Use Stage-1 AV (frozen) to draw concepts. Render. Train AR to           │
│  reconstruct h_ℓ from the rendered drawing.                              │
│  Loss: MSE(h_ℓ, ĥ_ℓ)                                                     │
│  Result: AR that maps drawings → activations                             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  STAGE 3: AV RL with reconstruction reward (GRPO)        (8-100 GPU-hrs) │
│  ────────────────────────────────────────────                            │
│  Now turn on activation injection.                                       │
│  Sample drawings from AV. Render. Reconstruct via AR. Compute reward.    │
│  GRPO update on AV. KL penalty to AV_SFT to prevent drift.               │
│  Result: AV produces drawings FAITHFUL to specific activations,          │
│          not just concept depictions                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  STAGE 4: evaluation                                       (1 GPU-hour)  │
│  ──────────────────                                                      │
│  Quantitative: FVE on held-out activations                               │
│  Qualitative:  layer-trajectory sweep on 50 interesting prompts          │
│  Baseline:     compare against text-NLA on same prompts                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 0 — Data preparation (CPU, several hours)

### (a) Stroke tokenizer

Pure Python module. ~50 lines. Functions:
- `encode_strokes(stroke_list) -> list[int]` (token ids)
- `decode_tokens(token_ids) -> list[stroke]`
- `bin_offset(value) -> int`
- `unbin_offset(bin) -> int`

### (b) SFT corpus

Download:
- **IAM On-Line Handwriting Database** (~12K sentences with stylus traces). `fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database`. Login-walled, request access ahead of time.
- **QuickDraw** (50M sketches, 345 categories). `github.com/googlecreativelab/quickdraw-dataset`. CC-BY-4.0. Use a subsample of ~100K drawings across categories.

Process each into `(text_caption, stroke_token_sequence)`:
- IAM: caption = the sentence; strokes = the timestamped pen trace, downsampled and normalised.
- QuickDraw: caption = `"draw a {category}"` (or richer GPT-generated captions if we want); strokes = the recorded stroke-3 sequence.

Output: a single jsonl file, ~150K samples.

### (c) Activation corpus

Choose layers to sweep. Default: `{4, 8, 12, 16, 20, 24}` for E2B (6 anchor layers).

Pull ~1M text snippets from FineWeb / The Pile. For each snippet:
1. Run Gemma 4 E2B forward (text-only).
2. Save the residual stream at each layer in the sweep set, at the final-token position.
3. Optionally save activations at multiple token positions (e.g., every 10th) for richer dataset.

Output:
- For per-layer training (option (a) in `02-architecture.md`): one shard per layer, each containing `(text_id, h_ℓ)` pairs.
- For layer-conditioned training (option (b)): one shard total, containing `(text_id, ℓ, h_ℓ)` triples.

Storage: ~5-10 GB total, depending on `d` and number of layers.

---

## Stage 1 — AV warm-start (1-3 GPU-hours)

**Goal: AV learns to draw concepts when prompted with text. No activation injection yet.**

```
for (caption, strokes) in sft_corpus:
    prompt    = "Visualize: " + caption + " <DRAW>"
    target    = strokes + [</DRAW>]
    loss      = cross_entropy(AV.forward(prompt + target), target)
    backprop into AV's new-vocab embed/lm_head rows + LoRA
```

Hyperparameters:
- LoRA rank: 16, applied to all attention layers.
- Optimizer: AdamW, lr 1e-4 for LoRA, 5e-4 for new embeddings.
- Batch size: 64-128 (large, H200 has the memory).
- Epochs: 3-5 over the 150K-sample corpus.

Result: AV can produce a recognisable cat when asked "draw a cat".

This stage is essentially "train a small text-to-sketch model". It biases the AV toward human-interpretable drawings so that Stage 3's RL doesn't find an "incomprehensible code" that fools the AR but reads as gibberish to humans.

---

## Stage 2 — AR supervised training (1-3 GPU-hours)

**Goal: AR learns to project rendered drawings back into activation space.**

```
for (text, h_ℓ, ℓ) in activation_corpus:
    summary  = summarise(text)             # cheap summarizer (Gemma 4 itself in text mode)
    drawing  = AV.generate(summary)        # text-conditioned, no inject
    png      = render(drawing)             # see 04-renderer.md
    ĥ_ℓ      = AR(png, ℓ)                  # truncated to ℓ, output through Linear(d,d)
    loss     = MSE(h_ℓ, ĥ_ℓ)
    backprop into AR's LoRA + Linear(d,d) only
```

The AV is frozen at the Stage 1 checkpoint. We're calibrating the AR against the AV's drawing distribution.

Hyperparameters:
- LoRA rank: 8, applied to truncated layers.
- Optimizer: AdamW, lr 1e-4.
- Batch size: 32-64 (vision encoder adds memory).
- Epochs: 2-3 over the activation corpus.

Result: AR can predict what activation a drawing would have produced if it had been the corresponding text.

---

## Stage 3 — AV RL with reconstruction reward (8-100+ GPU-hours)

**Goal: AV produces drawings faithful to *specific activations*, not generic concept depictions.**

This is the big one. RL with GRPO, as in NLA.

```
for (text, h_ℓ, ℓ) in activation_corpus:
    inject h_ℓ at <ACT_TOKEN> in AV's prompt
    
    sample G drawings from AV (group size G = 8 for GRPO)
    
    for each drawing in group:
        png       = render(drawing)
        ĥ_ℓ        = AR(png, ℓ)
        reward    = -log ‖h_ℓ - ĥ_ℓ‖²
    
    advantage  = (reward - group_mean(reward)) / group_std(reward)
    GRPO loss  = -advantage * log_prob(drawing) + β * KL(AV || AV_SFT_init)
    backprop into AV's trainable params
    
    # in parallel, AR continues training:
    AR_loss    = MSE(h_ℓ, AR(best_drawing_in_group, ℓ))
    backprop into AR
```

Hyperparameters (from NLA paper):
- GRPO group size: 8
- KL coefficient β: 0.01-0.1, tuned
- Reward: `-log ||h - ĥ||²` (log transform for numerical stability)
- Learning rate: lower than SFT, ~1e-5 for AV's params

**This is the most expensive stage by far.** NLA paper used 2×8=16 H100s for the RL phase. On 2× H200 we can do a mini-version with smaller batches but fewer total updates. Full reproduction is out of scope without more compute.

Result: faithful AV. The drawings now contain real information about the specific activation, not just a concept depiction.

---

## Stage 4 — Evaluation (1 GPU-hour)

See `05-evaluation.md` for the full eval protocol.

Outputs:
- FVE numbers per layer
- Layer-trajectory PNGs for 50 prompts
- Side-by-side comparison vs text NLA baseline
- Failure mode catalogue

---

## Compute estimates per stage

For one anchor layer (multiply by N for full per-layer sweep on Option A; ~1.5-2× for Option B hybrid):

| Stage | Min compute (v0) | Full compute (publishable) |
|---|---|---|
| 0. Data prep | CPU, 4-6 hrs | same |
| 1. AV SFT | 1-2 GPU-hrs on 1× H200 | 4-8 GPU-hrs |
| 2. AR supervised | 1-2 GPU-hrs on 1× H200 | 4-8 GPU-hrs |
| 3. AV GRPO | 8-16 GPU-hrs on 2× H200 | 24-48 GPU-hrs on 2× H200 |
| 4. Eval | 1 GPU-hr | same |

**For reference: Anthropic's NLA on Qwen-7B used 2× H100 (SFT) + 16× H100 (RL). Our Gemma 4 E2B is smaller, so per-layer total is ~10-30 H200-hours for a meaningful run, ~50-100 for high-fidelity.**

### Budget tiers

**3-hour budget (one anchor layer, current 2× H200):**
Day-0 sanity check (1 hr) + Stages 0 prep (done off-GPU) + Stage 1 (2 hrs). Exit with a text-to-stroke decoder, no faithfulness training. Useful sanity-check; not the publishable artefact.

**24-hour budget (one anchor layer):**
Day-0 (1) + Stage 1 (4) + Stage 2 (4) + Stage 3 mini-run (12) + Stage 4 (1). Full pipeline at one layer. Expected FVE 0.3-0.5. Single-layer visual lens, no sweep.

**3-day budget on 2× H200 (~150 GPU-hours):**
- **Option A (full per-layer):** 6 anchor layers × ~20 GPU-hrs each = ~120 GPU-hrs. Tight but doable. Layer sweep + faithfulness for the headline trajectory artefact.
- **Option B (hybrid):** ~40-60 GPU-hrs total (1 shared AV training + 6 adapters + 6 ARs). Buffer left over for re-running noisy layers.

**Week of 4-8× H200 (~600-1300 GPU-hours):**
- **Option A** at high fidelity. Multiple training runs per layer for hyperparameter sweep. Expected FVE 0.5-0.7. The full publishable artefact.
- Time for the text-NLA baseline as well, running fully independently.

### Recommended target

**Option A + 3-day budget on 2-4× H200** gets us the full layer-sweep visual artefact at meaningful (not best-in-class) FVE. That's the realistic publishable v1.

**Option B is the contingency** if compute tightens.
