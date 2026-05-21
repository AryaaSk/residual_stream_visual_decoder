# Viral Video Plan — Final Push (overnight to 08:16 UTC)

**Read this first after any context compaction.** SSH key expires 2026-05-21 08:16 UTC; ~1.5 hours from when this was written. Plan: generate every possible viral asset NOW, pull to local, then compose the final video.

## What user asked for (verbatim, paraphrased)

> "generate a lot of stuff: lots of drawing animations (upscaled, relatively trivial), see how the drawings change through the layers for a single token inference, see how the layers morph across multiple token inference, think of cool things, infographics, graphs, explanations — render everything and download locally ASAP"

## Trained layer ckpts available

Located under `checkpoints/overnight/` and `checkpoints/v2_0/`:

| layer | best variant                      | top-1 |
|------:|-----------------------------------|------:|
|    L3 | overnight/L3 (7.5K SFT)           |  25% |
|   L10 | v2_0/L10 (10K SFT baseline)       |  65% |
|   L10 | overnight/L10_filtered (8K)       | **75%** |
|   L15 | overnight/L15_filtered (8K)       |  45% |
|   L20 | overnight/L20_v2 (12.5K SFT)      |  35% |
|   L24 | overnight/L24 (8K from-scratch)   | **75%** |
|   L27 | overnight/L27 (8K from-scratch)   | TBD |
|   L29 | v2_0/L29 (10K SFT)                | **85%** 🏆 |

7 distinct layers covered. Strongest 4 for the cross-depth video: L3, L10_filtered, L24, L29 (already rendered as `artefacts/v3/cross_layer_BEST/`).

## Asset matrix — what to generate

### Asset 1: Drawing animations (upscaled MP4 per drawing)

For each (layer × hero_concept) pair, render the AV's generated stroke sequence as an MP4 with 4× display upscale at 24 fps. Each MP4 ~3-5 seconds, ~500 KB.

Hero concepts: cat, dog, elephant, sun, horse, flower, tree, fish, car, airplane (10).
Layers: L3, L10_filtered, L24, L29 (4).
Total: 40 MP4s.

Saves to `artefacts/v3/viral/anim/{layer}/{concept}.mp4`.

**Script:** `code/eval/render_animations.py` (new). For each (layer, concept), do best-of-N CLIP-rank-of-N drawings, save the winner as MP4 via `stroke_render(strokes, save_animation_path=...)`.

**ETA:** ~25 min on GPU with N=8 samples per (layer, concept).

### Asset 2: 7-layer cross-depth strips (finer-grain than the 4-layer)

Render the SAME 10 hero concepts at all 7 available layers (L3, L10_filt, L15, L20, L24, L27, L29). Output strips 7 panels wide.

**Script:** modified `code/eval/v3_cross_layer_video.py` with all 7 layer symlinks.

**Setup:** `checkpoints/overnight7/L03 L10 L15 L20 L24 L27 L29` symlinks.

**ETA:** ~40 min on GPU (slow because 7 ckpts × 10 prompts × 16 samples × Qwen evaluation).

### Asset 3: Per-token trajectory (single layer, multiple tokens)

For 4-6 hero prompts, run model.generate() to produce next tokens; at each token position extract h_L10 (or L29) and render. Stitch into MP4 showing drawing morph as model reads.

**Script:** `code/eval/token_trajectory.py` already exists; just run with our L10_filtered ckpt.

**ETA:** ~15 min.

### Asset 4: Per-token × cross-layer animation (the killer)

For each generated token, render the drawing at ALL 4 layers (L3/L10/L24/L29) simultaneously. As model reads "Once upon a time", show 4 panels that all morph together.

**Script:** `code/eval/cross_layer_token_trajectory.py` (new). For each token position, render 4 layers side-by-side; stack into MP4 across token positions.

**ETA:** ~20 min.

### Asset 5: Activation interpolation morphs

For pairs (cat ↔ elephant), (dog ↔ fish), (sun ↔ flower), interpolate h at L10 in 30 steps, render each, MP4 morph.

**Script:** `code/eval/interpolate_h.py` already exists.

**ETA:** ~15 min.

### Asset 6: Infographic plots (matplotlib, local CPU, fast)

- **Per-layer scoreboard bar chart** (already in build_viral_v3.py).
- **Concept × layer heatmap**: rows = 20 hero concepts, cols = 4 layers, cell color = log P(correct concept | image). Shows which concepts each layer is good at.
- **Discriminability matrix heatmap**: pull the 20×20 matrix from findings/v3/per_layer_eval/L29.
- **Filtered vs baseline comparison chart**: L10 baseline 65 vs filtered 75; L29 baseline 85 vs filtered 45. Shows filter is layer-specific.
- **Architecture diagram**: SVG/PNG showing Qwen → h at layer L → AV → drawing → frozen Qwen → captioner. Hand-drawn in PIL.

### Asset 7: Hero best-of grid

20 hero concept best-cosine drawings from L29 (the headline 85% layer), rendered at 4× upscale, 4×5 grid.

### Asset 7b: Novel / OOD prompts (user requested)

Try prompts NOT seen in training. Even if drawings are rough, it's interesting interpretability content:
- "Eiffel Tower in Paris"
- "I am feeling sad"
- "The colour purple"
- "A skyscraper at night"
- "A bowl of soup"
- "The ocean"
- "A guitar"
- "A robot"
- "Numbers"
- "A volcano erupting"
- "Snow falling"
- "A king's crown"

Render with L29 (strongest layer). Show what the model "draws" for never-trained concepts. The OOD generalization (or its limits) is its own viral angle.

**Script:** `code/eval/render_ood_animations.py` — same as render_animations but with custom prompt list and no concept-target scoring (just render best stroke draws).

### Asset 8: Composed viral video

Stitch the above into an 90-120s video:
- 0:00 title + architecture explainer (10s)
- 0:10 cross-depth strip animations (cat / dog / elephant / sun) (20s)
- 0:30 per-token trajectory (eiffel) (15s)
- 0:45 cross-layer × token (15s)
- 1:00 interpolation morph (15s)
- 1:15 scoreboard + key findings (10s)
- 1:25 hero gallery + outro (10s)

Music: optional ambient soundtrack later.

## Execution order (DO IN PARALLEL)

GPU 6 + GPU 7 both stay busy:

1. **NOW**: launch on whichever GPU is free
   - Asset 1 (animations) — fast, batched, can run on either GPU
   - Asset 4 (cross-layer × token) — uses one Qwen for inference + per-layer AV swaps
2. **AS GPU FREES**: launch
   - Asset 2 (7-layer strips)
   - Asset 3 (per-token)
   - Asset 5 (interp)
3. **LOCAL (no GPU needed)**:
   - Asset 6 (matplotlib plots)
   - Asset 7 (hero grid composition)
   - Asset 8 (ffmpeg video stitch)

## Files to write

- `code/eval/render_animations.py` (Asset 1)
- `code/eval/cross_layer_token_trajectory.py` (Asset 4)
- `code/eval/build_viral_v3_PLUS.py` (Asset 6 + 7 + 8 — extends `build_viral_v3.py`)
- `code/eval/concept_layer_heatmap.py` (Asset 6 sub-script)

## Pull schedule

After EACH iteration finishes, immediately `bin/h200 pull <findings or artefacts>` to local.

## Safety: if SSH key expires early

All ckpts (1.4 GB) and findings + artefacts are already local. Everything past this point CAN run locally (Mac has ffmpeg, PIL, matplotlib; just no Qwen-3.5-4B model for new generations). If SSH dies, fall back to composing only the assets we already have on disk.
