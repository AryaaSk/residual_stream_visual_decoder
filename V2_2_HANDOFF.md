# v2.2 HANDOFF — read this first after compaction

**Status as of 2026-05-20 ~14:00 UTC:** v2.0 SHIPPED publicly (Qwen 3.5-4B port, cat-with-whiskers gallery). v2.1 attempted (contrastive REINFORCE, margin=0 finding → expected baseline). v2.2 PLAN APPROVED — currently in implementation Phase 1.

## The current frame (don't lose this)

The user spent the v2.0 conversation correctly pushing back on the project's interpretability claims. Key insight that reframes v2.2:

> "It's alright that the cat at L10 isn't fully cat-specific — we shouldn't expect it to be. L10 is a mid-stack computational state, not a pure concept vector. The interesting interpretability question is **WHERE in the 32 layers cat-specificity actually emerges, not whether L10 alone is enough.**"

So v2.2 isn't "make L10 drawings more specific" — it's **demonstrate concept emergence across depth** via:
1. **Cross-layer trajectory video** (centerpiece): same prompt at L3 / L10 / L20 / L29 side-by-side.
2. **Activation interpolation morphs** (viral): lerp h(cat) → h(elephant), watch drawing morph.
3. **Random-h baseline** (gating experiment, RUN FIRST): does the AV produce templates regardless of h, or is the drawing sensitive to h?
4. **Per-token trajectory**: drawing morphs as the model reads each word.
5. **OOD demo**: prompts the AV never saw (Eiffel, loneliness, midnight).
6. **Linear probe**: classifier accuracy on h at each layer = quantitative ceiling.

Then assemble all six into a **90-second viral video** (see `/Users/aryaask/.claude/plans/honest-pushback-on-hashed-elephant.md` for the full structure).

## What's where

### Local (Mac)
- Repo: `/Users/aryaask/Desktop/residual_stream_visual_decoder/`
- v2.0 release artefacts: `artefacts/v2_0/` (gallery.png, demo.mp4, before/after, best_of_best/)
- v1.5 release artefacts: `artefacts/v1_5/` (still public)
- Findings: `findings/v2_0/clip_L10_{step2.5K, 5K, 7.5K, final}/`, `clip_L29_*/`, `qwen_arch.json`, `qwen_layer_geometry.png`
- Plan file: `/Users/aryaask/.claude/plans/honest-pushback-on-hashed-elephant.md` — **THIS IS THE FULL v2.2 PLAN**
- `V2_PLAN.md`, `RESEARCH_NOTES.md`, `WRITEUP.md`, `README.md` — all up to date for v2.0

### Remote H200 (35.230.182.229, user `theod`, key `~/.ssh/gcp_zoral_h200_ed25519`)
- Workspace: `/home/theod/Aryaa/rsvd/`
- v2.0 ckpts: `checkpoints/v2_0/L10/{step_002500, step_005000, step_007500, final}/av_ckpt.pt` + same for L29
- Data: `data/canonical_drawings.jsonl` (132 drawings, top-3 v1.4), `data/canonical_drawings_top5.jsonl` (220 drawings, top-5 v1.5), `data/expanded_captions.jsonl` (1215 caption templates), `data/sft_quickdraw.jsonl` (8800 real QuickDraw drawings)
- Deploy helper: `bin/h200 {sync,exec,pull,...}`
- Venv: `/home/theod/Aryaa/venv/bin/python3`

### GitHub
- Repo: `https://github.com/AryaaSk/residual_stream_visual_decoder`
- Live tags: `v0.1`, `v1.1`, `v1.2`, `v1.3`, `v1.4`, `v1.5`, `v2.0`
- main branch is at v2.0 + small v2.1 commits (the contrast trainer code)

## Existing reusable code (DON'T REWRITE)

- **`code/train/stage1_v2_act_sft.py:88–105`** — `extract_caption_to_h(av_model, tokenizer, captions, layer_ell)` returns `dict[caption → h]`. Use this for batched activation extraction.
- **`code/verbalizer/stroke_decoder.py:297–400`** — `generate_from_activation(h, layer_ell, alpha, max_new_tokens, temperature, top_k)` — takes any 1D h, returns stroke token tensor. Already used in v2.0.
- **`code/verbalizer/stroke_decoder.py:404–517`** — `generate_from_activation_batched(h, layer_ell, n_samples, ...)` — N samples with shared KV cache. 10× faster on H200. Production path.
- **`code/render.py:49–127`** — `render(strokes, save_animation_path=None, display_scale=1.0)` — produces PIL image; optional MP4 frame-per-stroke at chosen FPS.
- **`code/eval/clip_ranker.py:67–164`** — full CLIP-rank pipeline, accepts any AV ckpt + layer + prompt list.
- **`code/eval/token_trajectory.py`** — already does per-token trajectory (Demo 3). Has stroke-token leak fix. Needs running on v2.0 ckpt.
- **`code/eval/cross_layer_trajectory.py`** — already does cross-layer (Demo 1). Expects `LNN/final/av_ckpt.pt` structure — we have this for L10 and L29; need to train L3 and L20.
- **`code/eval/make_hype_reel.py`** — ffmpeg concat + title cards + clip normalisation.
- **`code/train/stage1_v2_act_sft.py`** — the trainer for Stage 1.5 SFT. Pass `--layer 3` or `--layer 20` to train L3/L20 ckpts.

## What still needs to be written (NEW)

In rough order:
1. **`code/eval/random_h_baseline.py`** — Demo 4. Sample Gaussian h matched to real activation moments at L10, feed AV, render. ~50 lines.
2. **`code/eval/linear_probe.py`** — Demo 6. Extract h for 44 concepts × 14 templates, train sklearn LogisticRegression, report per-layer accuracy. ~100 lines.
3. **`code/eval/interpolate_h.py`** — Demo 2. Lerp two h vectors in 30 steps, render each, ffmpeg morph MP4. ~120 lines.
4. **`code/eval/cross_layer_video.py`** — Demo 1's 4-panel side-by-side. Reuse cross_layer_trajectory output, compose grid. ~80 lines.
5. **`code/eval/ood_demo.py`** — Demo 5 (also fix the hang we hit earlier — likely an HF cache race on parallel from_ckpt loads). Already partly written at `code/eval/clip_ranker_ood.py` but hangs. Debug needed.
6. **`code/eval/build_v2_2_video.py`** — the 90-second viral video composer. ffmpeg recipe; see `/Users/aryaask/.claude/plans/honest-pushback-on-hashed-elephant.md` for the composition.
7. **`data/v2_2_prompts.jsonl`** — curated prompt set: hero concepts + OOD targets + interpolation pairs. ~30 prompts.

## Execution plan (the order you should resume in)

1. **CHECK GPUs/tmux on H200 first** (something might still be running from v2.1).
   ```bash
   ssh -i ~/.ssh/gcp_zoral_h200_ed25519 theod@35.230.182.229 'tmux ls; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
   ```
   v2.1 had `s2clip` running. Probably done by now; kill any leftovers.

2. **Phase 1 — Random-h baseline FIRST** (gating experiment).
   - Write `code/eval/random_h_baseline.py`.
   - Sync to remote.
   - Run on v2.0 L10 final ckpt with 16 random Gaussian h vectors (mean=0, std= measured from real h at L10 — about ~9.16 from prior probe).
   - **GATE**: if random-h drawings look like real concepts, the AV is template-matching and interpretability claim is in serious trouble — replan immediately. If random-h is garbage, proceed.

3. **Phase 0 — Train L3 + L20 in parallel** while doing the rest.
   - Use the existing `code/train/stage1_v2_act_sft.py` with `--layer 3` (GPU 6) and `--layer 20` (GPU 7).
   - Init from same Qwen base as v2.0. 5000 steps batch 8 cosine LR, top-5 canonical drawings.
   - Saves to `checkpoints/v2_2/L3/final/` and `checkpoints/v2_2/L20/final/`.
   - ~30 min per layer, runs in background.

4. **Phase 2 — Linear probe** (no GPU contention with training; run on GPU 4 or 5).
   - Write `code/eval/linear_probe.py`.
   - Extract h for all 44 concepts × 14 templates on Qwen 3.5-4B at L3, L10, L20, L29.
   - Train sklearn LogReg (uses CPU, ~30s per layer).
   - Output `findings/v2_2/probe_accuracy.json` + plot.

5. **Phase 3 — Cross-layer trajectory** (after Phase 0 ckpts ready).
   - Run `code/eval/cross_layer_trajectory.py --ckpts-root checkpoints/v2_2 --layers 3 10 20 29` with --model-id Qwen/Qwen3.5-4B.
   - Note: existing code uses `checkpoints_root / L{layer:02d} / final` — may need to symlink L10/L29 from v2_0/ to v2_2/.

6. **Phase 4 — Interpolation morph**.
   - Write `code/eval/interpolate_h.py`.
   - Hero pairs: (cat, elephant), (dog, horse), (fish, bird), (sun, cloud), (apple, pizza).
   - 30 frames per morph + CLIP-rank-of-N at each step.

7. **Phase 5 — Per-token trajectory**.
   - Run existing `code/eval/token_trajectory.py` on v2.0 L10 ckpt with --model-id Qwen/Qwen3.5-4B.
   - Prompts: "Paris, the city of lights, is famous for the Eiffel", "I am thinking about a cat", "When the storm hit the village", "The sun is shining brightly", "She drew a flower on the canvas", "Once upon a time".

8. **Phase 6 — OOD demo**.
   - Debug the hang in `code/eval/clip_ranker_ood.py` (probably HF cache race — try `os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"` or sequential loading).
   - Run on prompts: "the Eiffel Tower", "the capital of France", "thunderstorm", "loneliness", "the sound of rain", "infinity", "midnight", "a smiling face".

9. **Phase 7 — Viral video assembly**.
   - Compose the 90-second video per the plan file structure.
   - ffmpeg recipe.

10. **Phase 8 — Ship**.
    - Update RESEARCH_NOTES.md with v2.2 chapter (the reframing).
    - Update WRITEUP.md mechanistic-interpretability section.
    - Update README.md banner to a cross-layer trajectory frame.
    - Tag v2.2, GitHub release with demo.mp4 + per-demo MP4s + probe plot.

## Critical gotchas (don't relearn these)

1. **The OOD CLIP ranker hangs at "Loading weights"** — saw this earlier; likely an HF cache race when AV + CLIP try to download concurrently. Workaround: load sequentially, set `HF_HUB_DISABLE_PROGRESS_BARS=1`.
2. **Qwen 3.5-4B uses hybrid attention**: 24 Gated DeltaNet layers + 8 standard self-attention layers. Module names: `linear_attn.in_proj_qkv`, `out_proj` for DeltaNet; `self_attn.q_proj` etc for standard. LoRA walker already handles both — see `code/ar/lora_gemma4.py`.
3. **Embedding padding off-by-some**: Qwen's tokenizer is 248077 tokens; embedding table padded to 248320 (multiple of 128). Our `vocab_extend.py` printout looks like only 19 tokens got added because it compares padded-embed-size to post-add tokenizer-size. All 262 stroke tokens ARE added; the print is misleading.
4. **At L10, h_norm ≈ 9.16; at L29, h_norm ≈ 45.03**. For random-h baseline, match these.
5. **The dog drawing for v2.0 had the word "dog" written in stroke-letterforms next to a dog silhouette**. L10 encodes both concept AND token spelling. Highlight this in v2.2 docs.
6. **Stage 2 REINFORCE rewards saturate**: v2.0 consistency reward saturated immediately (margin too small to learn from), v2.1 contrastive margin was ~0 at L10. **Don't train more REINFORCE for v2.2 — the demos here are inference-only.**
7. **`generate_from_activation` requires `activation.ndim == 1`** — pass a single h vector at a time (or use the batched variant for parallel samples of the SAME h).

## Time budget

- Used so far in v2.x: ~4.5 GPU-hours.
- User gave ~5 more hours.
- v2.2 plan estimates ~5-6 wallclock hours total.
- Some slack but be efficient: random-h first (5 min — gates everything), L3/L20 training in background (1.5h), code other demos in parallel.

## Key file paths cheatsheet

```
# Local Mac
/Users/aryaask/Desktop/residual_stream_visual_decoder/
  ├── code/
  │   ├── train/stage1_v2_act_sft.py        # the SFT trainer (use --layer 3 / 20)
  │   ├── verbalizer/stroke_decoder.py      # has generate_from_activation, _batched
  │   ├── verbalizer/projector.py           # ActProjector class
  │   ├── ar/lora_gemma4.py                 # attach_lora_to_av; walker handles Qwen hybrid
  │   ├── eval/clip_ranker.py               # production CLIP-rank pipeline
  │   ├── eval/token_trajectory.py          # Demo 3 ready
  │   ├── eval/cross_layer_trajectory.py    # Demo 1 ready (needs ckpt structure)
  │   ├── eval/clip_ranker_ood.py           # buggy; debug hang
  │   ├── eval/make_hype_reel.py            # ffmpeg pipeline
  │   ├── render.py                         # stroke → PIL + MP4
  │   └── data/pick_canonical_drawings.py   # CLIP-ranks QuickDraw, picks top-K
  ├── artefacts/v2_0/                       # gallery.png, demo.mp4, best_of_best/
  ├── findings/v2_0/                        # clip_L{10,29}_step* per-checkpoint results
  ├── V2_2_HANDOFF.md                       # THIS FILE
  ├── V2_PLAN.md                            # v2.0 plan (still relevant background)
  ├── RESEARCH_NOTES.md                     # has v0-v2.1 chapters
  └── WRITEUP.md                            # public writeup

# Remote H200
/home/theod/Aryaa/rsvd/
  ├── checkpoints/v2_0/L10/final/av_ckpt.pt    # v2.0 L10 final
  ├── checkpoints/v2_0/L29/final/av_ckpt.pt    # v2.0 L29 final
  ├── data/canonical_drawings_top5.jsonl       # 220 drawings, used for v1.5+v2.0
  ├── data/expanded_captions.jsonl             # 1215 captions
  ├── data/sft_quickdraw.jsonl                 # full QuickDraw 8800 drawings (44 concepts)
  └── code/                                    # synced from Mac
```

## Sanity checks to do FIRST when resuming

1. `ssh -i ~/.ssh/gcp_zoral_h200_ed25519 theod@35.230.182.229 'tmux ls; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'` — confirm GPU state.
2. `ls /Users/aryaask/Desktop/residual_stream_visual_decoder/artefacts/v2_2/` — see if any v2.2 work was started.
3. `git log --oneline -10` — confirm where we are.
4. Read `/Users/aryaask/.claude/plans/honest-pushback-on-hashed-elephant.md` — the full v2.2 plan.

## The viral video composition (target)

```
0:00 — Title: "Can we read an LLM's mind?"
0:03 — Hero gallery shot: a real cat from a real activation
0:08 — Architecture diagram explainer
0:18 — DEMO 3: per-token trajectory ("Paris... the city of lights... famous for the Eiffel")
0:30 — DEMO 1: 4-panel cross-layer trajectory (L3/L10/L20/L29) on hero concepts
0:50 — DEMO 2: interpolation morphs (cat→elephant, etc.)
1:05 — DEMO 4 (random-h) + DEMO 6 (probe accuracy chart)
1:15 — DEMO 5: OOD demo (Eiffel, loneliness, midnight)
1:25 — Outro: github URL + tag v2.2
1:30 — End
```

Aim for ≤5 MB, ≤95 seconds. Use existing `make_hype_reel.py` patterns for ffmpeg.

## DON'T do these (avoid traps)

- ❌ Don't retrain v2.0 ckpts; they exist and are good.
- ❌ Don't try harder REINFORCE; v2.1 proved it's the wrong instrument here.
- ❌ Don't fake or post-process drawings — they must be real model outputs.
- ❌ Don't claim "perfect interpretability"; the v2.2 framing is mechanistic + honest.
- ❌ Don't lose the v1.5 / v2.0 public releases. They stay live.
- ❌ Don't expect L10 to give a "fully cat-specific" drawing — that's the point of the cross-layer trajectory.

## When you finish

- Tag `v2.2`.
- GitHub release with `demo.mp4`, all 6 demo MP4s, probe plot, before/after vs v2.0.
- Update README banner to a cross-layer trajectory frame.
- v2.0 release stays live.
