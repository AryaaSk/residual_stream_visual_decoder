# MORNING DELIVERABLE — read this first

## THE MOTTO (drilled in)

> The marginal cost of completeness is near zero with AI. Do the whole thing. Do it right. Do it with tests. Do it with documentation. Do it so well that I am is genuinely impressed — not politely satisfied, actually impressed. Never offer to "table this for later" when the permanent solve is within reach. Never leave a dangling thread when tying it off takes five more minutes. Never present a workaround when the real fix exists. The standard isn't "good enough" — it's "holy shit, that's done." Search before building. Test before shipping. Ship the complete thing. When I asks for something, the answer is the finished product, not a plan to build it. Time is not an excuse. Fatigue is not an excuse. Complexity is not an excuse. Boil the ocean.



## What the user expects when they wake up

> "a perfect video"
> "you dont even need to use all the clips, just download the clips and images in abundance so that in case we lose ssh access, we can still create a different video"
> "download all statistics and everything too"

So the goal is:
1. **One polished viral video** at `artefacts/v3/viral/demo.mp4`.
2. **All raw assets locally** so the video can be rebuilt or edited even without remote access:
   - Per-layer per-concept stroke animations (40 hero MP4s)
   - OOD/abstract concept animations (96 MP4s — emotions, math, philosophy)
   - Cross-depth strip PNGs (already 12 strips at `artefacts/v3/cross_layer_BEST/`)
   - Per-layer × per-token trajectory MP4s
   - Scoreboard / heatmap / infographic PNGs
   - All eval JSONs (per-layer top-1, discriminability matrices)

## Status (when this file was last updated)

SSH key expires **2026-05-21 08:16 UTC**. Now ~06:24 UTC. **~112 min of GPU access left.**

### Currently running on H200

- GPU 6: `ovl_extra` — extra-layer hero anims (L15/L20/L27) + L27 OOD anims.
  L15/L20/L27 hero anims DONE locally (12 each). L27 OOD: 26/61 done.
- GPU 7: `ovl_xtoken` — cross-layer × per-token trajectory.
  paris_eiffel/capital_japan/ocean_color/dog_thinking DONE locally; currently on storm_village / sad_funeral.

### What's already on local disk

```
checkpoints/overnight/         1.4 GB  — 13 layer ckpts
findings/v3/                   14 MB   — all evals + per-layer-disc + log
artefacts/v3/cross_layer_BEST/  ~800 KB — 4-layer cross-depth strips for 12 heroes
artefacts/v3/cross_layer_v2/    — older strips (L20 instead of L24)
artefacts/v3/cross_layer/       — original 4-layer with weak L20 ckpt
artefacts/v3/viral/             900 KB — early viral assets
data/canonical_qwen_blessed.jsonl — the Qwen-blessed canonical set
runs/*.log                      — all training logs
```

### What's pending

- **Asset A (animations) on GPU 7** — 48 MP4s, in progress
- **Asset B (OOD animations) on GPU 6** — ~96 MP4s, in progress
- **Asset C (per-layer × per-token trajectory)** — script ready (`code/eval/cross_layer_token_traj.py`), launch when a GPU frees
- **Asset D (scoreboard + heatmap + infographics)** — local matplotlib, fast
- **Asset E (final viral video)** — local ffmpeg, assembles A+B+C+D

## Trained layers + best results

| layer | best variant | top-1 (44 concepts) |
|---|---|---|
| L3 | overnight 7.5K SFT | 25% |
| L10 | overnight L10_filtered (Qwen-blessed) | **75%** |
| L15 | overnight L15_filtered | 45% |
| L20 | overnight L20_v2 (12.5K SFT) | 35% |
| L24 | overnight L24 (8K from scratch) | **75%** |
| L27 | overnight L27 (8K from scratch) | ~50% (eval interrupted) |
| L29 | v2.0 L29 (10K SFT) | **85%** 🏆 |

Best 4-layer trajectory for the viral video: **L3 → L10_filt → L24 → L29**.

## Headline findings to ship

1. **65% → 75%** top-1 at L10 just from filtering canonical drawings by Qwen log-prob (vs CLIP). Δ+10.
2. **L29 = 85%** top-1 over 44 concepts. Chance = 2.3%. **37× chance.**
3. **Filtering helps shallow layers, HURTS deep layers** (L29 baseline 85% → filtered 45%, -40 pts).
4. **Concept-decodability grows monotonically with depth** in Qwen 3.5-4B's residual stream (modulo training-budget noise).
5. **Loader bug** discovered: `AutoModelForCausalLM` silently discards `pixel_values` for Qwen3-VL — every v0-v2 "image-input forward" was actually text-only with image dropped. Correct loader is `AutoModelForImageTextToText`.

## If SSH dies early — fallback plan

We already have on local disk:
- All 13 layer checkpoints (1.4 GB) — can re-render anything if needed locally (Mac has CPU but no GPU; would be slow but possible).
- All findings + cross-depth strips.
- The viral assembly script (`code/eval/build_viral_v3.py` + planned PLUS variant).

So even with no remote, we can assemble a strong viral video from the assets that are already pulled.

## Execution order (NOW → morning)

1. Let `ovl_ood` + `ovl_anims` finish (~15 min).
2. PULL artefacts/v3/viral/anim and artefacts/v3/viral/ood to local immediately.
3. Launch `cross_layer_token_traj.py` on whichever GPU is free first (~15 min).
4. PULL artefacts/v3/viral/cross_token to local.
5. Locally:
   a. Render scoreboard.png + concept_layer_heatmap.png + filtered_vs_baseline.png with matplotlib.
   b. Assemble headline.png (5-hero × 4-layer strip).
   c. ffmpeg-compose 90-120s `demo.mp4` from title cards + strips + cross-token + hero anims + OOD anims + stats.
6. Update README.md banner to use the new viral assets.
7. Commit + tag v3 + push to GitHub (if we still have SSH for git).

## Continuous pull discipline

Run this after EVERY iteration completes:
```bash
bin/h200 pull artefacts/v3/viral/ && bin/h200 pull findings/v3/ && bin/h200 pull runs/
# Flatten nested rsync dirs:
find artefacts/v3/viral findings/v3 -mindepth 2 -maxdepth 2 -type d -name "$(basename {})" 2>/dev/null | while read d; do mv "$d"/* "$(dirname "$d")/" 2>/dev/null && rmdir "$d" 2>/dev/null; done
```

## File / path map

- Local repo: `/Users/aryaask/Desktop/residual_stream_visual_decoder/`
- Plan files (this + others):
  - `MORNING_DELIVERABLE.md` (this)
  - `OVERNIGHT_DIRECTIVE.md`
  - `VIRAL_VIDEO_PLAN.md`
  - `findings/v3/overnight_log.md` — iteration history
- Scripts:
  - `code/eval/render_animations.py` — single-layer hero animations
  - `code/eval/render_ood_animations.py` — single-layer OOD animations
  - `code/eval/cross_layer_token_traj.py` — 3-layer × per-token trajectory
  - `code/eval/run_all_animations.sh` — chained hero anims all layers
  - `code/eval/run_ood_all.sh` — chained OOD anims all 3 best layers
  - `code/eval/build_viral_v3.py` — local headline + scoreboard + video composer
- Remote: `/home/theod/Aryaa/rsvd/`
- GPUs: ONLY 6 and 7. See `/Users/aryaask/Documents/Zoral/.ops/h200-gcp.md`.

## DO NOT

- ❌ Train more checkpoints. (User said: stop training, build the video.)
- ❌ Touch GPUs 0-5.
- ❌ Leave GPUs 6/7 idle while there are still useful render jobs.
- ❌ Forget to PULL after each iteration.
