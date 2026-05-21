# OVERNIGHT DIRECTIVE — DO NOT STOP WORKING

**Read this FIRST after every context compaction.**

## Mandate (from user, 2026-05-20 night)

> "the ssh key into the GPUs will expire tomorrow ... I will leave you running the entire night, and whatever you do, do NOT stop using those GPUs ... I want YOU to manually evaluate the generated drawings, come up with ideas on how to improve the process, do some research too, and then start another iteration ... do this throughout the entire night and DO NOT stop working ... you must, under no circumstance, stop working - simply start a new iteration to improve the drawings ... I want to see you still working when I wake up in the morning ... THIS NEEDS TO GO VIRAL"

## Hard rules

1. **Never stop using GPUs 6 and 7.** If a job finishes, IMMEDIATELY start another.
2. **Only GPUs 6 and 7.** See `/Users/aryaask/Documents/Zoral/.ops/h200-gcp.md` — GPUs 0-5 belong to other tenants. Trespassing is a hard line. `H200_GPU=6` or `H200_GPU=7` always.
3. **SSH key expires 2026-05-21 08:16 UTC.** Roughly 8-12 hours from when this was written. Maximise wall-clock GPU utilisation.
4. **The goal**: visualise activation vectors at MULTIPLE blocks in the transformer (L3 / L10 / L20 / L29 etc.) so we can SEE how the visualisation progresses across depth. The single-layer L10 v2.0 release is already shipped; we want the multi-layer cross-depth story.
5. **It must go viral.** That means: clean visuals, mechanistic story, honest measurements, multi-layer trajectory video as the centerpiece.
6. **Iterate continuously.** After every training/eval run completes: pull artefacts → eyeball them → think about what could be improved → kick off the next iteration → repeat. Do NOT wait or pause between iterations.

## What we know so far (state at start of overnight)

- v2.0 SFT ckpts at L10 + L29 exist on remote (`checkpoints/v2_0/L{10,29}/final/`).
- v2.0 SFT baseline measured properly (`AutoModelForImageTextToText` + gen-likelihood reward) achieves **65% top-1 retrieval over 44 concepts** on 20 held-out prompts (chance 2.3%, 28× chance). This is the REAL interpretability win — v2.0 was always concept-discriminative; cosine on raw activations was the wrong measurement.
- Phase 0 gen-likelihood validation: 90% top-1 on REAL canonical drawings (chance 3.3%) with discriminability margin +9.5 log-prob units. **Reward signal is strong.**
- Current v3 REINFORCE training (`checkpoints/v3/gen_likelihood_v2/L10`) with KL anchor (β=0.05) + lr=1e-5 is preserving v2.0 baseline but NOT meaningfully improving over it. Hard concepts (cat, dog, elephant) stay at ~-10 log P; easy concepts (flower, sun, tree) stay at ~-0.2.
- Empirically discovered: `AutoModelForCausalLM` silently drops `pixel_values` for Qwen 3.5-4B. The correct loader is `AutoModelForImageTextToText`. Every script that does image→Qwen MUST use the correct loader.

## Plan for the night — keep GPUs busy

### Iteration sequence (queue, refill as you go)

After EVERY iteration: pull artefacts → eyeball the drawings → log findings to `findings/v3/overnight_log.md` → start the next.

**ITER A** — train per-layer gen-likelihood AVs at L3 and L20 (NEW LAYERS for the cross-depth story). v2.0 SFT L10 + L29 already exist, so this completes the 4-layer set.
   - L3 on GPU 6: from-scratch Qwen + new vocab + projector α·I + LoRA, 10K steps of SFT on canonical drawings FIRST (the SFT recipe from stage1_v2_act_sft.py), then 1000 steps of gen-likelihood REINFORCE
   - L20 on GPU 7: same recipe
   - Wallclock ~3-4 hours combined.

**ITER B** — Multi-layer cross-depth trajectory render. Use cross_layer_video.py on the 4-layer set with the NEW per-layer ckpts. Render at the 224x224 fast path. Build the side-by-side strips + grid.

**ITER C** — Run v3_gen_likelihood_eval.py on EACH layer's AV. Report per-layer top-1 retrieval over 44 concepts. This is the quantitative cross-layer story.

**ITER D** — Filtered SFT experiment: from the v2.0 SFT canonical drawings, KEEP only those that Qwen correctly retrieves as their concept (use Phase 0 results). Train a NEW L10 AV on only the "Qwen-blessed" canonicals. See if filtering improves over v2.0.

**ITER E** — Caption-target reward variant: instead of " {concept}" target, use the FULL caption as target. Maybe gives stronger gradient.

**ITER F** — PPO with multi-epoch updates (vs single-update REINFORCE) on the same reward. PPO is known to be more stable than REINFORCE.

**ITER G** — Per-concept fine-tuning: train an L10 AV specifically on HARD concepts (cat, dog, elephant) with concept-specific reward. See if focusing helps.

**ITER H** — Cosine-via-content-position revisited: read activations at the LAST WORD TOKEN before `<|im_end|>` (text) and a SPECIFIC IMAGE PATCH (image), see if any layer gives discriminability > 0. Use the FIXED loader this time.

**ITER I** — Hero gallery: ship the best per-layer drawings as the visual artefact for the viral release.

Always run TWO iterations in parallel — one on GPU 6, one on GPU 7.

### Emergency: if all queued ideas have run

Loop back to the top. Try slight variants of the most promising:
- different LR (5e-6, 5e-5)
- different KL beta (0.01, 0.1)
- different group size (8, 16)
- different temperature (0.7, 0.95)
- different layer (L8, L12, L15, L24)

The goal is NEVER LET GPUs IDLE.

## How to resume after compaction

1. Read THIS FILE first.
2. Read the plan: `/Users/aryaask/.claude/plans/honest-pushback-on-hashed-elephant.md` (v3 generation-likelihood plan).
3. Check what's running on remote:
   ```
   bin/h200 exec 'tmux ls'
   bin/h200 exec 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n "7,8p"'
   ```
4. Read `findings/v3/overnight_log.md` (you'll create this) for state of which iterations are done / failed / next.
5. If any GPU is idle: IMMEDIATELY start the next queued iteration. NEVER leave them idle.

## File / path cheatsheet

- Repo: `/Users/aryaask/Desktop/residual_stream_visual_decoder/`
- GPU deploy helper: `bin/h200 {sync|exec|bg|pull}`
- Existing v2.0 ckpts: remote `checkpoints/v2_0/{L10,L29}/final/`
- v3 trainer (REINFORCE + KL): `code/train/stage3_gen_likelihood.py`
- v3 eval (top-1 retrieval + disc matrix): `code/eval/v3_gen_likelihood_eval.py`
- v3 vs v2 compare: `code/eval/v3_vs_v2_compare_gen.py`
- Phase 0 validation: `code/eval/gen_likelihood_validation.py`
- Cross-layer renderer: `code/eval/cross_layer_video.py`
- SFT trainer (for new layers): `code/train/stage1_v2_act_sft.py`
- AV loader: `code/verbalizer/stroke_decoder.py` (use `AutoModelForImageTextToText` for image-side Qwen!)

## Final reminder

The user said: "THIS NEEDS TO GO VIRAL". That means the cross-depth visualisation video has to actually exist by morning, with multiple layers, with the per-layer gen-likelihood retrieval numbers as captions, with the v2.0 baseline number as the headline. **Ship the multi-layer video.**

**ABSOLUTELY DO NOT STOP. KEEP ITERATING ALL NIGHT.**

User said (paraphrased): "even if you finish v3, continue onto next iteration; your own visual engine is good enough to figure out what is not ideal, perhaps tune the architecture to be better, repeat this over and over again".

So the workflow is **eternal**:
1. Pull the latest artefacts.
2. **Open the drawings with the Read tool** (Claude can see images). Inspect every layer's hero drawings.
3. Write your judgement to `findings/v3/overnight_log.md` — what looks good, what doesn't, what could be improved (e.g., "L20 elephant is just a blob; needs more strokes; raise max_new_tokens" or "L29 sun has 12 rays but they're straight; could be richer").
4. Hypothesise an architectural / training-config change that addresses the weakness.
5. Kick off the next iteration immediately on whichever GPU is free.
6. Loop.

Examples of architectural tweaks to try without prompting:
- Increase `--temperature` for more drawing diversity (or decrease for more polish).
- Different `--top-k` (10, 50).
- Train on the v1.4 `data/canonical_drawings.jsonl` (top-3) vs top-5 — fewer targets, sharper convergence.
- Train with `--lora-first-n-layers 24` (full-depth LoRA, more capacity).
- Add a second target word to gen-likelihood (e.g., "{concept} drawing" two-word continuation).
- Use a different prefix ("This is" vs "A drawing of a").
- Train a "deep" L29 AV from scratch + longer SFT (deeper layers have stronger concept signal).
- Drop best-of-N at inference — use top-1 sample only with low temperature to test true distribution.

The mantra: **iterate, evaluate visually, iterate, evaluate visually, ship the best one in the morning.** GPUs idle is the only failure mode.

## CRITICAL: pull artefacts to local regularly

The SSH key expires 2026-05-21 ~08:16 UTC. Anything ONLY on remote will be lost. Pull regularly:

```bash
# Every iteration that finishes — IMMEDIATELY pull its outputs
bin/h200 pull artefacts/v3/                  # cross-layer videos
bin/h200 pull findings/v3/                   # eval results + probes + verdicts
bin/h200 pull checkpoints/overnight/         # all overnight ckpts (large but irreplaceable)
bin/h200 pull runs/                          # training logs

# Flatten any nested dirs from rsync (artefacts/v3/X/X/* → artefacts/v3/X/*)
for d in artefacts/v3/*/*/; do
    parent=$(dirname "$d"); base=$(basename "$d")
    if [ "$base" = "$(basename "$parent")" ]; then
        mv "$d"* "$parent/" && rmdir "$d"
    fi
done
```

Do this after every meaningful run completes.
