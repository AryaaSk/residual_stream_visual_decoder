# Plan: Residual Stream Visual Decoder — 7-Day Execution

## Context

We are building a "visual lens" into Gemma 4 E2B's residual stream: a stroke-output decoder that converts mid-stack activations into hand-drawn-style images (rendered from stroke tokens, with animation as a free bonus), jointly trained with an image-input reconstructor (using Gemma 4's native vision pathway) so the drawings must be informative enough to recover the original activation.

Architecturally a clone of Anthropic's Natural Language Autoencoders (NLAs, 2026), with the text channel swapped for a stroke channel.

The full design spec lives at `/Users/aryaask/Desktop/residual_stream_visual_decoder/` (7 markdown files). This plan is the **execution** layer: what to actually do, in what order, on which hardware, with what discipline.

**The user's headline ask:** highest-ROI single anchor layer trained end-to-end and producing visible drawings **by end of Day 1**. Remaining layers and text-NLA baseline roll out across Days 2-7.

---

## Goals & success criteria

| Milestone | Definition of done |
|---|---|
| **End of Day 1** | One layer fully trained (Stages 1-2-3). 50 probes rendered as PNG + MP4. FVE measured. Git repo with all artefacts pushed to remote. |
| **End of Day 3** | 3 additional anchor layers fully trained. Trajectory comparison grids for those layers + Day-1 layer. |
| **End of Day 5** | All 8 anchor layers fully trained. Text-NLA baseline trained at one mid-stack layer. |
| **End of Day 7** | Comparison grids (logit / tuned / text-NLA / visual-NLA) per probe per layer. Hero probes with polished MP4s. 60-sec hype reel. Publishable blog-post writeup. GitHub release with reproducibility kit. |

---

## Resource inventory

### Hardware

**Primary compute: H200 GCP instance** (per `.ops/h200-gcp.md`)
- External IP: `35.230.182.229`
- Instance name: `p460-8xh200-spot`
- User: `theod`, key: `~/.ssh/gcp_zoral_h200_ed25519`
- 8× H200 (141 GB each), but **we only get GPUs 6 and 7** (always set `CUDA_VISIBLE_DEVICES=6,7`)
- Shared box, spot instance (can be preempted)
- 316 GB free disk on `~/Aryaa/`
- Venv at `~/Aryaa/venv` (transformers 5.8.1)
- `bin/h200` deploy helper exists: `run`, `sync`, `ssh`, `nvidia`, `pull`

**Aspirational compute:** user may secure 4-5 H200s for the week. Plan compresses linearly if this materialises.

**Local: MacBook**
- For data prep, code authoring, git workflows, plotting, video assembly
- All non-GPU work happens here

### Critical risk: SSH key expiry

**Key expires 2026-05-21 08:16 UTC.** Today is 2026-05-19. That gives **~48 hours of guaranteed H200 access**.

- User is pursuing key renewal in parallel.
- Plan A: key renewed in time → full 7-day rollout proceeds.
- Plan B: key not renewed → Day 1 still produces a complete single-layer artefact within the 48-hour window. Worst case we ship that as v0.

Day 1 must therefore be **end-to-end complete in <24 wall-clock hours** and produce a checkpoint + 50-probe artefacts before the 48-hour countdown forces a stop.

---

## Repository

### Location & name

```
/Users/aryaask/Desktop/residual_stream_visual_decoder/
```

Git repo named `residual_stream_visual_decoder` (matches folder).

### Remote

Create a private GitHub repo `aryaask/residual_stream_visual_decoder` (private until we're ready to release). Push at least once per major milestone (every few hours during active work).

### Initial commit content

The 7 markdown design docs already in the folder become the initial commit. Then we add code/, research_log/, findings/, artefacts/ structure.

### Directory structure (after Day 1)

```
residual_stream_visual_decoder/
├── README.md                          (already exists)
├── 01-vision.md                       (already exists)
├── 02-architecture.md                 (already exists)
├── 03-training.md                     (already exists)
├── 04-renderer.md                     (already exists)
├── 05-evaluation.md                   (already exists)
├── 06-prior-work.md                   (already exists)
│
├── code/                              (all source code)
│   ├── stroke_tokenizer.py            (Cartesian stroke-5, 128-bin Δx/Δy)
│   ├── render.py                      (deterministic, PNG + MP4)
│   ├── lenses/
│   │   ├── logit_lens.py
│   │   └── tuned_lens.py
│   ├── av/
│   │   ├── vocab_extend.py            (Anole-style: add stroke tokens to Gemma 4)
│   │   ├── activation_injection.py    (<ACT_TOKEN> embedding surgery)
│   │   └── stroke_decoder.py          (the AV)
│   ├── ar/
│   │   └── reconstructor.py           (truncated Gemma 4 + Linear(d,d))
│   ├── data/
│   │   ├── iam_loader.py
│   │   ├── quickdraw_loader.py
│   │   └── activation_extractor.py
│   ├── train/
│   │   ├── stage1_av_sft.py
│   │   ├── stage2_ar_supervised.py
│   │   └── stage3_av_grpo.py
│   └── eval/
│       ├── day0_alignment.py          (the go/no-go gate)
│       ├── roi_ranking.py
│       ├── fve_metric.py
│       └── probe_sweep.py
│
├── research_log/                       (chronological findings, ONE FILE PER DAY)
│   ├── 2026-05-19-day1.md
│   ├── 2026-05-20-day2.md
│   └── ...
│
├── findings/                           (consolidated key results, plots)
│   ├── day0_alignment.png
│   ├── day0_alignment.json
│   ├── roi_ranking.json
│   ├── fve_per_layer.json
│   └── ...
│
├── artefacts/                          (per-probe outputs — small files in git, big in gitignored subdirs)
│   ├── per_probe/
│   │   ├── L{NN}/
│   │   │   ├── png/                   ← committed (small)
│   │   │   └── mp4/                   ← gitignored (large)
│   ├── comparison_grids/              ← committed (HTML/PNG)
│   ├── hero_probes/                   ← committed (showcase)
│   └── trajectory_morphs/             ← gitignored (per-probe videos)
│
├── checkpoints/                        (gitignored — too big, .safetensors)
│   ├── L{NN}/
│   │   ├── av/
│   │   ├── ar/
│   │   └── linear_d_d.pt
│   ├── text_nla_L{NN}/
│   └── tuned_lens.pt
│
├── bin/
│   ├── h200                            (local helper that wraps the existing /Users/aryaask/Documents/Zoral/bin/h200)
│   ├── commit-push                     (one-liner: git add . && git commit -m "$1" && git push)
│   └── new-day                         (creates today's research_log/ entry from template)
│
├── .gitignore
├── requirements.txt
└── pyproject.toml
```

### .gitignore

```
checkpoints/
*.safetensors
*.pt
*.pth
artefacts/per_probe/*/mp4/
artefacts/trajectory_morphs/
__pycache__/
*.pyc
.venv/
venv/
wandb/
runs/
*.log
.DS_Store
```

### Git workflow

- Commit **at every major milestone** (after each stage finishes, after Day-0 results, etc.). Push immediately.
- Commit messages in present tense, brief: `"day-0: cross-modal alignment plot, ROI ranking committed to L16"`.
- After every training run completes, ALSO write a research_log entry summarising what was tried, what worked, what didn't, FVE numbers, and any observations.

---

## Research logging discipline

This is a **research project**. Findings are the deliverable, not just the code. We document everything.

### Daily research log

`research_log/YYYY-MM-DD-day{N}.md` — one file per day, written as the day unfolds (not in arrears). Template:

```markdown
# Day N — YYYY-MM-DD

## Goals for today
- ...

## What happened (chronological)
- HH:MM  ...
- HH:MM  ...

## Findings
- Hypothesis tested: ...
- Result: ...
- Surprise: ...

## Decisions made
- Decision: ...  Reason: ...

## Open questions / risks
- ...

## End-of-day state
- Checkpoints saved at: ...
- FVE: ...
- Next: ...
```

### Findings folder

`findings/` holds canonical, named, reproducible artefacts referenced from the research logs:
- Numerical results as JSON
- Plots as PNG with the script that generated them committed alongside
- Each finding has a 1-paragraph commentary in the corresponding day's research log

### NOTE-DOWN-EVERYTHING rule

Anything we discover during execution — surprising activation statistics, training instabilities, hyperparameter sweet spots, layer-quality patterns, malformed AV outputs, failure modes, model quirks — gets written into the day's research log within the same hour. **A discovery not written down is a discovery that didn't happen.**

---

## Architecture recap (1-paragraph reference)

Three Gemma 4 E2B instances: **TARGET** (frozen, source of `h_ℓ`), **AV** (vocab-extended with 262 stroke tokens, activation injected via `<ACT_TOKEN>` embedding surgery, autoregressively emits stroke tokens, trained with GRPO), **AR** (truncated to first ℓ layers, reads rendered PNG through Gemma 4's vision encoder, outputs `ĥ_ℓ` through `Linear(d, d)`, trained with supervised MSE). Loss is round-trip `‖h_ℓ - ĥ_ℓ‖²`. Full details in `02-architecture.md`.

---

## Anchor layer strategy

**Target: 8 anchor layers**, evenly spaced. Default candidate set for E2B (~26 layers): `{3, 6, 10, 13, 16, 19, 22, 26}`.

**Day-0 ROI ranking overrides the default**: if early layers show no cross-modal alignment (likely), drop them and redistribute density to mid-late layers where alignment exists.

**Training order is ROI rank order**: highest-ROI layer first, so we have a working artefact as early as possible.

---

## DAY 1 — Hour-by-hour (target: visible artefact by hour 24)

This day is the headline. Everything else is rollout.

```
HOUR 0 ─────────────────────────────────────────────────────────────────
  LOCAL: copy this plan file → /Users/aryaask/Desktop/residual_stream_visual_decoder/07-execution-plan.md
         (becomes the 8th doc in the folder alongside README + 01-06).
  LOCAL: git init in residual_stream_visual_decoder. First commit (the 7 design docs + this execution plan).
         Create PUBLIC GitHub repo "aryaask/residual_stream_visual_decoder" via `gh repo create`.
         Push initial commit.
         Write `research_log/2026-05-19-day1.md` (today's date).
  LOCAL: scaffold directory tree (code/, findings/, artefacts/, checkpoints/, bin/).
         Add .gitignore, requirements.txt, pyproject.toml.
         Commit + push.

HOUR 0-1 ───────────────────────────────────────────────────────────────
  REMOTE: SSH to H200. Smoke test:
            CUDA_VISIBLE_DEVICES=6,7 python -c "
              from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
              m = AutoModelForCausalLM.from_pretrained('google/gemma-4-e2b-it', torch_dtype='bfloat16').cuda()
              t = AutoTokenizer.from_pretrained('google/gemma-4-e2b-it')
              # text forward
              # image forward (if vision encoder bundled)
            "
          Verify CUDA_VISIBLE_DEVICES isolation, free memory, vision encoder loads.
  LOCAL: write stroke_tokenizer.py (Cartesian stroke-5, 128-bin), render.py (PIL/Cairo, PNG + MP4).
         Unit-test both with hand-crafted stroke sequences. Commit + push.

HOUR 1-2 ───────────────────────────────────────────────────────────────
  LOCAL: write code/eval/day0_alignment.py (100 concepts × 26 layers cosine sim, plot)
         write code/eval/roi_ranking.py (alignment + variance + entropy → ROI score)
         bin/h200 sync; bin/h200 run day0_alignment.py
  REMOTE: ~30 min compute. Outputs findings/day0_alignment.png + .json
  LOCAL: pull artefacts. INSPECT THE PLOT. Write the result into research_log.
         Decision point: which layer is the top-1 ROI? Commit that layer choice in code (constant LAYER_TOP1).
         Commit + push.

HOUR 2-3 ───────────────────────────────────────────────────────────────
  LOCAL: write code/data/iam_loader.py, quickdraw_loader.py, activation_extractor.py.
         Kick off QuickDraw download on the Mac in background (~30 GB).
         Initiate IAM On-Line access request (login-walled, may need human approval).
  REMOTE: kick off activation corpus extraction in background (Gemma 4 forward on 100K text snippets at all sweep layers, save to ~/Aryaa/data/activations/).
         Estimated ~3-4 hours background time on 1 GPU.
  LOCAL: write code/lenses/tuned_lens.py.
  REMOTE: kick off tuned-lens training on the other GPU (~30 min).
  Commit + push.

HOUR 3-4 ───────────────────────────────────────────────────────────────
  LOCAL: write code/av/vocab_extend.py (Anole-style, add 262 stroke tokens to Gemma 4),
         code/av/activation_injection.py (<ACT_TOKEN> embedding replacement),
         code/av/stroke_decoder.py (AV wrapper class),
         code/ar/reconstructor.py (truncated Gemma 4 + Linear(d,d) wrapper),
         code/train/stage1_av_sft.py.
  Process QuickDraw + (if available) IAM into ~50K (caption, strokes) SFT pairs. Write to ~/Aryaa/data/sft.jsonl.
  Commit + push.

HOUR 4-6 ───────────────────────────────────────────────────────────────
  REMOTE: launch Stage 1 (AV SFT) at LAYER_TOP1.
            CUDA_VISIBLE_DEVICES=6,7 python -m code.train.stage1_av_sft \
              --layer LAYER_TOP1 --epochs 2 --batch 64
          Both H200 GPUs via FSDP. Logs streamed to ~/Aryaa/runs/stage1_L{NN}/.
          Estimated wall clock: ~2 hours (Anole-style PEFT, ~10M trainable params, ~50K examples).
  LOCAL: write code/train/stage2_ar_supervised.py while Stage 1 runs.
         Write code/eval/probe_sweep.py.
         Write code/eval/fve_metric.py.
  Hour 6: pull Stage 1 checkpoint, sanity-check by sampling drawings from text prompts.
         Save sample PNGs to findings/stage1_samples_L{NN}.png. Research log entry. Commit + push.

HOUR 6-9 ───────────────────────────────────────────────────────────────
  REMOTE: launch Stage 2 (AR supervised) at LAYER_TOP1.
            CUDA_VISIBLE_DEVICES=6,7 python -m code.train.stage2_ar_supervised \
              --layer LAYER_TOP1 --epochs 2 --batch 32
          Generates synthetic drawings from AV (frozen at Stage 1) for each activation, trains AR.
          Estimated wall clock: ~2-3 hours.
  LOCAL: write code/train/stage3_av_grpo.py. This is the biggest piece of code; review against NLA reference repo (kitft/natural_language_autoencoders).
  Hour 9: pull AR checkpoint, measure baseline FVE before RL (should be already > random thanks to Stage 2).
         Research log entry. Commit + push.

HOUR 9-22 ──────────────────────────────────────────────────────────────
  REMOTE: launch Stage 3 (AV GRPO) at LAYER_TOP1.
            CUDA_VISIBLE_DEVICES=6,7 python -m code.train.stage3_av_grpo \
              --layer LAYER_TOP1 --group-size 8 --kl-beta 0.05 --steps 5000
          GRPO with KL penalty to Stage 1 init. AR continues training in parallel.
          Reward: -log ||h - ĥ||². Save checkpoints every 500 steps.
          Monitor FVE on a held-out 1K-sample validation set every 500 steps.
          Estimated wall clock: ~12-13 hours on 2× H200 (probably can't squeeze 30 GPU-hrs into 15 wall-clock hours with only 2 GPUs; we accept this and run ~5000 steps instead of 10000).
          If FVE plateaus before then, stop early. Document in research log.
  LOCAL: write code/eval/render_probe_set.py.
         Curate the 50-probe set (50 prompts across categories: factual / arithmetic / multi-hop / emotional / ambiguous / code / list / negation).
         Save as findings/probe_set.json. Commit + push.

HOUR 22-23 ─────────────────────────────────────────────────────────────
  REMOTE: pull final Stage 3 checkpoint (`av_L{NN}_grpo_final.safetensors`, `ar_L{NN}_grpo_final.safetensors`).
  REMOTE: run probe_sweep.py over the 50-probe set.
            For each probe:
              h = target.forward(probe_text).activations[LAYER_TOP1]
              strokes = AV.generate(activation=h)
              png, mp4 = render(strokes, save_animation=True)
            Save 50 PNGs + 50 MP4s to artefacts/per_probe/L{NN}/.
          Estimated wall clock: ~30 min.
  LOCAL: pull artefacts. Compute FVE on the 50 probes.
         Inspect drawings visually. Note surprising / failed / striking ones.

HOUR 23-24 ─────────────────────────────────────────────────────────────
  LOCAL: build a quick HTML index page (`artefacts/per_probe/L{NN}/index.html`) showing all 50 PNG+MP4 side by side with their prompts.
         Cherry-pick 5 hero probes. Polish those into the start of artefacts/hero_probes/.
         Write findings/fve_per_layer.json (just the L{NN} entry for now).
         Update research_log/2026-05-19-day1.md with full results + reflections.
         Commit EVERYTHING. Push. Tag the commit `day1-complete`.

END OF DAY 1 — DELIVERABLE
  ✓ One trained visual NLA at LAYER_TOP1 (highest ROI)
  ✓ 50 probes rendered as PNG + animated MP4
  ✓ FVE measured and recorded
  ✓ Tuned lens trained across all 26 layers (supplementary)
  ✓ Day-0 alignment plot (the go/no-go gate result)
  ✓ All code committed and pushed
  ✓ Research log filled in
  ✓ Tagged commit `day1-complete` on GitHub
```

**If the 48-hour key window expires here, this is a shippable v0.**

---

## DAYS 2-7 — Rolling completion

### Day 2 (2026-05-20)

**Goal:** start 3 more anchor layers + kick off text-NLA baseline.

```
MORNING
  Confirm Day 1 results visible in artefacts/. Brief retro in research_log/2026-05-20-day2.md.
  
  REMOTE: kick off Stage 1 (AV SFT) for next 3 highest-ROI anchor layers in parallel.
          Each takes ~2 hours wall clock, on 2 GPUs we serialise → ~6 hours total.
          Alternative: train all 3 sequentially with FSDP using both GPUs each → also ~6 hours but cleaner.

AFTERNOON
  REMOTE: launch text-NLA baseline at mid-stack layer (e.g., L16). Same Stage 1 SFT (text output, no vocab extension).
          This is cheap relative to visual (no new vocab to learn), maybe 1 hour.
  
  REMOTE: as Stage 1 completes for each of the 3 new layers, immediately kick off Stage 2 (AR supervised).

EVENING
  Begin Stage 3 (AV GRPO) for the 3 new layers + text-NLA. Will run overnight + into Day 3.
  
  LOCAL: write code/eval/comparison_grid.py — generates per-probe per-layer 4-column grids (logit / tuned / text-NLA / visual-NLA).
  
  Commit + push. Research log. Tag `day2-complete`.
```

### Day 3 (2026-05-21) — **KEY EXPIRY DAY**

**Critical: key expires 08:16 UTC.** If renewal hasn't happened, we lose remote access at that hour.

```
PRE-EXPIRY (before 08:16 UTC)
  REMOTE: pull EVERYTHING that's been trained so far (all checkpoints, all runs, all logs).
  Confirm everything is in GitHub.

POST-EXPIRY OPTIONS
  Option A: Key renewed → continue as planned.
  Option B: Key not renewed → continue Days 3-7 work LOCAL ONLY:
    - Process artefacts we already pulled
    - Build comparison grids, hero probes, trajectory morph videos
    - Writeup
    - Visual NLA artefact is still publishable at the 4 layers we trained
```

**Assuming Plan A:** continue rolling out the remaining 4 anchor layers. Same pattern as Day 2 but in parallel with finalising the layers from Day 2.

### Days 4-5

Finish Stage 3 for all 8 visual anchor layers + text-NLA. Validate FVE across the layer sweep. Update `findings/fve_per_layer.json`.

### Day 6

Render the full 50-probe × 8-layer matrix:
- 400 PNGs + 400 MP4s for visual NLA
- 50 text outputs × 8 layers for text-NLA at L16 only (just 50 entries, since text-NLA is single-layer)
- All in `artefacts/per_probe/`

Build all comparison grids. Per probe, per layer: 4-column (logit / tuned / text-NLA / visual-NLA). Output to `artefacts/comparison_grids/`.

Build trajectory-morph videos: per probe, one video that morphs through the 8 layers' drawings smoothly. Output to `artefacts/trajectory_morphs/`.

### Day 7

Cherry-pick 10 hero probes. High-fidelity render of each (higher canvas resolution, longer MP4s, anti-aliasing dialed up).

Assemble 60-second hype reel: best 10-15 morph clips edited together with brief on-screen captions.

Writeup: blog-post format. Sections: motivation → architecture → key result (the trajectory grid) → quantitative (FVE table) → qualitative analysis of probes → comparison with text-NLA → limitations and future work.

GitHub release: tag `v0.1`. README updated with one-line install + one-line "generate a drawing for an arbitrary prompt" command.

Tag commit `complete`.

---

## Critical risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **SSH key expires before key renewal** | Medium-High | Day 1 must produce shippable artefact within first 24 hours. All checkpoints pulled to Mac immediately. Plan B path documented above. |
| **Day-0 alignment fails (no cross-modal overlap at any layer)** | Low-Medium | This is the project's go/no-go gate. If it fails, pivot to text-NLA only (Anthropic didn't release small-model NLAs, that's still novel). Document and ship. |
| **GPU preemption (spot instance)** | Medium | Checkpoint every 500 GRPO steps. Resume training from latest checkpoint is supported. Document the resume command in research log. |
| **FVE plateaus low** | Medium | Mitigations in order: increase LoRA rank, swap `Linear(d, d)` → 2-3 layer MLP, increase GRPO group size, fine-tune vision encoder on QuickDraw line art. Try one at a time, document results. |
| **AV reward hacking (generates gibberish that fools AR but isn't drawing-like)** | Low-Medium | Increase KL penalty β. Add a small discriminator loss for "drawing-like" using a fixed CLIP image embedding similarity to QuickDraw. |
| **AV converges to writing the concept as block letters** (because AR's frozen Gemma 4 backbone can OCR) | Medium | **Not a true risk — it's an interesting alternative outcome.** If β is too small the AV may discover that writing `cat` as strokes is the highest-bandwidth encoding. We get a different but still novel result ("Gemma 4 verbalises its activations as written text via its own motor channel"). To prevent: increase β. To encourage: decrease β. |
| **IAM On-Line login approval is slow** | Low | Plan does not depend on IAM (QuickDraw + synthetic captions are sufficient for SFT). IAM is a nice-to-have for cursive handwriting style. |
| **Run out of compute mid-week** | Medium | 6 layers instead of 8 if needed. Or 4 with denser training each. Highest-ROI layer always completes first. |
| **Vision encoder produces garbage on sparse line art** | Medium | If Day-0 shows poor alignment specifically because images are off-manifold for the vision encoder, do a 1-hour fine-tune of the vision encoder on QuickDraw rendered the same way our AV will render. |

---

## Verification — how to know each milestone landed

| Stage | Verification command |
|---|---|
| **Repo set up** | `cd ~/Desktop/residual_stream_visual_decoder && git log --oneline` shows initial commit. `git remote -v` shows GitHub remote. |
| **Day-0 alignment** | `findings/day0_alignment.png` exists. Cosine alignment for top layer beats control by > 0.1. |
| **ROI ranking** | `findings/roi_ranking.json` exists with 26 layers ranked. Top-1 layer documented in research_log. |
| **Tuned lens** | `checkpoints/tuned_lens.pt` exists. Sample logit predictions at layer 16 differ visibly from vanilla logit lens (more coherent across layers). |
| **Stage 1 AV** | `checkpoints/L{NN}/av/` exists. `python -m code.av.stroke_decoder --prompt "a cat"` produces a stroke sequence that renders to a recognisable cat (visual eyeball test). |
| **Stage 2 AR** | `checkpoints/L{NN}/ar/` exists. MSE on held-out activations < 0.5 × Var(h). |
| **Stage 3 GRPO** | FVE on held-out > 0.2 minimum (anything is better than text-NLA at 0.5 is gravy). Drawings remain visually coherent (not collapsed to noise). |
| **Day-1 probe sweep** | `artefacts/per_probe/L{NN}/` has 50 PNGs + 50 MP4s. `index.html` opens in browser and renders correctly. |
| **Each subsequent layer** | Same as Day-1 but for that layer's directory. |
| **Comparison grids** | `artefacts/comparison_grids/` has one HTML per probe with 4 columns × 8 layers populated. |
| **Hero probes** | 10 PNG + MP4 sets in `artefacts/hero_probes/` at higher fidelity than the regular probes. |
| **Hype reel** | `demo.mp4` at repo root, ~60 seconds. |
| **Writeup** | `WRITEUP.md` at repo root. Contains: motivation, architecture, FVE table, screenshots of trajectory grids, link to demo.mp4, limitations. |

---

## What to read for context (during execution)

- **The 7 design docs** at `/Users/aryaask/Desktop/residual_stream_visual_decoder/*.md`
- **`.ops/h200-gcp.md`** — H200 access, GPU allocation, `bin/h200` helper
- **`/Users/aryaask/Documents/Zoral/bin/h200`** — the actual h200 helper script we'll wrap
- **NLA reference repo** — github.com/kitft/natural_language_autoencoders — `nla_inference.py`, `nla/train_actor.py`, `nla/reward.py`, `nla/loss.py`, `configs/` for the GRPO recipe
- **Anole repo** — github.com/GAIR-NLP/anole — vocab extension pattern (~30 lines)
- **Cursive Transformer** — github.com/greydanus/cursivetransformer — sanity reference for stroke tokenisation patterns (though we use Cartesian, not polar)

---

## Storage strategy

Storage is split three ways: **Mac (small, durable, source of truth for code & critical artefacts)**, **GCP H200 (large, ephemeral, source of truth for raw data & in-flight training)**, **GitHub + HF Hub (durable, off-site backup)**.

### Volumetric breakdown (estimates)

| Asset | Size | Lives on |
|---|---|---|
| Source code | ~50 MB | Mac (committed to git) |
| Gemma 4 E2B model weights | ~5-10 GB | GCP only (HF cache, redownloadable) |
| QuickDraw raw NDJSON | ~30 GB | GCP only |
| QuickDraw processed subset | ~5 GB | GCP, optional Mac mirror |
| IAM On-Line | ~500 MB | GCP, optional Mac mirror |
| Activation corpus (8 layers × 100K samples) | ~5-10 GB | GCP only |
| SFT corpus (caption, strokes) | ~1 GB | GCP, Mac mirror |
| AV checkpoints (LoRA + new vocab, per layer) | ~50 MB each × 8 = ~400 MB | GCP + Mac (small enough) |
| AR checkpoints (LoRA + Linear(d,d), per layer) | ~100 MB each × 8 = ~800 MB | GCP + Mac |
| Tuned lens (26 Linear(d,d) adapters) | ~400 MB total | GCP + Mac |
| Per-probe PNGs (50 × 8 layers) | ~10 MB total | GCP + Mac + git |
| Per-probe MP4s (50 × 8 layers) | ~200 MB total | GCP + Mac (gitignored, kept locally) |
| Hero probes (10 polished) | ~50 MB | GCP + Mac + git |
| Comparison grids HTML | ~5 MB | Mac + git |
| Writeup + plots | ~10 MB | Mac + git |

**Total Mac requirement:** ~3-5 GB to mirror the critical artefacts. **Total GCP requirement:** ~50 GB peak (well within the 316 GB free).

### Pull discipline (the failsafe loop)

After **every** stage completes on a layer:

```
1. bin/h200 pull checkpoints/L{NN}/  → ~/Desktop/residual_stream_visual_decoder/checkpoints/L{NN}/
2. bin/h200 pull artefacts/per_probe/L{NN}/  → ~/Desktop/residual_stream_visual_decoder/artefacts/per_probe/L{NN}/
3. cd ~/Desktop/residual_stream_visual_decoder && git add -A && git commit -m "..." && git push
```

Mac mirrors EVERYTHING worth keeping. Git holds the small/source/results. GCP holds the raw datasets and in-flight training.

The Mac thus always has the LATEST snapshot of every completed layer, plus all source code, plus all small artefacts pushed to GitHub.

### Loss-of-SSH plan (Plan B again)

If we lose H200 access mid-week:
- Mac has: all source code, all completed-layer checkpoints, all rendered probe artefacts up to the loss moment.
- We can: render new probes locally from existing checkpoints (slow but possible on Mac), build comparison grids, write the writeup, ship v0.
- We cannot: train new layers, change architecture.

The pull-after-each-stage discipline guarantees no completed work is stranded on the H200.

### Confirmed user decisions (2026-05-19)

- **Mac has plenty of free space (20+ GB).** Mirror everything important locally: all final checkpoints per layer, processed SFT corpus subset, all per-probe PNGs and MP4s, hero artefacts, comparison grids. Skip only the raw QuickDraw NDJSON (~30 GB, stays on GCP only).
- **No HuggingFace Hub backup.** Rely on GitHub (for code + small artefacts) + Mac (for everything except raw datasets) + GCP (until SSH expires).
- **GitHub repo is PUBLIC from day 1.** Building in public. Standard research workflow with full transparency.

---

## Working agreements

1. **Use H200 GCP instance for all training**. Always `CUDA_VISIBLE_DEVICES=6,7`. Use `bin/h200 run` and `bin/h200 sync` for deploy.
2. **Source code lives on Mac; data and training live on GCP**. Rsync source via `bin/h200 sync`; pull artefacts via `bin/h200 pull`.
3. **Commit + push at every milestone**. No work sits uncommitted for more than an hour.
4. **Pull checkpoints + artefacts to Mac immediately after each training run completes**. The H200 is ephemeral.
5. **Research log is mandatory and continuous**. Anything surprising goes in the same hour it happens.
6. **Highest-ROI layer first, always**. Don't reorder for convenience.
7. **If a stage exceeds 2× its time budget, stop and assess**. Don't burn compute on a stuck run.
8. **All hyperparameters documented in research log**. Future-us must be able to reproduce.
9. **GitHub repo: `aryaask/residual_stream_visual_decoder` (PUBLIC from day 1)**. Build in public.
