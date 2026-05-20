# Residual Stream Visual Decoder

**Draw what Gemma 4 is thinking.**

![gallery](artefacts/v1_2/gallery.png)

Each panel above is a real drawing produced by sampling stroke tokens from an
*Activation Verbalizer* that received Gemma 4 E2B's residual-stream activation
at layer 12, injected via embedding-layer surgery. The drawings are not
generated from the prompt's text — they come from the *internal vector* the
model is processing when you ask it to think about a cat.

**v1.1 vs v1.2 — same model, same prompts, same injection layer:**

![Before / after](artefacts/v1_2/before_after_small.png)

Top row (v1.1): the architecture worked end-to-end but the AV had no learnable
surface to interpret the injected activation. Outputs were abstract structure.

Bottom row (v1.2): added a learnable activation projector + LoRA on the AV's
first 8 language layers + a supervised "activation → real drawing" Stage 1.5
training stage. Cat-shaped cats, dog-shaped dogs, flower-shaped flowers.

<!-- The hype reel + per-token + cross-layer galleries are inserted by build_index.py after training. -->

<p align="center">
  <video src="demo.mp4" width="640" controls muted loop autoplay></video>
</p>

```
┌─────────────────────┐  h_ℓ ∈ ℝ¹⁵³⁶  ┌────────────────────────┐
│ Gemma 4 E2B (frozen)│ ────────────▶ │ AV: stroke decoder      │
│  process prompt     │                │ vocab +262 stroke tokens│
└─────────────────────┘                │ activation injected at  │
                                       │ <ACT_TOKEN> embedding   │
                                       └────────┬───────────────┘
                                                │ stroke tokens
                                                ▼
                                       ┌────────────────┐
                                       │ deterministic  │  PNG + animated MP4
                                       │ renderer       │
                                       └────────┬───────┘
                                                ▼
                                       ┌─────────────────────────┐
                                       │ AR: truncated Gemma 4   │
                                       │ + backbone LoRA + Linear│
                                       │ reads PNG via vision    │
                                       └────────┬───────────────┘
                                                │ ĥ_ℓ
                                                ▼
                          MSE(h_ℓ, ĥ_ℓ) drives the iterative refinement loop
```

The recipe is Anthropic's [Natural Language Autoencoders (2026)](https://transformer-circuits.pub/2026/nla/index.html)
adapted from text output to vector strokes:

1. **Activation Verbalizer (AV)**: Gemma 4 E2B with 262 stroke tokens added to its vocabulary.
   The target activation is injected by overwriting `<ACT_TOKEN>`'s embedding row.
2. **Reconstructor (AR)**: truncated Gemma 4 E2B (first ℓ layers) + a custom LoRA on every
   attention `q/k/v/o_proj` (including the vision tower) + a `Linear(d, d)` head.
3. **Iterative joint training**: alternate AR supervised steps (with fresh AV-generated
   buffer to track distribution) and AV GRPO steps (with frozen AR providing reward).
   Five outer iterations.

## Try it on your own prompt

```bash
python demo.py "The capital of France is"
python demo.py "I am thinking about a dog." --layer 12 --open
python demo.py "What is 47 + 38?" --layer 12
```

Outputs land under `demo_output/<slug>/` — `drawing.png` (224×224, what AR saw),
`drawing_4x.png` (896×896, vector-rerendered for display), `drawing_4x.mp4`
(stroke-by-stroke animation).

## What's in this repo

| Path | Contents |
|---|---|
| `code/verbalizer/` | AV: vocab extension + activation injection + sampling |
| `code/ar/` | AR: truncated Gemma + Linear head + custom LoRA on Gemma4ClippableLinear |
| `code/train/` | Stage 1 SFT, Stage 2 v1/v2/v3/v4 ablations, Stage 3 GRPO, **Stage 4 iterative (v1.0)** |
| `code/eval/` | inject_demo, compare_av, measure_fve, alpha_sweep, activation_geometry, token_trajectory, cross_layer_trajectory, build_index |
| `code/lenses/` | logit lens + tuned lens (per-layer text-side baselines) |
| `code/render.py` | Deterministic Cartesian stroke-5 renderer with vector upscaling |
| `code/stroke_tokenizer.py` | 262-token stroke vocabulary (Δx, Δy, pen-state) |
| `code/tests/` | 32 unit tests (tokenizer, renderer, LoRA) |
| `bin/h200` | Sync/run/bg/pull helper for the GCP H200 instance |
| `WRITEUP.md` | The full v1.0 findings document |
| `INDEX.html` | Browsable landing page for all artefacts |
| `demo.py` | One-click demo: prompt → drawing |
| `artefacts/` | Per-probe drawings + per-token + cross-layer MP4s |
| `findings/` | Plots, JSON metrics, alpha sweep, FVE measurements |
| `research_log/` | Chronological journal of every experiment |
| `01-vision.md` … `07-execution-plan.md` | The original design docs |

## How it was built (3-day sprint)

- **v0.1** (Day 1-2): full architecture standup. FVE = 0 across 5 AR variants (Linear/MLP/centered/contrastive). Bottleneck identified.
- **v1.0** (Day 3-): custom LoRA on `Gemma4ClippableLinear` + iterative joint AR + AV training (the genuine NLA recipe). Unlocked the FVE bottleneck.

Read `WRITEUP.md` for the full story including all five engineering discoveries:
alpha tuning (1.0 → 0.5), AR data quality matters more than AR training length,
L16 is one of the most-clustered layers (use L3 / L12 / L24 instead),
frozen-backbone AR cannot break FVE 0 regardless of training scheme,
iterative joint training unlocks per-prompt discrimination.

## Reproduce

```bash
git clone https://github.com/AryaaSk/residual_stream_visual_decoder.git
cd residual_stream_visual_decoder
pip install -r requirements.txt
# Pull our trained checkpoints (TBD: Hugging Face release)
# Then:
python demo.py "your prompt here"
```

Training scripts live under `code/train/`. The v1.0 trainer is
`code/train/stage4_iterative.py`. See `07-execution-plan.md` for the full
recipe and hyperparameters.

## License

Code: MIT.
Checkpoints: derive from Gemma 4 E2B which is under [Google's Gemma Terms of Use](https://ai.google.dev/gemma/terms).
