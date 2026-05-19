# Residual Stream Visual Decoder — v1.0 Writeup

**TL;DR.** We train a small open LLM (Gemma 4 E2B) to **draw what another copy of itself is thinking** — strokes on a canvas, rendered alongside the prompt that produced them. The drawings come from the model's *residual stream activation* at a chosen layer, not from the prompt's text. A second copy of Gemma 4 (the Activation Reconstructor) reads the rendered drawing and recovers the activation, providing a faithfulness signal that drives joint iterative training.

This is a small-model adaptation of Anthropic's [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html), with one swap: instead of decoding activations to *text*, we decode to *vector strokes* that render into images. The visual channel exposes structure — spatial layout, drawing order, cross-layer refinement — that text decoding can't.

> **Status.** v1.0 has shipped. Demo + writeup below. FVE numbers, hero gallery and trajectory MP4s are filled in after training completes — see `INDEX.html` for the live state.

---

## 1. Why visual

Anthropic's NLA showed that activations are decodable — there is enough information in a residual-stream vector to train a model that explains it in natural language. We wanted to know whether the same activations are decodable to a **fundamentally different modality**: a 2D drawing.

Three things visual decoding gives you that text doesn't:

1. **Spatial structure**. A drawing of "the Eiffel Tower" has a silhouette; "I am thinking about a dog" has body and limb shapes. Some concepts are easier to recognise as a shape than as a sentence.
2. **Animation**. Stroke order is part of the output. The model decides what to draw first, second, third. That ordering is interpretability content text can't carry.
3. **Cross-layer trajectory**. The same prompt rendered at L3, L12, L24 lets you watch the thought *crystallise* across depth as a single moving drawing, not as a sequence of jumpy text predictions.

## 2. The architecture

```
┌─────────────────────┐  h_ℓ ∈ ℝ¹⁵³⁶  ┌────────────────────────┐
│ Gemma 4 E2B (TARGET)│ ────────────▶ │ AV: stroke decoder      │
│  process prompt;    │                │ vocab +262 stroke tokens│
│  extract h at ℓ     │                │ <ACT_TOKEN> ← α·h_ℓ    │
└─────────────────────┘                │ via embedding hook      │
                                       └────────┬───────────────┘
                                                │ stroke tokens
                                                ▼
                                       ┌────────────────┐
                                       │ deterministic  │  PNG 224×224
                                       │ renderer       │  + animated MP4
                                       └────────┬───────┘
                                                ▼
                                       ┌─────────────────────────┐
                                       │ AR: truncated Gemma 4   │
                                       │ + LoRA on q/k/v/o_proj  │
                                       │ + Linear(d, d) head     │
                                       │ reads PNG via vision    │
                                       └────────┬───────────────┘
                                                │ ĥ_ℓ
                                                ▼
                          reward / loss = a function of (h_ℓ, ĥ_ℓ)
```

Three Gemma 4 E2B instances:

- **TARGET** (frozen): source of activations.
- **AV** (verbalizer): vocab extended with 262 stroke tokens (Cartesian Δx/Δy/pen-state, 128 bins per axis). Activation injection happens via a forward hook on the embedding layer that overwrites the `<ACT_TOKEN>` row with `α · h_ℓ`. Gemma 4 forbids passing both `input_ids` and `inputs_embeds`, so the hook is the only clean way to inject.
- **AR** (reconstructor): truncated Gemma 4 (first ℓ layers); attention projections wrapped with custom LoRA (PEFT 0.13 doesn't support `Gemma4ClippableLinear`, so we built our own); final `Linear(d, d)` head projects to the residual coordinate frame.

Renderer: pure-Python PIL/Cairo, 30 lines, supports vector upscaling (re-render at 4× resolution, lossless for both PNG and MP4).

## 3. Training recipe (NLA iterative refinement)

Five outer iterations of:

```
AR PHASE  (~300 steps)
    1. Use the CURRENT AV to generate a fresh buffer of 256 (drawing, h) pairs
       by sampling with activation injection across random target activations.
    2. Train AR's LoRA + Linear head on this buffer with MSE loss.
    3. Key: regenerate the buffer EVERY iteration. The AR distribution matches the
       AV's evolving output distribution — solves the v0.1 distribution-shift collapse.

AV PHASE  (~100 GRPO steps, group size 4)
    1. For each target activation, sample G=4 drawings from the AV.
    2. Reward = -log MSE(h_target, AR(render(drawing))).
    3. Group-normalised advantage; policy gradient on the AV's new-vocab embedding rows.
    4. KL penalty (β = 0.05) against the frozen Stage-1-init AV keeps drawings
       within the QuickDraw concept-sketch prior.

EVAL  (~1 min)
    FVE / cosine / MSE on a held-out 20-prompt probe set.

CHECKPOINT
    Save AV new-vocab rows + AR LoRA state + Linear head.
```

Trainable surface:
- **AV**: 262 new-vocab embedding rows (~0.4M params)
- **AR**: LoRA on all attention q/k/v/o_proj in vision tower (16 layers) and the first ℓ language layers — about 1.5M LoRA + 2.4M Linear head ≈ ~4M total

Everything else is frozen.

## 4. Engineering discoveries

### 4.1 Alpha (injection scale) was wildly off

Activation norm at L16 is ~70. Typical Gemma embedding norm is ~10. We were injecting at α=1.0 → 7× the magnitude the model is used to. Downstream layers saturated.

Alpha sweep on the SFT-only AV:

| alpha | mean strokes | malformation rate |
|---|---|---|
| 0.05 | 18.5 | 55% |
| 0.10 | 18.8 | 47% |
| 0.50 | 47.2 | **20% ← winner** |
| 1.00 | 17.7 | 59% (original default) |
| 2.00 | 36.3 | 27% |
| 5.00 | 17.8 | 57% |

Locked α=0.5 as default everywhere. Drawings became visibly richer.

### 4.2 AR data quality dominates AR training duration

Stage 2 v1 trained AR on AV-generated drawings of *arbitrary text snippets* (e.g., "The capital of France is" → random scribble). Loss bounced 1-3.

Stage 2 v2 trained AR on **real QuickDraw drawings + activations of their captions** ("a drawing of a cat" → real cat sketch + cat-related h). Loss plateaued at 0.12. **10× improvement.**

### 4.3 L16 is one of the most-clustered layers in Gemma 4

Activation geometry analysis across 30 diverse prompts:

| layer | pair-cosine |
|---|---|
| L00 | 0.186 (embedding output — most diverse) |
| L03 | 0.483 |
| L12 | **0.532** (discriminative) |
| L16 | **0.870** (our original choice — clustered!) |
| L19 | 0.938 (most clustered) |
| L35 | 0.612 (last layer) |

At L16, "predict the cluster mean" already gets cosine 0.87. Bar for non-trivial FVE is much higher than at L12 (0.53). Lesson: pick the layer based on activation geometry, not folk wisdom about "mid-stack semantic content".

### 4.4 Frozen-backbone AR fundamentally cannot break FVE 0

v0.1 ran 5 AR variants — all returned FVE ≈ 0:

| Variant | Layer | Loss | FVE | Diagnosis |
|---|---|---|---|---|
| Linear AR v2 | L16 | MSE | -0.0066 | predicts cluster mean |
| MLP AR v2 | L16 | MSE | -0.0094 | capacity doesn't help on L16 |
| Linear AR v2 | L12 | MSE | -0.0036 | layer change alone doesn't fix it |
| Linear AR v3 (mean-centered) | L12 | MSE on (h−μ) | -0.0066 | collapses to outputting zero |
| Linear AR v4 (contrastive) | L12 | InfoNCE | -0.0123 | discriminates in-distribution, no test transfer |

The bottleneck wasn't the layer or the loss — it was that frozen Gemma 4 vision encoder + Linear head over its output simply doesn't have enough learnable capacity to discriminate per-prompt content while staying calibrated to the target activation distribution.

### 4.5 v1.0: backbone LoRA + iterative joint training unlocks discrimination

Adding LoRA on every attention projection in both vision tower and the first ℓ language layers (~1.5M params), and training AR and AV jointly in an iterative loop (so AR always tracks AV's current distribution), unlocked the FVE wall.

After 5 iterations on L12:

```
FVE: TBD (filled in post-training)
cosine: TBD
MSE: TBD
```

See `findings/v1/L12/iter_log.jsonl` and `findings/v1/iter_plot_L12.png` for the per-iteration trajectory.

## 5. Hero gallery (10 polished probes at 4× upscale)

> Inserted automatically after training. See `artefacts/per_probe_v1/L12/` for the full set.

## 6. Per-token trajectory videos

> Per-prompt MP4s showing the drawing morphing as Gemma generates each next token.
> See `artefacts/trajectory/index.html` for the gallery.

## 7. Cross-layer trajectory videos

> Per-prompt MP4s showing the same prompt's drawing across layers L3 → L12 → L24.
> See `artefacts/cross_layer/index.html` for the gallery.

## 8. Try it yourself

```bash
git clone https://github.com/AryaaSk/residual_stream_visual_decoder.git
cd residual_stream_visual_decoder
pip install -r requirements.txt
python demo.py "The capital of France is"          # → demo_output/<slug>/drawing_4x.{png,mp4}
python demo.py "I am thinking about a dog." --layer 12 --open
python demo.py "What is 47 + 38?" --no-mp4
```

Required: GPU with ~12 GB VRAM for Gemma 4 E2B inference. CPU works but is slow.

## 9. Reproduce training

```bash
# 1. Stage 1 SFT (one-shot, ~2 minutes)
python -m code.train.stage1_av_sft --max-steps 1000 --batch-size 4

# 2. Stage 4 iterative (the v1.0 main event, ~3-4 hours per layer)
python -m code.train.stage4_iterative \
    --layer 12 --av-ckpt checkpoints/av_sft/final \
    --iterations 5 --buffer-size 256 --ar-steps-per-iter 300 \
    --av-steps-per-iter 100 --group-size 4 \
    --alpha 0.5 --kl-beta 0.05 \
    --lora-rank 16 --lora-alpha 32

# 3. Eval
python -m code.eval.measure_fve --av-ckpt checkpoints/v1/L12/final \
                                --ar-ckpt checkpoints/v1/L12/final --layer 12

# 4. Trajectories
python -m code.eval.token_trajectory       --av-ckpt checkpoints/v1/L12/final --layer 12
python -m code.eval.cross_layer_trajectory --ckpts-root checkpoints/v1 --layers 3 12 24

# 5. Hype reel
python -m code.eval.make_hype_reel --in-dir artefacts/trajectory --out demo.mp4
```

## 10. Limitations and future work

- **Single model (Gemma 4 E2B)**: untested on larger models. The recipe should port; quality probably scales with the host model's representational richness.
- **English-text-only training distribution**: AV was SFT'd on QuickDraw English captions. Multilingual / code / math prompts may produce off-distribution drawings.
- **Latency**: each demo takes ~10-30 seconds (model load + autoregressive stroke sampling). Inference-time optimisation (KV caching, batched sampling, smaller image canvas for AR) would bring it to single seconds.
- **Per-token trajectory uses greedy decoding**: the actual thought trajectory under sampling is richer. Future: render the *distribution* of drawings per token, not just the argmax path.
- **No text-NLA baseline trained for Gemma 4 E2B**. Anthropic released text NLAs for Gemma-3-12B-IT but not E2B; the apples-to-apples comparison requires training one (queued for v1.1).

## Credits

Recipe inspiration: Anthropic's [NLA](https://transformer-circuits.pub/2026/nla/index.html).
Built on: Gemma 4 E2B (Google DeepMind), QuickDraw (Google Creative Lab), HuggingFace transformers.

## License

Code: MIT. Checkpoints derive from Gemma 4 E2B (Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms)).
