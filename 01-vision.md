# 01 — Vision

## What we are building

A **visual decoder of LLM residual streams**. Given any activation `h_ℓ` from a frozen Gemma 4 E2B at layer ℓ, our system produces a hand-drawn-style image that is **faithful** to the activation in the sense that another copy of Gemma 4 (the AR) can read the image and reconstruct `h_ℓ`.

The output is a sequence of pen strokes, rendered to a 224×224 PNG. Because the output is procedural (not a one-shot raster), we can also save the **animation** of the drawing emerging, which is itself an interpretability artefact.

## Why this is novel

The closest published work:

- **Visual Sketchpad** (Hu et al. 2024): LLMs given a Python draw API as a tool. Improves multimodal reasoning (+8-13% on geometry / spatial benchmarks). Drawing is a *tool call*, not a native output channel.
- **Anthropic NLA** (2026): trains LLMs to *verbalize* their own activations as text. Direct architectural ancestor of this project, but text-output only.
- **Multimodal Visualization-of-Thought (MVoT)** (2025): autoregressive raster image tokens interleaved with reasoning text. Same family but raster, not strokes; large model, not small.
- **MathCanvas** (2025): two-stage training for interleaved visual-textual reasoning. Pixel diagrams, not strokes.

The unoccupied square: **small open LLM × vector stroke output × for interpretability of internal states**. Nobody has shipped this. It's genuinely novel territory.

## Why strokes (not raster)

The decoder *could* emit raster pixels (Chameleon / Anole / Transfusion-style) or vector strokes (Cursive Transformer / SketchRNN-style). We chose strokes for four reasons:

1. **Animation comes for free.** A stroke sequence is intrinsically temporal. We save every intermediate frame and render an MP4 alongside the final PNG. A raster output cannot be animated, because it's produced in one shot. The animation reveals the *order* in which the model "thinks" the drawing should come together — itself an interpretability signal.

2. **Lower bandwidth, forced semantic compression.** A 200-stroke drawing is ~600 stroke tokens. A 224×224 raster is ~256 image tokens but each token is from a 8K-entry codebook (more bits per token, ~2KB total). Strokes force the model to commit to *structural* depictions (outlines, key features) rather than texture / shading detail, which is what we want for "what is the model thinking" — concept, not photograph.

3. **Discretisation is cheap and well-supported.** Cartesian stroke-5 with 128 position bins gives 261 vocab additions. Trivially compatible with the Anole-style vocabulary extension pattern.

4. **Architecture stays vanilla.** No diffusion, no MDN, no flow matching. Pure next-token cross-entropy at training time, pure argmax/sample at inference time. The entire architectural cleverness is "add new vocabulary rows".

## Why this matters

If it works:
- **A new interpretability tool.** Visual lens complements text-based logit lens, tuned lens, and NLAs. Different failure modes, different strengths. Together they cover more of the residual stream's information than any one alone.
- **A new modality of model communication.** Models that can sketch when text isn't enough. Same way humans reach for a whiteboard. Beyond interpretability, this enables future products (geometry assistants, diagrammatic reasoning, etc.).
- **A clean architectural template.** The recipe (vocab extension + NLA-style autoencoder + activation injection + vision pathway for reconstruction) is generic. Once it works for strokes it generalises to any structured visual modality (graphs, schemas, music notation, ...).

If it partly works (e.g., FVE ~0.3 instead of NLA's 0.6-0.8):
- The animated drawings are still qualitatively informative, even at lower numerical fidelity. The paper writes itself around the visualisation, not the metric.

If it doesn't work:
- We learn something concrete about modality alignment in multimodal LLMs (the Day-0 sanity check in `05-evaluation.md`).

## Project name rationale

"Residual stream visual decoder" is the most descriptive. Alternatives considered:
- "Visual NLA" (accurate, but assumes audience knows NLA)
- "Stroke lens" (catchy but loses the "visual decoding of residual stream" framing)
- "Visual logit lens" (misleading, we're not using the lm_head)

Going with the descriptive name in the folder; we can rename for any external artefact.
