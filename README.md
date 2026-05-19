# Residual Stream Visual Decoder

A "visual lens" for LLM activations. Trains a stroke-output decoder that converts mid-stack residual stream vectors from Gemma 4 E2B into hand-drawn-style images, jointly trained with an image-input reconstructor (using Gemma 4's native vision pathway) so the drawings must be informative enough to recover the original activation.

Inspired directly by Anthropic's **Natural Language Autoencoders (NLAs, 2026)**, which decode activations to text. We do the same thing but the output is **drawings**, not text, giving us a visual interface to the model's internal thoughts.

## Why it's interesting

The unoccupied square in the literature: small (≤4B) open LLM × vector stroke channel × interleaved with text reasoning × for interpretability. No published work fills this slot.

The drawings carry information that text NLA explanations cannot:
- **Spatial structure** that text struggles to convey (concept maps, geometric relationships, "fuzziness" of a half-formed thought)
- **Animation** showing how the drawing unfolds, providing a "stroke order" that hints at the order in which features come together
- **Continuous morph across layers** when you sweep ℓ: a more legible "thought trajectory" than the abrupt token changes you get from text logit-lens

## TL;DR architecture

```
target_activation (Gemma 4)
   │
   ▼
 [AV: stroke decoder] ─→ stroke tokens
                              │
                              ▼
                       [renderer]  ─→ animated MP4 + final PNG
                              │
                              ▼
                       [AR: truncated Gemma 4 reading PNG via vision pathway]
                              │
                              ▼
                       reconstructed activation
                              │
                              ▼
              MSE loss against original activation
```

Three Gemma 4 E2B instances:
- **TARGET** (frozen): source of activations.
- **AV** (verbalizer / stroke decoder): vocab extended with 262 stroke tokens, activation injected as a special token embedding, trained with GRPO.
- **AR** (reconstructor): truncated to first ℓ layers, reads rendered PNG through Gemma 4's existing vision encoder, output through `Linear(d, d)`, trained with supervised MSE.

Loss is round-trip activation MSE. Same recipe as NLA, swap text-channel for stroke-channel.

## Files in this folder

| File | Contents |
|---|---|
| `01-vision.md` | What we're building, why it's novel, why strokes specifically |
| `02-architecture.md` | Full technical architecture, every component spec'd, diagrams |
| `03-training.md` | The 4-stage training pipeline + compute estimates |
| `04-renderer.md` | The deterministic renderer + animation pipeline |
| `05-evaluation.md` | Day-0 sanity check, FVE metric, qualitative eval, baselines |
| `06-prior-work.md` | Anthropic NLA, Pi Zero, stroke models, multimodal LLMs |

## Status

Architecture spec complete. Awaiting compute allocation for Day-0 modality-alignment sanity check (1 GPU-hour, must pass before any training).

## Hardware

2× H200 confirmed for initial work. More TBD. See `03-training.md` for budget tiers (3-hr / 24-hr / week).
