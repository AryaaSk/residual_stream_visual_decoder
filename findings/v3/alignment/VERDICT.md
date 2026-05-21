# Phase 0 — text/image alignment verdict

Run: 2026-05-20, Qwen 3.5-4B, 80 (caption, real canonical drawing) pairs.

## Setup

For each pair we wrapped both inputs in the Qwen-VL chat template so the read
position is structurally identical (both end with `<|im_end|>\n`):

```python
# text
messages = [{"role": "user", "content": [{"type": "text", "text": caption}]}]
# image
messages = [{"role": "user", "content": [{"type": "image"}]}]
```

Then read `hidden_states[L][0, -1, :]` (the last token of the wrapper) and
computed `cosine(h_text, h_image)` at every layer L = 0..32.

## Result

Cosine vs layer is NOT monotonic, NOT late-layer-peaking. It's a middle-layer
plateau followed by sharp divergence at depth:

| layer | mean cosine | std   | interpretation                                         |
|------:|------------:|------:|--------------------------------------------------------|
|     0 |      +1.000 | 0.000 | embedding-only: both inputs end with same `\n`         |
|     3 |      +0.628 | 0.005 | early layers: positional dominance                     |
|  6-19 |   ~0.50-0.58| ~0.03 | **middle plateau** — best alignment                    |
|    15 |      +0.567 | 0.021 | sweet spot — high mean, low std → **chosen L\***       |
|    25 |      +0.213 | 0.045 | late layers: text and image representations diverge    |
|    29 |      +0.212 | 0.064 | nearly orthogonal                                      |
|    32 |      +0.264 | 0.018 | final layer: bottoming out                             |

## What this means for v3

- **The naive "unified stream converges at depth" hypothesis is wrong** for
  Qwen 3.5-4B last-token activations. Text and image inputs *diverge* with depth.
- **There is still a usable signal in middle layers** (L8-L19, cosine 0.48-0.58).
  Real canonical drawings match their captions at cosine 0.567 ± 0.02 at L15.
- **Chosen target layer: L\* = 15.** This is where:
  1. The mean is high enough that REINFORCE has meaningful signal to push.
  2. The std is low (0.021) so the per-pair signal is stable.
  3. The position is deep enough that we're past pure positional artifacts
     (L0-L2 are dominated by the shared closing `\n` token).
- **The chosen reward is `cosine(h_text(caption), h_image_only(drawing)) @ L15`.**
  Real-canonical-drawing baseline cosine at L15 is ~0.567 — the v3 AV is
  successful if its drawings push the cosine in that direction.

## Caveats

- The chat-template closing token introduces a position-aligned scaffold. We can
  see at L0 that both forwards produce identical embeddings (cosine 1.000) just
  because they share the same final `\n`. The cosine at L=15 (0.567) is the
  signal *above* that positional artifact, but isn't purely "concept similarity".
- The drop at late layers may reflect specialisation of last-token activations
  toward output-prediction logits, which diverge by modality. This is itself an
  interpretable finding about Qwen 3.5-4B's late-stack structure.
- A more principled experiment would compare at the *content* position (last text
  token before `<|im_end|>` for text; last image patch before `<|vision_end|>`
  for image). That confounds with positional encoding differences but removes the
  shared-closing-token artifact. **Future work.**

## v3 training decision

Proceed to Phase 1 with L\* = 15. Baseline cosine for AV-generated drawings
(before training) will be measured at probe step 0; the gate at step 500 checks
whether REINFORCE moves the reward EMA off that baseline.
