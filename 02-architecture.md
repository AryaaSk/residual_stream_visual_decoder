# 02 — Architecture

## High-level data flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   INPUT TEXT                                                             │
│   "The capital of France is"                                             │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────────┐                                                    │
│   │   TARGET        │  ← Gemma 4 E2B, FROZEN, never trained              │
│   │   (frozen)      │    extract one or more layer activations as it     │
│   │                 │    processes the input text                        │
│   └─────────────────┘                                                    │
│         │                                                                │
│         │ h_ℓ   (activation at layer ℓ, ℝᵈ where d ≈ 2048-4096)          │
│         ▼                                                                │
│   ┌─────────────────┐                                                    │
│   │   AV            │  ← Gemma 4 E2B, TRAINED, vocab extended with       │
│   │   (verbalizer / │    262 stroke tokens. Receives the activation in   │
│   │   stroke        │    place of a special prompt-token embedding.      │
│   │   decoder)      │    Autoregressively outputs stroke tokens.         │
│   └─────────────────┘                                                    │
│         │                                                                │
│         │ stroke tokens  [Δx_42, Δy_88, pen_down, ...]                   │
│         ▼                                                                │
│   ┌─────────────────┐                                                    │
│   │   RENDERER      │  ← pure Python (PIL/Cairo), no learned params      │
│   │ (deterministic) │    walks pen across 224×224 canvas, saves frames   │
│   └─────────────────┘                                                    │
│         │                                                                │
│         │ (a) final PNG (224×224) → fed to AR                            │
│         │ (b) animated MP4 / GIF  → for human inspection only            │
│         ▼                                                                │
│   ┌─────────────────┐                                                    │
│   │   AR            │  ← Gemma 4 E2B TRUNCATED to first ℓ layers,        │
│   │ (reconstructor) │    vision encoder kept intact, + Linear(d,d)       │
│   │                 │    Reads rendered PNG as image input.              │
│   └─────────────────┘                                                    │
│         │                                                                │
│         │ ĥ_ℓ   (reconstructed activation, ℝᵈ)                            │
│         ▼                                                                │
│   LOSS: ‖h_ℓ − ĥ_ℓ‖²    (or 2(1 − cos(h_ℓ, ĥ_ℓ)))                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

The loss being low means the drawing produced by AV contains enough information about the activation that a copy of Gemma 4 (the AR) can read the drawing as an image and recover what the original Gemma 4 was thinking. That is the visual lens.

## Choice of base model: Gemma 4 E2B

| Property | Why |
|---|---|
| **E2B size** | Small enough for fast iteration on 2× H200, big enough for rich mid-stack reasoning content |
| **Vision input native** | AR reads rendered drawing as an image through Gemma 4's existing vision encoder. No new modality on the AR side. |
| **Open weights** | Required for truncation surgery |
| **Same tokenizer family as Gemma 2/3** | Vocab extension recipe transfers |
| **"Thinking" variants exist (26B, 31B)** | When scaling up later, reasoning-rich activations are the most interesting to visualise |

The vision-input capability is the architectural unlock. We only need to teach **one** side (the AV / stroke decoder) a new output modality. The AR uses what Gemma 4 already knows for image processing.

---

## TARGET — the frozen model

Stock Gemma 4 E2B, no modifications. Two roles:

```
┌──────────────────────────────────────────────────────────────┐
│                  TARGET: Gemma 4 E2B (frozen)                │
│                                                              │
│   tokens ─→ embed ─→ ┌─────┐ ─→ ┌─────┐ ─→ ... ─→ ┌─────┐    │
│                      │ blk │    │ blk │            │ blk │    │
│                      │  1  │    │  2  │            │  L  │    │
│                      └──┬──┘    └──┬──┘            └──┬──┘    │
│                         │          │                  │       │
│                       residual stream (h_1, h_2, ..., h_L)    │
│                         │          │                  │       │
│                         ▼          ▼                  ▼       │
│                       extract   extract            extract    │
│                       (tap into residual at any layer ℓ)      │
└──────────────────────────────────────────────────────────────┘
```

We extract `h_ℓ ∈ ℝᵈ` at a chosen layer ℓ at a chosen token position (final token by default). `d` is Gemma 4 E2B's hidden size.

### All-layer sweep

For the trajectory visualisation, we want to see thoughts evolve across depth. **The AR is per-layer no matter what** (truncating to layer 8 vs layer 24 is literally a different network), so the only design choice is how much we share on the AV side. Three options:

#### Option A — Full per-layer (gold standard)

N independent `(AVℓ, ARℓ)` pairs, one per anchor layer. Each AV trains from scratch only on its layer's activation distribution.

- **Pros:** Each pair fits its layer optimally. Lowest quality risk. Matches Anthropic's released-checkpoint pattern (their Qwen-7B / Gemma-3-12B / etc. NLAs each cover one layer).
- **Cons:** N× training compute and N× artefact-management overhead.
- **For E2B + 6 anchor layers `{4, 8, 12, 16, 20, 24}`:** 6 full training runs.

#### Option B — Hybrid (shared AV body + per-layer input adapter + per-layer AR)

```
   h_ℓ                                                 
    │                                                  
    ▼                                                  
  Aℓ : ℝᵈ → ℝᵈ          ← per-layer Linear(d,d) adapter (~16M params each)
    │                       trained to rotate each layer's distribution
    │                       into the AV's canonical input space
    ▼                                                  
  α · Aℓ(h_ℓ)            ← injected into <ACT_TOKEN>'s embedding
    │                                                  
    ▼                                                  
  [shared AV body]       ← one set of vocab, embeddings, LoRA, lm_head
    │                       trained on activations from ALL layers via the adapters
    ▼                                                  
  stroke tokens                                        
```

- **Pros:** ~1.5-2× cost vs single-layer (one big AV + N small adapters + N ARs). AV body learns "drawing skills" from data across all layers. Adapters handle layer-specific statistics.
- **Cons:** Possible negative transfer if layer distributions are very different. Adapter capacity may be insufficient.

#### Option C — Layer-conditioned single AV (rejected)

One AV that takes layer id as a prompt token. No per-layer adapter, no per-layer body.

- **Pros:** Cheapest. ~1× cost.
- **Cons:** AV has to internally handle activations from all layers simultaneously. High risk of mediocre quality everywhere. Not used.

### Decision

**Gold standard: Option A.** Run full per-layer training across 6 anchor layers. With access to a few days on 2-4 H200s this is achievable.

**Pragmatic fallback: Option B.** If compute is constrained to ~1× single-layer cost, the hybrid gives most of the quality with controlled cost. Anchor layer adapters trained jointly with the shared AV body.

**Rejected: Option C.**

---

## AV — Activation Verbalizer / stroke decoder

The activation enters Gemma 4 via **token-embedding replacement**. This is the core NLA trick.

### Vocabulary extension (Anole pattern)

```
ORIGINAL Gemma 4 vocab (~256K tokens):
    [BOS, EOS, "the", "cat", "▁sat", ...]

ADDED stroke vocab (262 new tokens):
    [<DRAW>, </DRAW>,
     Δx_0,    Δx_1,    ..., Δx_127,        ← 128 horizontal-offset bins
     Δy_0,    Δy_1,    ..., Δy_127,        ← 128 vertical-offset bins
     pen_down, pen_up, pen_end,            ← 3 pen state tokens
     <ACT_TOKEN>]                          ← activation injection slot
```

Embedding matrix grows from `(V, d)` to `(V+262, d)`. Same for `lm_head`. **Only the new rows are trainable initially.** Backbone stays frozen except for a thin LoRA we add to attention layers.

### Activation injection mechanism

```
PROMPT: [BOS, "Visualize", "the", "following", "thought", ":",
         <LAYER_ℓ_token>, <ACT_TOKEN>, <DRAW>]

EMBEDDING STEP:
   normal flow:    token id → embedding matrix lookup → vector
   
   <ACT_TOKEN>:    instead of using the learned embedding row,
                   OVERWRITE it with α · h_ℓ
                   where α is a scalar (~1.0 to start, hand-tuned)

In code:
   input_embeds = model.embed_tokens(input_ids)
   act_pos = (input_ids == ACT_TOKEN_ID).nonzero()[0]
   input_embeds[act_pos] = alpha * h_l           # ← the surgery
   outputs = model(inputs_embeds=input_embeds, ...)
```

Three lines. That is the entire "activation injection" mechanism.

### Autoregressive stroke decoding

```
step 1:  context = "Visualize ... <ACT_TOKEN> <DRAW>"
         AV emits:  Δx_64        (centre horizontally)

step 2:  context = "... <DRAW> Δx_64"
         AV emits:  Δy_64        (centre vertically)

step 3:  context = "... <DRAW> Δx_64 Δy_64"
         AV emits:  pen_down     (pen touches canvas)

step 4:  context = "... pen_down"
         AV emits:  Δx_70        (move right by ~6 units)

... continues until ...

step N:  AV emits:  </DRAW>      (stop)
```

A 200-stroke drawing is ~600 stroke tokens plus markers ≈ 605 tokens. Comfortable within Gemma 4's context.

### Trainable surface

```
Component                    Frozen / Trained
─────────────────────────────────────────────
Embed rows for old tokens    FROZEN
Embed rows for new 262       TRAINED  (~262 × d ≈ 0.5M params)
Backbone weights             FROZEN  (LoRA on top)
LoRA (rank 16, all attn)     TRAINED  (~5-10M params)
lm_head rows for old tokens  FROZEN
lm_head rows for new 262     TRAINED  (~262 × d ≈ 0.5M params)
                            ─────────────────
                            Total trainable: ~10M params
```

Anole-style PEFT. Huge model, tiny trainable surface, fast convergence.

---

## AR — Activation Reconstructor

Reads the rendered drawing as an image, outputs a reconstructed activation in the same coordinate frame as `h_ℓ`.

### Truncation

```
FULL Gemma 4 E2B:                          AR (= truncated):
                                                                              
   image patches ─┐                          image patches ─┐               
   text tokens ───┼─→ blk 1 → ... → blk L     text tokens ─┼─→ blk 1 → ... → blk ℓ
                  │                                          │
                                                         TAKE final-token
                                                         activation at layer ℓ
                                                              │
                                                              ▼
                                                         Linear(d, d)
                                                              │
                                                              ▼
                                                             ĥ_ℓ
```

Why truncate? Because `h_ℓ` was the activation at layer ℓ in the original Gemma. We want `ĥ_ℓ` in the same coordinate frame (same layer of the same architecture).

**Important capability note:** the AR's backbone is **full pretrained Gemma 4 E2B**, including its vision encoder. This means **the AR can read text in images** (Gemma 4 was pretrained on internet-scale data including OCR-rich material). If the AV decides during Stage 3 RL that "writing the concept as letters" is a viable encoding (e.g., draw `cat` as block letters when h_ℓ = activation for "cat"), the AR will RECOGNISE the letters and produce a `ĥ_ℓ` close to the target. Reward flows. This is a real possibility, not a failure mode — see the "letter-writing convergence" item in `05-evaluation.md`. The KL penalty in Stage 3 is what arbitrates between "concept sketch" and "letter writing" as the model's chosen encoding.

### Forward pass

```
rendered PNG (224 × 224)
        │
        ▼
┌──────────────────────────┐
│ Gemma 4 vision encoder   │   ← Already in Gemma 4 E2B. Used as-is.
│ (ViT-style patch encoder)│
└──────────────────────────┘
        │
        │ image patch embeddings (sequence of vectors)
        ▼
PROMPT WRAPPER: prepend text tokens
   [token_embeds("This drawing depicts the thought:"),
    image_patch_embeds,
    token_embeds(".")]
        │
        ▼
┌──────────────────────────┐
│ Gemma 4 LLM, first ℓ     │   ← truncated copy of Gemma 4
│ layers only              │     same weights initially, fine-tuned with LoRA
└──────────────────────────┘
        │
        │ activations at layer ℓ
        ▼
SELECT activation at final token position (or pool image-patch positions)
        │
        ▼
┌──────────────────────────┐
│ Linear(d, d)             │   ← the only entirely new parameter block
│                          │     calibrates AR's "image-conditioned" activation
└──────────────────────────┘     space to target's "text-conditioned" space at ℓ
        │
        ▼
ĥ_ℓ
```

### Trainable surface

```
Component                          Frozen / Trained
─────────────────────────────────────────────────────
Vision encoder                     FROZEN
Embed rows                         FROZEN
Backbone first ℓ layers            FROZEN  (LoRA on top)
LoRA (rank 8, all attn in 1..ℓ)    TRAINED  (~3-5M params)
Linear(d, d) head                  TRAINED  (~4-16M params depending on d)
                                  ─────────────────
                                  Total trainable: ~10-20M params
```

---

## Stroke representation (formal)

```
A "drawing" = list of strokes, where each stroke is (Δx, Δy, pen_state).

  Δx, Δy:   integer offsets from current pen position, in pixels
            Range -64 to +63, quantised to 128 bins.
  pen_state ∈ { DOWN, UP, END }
            DOWN: pen touching canvas, drawing as it moves
            UP:   pen lifts at end of current segment (next move is a jump)
            END:  end of the whole drawing

Token stream per stroke = 3 tokens: [Δx_bin, Δy_bin, pen_state_token]

Total vocab additions:
  128  Δx_bin tokens
  128  Δy_bin tokens
    3  pen_state tokens (DOWN, UP, END)
    2  bracket tokens (<DRAW>, </DRAW>)
    1  injection token (<ACT_TOKEN>)
  ───
  262 new tokens
```

This is the entire modality bridge. No diffusion, no MDN, no flow matching. Just 262 new vocabulary entries.

See `04-renderer.md` for the renderer that turns these tokens into a PNG + animation.

---

## The load-bearing assumption

The architecture only works if **Gemma 4's residual stream at layer ℓ represents semantically-equivalent text and images in overlapping subspaces**.

If text-"France" and image-of-France produce activations in disjoint subspaces at layer ℓ, the AR can never reconstruct one from the other.

**Reasons to expect alignment:**
- Gemma 4 is natively multimodal. Vision patches project into the LLM's embedding space and flow through the same transformer stack as text. By design they share the residual stream.
- Mechanistic interpretability on multimodal LLMs has found shared concept features (e.g., "Empire State Building" activates similarly for image vs text).
- The `Linear(d, d)` head + AR's LoRA are specifically there to bridge any residual modality gap.

**Reasons to worry:**
- Modality-specific features persist even in unified models, especially in early/mid layers where modalities are still being "translated".
- Sparse line art is rare in Gemma 4's vision pretraining; our renderings may land off-manifold.
- Even if alignment exists in principle, our adapter may be too small to bridge.

**Mitigations:**
- Pick ℓ in the late stack (where multimodal models tend to converge to shared concept representations).
- Optional: brief vision-encoder fine-tune on QuickDraw line art to bring our renderings on-manifold.
- Bigger adapter if `Linear(d, d)` is insufficient: 2-3 layer MLP, or higher LoRA rank.

**Mandatory Day-0 sanity check** before any training: see `05-evaluation.md`.
