# 05 — Evaluation

Three layers of eval: a Day-0 viability check before any training, quantitative reconstruction metrics during/after training, and qualitative trajectory visualisation as the headline artefact.

## Day-0 sanity check (mandatory, ~1 GPU-hour)

**Question: does Gemma 4 represent semantically-equivalent text and images in overlapping subspaces at any layer?**

If no, the entire architecture is built on sand. We must measure this before committing to multi-stage training.

### Protocol

```
1. Take a list of 100 simple concepts (cat, France, triangle, fear, ...).
2. For each concept C:
   a. text_act[C, ℓ]  = Gemma 4 forward("I am thinking about " + C),
                        extract residual at layer ℓ at final token
   b. image_act[C, ℓ] = Gemma 4 forward(rendered QuickDraw drawing of C as image input),
                        extract residual at layer ℓ at last image-patch token
3. For each layer ℓ in {1, 4, 8, 12, 16, 20, 24, 26}:
   alignment_ℓ     = mean over concepts of cos(text_act[C, ℓ], image_act[C, ℓ])
   control_ℓ       = mean over random concept pairs of cos(text_act[C1, ℓ], image_act[C2, ℓ])
4. Plot alignment_ℓ and control_ℓ vs ℓ.
```

### Expected outcomes

| Outcome | Interpretation | Action |
|---|---|---|
| `alignment_ℓ ≫ control_ℓ` at any layer | Cross-modal alignment exists. Project viable. | Proceed; pick the best ℓ as default. |
| `alignment_ℓ ≈ control_ℓ` everywhere | No alignment. Assumption broken. | Re-think. Options: pretrain a vision adapter on line art, use a different target model, use a per-modality calibration head. |
| `alignment_ℓ` rises with depth, beats control only in late layers | Expected typical result. | Focus the experiment on late layers (ℓ ≈ 16-26 for E2B). |

This is also the cleanest "Day 1" deliverable. Even if nothing else lands, we have a measured answer to "do multimodal LLMs converge text and image concepts at deep layers?" which is itself a small interpretability result.

## Quantitative: Fraction of Variance Explained (FVE)

Standard NLA metric. For held-out `(h_ℓ, ĥ_ℓ)` pairs:

```
FVE_ℓ = 1 - Var(h_ℓ - ĥ_ℓ) / Var(h_ℓ)
```

NLA paper reports 0.6-0.8 on Claude-scale models with text channel. Expectations for us:

- **Visual NLA on Gemma 4 E2B, late layer:** 0.3-0.5 (lower than text due to modality gap; expected)
- **Text NLA on Gemma 4 E2B (our baseline), late layer:** 0.4-0.6 (lower than Claude due to smaller model)
- **Random baseline:** ~0 (random images shouldn't predict random activations)

### Reporting structure

```
                FVE @ ℓ=8   FVE @ ℓ=16   FVE @ ℓ=24
Text NLA       0.30          0.50          0.55
Visual NLA     0.15          0.35          0.45
Random         0.01          0.01          0.01
```

Compare visual vs text per layer. Compare both against random as sanity.

## Qualitative: Layer-trajectory visualisation (the headline)

This is the artefact that justifies the project's existence.

### Protocol

For each of 50 carefully-chosen probes (factual recall, arithmetic, multi-hop reasoning, emotional content, ambiguous phrasing):

1. Run target Gemma 4 forward.
2. Extract `h_ℓ` at each layer in sweep set.
3. For each layer, run AV with activation injection, render the drawing, save final PNG + MP4.
4. Lay out all layer-PNGs as a 1×N strip per prompt.
5. (Bonus) make a "morph" video that animates between layers, so you see a single moving drawing as ℓ increases.

### Probe set (50 prompts, by category)

| Category | Example prompts |
|---|---|
| **Factual recall** | "The capital of France is", "The author of Hamlet is" |
| **Arithmetic** | "What is 47 + 38?", "What is 12 × 13?" |
| **Multi-hop** | "The mother of Barack Obama's wife is named", "The capital of the country containing Mount Everest is" |
| **Emotional** | "She received the news and felt", "The funeral was somber and" |
| **Ambiguous** | "I saw the man with the telescope" |
| **Lists / structured** | "Three primary colours are" |
| **Negation** | "Paris is not the capital of" |
| **Code** | "def fibonacci(n):", "SELECT * FROM" |

### What we'll look for

- **Trajectory continuity.** Do drawings morph smoothly across layers (suggesting continuous refinement) or jump discretely (suggesting modular reasoning)?
- **Concept sharpening.** Does the drawing become more specific / detailed at later layers?
- **Multi-hop intermediates.** Does the "Obama's wife's mother" prompt show *Michelle Obama* concepts at intermediate layers before *Marian Robinson* concepts at later layers?
- **Failure cases.** Does the model "see" ambiguous prompts as multiple overlapping concepts (a man holding a telescope AND a man being viewed through one)?

### Comparison: text NLA on the same probes

For every probe and layer, generate both:
- Text NLA explanation: "the model is thinking about a European capital city"
- Visual NLA drawing: an outline that may or may not look like France

Lay them out side by side. The visual NLA's value-add is most visible when:
- The concept is **spatial** (drawings show layout, text struggles)
- The thought is **partially formed** (a vague drawing reads as vague; a vague text reads as wrong)
- The activation is **mixed** (drawing can show overlaid concepts; text picks one)

## Failure modes (and one "failure" that's secretly interesting)

| Symptom | Likely cause | Fix / framing |
|---|---|---|
| All drawings look the same | AV ignoring activation injection | Increase α, increase RL signal, verify `<ACT_TOKEN>` actually overridden |
| Reasonable drawings but FVE near 0 | Drawings depict the concept but AR's `Linear(d,d)` can't extract that into target's coordinate frame | Bigger AR adapter (2-3 layer MLP), brief vision-encoder fine-tune on QuickDraw style |
| FVE good but drawings are unrecognisable scribbles | RL found an "incomprehensible code" — AV invented an arbitrary visual encoding that AR (frozen Gemma 4 with vision) happens to decode | Increase KL penalty β toward SFT init |
| **AV writes the word for the concept as block letters** | **Not a failure** — Gemma 4 can OCR, so writing `cat` is a valid (and arguably more interpretable) encoding than sketching one. The reward signal will reinforce this if KL doesn't prevent it. | If you want concept sketches: increase β. If you want letter-writing as the result: decrease β and rerun. **Either output is publishable.** |
| Drawings only sensible at one layer | Layer-conditioned model not generalising | Switch to full per-layer pairs for tricky layers |
| All drawings look like cats / one category | SFT corpus too imbalanced | Re-weight QuickDraw categories uniformly during corpus build |

**Important context for the "letter writing" row:** the AR's backbone is full pretrained Gemma 4 E2B (with its vision encoder). Gemma 4 was pretrained on internet-scale image-text data including OCR-rich material, so it can read text in images. If AV writes `cat` in strokes, the AR's vision encoder will represent the image similarly to how it represents the word "cat" from text, and reconstruction succeeds. The β coefficient in the KL penalty is the knob that arbitrates between "sketch the concept" (β large → AV pulled toward QuickDraw prior) and "write the concept" (β small → AV free to discover letter-writing as the highest-bandwidth encoding).

## Stop criteria

- **Day-0 fails:** abort, pivot to text-only NLA experiment or rethink alignment assumption.
- **Stage 1 produces incoherent strokes:** debug data pipeline before going further.
- **Stage 3 FVE stays near baseline after 24 GPU-hours:** the RL signal isn't reaching the AV; debug reward, KL, group size.
- **Everything else:** ship v0, publish, iterate.
