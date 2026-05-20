# v2.0 — Port to Qwen 3.5-4B (unified vision+text stream) + frozen-AR REINFORCE

## Why v2.0 exists

v1.5 shipped a viral-quality release (cat with face, sun with rays, fish with tail) by leaning on two compromises we now want to remove:

1. **44-concept ceiling.** Stage 1.5 supervised SFT trained on 44 QuickDraw categories with top-5 canonical drawings per concept. Prompts outside that set decode to noise.
2. **Reward gameability.** The original "AR reads drawing, predicts h" reward (v1.0/v1.1) failed because the AR was being trained alongside the AV — they co-evolved to make abstract scribbles AR-decodable but human-unrecognisable. We patched around this with CLIP-ranked best-of-N at *inference* and canonical-drawing distillation at *training*, but the principled fix was always "use the target model itself as the frozen AR."

Gemma 4 made the principled fix awkward: its vision tower and text path produce activations in distributionally-different distributions at the same layer — same coordinate frame, but different upstream statistics. The cross-modal consistency reward we sketched in the v1.x conversation works, but has to engineer around that pathway mismatch.

**Qwen 3.5-4B fixes the architectural premise.** Released March 2026. The 4B variant is a *native unified-stream multimodal model* — text and image tokens go through the same embedding lookup, share the same residual stream, get processed by the same attention from layer 0 onward. No vision tower hanging off the side. That means **feeding an image back through Qwen produces an activation in the same distribution as the original text-prompt activation**, and the frozen-AR-via-target-model reward becomes trivially clean to implement.

v2.0 is the architectural port that lets that reward function actually work, then trains REINFORCE on a broader prompt set than the 44 QuickDraw concepts.

## What v2.0 keeps from v1.x

Almost everything. The codebase turned out to be remarkably model-agnostic — only a handful of `model_id` defaults reference Gemma, and the LoRA walker / vocab extension / activation injection / batched generation / CLIP ranker all use standard HuggingFace APIs that work on any causal LM.

| component | v1.x form | v2.0 change |
|---|---|---|
| StrokeDecoder (`code/verbalizer/stroke_decoder.py`) | Gemma 4 E2B base | Qwen 3.5-4B base (one default flip per call site) |
| ActProjector | `Linear(d, d)` init `α·I` | unchanged |
| AV-LoRA (`code/ar/lora_gemma4.py`) | walker filters on `Gemma4ClippableLinear` or `Linear` | for AV: already accepts plain `Linear` (works on Qwen unchanged); for AR (no longer used in v2.0): drop the `Gemma4ClippableLinear` filter |
| Vocab extension (262 stroke tokens) | `tokenizer.add_tokens` + `resize_token_embeddings` | unchanged (model-agnostic) |
| Stroke renderer (`code/render.py`) | PIL/Cairo, 4× upscale | unchanged |
| Stage 1 SFT (text → strokes) | text-conditioned CE | unchanged |
| Stage 1.5 act-SFT on canonical drawings | h-conditioned CE | unchanged |
| CLIP ranker for best-of-N inference | works on any AV ckpt | unchanged |
| Eval / gallery / hype-reel pipeline (`code/eval/*`) | model-agnostic | unchanged |
| H200 deploy + autoeval scripts (`bin/*`) | parameterised on `--model-id` | unchanged |

**What changes in v2.0:**
- Base model: Gemma 4 E2B → Qwen 3.5-4B.
- New: Stage 2 REINFORCE with Qwen-self-consistency reward (replacing the never-shipped v1.6 idea).
- Drawing data unrestricted at Stage 2: any prompt, not just QuickDraw concepts.

## The Stage 2 reward, precisely

```
for each training step:
  P ~ prompt_set
  h_text = Qwen(P).hidden_states[L][last_text_token]               # frozen target
  drawings = AV.sample(h_text, n=4)                                # group sample
  for d in drawings:
      image = render(d)
      h_img_then_text = Qwen(image + P).hidden_states[L][last_text_token]
      reward[d] = cosine(h_text, h_img_then_text)
      if n_strokes(d) < 15: reward[d] -= (15 - n_strokes(d)) * 0.05
      reward[d] += 0.1 * CLIP_sim(image, P)
  advantage[d] = (reward[d] - mean(reward)) / (std(reward) + 1e-6)
  loss = -Σ advantage[d].detach() * log_prob(d | h_text)
  loss += β * KL(av_current || av_init)        # anchor
  backprop on AV's projector + AV-LoRA + new-vocab rows
```

Qwen is frozen end-to-end. The only trainable parameters are the AV side, exactly as in v1.5. The reward is "feed the AV's drawing image back through the same model that produced the target activation, and check that prepending the image to the original text prompt doesn't shift what the model is thinking at layer L." If the drawing genuinely encodes the prompt's concept, the activation shouldn't move much. If it's noise, the activation shifts and reward drops.

Three guards against degenerate solutions:
- **Min-stroke penalty:** an empty drawing is almost-identity to the text-only forward (vision tokens for blank images carry minimal info), so reward stays high but we've learned nothing. Penalise drawings with <15 strokes.
- **CLIP regulariser (small weight, 0.1):** orthogonal "does the rendered image look like the concept name" signal. Catches images that fool Qwen-consistency but don't look like anything.
- **KL anchor to v2.0 Stage 1.5 init:** keeps the AV's stroke distribution close to "QuickDraw-style line drawing" rather than letting it drift into adversarial pixel-art that maximises the reward but isn't drawing-like.

## Layer selection

Phase 0 of v2.0 includes an activation-geometry probe on Qwen: feed 30 diverse prompts, compute pair-cosine of last-token activations per layer, pick the layer in the middle band with the lowest pair-cosine (= most discriminative). On Gemma that gave L12 (pair-cosine 0.532); v1.5 confirmed L12 produced visibly better drawings than L24 (0.870, too clustered).

**Working assumption** (overridden by probe): `L_primary = round(0.46 × num_hidden_layers)`. For a 36-layer Qwen that's L17; for 40-layer it's L18. The probe will refine this on actual data in <5 minutes.

**Secondary layer:** `L_late = round(0.92 × num_hidden_layers)`. Used for the cross-layer trajectory artefact and as a fallback if L_primary doesn't converge.

We train L_primary first (smoke-test goalpost), then L_late in parallel.

## What v2.0 SHIPS

If Phase 5 gate passes:
- Tag `v2.0`, GitHub release.
- `artefacts/v2_0/demo.mp4` (54-sec hype reel of v2.0 hero gallery).
- `artefacts/v2_0/gallery.png` (12-tile CLIP-ranked best-of-32 hero gallery).
- `artefacts/v2_0/before_after.png` (v1.5 vs v2.0 side-by-side on the same prompts).
- `artefacts/v2_0/ood_demo.png` (v2.0 on prompts v1.5 *couldn't* handle: Eiffel Tower, capital of France, thunderstorm, the concept of loneliness).
- README banner image updated; v1.5 demoted to "previous version".

If Phase 5 gate doesn't pass (Stage 2 REINFORCE blows up; SFT-only Qwen worse than v1.5):
- Don't tag v2.0.
- Ship a `RESEARCH_NOTES.md` chapter documenting the attempt and what we learned.
- v1.5 remains the public release.

## Budget

5 GPU-hours on 2× H200. Hard cap.

| phase | wallclock | purpose |
|---|---|---|
| 0 | 0.5h | Qwen sanity, arch probe, L_primary commit |
| 1 | 1.0h | code port (model_id defaults), 50-step smoke train |
| 2 | 1.5h | Stage 1 + Stage 1.5 SFT on Qwen, both layers parallel |
| 3 | 1.0h | implement + standalone-test the Qwen-self-consistency reward |
| 4 | 1.0h | Stage 2 REINFORCE, both layers parallel, ~3-5K steps |
| 5 | 1.0h | CLIP eval, hype reel, README/WRITEUP, GitHub release |

Phases 4 and 5 can overlap. Phases 0-2 don't need Phase 3's code; they're standalone.

## Files

**New:**
- `code/eval/qwen_sanity_check.py` — Phase 0 probe.
- `code/train/stage2_qwen_consistency.py` — the REINFORCE trainer.
- `data/v2_prompts.jsonl` — ~3K diverse prompts for Stage 2.
- `V2_PLAN.md` — this document.

**Edited:** `code/verbalizer/stroke_decoder.py`, `code/verbalizer/vocab_extend.py`, `code/ar/lora_gemma4.py`, `code/eval/clip_ranker.py`, `code/train/stage1_v2_act_sft.py` — `model_id` default flips. RESEARCH_NOTES.md + WRITEUP.md + README.md — v2.0 chapter.

**Reused as-is:** `code/eval/build_gallery.py`, `before_after_grid.py`, `make_hype_reel.py`, `best_across_ckpts.py`, `version_evolution.py`. `code/data/pick_canonical_drawings.py`, `build_expanded_corpus.py`. `code/render.py`, `code/stroke_tokenizer.py`, `code/verbalizer/projector.py`. `bin/h200`, `bin/autoeval_v1_5_clip.sh` (cloned to v2_0 variant).
