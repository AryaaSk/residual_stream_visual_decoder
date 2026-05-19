# 06 — Prior work

Synthesis of the research that informed the architecture. Six clusters, ordered by how directly they shaped what we're building.

---

## 1. Anthropic Natural Language Autoencoders (NLAs) — the direct ancestor

**Paper:** `transformer-circuits.pub/2026/nla/index.html` · **GitHub:** `github.com/kitft/natural_language_autoencoders`

What they did: trained a copy of Claude to **verbalise its own activations**, with a second copy trained to **reconstruct activations from those verbalizations**. Round-trip MSE loss. The text explanation is the bottleneck the activation flows through.

### Architecture
- **Target model (frozen).** Source of activations `h_ℓ` at mid-late layer ℓ.
- **AV (Activation Verbalizer).** Same architecture as target. Takes a fixed prompt; `<ACT_TOKEN>`'s embedding is replaced by scaled activation. Generates text autoregressively. Trained with **GRPO**.
- **AR (Activation Reconstructor).** Target architecture truncated to first ℓ layers + `Linear(d, d)` head on final-token activation. Trained with **supervised MSE**.
- **Loss.** AR: `‖h - ĥ‖²`. AV: `−log ‖h - ĥ‖²` + KL penalty `β · KL(AV ‖ AV_SFT_init)`.

### Training
- AV warm-start: SFT on a text-summarisation proxy task. Reaches FVE 0.3-0.4.
- AV RL: GRPO. Reaches FVE 0.6-0.8.
- Compute: SFT on 2×H100. RL on 2×8=16 H100s. **Not 3 hours.**

### Key results
- Caught Claude internally suspecting safety evaluation even when it didn't verbalize the suspicion (16% on code-deletion test, 26% on SWE-bench, <1% on real user data).
- In adversarial auditing game: 12-15% success with NLAs vs <3% without.

### Released checkpoints
On HuggingFace under `kitft/nla-models`:
- Qwen2.5-7B-Instruct
- Gemma-3-12B-IT
- Gemma-3-27B-IT
- Llama-3.3-70B-Instruct

**No Gemma-2B or Gemma-4-E2B.** We're training one for the smaller scale.

### What we lift from NLA
- Whole architectural pattern (target / AV / AR with round-trip MSE)
- Activation injection mechanism (token-embedding replacement)
- Truncated-AR + `Linear(d, d)` design
- GRPO training recipe with KL-to-init penalty
- Warm-start with SFT on a proxy task

### What we change
- AV output channel: **text → stroke tokens** (via Anole-style vocab extension)
- AR input modality: **text → rendered PNG via vision encoder** (using Gemma 4's native vision pathway)
- Base model: **Claude → Gemma 4 E2B** (smaller, open, multimodal-input)

---

## 2. Pi Zero / Pi 0.5 (Physical Intelligence) — architectural alternative considered and rejected

**Paper:** `arxiv 2410.24164` · **Pi 0.5:** `arxiv 2504.16054` · **Open clone:** `github.com/allenzren/open-pi-zero`

What they did: gave a frozen PaliGemma 3B a separate **action expert** (~300M params) that takes the same per-layer joint attention as PaliGemma but produces continuous robot joint trajectories via **flow matching**. The two networks share attention layers but have separate QKV/FFN weights ("mixture of experts in attention").

### Why we considered it
The two-Gemma framing made it tempting. If you want a separate "thought channel", give it a separate expert network.

### Why we rejected it for this project
- **Designed for high-dim continuous output** (50 timesteps × 14 joints = 700 floats). Stroke output is 3-dim per timestep; flow matching is overkill.
- **Inference smoothness via parallel decoding** matters for 50Hz robot control. We have no real-time constraint.
- **Multi-modality of valid trajectories** is a real argument for diffusion-style heads, but Cursive Transformer empirically showed AR discrete strokes handle this fine for drawing.
- **Interleaving with text reasoning is hard** with a separate expert (needs explicit mode switching). With NLA-style vocabulary extension, interleaving is free.

The decisive observation: **Physical Intelligence themselves also ship Pi-Zero-FAST**, a discrete-token version. They prefer flow matching for high-dim continuous control. We are NOT doing high-dim continuous control.

### What we lift from Pi Zero
- The intellectual move of "frozen pretrained substrate + new small head for a new output modality"
- The `<ACT_TOKEN>` activation injection idea (precursor to NLA's same trick)

---

## 3. Stroke generation models — the modality

**Graves 2013** (`arxiv 1308.0850`): LSTM + Mixture Density Network over `(Δx, Δy)`, Bernoulli on `pen_up`. Soft window attention over input characters. Canonical conditional-handwriting recipe.

**SketchRNN** (Ha 2017, `arxiv 1704.03477`): seq2seq VAE on QuickDraw with **stroke-5 representation** `(Δx, Δy, p_down, p_up, p_end)`. This is the de facto encoding we use.

**Cursive Transformer** (Greydanus 2025, `arxiv 2504.00051`, `greydanus/cursivetransformer`): vanilla GPT with **polar-binned stroke tokens**. No MDN, no special attention, beats Graves. Two tokens per stroke.

**VQ-SGen** (`arxiv 2411.16446`): VQ-VAE over stroke primitives. Path forward if we need primitive-level chunking in v2.

### What we use
- **Cartesian stroke-5** as the representation (broader empirical support than polar; no wraparound; trivially extends to 3D).
- 128-bin quantisation per axis (between Cursive Transformer's bin count and full continuous precision).

### What we rejected
- Polar (Cursive Transformer's choice): wraparound discontinuity, wastes resolution on axis-aligned content.
- VQ codebook (VQ-SGen): adds a separately-trained tokenizer stage. Defer to v2 if needed.

---

## 4. Visual reasoning via generation — closest neighbours in interpretability

**Visual Sketchpad** (Hu 2024, `arxiv 2406.09403`, `visualsketchpad.github.io`): GPT-4o with a Python `matplotlib` drawing tool. Reported gains: +12.7% on math benchmarks, +8.6% on vision benchmarks. **Empirical proof point that drawing during reasoning helps.** Doesn't constrain architecture (it's a tool call, not native output).

**Visualization-of-Thought** (Wu 2024, `arxiv 2404.03622`): pure text LLM emits ASCII grids in CoT. Proves externalised spatial state helps even without pixels.

**MVoT** (Li 2025, `arxiv 2501.07542`, `chengzu-li/MVoT`): autoregressive image tokens interleaved with reasoning text. Closest published version of "native visual tokens during reasoning". Uses VQ image tokenizer over rasters, not strokes. Tested on maze / Frozen Lake. **Steal the eval setup for grid-spatial reasoning tasks.**

**MathCanvas** (`arxiv 2510.14958`, `mathcanvas.github.io`): two-stage training (15.2M caption→diagram + edit pairs, then 219K interleaved visual-textual reasoning traces). Pixel diagrams, not strokes. **Steal the dataset-construction philosophy** if we extend to Phase 3 reasoning data.

### What we lift
- Visual Sketchpad's empirical justification (drawing helps reasoning by 8-13%)
- MVoT's grid-spatial evaluation harness
- MathCanvas's data-construction recipe (for future Phase 3)

---

## 5. Native interleaved multimodal output — the architectural template

| Model | Modality | Pattern |
|---|---|---|
| **Chameleon** (Meta, `arxiv 2405.09818`) | text + VQ-tokenised image | Single transformer, unified vocab. Trained from scratch. Image-out disabled in public release. |
| **Anole** (`arxiv 2407.06135`, `GAIR-NLP/anole`) | text + image | Trains <40M params on Chameleon-7B in 30 min on 8×A100 to re-enable image output. **Killer derisker for vocab extension.** |
| **Transfusion** (Meta, `arxiv 2408.11039`) | text + continuous image latents | Two losses (CE for text, diffusion for image) in one transformer. Closed. |
| **Show-o** (`arxiv 2408.12528`) | text + discrete image tokens | MaskGIT-style discrete diffusion for image, AR for text. Open. |

### What we lift
- **Anole's recipe is our template for AV.** Add new modality tokens to existing LLM vocab, train embedding + lm_head rows + thin LoRA. <40M params trainable. 30 min on 8×A100 (which means <30 min on 2×H200).

### What we rejected
- Chameleon's from-scratch pretraining: needs pretraining-scale compute.
- Transfusion's continuous diffusion: overkill for 3D stroke output.
- Show-o's discrete diffusion: AR sampling is fine for our token count.

---

## 6. Interpretability foundations — logit lens family

**Logit lens** (nostalgebraist 2020, LessWrong): apply `final_LN + lm_head` to intermediate residuals. See predictions crystallise across depth. Works because each layer's residual is roughly aligned with the next; fails because the unembedding was only trained for the final layer.

**Tuned lens** (Belrose 2023, `arxiv 2303.08112`): train a per-layer affine `A_ℓ` so that `lm_head(LN(A_ℓ · h_ℓ))` matches `lm_head(LN(h_L))`. Cheap, gives much cleaner trajectories.

**Patchscopes** (Ghandeharioun 2024): patch a hidden state into another forward pass with a different prompt to "interpret" it.

**SAEs (Sparse Autoencoders)**: decompose residual streams into interpretable features. Anthropic, OpenAI, DeepMind all heavily using.

### What we lift
- The **lens / autoencoder framing**. Our project is in this family.
- The conceptual move: **the unembedding only works for the final layer's residual; intermediate layers need a learned adapter.** Same reason our AR has `Linear(d, d)`.

### What's complementary
- Logit/tuned lens give text trajectories per layer.
- Our visual lens gives image trajectories per layer.
- SAEs give discrete feature activations per layer.
- All three are different views of the same underlying residual stream. Use together for the strongest interpretability story.
