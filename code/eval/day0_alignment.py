"""Day-0 sanity check: cross-modal alignment per layer in Gemma 4 E2B.

The go/no-go gate for the whole project. We ask: does Gemma 4 represent
semantically-equivalent text and images in overlapping subspaces at any layer?

If yes → there is structure for our AR to recover.
If no  → the architectural assumption is broken.

Protocol
--------
For each of 100 concepts (from a curated concept list):
    text_act[C, ℓ]  = Gemma 4 forward("I am thinking about " + C),
                      residual stream at layer ℓ, final-text-token position
    image_act[C, ℓ] = Gemma 4 forward(rendered QuickDraw-style sketch of C
                                       as image input + a neutral text prompt),
                      residual stream at layer ℓ, last-image-patch position

Then compute:
    alignment_ℓ = mean_C cos(text_act[C, ℓ], image_act[C, ℓ])
    control_ℓ   = mean_{C1≠C2} cos(text_act[C1, ℓ], image_act[C2, ℓ])

Outputs
-------
    findings/day0_alignment.json   per-layer alignment & control numbers
    findings/day0_alignment.png    line plot of both vs layer
    findings/day0_concepts.json    the concept list used (for reproducibility)

The "sketches" used for image input are generated procedurally with PIL
(simple icon-like shapes per concept) so this script has no external data
dependency.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

MODEL_ID_DEFAULT = "google/gemma-4-e2b-it"

# 50 concepts that exist in QuickDraw's categories (so we can use REAL human
# drawings for the image side of day-0). Procedural fallback works for anything,
# but produces alignment near zero because random scribbles aren't recognisable.
CONCEPTS = [
    "cat", "dog", "horse", "elephant", "fish", "bird", "snake", "spider",
    "apple", "banana", "carrot", "pizza", "donut", "cookie", "bread",
    "tree", "flower", "leaf", "mushroom", "cactus",
    "mountain", "cloud", "sun", "moon", "star", "rainbow",
    "house", "tent", "bridge", "tower",
    "car", "bicycle", "airplane", "boat", "train", "truck", "rocket",
    "chair", "table", "bed", "lamp", "clock", "door", "window", "key",
    "book", "pencil", "scissors", "phone", "umbrella",
]
assert len(CONCEPTS) == 50, f"Expected 50 concepts, got {len(CONCEPTS)}"


def render_concept_sketch_procedural(concept: str, size: int = 224) -> Image.Image:
    """Fallback procedural sketch (NOT actual concept depiction).

    Generates random strokes seeded by concept name. Used only when no
    QuickDraw data is available. Empirically: with these sketches the cross-modal
    alignment delta is near zero, because the model can't recognise concepts
    in random scribbles. Use ``render_concept_quickdraw`` instead whenever
    possible.
    """
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    seed = sum(ord(c) for c in concept)
    rng = np.random.default_rng(seed)
    n_strokes = rng.integers(3, 8)
    for _ in range(n_strokes):
        x0 = rng.integers(20, size - 20)
        y0 = rng.integers(20, size - 20)
        x1 = x0 + rng.integers(-80, 80)
        y1 = y0 + rng.integers(-80, 80)
        x1 = int(np.clip(x1, 0, size - 1))
        y1 = int(np.clip(y1, 0, size - 1))
        draw.line([(x0, y0), (x1, y1)], fill=0, width=2)
    return img


def render_concept_quickdraw(concept: str, quickdraw_dir: Path, size: int = 224, seed: int = 0) -> Image.Image | None:
    """Render an actual QuickDraw sketch of `concept` as a 224×224 grayscale PNG.

    Looks up `<quickdraw_dir>/<concept>.ndjson`, picks a deterministic sample
    (recognized=True if possible), and converts to our renderer's format using
    the QuickDraw → stroke-5 → render pipeline.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.quickdraw_loader import quickdraw_strokes_to_stroke5, stream_quickdraw_ndjson
    from render import render as stroke_render

    ndjson_path = quickdraw_dir / f"{concept}.ndjson"
    if not ndjson_path.exists():
        return None
    samples = list(stream_quickdraw_ndjson(ndjson_path, max_per_file=100))
    if not samples:
        return None
    obj = samples[seed % len(samples)]
    strokes = quickdraw_strokes_to_stroke5(obj["drawing"], target_size=size)
    img = stroke_render(strokes, canvas_size=size)
    return img


def render_concept_sketch(concept: str, size: int = 224, quickdraw_dir: Path | None = None) -> Image.Image:
    """Render an image of `concept`. Uses QuickDraw if available, otherwise procedural fallback."""
    if quickdraw_dir is not None:
        img = render_concept_quickdraw(concept, quickdraw_dir, size=size)
        if img is not None:
            return img
    return render_concept_sketch_procedural(concept, size=size)


def load_model_and_processor(model_id: str, device: str = "cuda"):
    """Load Gemma 4 E2B with the multimodal processor."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"[day0] loading {model_id} on {device}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"[day0] loaded. num layers = {model.config.text_config.num_hidden_layers}", flush=True)
    return model, processor


@torch.no_grad()
def extract_text_residuals(model, processor, text: str, device: str = "cuda") -> torch.Tensor:
    """Run text-only forward, return residuals at each layer (final-token position).

    Returns a tensor of shape (n_layers + 1, hidden_size) in float32 on CPU.
    """
    tok = processor.tokenizer
    inputs = tok(text, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    # hidden_states is a tuple of (embeddings, layer1, ..., layerN)
    # we want one tensor per layer, at the LAST token position
    residuals = []
    for h in out.hidden_states:
        residuals.append(h[0, -1, :].to(torch.float32).cpu())
    return torch.stack(residuals, dim=0)


@torch.no_grad()
def extract_image_residuals(model, processor, image: Image.Image, prompt: str = "What is in this image?", device: str = "cuda") -> torch.Tensor:
    """Run text+image forward, return residuals at each layer at the LAST token position.

    The "last token" after multimodal preprocessing is post-text, so it reflects
    the integrated representation of image + prompt.
    """
    # Use the chat template with an image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    residuals = []
    for h in out.hidden_states:
        residuals.append(h[0, -1, :].to(torch.float32).cpu())
    return torch.stack(residuals, dim=0)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    parser.add_argument("--output-dir", default="findings")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Use only first N concepts (for quick smoke test)")
    parser.add_argument("--quickdraw-dir", type=Path, default=Path("data/quickdraw"),
                        help="Directory containing per-concept QuickDraw NDJSON files. If files exist, use real sketches; else fall back to procedural.")
    parser.add_argument("--n-image-samples", type=int, default=3,
                        help="Average over this many different QuickDraw samples per concept (averaged image activation).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concepts = CONCEPTS if args.limit is None else CONCEPTS[: args.limit]
    print(f"[day0] using {len(concepts)} concepts", flush=True)

    model, processor = load_model_and_processor(args.model_id, args.device)
    n_layers = model.config.text_config.num_hidden_layers
    print(f"[day0] {n_layers} text layers + 1 embedding layer = {n_layers + 1} extractable positions", flush=True)

    text_acts: list[torch.Tensor] = []
    image_acts: list[torch.Tensor] = []

    quickdraw_dir = args.quickdraw_dir if args.quickdraw_dir.exists() else None
    print(f"[day0] image source: {'QuickDraw at ' + str(quickdraw_dir) if quickdraw_dir else 'PROCEDURAL FALLBACK (low signal expected)'}", flush=True)

    for i, concept in enumerate(concepts):
        print(f"[day0] {i+1:3d}/{len(concepts)}: {concept}", flush=True)
        text_input = f"I am thinking about a {concept}."
        try:
            text_acts.append(extract_text_residuals(model, processor, text_input, args.device))
        except Exception as e:
            print(f"  TEXT FAIL: {e}", flush=True)
            text_acts.append(None)

        # Average activations over multiple QuickDraw samples to reduce per-sample noise
        try:
            stacked = None
            n_used = 0
            for k in range(args.n_image_samples):
                if quickdraw_dir is not None:
                    sketch = render_concept_quickdraw(concept, quickdraw_dir, seed=k)
                    if sketch is None:
                        sketch = render_concept_sketch_procedural(concept)
                else:
                    sketch = render_concept_sketch_procedural(concept)
                act = extract_image_residuals(model, processor, sketch, device=args.device)
                stacked = act if stacked is None else stacked + act
                n_used += 1
            avg_act = stacked / max(1, n_used) if stacked is not None else None
            image_acts.append(avg_act)
        except Exception as e:
            print(f"  IMAGE FAIL: {e}", flush=True)
            image_acts.append(None)

    # Drop any failed concepts
    keep = [i for i in range(len(concepts)) if text_acts[i] is not None and image_acts[i] is not None]
    print(f"[day0] {len(keep)} concepts kept (out of {len(concepts)})", flush=True)

    n_positions = text_acts[keep[0]].shape[0]
    alignment = np.zeros(n_positions)
    control = np.zeros(n_positions)
    counts_align = 0
    counts_control = 0

    # Aligned pairs: same concept, text vs image
    for i in keep:
        for ell in range(n_positions):
            alignment[ell] += cosine_sim(text_acts[i][ell], image_acts[i][ell])
        counts_align += 1

    # Control pairs: random pairs of distinct concepts
    rng = np.random.default_rng(0)
    n_control_samples = min(500, len(keep) * (len(keep) - 1))
    for _ in range(n_control_samples):
        i, j = rng.choice(keep, size=2, replace=False)
        for ell in range(n_positions):
            control[ell] += cosine_sim(text_acts[i][ell], image_acts[j][ell])
        counts_control += 1

    alignment = alignment / max(1, counts_align)
    control = control / max(1, counts_control)
    delta = alignment - control

    # Write JSON
    result = {
        "model_id": args.model_id,
        "n_concepts_kept": int(len(keep)),
        "n_positions": int(n_positions),
        "alignment_per_layer": alignment.tolist(),
        "control_per_layer": control.tolist(),
        "delta_per_layer": delta.tolist(),
        "best_layer": int(np.argmax(delta)),
        "best_delta": float(np.max(delta)),
        "concepts": concepts,
    }
    json_path = output_dir / "day0_alignment.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[day0] wrote {json_path}", flush=True)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = list(range(n_positions))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(xs, alignment, label="aligned pairs (same concept)", linewidth=2)
        ax.plot(xs, control, label="control pairs (random concepts)", linewidth=2, alpha=0.7)
        ax.fill_between(xs, control, alignment, where=(alignment > control), alpha=0.15, color="green", label="alignment > control")
        ax.set_xlabel(f"layer index (0 = embedding, 1..{n_positions - 1} = transformer layers)")
        ax.set_ylabel("mean cosine similarity")
        ax.set_title(f"Day-0 cross-modal alignment in {args.model_id}\nbest layer = {result['best_layer']}, delta = {result['best_delta']:.3f}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plot_path = output_dir / "day0_alignment.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"[day0] wrote {plot_path}", flush=True)
    except ImportError:
        print(f"[day0] matplotlib not available, skipping plot")

    print(f"\n[day0] SUMMARY")
    print(f"  best layer:  {result['best_layer']}")
    print(f"  best delta:  {result['best_delta']:.4f}")
    print(f"  alignment[best]: {alignment[result['best_layer']]:.4f}")
    print(f"  control  [best]: {control[result['best_layer']]:.4f}")
    print(f"  top 5 layers by delta: {sorted(range(n_positions), key=lambda i: -delta[i])[:5]}")


if __name__ == "__main__":
    main()
