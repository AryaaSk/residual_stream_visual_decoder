"""The foundational NLA experiment for this project.

Hypothesis (the whole reason we ported to Qwen 3.5-4B):
    Because Qwen has a natively unified vision+text residual stream,
    h_text("a cat") and h_image(drawing_of_cat) should converge as depth
    grows. Early layers: modality-specific features → low cosine. Late
    layers: abstract concept space → high cosine.

If true, this gives us a free, frozen, principled AR: feed the drawing
back through Qwen, read the activation, compare to h_text. No trained
reconstructor needed.

If false, the project's premise needs revisiting — and we should ship that
honestly as a finding.

The experiment:
  For each (concept, real_canonical_drawing) in canonical_drawings_top5.jsonl:
    h_text  = Qwen(caption)[L][last_text_token]              for L in 0..32
    h_image = Qwen(images=rendered_drawing)[L][last_token]   for L in 0..32
    cos[L]  = cosine(h_text, h_image)
  Plot mean cos[L] over all (concept, drawing) pairs as a function of L.

The expected shape if hypothesis holds:
                                                 ●●●●●●●
  cosine            ●●●●●●●●●●●●●
       ●●●●●●●●●●●
       0    5   10   15   20   25   30
                       layer L

The expected shape if hypothesis fails:
       ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● (flat near 0 or near 1)

Output:
    findings/v2_2/text_image_alignment/per_layer.json
    findings/v2_2/text_image_alignment/per_layer.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render import render as stroke_render


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--data", type=Path,
                   default=Path("data/canonical_drawings_top5.jsonl"),
                   help="JSONL with caption + strokes")
    p.add_argument("--out-dir", type=Path,
                   default=Path("findings/v2_2/text_image_alignment"))
    p.add_argument("--max-pairs", type=int, default=80,
                   help="Subsample for speed; -1 means all")
    p.add_argument("--display-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[align] loading Qwen base {args.model_id} ...", flush=True)
    from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    try:
        proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        has_mm = True
        print("[align] AutoProcessor loaded (multimodal path)", flush=True)
    except Exception as e:
        proc = None
        has_mm = False
        print(f"[align] AutoProcessor unavailable ({e}); falling back to tokenizer-only", flush=True)
    # CRITICAL: AutoModelForImageTextToText loads the vision encoder. AutoModelForCausalLM
    # would silently drop pixel_values and make h_image identical for every input.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    device = next(model.parameters()).device

    # Inspect number of layers
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None)
    if n_layers is None and hasattr(cfg, "text_config"):
        n_layers = cfg.text_config.num_hidden_layers
    print(f"[align] model has {n_layers} hidden layers", flush=True)

    # Load drawings
    drawings = load_jsonl(args.data)
    if args.max_pairs > 0 and len(drawings) > args.max_pairs:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(drawings), size=args.max_pairs, replace=False)
        drawings = [drawings[i] for i in idx]
    print(f"[align] {len(drawings)} (caption, drawing) pairs", flush=True)

    # Group cosines by layer
    cos_by_layer: list[list[float]] = [[] for _ in range(n_layers + 1)]
    pair_cos: list[dict] = []

    # Use Qwen-VL chat template for BOTH paths so the read position is
    # structurally identical: input ends with `<|im_end|>\n`, we read the last
    # token activation. This avoids comparing "last word of caption" vs
    # "last image patch" which would be confounded by position.
    @torch.no_grad()
    def h_text_layers(caption: str):
        if has_mm:
            messages = [{"role": "user", "content": [{"type": "text", "text": caption}]}]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = proc(text=[text], images=None, return_tensors="pt").to(device)
        else:
            inputs = tok(caption, return_tensors="pt", add_special_tokens=True).to(device)
        out = model(**inputs, output_hidden_states=True, use_cache=False)
        return [hs[0, -1, :].detach().to(torch.float32).cpu() for hs in out.hidden_states]

    @torch.no_grad()
    def h_image_layers(image):
        if not has_mm:
            return None
        # Image-only path via chat template — NO caption, no text content beyond
        # the conversation wrapper tokens.
        messages = [{"role": "user", "content": [{"type": "image"}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        try:
            inputs = proc(text=[text], images=[image], return_tensors="pt").to(device)
            out = model(**inputs, output_hidden_states=True, use_cache=False)
            return [hs[0, -1, :].detach().to(torch.float32).cpu() for hs in out.hidden_states]
        except Exception as e:
            print(f"[align]   image-only chat-template path failed: {type(e).__name__}: {str(e)[:140]}", flush=True)
            return None

    t0 = time.time()
    n_done = 0
    n_skipped = 0
    for row in drawings:
        caption = row.get("caption")
        strokes_raw = row.get("strokes", [])
        if not caption or not strokes_raw:
            n_skipped += 1
            continue
        # Render the canonical drawing as a PIL image
        from stroke_tokenizer import Stroke
        stroke_objs = [Stroke(dx=s["dx"], dy=s["dy"], pen=s["pen"]) for s in strokes_raw]
        img = stroke_render(stroke_objs, display_scale=args.display_scale).convert("RGB")

        ht = h_text_layers(caption)
        hi = h_image_layers(img)
        if hi is None:
            n_skipped += 1
            continue
        if len(ht) != len(hi):
            print(f"[align]   layer count mismatch {len(ht)} vs {len(hi)}, skipping", flush=True)
            n_skipped += 1
            continue
        layer_cos: dict[int, float] = {}
        for L, (a, b) in enumerate(zip(ht, hi)):
            c = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            cos_by_layer[L].append(c)
            layer_cos[L] = round(c, 4)
        pair_cos.append({"caption": caption, "layer_cos": layer_cos})
        n_done += 1
        if n_done % 10 == 0:
            elapsed = time.time() - t0
            print(f"[align] {n_done}/{len(drawings)} pairs done  ({elapsed:.1f}s)", flush=True)

    print(f"\n[align] done: {n_done} pairs, {n_skipped} skipped, total {time.time()-t0:.1f}s", flush=True)

    # Aggregate
    mean_per_layer = []
    std_per_layer = []
    for L in range(len(cos_by_layer)):
        if cos_by_layer[L]:
            mean_per_layer.append(float(np.mean(cos_by_layer[L])))
            std_per_layer.append(float(np.std(cos_by_layer[L])))
        else:
            mean_per_layer.append(None)
            std_per_layer.append(None)

    results = {
        "model_id": args.model_id,
        "n_pairs_done": n_done,
        "n_pairs_skipped": n_skipped,
        "n_layers_incl_embed": len(cos_by_layer),
        "mean_cosine_per_layer": mean_per_layer,
        "std_cosine_per_layer": std_per_layer,
        "per_pair": pair_cos,
    }
    (args.out_dir / "per_layer.json").write_text(json.dumps(results, indent=2))
    print(f"[align] saved {args.out_dir / 'per_layer.json'}", flush=True)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ls = list(range(len(cos_by_layer)))
        means = np.array([m if m is not None else float("nan") for m in mean_per_layer])
        stds = np.array([s if s is not None else 0.0 for s in std_per_layer])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(Ls, means, marker="o", color="#3066be", label="mean cosine(h_text, h_image)")
        ax.fill_between(Ls, means - stds, means + stds, alpha=0.2, color="#3066be", label="±1 std")
        ax.set_xlabel("Layer L")
        ax.set_ylabel("cosine(h_text, h_image)")
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_title(f"Text/image alignment vs depth (Qwen 3.5-4B, n={n_done} canonical drawings)")
        ax.legend()
        for L in [0, len(Ls) // 2, len(Ls) - 1]:
            if means[L] == means[L]:  # not NaN
                ax.annotate(f"L{L}: {means[L]:.2f}", (L, means[L]),
                            xytext=(5, 5), textcoords="offset points", fontsize=9)
        plt.tight_layout()
        (args.out_dir / "per_layer.png").parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out_dir / "per_layer.png", dpi=120)
        print(f"[align] saved {args.out_dir / 'per_layer.png'}", flush=True)
    except Exception as e:
        print(f"[align] WARN: plot failed: {e}", flush=True)

    # Verdict
    print("\n=== verdict ===")
    for L in [0, 3, 10, 20, 29]:
        if L < len(mean_per_layer) and mean_per_layer[L] is not None:
            print(f"L{L:2d}: mean cosine = {mean_per_layer[L]:+.3f} ± {std_per_layer[L]:.3f}")


if __name__ == "__main__":
    main()
