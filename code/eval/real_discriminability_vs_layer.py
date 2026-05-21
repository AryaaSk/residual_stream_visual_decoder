"""Diagnose where in Qwen 3.5-4B's stack the foundational NLA signal actually exists.

For each layer L, compute the FULL pairwise cosine matrix between text and image
activations of N (caption, real_canonical_drawing) pairs:

    M_L[i, j] = cosine(h_text(caption_i)[L], h_image(drawing_j)[L])

Discriminability at L = mean(diag) - mean(off_diag)
Top-1 retrieval at L = fraction where argmax_j M_L[i,:] == i

If real canonical drawings ARE concept-discriminable at some layer L*, then v3
training has a real signal to optimise toward. If not at ANY layer, the
foundational architecture is unworkable with this base model and we ship the
negative finding honestly.

Output:
    findings/v3/real_discriminability/per_layer.json
    findings/v3/real_discriminability/per_layer.png
    findings/v3/real_discriminability/heatmaps/L{NN}.png    (sample heatmaps)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render import render as stroke_render
from stroke_tokenizer import Stroke


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# Pick one canonical drawing per concept; ensures pairwise concepts are all distinct
def one_drawing_per_concept(rows: list[dict]) -> list[dict]:
    by_concept: dict[str, dict] = {}
    for r in rows:
        cap = r.get("caption", "")
        # Concept ≈ last word of canonical caption "a drawing of a {concept}"
        concept = cap.replace("a drawing of an ", "").replace("a drawing of a ", "")
        concept = concept.replace("a drawing of ", "").strip().rstrip("s")
        if concept and concept not in by_concept:
            by_concept[concept] = {"concept": concept, "caption": cap, "strokes": r["strokes"]}
    return list(by_concept.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--data", type=Path,
                   default=Path("data/canonical_drawings_top5.jsonl"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("findings/v3/real_discriminability"))
    p.add_argument("--layers", type=int, nargs="+",
                   default=[0, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 29])
    p.add_argument("--max-concepts", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--display-scale", type=float, default=2.0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "heatmaps").mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    print(f"[disc] loading Qwen 3.5-4B (ImageTextToText) ...", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    device = next(model.parameters()).device

    print(f"[disc] loading canonical drawings ...", flush=True)
    raw = load_jsonl(args.data)
    rows = one_drawing_per_concept(raw)
    if len(rows) > args.max_concepts:
        rows = rows[: args.max_concepts]
    print(f"[disc] {len(rows)} distinct concepts", flush=True)

    # Pre-extract h_text and h_image (per layer, per concept)
    n = len(rows)
    n_layers_check = max(args.layers) + 1
    # Storage: layer → concept_idx → h vector (cpu)
    h_text_by_layer: dict[int, list[torch.Tensor]] = {L: [None] * n for L in args.layers}
    h_img_by_layer: dict[int, list[torch.Tensor]] = {L: [None] * n for L in args.layers}

    @torch.no_grad()
    def extract_all_layers_text(caption: str) -> list[torch.Tensor]:
        msgs = [{"role": "user", "content": [{"type": "text", "text": caption}]}]
        wrap = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        inp = proc(text=[wrap], images=None, return_tensors="pt").to(device)
        out = model(**inp, output_hidden_states=True, use_cache=False)
        return [hs[0, -1, :].detach().to(torch.float32).cpu() for hs in out.hidden_states]

    @torch.no_grad()
    def extract_all_layers_image(image) -> list[torch.Tensor]:
        msgs = [{"role": "user", "content": [{"type": "image"}]}]
        wrap = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        inp = proc(text=[wrap], images=[image], return_tensors="pt").to(device)
        out = model(**inp, output_hidden_states=True, use_cache=False)
        return [hs[0, -1, :].detach().to(torch.float32).cpu() for hs in out.hidden_states]

    t0 = time.time()
    for i, r in enumerate(rows):
        # text forward
        ht = extract_all_layers_text(r["caption"])
        # image forward
        stroke_objs = [Stroke(dx=s["dx"], dy=s["dy"], pen=s["pen"]) for s in r["strokes"]]
        img = stroke_render(stroke_objs, display_scale=args.display_scale).convert("RGB")
        hi = extract_all_layers_image(img)
        for L in args.layers:
            h_text_by_layer[L][i] = ht[L]
            h_img_by_layer[L][i] = hi[L]
        if (i + 1) % 5 == 0:
            print(f"[disc] {i+1}/{n} concepts extracted ({time.time()-t0:.1f}s)", flush=True)
    print(f"[disc] extraction done in {time.time()-t0:.1f}s", flush=True)

    # Compute pairwise matrix per layer + discriminability metrics
    per_layer = []
    for L in args.layers:
        HT = torch.stack(h_text_by_layer[L], dim=0)  # (n, d)
        HI = torch.stack(h_img_by_layer[L], dim=0)   # (n, d)
        # Normalize and matmul to get cosine matrix
        HT_n = HT / (HT.norm(dim=1, keepdim=True) + 1e-9)
        HI_n = HI / (HI.norm(dim=1, keepdim=True) + 1e-9)
        M = (HT_n @ HI_n.T).numpy()  # (n, n)
        diag = np.diag(M)
        off_mask = ~np.eye(n, dtype=bool)
        off = M[off_mask].reshape(n, n - 1).mean(axis=1)
        disc = (diag - off).mean()
        top1 = float(np.mean(np.argmax(M, axis=1) == np.arange(n)))
        rec = {
            "layer": L,
            "n_concepts": n,
            "diag_mean": float(diag.mean()),
            "diag_std": float(diag.std()),
            "off_diag_mean": float(off.mean()),
            "discriminability_mean": float(disc),
            "discriminability_per_concept": disc.tolist() if hasattr(disc, "tolist") else float(disc),
            "top1_retrieval": top1,
            "top1_chance": 1.0 / n,
        }
        per_layer.append(rec)
        print(f"[disc] L{L:2d}  diag={float(diag.mean()):+.3f}  off={float(off.mean()):+.3f}  disc={float(disc):+.4f}  top1={top1*100:.1f}%  (chance {100/n:.1f}%)", flush=True)
        # Save heatmap
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels([r["concept"] for r in rows], rotation=70, fontsize=7, ha="right")
            ax.set_yticklabels([r["concept"] for r in rows], fontsize=7)
            ax.set_xlabel("real image's concept")
            ax.set_ylabel("text caption's concept")
            ax.set_title(f"L{L}: diag={float(diag.mean()):+.3f}  off={float(off.mean()):+.3f}  disc={float(disc):+.3f}  top1={top1*100:.1f}%")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(args.out_dir / "heatmaps" / f"L{L:02d}.png", dpi=110)
            plt.close()
        except Exception as e:
            print(f"[disc] WARN: heatmap save failed: {e}", flush=True)

    (args.out_dir / "per_layer.json").write_text(json.dumps(per_layer, indent=2))

    # Summary plot: discriminability + top1 vs layer
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ls = [r["layer"] for r in per_layer]
        diags = [r["diag_mean"] for r in per_layer]
        offs = [r["off_diag_mean"] for r in per_layer]
        discs = [r["discriminability_mean"] for r in per_layer]
        top1s = [r["top1_retrieval"] * 100 for r in per_layer]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax1 = axes[0]
        ax1.plot(Ls, diags, "o-", color="#2a7", label="diag (correct pair)")
        ax1.plot(Ls, offs, "s-", color="#a72", label="off-diag (wrong pair)")
        ax1.plot(Ls, discs, "^-", color="#27a", label="discriminability (diag - off)")
        ax1.set_xlabel("Layer L")
        ax1.set_ylabel("cosine")
        ax1.axhline(0, color="grey", linewidth=0.5)
        ax1.legend()
        ax1.set_title(f"Cosine alignment vs layer (n={n} real concept pairs)")
        ax2 = axes[1]
        ax2.plot(Ls, top1s, "o-", color="#a27")
        ax2.axhline(100 / n, color="grey", linestyle="--", label=f"chance ({100/n:.1f}%)")
        ax2.set_xlabel("Layer L")
        ax2.set_ylabel("Top-1 retrieval (%)")
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.set_title("Top-1 retrieval vs layer")
        plt.tight_layout()
        plt.savefig(args.out_dir / "per_layer.png", dpi=120)
        print(f"[disc] saved {args.out_dir / 'per_layer.png'}", flush=True)
    except Exception as e:
        print(f"[disc] WARN: summary plot failed: {e}", flush=True)

    # Pick recommended layer = max discriminability above some sensible threshold
    best_L = max(per_layer, key=lambda r: r["discriminability_mean"])
    print(f"\n[disc] RECOMMENDED LAYER for v3 training: L{best_L['layer']}", flush=True)
    print(f"  discriminability = {best_L['discriminability_mean']:+.4f}", flush=True)
    print(f"  top-1 retrieval  = {best_L['top1_retrieval']*100:.1f}%  (chance {100/n:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
