"""OOD eval for v2.0 — prompts NOT in QuickDraw training. Standalone script."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402
from render import render as stroke_render  # noqa: E402

OOD = [
    ("eiffel_tower",  "The Eiffel Tower in Paris.",                    "the Eiffel Tower"),
    ("capital_paris", "The capital of France is Paris.",                "the city of Paris"),
    ("thunderstorm",  "A thunderstorm with lightning hit the village.", "a thunderstorm"),
    ("smile_face",    "I am picturing a smiling face.",                 "a smiling face"),
    ("triangle",      "Imagine a triangle inscribed in a circle.",      "a triangle"),
    ("circle",        "I am picturing a circle.",                       "a circle"),
    ("rainbow",       "A rainbow after the rain.",                      "a rainbow"),
    ("hand",          "I am picturing a human hand.",                   "a hand"),
    ("face",          "Imagine the face of an old man.",                "a face"),
    ("eye",           "An eye looking up at the sky.",                  "an eye"),
    ("book",          "An open book on a desk.",                        "a book"),
    ("bicycle",       "I am picturing a bicycle.",                      "a bicycle"),
]


def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    return CLIPModel.from_pretrained(name).to("cuda").eval(), CLIPProcessor.from_pretrained(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=240)
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()

    torch.manual_seed(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ood] loading AV {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    print(f"[ood] loaded AV; loading CLIP ...", flush=True)
    clip_model, clip_proc = load_clip()
    print(f"[ood] both loaded; starting OOD eval", flush=True)

    summary = []
    for slug, prompt, clip_text in OOD:
        print(f"\n[ood] === {slug}: {prompt!r}", flush=True)
        enc = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(av.device())
        with torch.no_grad():
            out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach()
        ids_list = av.generate_from_activation_batched(
            h, layer_ell=args.layer, n_samples=args.n_samples,
            alpha=args.alpha, max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
        candidates = []
        for i, ids in enumerate(ids_list):
            strokes, malformed = av.vocab.decode_tokens_with_stats(ids.tolist())
            if len(strokes) < 8:
                continue
            img = stroke_render(strokes, display_scale=2.0).convert("RGB")
            candidates.append({"sample": i, "strokes": strokes, "n_strokes": len(strokes), "img": img})
        if not candidates:
            print(f"[ood]   all candidates degenerate; skipping")
            continue
        imgs = [c["img"] for c in candidates]
        with torch.no_grad():
            inputs = clip_proc(text=[clip_text], images=imgs, return_tensors="pt", padding=True).to("cuda")
            scores = clip_model(**inputs).logits_per_image.squeeze(-1).tolist()
        for c, s in zip(candidates, scores):
            c["clip_score"] = float(s)
        candidates.sort(key=lambda c: c["clip_score"], reverse=True)
        top = candidates[0]
        png_path = args.out_dir / f"{slug}.png"
        mp4_path = args.out_dir / f"{slug}.mp4"
        img4 = stroke_render(top["strokes"], display_scale=4.0,
                             save_animation_path=str(mp4_path), fps=24)
        img4.save(png_path)
        print(f"[ood]   best CLIP={top['clip_score']:.2f}  n_strokes={top['n_strokes']}  → {png_path.name}", flush=True)
        summary.append({"slug": slug, "prompt": prompt, "clip_text": clip_text,
                        "best_clip": top["clip_score"], "n_strokes": top["n_strokes"]})

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[ood] DONE → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
