"""Activation interpolation — the smooth morph demo for v2.2.

Pick two prompts P_A, P_B. Extract h_A = Qwen(P_A).hidden_states[L][last]
and h_B similarly. For alpha in linspace(0, 1, n_steps):
    h_alpha = (1 - alpha) * h_A + alpha * h_B
Generate a drawing from each h_alpha (best-of-N CLIP-ranked against BOTH
prompts; we report which side wins at each step).

Compose the n_steps drawings into a smooth morph MP4: 5 frames per drawing
at 24 fps → ~2 seconds per pair if n_steps=10, ~6s if n_steps=30.

Smooth morph (frames blend continuously across alpha) → the AV is decoding
h *continuously* in residual-stream space.

Discrete snap at alpha=0.5 → the AV is template-matching: classifying h
into N buckets, then emitting whichever template wins.

Either outcome is informative; output the verdict in summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder
from render import render as stroke_render


# Default morph pairs: 5 pairs that should each give visually distinguishable endpoints.
DEFAULT_PAIRS = [
    ("cat_to_elephant",  "I am thinking about a cat.",     "I am thinking about an elephant.",
                          "a drawing of a cat",            "a drawing of an elephant"),
    ("dog_to_horse",     "I am thinking about a dog.",     "I am thinking about a horse.",
                          "a drawing of a dog",            "a drawing of a horse"),
    ("fish_to_bird",     "Imagine a fish.",                "I am picturing a bird flying across the sky.",
                          "a drawing of a fish",           "a drawing of a bird"),
    ("sun_to_cloud",     "The sun is shining.",            "I am picturing a cloud in the sky.",
                          "a drawing of a sun",            "a drawing of a cloud"),
    ("apple_to_pizza",   "I am thinking about an apple.",  "I am thinking about a pizza.",
                          "a drawing of an apple",         "a drawing of a pizza"),
]


def extract_h(av, prompt: str, layer: int) -> torch.Tensor:
    enc = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(av.device())
    with torch.no_grad():
        out = av.model(**enc, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer][0, -1, :].detach()


def best_candidate(av, h: torch.Tensor, layer: int, n_samples: int,
                   max_tokens: int, temperature: float, top_k: int,
                   clip_model, clip_proc, clip_texts: list[str]
                   ) -> tuple[Image.Image, list, dict]:
    gen_ids_list = av.generate_from_activation_batched(
        h.to(av.device()), layer_ell=layer, n_samples=n_samples,
        max_new_tokens=max_tokens, temperature=temperature, top_k=top_k,
    )
    candidates = []
    for ids in gen_ids_list:
        strokes, _ = av.vocab.decode_tokens_with_stats(ids.tolist())
        if len(strokes) < 8:
            continue
        img = stroke_render(strokes, display_scale=2.0).convert("RGB")
        candidates.append({"strokes": strokes, "img": img, "n_strokes": len(strokes)})
    if not candidates:
        return None, None, {"degenerate": True}
    imgs = [c["img"] for c in candidates]
    with torch.no_grad():
        inputs = clip_proc(text=clip_texts, images=imgs, return_tensors="pt", padding=True).to("cuda")
        out_clip = clip_model(**inputs)
        logits = out_clip.logits_per_image.detach().cpu()  # (N_img, N_text)
    # Pick the candidate whose MAX score across clip_texts is highest
    max_scores = logits.max(dim=1).values.tolist()
    winner_idx = int(np.argmax(max_scores))
    scores = logits[winner_idx].tolist()
    return candidates[winner_idx]["img"], candidates[winner_idx]["strokes"], {
        "scores": [round(s, 2) for s in scores],
        "winner_max": round(max(scores), 2),
        "n_strokes": candidates[winner_idx]["n_strokes"],
    }


def overlay_caption(img: Image.Image, text: str, sub: str = "") -> Image.Image:
    W = img.size[0]
    PAD = 80
    out = Image.new("RGB", (W, img.size[1] + PAD), color=(255, 255, 255))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = font_big
    draw.text((10, img.size[1] + 10), text, fill=(0, 0, 0), font=font_big)
    if sub:
        draw.text((10, img.size[1] + 45), sub, fill=(80, 80, 80), font=font_small)
    return out


def write_mp4(frames_pil: list[Image.Image], path: Path, fps: int = 24,
              hold_frames: int = 4, crossfade_frames: int = 4):
    import imageio.v3 as iio
    if not frames_pil:
        return
    W, H = frames_pil[0].size

    def arr(im: Image.Image) -> np.ndarray:
        return np.asarray(im, dtype=np.uint8)

    out_frames: list[np.ndarray] = []
    for i, im in enumerate(frames_pil):
        a = arr(im)
        for _ in range(hold_frames):
            out_frames.append(a)
        if i < len(frames_pil) - 1 and crossfade_frames > 0:
            nx = arr(frames_pil[i + 1])
            for j in range(1, crossfade_frames + 1):
                alpha = j / (crossfade_frames + 1)
                blended = (a * (1 - alpha) + nx * alpha).astype(np.uint8)
                out_frames.append(blended)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(out_frames, axis=0), fps=fps,
                codec="libx264", macro_block_size=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--n-steps", type=int, default=15,
                   help="Interpolation steps from alpha=0 to 1 inclusive")
    p.add_argument("--n-samples", type=int, default=16,
                   help="best-of-N CLIP-rank per interpolation step")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=240)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--display-scale", type=float, default=4.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[interp] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()

    print("[interp] loading CLIP ...", flush=True)
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(name).to("cuda").eval()
    clip_proc = CLIPProcessor.from_pretrained(name)

    alphas = list(np.linspace(0.0, 1.0, args.n_steps))
    summary = {"layer": args.layer, "n_steps": args.n_steps, "pairs": []}

    for tag, prompt_a, prompt_b, clip_a, clip_b in DEFAULT_PAIRS:
        t0 = time.time()
        h_a = extract_h(av, prompt_a, args.layer)
        h_b = extract_h(av, prompt_b, args.layer)
        clip_texts = [clip_a, clip_b]

        frames_pil: list[Image.Image] = []
        per_step: list[dict] = []
        all_dir = args.out_dir / "individual" / tag
        all_dir.mkdir(parents=True, exist_ok=True)
        for i, alpha in enumerate(alphas):
            h = (1 - float(alpha)) * h_a + float(alpha) * h_b
            img, strokes, meta = best_candidate(
                av, h, args.layer, args.n_samples, args.max_tokens,
                args.temperature, args.top_k, clip_model, clip_proc, clip_texts,
            )
            if img is None:
                print(f"[interp] {tag} alpha={alpha:.2f}: degenerate", flush=True)
                # placeholder blank frame
                blank = Image.new("RGB", (224 * int(args.display_scale), 224 * int(args.display_scale)),
                                  color=(245, 245, 245))
                frames_pil.append(overlay_caption(blank, f"{prompt_a}  →  {prompt_b}",
                                                  f"α={alpha:.2f}  (degenerate)"))
                per_step.append({"alpha": float(alpha), "degenerate": True})
                continue
            # Re-render at display scale
            full = stroke_render(strokes, display_scale=args.display_scale).convert("RGB")
            full.save(all_dir / f"step_{i:02d}_alpha{alpha:.2f}.png")
            side = clip_a if meta["scores"][0] > meta["scores"][1] else clip_b
            side_score = max(meta["scores"])
            other_score = min(meta["scores"])
            caption_main = f"{prompt_a.rstrip('.')}  →  {prompt_b.rstrip('.')}"
            caption_sub = f"α={alpha:.2f}   {clip_a}={meta['scores'][0]:.1f}   {clip_b}={meta['scores'][1]:.1f}"
            frames_pil.append(overlay_caption(full, caption_main, caption_sub))
            per_step.append({"alpha": float(alpha), "n_strokes": meta["n_strokes"],
                             "score_a": meta["scores"][0], "score_b": meta["scores"][1]})
        write_mp4(frames_pil, args.out_dir / f"{tag}.mp4")
        elapsed = time.time() - t0
        # Smoothness: max stepwise change in (score_b - score_a)
        diffs = []
        prev_diff = None
        for s in per_step:
            if "degenerate" in s:
                continue
            d = s["score_b"] - s["score_a"]
            if prev_diff is not None:
                diffs.append(abs(d - prev_diff))
            prev_diff = d
        max_step = max(diffs) if diffs else 0.0
        is_smooth = max_step < 4.0  # heuristic
        print(f"[interp] {tag}: max stepwise Δ(B-A)={max_step:.2f}  → {'SMOOTH' if is_smooth else 'SNAP'}  ({elapsed:.1f}s)", flush=True)
        summary["pairs"].append({"tag": tag, "prompt_a": prompt_a, "prompt_b": prompt_b,
                                  "clip_a": clip_a, "clip_b": clip_b,
                                  "per_step": per_step, "max_stepwise_delta": float(max_step),
                                  "is_smooth": is_smooth, "wallclock_s": elapsed})

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[interp] DONE → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
