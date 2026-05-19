"""Quick sanity-check sampler for the Stage-1 SFT'd AV.

Loads the AV checkpoint, generates drawings for a few canned text prompts,
renders to PNG (and optionally MP4), saves to findings/stage1_samples/.

Usage:
    python code/eval/quick_sample.py --av-ckpt checkpoints/av_sft/final
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder, SFT_PROMPT_TEMPLATE  # noqa: E402
from verbalizer.activation_injection import stroke_token_ids  # noqa: E402
from stroke_tokenizer import DRAW_CLOSE  # noqa: E402
from render import render as stroke_render  # noqa: E402


PROMPTS = [
    ("cat", "a drawing of a cat"),
    ("dog", "a drawing of a dog"),
    ("tree", "a drawing of a tree"),
    ("house", "a drawing of a house"),
    ("car", "a drawing of a car"),
    ("airplane", "a drawing of an airplane"),
    ("flower", "a drawing of a flower"),
    ("bicycle", "a drawing of a bicycle"),
]


@torch.no_grad()
def sample_drawing(av: StrokeDecoder, caption: str, max_tokens: int = 400, temperature: float = 1.0) -> list[int]:
    device = next(av.model.parameters()).device
    prompt = SFT_PROMPT_TEMPLATE.format(caption=caption)
    prompt_ids = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
    allowed = set(stroke_token_ids(av.vocab))
    allowed_ids = torch.tensor(list(allowed), device=device)
    mask = torch.full((av.model.get_input_embeddings().weight.shape[0],), float("-inf"), device=device)
    mask[allowed_ids] = 0.0

    gen: list[int] = []
    past = None
    cur = prompt_ids
    for _ in range(max_tokens):
        if past is None:
            out = av.model(input_ids=cur, use_cache=True)
        else:
            out = av.model(input_ids=torch.tensor([[gen[-1]]], device=device), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :] + mask
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        gen.append(nxt)
        if nxt == av.vocab.name_to_id[DRAW_CLOSE]:
            break
    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--av-ckpt", type=Path, required=True)
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it")
    parser.add_argument("--out-dir", type=Path, default=Path("findings/stage1_samples"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mp4", action="store_true", default=False)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[quick_sample] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()

    for slug, caption in PROMPTS:
        print(f"[quick_sample] sampling '{caption}'...", flush=True)
        gen_ids = sample_drawing(av, caption, temperature=args.temperature)
        strokes, malformed = av.vocab.decode_tokens_with_stats(gen_ids)
        png_path = args.out_dir / f"{slug}.png"
        mp4_path = args.out_dir / f"{slug}.mp4" if args.mp4 else None
        img = stroke_render(strokes, save_animation_path=str(mp4_path) if mp4_path else None, fps=24)
        img.save(png_path)
        print(f"  → {png_path.name}  strokes={len(strokes)}  tokens={len(gen_ids)}  malformed={malformed}", flush=True)

    # Build a quick index HTML
    html = ["<!doctype html><html><body><h2>Stage 1 samples</h2><div style='display:flex;flex-wrap:wrap;gap:1em'>"]
    for slug, caption in PROMPTS:
        html.append(f"<div style='text-align:center;border:1px solid #ddd;padding:.5em'>"
                    f"<img src='{slug}.png' style='width:200px'><br>{caption}</div>")
    html.append("</div></body></html>")
    (args.out_dir / "index.html").write_text("".join(html))
    print(f"[quick_sample] index: {args.out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
