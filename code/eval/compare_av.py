"""Side-by-side comparison of two AV checkpoints (e.g., Stage-1 vs Stage-3).

For each demo prompt, extracts h_ℓ from Gemma 4 and runs activation injection
through BOTH AVs in turn. Saves two PNGs and two MP4s per prompt + a side-by-side
HTML index.

Usage:
    python code/eval/compare_av.py \
        --av-a checkpoints/av_sft/final --label-a "Stage 1 (SFT only)" \
        --av-b checkpoints/av_grpo/L16/final --label-b "Stage 3 (RL faithful)" \
        --layer 16
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render import render as stroke_render  # noqa: E402
from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402

DEMO_TEXTS = [
    ("capital_france", "The capital of France is"),
    ("smile", "She smiled brightly at the surprise."),
    ("dog", "I am thinking about a dog."),
    ("triangle", "Imagine a triangle inscribed in a circle."),
    ("storm", "When the storm hit, the village"),
    ("paris", "Paris, the city of lights, is famous for the Eiffel"),
    ("math", "What is 47 + 38? The answer is"),
    ("emotion", "She received the news and felt deeply"),
    ("code", "def fibonacci(n):"),
    ("face", "I am picturing a smiling face."),
]


@torch.no_grad()
def run_av_on_all(av: StrokeDecoder, hs: list[torch.Tensor], layer: int,
                  alpha: float, max_tokens: int, temperature: float,
                  out_dir: Path, suffix: str) -> list[dict]:
    rows = []
    for (slug, text), h in zip(DEMO_TEXTS, hs):
        print(f"  {suffix}: {slug}", flush=True)
        ids = av.generate_from_activation(
            h, layer_ell=layer, alpha=alpha,
            max_new_tokens=max_tokens, temperature=temperature,
        )
        strokes, malformed = av.vocab.decode_tokens_with_stats(ids.tolist())
        png_path = out_dir / f"{slug}_{suffix}.png"
        mp4_path = out_dir / f"{slug}_{suffix}.mp4"
        img = stroke_render(strokes, save_animation_path=str(mp4_path), fps=24)
        img.save(png_path)
        # 4× display upscale (vector-lossless re-render at higher resolution)
        img_4x = stroke_render(strokes, display_scale=4.0)
        img_4x.save(out_dir / f"{slug}_{suffix}_4x.png")
        stroke_render(strokes, display_scale=4.0, save_animation_path=str(out_dir / f"{slug}_{suffix}_4x.mp4"), fps=24)
        rows.append({"slug": slug, "text": text, "strokes": len(strokes), "tokens": len(ids), "malformed": malformed})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--av-a", type=Path, required=True)
    parser.add_argument("--av-b", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=Path("findings/compare_av"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load AV-A and extract activations ONCE (since AV-A and AV-B share the
    # base Gemma 4, we can reuse h from either)
    print(f"[compare] loading {args.label_a} from {args.av_a}", flush=True)
    av_a = StrokeDecoder.from_ckpt(args.av_a, model_id=args.model_id)
    av_a.model.eval()
    device = av_a.device()

    hs = []
    for slug, text in DEMO_TEXTS:
        enc = av_a.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = av_a.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach().clone()
        hs.append(h)

    rows_a = run_av_on_all(av_a, hs, args.layer, args.alpha, args.max_tokens, args.temperature, args.out_dir, "A")

    # Free AV-A to free GPU memory before loading AV-B
    del av_a
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[compare] loading {args.label_b} from {args.av_b}", flush=True)
    av_b = StrokeDecoder.from_ckpt(args.av_b, model_id=args.model_id)
    av_b.model.eval()
    rows_b = run_av_on_all(av_b, hs, args.layer, args.alpha, args.max_tokens, args.temperature, args.out_dir, "B")

    # Build comparison HTML
    parts = [f"""<!doctype html><html><head><meta charset='utf-8'>
<title>AV comparison — layer {args.layer}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ border: 1px solid #ddd; padding: 0.5em; vertical-align: top; }}
th {{ background: #f7f7f7; }}
.prompt {{ font-size: 13px; color: #444; }}
.stats {{ font-size: 11px; color: #888; }}
img, video {{ width: 200px; background: #fff; border: 1px solid #eee; }}
</style></head><body>
<h1>AV comparison — layer {args.layer}</h1>
<p>For each prompt, h_ℓ extracted from Gemma 4 E2B at layer {args.layer}, then injected into each AV via embedding-layer hook.</p>
<table>
<tr><th>Prompt</th><th>{args.label_a} (PNG / MP4)</th><th>{args.label_b} (PNG / MP4)</th></tr>"""]
    for (slug, text), a, b in zip(DEMO_TEXTS, rows_a, rows_b):
        parts.append(f"""
<tr>
  <td><b>{slug}</b><br><span class='prompt'>{text}</span></td>
  <td><img src='{slug}_A_4x.png'><br><video src='{slug}_A_4x.mp4' controls muted loop></video><br>
      <span class='stats'>strokes={a['strokes']}, tokens={a['tokens']}, malformed={a['malformed']} · <a href='{slug}_A.png'>native 224px</a></span></td>
  <td><img src='{slug}_B_4x.png'><br><video src='{slug}_B_4x.mp4' controls muted loop></video><br>
      <span class='stats'>strokes={b['strokes']}, tokens={b['tokens']}, malformed={b['malformed']} · <a href='{slug}_B.png'>native 224px</a></span></td>
</tr>""")
    parts.append("</table></body></html>")
    (args.out_dir / "index.html").write_text("".join(parts))
    print(f"[compare] index: {args.out_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
