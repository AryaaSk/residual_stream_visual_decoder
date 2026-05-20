"""3-way AV comparison: SFT-only vs Stage-3-v1 vs Stage-3-v2.

For each of N demo prompts, extracts h_ℓ from Gemma 4 then runs activation
injection through all three AVs in turn. Saves PNG + MP4 per AV per prompt,
plus a 3-column HTML grid.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render import render as stroke_render  # noqa: E402
from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402

DEMO_TEXTS = [
    ("capital_france", "The capital of France is", "factual"),
    ("hamlet", "The author of Hamlet is", "factual"),
    ("smile", "She smiled brightly at the surprise.", "emotional"),
    ("dog", "I am thinking about a dog.", "concept"),
    ("cat", "I am thinking about a cat.", "concept"),
    ("triangle", "Imagine a triangle inscribed in a circle.", "spatial"),
    ("storm", "When the storm hit, the village", "narrative"),
    ("paris", "Paris, the city of lights, is famous for the Eiffel", "factual"),
    ("math", "What is 47 + 38? The answer is", "arithmetic"),
    ("emotion", "She received the news and felt deeply", "emotional"),
    ("code", "def fibonacci(n):", "code"),
    ("face", "I am picturing a smiling face.", "concept"),
    ("multihop", "The mother of Barack Obama's wife is named", "multihop"),
    ("primary", "The three primary colours are", "list"),
    ("neg", "Paris is not the capital of", "negation"),
]


@torch.no_grad()
def run_av(av: StrokeDecoder, hs: list[torch.Tensor], layer: int, alpha: float,
           max_tokens: int, out_dir: Path, suffix: str) -> list[dict]:
    rows = []
    for (slug, text, _), h in zip(DEMO_TEXTS, hs):
        ids = av.generate_from_activation(
            h, layer_ell=layer, alpha=alpha, max_new_tokens=max_tokens, temperature=1.0,
        )
        strokes, malformed = av.vocab.decode_tokens_with_stats(ids.tolist())
        png_path = out_dir / f"{slug}_{suffix}.png"
        mp4_path = out_dir / f"{slug}_{suffix}.mp4"
        png_path_4x = out_dir / f"{slug}_{suffix}_4x.png"
        mp4_path_4x = out_dir / f"{slug}_{suffix}_4x.mp4"
        img = stroke_render(strokes, save_animation_path=str(mp4_path), fps=24)
        img.save(png_path)
        img_4x = stroke_render(strokes, display_scale=4.0)
        img_4x.save(png_path_4x)
        stroke_render(strokes, display_scale=4.0, save_animation_path=str(mp4_path_4x), fps=24)
        rows.append({"slug": slug, "text": text, "strokes": len(strokes), "tokens": len(ids), "malformed": malformed})
        print(f"  {suffix}: {slug:15s} strokes={len(strokes):3d} malformed={malformed:3d}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--av-sft", type=Path, required=True)
    parser.add_argument("--av-v1", type=Path, required=True)
    parser.add_argument("--av-v2", type=Path, required=True)
    parser.add_argument("--label-sft", default="SFT only")
    parser.add_argument("--label-v1", default="Stage 3 v1 (AR v1)")
    parser.add_argument("--label-v2", default="Stage 3 v2 (AR v2)")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=Path("findings/compare_three"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Use SFT AV to extract activations (it's the base; all three share the same backbone for extraction)
    print(f"[3way] loading SFT AV from {args.av_sft}", flush=True)
    av_sft = StrokeDecoder.from_ckpt(args.av_sft, model_id=args.model_id)
    av_sft.model.eval()
    device = av_sft.device()

    hs = []
    for slug, text, _ in DEMO_TEXTS:
        enc = av_sft.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = av_sft.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach().clone()
        hs.append(h)

    rows_sft = run_av(av_sft, hs, args.layer, args.alpha, args.max_tokens, args.out_dir, "sft")

    del av_sft; gc.collect(); torch.cuda.empty_cache()

    print(f"[3way] loading v1 AV from {args.av_v1}", flush=True)
    av_v1 = StrokeDecoder.from_ckpt(args.av_v1, model_id=args.model_id)
    av_v1.model.eval()
    rows_v1 = run_av(av_v1, hs, args.layer, args.alpha, args.max_tokens, args.out_dir, "v1")
    del av_v1; gc.collect(); torch.cuda.empty_cache()

    print(f"[3way] loading v2 AV from {args.av_v2}", flush=True)
    av_v2 = StrokeDecoder.from_ckpt(args.av_v2, model_id=args.model_id)
    av_v2.model.eval()
    rows_v2 = run_av(av_v2, hs, args.layer, args.alpha, args.max_tokens, args.out_dir, "v2")
    del av_v2; gc.collect(); torch.cuda.empty_cache()

    # HTML grid
    parts = [f"""<!doctype html><html><head><meta charset='utf-8'>
<title>3-way AV comparison — layer {args.layer}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1400px; margin: 2em auto; padding: 0 1em; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ border: 1px solid #ddd; padding: 0.5em; vertical-align: top; text-align: center; }}
th {{ background: #f7f7f7; }}
.prompt {{ font-size: 13px; color: #444; text-align: left; }}
.cat {{ display: inline-block; padding: 2px 5px; background: #eef; border-radius: 4px; font-size: 11px; color: #557; }}
.stats {{ font-size: 11px; color: #888; }}
img {{ width: 180px; background: #fff; border: 1px solid #eee; }}
video {{ width: 180px; }}
</style></head><body>
<h1>Three-way AV comparison — layer {args.layer}, α={args.alpha}</h1>
<p>For each prompt, h_ℓ extracted from Gemma 4 E2B at layer {args.layer}, then injected into each AV. All three AVs share the same Stage-1 SFT base; v1 and v2 differ in the AR they were RL'd against.</p>
<table>
<tr><th>Prompt</th><th>{args.label_sft}</th><th>{args.label_v1}</th><th>{args.label_v2}</th></tr>"""]
    for i, (slug, text, cat) in enumerate(DEMO_TEXTS):
        s, a, b = rows_sft[i], rows_v1[i], rows_v2[i]
        parts.append(f"""<tr>
<td class='prompt'><span class='cat'>{cat}</span> <b>{slug}</b><br>{text}</td>
<td><img src='{slug}_sft_4x.png'><br><video src='{slug}_sft_4x.mp4' controls muted loop></video><br><span class='stats'>strokes={s['strokes']} malf={s['malformed']}</span></td>
<td><img src='{slug}_v1_4x.png'><br><video src='{slug}_v1_4x.mp4' controls muted loop></video><br><span class='stats'>strokes={a['strokes']} malf={a['malformed']}</span></td>
<td><img src='{slug}_v2_4x.png'><br><video src='{slug}_v2_4x.mp4' controls muted loop></video><br><span class='stats'>strokes={b['strokes']} malf={b['malformed']}</span></td>
</tr>""")
    parts.append("</table></body></html>")
    (args.out_dir / "index.html").write_text("".join(parts))
    print(f"[3way] wrote {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
