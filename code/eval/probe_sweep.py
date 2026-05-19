"""Per-layer probe sweep: render visualisation drawings for a fixed probe set.

For each probe (a text prompt), and each trained anchor layer:
    1. Run the target model on the probe, extract h_ℓ at the chosen layer.
    2. Pass h_ℓ to the trained AV (with activation injection) and sample strokes.
    3. Render the strokes to (a) final PNG, (b) animated MP4.
    4. Save artefacts.

Outputs go under: artefacts/per_probe/L{NN}/{probe_id}.png and .mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder  # noqa: E402
from render import render as stroke_render  # noqa: E402
from stroke_tokenizer import StrokeVocab  # noqa: E402

DEFAULT_PROBE_SET = [
    # Factual recall
    {"id": "factual_paris", "text": "The capital of France is", "category": "factual"},
    {"id": "factual_hamlet", "text": "The author of Hamlet is", "category": "factual"},
    {"id": "factual_water", "text": "The chemical formula for water is", "category": "factual"},
    {"id": "factual_speed_of_light", "text": "The speed of light in vacuum is", "category": "factual"},
    {"id": "factual_smallest_prime", "text": "The smallest prime number is", "category": "factual"},
    # Arithmetic
    {"id": "math_47_plus_38", "text": "What is 47 + 38? The answer is", "category": "arithmetic"},
    {"id": "math_12_times_13", "text": "What is 12 × 13? The answer is", "category": "arithmetic"},
    {"id": "math_sqrt_64", "text": "What is the square root of 64? It is", "category": "arithmetic"},
    {"id": "math_pi", "text": "Pi to two decimal places is", "category": "arithmetic"},
    {"id": "math_triangle_sides", "text": "A triangle has how many sides? It has", "category": "arithmetic"},
    # Multi-hop reasoning
    {"id": "multihop_obama_wife", "text": "Barack Obama's wife's name is", "category": "multihop"},
    {"id": "multihop_obama_wife_mother", "text": "The mother of Barack Obama's wife is named", "category": "multihop"},
    {"id": "multihop_capital_of_country_with_eiffel", "text": "The capital of the country that contains the Eiffel Tower is", "category": "multihop"},
    {"id": "multihop_continent_of_china", "text": "China is located in the continent of", "category": "multihop"},
    {"id": "multihop_president_country_eiffel", "text": "The current president of the country that contains the Eiffel Tower is named", "category": "multihop"},
    # Emotional / affective
    {"id": "emotion_funeral", "text": "The funeral was somber and", "category": "emotional"},
    {"id": "emotion_joy", "text": "Her face lit up with joy when she heard the", "category": "emotional"},
    {"id": "emotion_fear", "text": "His heart pounded with fear as the", "category": "emotional"},
    {"id": "emotion_calm", "text": "The lake was calm and", "category": "emotional"},
    {"id": "emotion_letter", "text": "She received the news and felt deeply", "category": "emotional"},
    # Ambiguous / structural
    {"id": "ambig_telescope", "text": "I saw the man with the telescope", "category": "ambiguous"},
    {"id": "ambig_visiting_relatives", "text": "Visiting relatives can be boring", "category": "ambiguous"},
    {"id": "ambig_bank", "text": "She went to the bank to deposit", "category": "ambiguous"},
    # Lists / structured
    {"id": "list_primary_colours", "text": "The three primary colours are", "category": "list"},
    {"id": "list_seasons", "text": "The four seasons of the year are", "category": "list"},
    {"id": "list_planets", "text": "The planets in our solar system are", "category": "list"},
    {"id": "list_continents", "text": "There are seven continents:", "category": "list"},
    # Negation
    {"id": "neg_paris", "text": "Paris is not the capital of", "category": "negation"},
    {"id": "neg_water_solid", "text": "Water is not in solid form when it", "category": "negation"},
    # Code
    {"id": "code_fib", "text": "def fibonacci(n):", "category": "code"},
    {"id": "code_sql", "text": "SELECT * FROM users WHERE", "category": "code"},
    {"id": "code_main", "text": "if __name__ == '__main__':", "category": "code"},
    {"id": "code_print_hello", "text": "print('Hello,", "category": "code"},
    # Narrative / open-ended
    {"id": "narrative_once", "text": "Once upon a time, in a kingdom far away,", "category": "narrative"},
    {"id": "narrative_storm", "text": "When the storm hit, the village", "category": "narrative"},
    {"id": "narrative_letter", "text": "When she opened the box, she found", "category": "narrative"},
    # Concept thinking
    {"id": "concept_cat", "text": "I am thinking about a cat.", "category": "concept"},
    {"id": "concept_house", "text": "I am picturing a small house with a red roof.", "category": "concept"},
    {"id": "concept_geometry", "text": "Imagine an equilateral triangle inscribed in a circle.", "category": "concept"},
    {"id": "concept_emotion", "text": "I am thinking about deep sadness.", "category": "concept"},
    {"id": "concept_motion", "text": "I am picturing a bird flying across the sky.", "category": "concept"},
    {"id": "concept_face", "text": "I am picturing a smiling face.", "category": "concept"},
    {"id": "concept_diagram", "text": "I'm drawing a flowchart with three boxes connected by arrows.", "category": "concept"},
    # Polysemy
    {"id": "poly_bark", "text": "The bark of the tree was rough, and", "category": "polysemy"},
    {"id": "poly_bank2", "text": "The river bank was muddy after the rain.", "category": "polysemy"},
    {"id": "poly_left", "text": "After she left, the room felt", "category": "polysemy"},
    # Comparative
    {"id": "comp_taller", "text": "An adult elephant is taller than a", "category": "comparative"},
    {"id": "comp_cold", "text": "Antarctica is colder than", "category": "comparative"},
    # Self / meta
    {"id": "meta_iam", "text": "I am a language model trained to", "category": "meta"},
    {"id": "meta_thinking", "text": "Right now I am thinking about", "category": "meta"},
]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--av-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artefacts/per_probe"))
    parser.add_argument("--probe-set", type=Path, default=None,
                        help="Optional JSON file with probe set; if absent, uses built-in DEFAULT_PROBE_SET")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--canvas-size", type=int, default=224)
    parser.add_argument("--no-mp4", action="store_true", help="skip mp4 generation (PNG only)")
    args = parser.parse_args()

    if args.probe_set is not None:
        probes = json.loads(args.probe_set.read_text())
    else:
        probes = DEFAULT_PROBE_SET

    layer_dir = args.out_dir / f"L{args.layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    (layer_dir / "png").mkdir(parents=True, exist_ok=True)
    (layer_dir / "mp4").mkdir(parents=True, exist_ok=True)

    # Save probe set for reproducibility
    with open(layer_dir / "probes.json", "w") as f:
        json.dump(probes, f, indent=2)

    # Load AV from Stage-1 av_ckpt.pt format (Anole-minimal: just new embedding rows)
    print(f"[probe] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id, device="cuda", dtype=torch.bfloat16)
    av.model.eval()

    # Use the target model (Gemma 4 without vocab extension) to extract clean h_ℓ.
    # For the Day-1 path we re-use the AV's backbone for extraction; it shares
    # the original weights (LoRA delta is small).
    from transformers import AutoTokenizer
    target_tok = AutoTokenizer.from_pretrained(args.model_id)
    target_model = av.model  # extracts hidden_states with LoRA still active; fine for Day-1

    for probe in probes:
        text = probe["text"]
        # Extract activation
        enc = target_tok(text, return_tensors="pt", add_special_tokens=True).to(av.device())
        out = target_model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach()

        # Generate strokes
        ids = av.generate_from_activation(
            h, layer_ell=args.layer, alpha=args.alpha,
            max_new_tokens=args.max_tokens, temperature=args.temperature,
        )
        strokes = av.vocab.decode_tokens(ids.tolist())
        png_path = layer_dir / "png" / f"{probe['id']}.png"
        mp4_path = None if args.no_mp4 else layer_dir / "mp4" / f"{probe['id']}.mp4"
        img = stroke_render(
            strokes, canvas_size=args.canvas_size,
            save_animation_path=str(mp4_path) if mp4_path is not None else None,
            fps=24,
        )
        img.save(png_path)
        print(f"[probe] {probe['id']:35s}  strokes={len(strokes):3d}  →  {png_path.name}", flush=True)

    # Generate an HTML index page
    html_path = layer_dir / "index.html"
    with open(html_path, "w") as f:
        f.write(f"""<!doctype html><html><head><meta charset='utf-8'><title>Visual NLA — L{args.layer:02d}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1em; }}
.cell {{ border: 1px solid #ddd; padding: 0.5em; border-radius: 8px; }}
.cell img, .cell video {{ width: 100%; height: auto; background: #fff; border: 1px solid #eee; }}
.prompt {{ font-size: 13px; color: #444; margin-top: 0.5em; }}
.cat {{ display: inline-block; padding: 2px 6px; background: #eef; border-radius: 4px; font-size: 11px; color: #557; }}
</style></head><body>
<h1>Visual NLA — Layer {args.layer}</h1>
<p>{len(probes)} probes. Each shows a PNG of the final drawing (left) and the animated emergence (right).</p>
<div class='grid'>""")
        for probe in probes:
            f.write(f"""<div class='cell'>
<img src='png/{probe['id']}.png' alt='{probe['id']}'>
{'' if args.no_mp4 else f"<video src='mp4/{probe['id']}.mp4' controls muted loop autoplay></video>"}
<div class='prompt'><span class='cat'>{probe['category']}</span> {probe['text']}</div>
</div>""")
        f.write("</div></body></html>")
    print(f"[probe] wrote index: {html_path}", flush=True)


if __name__ == "__main__":
    main()
