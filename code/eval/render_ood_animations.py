"""OOD / abstract / math prompts — render what the model "draws" for never-trained ideas.

Same animation render pipeline but the prompt set is NOVEL: emotions, abstract
concepts, math questions, sentences. No concept-target scoring (the AV has no
"correct" answer for these); just render the best stroke draws.

Output: artefacts/v3/viral/ood/{slug}.mp4 (animation) + {slug}.png (static)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verbalizer.stroke_decoder import StrokeDecoder
from render import render as stroke_render


OOD_PROMPTS = [
    # Famous places / things (not in QuickDraw 44)
    ("eiffel",        "The Eiffel Tower in Paris."),
    ("skyscraper",    "A skyscraper at night."),
    ("guitar",        "I am picturing a guitar."),
    ("robot",         "A friendly robot."),
    ("volcano",       "A volcano erupting."),
    ("crown",         "A king's crown."),

    # Emotions / abstract states
    ("happy",         "The feeling of being happy."),
    ("sad",           "I am feeling sad."),
    ("anger",         "I am feeling angry."),
    ("love",          "The feeling of love."),
    ("loneliness",    "A feeling of deep loneliness."),
    ("nostalgia",     "The feeling of nostalgia."),

    # Sensory / abstract perception
    ("colour_purple", "The colour purple."),
    ("colour_red",    "The colour red."),
    ("warmth",        "The warmth of the sun on your skin."),
    ("silence",       "Total silence."),
    ("rain_sound",    "The sound of rain on a window."),
    ("memory",        "An old memory."),

    # Math / numbers / quantities
    ("infinity",      "Infinity."),
    ("two_plus_two",  "Two plus two equals four."),
    ("triangle",      "A triangle inscribed in a circle."),
    ("pi",            "The number pi."),
    ("zero",          "The concept of zero."),
    ("many",          "A very large number of things."),

    # Time / motion
    ("midnight",      "It is midnight."),
    ("morning",       "Early morning sunrise."),
    ("flying",        "The sensation of flying."),
    ("falling",       "Falling through the air."),

    # Phrases / sentences
    ("once_upon",     "Once upon a time in a kingdom far away."),
    ("capital_japan", "The capital of Japan is Tokyo."),
    ("speed_of_light","The speed of light is constant."),
    ("rainbow",       "A rainbow appears after the rain."),

    # CRAZY ABSTRACT
    ("god",           "God."),
    ("death",         "Death."),
    ("dreams",        "Dreams."),
    ("the_universe",  "The entire universe."),
    ("consciousness", "Consciousness."),
    ("nothingness",   "Pure nothingness."),
    ("forever",       "Forever and ever."),
    ("inside_a_dream","Being inside a dream."),
    ("home",          "Home."),
    ("freedom",       "The feeling of freedom."),
    ("fear",          "Primal fear."),
    ("hope",          "Hope."),

    # Deep emotional / philosophical
    ("childhood",     "A memory from childhood."),
    ("first_love",    "Your first love."),
    ("grief",         "Profound grief."),
    ("longing",       "Quiet longing."),
    ("regret",        "Deep regret."),
    ("wonder",        "A sense of wonder."),

    # Pop culture / specific names
    ("paris",         "Paris."),
    ("tokyo",         "Tokyo."),
    ("the_beatles",   "The Beatles."),
    ("shakespeare",   "Shakespeare."),
    ("einstein",      "Einstein."),
    ("napoleon",      "Napoleon Bonaparte."),

    # Counterfactuals
    ("not_a_cat",     "Not a cat. Definitely not a cat."),
    ("draw_nothing",  "Draw nothing at all."),
    ("everything",    "Draw everything."),
    ("the_question",  "What is the question?"),
    ("the_answer",    "42."),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--layer-tag", default=None)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--max-tokens", type=int, default=240)
    p.add_argument("--display-scale", type=float, default=4.0)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--out-dir", type=Path, default=Path("artefacts/v3/viral/ood"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    tag = args.layer_tag or f"L{args.layer:02d}"
    out_dir = args.out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ood] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_ckpt(args.av_ckpt, model_id=args.model_id)
    av.model.eval()
    device = av.device()

    results = []
    for slug, prompt in OOD_PROMPTS:
        t0 = time.time()
        enc = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            out = av.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1, :].detach()
        ids_list = av.generate_from_activation_batched(
            h, layer_ell=args.layer, n_samples=args.n_samples,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
        )
        # Pick the longest stroke sequence (heuristic — "most-drawn")
        best = None
        for ids in ids_list:
            strokes, _ = av.vocab.decode_tokens_with_stats(ids.tolist())
            if len(strokes) < 8:
                continue
            if best is None or len(strokes) > len(best["strokes"]):
                best = {"strokes": strokes, "n_strokes": len(strokes)}
        if best is None:
            print(f"[ood] {slug:18s}  all degenerate", flush=True)
            continue
        mp4_path = out_dir / f"{slug}.mp4"
        png_path = out_dir / f"{slug}.png"
        stroke_render(best["strokes"], display_scale=args.display_scale,
                      save_animation_path=str(mp4_path), fps=args.fps)
        final_png = stroke_render(best["strokes"], display_scale=args.display_scale).convert("RGB")
        final_png.save(png_path)
        results.append({"slug": slug, "prompt": prompt, "n_strokes": best["n_strokes"]})
        print(f"[ood] {tag} {slug:18s} ({prompt[:40]!r:40s})  n_strokes={best['n_strokes']:3d}  ({time.time()-t0:.1f}s)", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"[ood] {tag} DONE → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
