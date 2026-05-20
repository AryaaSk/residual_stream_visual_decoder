# Morning — v1.1 shipped (honest negative result)

Everything is done. Tag `v1.1` is pushed; GitHub release is up:
https://github.com/AryaaSk/residual_stream_visual_decoder/releases/tag/v1.1

## What you'll see when you open the repo

- `README.md` — re-titled to honest "An attempt to draw what Gemma 4 is thinking" with a v1.1 status banner
- `WRITEUP.md` — new TL;DR table at top (cosine 0.51 L12 / 0.70 L24; FVE negative; visuals not recognisable), §4.6 auto-filled with full numbers, §10 known issues, §11 v1.2 candidates
- `artefacts/v1_1/demo.mp4` — 12-clip hype reel (hero inject_demo + cross-layer + per-token trajectories)
- `findings/v1_1/SUMMARY.json` — headline numbers in JSON
- `findings/v1_1/iter_plot_L{12,24}.png` — per-iter FVE / cosine / loss plots
- `findings/v1_1/inject_demo_L{12,24}/` — 26 prompts × 2 layers, hero prompts also at α ∈ {0.3, 0.5, 0.7, 1.0}
- `artefacts/v1_1/{trajectory_L12,cross_layer}/` — per-token + cross-layer MP4s

## The honest verdict

**The architecture works end-to-end.** Custom LoRA on `Gemma4ClippableLinear`, iterative joint AR/AV training, activation injection via embedding hook, 26-prompt eval with α-sweep, per-token + cross-layer trajectory MP4s — all run cleanly.

**The faithfulness signal is real but weak.** Held-out cosine 0.51 (L12) / 0.70 (L24) means the AR's reconstruction direction tracks the source prompt. You can tell `dog` and `eiffel` apart by stroke layout. FVE stays negative because the AR over-predicts magnitude (MSE loss has no variance-inflation penalty).

**The drawings are not recognisable.** I looked at the L24 hero outputs (dog, cat, eiffel, triangle, smile_face, capital_france) at α=0.3, 0.5, 0.7, 1.0. They are abstract structures — per-prompt different, but none of them looks like the thing. This is not a viral demo as-is.

## What I'd do for v1.2 (in §11 of WRITEUP)

1. **Cosine-based AR loss** (the metric that's actually working — optimise for it directly)
2. **InfoNCE discriminative AR** (closer to true NLA framing)
3. **Weaker AV KL anchor** (β = 0.005 not 0.05 — current anchor keeps stroke distribution too close to the narrow QuickDraw prior)
4. **Late layer L32/L34** probe
5. **Architectural fix for token_trajectory.py** — currently uses AV (extended vocab) as the next-token generator, so stroke tokens leak into the caption ("I am thinking about a dog. Specifically, a\<DX_092\>\<PEN_UP\>..."). The fix is a separate clean Gemma 4 for text generation + activation extraction.
6. **Larger AR LoRA (r=32)** if iter-buffer is the bottleneck.

Each is a single-knob experiment. #1 is cheapest and most-aligned-with-what-works; I'd start there.

## Decision points

- **Ship to Twitter as-is?** I'd hesitate. The result is honest but the visuals don't carry the "look what it can do" hook. Interpretability twitter does love negative-result writeups, but they usually pair the disappointment with one striking image. We don't have one.
- **Run v1.2 first?** Even just experiment #1 (cosine AR loss) is a 6-hour retrain. If you do, the H200 is still ours.
- **Keep the v1.1 release public?** Yes — it's an honest record of what 24 GPU-hours bought, the writeup is comprehensive, the code is clean, others can build on it.

## What's running / not running

- All training tmux sessions on H200 have ended naturally
- `autofinish` tmux on H200 completed at 05:19 UTC (status logged)
- Mac-side `autopull` exited cleanly at 05:28 after fetching SUMMARY.json and running fill_writeup.py
- Monitor `b35gu0nlb` timed out (expected — pipeline finished within the hour)
- No background processes to clean up
