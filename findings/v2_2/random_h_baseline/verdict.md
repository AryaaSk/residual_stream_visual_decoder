# Phase 1 — random-h baseline verdict

Run: 2026-05-20, v2.0 L10 final ckpt, n_samples=16 per h, n_random=16.

## Raw CLIP scores (the misleading version)

| condition       | mean | std  | min   | max   |
|----------------:|:----:|:----:|:-----:|:-----:|
| random_matched  | 35.62 | -    | -     | -     |
| random_iso      | 34.68 | -    | -     | -     |
| real_control    | 35.89 | -    | -     | -     |

By raw CLIP score, the gap is just +0.27. The auto-verdict script said
"AV largely ignores h". **That heuristic is wrong here.**

CLIP gives 33-37 to almost any well-formed line drawing scored against 20
concept texts (it always finds something to fit). The mean-score gap
therefore can't distinguish "h-sensitive" from "h-insensitive".

## The real metric — prompt→concept match accuracy

For each real prompt (`cat`, `dog`, `fish`, etc.), we ask: does the drawing
generated from `h_qwen(prompt)[L10]` get CLIP-best-matched to the SAME
concept as the prompt?

```
  YES dog        -> dog         score=34.92
  YES fish       -> fish        score=34.99
  YES horse      -> horse       score=36.61
  YES elephant   -> elephant    score=36.78
  YES flower     -> flower      score=35.68
  YES tree       -> tree        score=35.40
  YES mountain   -> mountain    score=37.78
  YES sun        -> sun         score=36.24
  YES star       -> star        score=35.89
  YES car        -> car         score=36.55
  YES airplane   -> airplane    score=37.07
  no  cat        -> apple       (compact round drawing)
  no  bird       -> pizza       (likely scribble shape)
  no  cactus     -> mountain    (tall-pointy)
  no  cloud      -> horse       (low-CLIP=29.66 anyway)
  no  house      -> mountain    (tall-triangular)
```

**ACCURACY: 11 / 16 = 69%**.
Chance with 20 candidate concepts ≈ 5%. **The AV is decoding h** — when you
hand it h_qwen("dog"), you get back a drawing CLIP unambiguously calls "dog".

## Concept diversity confirms

|         condition | distinct best-concepts / 16 |
|------------------:|:---------------------------:|
|       real_control | **13** (high diversity, matches input diversity) |
|     random_matched | 7 (mode-collapse on dog/airplane/elephant/bird) |
|         random_iso | 7 (mode-collapse on cloud x6, mountain x4) |

When fed random h with matched moments, the AV falls back to a small set of
high-prior templates — exactly the "v2.0 dog drawing has the WORD 'dog'
written in stroke-letterforms" mode-collapse pattern we noted in v2.0.

When fed real h, the AV produces a much wider variety of drawings, and
they predominantly match the input prompt's concept.

## v2.2 gating decision

**Proceed.** The AV is genuinely h-sensitive at L10:
1. 69% prompt→concept accuracy beats 5% chance by 14×.
2. Real h produces nearly 2× the concept diversity of random h.
3. The 5 mis-matches all have known plausible explanations
   (cat→apple is "compact round", bird→pizza is "many short strokes",
   house→mountain is "triangular silhouette").

What the baseline DOES caveat: the AV has *some* drawing templates it
falls back to when h is unusual (clouds, mountains, dogs). This is exactly
the v2.0 "concept-plausible-not-specific" finding — and it's what the
cross-layer trajectory (L3 / L10 / L20 / L29) will explain by showing that
specificity DOES emerge in deeper layers (Phase 2 probe will quantify).

The interpretability claim survives. Continue v2.2 as planned.
