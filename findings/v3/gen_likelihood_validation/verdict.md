# Phase 0 — generation-likelihood validation

Goal: verify that frozen Qwen 3.5-4B assigns higher log P(correct_concept | image, prompt) than for wrong concepts, on REAL canonical drawings. This is the foundational signal v3 trains on.

## Setup

- Loader: `AutoModelForImageTextToText` (the correct one for Qwen3-VL; `AutoModelForCausalLM` silently drops pixel_values).
- Prompt: chat-template-wrapped `<image>What is this a drawing of?<|im_end|>\n<|im_start|>assistant\nA drawing of a `
- For each (concept, canonical drawing), score every other concept's log-prob as the continuation. Pairwise 30×30 matrix.

## Results

| metric                 | value                                          |
|-----------------------:|:----------------------------------------------:|
| diag mean (correct)    | -1.373 ± 2.968          |
| off-diag mean (wrong)  | -10.911                              |
| discriminability       | **+9.5379** ± 3.724      |
| top-1 retrieval        | **90.0%**  (chance 3.3%)     |
| margin (correct − wrong) | +4.883 ± 3.894     |

## Verdict

STRONG signal — proceed to Phase 1 (training)
