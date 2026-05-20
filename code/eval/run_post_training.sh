#!/usr/bin/env bash
# Post-training pipeline for v2.2: runs all six demos sequentially.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH=code bash code/eval/run_post_training.sh
#
# Most demos auto-pick their GPU via CUDA_VISIBLE_DEVICES. Cross-layer is the
# longest and is the main GPU consumer (loads 4 AVs sequentially).
set -e
cd "$(dirname "$0")/../.."

PY=${PY:-/home/theod/Aryaa/venv/bin/python3}
export PYTHONPATH=code
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p runs artefacts/v2_2 findings/v2_2

echo "=== [$(date)] starting post-training pipeline ==="

echo "=== [$(date)] [1/5] linear probe (CPU-heavy, GPU light) ==="
$PY code/eval/linear_probe.py --layers 3 10 20 29 --out-dir findings/v2_2 \
    2>&1 | tee runs/probe.log

echo "=== [$(date)] [2/5] cross-layer trajectory (8 prompts × 4 layers) ==="
$PY code/eval/cross_layer_video.py \
    --ckpts-root checkpoints/v2_2 --layers 3 10 20 29 --n-samples 32 \
    --prompts-jsonl data/v2_2_prompts.jsonl \
    --out-dir artefacts/v2_2/cross_layer \
    2>&1 | tee runs/cross_layer.log

echo "=== [$(date)] [3/5] interpolation morphs (5 pairs × 15 steps × 16 samples) ==="
$PY code/eval/interpolate_h.py \
    --av-ckpt checkpoints/v2_2/L10/final --layer 10 \
    --n-steps 15 --n-samples 16 \
    --out-dir artefacts/v2_2/morph \
    2>&1 | tee runs/morph.log

echo "=== [$(date)] [4/5] per-token trajectory (6 prompts) ==="
$PY code/eval/token_trajectory.py \
    --av-ckpt checkpoints/v2_2/L10/final --layer 10 \
    --max-gen-tokens 15 \
    --out-dir artefacts/v2_2/per_token \
    2>&1 | tee runs/per_token.log

echo "=== [$(date)] [5/5] OOD demo (8 prompts) ==="
$PY code/eval/clip_ranker_ood.py \
    --av-ckpt checkpoints/v2_2/L10/final --layer 10 \
    --n-samples 16 \
    --out-dir findings/v2_2/ood \
    2>&1 | tee runs/ood.log

echo "=== [$(date)] post-training pipeline DONE ==="
