#!/usr/bin/env bash
# Render animations for ALL 4 priority layers sequentially.
set -e
cd "$(dirname "$0")/../.."

PY=${PY:-/home/theod/Aryaa/venv/bin/python3}
export PYTHONPATH=code
export HF_HUB_DISABLE_PROGRESS_BARS=1

declare -A LAYERS
LAYERS[3]="checkpoints/overnight/L3/final"
LAYERS[10]="checkpoints/overnight/L10_filtered/final"
LAYERS[24]="checkpoints/overnight/L24/final"
LAYERS[29]="checkpoints/v2_0/L29/final"

for L in 3 10 24 29; do
    ck=${LAYERS[$L]}
    tag="L$(printf '%02d' $L)"
    echo "=== [$(date)] Animations for ${tag} from ${ck} ==="
    $PY code/eval/render_animations.py \
        --av-ckpt "$ck" --layer "$L" --layer-tag "$tag" \
        --n-samples 12 --display-scale 4.0 --fps 24 \
        --out-dir artefacts/v3/viral/anim \
        2>&1
done

echo "=== [$(date)] all animations DONE ==="
