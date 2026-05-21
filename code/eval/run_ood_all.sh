#!/usr/bin/env bash
# OOD animations across all best layers (L10_filt, L24, L29)
set -e
cd "$(dirname "$0")/../.."

PY=${PY:-/home/theod/Aryaa/venv/bin/python3}
export PYTHONPATH=code
export HF_HUB_DISABLE_PROGRESS_BARS=1

declare -A LAYERS
LAYERS[10]="checkpoints/overnight/L10_filtered/final"
LAYERS[24]="checkpoints/overnight/L24/final"
LAYERS[29]="checkpoints/v2_0/L29/final"

for L in 10 24 29; do
    ck=${LAYERS[$L]}
    tag="L$(printf '%02d' $L)"
    echo "=== [$(date)] OOD animations for ${tag} from ${ck} ==="
    $PY code/eval/render_ood_animations.py \
        --av-ckpt "$ck" --layer "$L" --layer-tag "$tag" \
        --n-samples 8 --display-scale 4.0 --fps 24 \
        --out-dir artefacts/v3/viral/ood \
        2>&1
done

echo "=== [$(date)] all OOD animations DONE ==="
