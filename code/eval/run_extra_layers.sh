#!/usr/bin/env bash
# Hero + OOD animations at additional layers (L15, L27)
set -e
cd "$(dirname "$0")/../.."

PY=${PY:-/home/theod/Aryaa/venv/bin/python3}
export PYTHONPATH=code
export HF_HUB_DISABLE_PROGRESS_BARS=1

declare -A LAYERS
LAYERS[15]="checkpoints/overnight/L15_filtered/final"
LAYERS[20]="checkpoints/overnight/L20_v2/final"
LAYERS[27]="checkpoints/overnight/L27/final"

for L in 15 20 27; do
    ck=${LAYERS[$L]}
    tag="L$(printf '%02d' $L)"
    echo "=== [$(date)] Hero anims for ${tag} from ${ck} ==="
    $PY code/eval/render_animations.py \
        --av-ckpt "$ck" --layer "$L" --layer-tag "$tag" \
        --n-samples 12 --display-scale 4.0 --fps 24 \
        --out-dir artefacts/v3/viral/anim \
        2>&1
done

# Also OOD on L27 (the most novel, between L24 and L29)
echo "=== [$(date)] OOD anims L27 ==="
$PY code/eval/render_ood_animations.py \
    --av-ckpt checkpoints/overnight/L27/final --layer 27 --layer-tag L27 \
    --n-samples 8 --display-scale 4.0 --fps 24 \
    --out-dir artefacts/v3/viral/ood \
    2>&1

echo "=== [$(date)] all extra layer anims DONE ==="
