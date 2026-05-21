#!/usr/bin/env bash
# Run v3_gen_likelihood_eval.py on EACH of the per-layer AV checkpoints.
# Outputs go to findings/v3/per_layer_eval/L{NN}/
set -e
cd "$(dirname "$0")/../.."

PY=${PY:-/home/theod/Aryaa/venv/bin/python3}
export PYTHONPATH=code
export HF_HUB_DISABLE_PROGRESS_BARS=1

# CKPTS: layer → checkpoint dir
declare -A CKPTS
CKPTS[3]="checkpoints/overnight/L3/final"
CKPTS[10]="checkpoints/v2_0/L10/final"
CKPTS[20]="checkpoints/overnight/L20/final"
CKPTS[29]="checkpoints/v2_0/L29/final"

for L in 3 10 20 29; do
    ck="${CKPTS[$L]}"
    out="findings/v3/per_layer_eval/L${L}"
    echo "=== [$(date)] Eval L${L} from ${ck} ==="
    if [ ! -f "${ck}/av_ckpt.pt" ]; then
        echo "  SKIP: ${ck}/av_ckpt.pt not found"
        continue
    fi
    mkdir -p "$out"
    $PY code/eval/v3_gen_likelihood_eval.py \
        --av-ckpt "${ck}" --layer "$L" \
        --n-samples 8 --temperature 0.85 --top-k 25 \
        --max-tokens 200 --display-scale 1.0 \
        --out-dir "$out" \
        2>&1 | tee "$out/eval.log"
done

echo "=== [$(date)] all per-layer evals DONE ==="
