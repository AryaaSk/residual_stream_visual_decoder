#!/usr/bin/env bash
# CLIP-ranked autoeval for v1.3 training. As each step_NNNNNN ckpt appears,
# runs clip_ranker.py on it. The CLIP-ranked outputs are much higher quality
# than heuristic-ranked ones (CLIP measures actual semantic resemblance).

set +e
cd /home/theod/Aryaa/rsvd
LOG=runs/autoeval_v2_0_clip.log
VENV=/home/theod/Aryaa/venv/bin/python3
export PYTHONPATH=code

log() { echo "[ae-clip $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
heading() { echo "" | tee -a "$LOG"; log "=========== $* ==========="; }

heading "STARTUP (CLIP-ranked autoeval)"

declare -A SEEN
SEEN["L10_done"]=0
SEEN["L29_done"]=0

run_clip_on_ckpt() {
    local LAYER=$1
    local CKPT_DIR=$2
    local TAG=$3
    local GPU=$4
    local OUT="findings/v2_0/clip_L${LAYER}_${TAG}"
    if [[ -d "$OUT" ]]; then
        log "skip $OUT (exists)"
        return
    fi
    heading "clip_ranker L=${LAYER} ckpt=${CKPT_DIR} → $OUT (GPU $GPU)"
    CUDA_VISIBLE_DEVICES=$GPU $VENV code/eval/clip_ranker.py \
        --av-ckpt "$CKPT_DIR" --model-id Qwen/Qwen3.5-4B --layer "$LAYER" \
        --n-samples 32 --pick-k 3 --temperature 0.85 --top-k 25 \
        --out-dir "$OUT" 2>&1 | tee -a "$LOG"
    log "clip L=${LAYER} ${TAG} done"
}

heading "POLLING for v1.3 checkpoints"
while true; do
    L10_DONE=$(grep -c "^\[s15\] DONE → " runs/s20_L10.log 2>/dev/null)
    L29_DONE=$(grep -c "^\[s15\] DONE → " runs/s20_L29.log 2>/dev/null)

    for L in 10 29; do
        if [[ "$L" == "12" ]]; then EVAL_GPU=2; else EVAL_GPU=3; fi
        for CKPT in $(ls -1d checkpoints/v2_0/L${L}/step_* 2>/dev/null); do
            STEP=$(basename "$CKPT" | sed 's/step_//')
            KEY="L${L}_${STEP}"
            if [[ -z "${SEEN[$KEY]:-}" ]]; then
                SEEN[$KEY]=1
                run_clip_on_ckpt "$L" "$CKPT" "step${STEP}" "$EVAL_GPU"
            fi
        done
    done

    for L in 10 29; do
        if [[ "$L" == "10" ]]; then DONE_FLAG=$L10_DONE; EVAL_GPU=2; else DONE_FLAG=$L29_DONE; EVAL_GPU=3; fi
        if [[ "$DONE_FLAG" -ge 1 && "${SEEN[L${L}_final]:-0}" == "0" ]]; then
            SEEN[L${L}_final]=1
            FINAL_DIR="checkpoints/v2_0/L${L}/final"
            if [[ -d "$FINAL_DIR" ]]; then
                run_clip_on_ckpt "$L" "$FINAL_DIR" "final" "$EVAL_GPU"
            fi
        fi
    done

    if [[ "$L10_DONE" -ge 1 && "$L29_DONE" -ge 1 && "${SEEN[L10_final]}" == "1" && "${SEEN[L29_final]}" == "1" ]]; then
        break
    fi

    sleep 120
done

heading "CLIP AUTOEVAL COMPLETE. Outputs under findings/v2_0/clip_L*_*"
