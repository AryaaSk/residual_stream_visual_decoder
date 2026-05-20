#!/usr/bin/env bash
# CLIP-ranked autoeval for v1.3 training. As each step_NNNNNN ckpt appears,
# runs clip_ranker.py on it. The CLIP-ranked outputs are much higher quality
# than heuristic-ranked ones (CLIP measures actual semantic resemblance).

set +e
cd /home/theod/Aryaa/rsvd
LOG=runs/autoeval_v1_3_clip.log
VENV=/home/theod/Aryaa/venv/bin/python3
export PYTHONPATH=code

log() { echo "[ae-clip $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
heading() { echo "" | tee -a "$LOG"; log "=========== $* ==========="; }

heading "STARTUP (CLIP-ranked autoeval)"

declare -A SEEN
SEEN["L12_done"]=0
SEEN["L24_done"]=0

run_clip_on_ckpt() {
    local LAYER=$1
    local CKPT_DIR=$2
    local TAG=$3
    local GPU=$4
    local OUT="findings/v1_3/clip_L${LAYER}_${TAG}"
    if [[ -d "$OUT" ]]; then
        log "skip $OUT (exists)"
        return
    fi
    heading "clip_ranker L=${LAYER} ckpt=${CKPT_DIR} → $OUT (GPU $GPU)"
    CUDA_VISIBLE_DEVICES=$GPU $VENV code/eval/clip_ranker.py \
        --av-ckpt "$CKPT_DIR" --layer "$LAYER" \
        --n-samples 32 --pick-k 3 --temperature 0.85 --top-k 25 \
        --out-dir "$OUT" 2>&1 | tee -a "$LOG"
    log "clip L=${LAYER} ${TAG} done"
}

heading "POLLING for v1.3 checkpoints"
while true; do
    L12_DONE=$(grep -c "^\[s15\] DONE → " runs/s13_L12.log 2>/dev/null)
    L24_DONE=$(grep -c "^\[s15\] DONE → " runs/s13_L24.log 2>/dev/null)

    for L in 12 24; do
        if [[ "$L" == "12" ]]; then EVAL_GPU=2; else EVAL_GPU=3; fi
        for CKPT in $(ls -1d checkpoints/v1_3/L${L}/step_* 2>/dev/null); do
            STEP=$(basename "$CKPT" | sed 's/step_//')
            KEY="L${L}_${STEP}"
            if [[ -z "${SEEN[$KEY]:-}" ]]; then
                SEEN[$KEY]=1
                run_clip_on_ckpt "$L" "$CKPT" "step${STEP}" "$EVAL_GPU"
            fi
        done
    done

    for L in 12 24; do
        if [[ "$L" == "12" ]]; then DONE_FLAG=$L12_DONE; EVAL_GPU=2; else DONE_FLAG=$L24_DONE; EVAL_GPU=3; fi
        if [[ "$DONE_FLAG" -ge 1 && "${SEEN[L${L}_final]:-0}" == "0" ]]; then
            SEEN[L${L}_final]=1
            FINAL_DIR="checkpoints/v1_3/L${L}/final"
            if [[ -d "$FINAL_DIR" ]]; then
                run_clip_on_ckpt "$L" "$FINAL_DIR" "final" "$EVAL_GPU"
            fi
        fi
    done

    if [[ "$L12_DONE" -ge 1 && "$L24_DONE" -ge 1 && "${SEEN[L12_final]}" == "1" && "${SEEN[L24_final]}" == "1" ]]; then
        break
    fi

    sleep 120
done

heading "CLIP AUTOEVAL COMPLETE. Outputs under findings/v1_3/clip_L*_*"
