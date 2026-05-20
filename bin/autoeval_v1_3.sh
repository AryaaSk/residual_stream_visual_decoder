#!/usr/bin/env bash
# autoeval_v1_3.sh — autonomous eval loop for v1.3 training.
#
# Watches checkpoints/v1_3/L{12,24}/step_* directories. As each new step_N
# appears, runs fast best-of-N on it. Quality progression is logged.
# When training completes (DONE markers in s13_L*.log), runs the final
# best-of-N + builds hype reel.
#
# Designed to run inside a detached tmux on the H200.

set +e

cd /home/theod/Aryaa/rsvd
LOG=runs/autoeval_v1_3.log
VENV=/home/theod/Aryaa/venv/bin/python3
export PYTHONPATH=code

log() { echo "[autoeval $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
heading() { echo "" | tee -a "$LOG"; log "=========== $* ==========="; }

heading "STARTUP"

# Track which checkpoints we've already evaluated
declare -A SEEN
SEEN["L12_done"]=0
SEEN["L24_done"]=0

run_eval_on_ckpt() {
    local LAYER=$1
    local CKPT_DIR=$2     # full path
    local TAG=$3          # e.g. "step005000" or "final"
    local GPU=$4
    local OUT="findings/v1_3/fastN_L${LAYER}_${TAG}"
    if [[ -d "$OUT" ]]; then
        log "skip $OUT (exists)"
        return
    fi
    heading "fastN L=${LAYER} ckpt=${CKPT_DIR} → $OUT (GPU $GPU)"
    CUDA_VISIBLE_DEVICES=$GPU $VENV code/eval/fast_best_of_n.py \
        --av-ckpt "$CKPT_DIR" --layer "$LAYER" \
        --n-samples 16 --pick-k 3 --temperature 0.8 --top-k 20 \
        --out-dir "$OUT" 2>&1 | tee -a "$LOG"
    log "fastN L=${LAYER} ${TAG} done"
}

# Phase A — poll for new checkpoints AND completion
heading "POLLING for v1.3 checkpoints + DONE markers"
while true; do
    # Check for completion markers
    L12_DONE=$(grep -c "^\[s15\] DONE → " runs/s13_L12.log 2>/dev/null)
    L24_DONE=$(grep -c "^\[s15\] DONE → " runs/s13_L24.log 2>/dev/null)

    # Eval new step ckpts on the EVAL GPUs (0 for L12, 1 for L24).
    # The training GPUs are 7 (L12 training) and 6 (L24 training); we use
    # the spare ones to avoid contention.
    for L in 12 24; do
        if [[ "$L" == "12" ]]; then EVAL_GPU=0; else EVAL_GPU=1; fi
        # Look for step_NNNNNN dirs
        for CKPT in $(ls -1d checkpoints/v1_3/L${L}/step_* 2>/dev/null); do
            STEP=$(basename "$CKPT" | sed 's/step_//')
            KEY="L${L}_${STEP}"
            if [[ -z "${SEEN[$KEY]:-}" ]]; then
                SEEN[$KEY]=1
                run_eval_on_ckpt "$L" "$CKPT" "step${STEP}" "$EVAL_GPU"
            fi
        done
    done

    # If training has finished and we haven't run on `final`, do so now
    for L in 12 24; do
        if [[ "$L" == "12" ]]; then DONE_FLAG=$L12_DONE; EVAL_GPU=0; else DONE_FLAG=$L24_DONE; EVAL_GPU=1; fi
        if [[ "$DONE_FLAG" -ge 1 && "${SEEN[L${L}_final]:-0}" == "0" ]]; then
            SEEN[L${L}_final]=1
            FINAL_DIR="checkpoints/v1_3/L${L}/final"
            if [[ -d "$FINAL_DIR" ]]; then
                run_eval_on_ckpt "$L" "$FINAL_DIR" "final" "$EVAL_GPU"
            fi
        fi
    done

    # Exit when both layers are done AND we've eval'd both finals
    if [[ "$L12_DONE" -ge 1 && "$L24_DONE" -ge 1 && "${SEEN[L12_final]}" == "1" && "${SEEN[L24_final]}" == "1" ]]; then
        break
    fi

    sleep 120
done

heading "AUTOEVAL COMPLETE. Outputs under findings/v1_3/fastN_L*_*"
log "now you can build the hype reel from findings/v1_3/fastN_L12_final/ and L24_final/"
