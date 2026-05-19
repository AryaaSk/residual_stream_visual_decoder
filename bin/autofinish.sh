#!/usr/bin/env bash
# autofinish.sh — autonomous post-training pipeline for v1.1.
#
# Designed to run inside a detached tmux on the H200 alongside the training
# jobs. Polls the training logs every 60s; when both have written "DONE",
# kicks off the full eval + artefact pipeline (inject_demo, FVE measurements,
# cross-layer trajectory, hype reel). Everything lands on the remote disk
# under ~/Aryaa/rsvd/findings/v1_1/ and ~/Aryaa/rsvd/checkpoints/v1_1/ so
# the user can pull when they wake up.
#
# Run on the H200 (uses /home/theod/Aryaa/rsvd as cwd):
#     tmux new-session -d -s autofinish 'bash /home/theod/Aryaa/rsvd/bin/autofinish.sh'
#
# Designed to be IDEMPOTENT: if it crashes or is restarted, it skips work that
# already completed.

set +e  # don't bail on individual step failure — log and continue

cd /home/theod/Aryaa/rsvd
LOG=runs/autofinish.log
VENV=/home/theod/Aryaa/venv/bin/python3
export PYTHONPATH=code
export CUDA_VISIBLE_DEVICES=6,7  # use whichever's free; eval kernels work on either

log() { echo "[autofinish $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
heading() { echo "" | tee -a "$LOG"; log "============================ $* ============================"; }

heading "STARTUP"
log "PID=$$ PYTHONPATH=$PYTHONPATH"
log "waiting for both iter11L12.log and iter11L24.log to contain 'DONE' ..."

# Phase 0 — wait for both training jobs to complete (or error)
while true; do
    L12_DONE=$(grep -cE "^\[iter\] DONE → " runs/iter11L12.log 2>/dev/null)
    L24_DONE=$(grep -cE "^\[iter\] DONE → " runs/iter11L24.log 2>/dev/null)
    L12_ERR=$(grep -cE "^Traceback|^Error:|Killed|OOM" runs/iter11L12.log 2>/dev/null)
    L24_ERR=$(grep -cE "^Traceback|^Error:|Killed|OOM" runs/iter11L24.log 2>/dev/null)
    if [[ "$L12_DONE" -ge 1 && "$L24_DONE" -ge 1 ]]; then
        log "both training jobs complete"
        break
    fi
    if [[ "$L12_ERR" -ge 1 && "$L24_ERR" -ge 1 ]]; then
        log "WARN: both jobs have error markers; proceeding with whatever checkpoints exist"
        break
    fi
    sleep 60
done

# Phase 1 — inject_demo for each layer with the BEST iter checkpoint
# Includes alpha sweep on the hero subset so we have material to choose from
# even if alpha=0.5 produces weak visuals at this layer.
heading "PHASE 1: inject_demo per layer (26 prompts main pass + alpha sweep on 6 hero)"
for LAYER_DIR in checkpoints/v1_1/L12 checkpoints/v1_1/L24; do
    if [[ ! -d "$LAYER_DIR/final" ]]; then
        log "WARN: $LAYER_DIR/final missing, skipping inject_demo for that layer"
        continue
    fi
    LAYER=$(basename "$LAYER_DIR" | sed 's/^L//')
    OUT="findings/v1_1/inject_demo_L${LAYER}"
    log "inject_demo L=$LAYER → $OUT  (main α=0.5 + hero sweep α∈{0.3,0.7,1.0})"
    $VENV code/eval/inject_demo.py \
        --av-ckpt "$LAYER_DIR/final" --layer "$LAYER" \
        --alpha 0.5 --mp4 --out-dir "$OUT" \
        --alpha-sweep 0.3 0.7 1.0 \
        2>&1 | tee -a "$LOG"
done

# Phase 2 — FVE measurement (held-out probes)
heading "PHASE 2: FVE per layer (held-out)"
for LAYER in 12 24; do
    CKPT="checkpoints/v1_1/L${LAYER}/final"
    OUT="findings/v1_1/fve_heldout_L${LAYER}.json"
    if [[ ! -d "$CKPT" ]]; then
        log "WARN: $CKPT missing, skipping FVE for L$LAYER"
        continue
    fi
    log "measure_fve L=$LAYER (held-out) → $OUT"
    $VENV code/eval/measure_fve.py \
        --av-ckpt "$CKPT" --ar-ckpt "$CKPT" --layer "$LAYER" \
        --alpha 0.5 --n-samples-per-probe 2 \
        --out "$OUT" \
        2>&1 | tee -a "$LOG"
done

# Phase 3 — FVE measurement (training-distribution probes) — sanity diagnostic
heading "PHASE 3: FVE per layer (training distribution)"
for LAYER in 12 24; do
    CKPT="checkpoints/v1_1/L${LAYER}/final"
    OUT="findings/v1_1/fve_train_dist_L${LAYER}.json"
    if [[ ! -d "$CKPT" ]]; then
        log "WARN: $CKPT missing, skipping train-dist FVE for L$LAYER"
        continue
    fi
    log "measure_fve_train_dist L=$LAYER → $OUT"
    $VENV code/eval/measure_fve_train_dist.py \
        --av-ckpt "$CKPT" --ar-ckpt "$CKPT" --layer "$LAYER" \
        --alpha 0.5 --n-samples-per-probe 2 \
        --out "$OUT" \
        2>&1 | tee -a "$LOG"
done

# Phase 4 — per-iteration FVE plots
heading "PHASE 4: per-iteration FVE plots"
for LAYER in 12 24; do
    LOG_PATH="checkpoints/v1_1/L${LAYER}/iter_log.jsonl"
    OUT="findings/v1_1/iter_plot_L${LAYER}.png"
    if [[ ! -f "$LOG_PATH" ]]; then
        log "WARN: $LOG_PATH missing, skipping plot"
        continue
    fi
    $VENV code/eval/plot_iter_log.py --log "$LOG_PATH" --out "$OUT" --title "v1.1 L$LAYER" \
        2>&1 | tee -a "$LOG"
done

# Phase 5 — cross-layer trajectory (uses both L12 and L24)
heading "PHASE 5: cross-layer trajectory"
$VENV code/eval/cross_layer_trajectory.py \
    --ckpts-root checkpoints/v1_1 --layers 12 24 \
    --out-dir artefacts/v1_1/cross_layer \
    --alpha 0.5 \
    2>&1 | tee -a "$LOG"

# Phase 6 — per-token trajectory (uses L12)
heading "PHASE 6: per-token trajectory (L12)"
if [[ -d "checkpoints/v1_1/L12/final" ]]; then
    $VENV code/eval/token_trajectory.py \
        --av-ckpt checkpoints/v1_1/L12/final --layer 12 --alpha 0.5 \
        --out-dir artefacts/v1_1/trajectory_L12 \
        --max-gen-tokens 12 \
        2>&1 | tee -a "$LOG"
else
    log "skip token_trajectory (L12 final missing)"
fi

# Phase 7 — hype reel: hero inject_demo clips + cross-layer + per-token
heading "PHASE 7: 60-sec hype reel"
mkdir -p artefacts/v1_1/reel_src
# Hero clips from each layer (alpha=0.5 main pass, not the _a0.30 sweep variants)
for SLUG in dog cat eiffel capital_france triangle smile_face; do
    for LAYER in 12 24; do
        SRC="findings/v1_1/inject_demo_L${LAYER}/${SLUG}_4x.mp4"
        DST="artefacts/v1_1/reel_src/hero_L${LAYER}_${SLUG}.mp4"
        [[ -f "$SRC" ]] && cp "$SRC" "$DST"
    done
done
cp artefacts/v1_1/cross_layer/*.mp4 artefacts/v1_1/reel_src/ 2>/dev/null || true
cp artefacts/v1_1/trajectory_L12/*.mp4 artefacts/v1_1/reel_src/ 2>/dev/null || true
log "reel_src contents:"
ls artefacts/v1_1/reel_src/ | head -30 | tee -a "$LOG"
$VENV code/eval/make_hype_reel.py \
    --in-dir artefacts/v1_1/reel_src --out artefacts/v1_1/demo.mp4 \
    --max-clips 12 \
    2>&1 | tee -a "$LOG"

# Phase 8 — write a top-level summary JSON for easy reading
heading "PHASE 8: summary JSON"
$VENV - <<'EOF' 2>&1 | tee -a "$LOG"
import json, os
from pathlib import Path
out = {"layers": {}}
for layer in (12, 24):
    layer_out = {}
    for suffix in ("heldout", "train_dist"):
        path = Path(f"findings/v1_1/fve_{suffix}_L{layer}.json")
        if path.exists():
            try:
                with open(path) as f:
                    d = json.load(f)
                layer_out[suffix] = d.get("summary", {})
            except Exception as e:
                layer_out[suffix] = {"error": str(e)}
    iter_log = Path(f"checkpoints/v1_1/L{layer}/iter_log.jsonl")
    if iter_log.exists():
        with open(iter_log) as f:
            eval_rows = [json.loads(l) for l in f if '"phase": "EVAL"' in l]
        layer_out["per_iteration"] = [
            {"iter": r["iter"], "fve": r.get("fve"), "cosine": r.get("cosine"), "mse": r.get("mse")}
            for r in eval_rows
        ]
    out["layers"][f"L{layer}"] = layer_out
out["status"] = "DONE"
Path("findings/v1_1/SUMMARY.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
EOF

heading "AUTOFINISH COMPLETE"
log "all outputs under findings/v1_1/ and artefacts/v1_1/ — pull with:"
log "  bin/h200 pull checkpoints/v1_1/"
log "  bin/h200 pull findings/v1_1/"
log "  bin/h200 pull artefacts/v1_1/"
log "  bin/h200 pull runs/autofinish.log"
