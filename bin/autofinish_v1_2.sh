#!/usr/bin/env bash
# autofinish.sh — autonomous post-training pipeline for v1.1.
#
# Designed to run inside a detached tmux on the H200 alongside the training
# jobs. Polls the training logs every 60s; when both have written "DONE",
# kicks off the full eval + artefact pipeline (inject_demo, FVE measurements,
# cross-layer trajectory, hype reel). Everything lands on the remote disk
# under ~/Aryaa/rsvd/findings/v1_2/ and ~/Aryaa/rsvd/checkpoints/v1_2/ so
# the user can pull when they wake up.
#
# Run on the H200 (uses /home/theod/Aryaa/rsvd as cwd):
#     tmux new-session -d -s autofinish 'bash /home/theod/Aryaa/rsvd/bin/autofinish.sh'
#
# Designed to be IDEMPOTENT: if it crashes or is restarted, it skips work that
# already completed.

set +e  # don't bail on individual step failure — log and continue

cd /home/theod/Aryaa/rsvd
LOG=runs/autofinish_v1_2.log
VENV=/home/theod/Aryaa/venv/bin/python3
export PYTHONPATH=code
export CUDA_VISIBLE_DEVICES=6,7  # use whichever's free; eval kernels work on either

log() { echo "[autofinish $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
heading() { echo "" | tee -a "$LOG"; log "============================ $* ============================"; }

heading "STARTUP"
log "PID=$$ PYTHONPATH=$PYTHONPATH"
log "waiting for both s15_L12.log and s15_L24.log to contain '[s15] DONE' ..."

# Phase 0 — wait for both training jobs to complete (or error)
while true; do
    L12_DONE=$(grep -cE "^\[s15\] DONE → " runs/s15_L12.log 2>/dev/null)
    L24_DONE=$(grep -cE "^\[s15\] DONE → " runs/s15_L24.log 2>/dev/null)
    L12_ERR=$(grep -cE "^Traceback|^Error:|Killed|OOM" runs/s15_L12.log 2>/dev/null)
    L24_ERR=$(grep -cE "^Traceback|^Error:|Killed|OOM" runs/s15_L24.log 2>/dev/null)
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

# Phase 0.5 (v1.2): no-op. Stage 1.5 trainer writes directly to <layer_dir>/final
# (no FVE-best vs last-iter ambiguity in this stage).

# Phase 1 — inject_demo for each layer
# Hero clips also rendered at α=0.3, 0.7, 1.0 in case 0.5 is suboptimal here.
heading "PHASE 1: inject_demo per layer (26 prompts main pass + hero α sweep)"
for LAYER_DIR in checkpoints/v1_2/L12 checkpoints/v1_2/L24; do
    if [[ ! -d "$LAYER_DIR/final" ]]; then
        log "WARN: $LAYER_DIR/final missing, skipping inject_demo for that layer"
        continue
    fi
    LAYER=$(basename "$LAYER_DIR" | sed 's/^L//')
    OUT="findings/v1_2/inject_demo_L${LAYER}"
    log "inject_demo L=$LAYER → $OUT  (main α=0.5 + hero sweep α∈{0.3,0.7,1.0})"
    $VENV code/eval/inject_demo.py \
        --av-ckpt "$LAYER_DIR/final" --layer "$LAYER" \
        --alpha 0.5 --mp4 --out-dir "$OUT" \
        --alpha-sweep 0.3 0.7 1.0 \
        2>&1 | tee -a "$LOG"
done

# Phase 2/3/4 (FVE eval + iter plots) skipped for v1.2:
# Stage 1.5 retrains the AV only — no new AR — so FVE numbers from v1.1's AR
# applied to v1.2's AV would be misleading and slow. v1.2 ship gate is visual
# recognisability, not FVE. We do, however, plot the SFT loss curve.
heading "PHASE 2-4 SKIPPED for v1.2 (no AR retrain). Plotting SFT loss instead."
for LAYER in 12 24; do
    SFT_LOG="checkpoints/v1_2/L${LAYER}/train.jsonl"
    OUT="findings/v1_2/sft_loss_L${LAYER}.png"
    if [[ ! -f "$SFT_LOG" ]]; then
        continue
    fi
    $VENV - "$SFT_LOG" "$OUT" "$LAYER" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sft_log, out_path, layer = sys.argv[1], sys.argv[2], sys.argv[3]
xs, ys = [], []
for l in open(sft_log):
    d = json.loads(l)
    if "loss" in d and "step" in d:
        xs.append(d["step"]); ys.append(d["loss"])
plt.figure(figsize=(8, 4))
plt.plot(xs, ys, lw=0.8)
plt.xlabel("step"); plt.ylabel("CE loss"); plt.title(f"v1.2 Stage 1.5 SFT loss — L{layer}")
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(out_path, dpi=120)
print(f"wrote {out_path}")
PYEOF
done

# Phase 5 — cross-layer trajectory (uses both L12 and L24)
heading "PHASE 5: cross-layer trajectory"
$VENV code/eval/cross_layer_trajectory.py \
    --ckpts-root checkpoints/v1_2 --layers 12 24 \
    --out-dir artefacts/v1_2/cross_layer \
    --alpha 0.5 \
    2>&1 | tee -a "$LOG"

# Phase 6 — per-token trajectory (uses L12)
heading "PHASE 6: per-token trajectory (L12)"
if [[ -d "checkpoints/v1_2/L12/final" ]]; then
    $VENV code/eval/token_trajectory.py \
        --av-ckpt checkpoints/v1_2/L12/final --layer 12 --alpha 0.5 \
        --out-dir artefacts/v1_2/trajectory_L12 \
        --max-gen-tokens 12 \
        2>&1 | tee -a "$LOG"
else
    log "skip token_trajectory (L12 final missing)"
fi

# Phase 7 — hype reel: hero inject_demo clips + cross-layer + per-token
heading "PHASE 7: 60-sec hype reel"
mkdir -p artefacts/v1_2/reel_src
# Hero clips: in-distribution trained-concept prompts (v1.2 hero set)
for SLUG in cat dog fish flower sun house car airplane bird elephant; do
    for LAYER in 12 24; do
        SRC="findings/v1_2/inject_demo_L${LAYER}/${SLUG}_4x.mp4"
        DST="artefacts/v1_2/reel_src/hero_L${LAYER}_${SLUG}.mp4"
        [[ -f "$SRC" ]] && cp "$SRC" "$DST"
    done
done
cp artefacts/v1_2/cross_layer/*.mp4 artefacts/v1_2/reel_src/ 2>/dev/null || true
cp artefacts/v1_2/trajectory_L12/*.mp4 artefacts/v1_2/reel_src/ 2>/dev/null || true
log "reel_src contents:"
ls artefacts/v1_2/reel_src/ | head -30 | tee -a "$LOG"
$VENV code/eval/make_hype_reel.py \
    --in-dir artefacts/v1_2/reel_src --out artefacts/v1_2/demo.mp4 \
    --max-clips 12 \
    2>&1 | tee -a "$LOG"

# Phase 8 — write a top-level summary JSON for easy reading
heading "PHASE 8: summary JSON"
$VENV - <<'EOF' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
out = {"stage": "1.5 (act-conditioned SFT)", "layers": {}}
for layer in (12, 24):
    layer_out = {}
    train_log = Path(f"checkpoints/v1_2/L{layer}/train.jsonl")
    if train_log.exists():
        rows = [json.loads(l) for l in open(train_log) if l.strip()]
        if rows:
            layer_out["sft_loss_first"] = rows[0].get("loss")
            layer_out["sft_loss_last"] = rows[-1].get("loss")
            layer_out["n_steps_logged"] = len(rows)
    out["layers"][f"L{layer}"] = layer_out
out["status"] = "DONE"
Path("findings/v1_2/SUMMARY.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
EOF

heading "AUTOFINISH COMPLETE"
log "all outputs under findings/v1_2/ and artefacts/v1_2/ — pull with:"
log "  bin/h200 pull checkpoints/v1_2/"
log "  bin/h200 pull findings/v1_2/"
log "  bin/h200 pull artefacts/v1_2/"
log "  bin/h200 pull runs/autofinish.log"
