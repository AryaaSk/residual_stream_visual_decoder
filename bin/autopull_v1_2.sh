#!/usr/bin/env bash
# autopull.sh — Mac-side companion to bin/autofinish.sh (on the H200).
# Polls the remote every 5 min, pulls any new artefacts under findings/v1_2/
# and artefacts/v1_2/. Auto-terminates after 12 hours or when SUMMARY.json
# is present locally (whichever comes first).

set +e
cd "$(dirname "$0")/.."

LOG=runs/autopull_v1_2.log
mkdir -p runs

H200_KEY=${H200_KEY:-$HOME/.ssh/gcp_zoral_h200_ed25519}
H200_USER=${H200_USER:-theod}
H200_HOST=${H200_HOST:-35.230.182.229}
REMOTE_DIR=/home/theod/Aryaa/rsvd

log() { echo "[autopull $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "starting; will poll every 5 min; auto-stop after 12 hr or when SUMMARY.json present locally"

DEADLINE=$(( $(date +%s) + 43200 ))   # 12 hours
while [[ $(date +%s) -lt "$DEADLINE" ]]; do
    if [[ -f "findings/v1_2/SUMMARY.json" ]]; then
        log "SUMMARY.json present locally — pipeline finished, final pull"
        rsync -avz -e "ssh -i $H200_KEY -o StrictHostKeyChecking=no" \
            "$H200_USER@$H200_HOST:$REMOTE_DIR/findings/v1_2/" findings/v1_2/ 2>&1 | tail -3 | tee -a "$LOG"
        rsync -avz -e "ssh -i $H200_KEY -o StrictHostKeyChecking=no" \
            "$H200_USER@$H200_HOST:$REMOTE_DIR/artefacts/v1_2/" artefacts/v1_2/ 2>&1 | tail -3 | tee -a "$LOG"
        rsync -avz -e "ssh -i $H200_KEY -o StrictHostKeyChecking=no" \
            "$H200_USER@$H200_HOST:$REMOTE_DIR/runs/autofinish.log" runs/ 2>&1 | tail -3 | tee -a "$LOG"
        log "auto-filling WRITEUP from SUMMARY.json"
        /usr/bin/env python3 code/eval/fill_writeup.py \
            --summary findings/v1_2/SUMMARY.json \
            --writeup WRITEUP.md 2>&1 | tee -a "$LOG"
        log "done."
        exit 0
    fi
    rsync -avz --partial -e "ssh -i $H200_KEY -o StrictHostKeyChecking=no" \
        "$H200_USER@$H200_HOST:$REMOTE_DIR/findings/v1_2/" findings/v1_2/ 2>/dev/null | tail -3 >> "$LOG"
    rsync -avz --partial -e "ssh -i $H200_KEY -o StrictHostKeyChecking=no" \
        "$H200_USER@$H200_HOST:$REMOTE_DIR/artefacts/v1_2/" artefacts/v1_2/ 2>/dev/null | tail -3 >> "$LOG"
    rsync -avz --partial -e "ssh -i $H200_KEY -o StrictHostKeyChecking=no" \
        "$H200_USER@$H200_HOST:$REMOTE_DIR/runs/iter11L12.log" runs/ 2>/dev/null
    rsync -avz --partial -e "ssh -i $H200_KEY -o StrictHostKeyChecking=no" \
        "$H200_USER@$H200_HOST:$REMOTE_DIR/runs/iter11L24.log" runs/ 2>/dev/null
    rsync -avz --partial -e "ssh -i $H200_KEY -o StrictHostKeyChecking=no" \
        "$H200_USER@$H200_HOST:$REMOTE_DIR/runs/autofinish.log" runs/ 2>/dev/null
    log "polled (deadline $(( (DEADLINE - $(date +%s)) / 60 )) min away)"
    sleep 300
done

log "deadline hit, exiting"
