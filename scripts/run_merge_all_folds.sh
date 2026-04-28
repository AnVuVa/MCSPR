#!/bin/bash
# Sequential 4-fold MERGE baseline. Resume-safe (skips fold if GNN ckpt present).
# After all folds, runs canonical_eval --baseline merge to produce per-slide
# PCC(M)/PCC(H)/MSE/MAE in results/baselines/merge/canonical_summary.json.
#
# Usage:
#   bash scripts/run_merge_all_folds.sh
#   GPU=1 bash scripts/run_merge_all_folds.sh
#   FOLDS="0 1" bash scripts/run_merge_all_folds.sh

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
FOLDS="${FOLDS:-0 1 2 3}"
CONFIG="${CONFIG:-configs/merge_her2st_4fold.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-results/baselines/merge}"
LOG_DIR="logs"
MASTER="$LOG_DIR/merge_all_folds.log"
mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

log "=== MERGE sequential | GPU=$GPU folds=[$FOLDS] ==="
START_ALL=$(date +%s)

for FOLD in $FOLDS; do
    log "----- FOLD $FOLD START -----"
    F_START=$(date +%s)
    GPU="$GPU" CONFIG="$CONFIG" OUTPUT_BASE="$OUTPUT_BASE" FOLD="$FOLD" \
        bash scripts/run_merge_one_fold.sh
    rc=$?
    F_END=$(date +%s)
    log "----- FOLD $FOLD DONE rc=$rc wall=$((F_END - F_START))s -----"
    if [ $rc -ne 0 ]; then
        log "Fold $FOLD failed — aborting"
        exit $rc
    fi
done

END_ALL=$(date +%s)
log "=== ALL FOLDS COMPLETE wall=$((END_ALL - START_ALL))s ==="

log "Running canonical evaluation (--baseline merge)..."
python -u scripts/canonical_eval.py \
    --baseline merge \
    --config configs/her2st.yaml \
    --input_dir "$OUTPUT_BASE" 2>&1 | tee -a "$MASTER"
log "Done. canonical_summary.json at $OUTPUT_BASE/canonical_summary.json"
