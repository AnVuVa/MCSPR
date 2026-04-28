#!/bin/bash
# Sequential 4-fold TRIPLEX baseline on counts_svg target.
# Resume-safe (skips fold if full.json present).
#
# Usage:
#   bash scripts/run_triplex_all_folds.sh
#   GPU=1 bash scripts/run_triplex_all_folds.sh
#   FOLDS="0 1" bash scripts/run_triplex_all_folds.sh

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
FOLDS="${FOLDS:-0 1 2 3}"
CONFIG="${CONFIG:-configs/her2st.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/baselines/triplex}"
LOG_DIR="logs"
MASTER="$LOG_DIR/triplex_all_folds.log"
mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

log "=== TRIPLEX sequential | GPU=$GPU folds=[$FOLDS] config=$CONFIG ==="
START_ALL=$(date +%s)

for FOLD in $FOLDS; do
    log "----- FOLD $FOLD START -----"
    F_START=$(date +%s)
    GPU="$GPU" CONFIG="$CONFIG" OUTPUT_DIR="$OUTPUT_DIR" FOLD="$FOLD" \
        bash scripts/run_triplex_one_fold.sh
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

log "Running canonical evaluation (--baseline triplex)..."
python -u scripts/canonical_eval.py \
    --baseline triplex \
    --config "$CONFIG" \
    --input_dir "$OUTPUT_DIR" 2>&1 | tee -a "$MASTER"
log "Done. canonical_summary.json at $OUTPUT_DIR/canonical_summary.json"
