#!/bin/bash
# End-to-end TRIPLEX (no MCSPR, no Hydra) run for a single fold of HER2ST
# 4-fold patient-paired LOPCV. Counterpart to run_triplex_hydra_one_fold.sh.
#
# Usage:
#   bash scripts/run_triplex_one_fold.sh                    # fold 0, GPU 0
#   FOLD=2 GPU=1 bash scripts/run_triplex_one_fold.sh
#
# Outputs:
#   results/baselines/triplex/fold_{F}/best_model.pt
#   results/baselines/triplex/fold_{F}/full.json
#   logs/triplex_fold{F}.log

set -u
cd "$(dirname "$0")/.."

FOLD="${FOLD:-0}"
GPU="${GPU:-0}"
CONFIG="${CONFIG:-configs/her2st.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/baselines/triplex}"
LOG_DIR="logs"
LOG="${LOG_DIR}/triplex_fold${FOLD}.log"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

DONE="${OUTPUT_DIR}/fold_${FOLD}/full.json"
if [ -f "$DONE" ] && [ "${FORCE:-0}" != "1" ]; then
    log "Fold $FOLD: full.json exists at $DONE — skipping"
    exit 0
fi

log "=== TRIPLEX | fold=${FOLD} GPU=${GPU} config=${CONFIG} ==="
START=$(date +%s)

CUDA_VISIBLE_DEVICES="$GPU" python -u src/experiments/run_triplex.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --fold "$FOLD" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

END=$(date +%s)
log "exit=$rc | wall=$((END - START))s"
exit $rc
