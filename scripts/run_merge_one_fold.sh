#!/bin/bash
# End-to-end MERGE baseline for a single fold (CNN + GNN, pure baseline).
# Uses MCSPR's 4-fold patient-paired split + 300-SVG canonical counts.
#
# Usage:
#   bash scripts/run_merge_one_fold.sh                    # fold 0, GPU 0
#   FOLD=2 GPU=1 bash scripts/run_merge_one_fold.sh
#   CONFIG=configs/merge_her2st_4fold.yaml ...
#
# Outputs:
#   results/baselines/merge/fold_{F}/0/cnn/model_state_dict.pt
#   results/baselines/merge/fold_{F}/0/cnn/metrics.json
#   results/baselines/merge/fold_{F}/0/gnn/model_state_dict.pt
#   results/baselines/merge/fold_{F}/0/gnn/results.json
#   results/baselines/merge/preds/{slide}.npy           (300-gene per-slide)
#   logs/merge_fold{F}.log

set -u
cd "$(dirname "$0")/.."

# torch_geometric lives in the `merge` conda env, not MCSPR's default.
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate merge 2>/dev/null || true

FOLD="${FOLD:-0}"
GPU="${GPU:-0}"
CONFIG="${CONFIG:-configs/merge_her2st_4fold.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-results/baselines/merge}"
OUTPUT="${OUTPUT_BASE}/fold_${FOLD}"
LOG_DIR="logs"
LOG="${LOG_DIR}/merge_fold${FOLD}.log"

mkdir -p "$LOG_DIR" "$OUTPUT"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

GNN_DONE="${OUTPUT}/0/gnn/model_state_dict.pt"
if [ -f "$GNN_DONE" ] && [ "${FORCE:-0}" != "1" ]; then
    log "Fold $FOLD: GNN checkpoint exists at $GNN_DONE — skipping"
    exit 0
fi

log "=== MERGE end-to-end | fold=$FOLD GPU=$GPU ==="
START_TS=$(date +%s)

CUDA_VISIBLE_DEVICES="$GPU" python -u src/experiments/run_merge.py \
    -c "$CONFIG" -o "$OUTPUT" -d 0 -f "$FOLD" --mode all 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

END_TS=$(date +%s)
log "exit=$rc | wall=$((END_TS - START_TS))s"
exit $rc
