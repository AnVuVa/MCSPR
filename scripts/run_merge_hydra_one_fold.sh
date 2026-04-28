#!/bin/bash
# End-to-end HydraMERGE for a single fold (CNN + HydraGAT, weighted MSE).
# Reuses the MCSPR 4-fold patient-paired split + 300-SVG canonical counts +
# the per-fold module registry built by scripts/build_module_registry.py.
#
# Usage:
#   bash scripts/run_merge_hydra_one_fold.sh                    # fold 0, GPU 0
#   FOLD=2 GPU=0 bash scripts/run_merge_hydra_one_fold.sh
#   CONFIG=configs/merge_hydra_her2st_4fold.yaml ...
#
# Outputs:
#   results/baselines/merge_hydra/fold_{F}/0/cnn/model_state_dict.pt
#   results/baselines/merge_hydra/fold_{F}/0/cnn/metrics.json
#   results/baselines/merge_hydra/fold_{F}/0/gnn/model_state_dict.pt
#   results/baselines/merge_hydra/fold_{F}/0/gnn/results.json
#   results/baselines/merge_hydra/fold_{F}/full.json
#   results/baselines/merge_hydra/fold_{F}/module_breakdown.json
#   results/baselines/merge_hydra/fold_{F}/head_{k}.json
#   results/baselines/merge_hydra/preds/{slide}.npy
#   logs/merge_hydra_fold{F}.log

set -u
cd "$(dirname "$0")/.."

# torch_geometric + full ML stack live in the `cvst` conda env (sandbox).
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate cvst 2>/dev/null || true

FOLD="${FOLD:-0}"
GPU="${GPU:-0}"
CONFIG="${CONFIG:-configs/merge_hydra_her2st_4fold.yaml}"
# Use a placeholder string ('FOLDID') in the default to dodge bash's flaky
# brace-expansion behavior on `{F}` inside `${...:-default}` substitutions.
# Both this script and the runner accept either {F} or FOLDID as placeholder.
REGISTRY="${REGISTRY:-results/ablation/kmeans_y_elbow/fold_FOLDID/modules_foldFOLDID.json}"
OUTPUT_BASE="${OUTPUT_BASE:-results/baselines/merge_hydra}"
OUTPUT="${OUTPUT_BASE}/fold_${FOLD}"
LOG_DIR="logs"
LOG="${LOG_DIR}/merge_hydra_fold${FOLD}.log"

mkdir -p "$LOG_DIR" "$OUTPUT"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

DONE_FILE="${OUTPUT}/full.json"
if [ -f "$DONE_FILE" ] && [ "${FORCE:-0}" != "1" ]; then
    log "Fold $FOLD: full.json exists at $DONE_FILE — skipping"
    exit 0
fi

# Resolve registry path so we can warn early if it is missing. Use sed
# instead of bash's `${var//pat/repl}` to dodge interactions with extglob /
# brace-expansion edge cases on some shells.
RES_REG=$(echo "$REGISTRY" | sed -e "s/{F}/$FOLD/g" -e "s/FOLDID/$FOLD/g")
if [ ! -f "$RES_REG" ]; then
    log "ERROR: registry not found at $RES_REG"
    log "Build it: python scripts/build_module_registry.py --fold $FOLD --K 7"
    exit 1
fi

log "=== HydraMERGE end-to-end | fold=$FOLD GPU=$GPU registry=$RES_REG ==="
START_TS=$(date +%s)

CUDA_VISIBLE_DEVICES="$GPU" python -u src/experiments/run_merge_hydra.py \
    -c "$CONFIG" -o "$OUTPUT" --registry "$REGISTRY" \
    -d 0 -f "$FOLD" --mode all 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

END_TS=$(date +%s)
log "exit=$rc | wall=$((END_TS - START_TS))s"
exit $rc
