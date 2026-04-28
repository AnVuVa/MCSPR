#!/bin/bash
# Sequential 4-fold HydraSTNet run (folds 0..3) via the per-fold launcher.
# Each fold builds/uses its own registry and runs full train+eval.
# Resume-safe: a fold whose results/baselines/stnet_hydra/fold_{F}/full.json
# already exists will be skipped by run_stnet_hydra.py.
#
# Usage:
#   bash scripts/run_stnet_hydra_all_folds.sh           # GPU 0, K=7, folds 0..3
#   GPU=1 K=7 bash scripts/run_stnet_hydra_all_folds.sh # GPU 1
#   FOLDS="2 3" bash scripts/run_stnet_hydra_all_folds.sh # subset

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
K="${K:-7}"
FOLDS="${FOLDS:-0 1 2 3}"
LOG_DIR="logs"
MASTER="$LOG_DIR/stnet_hydra_all_folds.log"
mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

log "=== HydraSTNet sequential | GPU=$GPU K=$K folds=[$FOLDS] ==="
START_ALL=$(date +%s)

for FOLD in $FOLDS; do
    log "----- FOLD $FOLD START -----"
    FOLD_START=$(date +%s)
    GPU="$GPU" K="$K" FOLD="$FOLD" \
        bash scripts/run_stnet_hydra_one_fold.sh
    rc=$?
    FOLD_END=$(date +%s)
    log "----- FOLD $FOLD DONE rc=$rc wall=$((FOLD_END - FOLD_START))s -----"
    if [ $rc -ne 0 ]; then
        log "Fold $FOLD failed — aborting sequence"
        exit $rc
    fi
done

END_ALL=$(date +%s)
log "=== ALL FOLDS COMPLETE wall=$((END_ALL - START_ALL))s ==="

# Concise summary table across folds
log "=== Summary across folds ==="
python3 -c "
import json, os
from pathlib import Path
base = Path('results/baselines/stnet_hydra')
print(f\"{'fold':<6}{'n_val':<8}{'PCC(M)':<14}{'weighted':<12}{'best_train':<12}\")
for d in sorted(base.glob('fold_*')):
    fold_id = int(d.name.split('_')[1])
    p = d / 'module_breakdown.json'
    if not p.exists():
        print(f'{fold_id:<6}MISSING')
        continue
    b = json.loads(p.read_text())
    print(f\"{fold_id:<6}{b['n_val_slides']:<8}{b['pcc_m_per_slide_mean']:<14.4f}\"
          f\"{b['weighted_full_pcc']:<12.4f}{b['best_val_pcc_m_during_training']:<12.4f}\")
" 2>&1 | tee -a "$MASTER"
