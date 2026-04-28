#!/bin/bash
# End-to-end HydraTRIPLEX run for a single fold of the HER2ST 4-fold patient-paired
# LOPCV split. Reuses the per-fold gene module registry built for HydraSTNet
# (same KMeans-on-train-y partition, same SHA-256 verification, same fold split).
#
# Usage:
#   bash scripts/run_triplex_hydra_one_fold.sh                # fold 0, GPU 0
#   FOLD=2 GPU=1 bash scripts/run_triplex_hydra_one_fold.sh
#   K=8 FOLD=1 bash scripts/run_triplex_hydra_one_fold.sh
#   FORCE_REBUILD_REGISTRY=1 bash scripts/run_triplex_hydra_one_fold.sh
#
# Outputs:
#   results/ablation/kmeans_y_elbow/fold_{F}/modules_fold{F}.json
#   results/baselines/triplex_hydra/fold_{F}/best_model.pt
#   results/baselines/triplex_hydra/fold_{F}/training_log.json
#   results/baselines/triplex_hydra/fold_{F}/full.json
#   results/baselines/triplex_hydra/fold_{F}/head_{0..K-1}.json
#   results/baselines/triplex_hydra/fold_{F}/module_breakdown.json
#   logs/triplex_hydra_fold{F}.log

set -u
cd "$(dirname "$0")/.."

FOLD="${FOLD:-0}"
GPU="${GPU:-0}"
K="${K:-7}"
CONFIG="${CONFIG:-configs/her2st.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/baselines/triplex_hydra}"
REGISTRY="results/ablation/kmeans_y_elbow/fold_${FOLD}/modules_fold${FOLD}.json"
LOG_DIR="logs"
LOG="${LOG_DIR}/triplex_hydra_fold${FOLD}.log"
FORCE_REBUILD_REGISTRY="${FORCE_REBUILD_REGISTRY:-0}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== HydraTRIPLEX end-to-end | fold=${FOLD} GPU=${GPU} K=${K} ==="
log "config=${CONFIG} registry=${REGISTRY} output=${OUTPUT_DIR}"

# ----------------------------------------------------------------------------
# Step 1: ensure per-fold gene module registry exists (shared with HydraSTNet)
# ----------------------------------------------------------------------------
if [ -f "$REGISTRY" ] && [ "$FORCE_REBUILD_REGISTRY" != "1" ]; then
    log "[1/2] Registry already exists at $REGISTRY"
else
    log "[1/2] Building module registry (KMeans K=${K} on fold ${FOLD} training labels)"
    python -u scripts/build_module_registry.py \
        --fold "$FOLD" --K "$K" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        log "Registry build FAILED (rc=$rc)"; exit $rc
    fi
fi

python -u -c "
import json
r = json.load(open('${REGISTRY}'))
print(f'  registry: K={r[\"K\"]} sizes={r[\"module_sizes\"]} sum={sum(r[\"module_sizes\"])} '
      f'sha256={r[\"sha256\"][:12]}...')
" 2>&1 | tee -a "$LOG"

# ----------------------------------------------------------------------------
# Step 2: Train HydraTRIPLEX on this fold + canonical per-slide evaluation
# ----------------------------------------------------------------------------
log "[2/2] Training HydraTRIPLEX (fold ${FOLD}) — uses build_triplex_hydra_loaders"
START_TS=$(date +%s)

CUDA_VISIBLE_DEVICES="$GPU" python -u src/experiments/run_triplex_hydra.py \
    --config "$CONFIG" \
    --fold "$FOLD" \
    --registry "$REGISTRY" \
    --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
log "Training/eval exit=$rc | wall=${ELAPSED}s"
[ $rc -ne 0 ] && exit $rc

FOLD_DIR="${OUTPUT_DIR}/fold_${FOLD}"
log "=== Fold ${FOLD} summary ==="
if [ -f "${FOLD_DIR}/module_breakdown.json" ]; then
    python -u -c "
import json
b = json.load(open('${FOLD_DIR}/module_breakdown.json'))
print(f\"\\n  Metric: {b['metric']} (n_val_slides={b['n_val_slides']})\")
print(f\"  {'mod':<5}{'n_genes':<10}{'mean_PCC':<14}{'std':<10}\")
for m in b['modules']:
    print(f\"  {m['module_id']:<5}{m['n_genes']:<10}{m['module_mean_pcc']:<14.4f}{m['module_std_pcc']:<10.4f}\")
print(f\"  {'-'*40}\")
print(f\"  weighted_full_PCC = {b['weighted_full_pcc']:.4f}\")
print(f\"  per-slide PCC(M) mean (canonical)   = {b['pcc_m_per_slide_mean']:.4f}\")
print(f\"  best val PCC(M) during training     = {b['best_val_pcc_m_during_training']:.4f}\")
" 2>&1 | tee -a "$LOG"
fi

log "Done. Full log: ${LOG}"
