#!/bin/bash
# End-to-end HydraSTNet run for a single fold of the HER2ST 4-fold patient-paired
# LOPCV split (A+B | C+D | E+F | G+H per spec v2). Builds the per-fold gene
# module registry, then runs full train + canonical evaluation through the
# dedicated HydraSTNet dataloader (build_stnet_hydra_loaders), and prints the
# per-module breakdown at the end.
#
# Usage:
#   bash scripts/run_stnet_hydra_one_fold.sh                # fold 0, GPU 0
#   FOLD=2 GPU=1 bash scripts/run_stnet_hydra_one_fold.sh   # fold 2 on GPU 1
#   K=8 FOLD=1 bash scripts/run_stnet_hydra_one_fold.sh     # custom K
#   FORCE_REBUILD_REGISTRY=1 bash scripts/run_stnet_hydra_one_fold.sh
#
# Outputs:
#   results/ablation/kmeans_y_elbow/fold_{F}/modules_fold{F}.json
#   results/baselines/stnet_hydra/fold_{F}/best_model.pt
#   results/baselines/stnet_hydra/fold_{F}/training_log.json
#   results/baselines/stnet_hydra/fold_{F}/full.json
#   results/baselines/stnet_hydra/fold_{F}/head_{0..K-1}.json
#   results/baselines/stnet_hydra/fold_{F}/module_breakdown.json
#   logs/stnet_hydra_fold{F}.log

set -u
cd "$(dirname "$0")/.."   # repo root regardless of where the script is invoked

FOLD="${FOLD:-0}"
GPU="${GPU:-0}"
K="${K:-7}"
CONFIG="${CONFIG:-configs/her2st.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/baselines/stnet_hydra}"
REGISTRY="results/ablation/kmeans_y_elbow/fold_${FOLD}/modules_fold${FOLD}.json"
LOG_DIR="logs"
LOG="${LOG_DIR}/stnet_hydra_fold${FOLD}.log"
FORCE_REBUILD_REGISTRY="${FORCE_REBUILD_REGISTRY:-0}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== HydraSTNet end-to-end | fold=${FOLD} GPU=${GPU} K=${K} ==="
log "config=${CONFIG} registry=${REGISTRY} output=${OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# Step 1: Build the per-fold gene module registry (KMeans on training labels)
# -----------------------------------------------------------------------------
if [ -f "$REGISTRY" ] && [ "$FORCE_REBUILD_REGISTRY" != "1" ]; then
    log "[1/2] Registry already exists at $REGISTRY (set FORCE_REBUILD_REGISTRY=1 to rebuild)"
else
    log "[1/2] Building module registry (KMeans K=${K} on fold ${FOLD} training labels)"
    python -u scripts/build_module_registry.py \
        --fold "$FOLD" --K "$K" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        log "Registry build FAILED (rc=$rc)"; exit $rc
    fi
fi

# Show the registry's module sizes + sha256 for the run record
python -u -c "
import json, sys
r = json.load(open('${REGISTRY}'))
print(f'  registry: K={r[\"K\"]} sizes={r[\"module_sizes\"]} sum={sum(r[\"module_sizes\"])} '
      f'n_train_spots={r[\"n_train_spots\"]} sha256={r[\"sha256\"][:12]}...')
print(f'  train_slides ({len(r[\"train_slides\"])}): {r[\"train_slides\"]}')
" 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------------
# Step 2: Train HydraSTNet on this fold + canonical per-slide evaluation
# -----------------------------------------------------------------------------
log "[2/2] Training HydraSTNet (fold ${FOLD}) — uses build_stnet_hydra_loaders"
START_TS=$(date +%s)

CUDA_VISIBLE_DEVICES="$GPU" python -u src/experiments/run_stnet_hydra.py \
    --config "$CONFIG" \
    --fold "$FOLD" \
    --registry "$REGISTRY" \
    --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
log "Training/eval exit=$rc | wall=${ELAPSED}s"
[ $rc -ne 0 ] && exit $rc

# -----------------------------------------------------------------------------
# Summary print
# -----------------------------------------------------------------------------
FOLD_DIR="${OUTPUT_DIR}/fold_${FOLD}"
log "=== Fold ${FOLD} summary ==="
log "Artifacts:"
log "  $(ls -la ${FOLD_DIR} 2>/dev/null | grep -E 'json|pt' | awk '{print $NF, \"(\" $5, \"bytes)\"}' | tr '\n' ' ')"

if [ -f "${FOLD_DIR}/module_breakdown.json" ]; then
    python -u -c "
import json
b = json.load(open('${FOLD_DIR}/module_breakdown.json'))
print(f\"\\n  Metric: {b['metric']} (n_val_slides={b['n_val_slides']})\")
print(f\"  {'mod':<5}{'n_genes':<10}{'mean_PCC':<14}{'std':<10}\")
for m in b['modules']:
    print(f\"  {m['module_id']:<5}{m['n_genes']:<10}{m['module_mean_pcc']:<14.4f}{m['module_std_pcc']:<10.4f}\")
print(f\"  {'-'*40}\")
print(f\"  weighted_full_PCC (Σ m_k·μ_k / 300) = {b['weighted_full_pcc']:.4f}\")
print(f\"  per-slide PCC(M) mean (canonical)   = {b['pcc_m_per_slide_mean']:.4f}\")
print(f\"  best val PCC(M) during training     = {b['best_val_pcc_m_during_training']:.4f}\")
print(f\"  registry sha256: {b['registry_hash'][:16]}...\")
" 2>&1 | tee -a "$LOG"
fi

log "Done. Full log: ${LOG}"
