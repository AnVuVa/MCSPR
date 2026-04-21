#!/usr/bin/env bash
# Sequencer: waits for Run A (L_MSE) to finish, then launches Run B (L_norm),
# then runs diagnostic_eval on both. Safe to re-invoke if Run A already finished.
# Safety timeout: 5 hours per run.
set -u
cd "$(dirname "$0")/.."

MASTER_LOG="logs/diag_master.log"
mkdir -p logs results/diag
echo "[$(date)] sequencer START (pid=$$)" >> "$MASTER_LOG"

wait_marker() {
    local path="$1"
    local marker="$2"
    local max_sec="${3:-18000}"   # 5h default
    local waited=0
    while ! grep -q "$marker" "$path" 2>/dev/null; do
        sleep 60
        waited=$((waited + 60))
        if [ $waited -ge $max_sec ]; then
            echo "[$(date)] TIMEOUT waiting for '$marker' in $path after ${max_sec}s" >> "$MASTER_LOG"
            return 1
        fi
    done
    return 0
}

# --- Stage 1: Wait for Run A ---
echo "[$(date)] waiting for Run A marker in logs/diag_triplex_lmse_fold0.log" >> "$MASTER_LOG"
if ! wait_marker "logs/diag_triplex_lmse_fold0.log" "Run A finished"; then
    echo "[$(date)] ABORT: Run A never finished" >> "$MASTER_LOG"
    exit 1
fi
echo "[$(date)] Run A complete" >> "$MASTER_LOG"

# --- Stage 2: Launch Run B ---
echo "=== Run B: L_norm, fold 0, 50 epochs, MCSPR off, GPU 0 ===" >> logs/diag_triplex_lnorm_fold0.log
date >> logs/diag_triplex_lnorm_fold0.log
CUDA_VISIBLE_DEVICES=0 python -u src/experiments/run_triplex.py \
    --config configs/her2st_diag.yaml \
    --fold 0 \
    --use_normalized_mse \
    --output_dir results/diag/triplex_lnorm \
    >> logs/diag_triplex_lnorm_fold0.log 2>&1
RC=$?
echo "--- Run B finished at $(date), rc=$RC ---" >> logs/diag_triplex_lnorm_fold0.log
echo "[$(date)] Run B complete rc=$RC" >> "$MASTER_LOG"

# --- Stage 3: Diagnostic eval on both ---
for variant in triplex_lmse triplex_lnorm; do
    fold_dir="results/diag/${variant}/fold_0"
    out_json="${fold_dir}/diagnostic.json"
    if [ -f "${fold_dir}/best_model.pt" ]; then
        CUDA_VISIBLE_DEVICES=0 python -u scripts/diagnostic_eval.py \
            --config configs/her2st_diag.yaml \
            --fold 0 \
            --fold_dir "$fold_dir" \
            --out "$out_json" \
            > "logs/diag_eval_${variant}.log" 2>&1
        echo "[$(date)] eval ${variant} rc=$?" >> "$MASTER_LOG"
    else
        echo "[$(date)] SKIP eval ${variant} — no best_model.pt at $fold_dir" >> "$MASTER_LOG"
    fi
done

echo "[$(date)] sequencer DONE" >> "$MASTER_LOG"
