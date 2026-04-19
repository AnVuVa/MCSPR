#!/usr/bin/env bash
# Phase 1 launch — 50-run sweep across two GPUs (rebalanced).
# GPU 0: TRIPLEX × 8 folds (base + MCSPR) -> ablations × 2
#        (18 runs; TRIPLEX is the heaviest model — undivided GPU 0 attention)
# GPU 1: HisToGene × 8 folds (base + MCSPR) -> ST-Net × 8 folds (base + MCSPR)
#        (32 runs; HisToGene is lighter so GPU 1 absorbs all ST-Net work too)
#
# Expected wall-clock (30 min/TRIPLEX fold, 20 min/HisToGene, 15 min/ST-Net):
#   GPU 0: 16*30 + 2*30 = 540 min ≈ 9 h
#   GPU 1: 16*20 + 16*15 = 560 min ≈ 9.3 h
#   Limiting path: ~9.3 h
#
# Fires only when results/lambda_selection/her2st/selected_lambda.json exists.
# Per-fold failures are logged but do not abort the sweep.

set -uo pipefail

LAMBDA_PATH="results/lambda_selection/her2st/selected_lambda.json"
CONFIG="configs/her2st.yaml"
LOG_DIR="logs"

if [[ ! -f "$LAMBDA_PATH" ]]; then
  echo "ERROR: $LAMBDA_PATH not found. Run select_lambda.py first." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

SELECTED_LAMBDA=$(python -c "import json; d=json.load(open('$LAMBDA_PATH')); print(d.get('lambda', d.get('selected_lambda', d)))")
echo "Lambda path: $LAMBDA_PATH"
echo "Selected lambda: $SELECTED_LAMBDA"

run_fold() {
  local script=$1 fold=$2 gpu=$3 log=$4
  shift 4
  echo "[GPU $gpu] START $script fold=$fold -> $log"
  CUDA_VISIBLE_DEVICES=$gpu python src/experiments/${script}.py \
    --config "$CONFIG" --fold "$fold" "$@" \
    > "$log" 2>&1
  local exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    echo "[GPU $gpu] FAIL $script fold=$fold (exit $exit_code) — continuing"
  else
    echo "[GPU $gpu] DONE $script fold=$fold"
  fi
  return 0
}

# GPU 0: TRIPLEX × 8 folds (base + MCSPR) -> ablations (Fold 1, T=1 and no-EMA)
(
  for fold in 0 1 2 3 4 5 6 7; do
    run_fold run_triplex $fold 0 "$LOG_DIR/phase1_triplex_base_fold${fold}.log"
    run_fold run_triplex_mcspr $fold 0 "$LOG_DIR/phase1_triplex_mcspr_fold${fold}.log" \
      --lambda_path "$LAMBDA_PATH"
  done
  run_fold run_triplex_mcspr 1 0 "$LOG_DIR/ablation_T1_fold1.log" \
    --lambda_path "$LAMBDA_PATH" --n_contexts 1 \
    --output_dir results/ablation/T1
  run_fold run_triplex_mcspr 1 0 "$LOG_DIR/ablation_noema_fold1.log" \
    --lambda_path "$LAMBDA_PATH" --no_ema \
    --output_dir results/ablation/noema
) &
GPU0_PID=$!

# GPU 1: HisToGene × 8 folds (base + MCSPR) -> ST-Net × 8 folds (base + MCSPR)
(
  for fold in 0 1 2 3 4 5 6 7; do
    run_fold run_histogene $fold 1 "$LOG_DIR/phase1_histogene_base_fold${fold}.log"
    run_fold run_histogene_mcspr $fold 1 "$LOG_DIR/phase1_histogene_mcspr_fold${fold}.log" \
      --lambda_path "$LAMBDA_PATH"
  done
  for fold in 0 1 2 3 4 5 6 7; do
    run_fold run_stnet $fold 1 "$LOG_DIR/phase1_stnet_base_fold${fold}.log"
    run_fold run_stnet_mcspr $fold 1 "$LOG_DIR/phase1_stnet_mcspr_fold${fold}.log" \
      --lambda_path "$LAMBDA_PATH"
  done
) &
GPU1_PID=$!

echo "GPU 0 PID: $GPU0_PID | GPU 1 PID: $GPU1_PID"
echo "Monitor GPU 0: tail -f $LOG_DIR/phase1_triplex_base_fold0.log"
echo "Monitor GPU 1: tail -f $LOG_DIR/phase1_histogene_base_fold0.log"

wait $GPU0_PID $GPU1_PID
echo "Phase 1 complete. Next: python src/experiments/compare_results.py --config $CONFIG"
