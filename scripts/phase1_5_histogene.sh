#!/usr/bin/env bash
# Phase 1.5 — HisToGene baseline + MCSPR across 8 LOPCV folds.
# Assumes Phase 1 TRIPLEX and ST-Net queues have cleared.
# Usage: ./scripts/phase1_5_histogene.sh [GPU_IDX]
set -euo pipefail

GPU=${1:-1}
LAMBDA_PATH="results/lambda_selection/her2st/selected_lambda.json"
CONFIG="configs/her2st.yaml"

if [[ ! -f "$LAMBDA_PATH" ]]; then
  echo "ERROR: $LAMBDA_PATH not found." >&2
  exit 1
fi

mkdir -p logs

run_fold() {
  local script=$1 fold=$2 log=$3
  shift 3
  echo "[GPU $GPU] $script fold $fold → $log"
  CUDA_VISIBLE_DEVICES=$GPU python src/experiments/${script}.py \
    --config "$CONFIG" --fold "$fold" "$@" \
    > "$log" 2>&1
  local ec=$?
  if [[ $ec -ne 0 ]]; then
    echo "[GPU $GPU] FAILED fold $fold (exit $ec) — continuing"
  fi
  return 0
}

echo "Phase 1.5: HisToGene on GPU $GPU (lambda=$(python -c 'import json; print(json.load(open("'"$LAMBDA_PATH"'"))["selected_lambda"])'))"

for fold in 0 1 2 3 4 5 6 7; do
  run_fold run_histogene "$fold" \
    "logs/phase1_5_histogene_base_fold${fold}.log"
  run_fold run_histogene_mcspr "$fold" \
    "logs/phase1_5_histogene_mcspr_fold${fold}.log" \
    --lambda_path "$LAMBDA_PATH"
done

echo "Phase 1.5 complete."
