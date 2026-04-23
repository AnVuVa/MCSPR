#!/usr/bin/env bash
# Phase 1 GPU 1 queue only — used while GPU 0 is occupied by the L_norm diag.
# Mirrors phase1_resume.sh's GPU 1 subshell logic (ST-Net + HisToGene).
# When diag finishes, relaunch phase1_resume.sh to pick up the GPU 0 queue.
set -uo pipefail

LAMBDA_PATH="results/lambda_selection/her2st/selected_lambda.json"
CONFIG="configs/her2st.yaml"
DATASET="her2st"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

LAMBDA=$(python -c "import json; d=json.load(open('$LAMBDA_PATH')); print(d.get('selected_lambda', d.get('lambda')))")
log "Selected lambda: $LAMBDA"

already_done() {
  local arch=$1 fold=$2 out_root="results/${arch}/${DATASET}"
  [[ -f "${out_root}/fold_${fold}/training_log.json" ]]
}

run_fold() {
  local script=$1 fold=$2 gpu=$3 log_path=$4
  shift 4
  local arch=${script#run_}
  if already_done "$arch" "$fold"; then
    log "[GPU $gpu] SKIP $script fold=$fold — training_log.json exists"
    return 0
  fi
  log "[GPU $gpu] START $script fold=$fold -> $log_path"
  CUDA_VISIBLE_DEVICES=$gpu python -u src/experiments/${script}.py \
    --config "$CONFIG" --fold "$fold" "$@" \
    > "$log_path" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "[GPU $gpu] FAILED $script fold=$fold (exit $rc) — continuing"
  else
    log "[GPU $gpu] DONE $script fold=$fold"
  fi
  return 0
}

mkdir -p logs

log "GPU 1 queue started (stnet base+mcspr folds 0-7, then HisToGene)"
for fold in 0 1 2 3 4 5 6 7; do
  run_fold run_stnet $fold 1 "logs/phase1_stnet_base_fold${fold}.log"
  run_fold run_stnet_mcspr $fold 1 "logs/phase1_stnet_mcspr_fold${fold}.log" \
    --lambda_path "$LAMBDA_PATH"
done
log "GPU 1: ST-Net complete. Starting Phase 1.5 HisToGene."
bash scripts/phase1_5_histogene.sh 1
log "GPU 1 queue complete."
