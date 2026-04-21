#!/usr/bin/env bash
# Phase 1 Resume Orchestrator — replaces the dead master from phase1_launch.sh.
# Skips completed folds (training_log.json present), waits for any in-flight
# fold 0 process to release its GPU before starting the queue on that GPU.
#
# Launch:
#   nohup bash scripts/phase1_resume.sh >> logs/phase1_resume_master.log 2>&1 &
set -uo pipefail   # NOT -e: fold failures must not abort the sweep

LAMBDA_PATH="results/lambda_selection/her2st/selected_lambda.json"
CONFIG="configs/her2st.yaml"
DATASET="her2st"
PIDFILE="logs/phase1_resume.pid"
echo $$ > "$PIDFILE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [[ ! -f "$LAMBDA_PATH" ]]; then
  log "ERROR: $LAMBDA_PATH not found — aborting"
  exit 1
fi

LAMBDA=$(python -c "import json; d=json.load(open('$LAMBDA_PATH')); print(d.get('selected_lambda', d.get('lambda')))")
log "Selected lambda: $LAMBDA"

# ── In-flight PIDs to wait for (currently-running fold-0 processes) ────────
TRIPLEX_FOLD0_PID=431067   # GPU 0
STNET_FOLD0_PID=431961     # GPU 1 (already exited, but keep for safety)

wait_for_pid() {
  local pid=$1 label=$2
  if kill -0 "$pid" 2>/dev/null; then
    log "Waiting for $label (PID $pid) to finish before queuing on its GPU..."
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    log "$label (PID $pid) finished"
  else
    log "$label (PID $pid) already exited"
  fi
}

# ── Skip guard ──────────────────────────────────────────────────────────────
# A fold is "complete" when training_log.json is written — universal_trainer
# writes it at the END of the fold (after early stop or max_epochs).
# best_model.pt alone is NOT reliable: it is saved on every val improvement,
# so stale mid-training or old-session checkpoints would falsely trigger skip.
already_done() {
  local arch=$1 fold=$2 out_root=${3:-"results/${arch}/${DATASET}"}
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

run_ablation() {
  local tag=$1 fold=$2 gpu=$3 log_path=$4
  shift 4
  if [[ -f "results/ablation/${tag}/fold_${fold}/training_log.json" ]]; then
    log "[GPU $gpu] SKIP ablation $tag fold=$fold — already complete"
    return 0
  fi
  log "[GPU $gpu] START ablation $tag fold=$fold -> $log_path"
  CUDA_VISIBLE_DEVICES=$gpu python -u src/experiments/run_triplex_mcspr.py \
    --config "$CONFIG" --fold "$fold" \
    --lambda_path "$LAMBDA_PATH" \
    --output_dir "results/ablation/${tag}" \
    "$@" > "$log_path" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "[GPU $gpu] FAILED ablation $tag fold=$fold (exit $rc) — continuing"
  else
    log "[GPU $gpu] DONE ablation $tag fold=$fold"
  fi
  return 0
}

mkdir -p logs

# ── GPU 0 queue: TRIPLEX base → TRIPLEX MCSPR → ablations ──────────────────
(
  log "GPU 0 subshell started"
  wait_for_pid "$TRIPLEX_FOLD0_PID" "in-flight TRIPLEX fold 0"
  for fold in 0 1 2 3 4 5 6 7; do
    run_fold run_triplex $fold 0 \
      "logs/phase1_triplex_base_fold${fold}.log"
    run_fold run_triplex_mcspr $fold 0 \
      "logs/phase1_triplex_mcspr_fold${fold}.log" \
      --lambda_path "$LAMBDA_PATH"
  done
  run_ablation T1 1 0 \
    "logs/ablation_T1_fold1.log" --n_contexts 1
  run_ablation noema 1 0 \
    "logs/ablation_noema_fold1.log" --no_ema
  log "GPU 0 subshell complete"
) &
GPU0_PID=$!
log "GPU 0 subshell launched (PID $GPU0_PID)"

# ── GPU 1 queue: ST-Net base+MCSPR → HisToGene (Phase 1.5) ──────────────────
(
  log "GPU 1 subshell started"
  wait_for_pid "$STNET_FOLD0_PID" "in-flight ST-Net fold 0"
  for fold in 0 1 2 3 4 5 6 7; do
    run_fold run_stnet $fold 1 \
      "logs/phase1_stnet_base_fold${fold}.log"
    run_fold run_stnet_mcspr $fold 1 \
      "logs/phase1_stnet_mcspr_fold${fold}.log" \
      --lambda_path "$LAMBDA_PATH"
  done
  log "GPU 1: ST-Net complete. Starting Phase 1.5 HisToGene."
  bash scripts/phase1_5_histogene.sh 1
  log "GPU 1 subshell complete"
) &
GPU1_PID=$!
log "GPU 1 subshell launched (PID $GPU1_PID)"

log "Both subshells running. GPU0=$GPU0_PID GPU1=$GPU1_PID  orchestrator PID=$$"
log "PID file: $PIDFILE"
log "Monitor (orchestrator): tail -f logs/phase1_resume_master.log"

wait $GPU0_PID $GPU1_PID
log "Phase 1 + Phase 1.5 complete."
log "Next: python src/experiments/compare_results.py --config $CONFIG"
