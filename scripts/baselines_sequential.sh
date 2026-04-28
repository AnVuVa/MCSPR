#!/bin/bash
# Sequential baseline runs: TRIPLEX -> STNet -> HisToGene, all on HER2ST, GPU 1.
# RAM constraint (23 GiB WSL) forces serial execution.
set -u
cd /mnt/d/docker_machine/anvuva/MCSPR

LOG_DIR=logs
MASTER=$LOG_DIR/baselines_master.log
mkdir -p $LOG_DIR results/baselines

log() { echo "=== $(date '+%Y-%m-%d %H:%M:%S') $1 ===" | tee -a $MASTER; }

log "Sequential baselines starting"

log "[1/3] TRIPLEX baseline on GPU 1"
CUDA_VISIBLE_DEVICES=1 python -u src/experiments/run_triplex.py \
    --config configs/her2st.yaml \
    --output_dir results/baselines/triplex \
    > $LOG_DIR/baseline_triplex.log 2>&1
rc=$?
log "TRIPLEX exit=$rc"
[ $rc -ne 0 ] && { log "TRIPLEX failed — aborting sequence"; exit $rc; }

log "[2/3] STNet baseline on GPU 1 (HER2ST)"
CUDA_VISIBLE_DEVICES=1 python -u src/experiments/run_stnet.py \
    --config configs/her2st.yaml \
    --output_dir results/baselines/stnet \
    > $LOG_DIR/baseline_stnet.log 2>&1
rc=$?
log "STNet exit=$rc"
[ $rc -ne 0 ] && { log "STNet failed — aborting sequence"; exit $rc; }

log "[3/3] HisToGene baseline on GPU 1"
CUDA_VISIBLE_DEVICES=1 python -u src/experiments/run_histogene.py \
    --config configs/her2st.yaml \
    --output_dir results/baselines/histogene \
    > $LOG_DIR/baseline_histogene.log 2>&1
rc=$?
log "HisToGene exit=$rc"
[ $rc -ne 0 ] && { log "HisToGene failed — aborting sequence"; exit $rc; }

log "All baselines complete"
