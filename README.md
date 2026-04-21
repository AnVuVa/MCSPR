# MCSPR: Morphology-Conditioned Spatial Pathway Regularization

A drop-in auxiliary loss that aligns model-predicted expression with a
context-specific, histology-conditioned gene-module correlation structure
learned from training tissue. Paper: *(NeurIPS submission — citation pending)*.

This README is a **command reference**. Copy-paste everything you need
to prep data, train, evaluate, and run diagnostics.

---

## Table of contents

- [Requirements](#requirements)
- [Project layout](#project-layout)
- [Environment & health checks](#environment--health-checks)
- [Data preparation (one-time)](#data-preparation-one-time)
- [Precompute pipeline](#precompute-pipeline)
- [Lambda selection](#lambda-selection)
- [Training](#training)
- [L_norm diagnostic (per-gene variance-normalized MSE)](#l_norm-diagnostic-per-gene-variance-normalized-mse)
- [Evaluation](#evaluation)
- [Monitoring live runs](#monitoring-live-runs)
- [Result locations](#result-locations)
- [Troubleshooting & common patterns](#troubleshooting--common-patterns)
- [Key design decisions](#key-design-decisions)
- [Citation & license](#citation--license)

---

## Requirements

- Python 3.13 (earlier versions likely work; not tested)
- `torch`, `torchvision`, `numpy`, `scipy`, `scanpy`, `scikit-learn`,
  `pandas`, `pyyaml`, `Pillow`
- **Hardware tested:** 2× RTX 5070 Ti (16 GB each). TRIPLEX/HisToGene need
  ~8–12 GB at `batch_size=128` with mixed precision. ST-Net is lighter (~4–6 GB).
  CPU inference works (`--device cpu`) but is ~30× slower.
- Ciga ResNet18 SimCLR encoder at
  `/mnt/d/docker_machine/anvuva/TRIPLEX/weights/cigar/tenpercent_resnet18.ckpt`
  (falls back to ImageNet pretraining if absent).

No `setup.py` / `requirements.txt` — environment is managed externally.

---

## Project layout

```
mcspr/                              core MCSPR package
  core/loss.py                      MCSPRLoss (EMA-on-moments, bias-corrected)
  core/scheduler.py                 LambdaScheduler (constant-lambda protocol)
  prior/construction.py             NMF fitting + per-context C_prior
  metrics/smcs.py                   SMCS (structural module correlation score)
  validation/                       drift / generalization checks

src/
  data/
    loaders.py                      build_loaders, build_lopcv_folds
    dataset.py                      TRIPLEX/ST-Net dataset (counts_svg for 300g)
    histogene_dataset.py            HisToGene slide-level dataset
    precompute.py                   global_features, KMeans, NMF, C_prior, gene_var
  models/                           triplex.py, stnet.py, histogene.py, ciga_encoder.py
  training/
    trainer.py                      TRIPLEX trainer (TRIPLEX + TRIPLEX_MCSPR)
    universal_trainer.py            ST-Net + HisToGene trainer
    evaluate.py                     per-slide PCC(M)/PCC(H)/MSE/MAE + SMCS
  experiments/
    run_triplex.py, run_triplex_mcspr.py
    run_stnet.py, run_stnet_mcspr.py
    run_histogene.py, run_histogene_mcspr.py
    select_lambda.py                anti-leakage λ sweep
    compare_results.py              paired baseline vs MCSPR table
  losses/
    triplex_loss.py                 TRIPLEXLoss (4-branch MSE; never modified)
    normalized_mse.py               NormalizedMSELoss (gene-var-normalized MSE)

scripts/                            launch, eval, monitoring
configs/                            her2st.yaml, her2st_diag.yaml, stnet.yaml, scc.yaml
data/her2st/                        raw + precomputed artifacts (not versioned)
results/                            fold outputs, λ-selection artifacts
logs/                               training + orchestrator logs
```

---

## Environment & health checks

```bash
# GPU status (memory, util, per-PID mapping)
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

# Live orchestrator + training processes
ps -eo pid,etime,stat,cmd --sort=-etime | grep -E "phase1_resume|run_(stnet|triplex|histogene)" | grep -v grep

# Uptime / who / system load (sanity after unexpected orchestrator death)
uptime
```

State decoder: `Rl` = running multi-threaded (healthy), `Dl` = uninterruptible
sleep (usually disk I/O, transient), `S` = waiting on child, `Z` = zombie.
Transient `Dl` is normal during checkpoint writes; persistent `Dl` (>30 min)
suggests a stuck filesystem / deadlock.

---

## Data preparation (one-time)

HER2ST (Andersson et al., [`almaan/her2st`](https://github.com/almaan/her2st))
is the primary benchmark.

```bash
# 1. Clone dataset into data/her2st/
git clone https://github.com/almaan/her2st.git data/her2st

# 2. Alignment + conversion + gene selection
python scripts/verify_spot_ids.py
python scripts/convert_her2st_to_npy.py
python scripts/run_gate1.py              # seurat_v3 HVG ∩ Moran's I → 300-SVG panel
python scripts/run_gate2_enrichment.py   # NMF module × Hallmark MSigDB (validation only)
```

Produces `data/her2st/{barcodes, counts_svg, counts_spcs, tissue_positions,
wsi, wsi224, wsi448, features, features_full, features_svg, umi_counts}`.

---

## Precompute pipeline

```bash
# FULL precompute (destructive — re-runs KMeans + NMF; overwrites M, C_prior, etc.)
python src/data/precompute.py --config configs/her2st.yaml
```

Produces per LOPCV fold (no leakage — training slides only):

- `data/her2st/global_features/{slide}.npy` — 512-d Ciga ResNet18 patch embeddings
- `data/her2st/context_weights/fold_{i}/{slide}.npy` — soft KMeans assignments (6 contexts)
- `data/her2st/context_labels/fold_{i}/{slide}.npy` — hard assignments
- `data/her2st/nmf/fold_{i}/M.npy` — NMF loadings `(300, 15)`
- `data/her2st/nmf/fold_{i}/M_pinv.npy` — pseudoinverse `(15, 300)`
- `data/her2st/nmf/fold_{i}/C_prior.npy` — per-context correlation prior `(6, 15, 15)`
- `data/her2st/nmf/fold_{i}/gene_var.npy` — per-gene training variance `(300,)`

**WARNING — do not run full precompute while Phase 1 is training.** NMF and
KMeans are non-deterministic at the margin and will silently shift `C_prior`
between folds that have already completed and folds not yet started, breaking
paired comparisons. If you only need `gene_var.npy` added to existing folds,
use the one-off additive script below.

### One-off additive gene_var export (non-destructive)

```bash
# Adds gene_var.npy to each nmf/fold_{i}/ without touching M / C_prior / labels.
python - <<'PY'
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from src.data.loaders import build_lopcv_folds

base = Path('data/her2st')
sample_names = sorted([p.stem for p in (base / 'barcodes').glob('*.csv')])
for fold_idx, (train_samples, _) in enumerate(build_lopcv_folds(sample_names, 'her2st')):
    parts = [np.load(str(base / 'counts_svg' / f'{s}.npy')).astype(np.float32)
             for s in train_samples
             if (base / 'counts_svg' / f'{s}.npy').exists()]
    Y = np.concatenate(parts, axis=0)
    gene_var = Y.var(axis=0) + 1e-8
    out = base / 'nmf' / f'fold_{fold_idx}'
    out.mkdir(parents=True, exist_ok=True)
    np.save(str(out / 'gene_var.npy'), gene_var.astype(np.float32))
    print(f"fold_{fold_idx}: n_spots={Y.shape[0]} min={gene_var.min():.4e} max={gene_var.max():.4e}")
PY
```

---

## Lambda selection

```bash
python src/experiments/select_lambda.py --config configs/her2st.yaml
```

Anti-leakage protocol:
- Uses **Fold 1 training data only**, 80/20 internal split.
- Sweeps λ ∈ {0.01, 0.05, 0.10, 0.50}.
- Selects λ maximising Δ PCC(M) with drift guard (Q1 MSE increase < 5%).
- Write-once guard: overwriting `selected_lambda.json` requires `--force`.

Output: `results/lambda_selection/{dataset}/selected_lambda.json`.

**Current frozen value: λ = 0.10** (locked 2026-04-21; constant-λ schedule, no
warmup, no ramp). Historical archive: `results/lambda_005_archive/` contains
λ=0.05 runs for ablation. Override history: `selected_lambda.json.prev.json`
= λ=0.05, `.v1_warmup_ramp_010.json` = original warmup+ramp protocol.

Fold 1 is **reserved exclusively for λ selection** per the anti-leakage
protocol. Never use it for diagnostic or ablation experiments.

---

## Training

### Full Phase 1 orchestrator (overnight)

```bash
# Foreground
bash scripts/phase1_resume.sh

# Detached (survives shell death, own session — recommended for long runs)
setsid nohup bash scripts/phase1_resume.sh > logs/phase1_resume_lambda010.log 2>&1 &
disown
```

Orchestrates LOPCV over 8 patients on 2 GPUs:
- **GPU 0:** TRIPLEX baseline + MCSPR, folds 0–7
- **GPU 1:** ST-Net baseline + MCSPR, folds 0–7, then HisToGene Phase 1.5

Skip guard keys on `training_log.json` (end-of-fold marker) so interrupted
runs can resume safely. Partial `best_model.pt` without a log is treated as
incomplete and re-run.

Estimated wall-clock: ~13 h on 2× RTX 5070 Ti for full Phase 1.

### Individual experiment launches

```bash
# TRIPLEX — baseline, all folds
python src/experiments/run_triplex.py --config configs/her2st.yaml

# TRIPLEX — MCSPR, all folds
python src/experiments/run_triplex_mcspr.py \
    --config configs/her2st.yaml \
    --lambda_path results/lambda_selection/her2st/selected_lambda.json

# ST-Net — baseline / MCSPR
python src/experiments/run_stnet.py --config configs/her2st.yaml
python src/experiments/run_stnet_mcspr.py --config configs/her2st.yaml \
    --lambda_path results/lambda_selection/her2st/selected_lambda.json

# HisToGene — baseline / MCSPR (Phase 1.5, slide-level)
python src/experiments/run_histogene.py --config configs/her2st.yaml
python src/experiments/run_histogene_mcspr.py --config configs/her2st.yaml \
    --lambda_path results/lambda_selection/her2st/selected_lambda.json
```

### Common flags (all run_*.py scripts)

```bash
--fold N             # Run a single fold (0..n_folds-1). Default -1 = all folds.
--output_dir PATH    # Override default output directory (results/{arch}/{dataset}/)
--dry_run            # One batch forward + backward, print loss + grad norms, exit.
```

`run_triplex.py` additionally accepts:

```bash
--use_normalized_mse # Enable L_norm (per-gene variance-normalized MSE)
```

### Single-fold, single-GPU launch (pattern)

```bash
CUDA_VISIBLE_DEVICES=0 python -u src/experiments/run_triplex.py \
    --config configs/her2st.yaml \
    --fold 0 \
    --output_dir results/my_experiment \
    > logs/my_experiment.log 2>&1 &
```

### Dry-run before long training

```bash
CUDA_VISIBLE_DEVICES=0 python -u src/experiments/run_triplex.py \
    --config configs/her2st_diag.yaml \
    --fold 0 \
    --dry_run
```

Prints per-branch loss components, target encoder + fusion head gradient
norms. Use to verify wiring before committing to a multi-hour run.

---

## L_norm diagnostic (per-gene variance-normalized MSE)

L_norm divides each gene's squared residual by its training-fold variance,
making λ dimensionless and giving low-variance genes proportional weight.
Prerequisite: `data/her2st/nmf/fold_{i}/gene_var.npy` exists (see
[Precompute](#precompute-pipeline)).

### Enable via config

```yaml
# configs/her2st.yaml  or  configs/her2st_diag.yaml
training:
  use_normalized_mse: true
```

### Enable via CLI override (run_triplex.py only)

```bash
python src/experiments/run_triplex.py \
    --config configs/her2st.yaml \
    --use_normalized_mse \
    --fold 0 \
    --output_dir results/triplex_lnorm
```

### Paired L_norm vs L_MSE diagnostic on one fold

Fixed 50-epoch budget, early stopping disabled, same seed/fold/architecture.
The only variable is the loss function.

```bash
# Run A — L_MSE baseline
CUDA_VISIBLE_DEVICES=0 python -u src/experiments/run_triplex.py \
    --config configs/her2st_diag.yaml \
    --fold 0 \
    --output_dir results/diag/triplex_lmse \
    > logs/diag_triplex_lmse_fold0.log 2>&1 &

# Run B — L_norm (launch after A completes, or on a separate GPU)
CUDA_VISIBLE_DEVICES=0 python -u src/experiments/run_triplex.py \
    --config configs/her2st_diag.yaml \
    --fold 0 \
    --use_normalized_mse \
    --output_dir results/diag/triplex_lnorm \
    > logs/diag_triplex_lnorm_fold0.log 2>&1 &
```

### Automatic chained execution (A → B → eval)

```bash
setsid nohup bash scripts/diag_sequencer.sh > /dev/null 2>&1 < /dev/null &
disown
```

The sequencer:
1. Polls `logs/diag_triplex_lmse_fold0.log` for the "Run A finished" marker.
2. Launches Run B with `--use_normalized_mse`.
3. Runs `scripts/diagnostic_eval.py` on both fold_dirs.
4. Writes stage markers to `logs/diag_master.log`.

Safety timeout: 5h per run. Own session via setsid — survives shell death.

### Evaluate diagnostic results

```bash
# Per-run PCC(M) / MSE / pooled RVD
python scripts/diagnostic_eval.py \
    --config configs/her2st_diag.yaml \
    --fold 0 \
    --fold_dir results/diag/triplex_lmse/fold_0 \
    --out results/diag/triplex_lmse/fold_0/diagnostic.json

# Read summary
cat results/diag/triplex_lmse/fold_0/diagnostic.json
cat results/diag/triplex_lnorm/fold_0/diagnostic.json
```

---

## Evaluation

### Phase 1 pooled pipeline

```bash
# 1. Cache y_hat / y_true / sample_idx on val set (skips folds with cache)
python scripts/run_inference.py --config configs/her2st.yaml --device cuda
python scripts/run_inference.py --config configs/her2st.yaml --device cuda --force  # recompute

# 2. Pooled metrics + paired MCSPR − baseline deltas
python scripts/eval_pooled_pcc.py

# 3. Per-fold top-K table with RVD + Q1_MSE columns
python scripts/report_phase1_topk.py --config configs/her2st.yaml

# 4. Summary comparison
python src/experiments/compare_results.py --config configs/her2st.yaml
```

Outputs:
- `results/{arch}/her2st/fold_{i}/test_predictions.npz` (y_hat, y_true, sample_idx, batch_sizes)
- `logs/pooled_pcc_results.json` (pooled PCC-10/50/200, RVD, Q1_MSE, per-fold + aggregate + paired Δ)

`eval_pooled_pcc.py` reports **pooled** (all val spots across slides, per-gene
PCC over the pool — matches TRIPLEX paper convention) and **per-slide**
(per-gene PCC within each slide, then averaged — matches training-loop
convention).

### Metric definitions

- **PCC-K**: mean per-gene Pearson correlation over the top-K genes ranked by
  their pooled PCC. K ∈ {10, 50, 200}. Higher is better.
- **RVD** = mean_j [(var_pred_j − var_true_j)² / var_true_j²], masked to
  genes with var_true > 1e-8. Lower is better (predictions should match
  gene variance).
- **Q1 MSE**: mean MSE over the bottom 25% of genes by per-gene MSE (the
  "easy" tail). Lower is better.

---

## Monitoring live runs

```bash
# Phase 1 orchestrator
tail -f logs/phase1_resume_lambda010.log

# Individual fold logs (follow the pattern name in phase1_resume.sh)
tail -f logs/phase1_triplex_mcspr_fold0.log
tail -f logs/phase1_stnet_mcspr_fold5.log

# Diagnostic chain
cat logs/diag_master.log
tail -20 logs/diag_triplex_lmse_fold0.log
tail -20 logs/diag_triplex_lnorm_fold0.log

# Live process table, sorted by uptime
ps -eo pid,etime,stat,wchan:30,cmd --sort=-etime | grep -E "phase1_resume|run_(stnet|triplex|histogene)|diag_sequencer" | grep -v grep

# GPU memory + util snapshot
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

# Find which PID owns GPU memory
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

### If orchestrator dies

```bash
# 1. Inspect last log lines — look for a 'DONE' or the stage it was in
tail -50 logs/phase1_resume_lambda010.log

# 2. Confirm no orphan python children still running
ps -eo pid,etime,cmd | grep -E "run_.*_mcspr|run_(stnet|triplex|histogene)" | grep -v grep

# 3. Re-launch in a fresh session (skip guard will resume from training_log.json)
setsid nohup bash scripts/phase1_resume.sh > logs/phase1_resume_lambda010_resume.log 2>&1 &
disown
```

---

## Result locations

```
results/
  triplex/her2st/fold_{0..7}/          baseline TRIPLEX
    best_model.pt                      best-val-PCC-M checkpoint
    training_log.json                  per-epoch log (end-of-fold marker)
    metrics.json                       final PCC/MSE/MAE on best_model
    test_predictions.npz               y_hat, y_true, sample_idx, batch_sizes
  triplex_mcspr/her2st/fold_{0..7}/    TRIPLEX + MCSPR
  stnet/her2st/fold_{0..7}/            baseline ST-Net
  stnet_mcspr/her2st/fold_{0..7}/
  histogene/her2st/fold_{0..7}/        slide-level HisToGene
  histogene_mcspr/her2st/fold_{0..7}/
  diag/                                L_norm diagnostic outputs
    triplex_lmse/fold_0/
    triplex_lnorm/fold_0/
  lambda_selection/her2st/
    selected_lambda.json               current frozen λ
    selected_lambda.json.prev.json     previous λ (audit)
    topk_summary.json                  historical λ sweep
  lambda_005_archive/                  archived λ=0.05 Phase 1 runs
  quarantine/                          retired protocols (warmup+ramp, etc.)
```

---

## Troubleshooting & common patterns

### Orchestrator keeps dying overnight

Use `setsid nohup` so the process survives shell logout and WSL2 host events:

```bash
setsid nohup bash scripts/phase1_resume.sh > logs/phase1.log 2>&1 < /dev/null &
disown
```

### Inference cache shape mismatch (per-slide PCC doesn't match training)

The cache must store `sample_idx` — per-spot slide assignments used by
TRIPLEX's `evaluate.py`. Older caches without `sample_idx` fall back to
per-batch grouping (correct for ST-Net but wrong for TRIPLEX). Regenerate:

```bash
python scripts/run_inference.py --config configs/her2st.yaml --force
```

### Fold 1 accidentally used for a non-λ experiment

Don't. Fold 1 is reserved for λ selection. Use any other fold (0, 2–7).

### Training says "SKIP fold N — training_log.json exists" but you want a rerun

Delete or move the completed fold's log:

```bash
mv results/triplex/her2st/fold_N/training_log.json \
   results/triplex/her2st/fold_N/training_log.json.prev
```

Phase 1 orchestrator will re-run that fold on next launch.

### GPU 1 shows high memory but 0% util

Likely a python process in D-state (disk-wait) mid-checkpoint-write. Normal
if transient (<30 min). If persistent, check `ps -eo stat,wchan,cmd` for the
wait channel — `ext4_*` or similar suggests slow disk, not deadlock.

### MCSPR loss degrades RVD vs baseline

Expected directional effect on paired deltas — the regulariser trades a
small raw-variance fit loss for correlation structure. Use `eval_pooled_pcc.py`
to check the full metric panel rather than optimising for a single number.
If Δ RVD is large and Δ PCC-K is also negative, re-check `selected_lambda.json`
(current: λ=0.10) and whether the `use_normalized_mse` flag is on.

---

## Key design decisions

- **B = 15 NMF modules.** Reconstruction-R² plateau meets identifiability
  floor; higher ranks introduce rank-deficient modules on small folds.
- **n_contexts = 6.** Soft KMeans on 512-d Ciga embeddings. Cross-slide
  EMA stabilises `C_prior` across the HER2ST cohort.
- **Constant λ protocol (no warmup, no ramp)** from epoch 0. Replay analysis
  of warmup+ramp at λ_max=0.10 showed monotonic val-PCC(M) degradation after
  ramp-in on 5/6 folds: co-training with MSE from the first step beats
  "MSE pretraining → regulariser injection".
- **Current λ = 0.10**, locked 2026-04-21 after λ=0.05 runs produced
  negative pooled PCC-K deltas on 3/3 ST-Net + 2/2 TRIPLEX folds.
  λ=0.05 archived at `results/lambda_005_archive/`.
- **MCSPRLoss internals**: EMA on moments (μ, Σ), never on normalized
  correlations (E[ratio] ≠ ratio of E[]). Adam-style bias correction with
  zero-consistent warm start (`EMA_1 = (1-β)·x_1`), so the correction is
  exact from step 1 even across cross-slide context re-initialization.
  Numerator of `C_hat_t` is current-batch `Sigma_t` (gradient flows);
  denominator is bias-corrected EMA (stable, no gradient). FP32 cast on
  Frobenius loss to survive AMP.
- **SVG-300 panel** (`seurat_v3` HVG ∩ Moran's I rank) replaces the
  previous 250-gene MERGE panel.

---

## Citation & license

```
@inproceedings{mcspr2026,
  title     = {MCSPR: Morphology-Conditioned Spatial Pathway Regularization},
  author    = {(authors pending)},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

MIT license — see `LICENSE`. Cloned HER2ST data retains the upstream
license; Ciga ResNet18 encoder is redistributed under its original terms.
