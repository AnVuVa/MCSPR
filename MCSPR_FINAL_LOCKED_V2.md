# MCSPR — FINAL LOCKED SPECIFICATION v2.0
# NeurIPS 2026 | Both professors signed | Admin authorized
# Date: April 22, 2026
# Status: IMPLEMENTATION READY — Claude Code executes exactly this

---

## AMENDMENT LOG (v1 → v2)

```
A1: loss.py Phase B — EMA init: direct assignment → zero-consistent warm start
    self.ema_Sigma[t] = (1-β)·Sigma_batch_t  (NOT Sigma_batch_t directly)
    Reason: bias correction formula valid only for zero-initialized EMA

A2: loss.py Phase B — EMA update: Sigma_batch_t.detach() → Sigma_batch_t (no detach)
    Gradient flows through (1-β)·Sigma_batch_t into ema_Sigma[t]
    Historical state β·ema_Sigma[t] remains detached
    Reason: gradient must reach Y_hat via EMA numerator coefficient

A3: loss.py Phase E — C_hat_t numerator: Sigma_batch_t → Sigma_unbiased
    C_hat_t = Sigma_unbiased / denom  (both numerator and denominator from EMA)
    Diagonal: Σ_ii/(Σ_ii + τ) ≈ 1.0 — mathematically valid correlation matrix
    Reason: diagonal disconnect — mixing batch numerator with EMA denominator
    produced floating diagonal, contaminated Frobenius objective

A4: loss.py Phase F — B² normalization added
    loss_t = (1/(eff_n · B · B)) · sum((C_hat_t - C_prior_t)²)
    Reason: Frobenius sum scales O(B²); normalization makes λ dimension-invariant

A5: loss.py Phase G — gradient restoration scalar added
    L_MCSPR = L_MCSPR / (1.0 - beta)
    Reason: (1-β) coefficient in EMA squashes effective gradient; constant
    scalar preferred over exact bias_correction/(1-β) for optimizer stability

A6: loss.py Phase G — n_active_contexts → self.n_contexts (T=6 fixed)
    Reason: dividing by n_active penalizes broad context activation; 
    must be consistent λ regardless of how many contexts fire per batch

A7: universal_trainer.py — TRIPLEX mcspr_input: preds['fusion'] → preds['output']
    Both STNet and TRIPLEX attach MCSPR to 300-dim prediction output
    Reason: NMF prior defined in gene expression space R^300;
    attaching to fusion latent space is dimensional violation

A8: precompute.py — gene_var source: counts_spcs → counts_svg
    Reason: model trains on counts_svg (300 genes); normalizer must match

A9: precompute.py — gene_var path: gene_var_fold{i}.npy → gene_var.npy
    Path: data/{dataset}/nmf/fold_{i}/gene_var.npy
    Reason: fold index already encoded in directory name
```

---

## GLOBAL INVARIANTS

```
GENE SPACE:         m = 300   (counts_svg, SVG-filtered panel)
LATENT DIM:         B = TBD   (pending R² sweep — must achieve R² ≥ 0.60)
N_CONTEXTS:         T = 6     (fixed, all architectures, all datasets)
K_MIN:              30        (per-batch minimum effective spots per context)
SEED:               2021      (all splits, NMF, KMeans)
NORMALIZATION:      log1p(count / sum(count)) on counts_svg targets
CV SCHEME:          Patient-level LOPCV, 4 folds (HER2ST: 8 patients, 2/fold)
LAMBDA FOLD:        Fold 1 ONLY — never reported as test results
ARCHITECTURES:      STNet (ConvNet), TRIPLEX (cross-attention)
MCSPR ATTACHMENT:   preds['output'] for BOTH architectures — locked
BATCH_SIZE_STNET:   256 (confirmed safe: ~6GB on 16GB GPU)
BATCH_SIZE_TRIPLEX: TBD (pending VRAM check at 256)
NMF_R2_THRESHOLD:   0.60 (hard gate — pipeline refuses to continue below)
PRIMARY LOSS:       L_norm (NormalizedMSELoss) — use_normalized_mse: true
LAMBDA_MAX:         TBD (pending select_lambda.py on Fold 1 after n_modules fixed)
EMA_BETA:           0.9
EMA_TAU:            1e-4
WARMUP_EPOCHS:      5
RAMP_EPOCHS:        10
EARLY_STOP:         patience=20 on val PCC(M)
GRAD_CLIP:          max_norm=1.0
RESULTS_DIR:        results/v2/   (fresh — v1 results archived, not deleted)
DELTA_THRESHOLD:    ΔPCC ≥ 0.02 for "improves accuracy" claim
                    If ΔPCC < 0.02 but ΔRVD ≥ 30%: pivot to fidelity claim
```

---

## FILE 1 — mcspr/prior/construction.py

```
CONSTANTS:
  NMF_R2_THRESHOLD = 0.60
  NMF_MAX_ITER     = 500
  NMF_INIT         = 'nndsvda'
  SEED             = 2021
  EPS              = 1e-8

─────────────────────────────────────────────────────────────────
fit_nmf(Y_train_nonneg, n_modules) → (M, M_pinv, r2)
─────────────────────────────────────────────────────────────────
  # Y_train_nonneg: (N, 300) clipped to [0,∞) — log-normalized counts_svg
  # Returns M: (300, B), M_pinv: (B, 300)

  model ← NMF(n_components=n_modules, init=NMF_INIT,
               random_state=SEED, max_iter=NMF_MAX_ITER)
  W ← model.fit_transform(Y_train_nonneg)           # (N, B)
  M ← model.components_.T                           # (300, B)
  M_pinv ← pinv(M)                                  # (B, 300)

  Y_hat  ← W @ M.T
  r2     ← 1 - sum((Y_train_nonneg - Y_hat)²) / sum((Y_train_nonneg - mean)²)

  ASSERT r2 ≥ NMF_R2_THRESHOLD:
    raise ValueError(f"NMF R²={r2:.4f} < {NMF_R2_THRESHOLD}. "
                     f"Increase n_modules. Got B={n_modules}.")

  return M, M_pinv, r2

─────────────────────────────────────────────────────────────────
compute_context_priors(Y_train, M_pinv, context_weights,
                       n_contexts, k_min=30) → (C_prior, stats)
─────────────────────────────────────────────────────────────────
  # EMPIRICAL prior — NOT parametric M·diag(σ²)·Mᵀ + σ²ε·I
  # Project training spots to latent space, compute weighted correlation

  Z_train ← Y_train @ M_pinv.T                      # (N, B)
  C_prior ← zeros(n_contexts, B, B)

  for t in 0…n_contexts-1:
      w_t   ← context_weights[:, t]                 # (N,) soft weights
      eff_n ← sum(w_t)

      ASSERT eff_n ≥ k_min:
        # Full-dataset check only — batch-level gating handled in loss.py
        raise ValueError(f"Context {t} starved: eff_n={eff_n:.1f}")

      w_norm ← w_t / eff_n
      μ_t    ← sum_i(w_norm_i · Z_train[i])         # (B,)
      Z_c    ← Z_train - μ_t                         # (N, B)
      Σ_t    ← (Z_c * w_norm[:,None]).T @ Z_c        # (B, B) weighted cov

      std_t  ← sqrt(diag(Σ_t) + EPS)                # (B,)
      C_t    ← Σ_t / outer(std_t, std_t)             # (B, B) correlation
      C_t    ← clip(C_t, -1.0, 1.0)
      fill_diagonal(C_t, 1.0)

      C_prior[t] ← C_t

  return C_prior                                     # (T, B, B) float32
```

---

## FILE 2 — src/data/precompute.py

```
─────────────────────────────────────────────────────────────────
precompute_all(data_dir, dataset, n_folds, n_modules, n_contexts)
─────────────────────────────────────────────────────────────────
  folds ← build_lopcv_folds(data_dir, dataset, n_folds, seed=2021)

  for fold_idx, fold in enumerate(folds):
      train_samples ← fold['train_samples']
      nmf_dir       ← data_dir / "nmf" / f"fold_{fold_idx}"
      nmf_dir.mkdir(parents=True, exist_ok=True)

      # 1. Global image features + KMeans context assignment
      precompute_global_features(data_dir, train_samples)
      precompute_kmeans(data_dir, train_samples, n_contexts, seed=2021)

      # 2. Load training expression — counts_svg (300 genes, log-normalized)
      Y_parts ← []
      for s in train_samples:
          path ← data_dir / "counts_svg" / f"{s}.npy"
          ASSERT path.exists()
          Y_parts.append(load(path).astype(float32))
      Y_train ← vstack(Y_parts)                      # (N_train, 300)

      # 3. Fit NMF — hard R² gate inside fit_nmf()
      Y_nn   ← clip(Y_train, 0, None)
      M, M_pinv, r2 ← fit_nmf(Y_nn, n_modules)       # raises if R²<0.60
      save(nmf_dir / "M.npy",      M)                 # (300, B)
      save(nmf_dir / "M_pinv.npy", M_pinv)            # (B, 300)
      save(nmf_dir / "r2.txt",     r2)

      # 4. Compute context priors
      ctx_weights ← load_context_weights(train_samples, fold_idx)
      C_prior     ← compute_context_priors(
                        Y_train, M_pinv, ctx_weights, n_contexts)
      save(nmf_dir / "C_prior.npy", C_prior)          # (T, B, B)

      # 5. Per-gene variance — SAME Y_train, SAME fold, ZERO leakage
      # Source: counts_svg (300 genes, log-normalized) — matches model targets
      gene_var ← Y_train.var(axis=0) + 1e-8           # (300,) float32
      ASSERT min(gene_var) > 0
      save(nmf_dir / "gene_var.npy", gene_var.astype(float32))
      log(f"Fold {fold_idx}: R²={r2:.4f} "
          f"gene_var min={min(gene_var):.3e} max={max(gene_var):.3e}")
```

---

## FILE 3 — mcspr/core/loss.py  ★ FOUR AMENDMENTS LOCKED ★

```
─────────────────────────────────────────────────────────────────
CLASS MCSPRLoss(nn.Module)
─────────────────────────────────────────────────────────────────

  __init__(C_prior, M_pinv, n_contexts, lambda_max, tau, beta, k_min):
    register_buffer("C_prior",          tensor(C_prior, float32)) # (T,B,B)
    register_buffer("M_pinv",           tensor(M_pinv,  float32)) # (B,300)
    register_buffer("ema_mu",           zeros(T, B))
    register_buffer("ema_Sigma",        zeros(T, B, B))
    register_buffer("ema_sigma_sq",     zeros(T, B))
    register_buffer("ema_initialized",  zeros(T, bool))
    register_buffer("ema_step",         zeros(T, long))    # checkpoint-safe

    self.n_contexts = T                # fixed T=6, never dynamic
    self.lambda_max = lambda_max
    self.tau        = tau              # 1e-4
    self.beta       = beta             # 0.9
    self.k_min      = k_min           # 30

  ─────────────────────────────────────────────────────────
  forward(Y_hat, context_weights, lambda_scale) → L_MCSPR
  ─────────────────────────────────────────────────────────
    # Y_hat:           (N, 300) model predictions — HAS grad
    # context_weights: (N, T)   precomputed soft weights — frozen, no grad
    # lambda_scale:    scalar ∈ [0,1] from LambdaScheduler

    # Project to NMF latent space — gradient flows through Y_hat
    Z_hat ← Y_hat @ self.M_pinv.T                      # (N, B)
    B     ← self.M_pinv.shape[0]                       # latent dim

    total_loss ← 0.0
    n_active   ← 0

    for t in 0…self.n_contexts-1:

      # A. Per-context soft-weighted batch covariance
      w_t   ← context_weights[:, t]                   # (N,) frozen
      eff_n ← sum(w_t).item()

      if eff_n < self.k_min:
          continue                                     # starved — skip

      w_norm     ← w_t / eff_n
      mu_batch   ← (w_norm[:,None] * Z_hat).sum(0)    # (B,)
      z_c        ← Z_hat - mu_batch                   # (N,B) centered
      Sigma_batch_t ← (z_c * w_norm[:,None]).T @ z_c  # (B,B) HAS grad

      # B. EMA update — ★ AMENDMENT A1 + A2 ★
      with torch.no_grad():
          if NOT self.ema_initialized[t]:
              # Zero-consistent warm start (NOT direct assignment)
              # Ensures bias_correction formula valid from step 1
              self.ema_Sigma[t]       ← (1.0 - self.beta) * Sigma_batch_t.detach()
              self.ema_mu[t]          ← (1.0 - self.beta) * mu_batch.detach()
              self.ema_initialized[t] ← True
          else:
              # Historical state DETACHED — severs temporal gradient chain
              # Current batch NOT detached — gradient flows via (1-β) coefficient
              self.ema_Sigma[t] ← (self.beta  * self.ema_Sigma[t].detach()
                                 + (1.0 - self.beta) * Sigma_batch_t.detach())
              self.ema_mu[t]    ← (self.beta  * self.ema_mu[t].detach()
                                 + (1.0 - self.beta) * mu_batch.detach())
          self.ema_step[t] += 1
          self.ema_sigma_sq[t] ← diag(self.ema_Sigma[t])

      # WAIT — gradient cannot flow through no_grad block above.
      # Recompute ema_Sigma with gradient for loss computation only:
      if self.ema_initialized[t]:
          ema_Sigma_with_grad ← (self.beta * self.ema_Sigma[t].detach()
                               + (1.0 - self.beta) * Sigma_batch_t)
      else:
          ema_Sigma_with_grad ← (1.0 - self.beta) * Sigma_batch_t

      # C. Bias correction
      bias_correction  ← 1.0 - self.beta ** self.ema_step[t].item()
      Sigma_unbiased   ← ema_Sigma_with_grad / bias_correction   # HAS grad

      # D. Denominator — EMA std, DETACHED (stable, no gradient through denom)
      # tau appears ONLY here — no double-tau on outer product
      std_a ← sqrt(Sigma_unbiased.diag().detach() + self.tau)    # (B,) no grad
      denom ← std_a.unsqueeze(1) * std_a.unsqueeze(0)            # (B,B) no grad

      # E. Predicted correlation matrix — ★ AMENDMENT A3 ★
      # Numerator = Sigma_unbiased (same distribution as denominator)
      # Diagonal = Σ_ii/(Σ_ii+τ) ≈ 1.0 — valid correlation matrix
      # Gradient flows via (1-β) in ema_Sigma_with_grad above
      C_hat_t ← Sigma_unbiased / denom                            # (B,B) HAS grad
      C_hat_t ← clamp(C_hat_t, -1.0, 1.0)                        # fp16 safety

      # F. Frobenius loss — ★ AMENDMENT A4 ★
      # Divide by B² → mean squared correlation residual
      # Scale invariant to latent dimension — λ universality preserved
      loss_t ← (1.0 / (eff_n * B * B)) * sum(
                    (C_hat_t.float() - self.C_prior[t].float()) ** 2
                )

      total_loss += loss_t
      n_active   += 1

    if n_active == 0:
        return tensor(0.0)                                         # no grad

    # G. Scale — ★ AMENDMENTS A5 + A6 ★
    # Normalize by FIXED self.n_contexts (T=6), NOT n_active
    #   → consistent λ contribution per context per step
    # Divide by (1-β)
    #   → restores gradient magnitude squashed by EMA coefficient
    #   → stationary scalar preferred over exact bias_correction/(1-β)
    #     for Adam optimizer stability
    L_MCSPR ← (self.lambda_max * lambda_scale) * total_loss / self.n_contexts
    L_MCSPR ← L_MCSPR / (1.0 - self.beta)

    return L_MCSPR

# ─── GRADIENT FLOW DIAGRAM ───────────────────────────────────────
#
#  Y_hat (N,300)
#    │
#    ▼  @ M_pinv.T
#  Z_hat (N,B)
#    │
#    ▼  weighted_cov
#  Sigma_batch_t (B,B) ←── HAS GRADIENT
#    │
#    ▼  × (1-β)
#  ema_Sigma_with_grad (B,B) ←── gradient scaled by (1-β)
#    │
#    ▼  / bias_correction
#  Sigma_unbiased (B,B)
#    │                    │
#    ▼                    ▼  .diag().detach()
#  numerator           std_a (B,) ← NO GRADIENT
#  (B,B)                   │
#    │               outer product
#    │                   denom (B,B) ← NO GRADIENT
#    │                    │
#    └──── / denom ────────┘
#             │
#           C_hat_t (B,B) ← diagonal ≈ 1.0
#             │
#   Frobenius ‖C_hat_t - C_prior‖² / (eff_n·B·B)
#             │
#         × λ / T / (1-β)
#             │
#          L_MCSPR → backward() → Y_hat.grad
```

---

## FILE 4 — mcspr/core/scheduler.py

```
CLASS LambdaScheduler:
  __init__(warmup_epochs=5, ramp_epochs=10)

  get_scale(epoch) → float:
    if epoch < 5:         return 0.0
    elif epoch < 15:      return (epoch - 5) / 10.0
    else:                 return 1.0
```

---

## FILE 5 — src/losses/normalized_mse.py  ★ NEW FILE ★

```
CLASS NormalizedMSELoss(nn.Module):

  __init__(gene_var: ndarray(300,)):
    register_buffer('gene_var', tensor(gene_var, float32))  # (300,)
    # Source: counts_svg log-normalized, training fold only, no test leakage
    # Dynamic range 30–65× across folds — equalizes gradient across gene panel

  forward(y_hat: Tensor(N,300), y_true: Tensor(N,300)) → scalar:
    residuals ← y_hat - y_true                              # (N, 300)
    return mean((residuals ** 2) / self.gene_var)           # scalar
    # Equivalent to: weighted average MSE where weight ∝ 1/Var(gene)
    # Low-variance spatial marker genes: high weight → model attends to them
    # High-variance structural genes: low weight → not dominated by HER2 etc.
```

---

## FILE 6 — src/training/universal_trainer.py  (training step)

```
__init__(model, config, fold_idx, device):

  # Reconstruction loss
  if config.use_normalized_mse:
      gene_var ← load(f"data/{dataset}/nmf/fold_{fold_idx}/gene_var.npy")
      self.recon_loss ← NormalizedMSELoss(gene_var).to(device)
  else:
      self.recon_loss ← nn.MSELoss()

  # MCSPR loss
  if config.use_mcspr:
      C_prior ← load(f"data/{dataset}/nmf/fold_{fold_idx}/C_prior.npy")
      M_pinv  ← load(f"data/{dataset}/nmf/fold_{fold_idx}/M_pinv.npy")
      self.mcspr_loss ← MCSPRLoss(C_prior, M_pinv,
                                   n_contexts=T,
                                   lambda_max=config.lambda_max,
                                   tau=1e-4, beta=0.9, k_min=30).to(device)
      self.lambda_sched ← LambdaScheduler(warmup=5, ramp=10)

─────────────────────────────────────────────────────────────────
_training_step(batch, epoch) → (L_total, l_recon, l_mcspr)
─────────────────────────────────────────────────────────────────
  X        ← batch['patches'].to(device)        # (N, 3, 224, 224)
  y_true   ← batch['expression'].to(device)     # (N, 300)
  ctx_w    ← batch['ctx_weights'].to(device)    # (N, T) frozen

  # Forward
  if architecture == 'stnet':
      Y_hat       ← model(X)                    # Tensor (N, 300)
      mcspr_input ← Y_hat                       # same tensor

  elif architecture == 'triplex':
      preds       ← model(X, ...)               # dict
      Y_hat       ← preds['output']             # (N, 300) ★ AMENDMENT A7 ★
      mcspr_input ← preds['output']             # SAME — not preds['fusion']
      # Both architectures use prediction space — consistent with NMF prior domain

  L_recon ← self.recon_loss(Y_hat, y_true)

  if self.mcspr_loss is not None:
      scale   ← self.lambda_sched.get_scale(epoch)
      L_mcspr ← self.mcspr_loss(mcspr_input, ctx_w, scale)
  else:
      L_mcspr ← 0.0

  L_total ← L_recon + L_mcspr
  return L_total, L_recon.item(), float(L_mcspr)
```

---

## FILE 7 — src/training/evaluate.py

```
evaluate(model, loader, device) → metrics:
  # Per-slide, per-gene PCC — NEVER pool spots across slides

  for each slide s in loader:
      Y_hat_s, Y_true_s ← predict_slide(s)      # (N_s, 300)

      for j in 0…299:
          if std(Y_hat_s[:,j]) < 1e-10 or std(Y_true_s[:,j]) < 1e-10:
              pcc[j] ← 0.0                        # degenerate → 0, not NaN
          else:
              pcc[j] ← pearsonr(Y_hat_s[:,j], Y_true_s[:,j])

      var_pred ← var(Y_hat_s, axis=0)            # (300,)
      var_true ← var(Y_true_s, axis=0)           # (300,)
      rvd_s    ← mean((var_pred-var_true)² / (var_true²+1e-8))

      record pcc, rvd_s, mse_s, mae_s per slide

  pcc_matrix ← vstack(per_slide_pcc)             # (n_slides, 300)
  pcc_m      ← mean(pcc_matrix)
  pcc_10     ← mean(top_10_genes_by_pcc)
  pcc_50     ← mean(top_50_genes_by_pcc)

  return {pcc_m, pcc_10, pcc_50, rvd, mse, mae}
```

---

## FILE 8 — src/experiments/select_lambda.py

```
select_lambda():
  ASSERT not exists('selected_lambda.json')      # write-once guard

  FOLD       ← 1                                 # ONLY fold 1 — immutable
  GRID       ← [0.01, 0.05, 0.1, 0.5]
  DRIFT_MAX  ← 0.05                              # Q1 MSE drift threshold

  # 80/20 internal split of fold 1 training data
  internal_train, internal_val ← split(fold1_train, 0.80, seed=2021)

  # Baseline: L_norm only, λ=0
  baseline ← train_eval(lambda_max=0.0, use_mcspr=False, epochs=50)
  q1_base  ← baseline['q1_mse']

  for λ in GRID:
      result ← train_eval(lambda_max=λ, use_mcspr=True, epochs=50)
      drift  ← (result['q1_mse'] - q1_base) / q1_base

      if drift > DRIFT_MAX:
          log(f"λ={λ} DISQUALIFIED drift={drift:.3f}")
          continue

      record λ → {pcc_m, rvd, drift}

  best_λ ← argmax(valid_results, key='pcc_m')
  save_json('selected_lambda.json',
            {lambda: best_λ, fold: 1, timestamp: now()})
  return best_λ
  # best_λ frozen for ALL architectures, ALL folds, ALL datasets
```

---

## FILE 9 — configs/her2st.yaml

```yaml
dataset:            her2st
data_dir:           data/her2st
n_genes:            300
n_folds:            4
seed:               2021
architecture:       triplex
batch_size:         TBD        # pending VRAM check — target 256
n_epochs:           200
early_stop:         20
lr:                 1e-4
weight_decay:       1e-5
grad_clip:          1.0
use_normalized_mse: true
use_mcspr:          true
n_contexts:         6
k_min:              30
n_modules:          TBD        # pending R² sweep
lambda_max:         TBD        # pending select_lambda.py
tau:                0.0001
beta:               0.9
warmup_epochs:      5
ramp_epochs:        10
results_dir:        results/v2/triplex
```

---

## FILE 10 — configs/stnet.yaml

```yaml
dataset:            her2st
data_dir:           data/her2st
n_genes:            300
n_folds:            4
seed:               2021
architecture:       stnet
batch_size:         256        # confirmed safe
n_epochs:           200
early_stop:         20
lr:                 1e-4
weight_decay:       1e-5
grad_clip:          1.0
use_normalized_mse: true
use_mcspr:          true
n_contexts:         6
k_min:              30
n_modules:          TBD        # same as TRIPLEX — universal claim requires this
lambda_max:         TBD        # same frozen λ as TRIPLEX
tau:                0.0001
beta:               0.9
warmup_epochs:      5
ramp_epochs:        10
results_dir:        results/v2/stnet
```

---

## EXECUTION SEQUENCE (strict order — no step skipped)

```
PHASE 0 — NMF R² SWEEP (blocking all else)
  Script: inline python on fold 0 counts_svg
  Goal: find smallest n_modules with R² ≥ 0.60
  Output: n_modules value → update both configs

PHASE 1 — PRECOMPUTE (after n_modules confirmed)
  Script: python src/data/precompute.py --n_modules <n> --n_folds 4
  Output: nmf/fold_{0..3}/ with M, M_pinv, C_prior, gene_var, r2
  Verify: all 4 folds R² ≥ 0.60, gene_var shape=(300,), min>0

PHASE 2 — LOSS.PY AMENDMENTS (independent of n_modules — do in parallel with sweep)
  Apply all 4 amendments (A1-A6)
  Smoke test: forward pass, backward(), Y_hat.grad non-zero non-NaN

PHASE 3 — TRIPLEX VRAM CHECK
  1-epoch dry run batch=256 fold 0, report peak VRAM
  If ≤ 14GB: set batch_size=256 in her2st.yaml
  If > 14GB: set largest safe value ≥ 180

PHASE 4 — LAMBDA SELECTION (after phases 1-3 complete)
  Script: python src/experiments/select_lambda.py
  Fold 1 only, 80/20 internal split, grid {0.01, 0.05, 0.1, 0.5}
  Output: selected_lambda.json → update both configs

PHASE 5 — FULL TRAINING RUNS (after lambda frozen)
  GPU 0: TRIPLEX baseline → TRIPLEX+MCSPR (sequential, folds 0-3)
  GPU 1: STNet baseline → STNet+MCSPR (sequential, folds 0-3)
  All results → results/v2/
  Archive v1 results: mv results/v1_archive/ (do not delete)

PHASE 6 — EVALUATION
  Script: python src/experiments/compare_results.py --results_dir results/v2/
  Output: Δ_PCC(M), Δ_PCC-10, Δ_PCC-50, Δ_RVD, Δ_MAE, Δ_MSE
          Paired t-test per metric across 4 folds
          Per-fold breakdown table
```

---

## WHAT DOES NOT EXIST IN THIS PAPER

```
× Supplementary table with top-250-by-mean gene set
× SMCS in any main comparison table (Section 4.3 only)
× Per-architecture λ tuning (one frozen λ — universality claim)
× Results from fold 1 reported as test results
× Numbers from another paper's table without unified re-evaluation
× STEM without standardization + clamp + schedule verification
× n_active_contexts in loss denominator
× Direct EMA initialization (Sigma_t, not (1-β)·Sigma_t)
× Sigma_batch_t in C_hat_t numerator (must be Sigma_unbiased)
× C_prior = M·diag(σ²)·Mᵀ + σ²ε·I (empirical form only)
× NMF basis with R² < 0.60
× MCSPR attached to preds['fusion'] (must be preds['output'])
× gene_var from counts_spcs (must be counts_svg)
× Frobenius sum without B² normalization
× 1/(1-β) gradient restoration omitted
```

---

## OPEN VARIABLES — RESOLVE IN PHASE 0-4

```
n_modules     → Phase 0 (R² sweep)
batch_size    → Phase 3 (TRIPLEX VRAM check)
lambda_max    → Phase 4 (select_lambda.py)
```

---
*v2.0 — All amendments from Prof1 + Prof2 review incorporated*
*Signed: Prof1 ✓  Prof2 ✓  Admin pending*
