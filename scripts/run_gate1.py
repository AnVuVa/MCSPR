"""
Gate 1 runner — uses almaan/her2st full transcriptome.
Calls the existing gene_selection_analysis logic with correct data paths.

seurat_v3 HVG on raw integer counts, Moran's I on log-normalized full transcriptome.
"""
import json
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp
from collections import defaultdict

# -- Config --
DATA_DIR     = Path("data/her2st")
RAW_DIR      = Path("data/her2st/umi_counts")       # full ~12k gene raw counts
GENE_LIST    = Path("data/her2st/features_full/gene_names.json")
OUTPUT_DIR   = Path("results/pre_training_gates/gene_selection")
N_HMHVG      = 300
N_MEAN        = 250
FOLD_IDX      = 0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Patient-level LOPCV fold 0 --
feature_dir  = DATA_DIR / "features"
sample_names = sorted([f.stem for f in feature_dir.glob("*.csv")])
print(f"Total samples: {len(sample_names)}")

patient_map = defaultdict(list)
for s in sample_names:
    # MERGE uses bare names: A1, B2, etc. Patient = first letter.
    pid = s[0].upper()
    patient_map[pid].append(s)
patients = sorted(patient_map.keys())
held = patients[FOLD_IDX]
test_samples  = patient_map[held]
train_samples = [s for p in patients if p != held for s in patient_map[p]]
print(f"Fold {FOLD_IDX}: held={held}, train={len(train_samples)}, "
      f"test={len(test_samples)}")

# -- Load gene names --
with open(GENE_LIST) as f:
    gene_names = json.load(f)
print(f"Full gene set: {len(gene_names)} genes")

# -- Load raw counts (training only) --
print("Loading raw integer counts...")
raw_arrays = []
for s in train_samples:
    p = RAW_DIR / f"{s}.npy"
    if p.exists():
        raw_arrays.append(np.load(p))
Y_raw = np.concatenate(raw_arrays, axis=0).astype(np.float32)
print(f"Raw count matrix: {Y_raw.shape}")

# Integer check -- MANDATORY before seurat_v3
frac_nonint = (~np.isclose(Y_raw, Y_raw.astype(np.int32), atol=0.01)).mean()
assert frac_nonint < 0.01, f"Non-integer data passed to seurat_v3: {frac_nonint:.1%}"
print(f"Integer check: PASS ({frac_nonint:.4%} non-integer)")

# -- seurat_v3 HVG on raw counts --
print("Running seurat_v3 HVG selection...")
adata = sc.AnnData(X=Y_raw)
adata.var_names = gene_names
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=N_HMHVG)

hvg_df     = adata.var[['highly_variable', 'means', 'variances', 'variances_norm']]
hmhvg_g    = hvg_df[hvg_df['highly_variable']].index.tolist()
top_mean_g = [gene_names[i] for i in np.argsort(Y_raw.mean(0))[::-1][:N_MEAN]]

gene_idx      = {g: i for i, g in enumerate(gene_names)}
hmhvg_idx     = [gene_idx[g] for g in hmhvg_g if g in gene_idx]
top_mean_idx  = [gene_idx[g] for g in top_mean_g if g in gene_idx]
overlap       = len(set(hmhvg_g) & set(top_mean_g))

print(f"HMHVG: {len(hmhvg_g)} genes | Top-{N_MEAN}-by-mean: {len(top_mean_g)} genes")
print(f"Overlap: {overlap} ({overlap / N_HMHVG:.1%})")

# -- Moran's I on log-normalized full transcriptome (training slides only) --
print("\nComputing Moran's I on log-normalized counts...")
mi_hmhvg, mi_top_mean = [], []

for sample in train_samples:
    coord_p = DATA_DIR / "tissue_positions" / f"{sample}.csv"
    raw_p   = RAW_DIR / f"{sample}.npy"
    if not coord_p.exists() or not raw_p.exists():
        continue

    # Compute log1p(count/sum * 1e4) from raw counts for Moran's I
    Y_full = np.load(raw_p).astype(np.float32)
    library_size = Y_full.sum(axis=1, keepdims=True)
    Y_log = np.log1p(Y_full / (library_size + 1e-8) * 1e4)

    pos_df = pd.read_csv(coord_p, index_col=0)
    if "array_row" in pos_df.columns:
        coords = pos_df[["array_row", "array_col"]].values
    else:
        coords = pos_df.iloc[:, -2:].values

    if Y_log.shape[0] != len(coords):
        print(f"  WARN: {sample} shape mismatch, skipping")
        continue

    N = Y_log.shape[0]
    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)

    # Build spatial weight matrix (Chebyshev distance <= 1)
    W = np.zeros((N, N), np.float32)
    for i in range(N):
        dr = np.abs(rows - rows[i])
        dc = np.abs(cols - cols[i])
        nb = (np.maximum(dr, dc) <= 1) & (np.arange(N) != i)
        W[i, nb] = 1.0
    rs = W.sum(1, keepdims=True)
    rs[rs == 0] = 1
    W /= rs
    S0 = W.sum()

    # Moran's I per gene
    Z = Y_log - Y_log.mean(0)
    WZ = W @ Z
    num = (Z * WZ).sum(0)
    den = (Z * Z).sum(0)
    den = np.where(den > 1e-8, den, 1e-8)
    mi = (N / S0) * (num / den)

    mi_hmhvg.extend(mi[hmhvg_idx].tolist())
    mi_top_mean.extend(mi[top_mean_idx].tolist())
    print(f"  {sample}: {N} spots, mi_hmhvg_mean={mi[hmhvg_idx].mean():.4f}, "
          f"mi_topmean_mean={mi[top_mean_idx].mean():.4f}")

mi_h = np.array(mi_hmhvg)
mi_m = np.array(mi_top_mean)

mw_stat, mw_p = mannwhitneyu(mi_h, mi_m, alternative='greater')
ks_stat, ks_p = ks_2samp(mi_h, mi_m)
gate1 = mw_p < 0.05

# -- Output --
print("\n" + "=" * 60)
print("GATE 1 RESULTS")
print("=" * 60)
print(f"HMHVG    Moran's I: mean={mi_h.mean():.4f}  median={np.median(mi_h):.4f}")
print(f"Top-mean Moran's I: mean={mi_m.mean():.4f}  median={np.median(mi_m):.4f}")
print(f"Mann-Whitney U (one-sided, HMHVG > top-mean):")
print(f"  stat = {mw_stat:.1f}")
print(f"  p    = {mw_p:.4e}")
print(f"KS test (two-sided):")
print(f"  stat = {ks_stat:.4f}")
print(f"  p    = {ks_p:.4e}")
print(f"Overlap (HMHVG ^ top-250-by-mean): {overlap}/{N_HMHVG} = {overlap / N_HMHVG:.1%}")
print(f"GATE 1: {'PASS' if gate1 else 'FAIL'} (threshold: p < 0.05)")
print("=" * 60)

summary = {
    "fold": FOLD_IDX,
    "n_train_samples": len(train_samples),
    "n_full_genes": len(gene_names),
    "n_hmhvg": len(hmhvg_g),
    "n_top_mean": len(top_mean_g),
    "overlap_count": overlap,
    "overlap_pct": float(overlap / N_HMHVG),
    "morans_i_hmhvg_mean": float(mi_h.mean()),
    "morans_i_hmhvg_median": float(np.median(mi_h)),
    "morans_i_topmean_mean": float(mi_m.mean()),
    "morans_i_topmean_median": float(np.median(mi_m)),
    "mann_whitney_stat": float(mw_stat),
    "mann_whitney_p": float(mw_p),
    "ks_statistic": float(ks_stat),
    "ks_p": float(ks_p),
    "gene_selection_gate": "PASS" if gate1 else "FAIL",
    "hmhvg_genes": hmhvg_g,
}
with open(OUTPUT_DIR / "gene_selection_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
with open(OUTPUT_DIR / "hmhvg_genes.json", "w") as f:
    json.dump(hmhvg_g, f, indent=2)
print(f"\nResults saved to {OUTPUT_DIR}")
