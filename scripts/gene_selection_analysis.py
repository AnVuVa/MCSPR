"""Gene selection analysis — HER2ST Fold 0.

Two data matrices required:
  Y_raw:  umi_counts/*.npy  — raw integer UMI counts -> seurat_v3
  Y_log:  counts_spcs/*.npy — log-normalized floats  -> Moran's I

seurat_v3 REQUIRES raw integer counts. Passing log-normalized data
produces either a crash or silent integer coercion — both are wrong.
"""

import json
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR   = Path("data/her2st")
OUTPUT_DIR = Path("results/pre_training_gates/gene_selection")
N_HMHVG    = 300
N_MEAN     = 250
FOLD_IDX   = 0


def get_patient_id(sample_name: str) -> str:
    """A1 -> A, B2 -> B, etc. (MERGE bare naming convention)."""
    return sample_name[0].upper()


def build_lopcv_fold(sample_names, fold_idx):
    """Patient-level LOPCV for HER2ST (8 patients)."""
    from collections import defaultdict

    patient_map = defaultdict(list)
    for s in sample_names:
        patient_map[get_patient_id(s)].append(s)
    patients = sorted(patient_map.keys())
    held = patients[fold_idx]
    test = patient_map[held]
    train = [s for p in patients if p != held for s in patient_map[p]]
    return train, test


def load_matrix(directory: Path, samples: List[str],
                label: str) -> np.ndarray:
    arrays = []
    for s in samples:
        p = directory / f"{s}.npy"
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")
        arrays.append(np.load(p))
    return np.concatenate(arrays, axis=0)


def assert_integer_counts(Y: np.ndarray, name: str):
    """Crash loudly if non-integer data is passed to seurat_v3."""
    assert (Y >= 0).all(), f"{name}: negative values found"
    frac_nonint = (~np.isclose(Y, Y.astype(np.int32), atol=0.01)).mean()
    if frac_nonint > 0.01:
        raise ValueError(
            f"{name}: {frac_nonint:.1%} non-integer values detected.\n"
            f"seurat_v3 requires raw integer UMI counts, not log-normalized data.\n"
            f"Check that you are loading from umi_counts/, not counts_spcs/."
        )


def compute_hmhvg(Y_raw, gene_names, n_top=300):
    assert_integer_counts(Y_raw, "Y_raw")
    adata = sc.AnnData(X=Y_raw.astype(np.float32))
    adata.var_names = gene_names
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3',
                                    n_top_genes=n_top)
    hvg_df = adata.var[['highly_variable', 'means',
                        'variances', 'variances_norm']].copy()
    return hvg_df[hvg_df['highly_variable']].index.tolist(), hvg_df


def compute_top_mean_genes(Y_raw, gene_names, n_top=250):
    idx = np.argsort(Y_raw.mean(axis=0))[::-1][:n_top]
    return [gene_names[i] for i in idx]


def build_weights(coords: np.ndarray, radius: int = 1) -> np.ndarray:
    N = len(coords)
    W = np.zeros((N, N), dtype=np.float32)
    r, c = coords[:, 0].astype(int), coords[:, 1].astype(int)
    for i in range(N):
        dr = np.abs(r - r[i])
        dc = np.abs(c - c[i])
        nb = (np.maximum(dr, dc) <= radius) & (np.arange(N) != i)
        W[i, nb] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return W / rs


def morans_i(Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Moran's I for all genes simultaneously. Y: (N,m), W: (N,N)."""
    N, m = Y.shape
    S0 = W.sum()
    if S0 == 0:
        return np.zeros(m)
    Z = Y - Y.mean(axis=0)
    WZ = W @ Z
    num = (Z * WZ).sum(axis=0)
    den = np.where((Z * Z).sum(axis=0) > 1e-8,
                   (Z * Z).sum(axis=0), 1e-8)
    return ((N / S0) * (num / den)).astype(np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gene names from first sample
    feature_dir = DATA_DIR / "features"
    sample_names = sorted([f.stem for f in feature_dir.glob("*.csv")])
    gene_names = pd.read_csv(feature_dir / f"{sample_names[0]}.csv",
                             header=None)[0].tolist()
    print(f"Total genes in dataset: {len(gene_names)}")

    # Fold 0 training samples
    train_samples, test_samples = build_lopcv_fold(sample_names, FOLD_IDX)
    print(f"Fold {FOLD_IDX}: {len(train_samples)} train, "
          f"{len(test_samples)} test samples")
    print(f"Held-out patient: {get_patient_id(test_samples[0])}")

    # Load two separate matrices
    print("\nLoading raw UMI counts (for seurat_v3)...")
    Y_raw = load_matrix(DATA_DIR / "umi_counts", train_samples, "raw counts")
    print("Loading log-normalized counts (for Moran's I)...")
    Y_log = load_matrix(DATA_DIR / "counts_spcs", train_samples, "log counts")

    assert Y_raw.shape == Y_log.shape, \
        f"Shape mismatch: raw {Y_raw.shape} vs log {Y_log.shape}"
    print(f"Training matrix: {Y_raw.shape[0]} spots x {Y_raw.shape[1]} genes")

    # Gene selection
    print("\n-- Selecting HMHVG (seurat_v3 on raw counts) --")
    hmhvg_genes, hvg_df = compute_hmhvg(Y_raw, gene_names, N_HMHVG)
    top_mean_genes = compute_top_mean_genes(Y_raw, gene_names, N_MEAN)

    gene_idx     = {g: i for i, g in enumerate(gene_names)}
    hmhvg_idx    = [gene_idx[g] for g in hmhvg_genes if g in gene_idx]
    top_mean_idx = [gene_idx[g] for g in top_mean_genes if g in gene_idx]
    overlap      = len(set(hmhvg_genes) & set(top_mean_genes))
    overlap_pct  = overlap / N_HMHVG

    print(f"HMHVG: {len(hmhvg_genes)} genes")
    print(f"Top-{N_MEAN}-by-mean: {len(top_mean_genes)} genes")
    print(f"Overlap: {overlap} genes ({overlap_pct:.1%})")

    # Moran's I on log-normalized data
    print("\n-- Computing Moran's I (log-normalized, per slide) --")
    all_hmhvg_mi, all_top_mean_mi = [], []
    slides_done = 0

    for sample in train_samples:
        expr_p  = DATA_DIR / "counts_spcs" / f"{sample}.npy"
        coord_p = DATA_DIR / "tissue_positions" / f"{sample}.csv"
        if not expr_p.exists() or not coord_p.exists():
            print(f"  WARN: missing {sample}, skipping")
            continue
        Y      = np.load(expr_p)
        pos_df = pd.read_csv(coord_p, index_col=0)
        # Handle different column naming conventions
        if 'array_row' in pos_df.columns:
            coords = pos_df[['array_row', 'array_col']].values
        else:
            # Fallback: use last two numeric columns
            coords = pos_df.iloc[:, -2:].values
        if Y.shape[0] != len(coords):
            continue
        W  = build_weights(coords)
        mi = morans_i(Y, W)
        all_hmhvg_mi.extend(mi[hmhvg_idx].tolist())
        all_top_mean_mi.extend(mi[top_mean_idx].tolist())
        slides_done += 1
        print(f"  {sample}: {Y.shape[0]} spots done")

    print(f"\nMoran's I computed on {slides_done} slides.")

    mi_hmhvg    = np.array(all_hmhvg_mi)
    mi_top_mean = np.array(all_top_mean_mi)

    # Statistical tests
    mw_stat, mw_p = mannwhitneyu(mi_hmhvg, mi_top_mean, alternative='greater')
    ks_stat, ks_p = ks_2samp(mi_hmhvg, mi_top_mean)

    print(f"\n-- Statistical Tests --")
    print(f"  HMHVG    mean={mi_hmhvg.mean():.4f}  "
          f"median={np.median(mi_hmhvg):.4f}")
    print(f"  Top-mean mean={mi_top_mean.mean():.4f}  "
          f"median={np.median(mi_top_mean):.4f}")
    print(f"  Mann-Whitney U (one-sided, HMHVG > top-mean): "
          f"stat={mw_stat:.1f}  p={mw_p:.2e}")
    print(f"  KS test (two-sided): stat={ks_stat:.4f}  p={ks_p:.2e}")

    gate = "PASS" if mw_p < 0.05 else "FAIL"
    print(f"  Gene selection gate: {gate}")

    # ECDF data for paper figure
    def ecdf(x):
        xs = np.sort(x)
        return xs, np.arange(1, len(xs) + 1) / len(xs)

    xs_h, ys_h = ecdf(mi_hmhvg)
    xs_m, ys_m = ecdf(mi_top_mean)

    pd.concat([
        pd.DataFrame({"morans_i": xs_h, "ecdf": ys_h, "gene_set": "HMHVG"}),
        pd.DataFrame({"morans_i": xs_m, "ecdf": ys_m,
                       "gene_set": "top_250_mean"}),
    ]).to_csv(OUTPUT_DIR / "morans_i_ecdf.csv", index=False)

    # Save all outputs
    hvg_df.to_csv(OUTPUT_DIR / "hvg_statistics.csv")
    with open(OUTPUT_DIR / "hmhvg_genes.json", "w") as f:
        json.dump(hmhvg_genes, f, indent=2)
    with open(OUTPUT_DIR / "top_mean_genes.json", "w") as f:
        json.dump(top_mean_genes, f, indent=2)

    summary = {
        "dataset": "her2st", "fold": FOLD_IDX,
        "n_train_samples": len(train_samples),
        "n_hmhvg": len(hmhvg_genes),
        "n_top_mean": len(top_mean_genes),
        "overlap_count": overlap,
        "overlap_pct": float(overlap_pct),
        "morans_i": {
            "hmhvg_mean":    float(mi_hmhvg.mean()),
            "hmhvg_median":  float(np.median(mi_hmhvg)),
            "top_mean_mean": float(mi_top_mean.mean()),
            "top_mean_median": float(np.median(mi_top_mean)),
        },
        "mann_whitney_p": float(mw_p),
        "ks_statistic":   float(ks_stat),
        "ks_p":           float(ks_p),
        "gene_selection_gate": gate,
        "seurat_v3_on_raw_counts": True,
        "morans_i_on_lognorm": True,
    }
    with open(OUTPUT_DIR / "gene_selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_DIR}")
    print(f"\n{'='*50}")
    print(f"GENE SELECTION GATE: {gate}")
    print(f"Mann-Whitney p = {mw_p:.2e}")
    print(f"KS statistic  = {ks_stat:.4f}")
    print(f"Overlap       = {overlap_pct:.1%}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
