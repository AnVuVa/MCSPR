"""Extract Spatially Variable Genes (SVG) using:
  1. Filter: drop genes in bottom 50th percentile of mean expression
  2. Sort: rank remaining by mean Moran's I
  3. Select: top 300
  4. Report: mean Moran's I of SVG set vs top-250-by-mean

Professor's safeguard: minimum mean expression filter prevents near-zero
dropout genes with artificially inflated Moran's I from entering the set.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR   = Path("data/her2st")
RAW_DIR    = Path("data/her2st/umi_counts")
GENE_LIST  = Path("data/her2st/features_full/gene_names.json")
OUTPUT_DIR = Path("results/pre_training_gates/gene_selection")
N_SVG      = 300
FOLD_IDX   = 0

with open(GENE_LIST) as f:
    gene_names = json.load(f)

# -- Fold 0 training samples --
feature_dir  = DATA_DIR / "features"
sample_names = sorted([f.stem for f in feature_dir.glob("*.csv")])
patient_map  = defaultdict(list)
for s in sample_names:
    pid = s.replace("her2st_","").replace("HER2ST_","")[0].upper()
    patient_map[pid].append(s)
patients      = sorted(patient_map.keys())
held          = patients[FOLD_IDX]
train_samples = [s for p in patients if p != held for s in patient_map[p]]

# -- Compute per-gene mean expression across training slides --
print("Computing per-gene mean expression (training slides)...")
mean_arrays = []
for s in train_samples:
    p = RAW_DIR / f"{s}.npy"
    if p.exists():
        mean_arrays.append(np.load(p).mean(axis=0))   # (m,) mean per gene per slide
gene_mean_expr = np.stack(mean_arrays).mean(axis=0)   # (m,) grand mean
print(f"  Gene mean expression range: [{gene_mean_expr.min():.3f}, {gene_mean_expr.max():.1f}]")

# -- Compute per-gene mean Moran's I across training slides --
print("Computing per-gene mean Moran's I (log1p-normalized, training slides)...")
mi_accumulator = []   # list of (m,) arrays, one per slide

for sample in train_samples:
    raw_p   = RAW_DIR / f"{sample}.npy"
    coord_p = DATA_DIR / "tissue_positions" / f"{sample}.csv"
    if not raw_p.exists() or not coord_p.exists():
        continue

    Y_full = np.load(raw_p).astype(np.float32)
    lib    = Y_full.sum(axis=1, keepdims=True)
    Y_log  = np.log1p(Y_full / (lib + 1e-8) * 1e4)

    coords = pd.read_csv(coord_p, index_col=0)[['array_row','array_col']].values
    N = Y_log.shape[0]
    if N != len(coords):
        continue

    rows, cols = coords[:,0].astype(int), coords[:,1].astype(int)
    W = np.zeros((N, N), np.float32)
    for i in range(N):
        dr = np.abs(rows - rows[i]); dc = np.abs(cols - cols[i])
        nb = (np.maximum(dr, dc) <= 1) & (np.arange(N) != i)
        W[i, nb] = 1.0
    rs = W.sum(1, keepdims=True); rs[rs==0] = 1; W /= rs
    S0 = W.sum()

    Z   = Y_log - Y_log.mean(0)
    WZ  = W @ Z
    num = (Z * WZ).sum(0)
    den = np.where((Z*Z).sum(0) > 1e-8, (Z*Z).sum(0), 1e-8)
    mi  = (N / S0) * (num / den)
    mi_accumulator.append(mi)
    print(f"  {sample}: {N} spots, mean MI = {mi.mean():.4f}")

gene_mean_mi = np.stack(mi_accumulator).mean(axis=0)   # (m,)
print(f"\nMoran's I computed for {len(gene_names)} genes across {len(mi_accumulator)} slides")

# -- Step 1: Filter -- drop bottom 50th percentile mean expression --
expr_threshold = np.percentile(gene_mean_expr, 50)
expr_filter    = gene_mean_expr >= expr_threshold
n_pass_filter  = expr_filter.sum()
print(f"\nExpression filter (>= p50 = {expr_threshold:.4f}): {n_pass_filter}/{len(gene_names)} genes pass")

# -- Step 2: Sort remaining genes by mean Moran's I --
filtered_mi   = np.where(expr_filter, gene_mean_mi, -999.0)
ranked_idx    = np.argsort(filtered_mi)[::-1]

# -- Step 3: Top 300 --
svg_idx   = ranked_idx[:N_SVG]
svg_genes = [gene_names[i] for i in svg_idx]
svg_mi    = gene_mean_mi[svg_idx]

print(f"\nTop {N_SVG} SVG genes selected")
print(f"  SVG mean Moran's I:      {svg_mi.mean():.4f}")
print(f"  SVG min  Moran's I:      {svg_mi.min():.4f}  (cutoff)")
print(f"  SVG mean expression:     {gene_mean_expr[svg_idx].mean():.4f}")

# -- Step 4: Compare to top-250-by-mean --
top250_idx  = np.argsort(gene_mean_expr)[::-1][:250]
top250_mi   = gene_mean_mi[top250_idx]
top250_mean_expr = gene_mean_expr[top250_idx]

overlap     = len(set(svg_genes) & set([gene_names[i] for i in top250_idx]))

print(f"\nComparison -- SVG (300) vs Top-250-by-mean:")
print(f"  SVG    mean MI: {svg_mi.mean():.4f}")
print(f"  Top250 mean MI: {top250_mi.mean():.4f}")
print(f"  Delta MI:       {svg_mi.mean() - top250_mi.mean():+.4f}")
print(f"  Overlap:        {overlap}/{N_SVG} ({overlap/N_SVG:.1%})")

gate1_closed = svg_mi.mean() > top250_mi.mean()
print(f"\nGATE 1 FINAL STATUS: {'CLOSED -- SVG beats top-250 on Moran I' if gate1_closed else 'STILL OPEN -- investigate'}")

# -- Save outputs --
with open(OUTPUT_DIR / "svg_genes.json", "w") as f:
    json.dump(svg_genes, f, indent=2)

summary = {
    "n_genes_full_transcriptome":   len(gene_names),
    "n_genes_pass_expr_filter":     int(n_pass_filter),
    "expr_filter_percentile":       50,
    "expr_threshold":               float(expr_threshold),
    "n_svg_selected":               N_SVG,
    "svg_mean_morans_i":            float(svg_mi.mean()),
    "svg_min_morans_i":             float(svg_mi.min()),
    "top250_mean_morans_i":         float(top250_mi.mean()),
    "delta_morans_i":               float(svg_mi.mean() - top250_mi.mean()),
    "overlap_svg_top250":           overlap,
    "overlap_pct":                  float(overlap / N_SVG),
    "gate1_closed":                 bool(gate1_closed),
    "gene_selection_method":        "top_300_by_morans_i_filtered_p50_expression",
}
with open(OUTPUT_DIR / "svg_selection_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved: {OUTPUT_DIR}/svg_genes.json")
print(f"Saved: {OUTPUT_DIR}/svg_selection_summary.json")
