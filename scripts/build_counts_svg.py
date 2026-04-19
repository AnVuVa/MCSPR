"""Build counts_svg/{sample}.npy and features_svg/{sample}.csv for Option D.

Indexes umi_counts (N, 11870) by svg_genes.json order, lib-normalizes,
applies log1p. Matches the normalization inside run_precompute.py exactly:
    lib = Y_raw.sum(axis=1, keepdims=True)
    Y_log = np.log1p(Y_raw / (lib + 1e-8) * 1e4)
    Y_svg = Y_log[:, svg_idx]
"""
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/her2st")
RAW_DIR = DATA_DIR / "umi_counts"
OUT_COUNTS = DATA_DIR / "counts_svg"
OUT_FEAT = DATA_DIR / "features_svg"
SVG_PATH = Path("results/pre_training_gates/gene_selection/svg_genes.json")

OUT_COUNTS.mkdir(parents=True, exist_ok=True)
OUT_FEAT.mkdir(parents=True, exist_ok=True)

with open(SVG_PATH) as f:
    svg_genes = json.load(f)
with open(DATA_DIR / "features_full" / "gene_names.json") as f:
    all_genes = json.load(f)
gene_idx_map = {g: i for i, g in enumerate(all_genes)}
svg_idx = np.array([gene_idx_map[g] for g in svg_genes])
assert len(svg_idx) == 300, f"expected 300 SVG genes, got {len(svg_idx)}"
print(f"SVG gene set: {len(svg_genes)} genes (all present in umi_counts)")

samples = sorted(p.stem for p in RAW_DIR.glob("*.npy"))
print(f"Samples: {len(samples)}")

for s in samples:
    raw = np.load(RAW_DIR / f"{s}.npy").astype(np.float32)
    lib = raw.sum(axis=1, keepdims=True)
    Y_log = np.log1p(raw / (lib + 1e-8) * 1e4)
    Y_svg = Y_log[:, svg_idx].astype(np.float32)
    np.save(OUT_COUNTS / f"{s}.npy", Y_svg)
    with open(OUT_FEAT / f"{s}.csv", "w") as f:
        for g in svg_genes:
            f.write(g + "\n")
    print(f"  {s}: {raw.shape} -> {Y_svg.shape}")

print(f"\nWrote {len(samples)} counts_svg files to {OUT_COUNTS}")
print(f"Wrote {len(samples)} features_svg files to {OUT_FEAT}")
