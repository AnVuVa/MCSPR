"""Build SPCS-smoothed counts on the canonical 300-SVG panel.

For each slide:
  1. Load raw UMI counts (n_spots × 11870 full panel).
  2. Run SPCS (Python port of MERGE/scripts/smooth.ipynb's R algorithm) on
     the full panel — libsize normalization must use the full denominator.
  3. Subset SPCS output to the 300 SVG genes from locked_gene_set.json.
  4. Save (n_spots, 300) float32 to data/her2st/counts_svg_spcs/{slide}.npy.

Defaults: tau_p=16, tau_s=2, alpha=0.6, beta=0.4, is_hexa=True (HER2ST is on
a hexagonal Visium-style grid). zero_cutoff=1.0 to preserve all genes
(the SVG panel is already pre-filtered for spatial variability).
"""

from __future__ import annotations

import json
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.spcs import spcs_smooth, SPCSParams


def main():
    root = Path("data/her2st")
    out_dir = root / "counts_svg_spcs"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_names = json.load(open(root / "features_full" / "gene_names.json"))
    name_to_idx = {n: i for i, n in enumerate(full_names)}
    svg_names = json.load(open(root / "locked_gene_set.json"))["gene_names"]
    svg_idx = np.array([name_to_idx[g] for g in svg_names], dtype=np.int64)
    assert svg_idx.shape == (300,)

    umi_paths = sorted(glob(str(root / "umi_counts" / "*.npy")))
    tp_paths = sorted(glob(str(root / "tissue_positions" / "*.csv")))
    assert len(umi_paths) == len(tp_paths) == 36

    params = SPCSParams(zero_cutoff=1.0)  # keep all 11870, subset later

    for u, t in tqdm(list(zip(umi_paths, tp_paths)), desc="slides"):
        name = Path(u).stem
        counts = np.load(u).astype(np.float64)
        tp = pd.read_csv(t)
        in_t = tp[tp["in_tissue"] == 1]
        ar = in_t["array_row"].values
        ac = in_t["array_col"].values
        assert counts.shape[0] == len(ar)

        smoothed_full, _ = spcs_smooth(counts, ar, ac, params)
        smoothed_svg = smoothed_full[:, svg_idx].astype(np.float32)
        np.save(out_dir / f"{name}.npy", smoothed_svg, allow_pickle=False)

    a = np.load(out_dir / f"{Path(umi_paths[0]).stem}.npy")
    print(f"\nWrote {len(umi_paths)} slides to {out_dir}")
    print(f"Sanity: {Path(umi_paths[0]).stem} shape={a.shape} "
          f"min={a.min():.3f} max={a.max():.3f} mean={a.mean():.3f}")


if __name__ == "__main__":
    main()
