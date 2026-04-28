"""Validate the SPCS Python port against MERGE's R-produced counts_spcs.

For each MERGE slide:
  1. Load raw umi_counts (n_spots × 11870), subset to the 250-gene panel
     listed in features/{slide}.csv.
  2. Run our Python SPCS port.
  3. Compare to /mnt/d/docker_machine/anvuva/MERGE/data/her2st/counts_spcs/{slide}.npy.

Reports per-slide max-abs diff, mean-abs diff, and per-gene Pearson
correlation between port output and R reference. Bit-exact match is
unrealistic (rPCA + Pearson dist via different libs), but per-gene PCC
should be >0.99 if the port is correct.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.spcs import spcs_smooth, SPCSParams


MERGE_ROOT = Path("/mnt/d/docker_machine/anvuva/MERGE/data/her2st")


def _load_slide(slide: str):
    raw = np.load(MERGE_ROOT / "umi_counts" / f"{slide}.npy")  # (n, 11870)
    full_names = json.load(open(MERGE_ROOT / "features_full" / "gene_names.json"))
    name_to_idx = {n: i for i, n in enumerate(full_names)}
    panel_names = pd.read_csv(MERGE_ROOT / "features" / f"{slide}.csv",
                              header=None)[0].tolist()
    panel_idx = np.array([name_to_idx[g] for g in panel_names], dtype=np.int64)
    counts = raw[:, panel_idx]

    tp = pd.read_csv(MERGE_ROOT / "tissue_positions" / f"{slide}.csv")
    in_t = tp[tp["in_tissue"] == 1]
    return counts, in_t["array_row"].values, in_t["array_col"].values, panel_names


def _per_gene_pcc(a: np.ndarray, b: np.ndarray):
    pccs = []
    for k in range(a.shape[1]):
        if a[:, k].std() < 1e-9 or b[:, k].std() < 1e-9:
            continue
        r, _ = pearsonr(a[:, k], b[:, k])
        pccs.append(r)
    return np.array(pccs)


def main():
    slides = sorted(p.stem for p in (MERGE_ROOT / "counts_spcs").glob("*.npy"))[:6]
    print(f"Validating on {len(slides)} slides: {slides}\n")
    print(f"{'slide':6s} {'n_spots':>8s} {'n_genes':>8s} {'max|diff|':>10s} "
          f"{'mean|diff|':>11s} {'pcc_med':>8s} {'pcc_min':>8s} {'pcc>0.95':>9s}")
    for slide in slides:
        counts, ar, ac, names = _load_slide(slide)
        ref = np.load(MERGE_ROOT / "counts_spcs" / f"{slide}.npy")  # (n, 250)
        out, keep = spcs_smooth(counts, ar, ac, SPCSParams())
        # Re-index ref to the same surviving genes.
        ref_kept = ref[:, keep]
        diff = np.abs(out - ref_kept)
        pccs = _per_gene_pcc(out, ref_kept)
        print(f"{slide:6s} {counts.shape[0]:>8d} {keep.sum():>8d} "
              f"{diff.max():>10.4f} {diff.mean():>11.5f} "
              f"{np.median(pccs):>8.4f} {pccs.min():>8.4f} "
              f"{(pccs > 0.95).sum():>4d}/{pccs.size:<4d}")


if __name__ == "__main__":
    main()
