"""Build SPCS-style 8n-smoothed counts on the canonical 300-SVG panel.

For each slide:
  1. Load raw UMI counts (n_spots, 11870).
  2. Library-size normalize (denominator = total UMIs across all 11870 genes).
  3. log = scprep.transform.log (matches upstream MERGE/scripts/8n_smoothing.py).
  4. 8-neighbor mean smoothing on the array_row/array_col grid, applied
     IN-PLACE in iteration order (faithful to upstream — later spots see
     already-smoothed neighbors).
  5. Subset to the 300 SVG genes from data/her2st/locked_gene_set.json.
  6. Save (n_spots, 300) to data/her2st/counts_svg_smoothed8n/{slide}.npy.

Run inside the `merge` conda env (needs scprep).
"""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import scprep as scp
from tqdm import tqdm


_DELTA = np.array([
    [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, -1], [1, -1], [-1, 1],
    [0, 0],
])


def _smooth_inplace(counts: np.ndarray, x_coords, y_coords) -> np.ndarray:
    coord_to_idx = {
        f"{round(x)}_{round(y)}": i
        for i, (x, y) in enumerate(zip(x_coords, y_coords))
    }
    for x, y in zip(x_coords, y_coords):
        center_idx = coord_to_idx[f"{round(x)}_{round(y)}"]
        neighbor_idxs = [
            coord_to_idx[k]
            for dx, dy in _DELTA
            if (k := f"{round(x + dx)}_{round(y + dy)}") in coord_to_idx
        ]
        counts[center_idx] = counts[neighbor_idxs].mean(axis=0)
    return counts


def main():
    root = Path("data/her2st")
    out_dir = root / "counts_svg_smoothed8n"
    out_dir.mkdir(parents=True, exist_ok=True)

    # SVG gene index in the full 11870-panel.
    full_names = json.load(open(root / "features_full" / "gene_names.json"))
    name_to_idx = {n: i for i, n in enumerate(full_names)}
    svg_names = json.load(open(root / "locked_gene_set.json"))["gene_names"]
    svg_idx = np.array([name_to_idx[g] for g in svg_names], dtype=np.int64)
    assert svg_idx.shape == (300,)

    umi_paths = sorted(glob(str(root / "umi_counts" / "*.npy")))
    tp_paths = sorted(glob(str(root / "tissue_positions" / "*.csv")))
    assert len(umi_paths) == len(tp_paths) == 36
    for u, t in zip(umi_paths, tp_paths):
        assert Path(u).stem == Path(t).stem

    for u, t in tqdm(list(zip(umi_paths, tp_paths)), desc="slides"):
        name = Path(u).stem
        counts = np.load(u).astype(np.float64)
        tp = pd.read_csv(t)
        in_tissue = tp[tp["in_tissue"] == 1]
        x = in_tissue["array_row"].values
        y = in_tissue["array_col"].values
        assert counts.shape[0] == len(x), (
            f"{name}: counts {counts.shape[0]} vs in_tissue {len(x)}"
        )

        counts = scp.transform.log(scp.normalize.library_size_normalize(counts))
        counts = _smooth_inplace(counts, x, y)

        np.save(out_dir / f"{name}.npy", counts[:, svg_idx].astype(np.float32),
                allow_pickle=False)

    print(f"\nWrote {len(umi_paths)} slides to {out_dir}")
    a = np.load(out_dir / f"{Path(umi_paths[0]).stem}.npy")
    print(f"Sanity: {Path(umi_paths[0]).stem} shape={a.shape} "
          f"min={a.min():.3f} max={a.max():.3f} mean={a.mean():.3f}")


if __name__ == "__main__":
    main()
