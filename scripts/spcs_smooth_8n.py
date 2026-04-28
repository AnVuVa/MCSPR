"""SPCS-style 8-neighbor smoothing of UMI counts (ported from
MERGE/scripts/8n_smoothing.py).

For each spot, replaces its log-library-size-normalized count vector with the
mean of its own + 8 nearest neighbors' (in array_row/array_col grid) vectors.
This is the data-prep step that produces MERGE's `counts_spcs_to_8n` panel.

Usage:
    python scripts/spcs_smooth_8n.py \
        -i data/her2st \
        -o data/her2st/counts_spcs_to_8n_v2

Reads `data/<input>/umi_counts/{slide}.npy` (raw counts, n_spots x n_genes)
and `data/<input>/tissue_positions/{slide}.csv` (with array_row, array_col,
in_tissue), writes `<output>/{slide}.npy` of identical shape but smoothed
+ log-normalized.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import scprep as scp
from tqdm import tqdm


# Self + 8 neighbors on the array_row / array_col grid.
_DELTA = np.array([
    [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, -1], [1, -1], [-1, 1],
    [0, 0],
])


def _smooth_one_slide(counts: np.ndarray, x_coords, y_coords) -> np.ndarray:
    """Per-spot mean over self + 8-neighbors, applied in-place in iteration
    order — matches upstream MERGE/scripts/8n_smoothing.py exactly: spots
    iterated later see neighbors that have already been overwritten."""
    # Library-size normalize then natural log (same transform as upstream).
    counts = scp.transform.log(scp.normalize.library_size_normalize(counts))

    # Index spots by (round(x), round(y)) so we can look up neighbors.
    coord_to_idx = {
        f"{round(x)}_{round(y)}": i
        for i, (x, y) in enumerate(zip(x_coords, y_coords))
    }

    for x, y in zip(x_coords, y_coords):
        center_idx = coord_to_idx[f"{round(x)}_{round(y)}"]
        neighbor_idxs = []
        for dx, dy in _DELTA:
            key = f"{round(x + dx)}_{round(y + dy)}"
            if key in coord_to_idx:
                neighbor_idxs.append(coord_to_idx[key])
        counts[center_idx] = counts[neighbor_idxs].mean(axis=0)
    return counts


def main(args):
    input_dir = args.input
    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    umi_paths = sorted(glob(os.path.join(input_dir, "umi_counts", "*.npy")))
    tp_paths = sorted(glob(os.path.join(input_dir, "tissue_positions", "*.csv")))

    for umi_path, tp_path in zip(umi_paths, tp_paths):
        assert Path(umi_path).stem == Path(tp_path).stem, (
            f"Filename mismatch: {umi_path} vs {tp_path}"
        )

    for umi_path, tp_path in tqdm(list(zip(umi_paths, tp_paths))):
        counts = np.load(umi_path)
        # tissue_positions can be header-less or headed; try both.
        try:
            tp = pd.read_csv(tp_path)
            in_tissue = tp[tp["in_tissue"] == 1]
            x = in_tissue["array_row"].values
            y = in_tissue["array_col"].values
        except KeyError:
            tp = pd.read_csv(tp_path, header=None)
            in_tissue = tp[tp[1] == 1]
            x = in_tissue[2].values
            y = in_tissue[3].values

        smoothed = _smooth_one_slide(counts, x, y)
        np.save(
            os.path.join(output_dir, Path(umi_path).stem + ".npy"),
            smoothed, allow_pickle=False,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Dataset root containing umi_counts/ and tissue_positions/",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output dir for smoothed .npy per slide",
    )
    args = parser.parse_args()
    main(args)
