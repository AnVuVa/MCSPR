"""
Convert raw almaan/her2st .tsv.gz count matrices to .npy format.

For each sample:
  1. Load raw integer count matrix from ST-cnts/{sample}.tsv.gz
  2. Align spot ordering to match MERGE tissue_positions
  3. Intersect genes across all samples to get a shared gene set
  4. Save as (N_spots, N_genes) float32 .npy

Output:
  data/her2st/umi_counts/{sample}.npy     — raw integer counts
  data/her2st/features_full/gene_names.json — shared gene list
"""

import argparse
import gzip
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def load_raw_tsv(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    """Load a gzipped TSV count matrix.
    Returns (spot_ids, gene_names, count_matrix).
    """
    with gzip.open(path, "rt") as f:
        header = f.readline().strip().split("\t")
        genes = header[1:]  # first col is spot ID header

        spots = []
        rows = []
        for line in f:
            parts = line.strip().split("\t")
            spots.append(parts[0])
            rows.append([float(x) for x in parts[1:]])

    return spots, genes, np.array(rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--her2st_dir", type=Path, required=True)
    parser.add_argument("--merge_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_full_dir = args.merge_dir / "features_full"
    feature_full_dir.mkdir(parents=True, exist_ok=True)

    # Discover samples
    feature_dir = args.merge_dir / "features"
    sample_names = sorted([f.stem for f in feature_dir.glob("*.csv")])
    print(f"Converting {len(sample_names)} samples...")

    # First pass: find shared gene set across all samples
    print("Pass 1: Finding shared gene set...")
    shared_genes = None
    for sample in sample_names:
        raw_path = args.her2st_dir / "data" / "ST-cnts" / f"{sample}.tsv.gz"
        if not raw_path.exists():
            print(f"  WARN: {raw_path} not found, skipping")
            continue
        with gzip.open(raw_path, "rt") as f:
            header = f.readline().strip().split("\t")
            genes = set(header[1:])
        if shared_genes is None:
            shared_genes = genes
        else:
            shared_genes = shared_genes & genes

    shared_genes_list = sorted(shared_genes)
    print(f"Shared gene set: {len(shared_genes_list)} genes")

    # Use first sample as reference for gene ordering
    raw_ref = args.her2st_dir / "data" / "ST-cnts" / f"{sample_names[0]}.tsv.gz"
    ref_spots, ref_genes, _ = load_raw_tsv(raw_ref)
    # Keep reference gene order but filter to shared set
    gene_order = [g for g in ref_genes if g in shared_genes]
    gene_to_idx = {g: i for i, g in enumerate(ref_genes)}
    print(f"Reference gene set ({sample_names[0]}): {len(ref_genes)} genes")
    print(f"Using {len(gene_order)} shared genes in reference order")

    # Second pass: convert each sample
    print(f"\nPass 2: Converting...")
    n_converted = 0
    for sample in sample_names:
        raw_path = args.her2st_dir / "data" / "ST-cnts" / f"{sample}.tsv.gz"
        if not raw_path.exists():
            continue

        spots, genes, counts = load_raw_tsv(raw_path)

        # Build gene index for this sample
        sample_gene_idx = {g: i for i, g in enumerate(genes)}

        # Reorder genes to match reference gene_order
        col_indices = []
        for g in gene_order:
            if g in sample_gene_idx:
                col_indices.append(sample_gene_idx[g])
            else:
                col_indices.append(-1)  # missing gene

        # Build aligned matrix
        N = len(spots)
        M = len(gene_order)
        aligned = np.zeros((N, M), dtype=np.float32)
        for j, ci in enumerate(col_indices):
            if ci >= 0:
                aligned[:, j] = counts[:, ci]

        # Align spot ordering to match MERGE tissue_positions
        merge_pos = args.merge_dir / "tissue_positions" / f"{sample}.csv"
        if merge_pos.exists():
            pos_df = pd.read_csv(merge_pos, index_col=0)
            merge_spot_order = [str(s) for s in pos_df.index.tolist()]

            spot_to_row = {s: i for i, s in enumerate(spots)}
            reorder_idx = []
            for ms in merge_spot_order:
                if ms in spot_to_row:
                    reorder_idx.append(spot_to_row[ms])
                else:
                    reorder_idx.append(-1)

            if -1 in reorder_idx:
                n_missing = reorder_idx.count(-1)
                print(f"  WARN: {sample}: {n_missing} MERGE spots not found in raw")
                reorder_idx = [i for i in reorder_idx if i >= 0]

            aligned = aligned[reorder_idx]

        # Save
        out_path = args.output_dir / f"{sample}.npy"
        np.save(out_path, aligned)
        n_converted += 1
        print(f"  {sample}: saved {aligned.shape} -> {out_path}")

    # Save gene list
    gene_path = feature_full_dir / "gene_names.json"
    with open(gene_path, "w") as f:
        json.dump(gene_order, f, indent=2)

    print(f"\nConversion complete: {n_converted} samples")
    print(f"Gene count: {len(gene_order)}")
    print(f"Full gene list saved to {gene_path}")


if __name__ == "__main__":
    main()
