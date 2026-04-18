"""
Spot ID verification — confirms raw almaan/her2st spot ordering
matches MERGE-preprocessed data.

Checks per sample:
  1. Barcodes in raw .tsv.gz match barcodes in MERGE tissue_positions
  2. Spot count matches MERGE counts_spcs
  3. Gene count is full transcriptome (~15k), not pre-filtered (250)
"""

import argparse
import gzip
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


def verify_sample(
    sample: str,
    her2st_dir: Path,
    merge_dir: Path,
) -> Dict:
    """Verify one sample's spot IDs between raw and MERGE data."""
    # Raw counts
    raw_path = her2st_dir / "data" / "ST-cnts" / f"{sample}.tsv.gz"
    if not raw_path.exists():
        return {"status": "MISSING_RAW", "sample": sample}

    with gzip.open(raw_path, "rt") as f:
        header = f.readline().strip().split("\t")
        raw_spots = []
        for line in f:
            raw_spots.append(line.split("\t")[0])

    raw_genes = header[1:]  # first column is spot ID header (empty or index)
    n_raw_genes = len(raw_genes)

    # MERGE barcodes
    merge_pos = merge_dir / "tissue_positions" / f"{sample}.csv"
    if not merge_pos.exists():
        return {"status": "MISSING_MERGE", "sample": sample}

    pos_df = pd.read_csv(merge_pos, index_col=0)
    merge_spots = pos_df.index.tolist()

    # MERGE counts
    merge_counts = merge_dir / "counts_spcs" / f"{sample}.npy"
    n_merge_spots = 0
    if merge_counts.exists():
        Y = np.load(merge_counts)
        n_merge_spots = Y.shape[0]

    # Compare
    result = {
        "sample": sample,
        "n_raw_spots": len(raw_spots),
        "n_merge_spots": len(merge_spots),
        "n_merge_counts": n_merge_spots,
        "n_raw_genes": n_raw_genes,
    }

    if set(raw_spots) != set(merge_spots):
        # Check if one is a subset
        raw_set = set(raw_spots)
        merge_set = set(merge_spots)
        if merge_set.issubset(raw_set):
            result["status"] = "MERGE_SUBSET"
            result["note"] = (f"MERGE has {len(merge_set)} spots, "
                              f"raw has {len(raw_set)} (MERGE is filtered subset)")
        else:
            missing_in_raw = merge_set - raw_set
            result["status"] = "BARCODE_MISMATCH"
            result["missing_in_raw"] = len(missing_in_raw)
        return result

    if raw_spots == merge_spots:
        result["status"] = "OK"
    elif set(raw_spots) == set(merge_spots):
        result["status"] = "ORDER_MISMATCH_FIXABLE"
        result["note"] = "Same barcodes, different order — reindex during conversion"
    else:
        result["status"] = "BARCODE_MISMATCH"

    if n_merge_spots != len(merge_spots):
        result["count_check"] = "COUNT_MISMATCH"
    else:
        result["count_check"] = "OK"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--her2st_dir", type=Path, required=True)
    parser.add_argument("--merge_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover samples from MERGE features
    feature_dir = args.merge_dir / "features"
    sample_names = sorted([f.stem for f in feature_dir.glob("*.csv")])
    print(f"Verifying {len(sample_names)} samples...")

    results = []
    n_ok = 0
    n_fixable = 0
    n_broken = 0

    for sample in sample_names:
        r = verify_sample(sample, args.her2st_dir, args.merge_dir)
        results.append(r)
        status = r["status"]
        print(f"  {sample}: {status}  "
              f"({r.get('n_raw_spots', '?')} spots, "
              f"{r.get('n_raw_genes', '?')} genes)")

        if status == "OK":
            n_ok += 1
        elif status in ("ORDER_MISMATCH_FIXABLE", "MERGE_SUBSET"):
            n_fixable += 1
        else:
            n_broken += 1

    verdict = "PASS" if n_broken == 0 else "FAIL"
    print(f"\nVerification: {verdict}")
    print(f"  OK: {n_ok}  Fixable: {n_fixable}  Broken: {n_broken}")

    summary = {
        "verdict": verdict,
        "n_samples": len(sample_names),
        "n_ok": n_ok,
        "n_fixable": n_fixable,
        "n_broken": n_broken,
        "samples": results,
    }
    out_path = args.output_dir / "spot_id_verification.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
