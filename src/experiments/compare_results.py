"""Compare TRIPLEX vs TRIPLEX+MCSPR results.

Usage: python src/experiments/compare_results.py --dataset her2st

Loads both summary JSONs and prints delta table with significance test.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description="Compare results")
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name (her2st, stnet, scc)",
    )
    args = parser.parse_args()

    base_path = Path(f"results/triplex/{args.dataset}/summary.json")
    mcspr_path = Path(f"results/triplex_mcspr/{args.dataset}/summary.json")

    if not base_path.exists():
        print(f"ERROR: {base_path} not found. Run run_triplex.py first.")
        return
    if not mcspr_path.exists():
        print(f"ERROR: {mcspr_path} not found. Run run_triplex_mcspr.py first.")
        return

    with open(base_path) as f:
        base = json.load(f)
    with open(mcspr_path) as f:
        mcspr = json.load(f)

    # Metrics to compare
    metrics = [
        ("PCC(M)", "pcc_m", True),     # higher is better
        ("PCC(H)", "pcc_h", True),     # higher is better
        ("MSE", "mse", False),          # lower is better
        ("MAE", "mae", False),          # lower is better
    ]

    # Header
    print(f"\n{'='*72}")
    print(f"  COMPARISON: {args.dataset.upper()}")
    print(f"  TRIPLEX vs TRIPLEX+MCSPR")
    print(f"{'='*72}")
    print(
        f"  {'Metric':<10} {'TRIPLEX':>16} {'TRIPLEX+MCSPR':>16} "
        f"{'Delta':>10} {'Delta%':>8} {'p-val':>8}"
    )
    print(f"  {'-'*68}")

    for label, key, higher_better in metrics:
        b_val = base[key]
        m_val = mcspr[key]
        b_std = base.get(f"{key}_std", 0)
        m_std = mcspr.get(f"{key}_std", 0)

        delta = m_val - b_val
        delta_pct = (delta / abs(b_val)) * 100 if abs(b_val) > 1e-10 else 0.0

        # Direction indicator
        if higher_better:
            better = delta > 0
        else:
            better = delta < 0
        indicator = "+" if better else "-" if delta != 0 else "="

        # Paired t-test across folds
        p_val = np.nan
        if "per_fold" in base and "per_fold" in mcspr:
            b_folds = [f[key] for f in base["per_fold"]]
            m_folds = [f[key] for f in mcspr["per_fold"]]
            if len(b_folds) == len(m_folds) and len(b_folds) >= 2:
                _, p_val = ttest_rel(b_folds, m_folds)

        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
        sig = "*" if not np.isnan(p_val) and p_val < 0.05 else ""

        print(
            f"  {label:<10} "
            f"{b_val:>7.4f}+/-{b_std:.4f} "
            f"{m_val:>7.4f}+/-{m_std:.4f} "
            f"{indicator}{abs(delta):>8.4f} "
            f"{delta_pct:>7.1f}% "
            f"{p_str:>6}{sig}"
        )

    # SMCS (only for MCSPR)
    if mcspr.get("smcs_overall"):
        print(f"\n  SMCS (TRIPLEX+MCSPR only): {mcspr['smcs_overall']:.4f}")

    print(f"\n  * p < 0.05 (paired t-test across folds)")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
