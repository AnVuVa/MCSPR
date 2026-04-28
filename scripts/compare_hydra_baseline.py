"""Per-module ΔPCC: HydraSTNet vs STNet baseline, sliced by the same registry.

Loads the registry for each fold, the Hydra per-head results, and the baseline
fold's per-gene PCCs; reports per-module ΔPCC = mean(Hydra gene PCC over module
genes) - mean(baseline gene PCC over the same module genes). Crucial: the
baseline number is *sliced from the same idx_list[k]*, never re-trained.

Usage:
  python scripts/compare_hydra_baseline.py \
      --hydra_dir   results/baselines/stnet_hydra \
      --baseline_dir results/baselines/stnet \
      [--folds 0,1,2,3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_baseline_pcc_per_gene(baseline_dir: Path, fold: int) -> dict:
    """STNet baseline saves per-fold canonical_metrics.json with per-slide
    pcc_m + top50 indices but NOT per-gene PCC. Recompute per-gene from the
    saved best_model.pt would re-run inference. Cheaper: extract per-gene PCC
    from the canonical evaluator's intermediate state — but that isn't saved.

    For now, we expect the user to first run scripts/save_baseline_per_gene.py
    (or have a per-gene record at baseline_dir/fold_{F}/per_gene.json). If not
    found, raise a clear error pointing them at the next step.
    """
    p = baseline_dir / f"fold_{fold}" / "per_gene.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing baseline per-gene PCC at {p}.\n"
            f"Run: python scripts/canonical_eval.py --baseline stnet "
            f"--input_dir {baseline_dir} --emit_per_gene  (TODO add this flag)"
            f"\n— or compute from best_model.pt and write a {{gene: pcc}} json."
        )
    return json.loads(p.read_text())["pcc_per_gene"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hydra_dir", required=True,
                        help="e.g. results/baselines/stnet_hydra")
    parser.add_argument("--baseline_dir", required=True,
                        help="e.g. results/baselines/stnet")
    parser.add_argument("--folds", default="0,1,2,3")
    args = parser.parse_args()

    folds = [int(f) for f in args.folds.split(",")]
    hydra_dir = Path(args.hydra_dir)
    baseline_dir = Path(args.baseline_dir)

    print(f"{'fold':<6}{'module':<8}{'n_genes':<10}"
          f"{'PCC(hydra)':<14}{'PCC(base)':<14}{'Δ':<10}")
    deltas_per_module: dict[int, list[float]] = {}

    for fold in folds:
        full_hydra = json.loads(
            (hydra_dir / f"fold_{fold}" / "full.json").read_text()
        )
        registry_hash = full_hydra["registry_hash"]
        # Re-derive module → gene mapping from a head_k.json file
        head_files = sorted(
            (hydra_dir / f"fold_{fold}").glob("head_*.json"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        # Optional baseline per-gene
        try:
            base_per_gene = _load_baseline_pcc_per_gene(baseline_dir, fold)
        except FileNotFoundError as e:
            print(f"\nSKIP baseline comparison for fold {fold}: {e}")
            continue

        for hf in head_files:
            head = json.loads(hf.read_text())
            if head["registry_hash"] != registry_hash:
                print(f"  WARN: registry_hash mismatch on {hf}")
            k = head["module_id"]
            genes_k = head["gene_names"]
            pcc_h = head["pcc_module_mean"]
            pcc_b = float(np.nanmean([base_per_gene[g] for g in genes_k]))
            delta = pcc_h - pcc_b
            deltas_per_module.setdefault(k, []).append(delta)
            print(f"{fold:<6}{k:<8}{len(genes_k):<10}"
                  f"{pcc_h:<14.4f}{pcc_b:<14.4f}{delta:<+10.4f}")

    if deltas_per_module:
        print("\n=== Per-module ΔPCC (mean across folds) ===")
        for k in sorted(deltas_per_module):
            ds = deltas_per_module[k]
            print(f"  module {k}: {np.mean(ds):+.4f} ± {np.std(ds):.4f} "
                  f"(folds={len(ds)})")


if __name__ == "__main__":
    main()
