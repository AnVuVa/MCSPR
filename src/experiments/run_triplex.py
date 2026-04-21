"""Train TRIPLEX (no MCSPR) across all LOPCV folds.

Usage: python src/experiments/run_triplex.py --config configs/her2st.yaml [--dry_run]
Saves results to results/triplex/{dataset}/
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loaders import build_loaders, build_lopcv_folds
from src.data.precompute import precompute_all
from src.models.triplex import TRIPLEX
from src.training.trainer import train_one_fold


def set_seed(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Train TRIPLEX")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Run one batch and exit"
    )
    parser.add_argument(
        "--fold", type=int, default=-1,
        help="Run a single fold (0..n_folds-1). Default -1 runs all folds."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override default output directory."
    )
    parser.add_argument(
        "--use_normalized_mse", action="store_true",
        help="Override config: enable L_norm (per-gene variance-normalized MSE)."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.use_normalized_mse:
        config.setdefault("training", {})["use_normalized_mse"] = True

    dataset = config["dataset"]
    data_dir = config["data_dir"]
    n_genes = config.get("n_genes", 250)
    seed = config.get("training", {}).get("seed", 2021)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)

    # Discover samples
    bc_dir = Path(data_dir) / "barcodes"
    if not bc_dir.exists():
        print(f"ERROR: {bc_dir} not found. Populate data directory first.")
        return

    sample_names = sorted([p.stem for p in bc_dir.glob("*.csv")])
    print(f"Dataset: {dataset}, {len(sample_names)} samples")

    # Precompute if needed
    gf_dir = Path(data_dir) / "global_features"
    if not gf_dir.exists() or len(list(gf_dir.glob("*.npy"))) == 0:
        print("\n=== Running precomputation ===")
        precompute_all(data_dir, dataset, config)

    # Build folds
    folds = build_lopcv_folds(sample_names, dataset)
    n_folds = len(folds)
    print(f"\n{n_folds} LOPCV folds:")
    for i, (train, test) in enumerate(folds):
        test_patients = set()
        from src.data.loaders import get_patient_id

        for s in test:
            test_patients.add(get_patient_id(s, dataset))
        print(
            f"  Fold {i}: {len(train)} train, {len(test)} test | "
            f"test patient(s): {sorted(test_patients)}"
        )

    output_dir = args.output_dir if args.output_dir else f"results/triplex/{dataset}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fold_results = []

    fold_range = [args.fold] if args.fold >= 0 else range(n_folds)
    if args.fold >= 0 and not (0 <= args.fold < n_folds):
        raise ValueError(f"--fold {args.fold} out of range [0, {n_folds})")

    for fold_idx in fold_range:
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{n_folds-1}")
        print(f"{'='*60}")

        set_seed(seed)

        train_loader, val_loader = build_loaders(
            data_dir, dataset, fold_idx, config, sample_names
        )

        if args.dry_run and fold_idx == 0:
            # Dry run checks
            print("\n--- Dry Run Checks ---")
            print(f"Train loader: {len(train_loader)} batches")
            print(f"Val loader: {len(val_loader)} batches")

            batch = next(iter(train_loader))
            print(f"\nBatch shapes:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape} {v.dtype}")
                else:
                    print(f"  {k}: {type(v).__name__}")

            model = TRIPLEX(config, n_genes=n_genes)
            model = model.to(device)
            print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Forward + backward
            train_one_fold(
                model, train_loader, val_loader, config,
                fold_idx=0, output_dir=output_dir,
                device=device, mcspr_artifacts=None,
                dry_run=True,
            )
            print("\nDry run complete.")
            return

        model = TRIPLEX(config, n_genes=n_genes)
        metrics = train_one_fold(
            model, train_loader, val_loader, config,
            fold_idx=fold_idx, output_dir=output_dir,
            device=device, mcspr_artifacts=None,
            gene_names=train_loader.dataset.gene_names,
        )
        fold_results.append(metrics)

    # Aggregate results
    if fold_results:
        summary = {
            "dataset": dataset,
            "n_folds": n_folds,
            "pcc_m": float(np.mean([r["pcc_m"] for r in fold_results])),
            "pcc_m_std": float(np.std([r["pcc_m"] for r in fold_results])),
            "pcc_h": float(np.mean([r["pcc_h"] for r in fold_results])),
            "pcc_h_std": float(np.std([r["pcc_h"] for r in fold_results])),
            "mse": float(np.mean([r["mse"] for r in fold_results])),
            "mse_std": float(np.std([r["mse"] for r in fold_results])),
            "mae": float(np.mean([r["mae"] for r in fold_results])),
            "mae_std": float(np.std([r["mae"] for r in fold_results])),
            "per_fold": [
                {
                    "pcc_m": r["pcc_m"],
                    "pcc_h": r["pcc_h"],
                    "mse": r["mse"],
                    "mae": r["mae"],
                }
                for r in fold_results
            ],
        }

        with open(f"{output_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS — TRIPLEX on {dataset}")
        print(f"{'='*60}")
        print(f"  PCC(M): {summary['pcc_m']:.4f} +/- {summary['pcc_m_std']:.4f}")
        print(f"  PCC(H): {summary['pcc_h']:.4f} +/- {summary['pcc_h_std']:.4f}")
        print(f"  MSE:    {summary['mse']:.4f} +/- {summary['mse_std']:.4f}")
        print(f"  MAE:    {summary['mae']:.4f} +/- {summary['mae_std']:.4f}")


if __name__ == "__main__":
    main()
