"""
Train ST-Net baseline (no MCSPR) -- all LOPCV folds.

Usage:
    python src/experiments/run_stnet.py --config configs/her2st.yaml [--dry_run]

Saves results to: results/stnet/{dataset}/
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loaders import build_loaders, build_lopcv_folds
from src.models.stnet import STNet
from src.training.universal_trainer import train_one_fold, _evaluate


def set_seed(seed: int = 2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _aggregate(metrics_list, model_name, dataset):
    keys = ["pcc_m", "pcc_m_std", "mse", "mae", "rvd", "q1_mse"]
    summary = {"model": model_name, "dataset": dataset,
               "n_folds": len(metrics_list)}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train ST-Net baseline")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run one batch and exit")
    parser.add_argument("--fold", type=int, default=-1,
                        help="Run a single fold (0..n_folds-1). "
                             "Default -1 runs all folds.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override default output directory.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    data_dir = config["data_dir"]
    n_genes = config.get("n_genes", 250)
    seed = config.get("training", {}).get("seed", 2021)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f"results/stnet/{dataset}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    # Discover samples from barcodes directory (same as TRIPLEX experiments)
    bc_dir = Path(data_dir) / "barcodes"
    if not bc_dir.exists():
        print(f"ERROR: {bc_dir} not found. Populate data directory first.")
        return
    sample_names = sorted([p.stem for p in bc_dir.glob("*.csv")])
    print(f"Dataset: {dataset}, {len(sample_names)} samples")

    folds = build_lopcv_folds(sample_names, dataset)
    n_folds = len(folds)
    print(f"{n_folds} LOPCV folds")

    all_fold_metrics = []

    fold_range = [args.fold] if args.fold >= 0 else range(n_folds)
    if args.fold >= 0 and not (0 <= args.fold < n_folds):
        raise ValueError(f"--fold {args.fold} out of range [0, {n_folds})")

    for fold_idx in fold_range:
        print(f"\n{'=' * 50}")
        print(f"FOLD {fold_idx}/{n_folds - 1} -- ST-Net baseline")
        print(f"{'=' * 50}")

        set_seed(seed)  # reset per fold for reproducibility

        train_loader, val_loader = build_loaders(
            data_dir, dataset, fold_idx, config, sample_names
        )

        model = STNet(
            n_genes=n_genes,
            pretrained=True,
            dropout=config.get("model", {}).get("stnet_dropout", 0.0),
        )

        if args.dry_run and fold_idx == 0:
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

            model = model.to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\nModel parameters: {n_params:,}")

            train_one_fold(
                model=model,
                model_type="stnet",
                train_loader=train_loader,
                val_loader=val_loader,
                config=config.get("training", {}),
                fold_idx=0,
                output_dir=output_dir,
                mcspr_artifacts=None,
                dry_run=True,
            )
            print("\nDry run complete.")
            return

        fold_result = train_one_fold(
            model=model,
            model_type="stnet",
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.get("training", {}),
            fold_idx=fold_idx,
            output_dir=output_dir,
            mcspr_artifacts=None,
            dry_run=False,
        )

        # Full evaluation with best checkpoint
        ckpt_path = output_dir / f"fold_{fold_idx}" / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model = model.to(device)
        val_metrics = _evaluate(model, "stnet", val_loader, device)

        with open(output_dir / f"fold_{fold_idx}" / "metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=2)

        all_fold_metrics.append(val_metrics)
        print(f"Fold {fold_idx}: PCC(M)={val_metrics['pcc_m']:.4f} "
              f"RVD={val_metrics['rvd']:.4f}")

    # Aggregate summary
    if all_fold_metrics:
        summary = _aggregate(all_fold_metrics, "stnet", dataset)
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 50}")
        print(f"FINAL RESULTS -- ST-Net on {dataset}")
        print(f"{'=' * 50}")
        print(f"  PCC(M): {summary.get('pcc_m_mean', 0):.4f} "
              f"+/- {summary.get('pcc_m_std', 0):.4f}")
        print(f"  MSE:    {summary.get('mse_mean', 0):.4f}")
        print(f"  RVD:    {summary.get('rvd_mean', 0):.4f}")
        print(f"\nSummary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
