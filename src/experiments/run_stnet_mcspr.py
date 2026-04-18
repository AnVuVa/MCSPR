"""
Train ST-Net + MCSPR -- all LOPCV folds.

Usage:
    python src/experiments/run_stnet_mcspr.py --config configs/her2st.yaml [--dry_run]

Requires:
  - Lambda already selected (from select_lambda.py)
  - Precomputed NMF artifacts: data/{dataset}/nmf/fold_{i}/M_pinv.npy, C_prior.npy
  - Precomputed context weights: data/{dataset}/context_weights/{sample}.npy

Saves results to: results/stnet_mcspr/{dataset}/
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


def load_mcspr_artifacts(data_dir: str, fold_idx: int,
                         n_modules: int, n_contexts: int) -> dict:
    """Load precomputed NMF artifacts for this fold's training data."""
    nmf_dir = Path(data_dir) / "nmf" / f"fold_{fold_idx}"
    M_pinv = np.load(nmf_dir / "M_pinv.npy")
    C_prior = np.load(nmf_dir / "C_prior.npy")
    return {
        "M_pinv": M_pinv,
        "C_prior": C_prior,
        "n_modules": n_modules,
        "n_contexts": n_contexts,
    }


def _aggregate(metrics_list, model_name, dataset, frozen_lambda=None):
    keys = ["pcc_m", "pcc_m_std", "mse", "mae", "rvd", "q1_mse"]
    summary = {
        "model": model_name, "dataset": dataset,
        "n_folds": len(metrics_list),
        "frozen_lambda": frozen_lambda,
    }
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train ST-Net + MCSPR")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fold", type=int, default=-1,
                        help="Run a single fold (0..n_folds-1). "
                             "Default -1 runs all folds.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override default output directory.")
    parser.add_argument("--lambda_path", type=str, default=None,
                        help="Override default selected_lambda.json path.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Force MCSPR enabled
    config.setdefault("mcspr", {})["enabled"] = True

    dataset = config["dataset"]
    data_dir = config["data_dir"]
    n_genes = config.get("n_genes", 250)
    seed = config.get("training", {}).get("seed", 2021)
    mc = config.get("mcspr", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f"results/stnet_mcspr/{dataset}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen lambda (selected on Fold 1 internal 80/20 split)
    lambda_path = Path(args.lambda_path) if args.lambda_path else Path(
        f"results/lambda_selection/{dataset}/selected_lambda.json"
    )
    frozen_lambda = None
    if lambda_path.exists():
        with open(lambda_path) as f:
            lambda_config = json.load(f)
        frozen_lambda = float(lambda_config["selected_lambda"])
        print(f"Frozen lambda: {frozen_lambda} (from {lambda_path})")
    elif args.dry_run:
        print(f"(dry-run) lambda file {lambda_path} not found; "
              f"using config default lambda_max="
              f"{config.get('mcspr', {}).get('lambda_max')}")
    else:
        raise FileNotFoundError(
            f"Lambda not selected yet. Run select_lambda.py first.\n"
            f"Expected: {lambda_path}"
        )

    set_seed(seed)

    # Discover samples
    bc_dir = Path(data_dir) / "barcodes"
    if not bc_dir.exists():
        print(f"ERROR: {bc_dir} not found. Populate data directory first.")
        return
    sample_names = sorted([p.stem for p in bc_dir.glob("*.csv")])
    print(f"Dataset: {dataset}, {len(sample_names)} samples (MCSPR enabled)")

    folds = build_lopcv_folds(sample_names, dataset)
    n_folds = len(folds)
    print(f"{n_folds} LOPCV folds (identical to ST-Net baseline splits)")

    all_fold_metrics = []

    fold_range = [args.fold] if args.fold >= 0 else range(n_folds)
    if args.fold >= 0 and not (0 <= args.fold < n_folds):
        raise ValueError(f"--fold {args.fold} out of range [0, {n_folds})")

    for fold_idx in fold_range:
        print(f"\n{'=' * 50}")
        print(f"FOLD {fold_idx}/{n_folds - 1} -- ST-Net + MCSPR  "
              f"lambda={frozen_lambda}")
        print(f"{'=' * 50}")

        set_seed(seed)

        # CRITICAL: same fold splits as run_stnet.py
        train_loader, val_loader = build_loaders(
            data_dir, dataset, fold_idx, config, sample_names
        )

        mcspr_artifacts = load_mcspr_artifacts(
            data_dir=data_dir,
            fold_idx=fold_idx,
            n_modules=mc.get("n_modules", 15),
            n_contexts=mc.get("n_contexts", 6),
        )
        print(f"  Loaded MCSPR artifacts: M_pinv {mcspr_artifacts['M_pinv'].shape}, "
              f"C_prior {mcspr_artifacts['C_prior'].shape}")

        model = STNet(
            n_genes=n_genes,
            pretrained=True,
            dropout=config.get("model", {}).get("stnet_dropout", 0.0),
        )

        # Merge training config with MCSPR config, override lambda if loaded
        train_config = {**config.get("training", {}), **mc}
        if frozen_lambda is not None:
            train_config["lambda_max"] = frozen_lambda

        if args.dry_run and fold_idx == 0:
            print("\n--- Dry Run Checks (ST-Net+MCSPR) ---")
            model = model.to(device)
            train_one_fold(
                model=model,
                model_type="stnet",
                train_loader=train_loader,
                val_loader=val_loader,
                config=train_config,
                fold_idx=0,
                output_dir=output_dir,
                mcspr_artifacts=mcspr_artifacts,
                dry_run=True,
            )
            print("\nDry run complete (MCSPR).")
            return

        fold_result = train_one_fold(
            model=model,
            model_type="stnet",
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            fold_idx=fold_idx,
            output_dir=output_dir,
            mcspr_artifacts=mcspr_artifacts,
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
        summary = _aggregate(all_fold_metrics, "stnet_mcspr", dataset,
                             frozen_lambda=frozen_lambda)
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 50}")
        print(f"FINAL RESULTS -- ST-Net+MCSPR on {dataset}")
        print(f"{'=' * 50}")
        print(f"  PCC(M): {summary.get('pcc_m_mean', 0):.4f} "
              f"+/- {summary.get('pcc_m_std', 0):.4f}")
        print(f"  MSE:    {summary.get('mse_mean', 0):.4f}")
        print(f"  RVD:    {summary.get('rvd_mean', 0):.4f}")
        print(f"  Lambda: {frozen_lambda}")
        print(f"\nSummary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
