"""Train TRIPLEX+MCSPR across all LOPCV folds.

Usage: python src/experiments/run_triplex_mcspr.py --config configs/her2st.yaml [--dry_run]
Saves results to results/triplex_mcspr/{dataset}/

REQUIRES: precomputed NMF and C_prior (from run_triplex.py or precompute_all()).
Same LOPCV splits and seed as TRIPLEX for fair comparison.
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
    parser = argparse.ArgumentParser(description="Train TRIPLEX+MCSPR")
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
        "--lambda_path", type=str, default=None,
        help="Path to selected_lambda.json (overrides default)."
    )
    parser.add_argument(
        "--n_contexts", type=int, default=None,
        help="Override config n_contexts. n_contexts=1 collapses "
             "C_prior to a single global prior (ablation)."
    )
    parser.add_argument(
        "--no_ema", action="store_true",
        help="Disable EMA on module moments (ablation: set beta=0.0)."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Force MCSPR enabled
    config.setdefault("mcspr", {})["enabled"] = True

    # CLI overrides for ablations
    if args.n_contexts is not None:
        config["mcspr"]["n_contexts"] = int(args.n_contexts)
    if args.no_ema:
        config["mcspr"]["beta"] = 0.0   # EMA decay=0 -> no moment smoothing

    # Load frozen lambda if available (CLI override takes precedence)
    dataset = config["dataset"]
    lambda_path = (
        Path(args.lambda_path) if args.lambda_path
        else Path(f"results/lambda_selection/{dataset}/selected_lambda.json")
    )
    if lambda_path.exists():
        with open(lambda_path) as lf:
            _lam_cfg = json.load(lf)
        config["mcspr"]["lambda_max"] = float(_lam_cfg["selected_lambda"])
        print(f"Using lambda={_lam_cfg['selected_lambda']} from {lambda_path}")
    elif args.dry_run:
        print(f"(dry-run) lambda file {lambda_path} not found; "
              f"using config default lambda_max="
              f"{config['mcspr'].get('lambda_max')}")

    data_dir = config["data_dir"]
    n_genes = config.get("n_genes", 250)
    seed = config.get("training", {}).get("seed", 2021)
    mc = config.get("mcspr", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)

    # Discover samples
    bc_dir = Path(data_dir) / "barcodes"
    if not bc_dir.exists():
        print(f"ERROR: {bc_dir} not found. Populate data directory first.")
        return

    sample_names = sorted([p.stem for p in bc_dir.glob("*.csv")])
    print(f"Dataset: {dataset}, {len(sample_names)} samples (MCSPR enabled)")

    # Precompute if needed
    gf_dir = Path(data_dir) / "global_features"
    if not gf_dir.exists() or len(list(gf_dir.glob("*.npy"))) == 0:
        print("\n=== Running precomputation ===")
        precompute_all(data_dir, dataset, config)

    # Build folds — SAME splits as TRIPLEX (same seed, same function)
    folds = build_lopcv_folds(sample_names, dataset)
    n_folds = len(folds)
    print(f"\n{n_folds} LOPCV folds (identical to TRIPLEX splits)")

    output_dir = (
        args.output_dir if args.output_dir
        else f"results/triplex_mcspr/{dataset}"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fold_results = []

    fold_range = [args.fold] if args.fold >= 0 else range(n_folds)
    if args.fold >= 0 and not (0 <= args.fold < n_folds):
        raise ValueError(f"--fold {args.fold} out of range [0, {n_folds})")

    for fold_idx in fold_range:
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{n_folds-1} (TRIPLEX+MCSPR)")
        print(f"{'='*60}")

        # Load precomputed MCSPR artifacts for this fold
        split_id = f"fold_{fold_idx}"
        nmf_dir = Path(data_dir) / "nmf" / split_id

        if not nmf_dir.exists():
            print(
                f"  NMF artifacts not found at {nmf_dir}. "
                f"Run precompute_all() first."
            )
            # Attempt precomputation for this fold
            from src.data.precompute import (
                precompute_context_clusters,
                precompute_nmf_and_prior,
            )

            train_samples = folds[fold_idx][0]
            precompute_context_clusters(
                data_dir, dataset, train_samples, split_id,
                mc.get("n_contexts", 6),
            )
            precompute_nmf_and_prior(
                data_dir, dataset, split_id, train_samples,
                mc.get("n_modules", 15), mc.get("n_contexts", 6),
            )

        M_pinv = np.load(str(nmf_dir / "M_pinv.npy"))
        C_prior = np.load(str(nmf_dir / "C_prior.npy"))

        # Ablation: collapse to a global (T=1) prior by averaging over contexts.
        if int(mc.get("n_contexts", 6)) != C_prior.shape[0]:
            target_T = int(mc.get("n_contexts", 6))
            if target_T == 1:
                C_prior = C_prior.mean(axis=0, keepdims=True)
                print(f"  Ablation: C_prior collapsed to global "
                      f"(T=1), shape={C_prior.shape}")
            else:
                raise ValueError(
                    f"--n_contexts={target_T} cannot be derived from "
                    f"precomputed C_prior shape {C_prior.shape}. "
                    f"Only T=1 collapse is supported without re-running precompute."
                )

        mcspr_artifacts = {
            "M_pinv": M_pinv,
            "C_prior": C_prior,
            "n_modules": mc.get("n_modules", 15),
            "n_contexts": mc.get("n_contexts", 6),
        }
        print(
            f"  Loaded MCSPR artifacts: M_pinv {M_pinv.shape}, "
            f"C_prior {C_prior.shape}"
        )

        # Same seed as TRIPLEX for identical weight init
        set_seed(seed)

        train_loader, val_loader = build_loaders(
            data_dir, dataset, fold_idx, config, sample_names
        )

        if args.dry_run and fold_idx == 0:
            print("\n--- Dry Run Checks (TRIPLEX+MCSPR) ---")
            batch = next(iter(train_loader))
            print(f"\nBatch shapes:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape} {v.dtype}")

            model = TRIPLEX(config, n_genes=n_genes)
            model = model.to(device)

            train_one_fold(
                model, train_loader, val_loader, config,
                fold_idx=0, output_dir=output_dir,
                device=device,
                mcspr_artifacts=mcspr_artifacts,
                dry_run=True,
            )
            print("\nDry run complete (MCSPR).")
            return

        model = TRIPLEX(config, n_genes=n_genes)
        metrics = train_one_fold(
            model, train_loader, val_loader, config,
            fold_idx=fold_idx, output_dir=output_dir,
            device=device,
            mcspr_artifacts=mcspr_artifacts,
            gene_names=train_loader.dataset.gene_names,
        )
        fold_results.append(metrics)

    # Aggregate
    if fold_results:
        summary = {
            "dataset": dataset,
            "n_folds": n_folds,
            "mcspr_enabled": True,
            "pcc_m": float(np.mean([r["pcc_m"] for r in fold_results])),
            "pcc_m_std": float(np.std([r["pcc_m"] for r in fold_results])),
            "pcc_h": float(np.mean([r["pcc_h"] for r in fold_results])),
            "pcc_h_std": float(np.std([r["pcc_h"] for r in fold_results])),
            "mse": float(np.mean([r["mse"] for r in fold_results])),
            "mse_std": float(np.std([r["mse"] for r in fold_results])),
            "mae": float(np.mean([r["mae"] for r in fold_results])),
            "mae_std": float(np.std([r["mae"] for r in fold_results])),
            "smcs_overall": float(
                np.mean(
                    [
                        r.get("smcs_overall", 0.0) or 0.0
                        for r in fold_results
                    ]
                )
            ),
            "per_fold": [
                {
                    "pcc_m": r["pcc_m"],
                    "pcc_h": r["pcc_h"],
                    "mse": r["mse"],
                    "mae": r["mae"],
                    "smcs": r.get("smcs_overall"),
                }
                for r in fold_results
            ],
        }

        with open(f"{output_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS — TRIPLEX+MCSPR on {dataset}")
        print(f"{'='*60}")
        print(f"  PCC(M): {summary['pcc_m']:.4f} +/- {summary['pcc_m_std']:.4f}")
        print(f"  PCC(H): {summary['pcc_h']:.4f} +/- {summary['pcc_h_std']:.4f}")
        print(f"  MSE:    {summary['mse']:.4f} +/- {summary['mse_std']:.4f}")
        print(f"  MAE:    {summary['mae']:.4f} +/- {summary['mae_std']:.4f}")
        print(f"  SMCS:   {summary['smcs_overall']:.4f}")


if __name__ == "__main__":
    main()
