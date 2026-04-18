"""Train HisToGene + MCSPR -- all LOPCV folds.

Graph-based ViT + MCSPR prior. Same whole-slide loader as baseline;
MCSPR loss is applied after the graph forward pass via universal_trainer.

Usage:
    python src/experiments/run_histogene_mcspr.py --config configs/her2st.yaml \\
        [--lambda_path results/lambda_selection/her2st/selected_lambda.json] \\
        [--dry_run]

Saves results to: results/histogene_mcspr/{dataset}/
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

from src.data.histogene_loaders import (
    build_histogene_loaders,
    build_lopcv_folds,
)
from src.models.histogene import HisToGene
from src.training.universal_trainer import train_one_fold, _evaluate


OOM_RETRIES = 3
OOM_DECREMENT = 128


def set_seed(seed: int = 2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_mcspr_artifacts(data_dir, fold_idx, n_modules, n_contexts):
    nmf_dir = Path(data_dir) / "nmf" / f"fold_{fold_idx}"
    return {
        "M_pinv": np.load(nmf_dir / "M_pinv.npy"),
        "C_prior": np.load(nmf_dir / "C_prior.npy"),
        "n_modules": n_modules,
        "n_contexts": n_contexts,
    }


def _aggregate(metrics_list, model_name, dataset, extra=None):
    keys = ["pcc_m", "pcc_m_std", "mse", "mae", "rvd", "q1_mse"]
    summary = {"model": model_name, "dataset": dataset,
               "n_folds": len(metrics_list)}
    if extra:
        summary.update(extra)
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))
    return summary


def _build_histogene(config):
    hg_cfg = config.get("model", {}).get("histogene", {})
    return HisToGene(
        n_genes=config.get("n_genes", 250),
        d_model=hg_cfg.get("d_model", config.get("model", {}).get("d_model", 512)),
        n_layers=hg_cfg.get("n_layers", 4),
        num_heads=hg_cfg.get("num_heads", 8),
        mlp_ratio=hg_cfg.get("mlp_ratio", 4.0),
        dropout=hg_cfg.get("dropout", 0.2),
        use_precomputed=hg_cfg.get("use_precomputed", False),
        d_feat=hg_cfg.get("d_feat", 512),
        build_spatial_graph=hg_cfg.get("build_spatial_graph", False),
        k_neighbors=hg_cfg.get("k_neighbors", 8),
    )


def _train_with_oom_guard(
    build_loaders_fn, build_model_fn, train_kwargs_fn, initial_max_spots,
):
    max_spots = initial_max_spots
    last_err = None
    for attempt in range(OOM_RETRIES):
        try:
            train_loader, val_loader = build_loaders_fn(max_spots)
            model = build_model_fn()
            return train_one_fold(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                **train_kwargs_fn(max_spots),
            ), (train_loader, val_loader, model, max_spots)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e)
            last_err = e
            if "out of memory" not in msg.lower():
                raise
            torch.cuda.empty_cache()
            new_spots = max(128, max_spots - OOM_DECREMENT)
            print(f"  OOM at max_spots={max_spots}; retry "
                  f"{attempt + 1}/{OOM_RETRIES} with max_spots={new_spots}")
            if new_spots == max_spots:
                break
            max_spots = new_spots
    raise RuntimeError(
        f"OOM persisted after {OOM_RETRIES} retries "
        f"(final max_spots={max_spots}): {last_err}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train HisToGene + MCSPR")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fold", type=int, default=-1,
                        help="Run a single fold. Default -1 runs all folds.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lambda_path", type=str, default=None,
                        help="Override default selected_lambda.json path.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.setdefault("mcspr", {})["enabled"] = True

    dataset = config["dataset"]
    data_dir = config["data_dir"]
    seed = config.get("training", {}).get("seed", 2021)
    mc = config.get("mcspr", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f"results/histogene_mcspr/{dataset}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    lambda_path = Path(args.lambda_path) if args.lambda_path else Path(
        f"results/lambda_selection/{dataset}/selected_lambda.json"
    )
    frozen_lambda = None
    if lambda_path.exists():
        with open(lambda_path) as f:
            frozen_lambda = float(json.load(f)["selected_lambda"])
        print(f"Frozen lambda: {frozen_lambda} (from {lambda_path})")
    elif args.dry_run:
        print(f"(dry-run) lambda file {lambda_path} not found; "
              f"will use config default lambda_max="
              f"{mc.get('lambda_max')}")
    else:
        raise FileNotFoundError(
            f"Lambda not selected yet. Run select_lambda.py first. "
            f"Expected: {lambda_path}"
        )

    set_seed(seed)

    feature_dir = Path(data_dir) / "features"
    sample_names = sorted([f.stem for f in feature_dir.glob("*.csv")])
    if not sample_names:
        print(f"ERROR: no feature files in {feature_dir}.")
        return
    print(f"Dataset: {dataset}, {len(sample_names)} samples (MCSPR enabled)")

    folds = build_lopcv_folds(sample_names, dataset)
    n_folds = len(folds)
    print(f"{n_folds} LOPCV folds")

    fold_range = [args.fold] if args.fold >= 0 else range(n_folds)
    if args.fold >= 0 and not (0 <= args.fold < n_folds):
        raise ValueError(f"--fold {args.fold} out of range [0, {n_folds})")

    initial_max_spots = config.get("training", {}).get("max_spots", 1024)
    all_fold_metrics = []

    for fold_idx in fold_range:
        print(f"\n{'=' * 50}")
        print(f"FOLD {fold_idx}/{n_folds - 1} -- HisToGene + MCSPR  "
              f"lambda={frozen_lambda}")
        print(f"{'=' * 50}")
        set_seed(seed)

        mcspr_artifacts = load_mcspr_artifacts(
            data_dir, fold_idx,
            mc.get("n_modules", 15),
            mc.get("n_contexts", 6),
        )
        print(f"  Loaded MCSPR artifacts: "
              f"M_pinv {mcspr_artifacts['M_pinv'].shape}, "
              f"C_prior {mcspr_artifacts['C_prior'].shape}")

        train_config = {**config.get("training", {}), **mc}
        if frozen_lambda is not None:
            train_config["lambda_max"] = frozen_lambda

        def _build_loaders(max_spots):
            loader_cfg = {
                **config,
                "max_spots": max_spots,
                "seed": seed,
                "num_workers": config.get("training", {}).get("num_workers", 2),
                "patch_size": config.get("patch_size", 224),
                "n_genes": config.get("n_genes", 250),
            }
            return build_histogene_loaders(
                data_dir, dataset, fold_idx, loader_cfg, context_dir=None,
            )

        def _build_model():
            return _build_histogene(config)

        def _train_kwargs(max_spots):
            return dict(
                model_type="histogene",
                config=train_config,
                fold_idx=fold_idx,
                output_dir=output_dir,
                mcspr_artifacts=mcspr_artifacts,
                dry_run=bool(args.dry_run),
            )

        if args.dry_run and fold_idx == fold_range[0]:
            train_loader, val_loader = _build_loaders(initial_max_spots)
            print(f"Train loader: {len(train_loader)} batches (whole-slide)")
            model = _build_model().to(device)
            print(f"HisToGene parameters: "
                  f"{sum(p.numel() for p in model.parameters()):,}")
            train_one_fold(
                model=model,
                model_type="histogene",
                train_loader=train_loader,
                val_loader=val_loader,
                config=train_config,
                fold_idx=fold_idx,
                output_dir=output_dir,
                mcspr_artifacts=mcspr_artifacts,
                dry_run=True,
            )
            print("Dry run complete (HisToGene + MCSPR).")
            return

        _fold_result, (_, val_loader, model, final_max_spots) = (
            _train_with_oom_guard(
                _build_loaders, _build_model, _train_kwargs,
                initial_max_spots,
            )
        )
        if final_max_spots != initial_max_spots:
            print(f"  (completed at max_spots={final_max_spots})")

        ckpt_path = output_dir / f"fold_{fold_idx}" / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            model = model.to(device)
        val_metrics = _evaluate(model, "histogene", val_loader, device)

        with open(output_dir / f"fold_{fold_idx}" / "metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=2)

        all_fold_metrics.append(val_metrics)
        print(f"Fold {fold_idx}: PCC(M)={val_metrics['pcc_m']:.4f} "
              f"RVD={val_metrics.get('rvd', 0.0):.4f}")

    if all_fold_metrics:
        summary = _aggregate(
            all_fold_metrics, "histogene_mcspr", dataset,
            extra={"frozen_lambda": frozen_lambda},
        )
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'=' * 50}")
        print(f"FINAL RESULTS -- HisToGene+MCSPR on {dataset}")
        print(f"{'=' * 50}")
        print(f"  PCC(M): {summary.get('pcc_m_mean', 0):.4f} "
              f"+/- {summary.get('pcc_m_std', 0):.4f}")
        print(f"  MSE:    {summary.get('mse_mean', 0):.4f}")
        print(f"  RVD:    {summary.get('rvd_mean', 0):.4f}")
        print(f"  Lambda: {frozen_lambda}")


if __name__ == "__main__":
    main()
