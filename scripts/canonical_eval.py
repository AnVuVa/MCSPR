"""Unified per-slide evaluation for all baselines.

Rationale: TRIPLEX/STNet/HisToGene were trained with three different
evaluator functions. evaluate_fold (TRIPLEX) groups by sample_idx and uses
scipy.stats.pearsonr; universal_trainer._evaluate (STNet/HisToGene) treats
each DataLoader batch as a "slide" and uses a vectorized Pearson. To compare
apples-to-apples we re-evaluate every fold's best_model.pt with one
canonical formula: per-slide grouping by sample_idx, scipy pearsonr,
NaN-safe std-guard, reporting PCC(M), PCC(H top-50), MSE, MAE, RVD.

Usage:
    python scripts/canonical_eval.py \
        --baseline triplex --config configs/her2st.yaml \
        --input_dir results/baselines/triplex

    # Run across all three baselines:
    python scripts/canonical_eval.py --all --config configs/her2st.yaml \
        --base_dir results/baselines
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import build_loaders, build_lopcv_folds, build_stnet_loaders
from src.data.histogene_loaders import build_histogene_loaders
from src.models.triplex import TRIPLEX
from src.models.stnet import STNet
from src.models.histogene import HisToGene


PATCH_BASED = ("triplex", "stnet")
GRAPH_BASED = ("histogene",)


def _per_gene_pcc(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Per-gene Pearson via scipy; returns 0 for zero-variance columns."""
    n_genes = y_hat.shape[1]
    pccs = np.zeros(n_genes, dtype=np.float64)
    for j in range(n_genes):
        if np.std(y_hat[:, j]) < 1e-10 or np.std(y_true[:, j]) < 1e-10:
            pccs[j] = 0.0
        else:
            r, _ = pearsonr(y_hat[:, j], y_true[:, j])
            pccs[j] = r if not np.isnan(r) else 0.0
    return pccs


def _per_slide_metrics(slide_preds, slide_trues):
    """Compute canonical metrics from per-sample grouped preds/trues."""
    slide_pcc_m, slide_mse, slide_mae, slide_rvd = [], [], [], []
    all_gene_pccs = []
    n_spots_per_slide = {}

    for sid in sorted(slide_preds.keys()):
        y_hat = np.stack(slide_preds[sid]).astype(np.float64)
        y_true = np.stack(slide_trues[sid]).astype(np.float64)
        n_spots_per_slide[sid] = y_hat.shape[0]

        gene_pccs = _per_gene_pcc(y_hat, y_true)
        slide_pcc_m.append(float(np.nanmean(gene_pccs)))
        all_gene_pccs.append(gene_pccs)

        slide_mse.append(float(np.mean((y_hat - y_true) ** 2)))
        slide_mae.append(float(np.mean(np.abs(y_hat - y_true))))

        var_pred = y_hat.var(axis=0)
        var_true = y_true.var(axis=0)
        rvd = float(np.mean((var_pred - var_true) ** 2 / (var_true ** 2 + 1e-8)))
        slide_rvd.append(rvd)

    mean_gene_pccs = np.nanmean(np.stack(all_gene_pccs), axis=0)
    top50_idx = np.argsort(mean_gene_pccs)[::-1][:50]
    pcc_h = float(np.nanmean(mean_gene_pccs[top50_idx]))
    pcc_h_std = float(np.nanstd(mean_gene_pccs[top50_idx]))

    return {
        "pcc_m": float(np.mean(slide_pcc_m)),
        "pcc_m_std": float(np.std(slide_pcc_m)),
        "pcc_h": pcc_h,
        "pcc_h_std": pcc_h_std,
        "mse": float(np.mean(slide_mse)),
        "mse_std": float(np.std(slide_mse)),
        "mae": float(np.mean(slide_mae)),
        "mae_std": float(np.std(slide_mae)),
        "rvd": float(np.mean(slide_rvd)),
        "rvd_std": float(np.std(slide_rvd)),
        "q1_mse": float(np.percentile(slide_mse, 25)),
        "n_slides": len(slide_pcc_m),
        "n_spots_per_slide": {
            str(k): v for k, v in n_spots_per_slide.items()
        },
        "top50_gene_indices": top50_idx.tolist(),
    }


def _extract_y_hat(preds, model_type):
    """Unify the heterogeneous return types of the three baselines."""
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(preds, dict):
        # Spec v2 A7: attach to 'output' (gene space) not 'fusion' (latent).
        # Use 'output' when present; fall back to 'fusion' for TRIPLEX when
        # output is absent on older checkpoints.
        return preds.get("output", preds.get("fusion"))
    return preds


def _build_val_loader(baseline, config, fold_idx, data_dir, dataset, sample_names):
    if baseline == "triplex":
        _, val_loader = build_loaders(
            data_dir, dataset, fold_idx, config, sample_names,
        )
    elif baseline == "stnet":
        _, val_loader = build_stnet_loaders(
            data_dir, dataset, fold_idx, config, sample_names,
        )
    elif baseline == "histogene":
        loader_cfg = {
            **config,
            "max_spots": config.get("training", {}).get(
                "histogene_val_max_spots", 712,
            ),
            "val_max_spots": config.get("training", {}).get(
                "histogene_val_max_spots", 712,
            ),
            "seed": config.get("training", {}).get("seed", 2021),
            "num_workers": 0,
            "patch_size": config.get("patch_size", 224),
            "n_genes": config.get("n_genes", 300),
        }
        _, val_loader = build_histogene_loaders(
            data_dir, dataset, fold_idx, loader_cfg, context_dir=None,
        )
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    return val_loader


def _build_model(baseline, config):
    n_genes = config.get("n_genes", 300)
    if baseline == "triplex":
        return TRIPLEX(config, n_genes=n_genes)
    if baseline == "stnet":
        return STNet(
            n_genes=n_genes,
            pretrained=False,  # weights loaded from checkpoint
            dropout=config.get("model", {}).get("stnet_dropout", 0.0),
        )
    if baseline == "histogene":
        hg_cfg = config.get("model", {}).get("histogene", {})
        return HisToGene(
            n_genes=n_genes,
            d_model=hg_cfg.get(
                "d_model", config.get("model", {}).get("d_model", 512)
            ),
            n_layers=hg_cfg.get("n_layers", 4),
            num_heads=hg_cfg.get("num_heads", 8),
            mlp_ratio=hg_cfg.get("mlp_ratio", 4.0),
            dropout=hg_cfg.get("dropout", 0.2),
            use_precomputed=hg_cfg.get("use_precomputed", False),
            d_feat=hg_cfg.get("d_feat", 512),
            build_spatial_graph=hg_cfg.get("build_spatial_graph", False),
            k_neighbors=hg_cfg.get("k_neighbors", 8),
            use_grad_checkpoint=hg_cfg.get("use_grad_checkpoint", True),
            cnn_chunk_size=hg_cfg.get("cnn_chunk_size", 128),
        )
    raise ValueError(f"Unknown baseline: {baseline}")


def _load_checkpoint(model, ckpt_path, baseline):
    """Handle the two checkpoint formats across baselines."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state)
    return model


def _run_patch_based(model, val_loader, device):
    """Collect per-sample predictions for TRIPLEX / STNet."""
    slide_preds, slide_trues = {}, {}
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            preds = model(batch)
            y_hat = _extract_y_hat(preds, None).cpu().numpy()
            y_true = batch["expression"].cpu().numpy()
            sids = batch["sample_idx"].cpu().numpy()
            for i in range(y_hat.shape[0]):
                sid = int(sids[i])
                slide_preds.setdefault(sid, []).append(y_hat[i])
                slide_trues.setdefault(sid, []).append(y_true[i])
    return slide_preds, slide_trues


def _run_graph_based(model, val_loader, device):
    """Collect per-sample predictions for HisToGene (one slide per batch)."""
    slide_preds, slide_trues = {}, {}
    model.eval()
    with torch.no_grad():
        for batch_list in val_loader:
            for slide_dict in batch_list:
                patches = slide_dict["patches"].to(device)
                grid_norm = slide_dict["grid_norm"].to(device)
                Y_true = slide_dict["expression"].cpu().numpy()
                Y_hat, _ = model(patches, grid_norm)
                Y_hat = Y_hat.cpu().numpy()
                sid = slide_dict["sample_name"]
                # graph-based has no sample_idx — use sample_name string as key
                for i in range(Y_hat.shape[0]):
                    slide_preds.setdefault(sid, []).append(Y_hat[i])
                    slide_trues.setdefault(sid, []).append(Y_true[i])
    return slide_preds, slide_trues


def _evaluate_merge_from_preds(baseline, config, input_dir):
    """Evaluate MERGE from its saved per-slide predictions.

    MERGE saves preds at <output_dir>/preds/{slide}.npy of shape
    (n_spots_total_for_slide, n_genes), with zero rows for spots whose count
    sum was zero (those were filtered out of training; MERGE backfills with
    np.zeros when writing). We exclude those rows before scoring so canonical
    metrics aren't artificially inflated by predicted-vs-true zeros.

    Folds are inferred from the manifest at config['merge_split_manifest'] OR
    from the per-fold directories in input_dir. Ground truth is read from the
    canonical 300-SVG counts at <data_dir>/counts_svg/{slide}.npy.
    """
    import json as _json

    data_dir = Path(config["data_dir"])
    dataset = config["dataset"]
    counts_dir = data_dir / "counts_svg"

    # Resolve manifest — prefer explicit override, else MERGE's standard path
    manifest_path = (
        config.get("merge_split_manifest")
        or "/mnt/d/docker_machine/anvuva/MERGE/data/her2st/splits/"
           "her2st_patient_split_manifest_4fold.json"
    )
    with open(manifest_path) as f:
        manifest = _json.load(f)
    folds = manifest["folds"]
    n_folds = len(folds)
    print(f"\n=== Canonical eval: {baseline} (from preds) ===")
    print(f"Dataset={dataset} | n_folds={n_folds} | manifest={manifest_path}")

    all_fold_metrics = []
    input_dir = Path(input_dir)
    for fld in folds:
        fold_idx = fld["fold"]
        val_slides = fld["val_slides"]
        # MERGE writes preds to <fold_output>/preds/{slide}.npy where
        # fold_output is one level above the run subdirectory (0/cnn etc).
        # Try both layouts.
        candidate_pred_dirs = [
            input_dir / f"fold_{fold_idx}" / "preds",
            input_dir / f"fold_{fold_idx}" / "0" / "preds",
            input_dir / "preds",  # in case all folds dump to same dir
        ]
        pred_dir = next((p for p in candidate_pred_dirs if p.exists()), None)
        if pred_dir is None:
            print(f"  Fold {fold_idx}: SKIP (no preds dir found in "
                  f"{[str(p) for p in candidate_pred_dirs]})")
            continue

        slide_preds, slide_trues = {}, {}
        for slide in val_slides:
            pred_path = pred_dir / f"{slide}.npy"
            true_path = counts_dir / f"{slide}.npy"
            if not pred_path.exists() or not true_path.exists():
                print(f"  Fold {fold_idx}: missing {slide} (pred={pred_path.exists()}, "
                      f"true={true_path.exists()})")
                continue
            y_hat_full = np.load(str(pred_path)).astype(np.float64)
            y_true_full = np.load(str(true_path)).astype(np.float64)
            if y_hat_full.shape != y_true_full.shape:
                print(f"  Fold {fold_idx}: shape mismatch {slide} pred="
                      f"{y_hat_full.shape} true={y_true_full.shape}")
                continue
            # Drop rows where the truth was all-zero (MERGE's nonzero filter).
            mask = y_true_full.sum(axis=1) > 0
            y_hat = y_hat_full[mask]
            y_true = y_true_full[mask]
            if y_hat.shape[0] == 0:
                continue
            slide_preds[slide] = list(y_hat)
            slide_trues[slide] = list(y_true)

        if not slide_preds:
            continue

        metrics = _per_slide_metrics(slide_preds, slide_trues)
        metrics["fold"] = fold_idx
        all_fold_metrics.append(metrics)
        out_dir = (input_dir / f"fold_{fold_idx}")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "canonical_metrics.json", "w") as f:
            _json.dump(metrics, f, indent=2)
        print(
            f"  Fold {fold_idx}: PCC(M)={metrics['pcc_m']:.4f}±"
            f"{metrics['pcc_m_std']:.4f} PCC(H)={metrics['pcc_h']:.4f} "
            f"MSE={metrics['mse']:.4f} MAE={metrics['mae']:.4f} "
            f"n_slides={metrics['n_slides']}"
        )

    if not all_fold_metrics:
        return None

    keys = ["pcc_m", "pcc_m_std", "pcc_h", "pcc_h_std",
            "mse", "mae", "rvd", "q1_mse"]
    summary = {
        "baseline": baseline,
        "dataset": dataset,
        "n_folds_evaluated": len(all_fold_metrics),
        "per_fold": all_fold_metrics,
    }
    for k in keys:
        vals = [m[k] for m in all_fold_metrics if k in m]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std_across_folds"] = float(np.std(vals))

    with open(input_dir / "canonical_summary.json", "w") as f:
        _json.dump(summary, f, indent=2)
    print(
        f"  SUMMARY: PCC(M)={summary['pcc_m_mean']:.4f}±"
        f"{summary['pcc_m_std_across_folds']:.4f} "
        f"PCC(H)={summary['pcc_h_mean']:.4f} "
        f"MSE={summary['mse_mean']:.4f} MAE={summary['mae_mean']:.4f}"
    )
    return summary


def evaluate_baseline(baseline, config, input_dir, device):
    if baseline == "merge":
        return _evaluate_merge_from_preds(baseline, config, input_dir)

    data_dir = config["data_dir"]
    dataset = config["dataset"]
    n_folds = config.get("n_folds", 4)

    if baseline in ("triplex", "stnet"):
        bc_dir = Path(data_dir) / "barcodes"
        sample_names = sorted([p.stem for p in bc_dir.glob("*.csv")])
    else:
        feat_dir = Path(data_dir) / "features"
        sample_names = sorted([p.stem for p in feat_dir.glob("*.csv")])

    print(f"\n=== Canonical eval: {baseline} ===")
    print(f"Dataset={dataset} | n_folds={n_folds} | samples={len(sample_names)}")

    all_fold_metrics = []
    for fold_idx in range(n_folds):
        fold_dir = Path(input_dir) / f"fold_{fold_idx}"
        ckpt_path = fold_dir / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  Fold {fold_idx}: SKIP (no best_model.pt)")
            continue

        val_loader = _build_val_loader(
            baseline, config, fold_idx, data_dir, dataset, sample_names,
        )
        model = _build_model(baseline, config)
        model = _load_checkpoint(model, ckpt_path, baseline).to(device)

        if baseline in PATCH_BASED:
            slide_preds, slide_trues = _run_patch_based(model, val_loader, device)
        else:
            slide_preds, slide_trues = _run_graph_based(model, val_loader, device)

        metrics = _per_slide_metrics(slide_preds, slide_trues)
        metrics["fold"] = fold_idx
        all_fold_metrics.append(metrics)

        with open(fold_dir / "canonical_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(
            f"  Fold {fold_idx}: PCC(M)={metrics['pcc_m']:.4f}±"
            f"{metrics['pcc_m_std']:.4f} PCC(H)={metrics['pcc_h']:.4f} "
            f"MSE={metrics['mse']:.4f} n_slides={metrics['n_slides']}"
        )

        del model
        torch.cuda.empty_cache()

    if not all_fold_metrics:
        return None

    keys = ["pcc_m", "pcc_m_std", "pcc_h", "pcc_h_std", "mse", "mae", "rvd", "q1_mse"]
    summary = {
        "baseline": baseline,
        "dataset": dataset,
        "n_folds_evaluated": len(all_fold_metrics),
        "per_fold": all_fold_metrics,
    }
    for k in keys:
        vals = [m[k] for m in all_fold_metrics if k in m]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std_across_folds"] = float(np.std(vals))

    with open(Path(input_dir) / "canonical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"  SUMMARY: PCC(M)={summary['pcc_m_mean']:.4f}±"
        f"{summary['pcc_m_std_across_folds']:.4f} "
        f"PCC(H)={summary['pcc_h_mean']:.4f} MSE={summary['mse_mean']:.4f}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--baseline", type=str,
                        choices=["triplex", "stnet", "histogene", "merge"])
    parser.add_argument("--input_dir", type=str,
                        help="Directory containing fold_{i}/best_model.pt")
    parser.add_argument("--all", action="store_true",
                        help="Run all three baselines.")
    parser.add_argument("--base_dir", type=str, default="results/baselines",
                        help="Parent dir when --all (expects "
                             "base_dir/{triplex,stnet,histogene}).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.all:
        base = Path(args.base_dir)
        for baseline in ("triplex", "stnet", "histogene"):
            input_dir = base / baseline
            if not input_dir.exists():
                print(f"Missing: {input_dir} — skipping")
                continue
            evaluate_baseline(baseline, config, str(input_dir), device)
    else:
        if not args.baseline or not args.input_dir:
            parser.error("--baseline and --input_dir are required "
                         "when --all is not set.")
        evaluate_baseline(args.baseline, config, args.input_dir, device)


if __name__ == "__main__":
    main()
