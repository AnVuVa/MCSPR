"""Canonical per-slide evaluation for HydraTRIPLEX (PCC(M)/PCC(H)/MSE/MAE).

Reuses the exact `_per_slide_metrics` calculator as canonical_eval.py
(STNet/HisToGene/TRIPLEX/MERGE/HydraSTNet).

Usage:
  python scripts/canonical_eval_triplex_hydra.py \
      --config configs/her2st.yaml \
      --input_dir results/baselines/triplex_hydra \
      [--folds 0,1,2,3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.canonical_eval import _per_slide_metrics  # noqa: E402
from src.data.loaders import build_triplex_hydra_loaders  # noqa: E402
from src.models.triplex_hydra import HydraTRIPLEX  # noqa: E402
from src.training.hydra_helpers import load_registry, verify_modules  # noqa: E402


def _registry_path_for_fold(fold: int) -> Path:
    return Path(
        f"results/ablation/kmeans_y_elbow/fold_{fold}/modules_fold{fold}.json"
    )


def _eval_one_fold(
    fold: int, config: dict, input_dir: Path, device: torch.device
) -> dict | None:
    fold_dir = input_dir / f"fold_{fold}"
    ckpt_path = fold_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  Fold {fold}: SKIP (no best_model.pt at {ckpt_path})")
        return None

    registry = load_registry(_registry_path_for_fold(fold))
    train_loader, val_loader = build_triplex_hydra_loaders(
        data_dir=config["data_dir"], dataset=config["dataset"],
        fold_idx=fold, config=config,
    )
    idx_list = verify_modules(val_loader.dataset, registry, fold_idx=fold)
    module_sizes = [len(idx) for idx in idx_list]

    model = HydraTRIPLEX(
        config=config, module_sizes=module_sizes, idx_list=idx_list,
        d_hidden=config.get("model", {}).get("hydra_d_hidden", 128),
    ).to(device)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sample_names = val_loader.dataset.sample_names

    slide_preds_lists: dict[str, list[np.ndarray]] = {}
    slide_trues_lists: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for batch in val_loader:
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            preds, _ = model(batch_t)
            full = preds["fusion"].cpu().numpy()
            y_true = batch_t["expression"].cpu().numpy()
            sids = batch_t["sample_idx"].cpu().numpy()
            for i in range(full.shape[0]):
                sid = int(sids[i])
                slide_name = sample_names[sid]
                slide_preds_lists.setdefault(slide_name, []).append(full[i])
                slide_trues_lists.setdefault(slide_name, []).append(y_true[i])

    metrics = _per_slide_metrics(slide_preds_lists, slide_trues_lists)
    metrics["fold"] = fold
    with open(fold_dir / "canonical_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"  Fold {fold}: PCC(M)={metrics['pcc_m']:.4f}±"
        f"{metrics['pcc_m_std']:.4f} PCC(H)={metrics['pcc_h']:.4f} "
        f"MSE={metrics['mse']:.4f} MAE={metrics['mae']:.4f} "
        f"n_slides={metrics['n_slides']}"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    input_dir = Path(args.input_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    folds = [int(f) for f in args.folds.split(",")]

    print(f"=== Canonical eval: triplex_hydra ===")
    print(f"Dataset={config['dataset']} | input_dir={input_dir} | device={device}")

    all_fold = []
    for f in folds:
        m = _eval_one_fold(f, config, input_dir, device)
        if m is not None:
            all_fold.append(m)

    if not all_fold:
        print("No folds evaluated."); return

    keys = ["pcc_m", "pcc_m_std", "pcc_h", "pcc_h_std",
            "mse", "mae", "rvd", "q1_mse"]
    summary = {
        "baseline": "triplex_hydra",
        "dataset": config["dataset"],
        "n_folds_evaluated": len(all_fold),
        "per_fold": all_fold,
    }
    for k in keys:
        vals = [m[k] for m in all_fold]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std_across_folds"] = float(np.std(vals))

    out = input_dir / "canonical_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\n  SUMMARY: PCC(M)={summary['pcc_m_mean']:.4f}±"
        f"{summary['pcc_m_std_across_folds']:.4f} "
        f"PCC(H)={summary['pcc_h_mean']:.4f} "
        f"MSE={summary['mse_mean']:.4f} MAE={summary['mae_mean']:.4f}"
    )
    print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
