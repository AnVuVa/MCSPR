"""Load each completed checkpoint, run validation inference, save
test_predictions.npz alongside best_model.pt.

Each npz file contains:
  y_hat       (N_spots, n_genes)
  y_true      (N_spots, n_genes)
  slide_sizes (N_slides,) — number of spots per slide, so per-slide views
                           can be reconstructed via np.split.

Runs on CPU by default to avoid contending with training orchestrator.
Skips folds that already have a current cache.

Usage:
    python scripts/run_inference.py --config configs/her2st.yaml
    python scripts/run_inference.py --config configs/her2st.yaml --force
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import build_loaders
from src.models.triplex import TRIPLEX
from src.models.stnet import STNet


ARCHS = ["triplex", "triplex_mcspr", "stnet", "stnet_mcspr"]


def build_model(arch, config):
    n_genes = config.get("n_genes", 300)
    if arch.startswith("triplex"):
        return TRIPLEX(config, n_genes=n_genes)
    if arch.startswith("stnet"):
        mc = config.get("model", {}).get("stnet", {})
        return STNet(
            n_genes=n_genes,
            pretrained=mc.get("pretrained", True),
            dropout=mc.get("dropout", 0.2),
        )
    raise ValueError(arch)


def load_state(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    return model


def run_inference(model, val_loader, device):
    model.eval()
    yhat_list, ytrue_list, sid_list, batch_sizes = [], [], [], []
    sample_idx_available = None
    with torch.no_grad():
        for batch in val_loader:
            slide = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            preds = model(slide)
            if isinstance(preds, tuple):
                preds = preds[0]
            if isinstance(preds, dict):
                yh = preds.get("fusion", preds.get("output"))
            else:
                yh = preds
            yh_np = yh.cpu().numpy()
            yt_np = slide["expression"].cpu().numpy()
            yhat_list.append(yh_np)
            ytrue_list.append(yt_np)
            batch_sizes.append(yh_np.shape[0])
            if "sample_idx" in slide:
                sid_list.append(slide["sample_idx"].cpu().numpy())
                sample_idx_available = True
            elif sample_idx_available is None:
                sample_idx_available = False
    y_hat = np.concatenate(yhat_list, axis=0)
    y_true = np.concatenate(ytrue_list, axis=0)
    batch_sizes = np.asarray(batch_sizes, dtype=np.int64)
    if sample_idx_available:
        sample_idx = np.concatenate(sid_list, axis=0).astype(np.int64)
    else:
        # Fall back: one sample_idx per BATCH (matches universal_trainer
        # assumption that non-graph batches = slides).
        sample_idx = np.repeat(np.arange(len(batch_sizes)), batch_sizes).astype(np.int64)
    return y_hat, y_true, sample_idx, batch_sizes


def find_completed_folds(dataset):
    tasks = []
    for arch in ARCHS:
        for fold in range(8):
            log = Path(f"results/{arch}/{dataset}/fold_{fold}/training_log.json")
            ckpt = Path(f"results/{arch}/{dataset}/fold_{fold}/best_model.pt")
            if log.exists() and ckpt.exists():
                tasks.append((arch, fold, str(ckpt)))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing prediction caches")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    dataset = config["dataset"]
    data_dir = config["data_dir"]

    device = torch.device(args.device)
    tasks = find_completed_folds(dataset)
    print(f"Found {len(tasks)} completed folds. Device: {device}\n")

    ok, skip, fail = 0, 0, 0
    for arch, fold, ckpt_path in tasks:
        fold_dir = Path(ckpt_path).parent
        cache_path = fold_dir / "test_predictions.npz"
        label = f"{arch:<16} fold {fold}"
        if cache_path.exists() and not args.force:
            print(f"  SKIP {label}  (cache exists: {cache_path})")
            skip += 1
            continue

        try:
            _, val_loader = build_loaders(
                data_dir=data_dir, dataset=dataset, fold_idx=fold,
                config=config,
            )
            model = build_model(arch, config)
            model = load_state(model, ckpt_path)
            model = model.to(device)
            y_hat, y_true, sample_idx, batch_sizes = run_inference(model, val_loader, device)
            n_slides = int(len(np.unique(sample_idx)))
            np.savez(cache_path,
                     y_hat=y_hat, y_true=y_true,
                     sample_idx=sample_idx, batch_sizes=batch_sizes)
            print(f"  OK   {label}  spots={y_hat.shape[0]:>5} slides={n_slides} "
                  f"batches={len(batch_sizes)} -> {cache_path}")
            ok += 1
            del model, y_hat, y_true, val_loader
        except Exception as e:
            print(f"  FAIL {label}  {type(e).__name__}: {e}")
            fail += 1

    print(f"\nDone: {ok} written, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    main()
