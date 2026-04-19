"""Recompute PCC(M)/PCC(H)/PCC-10/PCC-50/PCC-200 for each lambda's best checkpoint.

Uses the same Fold-1 internal 80/20 split as select_lambda.py so metrics are
comparable to the values in selected_lambda.json.

Top-K definitions follow the TRIPLEX paper:
  PCC(M):   mean per-gene PCC over all 300 genes (pooled across val spots)
  PCC(H):   mean per-gene PCC over top-50 genes ranked by PCC (synonym with PCC-50)
  PCC-K:    mean per-gene PCC over top-K genes ranked by PCC

Run:
    python scripts/eval_lambda_topk.py --config configs/her2st.yaml
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

from src.data.dataset import STDataset
from src.data.loaders import build_lopcv_folds, slide_collate_fn, SlideBatchSampler
from src.models.triplex import TRIPLEX
from src.experiments.select_lambda import (
    FOLD_FOR_SELECTION,
    split_internal,
)
from torch.utils.data import DataLoader


TOP_K_LIST = [10, 50, 200]


def build_val_loader(config: dict, val_samples, data_dir, dataset, fold_idx):
    base = Path(data_dir)
    ctx_dir = base / "context_weights" / f"fold_{fold_idx}"
    if not ctx_dir.exists():
        ctx_dir = None
    gf_dir = base / "global_features"
    if not gf_dir.exists():
        gf_dir = None
    tc = config.get("training", {})
    mc = config.get("mcspr", {})

    val_ds = STDataset(
        data_dir=data_dir,
        dataset=dataset,
        sample_names=val_samples,
        n_genes=config.get("n_genes", 250),
        augment=False,
        n_contexts=mc.get("n_contexts", 6),
        context_dir=str(ctx_dir) if ctx_dir else None,
        global_feat_dir=str(gf_dir) if gf_dir else None,
    )
    sampler = SlideBatchSampler(
        val_ds, batch_size=tc.get("batch_size", 128),
        shuffle=False, drop_last=False,
    )
    return DataLoader(
        val_ds, batch_sampler=sampler,
        collate_fn=slide_collate_fn, num_workers=0, pin_memory=False,
    )


def per_gene_pcc(Y_hat: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
    n = Y_hat.shape[1]
    out = np.zeros(n)
    for j in range(n):
        if np.std(Y_hat[:, j]) < 1e-10 or np.std(Y_true[:, j]) < 1e-10:
            out[j] = 0.0
        else:
            out[j], _ = pearsonr(Y_hat[:, j], Y_true[:, j])
    return out


def run_inference(model, val_loader, device):
    model.eval()
    slide_yhat = {}
    slide_ytrue = {}
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
            yh = yh.cpu().numpy()
            yt = slide["expression"].cpu().numpy()
            s_idx = slide["sample_idx"].cpu().numpy()
            for i in range(yh.shape[0]):
                sid = int(s_idx[i])
                slide_yhat.setdefault(sid, []).append(yh[i])
                slide_ytrue.setdefault(sid, []).append(yt[i])
    return slide_yhat, slide_ytrue


def compute_topk_metrics(slide_yhat, slide_ytrue):
    # Pool spots across slides then compute per-gene PCC, as TRIPLEX paper.
    ids = sorted(slide_yhat.keys())
    Yh = np.concatenate([np.stack(slide_yhat[s]) for s in ids], axis=0)
    Yt = np.concatenate([np.stack(slide_ytrue[s]) for s in ids], axis=0)

    gene_pcc = per_gene_pcc(Yh, Yt)  # (n_genes,)

    result = {
        "pcc_m": float(np.nanmean(gene_pcc)),
        "pcc_m_std": float(np.nanstd(gene_pcc)),
    }
    order = np.argsort(gene_pcc)[::-1]  # descending
    for k in TOP_K_LIST:
        k_eff = min(k, len(gene_pcc))
        top = gene_pcc[order[:k_eff]]
        result[f"pcc_{k}"] = float(np.nanmean(top))
    # PCC(H) == PCC-50 per TRIPLEX convention
    result["pcc_h"] = result["pcc_50"]
    return result, gene_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    data_dir = config["data_dir"]
    out_dir = Path(f"results/lambda_selection/{dataset}")

    # Rebuild the same Fold-1 internal 80/20 val set
    bc_dir = Path(data_dir) / "barcodes"
    sample_names = sorted([p.stem for p in bc_dir.glob("*.csv")])
    folds = build_lopcv_folds(sample_names, dataset)
    fold_1_train, _ = folds[FOLD_FOR_SELECTION]
    _, internal_val = split_internal(fold_1_train)
    print(f"Internal val samples: {internal_val}")

    val_loader = build_val_loader(
        config, internal_val, data_dir, dataset, FOLD_FOR_SELECTION
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lambda_dirs = sorted(out_dir.glob("lambda_*"))
    header = f"{'lambda':>8}  {'PCC(M)':>8}  {'PCC(H)':>8}  {'PCC-10':>8}  {'PCC-50':>8}  {'PCC-200':>8}"
    print("\n" + header)
    print("-" * len(header))

    rows = []
    for d in lambda_dirs:
        ckpt = d / "fold_99" / "best_model.pt"
        if not ckpt.exists():
            continue
        lam = float(d.name.replace("lambda_", ""))
        model = TRIPLEX(config, n_genes=config.get("n_genes", 250))
        sd = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(sd["model"])
        model = model.to(device)

        slide_yhat, slide_ytrue = run_inference(model, val_loader, device)
        metrics, _ = compute_topk_metrics(slide_yhat, slide_ytrue)
        row = {"lambda": lam, **metrics}
        rows.append(row)

        print(
            f"{lam:>8.3f}  "
            f"{metrics['pcc_m']:>8.4f}  "
            f"{metrics['pcc_h']:>8.4f}  "
            f"{metrics['pcc_10']:>8.4f}  "
            f"{metrics['pcc_50']:>8.4f}  "
            f"{metrics['pcc_200']:>8.4f}"
        )

        # Save augmented per-lambda result
        per = d / "result_topk.json"
        with open(per, "w") as f:
            json.dump(row, f, indent=2)

        # Cleanup
        del model, sd, slide_yhat, slide_ytrue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary JSON
    summary = out_dir / "topk_summary.json"
    with open(summary, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved summary: {summary}")


if __name__ == "__main__":
    main()
