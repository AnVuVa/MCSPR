"""Recompute the Phase 1 metric pool for every completed fold:
PCC(M), PCC(H), PCC-10, PCC-50, PCC-200, PCC-300, RVD, Q1_MSE.

Pool spots across the full LOPCV val set (patient held-out), then compute
per-gene statistics (PCC, variance, MSE). PCC-K follows the TRIPLEX paper
convention. RVD and Q1_MSE use the gene-pooled analogues of the per-slide
definitions in src/training/universal_trainer.py:
  - RVD    = mean_j ((var_j(Y_hat) - var_j(Y_true))^2 / (var_j(Y_true)^2 + eps))
  - Q1_MSE = 25th percentile of per-gene pooled MSE.

Runs on CPU so it does not steal GPU from the training orchestrator.

Usage:
    python scripts/report_phase1_topk.py --config configs/her2st.yaml
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

from src.data.loaders import build_loaders
from src.models.triplex import TRIPLEX
from src.models.stnet import STNet


TOP_K_LIST = [10, 50, 200, 300]
ARCHS = ["triplex", "triplex_mcspr", "stnet", "stnet_mcspr"]


def per_gene_pcc(Y_hat: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
    n = Y_hat.shape[1]
    out = np.zeros(n)
    for j in range(n):
        if np.std(Y_hat[:, j]) < 1e-10 or np.std(Y_true[:, j]) < 1e-10:
            out[j] = 0.0
        else:
            out[j], _ = pearsonr(Y_hat[:, j], Y_true[:, j])
    return out


def pooled_rvd_q1mse(Y_hat: np.ndarray, Y_true: np.ndarray):
    # Gene-pooled analogues of universal_trainer's per-slide definitions.
    var_pred = Y_hat.var(axis=0)
    var_true = Y_true.var(axis=0)
    rvd = float(np.mean((var_pred - var_true) ** 2 / (var_true ** 2 + 1e-8)))
    per_gene_mse = ((Y_hat - Y_true) ** 2).mean(axis=0)
    q1_mse = float(np.percentile(per_gene_mse, 25))
    return rvd, q1_mse


def run_inference(model, arch, val_loader, device):
    model.eval()
    Yh_list, Yt_list = [], []
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
            Yh_list.append(yh.cpu().numpy())
            Yt_list.append(slide["expression"].cpu().numpy())
    Yh = np.concatenate(Yh_list, axis=0)
    Yt = np.concatenate(Yt_list, axis=0)
    return Yh, Yt


def topk_metrics(gene_pcc):
    out = {"pcc_m": float(np.nanmean(gene_pcc))}
    order = np.argsort(gene_pcc)[::-1]
    for k in TOP_K_LIST:
        k_eff = min(k, len(gene_pcc))
        out[f"pcc_{k}"] = float(np.nanmean(gene_pcc[order[:k_eff]]))
    out["pcc_h"] = out["pcc_50"]
    return out


def build_model(arch, config):
    n_genes = config.get("n_genes", 250)
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
    parser.add_argument("--out", type=str,
                        default="results/phase1_topk_partial.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    dataset = config["dataset"]
    data_dir = config["data_dir"]

    device = torch.device("cpu")  # keep GPUs for training orchestrator
    tasks = find_completed_folds(dataset)
    print(f"Found {len(tasks)} completed folds. Evaluating on CPU...\n")

    header = (
        f"{'arch':<16} {'fold':>4} "
        f"{'PCC(M)':>8} {'PCC(H)':>8} "
        f"{'PCC-10':>8} {'PCC-50':>8} {'PCC-200':>8} {'PCC-300':>8} "
        f"{'RVD':>8} {'Q1_MSE':>8}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for arch, fold, ckpt_path in tasks:
        try:
            _, val_loader = build_loaders(
                data_dir=data_dir, dataset=dataset, fold_idx=fold,
                config=config,
            )
            model = build_model(arch, config)
            model = load_state(model, ckpt_path)
            model = model.to(device)
            Yh, Yt = run_inference(model, arch, val_loader, device)
            gene_pcc = per_gene_pcc(Yh, Yt)
            m = topk_metrics(gene_pcc)
            rvd, q1_mse = pooled_rvd_q1mse(Yh, Yt)
            m["rvd"] = rvd
            m["q1_mse"] = q1_mse
            m.update({"arch": arch, "fold": fold,
                      "n_spots": int(Yh.shape[0]),
                      "n_genes": int(Yh.shape[1])})
            results.append(m)
            print(f"{arch:<16} {fold:>4} "
                  f"{m['pcc_m']:>8.4f} {m['pcc_h']:>8.4f} "
                  f"{m['pcc_10']:>8.4f} {m['pcc_50']:>8.4f} "
                  f"{m['pcc_200']:>8.4f} {m['pcc_300']:>8.4f} "
                  f"{m['rvd']:>8.4f} {m['q1_mse']:>8.4f}")
            del model, Yh, Yt, gene_pcc, val_loader
        except Exception as e:
            print(f"{arch:<16} {fold:>4}   ERROR: {e}")

    # Group means + MCSPR vs base deltas
    print("\n=== group means (over completed folds) ===")
    for arch in ARCHS:
        rows = [r for r in results if r["arch"] == arch]
        if not rows:
            continue
        print(f"  {arch:<14}  n={len(rows):d}  "
              f"PCC(M)={np.mean([r['pcc_m'] for r in rows]):.4f}  "
              f"PCC(H)={np.mean([r['pcc_h'] for r in rows]):.4f}  "
              f"PCC-10={np.mean([r['pcc_10'] for r in rows]):.4f}  "
              f"PCC-200={np.mean([r['pcc_200'] for r in rows]):.4f}  "
              f"PCC-300={np.mean([r['pcc_300'] for r in rows]):.4f}  "
              f"RVD={np.mean([r['rvd'] for r in rows]):.4f}  "
              f"Q1_MSE={np.mean([r['q1_mse'] for r in rows]):.4f}")

    print("\n=== MCSPR vs baseline (paired folds) ===")
    by_key = {(r['arch'], r['fold']): r for r in results}
    for arch in ["stnet", "triplex"]:
        paired = [(by_key[(arch, f)], by_key[(f"{arch}_mcspr", f)])
                  for f in range(8)
                  if (arch, f) in by_key and (f"{arch}_mcspr", f) in by_key]
        if not paired:
            continue
        for m_base, m_mcspr in paired:
            f = m_base["fold"]
            dm = m_mcspr['pcc_m'] - m_base['pcc_m']
            dh = m_mcspr['pcc_h'] - m_base['pcc_h']
            d10 = m_mcspr['pcc_10'] - m_base['pcc_10']
            d200 = m_mcspr['pcc_200'] - m_base['pcc_200']
            drvd = m_mcspr['rvd'] - m_base['rvd']
            dq1 = m_mcspr['q1_mse'] - m_base['q1_mse']
            print(f"  {arch:<8} fold {f}:  "
                  f"ΔPCC(M)={dm:+.4f}  ΔPCC(H)={dh:+.4f}  "
                  f"ΔPCC-10={d10:+.4f}  ΔPCC-200={d200:+.4f}  "
                  f"ΔRVD={drvd:+.4f}  ΔQ1_MSE={dq1:+.4f}")
        # mean delta
        dms = [a['pcc_m'] - b['pcc_m'] for b, a in paired]
        dhs = [a['pcc_h'] - b['pcc_h'] for b, a in paired]
        drvds = [a['rvd'] - b['rvd'] for b, a in paired]
        dq1s = [a['q1_mse'] - b['q1_mse'] for b, a in paired]
        print(f"  {arch:<8}  mean Δ  "
              f"PCC(M)={np.mean(dms):+.4f} ({np.sum([d>0 for d in dms])}/{len(dms)} wins)  "
              f"PCC(H)={np.mean(dhs):+.4f}  "
              f"RVD={np.mean(drvds):+.4f} (MCSPR better if <0)  "
              f"Q1_MSE={np.mean(dq1s):+.4f} (MCSPR better if <0)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
