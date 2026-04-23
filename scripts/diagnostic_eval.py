"""L_norm vs L_MSE diagnostic evaluator.

Loads a fold-trained TRIPLEX checkpoint, runs val inference, computes:
  PCC(M)       — per-gene PCC per slide, averaged (matches evaluate.py)
  PCC-K        — for K in {10, 50, 200, 300}: top-K genes by mean per-gene
                 PCC across slides, then mean PCC over those K genes.
                 Note PCC-300 == PCC(M) by identity; both are reported.
  MSE          — per-slide averaged (raw, all genes)
  MSE-Qi       — for i in {1,2,3,4}: per-slide MSE restricted to gene
                 quartile Qi of training-set variance (Q1=lowest-var 75 genes,
                 Q4=highest-var 75), then averaged across slides.
                 Uses data_dir/nmf/fold_{fold}/gene_var.npy as reference.
  RVD          — pooled across slides:
                 RVD = mean_j [(var_pred_j - var_true_j)^2 / var_true_j^2]
                 with mask var_true > 1e-8 (matches eval_pooled_pcc.py)

Usage:
  python scripts/diagnostic_eval.py --config configs/her2st_diag.yaml \
      --fold 0 --fold_dir results/diag/triplex_lmse/fold_0 \
      --out results/diag/triplex_lmse/fold_0/diagnostic.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loaders import build_loaders
from src.models.triplex import TRIPLEX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--fold_dir", required=True,
                    help="Path containing best_model.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ckpt = Path(args.fold_dir) / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _, val_loader = build_loaders(
        config["data_dir"], config["dataset"], args.fold, config
    )

    model = TRIPLEX(config, n_genes=config.get("n_genes", 300))
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model = model.to(device).eval()

    slide_yhat, slide_ytrue = {}, {}
    with torch.no_grad():
        for batch in val_loader:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
            preds, _ = model(b)
            yh = preds["fusion"].cpu().numpy()
            yt = b["expression"].cpu().numpy()
            sid = b["sample_idx"].cpu().numpy()
            for i in range(yh.shape[0]):
                s = int(sid[i])
                slide_yhat.setdefault(s, []).append(yh[i])
                slide_ytrue.setdefault(s, []).append(yt[i])

    n_genes = config.get("n_genes", 300)

    # Training-set gene variance → variance quartile partition.
    gene_var_path = Path(config["data_dir"]) / "nmf" / f"fold_{args.fold}" / "gene_var.npy"
    gene_var = np.load(str(gene_var_path))
    if gene_var.shape[0] != n_genes:
        raise ValueError(f"gene_var length {gene_var.shape[0]} != n_genes {n_genes}")
    q_order = np.argsort(gene_var)
    quartile_idx = {}
    q_size = n_genes // 4
    for i in range(4):
        lo = i * q_size
        hi = (i + 1) * q_size if i < 3 else n_genes
        quartile_idx[f"Q{i+1}"] = q_order[lo:hi]

    # Per-slide per-gene PCC (matrix: S × G), per-slide MSE (all + quartile).
    slide_pcc_m = []
    slide_mse_all = []
    slide_mse_q = {f"Q{i+1}": [] for i in range(4)}
    per_slide_gene_pccs = []  # shape (S, n_genes), NaN where std too small
    all_yh_parts, all_yt_parts = [], []

    slide_ids_sorted = sorted(slide_yhat.keys())
    for s in slide_ids_sorted:
        yh = np.stack(slide_yhat[s])
        yt = np.stack(slide_ytrue[s])
        all_yh_parts.append(yh); all_yt_parts.append(yt)

        gene_pccs = np.full(n_genes, np.nan, dtype=np.float64)
        for j in range(n_genes):
            if yt[:, j].std() < 1e-10 or yh[:, j].std() < 1e-10:
                continue
            r, _ = pearsonr(yh[:, j], yt[:, j])
            if not np.isnan(r):
                gene_pccs[j] = r
        per_slide_gene_pccs.append(gene_pccs)

        valid = gene_pccs[~np.isnan(gene_pccs)]
        slide_pcc_m.append(float(valid.mean()) if valid.size else 0.0)
        slide_mse_all.append(float(np.mean((yh - yt) ** 2)))
        for qk, idx in quartile_idx.items():
            slide_mse_q[qk].append(float(np.mean((yh[:, idx] - yt[:, idx]) ** 2)))

    # PCC-K: mean per-gene PCC across slides (ignore NaN), rank, top-K mean.
    per_slide_gene_pccs_np = np.stack(per_slide_gene_pccs)  # (S, G)
    mean_gene_pcc = np.nanmean(per_slide_gene_pccs_np, axis=0)  # (G,)
    pcc_k = {}
    for K in (10, 50, 200, 300):
        K_eff = min(K, n_genes)
        top_idx = np.argsort(mean_gene_pcc)[::-1][:K_eff]
        pcc_k[f"pcc_top{K}"] = float(np.nanmean(mean_gene_pcc[top_idx]))

    # RVD (pooled, unchanged).
    all_yh = np.vstack(all_yh_parts)
    all_yt = np.vstack(all_yt_parts)
    var_pred = all_yh.var(axis=0)
    var_true = all_yt.var(axis=0)
    mask = var_true > 1e-8
    rvd = float(((var_pred[mask] - var_true[mask]) ** 2 / var_true[mask] ** 2).mean())

    mse_q_means = {qk: float(np.mean(vals)) for qk, vals in slide_mse_q.items()}

    result = {
        "fold": args.fold,
        "fold_dir": args.fold_dir,
        "n_slides": len(slide_yhat),
        "n_genes": n_genes,
        "pcc_m_per_slide_mean": float(np.mean(slide_pcc_m)),
        "pcc_m_per_slide_std":  float(np.std(slide_pcc_m)),
        **pcc_k,
        "mse_per_slide_mean":   float(np.mean(slide_mse_all)),
        "mse_q_per_slide_mean": mse_q_means,
        "rvd_pooled":           rvd,
        "slide_ids":            slide_ids_sorted,
        "slide_pcc_m":          slide_pcc_m,
        "slide_mse":            slide_mse_all,
        "slide_mse_by_quartile": slide_mse_q,
        "gene_var_quartile_ranges": {
            qk: [float(gene_var[idx].min()), float(gene_var[idx].max())]
            for qk, idx in quartile_idx.items()
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{args.fold_dir}")
    print(f"  PCC(M)   per-slide: {result['pcc_m_per_slide_mean']:.4f} ± {result['pcc_m_per_slide_std']:.4f}")
    print(f"  PCC-10 / 50 / 200 / 300: "
          f"{pcc_k['pcc_top10']:.4f} / {pcc_k['pcc_top50']:.4f} / "
          f"{pcc_k['pcc_top200']:.4f} / {pcc_k['pcc_top300']:.4f}")
    print(f"  MSE      per-slide: {result['mse_per_slide_mean']:.4f}")
    print(f"  MSE Q1/Q2/Q3/Q4  : "
          f"{mse_q_means['Q1']:.4f} / {mse_q_means['Q2']:.4f} / "
          f"{mse_q_means['Q3']:.4f} / {mse_q_means['Q4']:.4f}")
    print(f"  RVD      pooled   : {result['rvd_pooled']:.4f}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
