"""L_norm vs L_MSE diagnostic evaluator.

Loads a fold-trained TRIPLEX checkpoint, runs val inference, computes:
  PCC(M)  — per-gene PCC per slide, averaged (matches evaluate.py)
  MSE     — per-slide averaged
  RVD     — pooled across slides:
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

    slide_pcc_m, slide_mse = [], []
    all_yh_parts, all_yt_parts = [], []
    for s in sorted(slide_yhat):
        yh = np.stack(slide_yhat[s])
        yt = np.stack(slide_ytrue[s])
        all_yh_parts.append(yh); all_yt_parts.append(yt)
        pccs = []
        for j in range(yh.shape[1]):
            if yt[:, j].std() < 1e-10 or yh[:, j].std() < 1e-10:
                continue
            r, _ = pearsonr(yh[:, j], yt[:, j])
            if not np.isnan(r):
                pccs.append(r)
        slide_pcc_m.append(float(np.mean(pccs)) if pccs else 0.0)
        slide_mse.append(float(np.mean((yh - yt) ** 2)))

    all_yh = np.vstack(all_yh_parts)
    all_yt = np.vstack(all_yt_parts)
    var_pred = all_yh.var(axis=0)
    var_true = all_yt.var(axis=0)
    mask = var_true > 1e-8
    rvd = float(((var_pred[mask] - var_true[mask]) ** 2 / var_true[mask] ** 2).mean())

    result = {
        "fold": args.fold,
        "fold_dir": args.fold_dir,
        "n_slides": len(slide_yhat),
        "pcc_m_per_slide_mean": float(np.mean(slide_pcc_m)),
        "pcc_m_per_slide_std":  float(np.std(slide_pcc_m)),
        "mse_per_slide_mean":   float(np.mean(slide_mse)),
        "rvd_pooled":           rvd,
        "slide_ids":            sorted(slide_yhat.keys()),
        "slide_pcc_m":          slide_pcc_m,
        "slide_mse":            slide_mse,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{args.fold_dir}")
    print(f"  PCC(M) per-slide: {result['pcc_m_per_slide_mean']:.4f} ± {result['pcc_m_per_slide_std']:.4f}")
    print(f"  MSE    per-slide: {result['mse_per_slide_mean']:.4f}")
    print(f"  RVD    pooled   : {result['rvd_pooled']:.4f}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
