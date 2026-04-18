"""Evaluation metrics: PCC(M), PCC(H), MSE, MAE, SMCS.

All metrics use per-slide computation then averaged over slides.
This matches TRIPLEX paper methodology.
"""

import numpy as np
import torch
from scipy.stats import pearsonr
from typing import Dict, List, Optional


def _per_gene_pcc(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Compute PCC for each gene. Returns (n_genes,) array."""
    n_genes = y_hat.shape[1]
    pccs = np.zeros(n_genes)
    for j in range(n_genes):
        if np.std(y_hat[:, j]) < 1e-10 or np.std(y_true[:, j]) < 1e-10:
            pccs[j] = 0.0
        else:
            pccs[j], _ = pearsonr(y_hat[:, j], y_true[:, j])
    return pccs


def evaluate_fold(
    model,
    val_loader,
    config: dict,
    gene_names: List[str],
    device: str = "cuda",
    mcspr_artifacts: Optional[Dict] = None,
) -> Dict:
    """Evaluate one fold on validation set.

    Metrics:
      PCC(M): mean PCC over all 250 genes, per slide, then averaged.
      PCC(H): mean PCC over top-50 highly predictive genes.
      MSE: mean over all spots x genes, per slide -> average.
      MAE: same structure as MSE.
      SMCS: if mcspr_artifacts provided, compute from test ground truth.
    """
    model.eval()

    # Collect predictions per slide
    slide_preds: Dict[int, List[np.ndarray]] = {}
    slide_trues: Dict[int, List[np.ndarray]] = {}
    slide_ctx_labels: Dict[int, List[np.ndarray]] = {}

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            b = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            preds, _ = model(b)
            y_hat = preds["fusion"].cpu().numpy()
            y_true = b["expression"].cpu().numpy()
            sample_indices = b["sample_idx"].cpu().numpy()

            for i in range(y_hat.shape[0]):
                sid = int(sample_indices[i])
                slide_preds.setdefault(sid, []).append(y_hat[i])
                slide_trues.setdefault(sid, []).append(y_true[i])

                if "context_weights" in b:
                    ctx_w = b["context_weights"][i].cpu().numpy()
                    slide_ctx_labels.setdefault(sid, []).append(
                        int(np.argmax(ctx_w))
                    )

    # Per-slide metrics
    slide_pcc_m = []
    slide_pcc_h_genes = []  # per-gene PCCs for PCC(H) ranking
    slide_mse = []
    slide_mae = []
    per_slide_results = {}
    all_gene_pccs = []

    slide_ids = sorted(slide_preds.keys())
    for sid in slide_ids:
        y_hat_s = np.stack(slide_preds[sid])  # (N_s, 250)
        y_true_s = np.stack(slide_trues[sid])

        # PCC(M): mean PCC over all genes for this slide
        gene_pccs = _per_gene_pcc(y_hat_s, y_true_s)
        pcc_m_slide = float(np.nanmean(gene_pccs))
        slide_pcc_m.append(pcc_m_slide)
        all_gene_pccs.append(gene_pccs)

        # MSE per slide
        mse_slide = float(np.mean((y_hat_s - y_true_s) ** 2))
        slide_mse.append(mse_slide)

        # MAE per slide
        mae_slide = float(np.mean(np.abs(y_hat_s - y_true_s)))
        slide_mae.append(mae_slide)

        per_slide_results[f"slide_{sid}"] = {
            "pcc_m": pcc_m_slide,
            "mse": mse_slide,
            "mae": mae_slide,
            "n_spots": y_hat_s.shape[0],
        }

    # Aggregate
    pcc_m = float(np.mean(slide_pcc_m))
    pcc_m_std = float(np.std(slide_pcc_m))

    # PCC(H): top-50 genes by mean PCC rank across slides
    mean_gene_pccs = np.nanmean(np.stack(all_gene_pccs), axis=0)
    top50_idx = np.argsort(mean_gene_pccs)[::-1][:50]
    top50_pccs = mean_gene_pccs[top50_idx]
    pcc_h = float(np.nanmean(top50_pccs))
    pcc_h_std = float(np.nanstd(top50_pccs))

    mse = float(np.mean(slide_mse))
    mae = float(np.mean(slide_mae))

    result = {
        "pcc_m": pcc_m,
        "pcc_m_std": pcc_m_std,
        "pcc_h": pcc_h,
        "pcc_h_std": pcc_h_std,
        "mse": mse,
        "mae": mae,
        "smcs_overall": None,
        "per_slide": per_slide_results,
        "top50_gene_indices": top50_idx.tolist(),
    }

    # SMCS (if mcspr_artifacts provided)
    if mcspr_artifacts is not None:
        from mcspr.metrics import compute_smcs

        # Pool all predictions and truths
        all_preds = np.concatenate(
            [np.stack(slide_preds[s]) for s in slide_ids], axis=0
        )
        all_trues = np.concatenate(
            [np.stack(slide_trues[s]) for s in slide_ids], axis=0
        )
        all_ctx = np.concatenate(
            [np.array(slide_ctx_labels.get(s, [0] * len(slide_preds[s])))
             for s in slide_ids],
            axis=0,
        )

        smcs_scores = compute_smcs(
            Y_pred=all_preds,
            Y_true=all_trues,
            M_pinv=mcspr_artifacts["M_pinv"],
            context_labels=all_ctx,
            n_contexts=mcspr_artifacts["n_contexts"],
        )
        result["smcs_overall"] = smcs_scores["smcs_overall"]
        result["smcs_per_context"] = {
            k: v for k, v in smcs_scores.items() if k.startswith("ctx_")
        }

    model.train()
    return result
