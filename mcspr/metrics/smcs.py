import numpy as np
from typing import Dict, Optional


def _soft_pearson_matrix(
    Z: np.ndarray, w: np.ndarray, tau: float = 1e-6
) -> np.ndarray:
    """Soft-weighted Pearson correlation matrix.

    CRITICAL: SMCS targets test-set ground truth ONLY.
    C_prior (training prior) is NEVER used as the SMCS target.
    """
    w_norm = w / (w.sum() + tau)
    mu = (w_norm[:, None] * Z).sum(axis=0)
    Z_c = Z - mu
    Sigma = (Z_c * w_norm[:, None]).T @ Z_c
    std = np.sqrt(np.diag(Sigma) + tau)
    C = Sigma / (std[:, None] * std[None, :])
    np.fill_diagonal(C, 1.0)
    return C


def compute_smcs(
    Y_pred: np.ndarray,
    Y_true: np.ndarray,
    M_pinv: np.ndarray,
    context_labels: np.ndarray,
    n_contexts: int,
    tau: float = 1e-6,
    weighted_contexts: Optional[np.ndarray] = None,
) -> Dict:
    Z_pred = Y_pred @ M_pinv.T
    Z_true = Y_true @ M_pinv.T
    B = Z_pred.shape[1]

    results: Dict = {}
    total_score = 0.0
    total_weight = 0.0

    for t in range(n_contexts):
        if weighted_contexts is not None:
            w_t = weighted_contexts[:, t]
        else:
            w_t = (context_labels == t).astype(float)

        eff_n = w_t.sum()
        if eff_n < 10:
            results[f"ctx_{t}_skipped"] = True
            continue

        C_pred_t = _soft_pearson_matrix(Z_pred, w_t, tau)
        C_true_t = _soft_pearson_matrix(Z_true, w_t, tau)

        frob_dist = np.linalg.norm(C_pred_t - C_true_t, "fro")
        max_dist = np.sqrt(2.0 * B * B)
        smcs_t = 1.0 - frob_dist / max_dist

        results[f"ctx_{t}_smcs"] = float(smcs_t)
        results[f"ctx_{t}_eff_n"] = float(eff_n)
        results[f"ctx_{t}_frob_dist"] = float(frob_dist)

        total_score += smcs_t * eff_n
        total_weight += eff_n

    results["smcs_overall"] = (
        float(total_score / total_weight) if total_weight > 0 else 0.0
    )
    return results


def smcs_sensitivity_analysis(
    Y_pred: np.ndarray,
    Y_true: np.ndarray,
    module_definitions: Dict[str, np.ndarray],
    context_labels: np.ndarray,
    n_contexts: int,
) -> Dict:
    results: Dict = {}
    scores: Dict = {}

    for name, M_pinv in module_definitions.items():
        smcs_result = compute_smcs(
            Y_pred, Y_true, M_pinv, context_labels, n_contexts
        )
        scores[name] = smcs_result["smcs_overall"]
        results[name] = smcs_result

    all_scores = list(scores.values())
    score_range = max(all_scores) - min(all_scores) if all_scores else 0.0

    results["score_range"] = score_range
    results["scores_by_definition"] = scores
    if score_range > 0.1:
        results["stability_warning"] = (
            "SMCS is unstable across module definitions (range > 0.1)"
        )

    return results
