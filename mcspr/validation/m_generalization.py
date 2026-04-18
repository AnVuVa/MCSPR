import numpy as np
from typing import Dict, List, Optional, Tuple


def run_m_generalization_test(
    Y_hat_test: np.ndarray,
    Y_true_test: np.ndarray,
    M: np.ndarray,
    M_pinv: np.ndarray,
    gene_names: Optional[List[str]] = None,
    threshold_r: float = 0.10,
    verbose: bool = True,
) -> Tuple[bool, Dict]:
    """M-generalization test.

    Threshold is on NORMALIZED metric, not raw covariance:
    R_j = |Cov(delta_j, y_j)| / Var(y_j)
    A threshold of 0.10 means the off-manifold component accounts for >10% of
    the true gene's variance. Raw covariance thresholds are biologically
    meaningless because covariance is not scale-invariant.
    """
    n_spots, n_genes = Y_hat_test.shape
    n_modules = M.shape[1]

    # Step 1: On-manifold projection
    Z_hat = Y_hat_test @ M_pinv.T  # (n, B)
    Y_hat_on = Z_hat @ M.T  # (n, m)

    # Step 2: Off-manifold residual
    delta = Y_hat_test - Y_hat_on  # (n, m)

    # Step 3: Orthogonality sanity check
    ortho_check = float(np.abs(delta.T @ Y_hat_on).max())
    assert ortho_check < 1e-3, (
        f"Orthogonality check failed: max |delta.T @ Y_hat_on| = {ortho_check:.6f}"
    )

    # Step 4: Per-gene R metric — R_j = |Cov(delta_j, y_j)| / Var(y_j)
    R = np.zeros(n_genes)
    for j in range(n_genes):
        cov_dj_yj = np.cov(delta[:, j], Y_true_test[:, j])[0, 1]
        var_yj = np.var(Y_true_test[:, j])
        R[j] = np.abs(cov_dj_yj) / (var_yj + 1e-8)

    # Step 5: Stratify genes into quartiles Q1-Q4 by loading magnitude
    loading_magnitude = np.linalg.norm(M, axis=1)  # (m,) per gene
    quartile_bounds = np.percentile(loading_magnitude, [25, 50, 75])
    quartile_labels = np.digitize(
        loading_magnitude, quartile_bounds
    )  # 0=Q1, 1=Q2, 2=Q3, 3=Q4

    quartile_names = ["Q1_low", "Q2", "Q3", "Q4_high"]

    # Step 6: Per-quartile report
    quartiles: Dict = {}
    for q in range(4):
        mask = quartile_labels == q
        if mask.sum() == 0:
            continue
        q_name = quartile_names[q]
        R_q = R[mask]
        quartiles[q_name] = {
            "n_genes": int(mask.sum()),
            "median_R": float(np.median(R_q)),
            "90th_pct_R": float(np.percentile(R_q, 90)),
            "mean_loading": float(loading_magnitude[mask].mean()),
        }

    # Step 7: Decision — FAIL if Q4 90th_pct_R >= threshold_r
    q4_90th = quartiles.get("Q4_high", {}).get("90th_pct_R", 0.0)
    passed = q4_90th < threshold_r

    if not passed:
        decision = (
            f"FAIL: Q4 90th percentile R = {q4_90th:.4f} >= {threshold_r}. "
            "Off-manifold predictions correlate with true expression for "
            "high-loading genes. Consider increasing n_modules or using "
            "head-only MCSPR."
        )
    else:
        decision = (
            f"PASS: Q4 90th percentile R = {q4_90th:.4f} < {threshold_r}"
        )

    if verbose:
        print(decision)

    report = {
        "n_spots": n_spots,
        "n_genes": n_genes,
        "n_modules": n_modules,
        "orthogonality_check": ortho_check,
        "quartiles": quartiles,
        "decision": decision,
        "passed": passed,
    }

    return passed, report
