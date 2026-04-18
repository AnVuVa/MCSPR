import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple

from scipy.stats import spearmanr


def fit_nmf(
    Y_train: np.ndarray,
    n_components: int = 15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    from sklearn.decomposition import NMF

    model = NMF(
        n_components=n_components,
        init="nndsvd",
        max_iter=500,
        random_state=random_state,
    )
    Z = model.fit_transform(Y_train)  # (n_train, B)
    M = model.components_.T  # (m, B)
    M_pinv = np.linalg.pinv(M)  # (B, m)

    Y_reconstructed = Z @ M.T
    ss_res = np.sum((Y_train - Y_reconstructed) ** 2)
    ss_tot = np.sum((Y_train - Y_train.mean(axis=0)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    if r2 < 0.60:
        warnings.warn(
            f"NMF reconstruction R² = {r2:.4f} < 0.60. "
            "Consider increasing n_components."
        )

    return M, M_pinv, r2


def compute_context_priors(
    Y_train: np.ndarray,
    M_pinv: np.ndarray,
    context_labels: np.ndarray,
    context_weights: np.ndarray,
    n_contexts: int,
) -> Tuple[np.ndarray, Dict]:
    Z_train = Y_train @ M_pinv.T  # (n_train, B)
    B = Z_train.shape[1]
    C_prior = np.zeros((n_contexts, B, B))
    stats: Dict = {}

    for t in range(n_contexts):
        w_t = context_weights[:, t]
        eff_n = w_t.sum()

        if eff_n < 10:
            C_prior[t] = np.eye(B)
            stats[f"ctx_{t}_warning"] = (
                f"eff_n={eff_n:.1f} < 10, using identity prior"
            )
            continue

        mu_t = (w_t[:, None] * Z_train).sum(axis=0) / eff_n
        Z_c = Z_train - mu_t
        Sigma_t = (Z_c * (w_t / eff_n)[:, None]).T @ Z_c

        std_t = np.sqrt(np.diag(Sigma_t) + 1e-8)
        C_prior[t] = Sigma_t / (std_t[:, None] * std_t[None, :])
        C_prior[t] = np.clip(C_prior[t], -1.0, 1.0)
        np.fill_diagonal(C_prior[t], 1.0)

        stats[f"ctx_{t}_eff_n"] = float(eff_n)
        stats[f"ctx_{t}_mean_abs_corr"] = float(np.abs(C_prior[t]).mean())

    return C_prior, stats


def validate_prior(
    C_prior: np.ndarray,
    M: np.ndarray,
    gene_names: Optional[List[str]],
    context_labels_per_slide: Dict[str, np.ndarray],
    spot_coords_per_slide: Dict[str, np.ndarray],
    min_spatial_contiguity: float = 0.6,
    min_cross_patient_rho: float = 0.7,
    verbose: bool = True,
) -> Tuple[List[int], Dict]:
    T, B, _ = C_prior.shape
    slide_ids = list(context_labels_per_slide.keys())
    n_slides = len(slide_ids)
    report: Dict = {}
    valid_contexts: List[int] = []

    upper_row, upper_col = np.triu_indices(B, k=1)

    for t in range(T):
        ctx_report: Dict = {}

        # --- Step 2: Gene enrichment (flag for manual MSigDB review) ---
        C_t = C_prior[t].copy()
        np.fill_diagonal(C_t, 0.0)
        off_diag = C_t[upper_row, upper_col]
        sorted_idx = np.argsort(np.abs(off_diag))[::-1][:10]

        top_pairs = []
        for idx in sorted_idx:
            a, b = int(upper_row[idx]), int(upper_col[idx])
            gene_a = (
                gene_names[int(np.argmax(M[:, a]))]
                if gene_names is not None
                else f"gene_{int(np.argmax(M[:, a]))}"
            )
            gene_b = (
                gene_names[int(np.argmax(M[:, b]))]
                if gene_names is not None
                else f"gene_{int(np.argmax(M[:, b]))}"
            )
            top_pairs.append(
                {
                    "programs": (a, b),
                    "correlation": float(C_prior[t, a, b]),
                    "top_gene_a": gene_a,
                    "top_gene_b": gene_b,
                }
            )
        ctx_report["top_gene_pairs"] = top_pairs
        ctx_report["msigdb_review_required"] = True

        # --- Step 3: Spatial coherence ---
        contiguity_per_slide = []
        for slide_id in slide_ids:
            labels = context_labels_per_slide[slide_id]
            coords = spot_coords_per_slide[slide_id]
            mask_t = labels == t
            n_t = int(mask_t.sum())
            if n_t == 0:
                continue

            coords_t = coords[mask_t]
            if n_t == 1:
                contiguity_per_slide.append(0.0)
                continue

            # Pairwise distances among context-t spots
            diffs = coords_t[None, :, :] - coords_t[:, None, :]  # (n_t, n_t, d)
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))  # (n_t, n_t)
            np.fill_diagonal(dists, np.inf)
            has_neighbor = (dists <= 2.0).any(axis=1)
            contiguity_per_slide.append(float(has_neighbor.mean()))

        mean_contiguity = (
            float(np.mean(contiguity_per_slide))
            if contiguity_per_slide
            else 0.0
        )
        spatial_pass = mean_contiguity >= min_spatial_contiguity
        ctx_report["spatial_contiguity"] = mean_contiguity
        ctx_report["spatial_pass"] = spatial_pass

        # --- Step 4: Cross-patient stability ---
        if n_slides < 2:
            stability_pass = False
            ctx_report["stability_warning"] = "Fewer than 2 slides available"
            ctx_report["stability_pass"] = False
        else:
            half1_ids = slide_ids[: n_slides // 2]
            half2_ids = slide_ids[n_slides // 2 :]

            C_t_offdiag = C_prior[t][upper_row, upper_col]
            n_top = min(50, len(C_t_offdiag))
            top50_idx = np.argsort(np.abs(C_t_offdiag))[::-1][:n_top]

            # Per-slide context prevalence vectors for each half
            def _ctx_prevalence(ids: List[str]) -> np.ndarray:
                return np.array(
                    [
                        (context_labels_per_slide[sid] == t).mean()
                        for sid in ids
                    ]
                )

            prev_h1 = _ctx_prevalence(half1_ids)
            prev_h2 = _ctx_prevalence(half2_ids)

            min_len = min(len(prev_h1), len(prev_h2))
            if min_len >= 3:
                rho, _ = spearmanr(
                    np.sort(prev_h1)[::-1][:min_len],
                    np.sort(prev_h2)[::-1][:min_len],
                )
                if np.isnan(rho):
                    rho = 0.0
            else:
                rho = 0.0

            stability_pass = rho >= min_cross_patient_rho
            ctx_report["cross_patient_rho"] = float(rho)
            ctx_report["stability_pass"] = stability_pass
            ctx_report["top50_offdiag_entries"] = C_t_offdiag[top50_idx].tolist()

        # --- Decision: context passes if Step 3 AND Step 4 both pass ---
        passes = spatial_pass and stability_pass
        ctx_report["valid"] = passes
        report[f"ctx_{t}"] = ctx_report

        if passes:
            valid_contexts.append(t)

        if verbose:
            status = "PASS" if passes else "FAIL"
            print(
                f"Context {t}: {status} "
                f"(spatial={spatial_pass}, stability={stability_pass})"
            )

    report["fallback_triggered"] = len(valid_contexts) < T - 1

    return valid_contexts, report
