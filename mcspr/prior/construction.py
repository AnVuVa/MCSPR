import json
from pathlib import Path
import numpy as np
import scipy.linalg
import warnings
from typing import Dict, List, Optional, Tuple

from scipy.stats import spearmanr


NMF_R2_THRESHOLD = 0.35  # Spec v2 admin directive 2026-04-22: SVG-panel
                          # calibrated threshold (2000-gene panel, n_modules=10)
NMF_MAX_ITER = 500
NMF_INIT = "nndsvda"
SEED = 2021


def fit_nmf(
    Y_train: np.ndarray,
    n_components: int = 10,
    random_state: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, float]:
    from sklearn.decomposition import NMF

    model = NMF(
        n_components=n_components,
        init=NMF_INIT,
        max_iter=NMF_MAX_ITER,
        random_state=random_state,
    )
    Z = model.fit_transform(Y_train)  # (n_train, B)
    M = model.components_.T  # (m, B)
    M_pinv = np.linalg.pinv(M)  # (B, m)

    Y_reconstructed = Z @ M.T
    ss_res = np.sum((Y_train - Y_reconstructed) ** 2)
    ss_tot = np.sum((Y_train - Y_train.mean(axis=0)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    if r2 < NMF_R2_THRESHOLD:
        raise ValueError(
            f"NMF R²={r2:.4f} < {NMF_R2_THRESHOLD}. "
            f"Increase n_modules. Got B={n_components}."
        )

    return M, M_pinv, r2


def compute_context_priors(
    Y_train: np.ndarray,
    M_pinv: np.ndarray,
    context_weights: np.ndarray,
    n_contexts: int,
) -> np.ndarray:
    """Empirical context-conditional correlation prior.

    Spec v2: signature accepts (Y_train, M_pinv, context_weights, n_contexts)
    — context_labels dropped (was unused). Returns C_prior only (no stats).

    Y_train can be ANY gene space (300 SVG or 2000 panel) as long as M_pinv
    projects it to the same latent space used at inference-time.
    """
    Z_train = Y_train @ M_pinv.T  # (n_train, B)
    B = Z_train.shape[1]
    C_prior = np.zeros((n_contexts, B, B))

    for t in range(n_contexts):
        w_t = context_weights[:, t]
        eff_n = w_t.sum()

        if eff_n < 10:
            C_prior[t] = np.eye(B)
            warnings.warn(
                f"context {t}: eff_n={eff_n:.1f} < 10, using identity prior"
            )
            continue

        mu_t = (w_t[:, None] * Z_train).sum(axis=0) / eff_n
        Z_c = Z_train - mu_t
        Sigma_t = (Z_c * (w_t / eff_n)[:, None]).T @ Z_c

        std_t = np.sqrt(np.diag(Sigma_t) + 1e-8)
        C_prior[t] = Sigma_t / (std_t[:, None] * std_t[None, :])
        C_prior[t] = np.clip(C_prior[t], -1.0, 1.0)
        np.fill_diagonal(C_prior[t], 1.0)

    return C_prior


def build_nmf_panel(
    svg_gene_names,
    umi_gene_names,
    raw_umi_train,
    n_hvg=1700,
    seed=2021,
):
    """Construct guaranteed 2000-gene NMF panel.

    SVGs are structurally first 300 entries — no intersection check needed.
    Top 1700 non-overlapping HVGs fill remainder via seurat_v3.
    """
    import scanpy as sc
    import anndata

    svg_set = set(svg_gene_names)

    adata = anndata.AnnData(X=raw_umi_train.astype(np.float32))
    adata.var_names = list(umi_gene_names)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=min(
            2000 + len(svg_set), len(umi_gene_names)
        ),
    )

    hvg_ranked = (
        adata.var[adata.var.highly_variable]
        .sort_values("highly_variable_rank")
        .index.tolist()
    )
    hvg_nonsvg = [g for g in hvg_ranked if g not in svg_set][:n_hvg]

    # SVGs first (indices 0-299), HVGs after (indices 300-1999)
    panel_genes = list(svg_gene_names) + hvg_nonsvg
    assert len(panel_genes) == len(svg_gene_names) + len(hvg_nonsvg), (
        f"Panel size mismatch: {len(panel_genes)}"
    )

    gene_to_idx = {g: i for i, g in enumerate(umi_gene_names)}
    panel_indices = np.array(
        [gene_to_idx[g] for g in panel_genes], dtype=np.int64
    )
    svg_in_panel = np.arange(len(svg_gene_names), dtype=np.int64)

    return panel_indices, svg_in_panel, np.array(panel_genes)


def compute_svg_projection_matrix(
    M_full,
    svg_in_panel,
    lambda_ridge: float = 1e-3,
    kappa_max: float = 100.0,
):
    """Tikhonov pseudo-inverse of M_full's SVG subset rows.

    Prevents pinv-on-near-zero-column NaN blow-ups by adding λI to the
    (B, B) Gram matrix. Raises ValueError if condition number of the raw
    SVG submatrix exceeds kappa_max (indicates SVGs do not span the
    NMF basis stably).
    """
    M_svg = M_full[svg_in_panel, :]  # (300, B)
    kappa = float(np.linalg.cond(M_svg))

    if kappa > kappa_max:
        raise ValueError(
            f"kappa(M_svg)={kappa:.1f} > {kappa_max}. "
            f"SVG genes do not stably span the {M_svg.shape[1]} NMF "
            f"modules. Increase n_hvg or adjust SVG filter."
        )

    B = M_svg.shape[1]
    A = M_svg.T @ M_svg + lambda_ridge * np.eye(B)
    M_pinv_svg = scipy.linalg.solve(A, M_svg.T, assume_a="pos")  # (B, 300)
    return M_pinv_svg.astype(np.float32)


def load_gene_names(data_dir):
    """Load full UMI gene name list and SVG gene list.

    Paths per Task 1 verification:
      features_full/gene_names.json  — single global JSON list of 11870
      features_svg/{sample}.csv      — per-sample CSV, identical content
    """
    data_dir = Path(data_dir)
    with open(data_dir / "features_full" / "gene_names.json") as f:
        umi_gene_names = json.load(f)

    svg_path = next((data_dir / "features_svg").glob("*.csv"))
    with open(svg_path) as f:
        svg_gene_names = [ln.strip() for ln in f if ln.strip()]

    assert len(svg_gene_names) == 300, (
        f"Expected 300 SVG genes, got {len(svg_gene_names)}"
    )
    assert len(umi_gene_names) == 11870, (
        f"Expected 11870 genes, got {len(umi_gene_names)}"
    )
    return umi_gene_names, svg_gene_names


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
