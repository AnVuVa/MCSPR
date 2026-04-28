"""SPCS (Spatial and Pattern Combined Smoothing) port from R to Python.

Faithful port of MERGE/scripts/smooth.ipynb (which itself wraps the original
SPCS R package by Liu et al.).

Pipeline per slide (n_spots × n_genes raw UMI counts):
  1. selectGenes: drop genes with > zero_cutoff * n_spots zero entries.
  2. NormalizeLibrarySize: per-spot CPM (counts / libsize * scale).
  3. log2(x + 1); NaN → 0.
  4. Pattern step:
     a. randomized PCA on (centered) spot×gene matrix → spots × d (d=10).
     b. Pearson distance between spot PCA rows: D = 1 - cor.
     c. Contribution: r = exp(-4.5 · D²); zero out r where D>1; zero diag.
     d. For each spot, keep top-tau_p neighbors by r; normalize weights.
  5. Spatial step: Manhattan distance between (array_row, array_col); for each
     spot keep neighbors with 1 ≤ d ≤ (2·tau_s if is_hexa else tau_s); weight
     1/d, then normalize.
  6. Combined: smoothed = exp·(1-α) + (s_exp·β + p_exp·(1-β))·α.
  7. NaN → global mean.

Returns the smoothed expression matrix (n_spots × n_genes_kept) plus the
boolean keep mask so callers can re-index back to the input gene order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.utils.extmath import randomized_svd


@dataclass
class SPCSParams:
    tau_p: int = 16          # top-K pattern neighbors per spot
    tau_s: int = 2           # spatial radius (Manhattan); doubled for hex grids
    alpha: float = 0.6       # smooth-vs-original mix
    beta: float = 0.4        # spatial-vs-pattern mix inside the smooth term
    is_hexa: bool = True     # HER2ST / Visium hex grid
    pca_dim: int = 10
    pca_oversample: int = 10
    pca_n_iter: int = 7
    seed: int = 42
    libsize_scale: float = 10000.0
    zero_cutoff: float = 0.7  # selectGenes zero-fraction threshold


def _select_genes(counts: np.ndarray, zero_cutoff: float) -> np.ndarray:
    """Boolean mask over genes: True = keep. Matches R `selectGenes` with
    var_cutoff = 0 (only zero-rate filter)."""
    n_spots = counts.shape[0]
    n_zeros_per_gene = (counts == 0).sum(axis=0)
    return n_zeros_per_gene <= zero_cutoff * n_spots


def _logcpm(counts: np.ndarray, scale: float) -> np.ndarray:
    libsize = counts.sum(axis=1, keepdims=True).astype(np.float64)
    libsize[libsize == 0] = 1.0  # avoid /0 — R produces NaN then sets to 0
    cpm = counts.astype(np.float64) / libsize * scale
    out = np.log2(cpm + 1)
    return np.nan_to_num(out, nan=0.0)


def _pattern_neighbors(log_cpm: np.ndarray, tau_p: int, p: SPCSParams):
    """Returns list of (idx_array, weight_array) per spot."""
    X = log_cpm - log_cpm.mean(axis=0, keepdims=True)
    U, S, _ = randomized_svd(
        X, n_components=p.pca_dim, n_oversamples=p.pca_oversample,
        n_iter=p.pca_n_iter, random_state=p.seed,
    )
    pcs = U * S  # (n_spots, pca_dim)

    # Pearson distance via corrcoef (rows = observations).
    cor = np.corrcoef(pcs)
    dist = 1.0 - cor
    rmat = np.exp(-4.5 * dist ** 2)
    rmat[dist > 1] = 0.0
    np.fill_diagonal(rmat, 0.0)

    n = log_cpm.shape[0]
    nbrs = []
    for i in range(n):
        # R indexes mat.pcontribution[, i] — column i is the contribution from
        # other spots TO spot i. corrcoef is symmetric so col == row, fine.
        col = rmat[:, i]
        nz = np.where(col > 0)[0]
        if nz.size == 0:
            nbrs.append((np.empty(0, dtype=np.int64), np.empty(0)))
            continue
        if tau_p > 0 and nz.size > tau_p:
            top = nz[np.argsort(-col[nz])[:tau_p]]
        else:
            top = nz
        w = col[top]
        w = w / (w.sum() + 1e-8)
        nbrs.append((top.astype(np.int64), w))
    return nbrs


def _spatial_neighbors(array_row: np.ndarray, array_col: np.ndarray,
                       tau_s: int, is_hexa: bool):
    pos = np.stack([array_row, array_col], axis=1).astype(np.int64)
    eff = tau_s * 2 if is_hexa else tau_s
    n = pos.shape[0]
    nbrs = []
    for i in range(n):
        d = np.abs(pos - pos[i]).sum(axis=1)
        mask = (d > 0) & (d <= eff)
        idx = np.where(mask)[0]
        if idx.size == 0:
            nbrs.append((np.empty(0, dtype=np.int64), np.empty(0)))
            continue
        w = 1.0 / d[idx].astype(np.float64)
        w = w / (w.sum() + 1e-8)
        nbrs.append((idx.astype(np.int64), w))
    return nbrs


def _apply_neighbors(log_cpm: np.ndarray, nbrs):
    out = np.zeros_like(log_cpm)
    for i, (idx, w) in enumerate(nbrs):
        if idx.size > 0:
            out[i] = w @ log_cpm[idx]
    return out


def spcs_smooth(counts: np.ndarray, array_row: np.ndarray,
                array_col: np.ndarray, params: SPCSParams = SPCSParams()):
    """Apply SPCS to a single slide.

    Args:
        counts: (n_spots, n_genes) raw UMI counts.
        array_row, array_col: (n_spots,) integer spot grid coordinates.
        params: SPCSParams.

    Returns:
        smoothed: (n_spots, n_genes_kept) float64 — log2(CPM+1) smoothed.
        keep: (n_genes,) bool mask — True for genes that survived selectGenes.
    """
    keep = _select_genes(counts, params.zero_cutoff)
    counts_kept = counts[:, keep]
    log_cpm = _logcpm(counts_kept, params.libsize_scale)

    p_nbrs = _pattern_neighbors(log_cpm, params.tau_p, params)
    s_nbrs = _spatial_neighbors(array_row, array_col,
                                params.tau_s, params.is_hexa)

    s_exp = _apply_neighbors(log_cpm, s_nbrs)
    p_exp = _apply_neighbors(log_cpm, p_nbrs)

    a, b = params.alpha, params.beta
    smoothed = log_cpm * (1 - a) + (s_exp * b + p_exp * (1 - b)) * a

    if np.isnan(smoothed).any():
        m = smoothed[~np.isnan(smoothed)].mean()
        smoothed = np.where(np.isnan(smoothed), m, smoothed)
    return smoothed, keep
