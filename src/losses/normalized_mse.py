import numpy as np
import torch
import torch.nn as nn


class NormalizedMSELoss(nn.Module):
    """
    Per-gene variance-normalized MSE loss (L_norm).

    Divides each gene's squared residual by its training-fold variance,
    making lambda a dimensionless ratio independent of gene expression scale.

    gene_var: np.ndarray of shape (m_genes,), computed from counts_svg
              log-normalized training targets (fold-specific, no test leakage).
              Spec v2 amendment A8: source is counts_svg (matches 300-gene
              model target), not counts_spcs.
    """

    def __init__(self, gene_var: np.ndarray):
        super().__init__()
        self.register_buffer(
            'gene_var',
            torch.tensor(gene_var, dtype=torch.float32)
        )

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_hat, y_true: (N_spots, m_genes), log-normalized space
        residuals = y_hat - y_true                          # (N, m)
        return ((residuals ** 2) / self.gene_var).mean()
