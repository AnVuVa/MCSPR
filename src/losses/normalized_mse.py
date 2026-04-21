import numpy as np
import torch
import torch.nn as nn


class NormalizedMSELoss(nn.Module):
    """
    Per-gene variance-normalized MSE loss (L_norm).

    Divides each gene's squared residual by its training-fold variance,
    making lambda a dimensionless ratio independent of gene expression scale.

    gene_var: np.ndarray of shape (m_genes,), computed from log-normalized
              SPCS-smoothed training targets (fold-specific, no test leakage).
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
