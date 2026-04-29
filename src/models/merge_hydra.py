"""HydraMERGE: MERGE GAT backbone with K parallel output heads.

Surgical change vs the MERGE baseline (src/models/merge.py):
  * Stage 1 CNN (CNN_Predictor) is reused unchanged — patch embeddings are
    shared across all heads, so the precompute path is byte-identical.
  * Stage 2 GNN keeps the first three GATConv layers identical
    (256 -> 448*8 -> 384*8 -> 256*8) and replaces the single final
    GATConv(256*8 -> num_genes) with K parallel single-head GATConvs, each
    producing m_k genes. The K outputs are reassembled to (n_spots, num_genes)
    in canonical column order via idx_list (per-fold registry).

Forward returns a dict so the runner can apply the hydra-weighted MSE on the
per-head outputs (mathematically identical to F.mse_loss on the reassembled
full 300, see src/training/hydra_helpers.per_head_loss).
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.utils import dropout_edge


class HydraGATNet(nn.Module):
    """4-layer GAT — first three layers identical to baseline GATNet; final
    GATConv replaced by K parallel single-head GATConvs (one per module).

    Args:
        module_sizes: list of K integers; m_k = number of genes head k owns.
                      sum(module_sizes) must equal num_genes.
        idx_list:     idx_list[k] = canonical column indices the k-th head
                      writes into when reassembling the (n_spots, num_genes)
                      output.
        num_heads:    GAT heads for the shared trunk (matches baseline = 8).
        drop_edge:    edge dropout probability (matches baseline = 0.2).
    """

    def __init__(
        self,
        module_sizes: List[int],
        idx_list: List[List[int]],
        num_heads: int = 8,
        drop_edge: float = 0.2,
    ):
        super().__init__()
        assert len(module_sizes) == len(idx_list)
        self.module_sizes = list(module_sizes)
        self.idx_list = [list(idx) for idx in idx_list]
        self.K = len(module_sizes)
        self.num_genes = sum(self.module_sizes)
        self.drop_edge = drop_edge

        dim1, dim2, dim3 = 448, 384, 256
        self.nn1 = GATConv(256, dim1, num_heads)
        self.layer_norm1 = LayerNorm(dim1 * num_heads)
        self.nn2 = GATConv(dim1 * num_heads, dim2, num_heads)
        self.layer_norm2 = LayerNorm(dim2 * num_heads)
        self.nn3 = GATConv(dim2 * num_heads, dim3, num_heads)
        self.layer_norm3 = LayerNorm(dim3 * num_heads)

        # K parallel heads — each is a single-head GATConv producing m_k genes,
        # mirroring the baseline's final GATConv(dim3*num_heads -> num_genes).
        self.heads = nn.ModuleList([
            GATConv(dim3 * num_heads, m_k) for m_k in self.module_sizes
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict:
        edge_index, _ = dropout_edge(
            edge_index, p=self.drop_edge, training=self.training,
        )
        x = F.relu(self.nn1(x, edge_index))
        x = self.layer_norm1(x)
        x = F.relu(self.nn2(x, edge_index))
        x = self.layer_norm2(x)
        x = F.relu(self.nn3(x, edge_index))
        x = self.layer_norm3(x)

        per_head = [head(x, edge_index) for head in self.heads]
        full = torch.empty(
            per_head[0].shape[0], self.num_genes,
            dtype=per_head[0].dtype, device=per_head[0].device,
        )
        for k, idx in enumerate(self.idx_list):
            full[:, idx] = per_head[k]
        return {"fusion": full, "fusion_per_head": per_head}
