import torch
import torch.nn as nn
import math


class RelativePositionBias(nn.Module):
    """Relative position encoding for 5x5 neighbor grid.

    Encodes (dx, dy) offsets between all 25 patches in the neighbor view.
    Offsets range from -4 to +4 in each axis.
    """

    def __init__(self, num_heads, grid_size=5):
        super().__init__()
        self.num_heads = num_heads
        self.grid_size = grid_size
        max_offset = 2 * grid_size - 1  # 9
        self.bias_table = nn.Parameter(
            torch.zeros(max_offset * max_offset, num_heads)
        )
        nn.init.trunc_normal_(self.bias_table, std=0.02)

        # Precompute relative position index
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(grid_size), torch.arange(grid_size), indexing="ij"
            )
        )  # (2, 5, 5)
        coords_flat = coords.reshape(2, -1)  # (2, 25)
        # Pairwise offsets
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, 25, 25)
        rel[0] += grid_size - 1  # shift to non-negative
        rel[1] += grid_size - 1
        rel_idx = rel[0] * max_offset + rel[1]  # (25, 25)
        self.register_buffer("rel_idx", rel_idx.long())

    def forward(self):
        # Returns (num_heads, 25, 25)
        bias = self.bias_table[self.rel_idx]  # (25, 25, num_heads)
        return bias.permute(2, 0, 1)


class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias=None):
        # x: (B, N, d)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, attn_mask=attn_bias
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class NeighborEncoder(nn.Module):
    """NEM — Neighbor view encoder.

    Input: pre-extracted neighbor features (B, 25, 512)
    Transformer with relative position encoding -> avg pool -> FC predictor.
    """

    def __init__(
        self,
        n_genes=250,
        depth=3,
        num_heads=16,
        mlp_ratio=1,
        dropout=0.3,
        d_model=512,
    ):
        super().__init__()
        self.d_model = d_model
        self.rel_pos = RelativePositionBias(num_heads, grid_size=5)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.predictor = nn.Linear(d_model, n_genes)

    def forward(self, neighbor_feats):
        # neighbor_feats: (B, 25, 512)
        x = neighbor_feats
        attn_bias = self.rel_pos()  # (num_heads, 25, 25)
        # Expand for batch: MultiheadAttention expects (B*num_heads, N, N) or
        # broadcast-compatible shape. We repeat for batch dimension.
        B = x.shape[0]
        num_heads = attn_bias.shape[0]
        bias = attn_bias.unsqueeze(0).expand(B, -1, -1, -1)
        bias = bias.reshape(B * num_heads, 25, 25)

        for block in self.blocks:
            x = block(x, attn_bias=bias)

        x = self.norm(x)  # (B, 25, d_model)
        pooled = x.mean(dim=1)  # (B, d_model)
        pred = self.predictor(pooled)  # (B, n_genes)
        return x, pred
