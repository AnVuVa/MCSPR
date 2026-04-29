"""
HisToGene backbone.

Original paper: Pang et al. 2021 "Leveraging information in spatial
transcriptomics to predict super-resolution gene expression from
histology images in tumors"

Architecture:
  - Input: N spot patches, each (3, 224, 224) consistent with TRIPLEX
  - Spatial position encoding from grid coordinates
  - Vision Transformer (ViT) with spatial attention
  - Output: (N, n_genes) gene expression prediction

Integration with MCSPR:
  - Model receives FULL SLIDE (all N spots) in one forward pass
  - Positional graph between spots is computed from grid_coords
  - MCSPR attaches to the final prediction tensor Y_hat: (N, n_genes)
  - No modification to the model — MCSPR operates on the output only

Key hyperparameters from TRIPLEX Table 14 (SCC dataset, closest to our use):
  n_layers: 4-5, dim: 512-2048, num_heads: 4-16, dropout: 0.1-0.4
  These are tuned per dataset via the same hyperparameter protocol.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from typing import Optional, Tuple
import math


class SpatialPositionEncoding(nn.Module):
    """
    Learnable spatial position encoding from (row, col) grid coordinates.
    Produces a continuous positional embedding from normalized coordinates.
    Different from APEG (which uses convolution) — this is a simpler MLP
    approach appropriate for HisToGene's ViT backbone.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, grid_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_norm: (N, 2) normalized coordinates in [0, 1]
        Returns:
            pos_enc:   (N, d_model)
        """
        return self.mlp(grid_norm)


class HisToGeneTransformerBlock(nn.Module):
    """Single ViT transformer block with pre-norm."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_grad_checkpoint = use_grad_checkpoint
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def _inner(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(
            x_norm.unsqueeze(0),
            x_norm.unsqueeze(0),
            x_norm.unsqueeze(0),
            attn_mask=attn_mask,
        )
        x = x + x_attn.squeeze(0)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_grad_checkpoint and self.training:
            return ckpt.checkpoint(
                self._inner, x, attn_mask, use_reentrant=False
            )
        return self._inner(x, attn_mask)


class PatchEmbedding(nn.Module):
    """
    Patch embedding via lightweight CNN.
    Maps (3, 224, 224) spot image to d_model-dim token.

    For precomputed features: accepts (d_feat,) and projects to d_model.
    """

    def __init__(
        self,
        d_model: int,
        use_precomputed: bool = False,
        d_feat: int = 512,
        use_grad_checkpoint: bool = False,
        cnn_chunk_size: int = 128,
    ):
        super().__init__()
        self.use_precomputed = use_precomputed
        self.use_grad_checkpoint = use_grad_checkpoint
        self.cnn_chunk_size = cnn_chunk_size

        if use_precomputed:
            self.proj = nn.Linear(d_feat, d_model)
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.proj = nn.Linear(128, d_model)

    def _cnn_chunked(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], self.cnn_chunk_size):
            chunk = x[i : i + self.cnn_chunk_size]
            if self.use_grad_checkpoint and self.training:
                outs.append(ckpt.checkpoint(self.cnn, chunk, use_reentrant=False))
            else:
                outs.append(self.cnn(chunk))
        return torch.cat(outs, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, H, W) raw patches OR (N, d_feat) precomputed features
        Returns:
            tokens: (N, d_model)
        """
        if self.use_precomputed:
            return self.proj(x)
        return self.proj(self._cnn_chunked(x))


class HisToGene(nn.Module):
    """
    HisToGene: spatial ViT for gene expression prediction.

    Processes all N spots from a slide simultaneously, maintaining
    the spatial graph structure required for positional attention.

    Args:
        n_genes:          Number of target genes (250 or 300 for HMHVG)
        d_model:          Transformer hidden dimension (default 512)
        n_layers:         Number of transformer blocks
        num_heads:        Attention heads
        mlp_ratio:        FFN expansion ratio
        dropout:          Dropout probability
        use_precomputed:  If True, accepts precomputed patch features (N, 512)
                          rather than raw patches (N, 3, 224, 224)
        d_feat:           Dimension of precomputed features (default 512)
        build_spatial_graph: If True, compute proximity-based attention mask
                             restricting attention to k-nearest neighbors.
                             If False, full attention over all spots.
    """

    def __init__(
        self,
        n_genes: int = 250,
        d_model: int = 512,
        n_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
        use_precomputed: bool = False,
        d_feat: int = 512,
        build_spatial_graph: bool = False,
        k_neighbors: int = 8,
        use_grad_checkpoint: bool = False,
        cnn_chunk_size: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_genes = n_genes
        self.build_spatial_graph = build_spatial_graph
        self.k_neighbors = k_neighbors
        self.use_grad_checkpoint = use_grad_checkpoint

        self.patch_embed = PatchEmbedding(
            d_model, use_precomputed, d_feat,
            use_grad_checkpoint=use_grad_checkpoint,
            cnn_chunk_size=cnn_chunk_size,
        )
        self.pos_enc = SpatialPositionEncoding(d_model)
        self.blocks = nn.ModuleList([
            HisToGeneTransformerBlock(
                d_model, num_heads, mlp_ratio, dropout,
                use_grad_checkpoint=use_grad_checkpoint,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.predictor = nn.Linear(d_model, n_genes)

    def _build_proximity_mask(
        self,
        grid_norm: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Build attention mask: spot i can attend to its k nearest neighbors.
        Returns (N, N) bool mask where True = BLOCK attention (PyTorch convention).
        Spots can always attend to themselves.
        """
        N = grid_norm.shape[0]
        diff = grid_norm.unsqueeze(0) - grid_norm.unsqueeze(1)  # (N, N, 2)
        dists = (diff ** 2).sum(-1)  # (N, N)
        topk_vals, _ = torch.topk(
            dists, k=min(k + 1, N), largest=False, dim=1
        )
        threshold = topk_vals[:, -1].unsqueeze(1)
        block_mask = dists > threshold  # (N, N)
        block_mask.fill_diagonal_(False)
        return block_mask

    def forward(
        self,
        patches: torch.Tensor,
        grid_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass over all N spots in one slide.

        Args:
            patches:   (N, 3, H, W) or (N, d_feat) precomputed
            grid_norm: (N, 2) normalized coords in [0,1]

        Returns:
            Y_hat:   (N, n_genes) gene expression predictions
            tokens:  (N, d_model) latent tokens (for downstream tasks)
        """
        tokens = self.patch_embed(patches)
        pos = self.pos_enc(grid_norm)
        tokens = tokens + pos

        attn_mask = None
        if self.build_spatial_graph and self.training:
            attn_mask = self._build_proximity_mask(
                grid_norm, self.k_neighbors
            )

        for block in self.blocks:
            tokens = block(tokens, attn_mask=attn_mask)

        tokens = self.norm(tokens)
        Y_hat = self.predictor(tokens)

        return Y_hat, tokens
