import torch
import torch.nn as nn


class APEGBlock(nn.Module):
    """Atypical Position Encoding Generator.

    Takes tokens at known grid positions, places them on a 2D spatial grid,
    applies depthwise 2D convolution, and re-extracts the positional signal.
    """

    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv2d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )

    def forward(self, tokens, coords, grid_h, grid_w):
        # tokens: (N, d), coords: (N, 2) [row, col]
        d = tokens.shape[1]
        grid = tokens.new_zeros(1, d, grid_h, grid_w)  # (1, d, H, W)

        rows = coords[:, 0].long()
        cols = coords[:, 1].long()
        grid[0, :, rows, cols] = tokens.T  # place tokens at positions

        conv_out = self.conv(grid)  # (1, d, H, W)

        # Re-extract at original positions
        pos_signal = conv_out[0, :, rows, cols].T  # (N, d)
        return pos_signal


class GlobalTransformerBlock(nn.Module):

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

    def forward(self, x):
        # x: (1, N, d) — single slide batch
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GlobalEncoder(nn.Module):
    """GEM — Global encoder with APEG.

    Input: pre-extracted global features for ALL spots in a WSI: (N, 512)
           spot coordinates: (N, 2) [grid row, col]

    APEG runs after each transformer block:
    1. Pass through transformer block
    2. Build spatial grid, apply depthwise conv, re-extract tokens
    3. Add positional signal to tokens
    4. Repeat for remaining blocks
    """

    def __init__(
        self,
        n_genes=250,
        depth=3,
        num_heads=16,
        mlp_ratio=4,
        dropout=0.3,
        d_model=512,
    ):
        super().__init__()
        self.d_model = d_model

        self.blocks = nn.ModuleList(
            [
                GlobalTransformerBlock(d_model, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.apeg_blocks = nn.ModuleList(
            [APEGBlock(d_model) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.predictor = nn.Linear(d_model, n_genes)

    def forward(self, global_feats, coords):
        # global_feats: (N, 512), coords: (N, 2)
        N = global_feats.shape[0]
        grid_h = int(coords[:, 0].max().item()) + 1
        grid_w = int(coords[:, 1].max().item()) + 1

        x = global_feats.unsqueeze(0)  # (1, N, d)

        for block, apeg in zip(self.blocks, self.apeg_blocks):
            x = block(x)  # (1, N, d)
            # APEG: add spatial position signal
            pos_signal = apeg(x.squeeze(0), coords, grid_h, grid_w)  # (N, d)
            x = x + pos_signal.unsqueeze(0)

        x = self.norm(x).squeeze(0)  # (N, d_model)
        pred = self.predictor(x)  # (N, n_genes)
        return x, pred
