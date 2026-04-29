import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):

    def __init__(self, d_model, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, kv):
        # query: (B, 1, d), kv: (B, S, d)
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        out = query + attn_out
        out = out + self.mlp(self.norm_ff(out))
        return out  # (B, 1, d)


class FusionLayer(nn.Module):
    """Cross-attention fusion (Eq 13 from paper).

    For each spot i:
      - Global token is the QUERY
      - Target tokens are KEY/VALUE: z_GT = CrossAttn(Q=z_Gl_i, KV=z_Ta_i)
      - Neighbor tokens are KEY/VALUE: z_GN = CrossAttn(Q=z_Gl_i, KV=z_Ne_i)
      - Fusion: z_GTN = z_GT + z_GN
      - Predictor FC on z_GTN
    """

    def __init__(
        self,
        n_genes=250,
        d_model=512,
        num_heads=4,
        mlp_ratio=4,
        depth=1,
        dropout=0.2,
    ):
        super().__init__()
        self.target_cross_attn = nn.ModuleList(
            [
                CrossAttentionBlock(d_model, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.neighbor_cross_attn = nn.ModuleList(
            [
                CrossAttentionBlock(d_model, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.predictor = nn.Linear(d_model, n_genes)

    def forward(self, z_gl, z_ta, z_ne):
        # z_gl: (B, d)  — global tokens for target spots
        # z_ta: (B, 49, d) — target tokens
        # z_ne: (B, 25, d) — neighbor tokens
        q = z_gl.unsqueeze(1)  # (B, 1, d)

        # Cross-attend to target tokens
        z_gt = q
        for block in self.target_cross_attn:
            z_gt = block(z_gt, z_ta)  # (B, 1, d)

        # Cross-attend to neighbor tokens
        z_gn = q
        for block in self.neighbor_cross_attn:
            z_gn = block(z_gn, z_ne)  # (B, 1, d)

        # Fusion: z_GTN = z_GT + z_GN
        z_ftn = (z_gt + z_gn).squeeze(1)  # (B, d)
        fusion_pred = self.predictor(z_ftn)  # (B, n_genes)
        return z_ftn, fusion_pred
