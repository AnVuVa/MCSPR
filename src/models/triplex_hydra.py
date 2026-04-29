"""HydraTRIPLEX — TRIPLEX with K parallel HydraHeads on the fusion output.

Surgical change vs baseline TRIPLEX: only the FusionLayer's final
`nn.Linear(d_model, n_genes)` is replaced with K parallel HydraHeads (one per
gene module from the per-fold k-means registry). The three auxiliary
predictors (target / neighbor / global) keep their full (B, n_genes) heads
unchanged — they're aux losses scaled by alpha and not the canonically
evaluated output.

Why fusion-only Hydra:
  * The fusion head is the canonically evaluated output, so it gets the same
    "module-private parameters + per-gene gradient = baseline" property as
    HydraSTNet.
  * Aux heads still see the full panel and provide their usual regularization
    signal to the encoders; we don't lose TRIPLEX's training dynamics.
  * Cheap surgery (one Linear -> K MLPs) and trivial parameter delta.

Returned `preds['fusion']` is reassembled to (B, n_genes) so the existing
`TRIPLEXLoss` and downstream code keep working without modification. The
per-head list is also exposed at `preds['fusion_per_head']` for the
weighted-by-module-size Hydra loss path.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.models.triplex import TRIPLEX
from src.models.stnet_hydra import HydraHead


class FusionHydraPredictor(nn.Module):
    """K parallel HydraHeads + canonical-order reassembly.

    forward(z): returns (full_pred, per_head_list) where
      full_pred:  (B, n_genes) reassembled in registry-canonical order
      per_head_list: List[Tensor] of length K, head k is (B, m_k)
    """

    def __init__(
        self,
        d_in: int,
        module_sizes: List[int],
        idx_list: List[List[int]],
        d_hidden: int = 128,
    ):
        super().__init__()
        assert len(module_sizes) == len(idx_list)
        self.module_sizes = list(module_sizes)
        # Store gene-index lists as buffers so device/dtype follow the module.
        # Buffers don't accept python lists, so we store as long tensors.
        for k, idx in enumerate(idx_list):
            self.register_buffer(
                f"idx_{k}", torch.tensor(idx, dtype=torch.long), persistent=False
            )
        self.K = len(module_sizes)
        self.n_genes = sum(module_sizes)
        self.heads = nn.ModuleList(
            [HydraHead(d_in, d_hidden, m_k) for m_k in module_sizes]
        )

    def _idx(self, k: int) -> torch.Tensor:
        return getattr(self, f"idx_{k}")

    def forward(self, z: torch.Tensor):
        per_head = [head(z) for head in self.heads]
        full = torch.empty(
            z.shape[0], self.n_genes, dtype=per_head[0].dtype, device=z.device,
        )
        for k, h in enumerate(per_head):
            full[:, self._idx(k)] = h
        return full, per_head


class HydraTRIPLEX(TRIPLEX):
    """TRIPLEX subclass with K-head fusion output.

    Args:
        config:        same TRIPLEX config dict
        module_sizes:  list of K ints; sum == n_genes (e.g. 300)
        idx_list:      list of K lists; idx_list[k] is the column indices in
                       y/full-pred for head k. Order must match `module_sizes`.
        d_hidden:      hidden width of each HydraHead's first Linear.
    """

    def __init__(
        self,
        config: dict,
        module_sizes: List[int],
        idx_list: List[List[int]],
        d_hidden: int = 128,
    ):
        n_genes = sum(module_sizes)
        super().__init__(config, n_genes=n_genes)
        d_model = config["model"].get("d_model", 512)
        # Surgically replace the fusion layer's single Linear predictor with K
        # HydraHeads. Cross-attention blocks and z_ftn computation are kept.
        self.fusion_layer.predictor = FusionHydraPredictor(
            d_in=d_model,
            module_sizes=module_sizes,
            idx_list=idx_list,
            d_hidden=d_hidden,
        )
        self.module_sizes = list(module_sizes)
        self.K = len(module_sizes)
        self.n_genes = n_genes

    def forward(self, batch):
        # Re-implement TRIPLEX.forward minimally to capture per-head fusion
        # without invoking the parent's `pred_fusion = self.predictor(z_ftn)`
        # contract (which now returns a tuple).
        target_img = batch["target_img"]
        neighbor_imgs = batch["neighbor_imgs"]
        global_features = batch["global_features"]
        spot_coords = batch["spot_coords"]
        target_spot_idx = batch["target_spot_idx"]

        z_ta, pred_ta = self.target_encoder(target_img)
        z_gl_all, pred_gl_all = self.global_encoder(global_features, spot_coords)
        neighbor_feats = self._extract_neighbor_features(neighbor_imgs)
        z_ne, pred_ne = self.neighbor_encoder(neighbor_feats)
        z_gl = z_gl_all[target_spot_idx]
        pred_gl = pred_gl_all[target_spot_idx]

        # Fusion cross-attention is unchanged; only the predictor is hydra.
        q = z_gl.unsqueeze(1)
        z_gt = q
        for block in self.fusion_layer.target_cross_attn:
            z_gt = block(z_gt, z_ta)
        z_gn = q
        for block in self.fusion_layer.neighbor_cross_attn:
            z_gn = block(z_gn, z_ne)
        z_ftn = (z_gt + z_gn).squeeze(1)
        full_fusion, fusion_per_head = self.fusion_layer.predictor(z_ftn)

        preds = {
            "target": pred_ta,
            "neighbor": pred_ne,
            "global": pred_gl,
            "fusion": full_fusion,
            "fusion_per_head": fusion_per_head,
            "output": full_fusion,
        }
        tokens = {
            "z_ta": z_ta,
            "z_ne": z_ne,
            "z_gl": z_gl,
            "z_ftn": z_ftn,
        }
        return preds, tokens
