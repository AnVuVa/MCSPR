"""HydraTRIPLEX loss: hydra-weighted fusion + standard aux branches.

Mirrors TRIPLEXLoss exactly for target / neighbor / global branches (each
keeps its full-300 head and the (1-alpha) hard + alpha soft-target MSE).
The fusion branch becomes the weighted-by-module-size hydra MSE used by
HydraSTNet, mathematically identical to MSE on the reassembled (B, 300)
prediction.

Soft-target detail: aux branches' soft target is the **detached, reassembled
(B, 300) fusion prediction** — same value the baseline would see — so the
soft-target gradient pathway is preserved bit-for-bit (the per-gene gradient
at each aux head is unchanged by the hydra refactor).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class HydraTRIPLEXLoss(nn.Module):
    def __init__(
        self,
        idx_list: List[List[int]],
        alpha: float = 0.5,
        return_per_head: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.idx_list = [list(idx) for idx in idx_list]
        self.K = len(idx_list)
        self.return_per_head = return_per_head
        self.mse = nn.MSELoss()

    def _hydra_weighted_mse(self, per_head, y):
        """Σ_k m_k · MSE_k / Σ_k m_k  ==  MSE on the full (B, 300)."""
        sse_total = 0.0
        n_elements = 0
        head_losses = []
        for k in range(self.K):
            target_k = y[:, self.idx_list[k]]
            mse_k = F.mse_loss(per_head[k], target_k)
            head_losses.append(mse_k)
            sse_total = sse_total + mse_k * per_head[k].numel()
            n_elements += per_head[k].numel()
        return sse_total / n_elements, head_losses

    def forward(self, preds, y):
        q_fusion_full = preds["fusion"]               # (B, 300) reassembled
        q_fusion_per_head = preds["fusion_per_head"]  # list of K tensors
        q_fusion_sg_full = q_fusion_full.detach()

        components = {}
        branch_loss_sum = torch.tensor(0.0, device=y.device, dtype=y.dtype)
        for branch in ("target", "neighbor", "global"):
            q_j = preds[branch]
            l_gt = self.mse(q_j, y)
            l_soft = self.mse(q_j, q_fusion_sg_full)
            l_j = (1.0 - self.alpha) * l_gt + self.alpha * l_soft
            components[f"loss_{branch}"] = l_j.item()
            components[f"loss_{branch}_gt"] = l_gt.item()
            components[f"loss_{branch}_soft"] = l_soft.item()
            branch_loss_sum = branch_loss_sum + l_j

        l_fusion, head_losses = self._hydra_weighted_mse(
            q_fusion_per_head, y,
        )
        components["loss_fusion"] = l_fusion.item()

        total = branch_loss_sum + l_fusion
        components["loss_total"] = total.item()

        if self.return_per_head:
            return total, components, head_losses
        return total, components
