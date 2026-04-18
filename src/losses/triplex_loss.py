import torch
import torch.nn as nn


class TRIPLEXLoss(nn.Module):
    """TRIPLEX fusion loss — Equations 14-16 from paper.

    For j in {target, neighbor, global}:
      L^j = (1 - alpha) * MSE(q^j, y) + alpha * MSE(q^j, q^F.detach())
    L^F = MSE(q^F, y)
    L_TRIPLEX = L^target + L^neighbor + L^global + L^F

    q^F is DETACHED before use as soft target in L^j.
    """

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, preds, y):
        """
        preds: dict with 'target', 'neighbor', 'global', 'fusion'
            each (B, n_genes)
        y: (B, n_genes) ground truth
        Returns: total loss (scalar), loss_components dict
        """
        q_fusion = preds["fusion"]
        q_fusion_sg = q_fusion.detach()  # stop-grad for soft target

        components = {}
        branch_loss_sum = torch.tensor(0.0, device=y.device, dtype=y.dtype)

        for branch in ["target", "neighbor", "global"]:
            q_j = preds[branch]
            l_gt = self.mse(q_j, y)
            l_soft = self.mse(q_j, q_fusion_sg)
            l_j = (1.0 - self.alpha) * l_gt + self.alpha * l_soft
            components[f"loss_{branch}"] = l_j.item()
            components[f"loss_{branch}_gt"] = l_gt.item()
            components[f"loss_{branch}_soft"] = l_soft.item()
            branch_loss_sum = branch_loss_sum + l_j

        l_fusion = self.mse(q_fusion, y)
        components["loss_fusion"] = l_fusion.item()

        total = branch_loss_sum + l_fusion
        components["loss_total"] = total.item()

        return total, components
