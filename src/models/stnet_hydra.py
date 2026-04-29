"""Hydra variant of ST-Net: same DenseNet121 backbone, K parallel MLP heads.

One head per gene module. The backbone is byte-identical to the baseline so
that any difference between STNet and HydraSTNet is attributable to the head
partitioning and per-head capacity, not to the feature extractor.

Forward returns a list of K tensors (preds[k] has shape (B, m_k)). The caller
concatenates them in canonical column order for full-300-gene comparison —
that re-assembly lives in run_stnet_hydra.py / hydra_helpers.py so the model
itself stays free of registry knowledge.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as models


class HydraHead(nn.Module):
    """Linear → ReLU → Linear, mirrors the spec sketch."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class HydraSTNet(nn.Module):
    """ST-Net Hydra: DenseNet121 backbone + K parallel HydraHeads.

    Args:
        module_sizes: list of K integers; m_k = number of genes head k owns.
                      sum(module_sizes) must equal n_genes (e.g. 300).
        d_hidden:     hidden width of every HydraHead's first Linear (default 128).
        pretrained:   load ImageNet DenseNet121 weights (matches baseline).
    """

    def __init__(
        self,
        module_sizes: List[int],
        d_hidden: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()
        # Identical to STNet baseline backbone, see src/models/stnet.py
        weights = (
            models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        base = models.densenet121(weights=weights)
        d = base.classifier.in_features  # 1024
        self.features = base.features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.module_sizes = list(module_sizes)
        self.heads = nn.ModuleList(
            [HydraHead(d, d_hidden, m_k) for m_k in self.module_sizes]
        )
        self.n_genes = sum(self.module_sizes)
        self.K = len(self.module_sizes)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = self.avg_pool(z)
        return self.flatten(z)  # (B, 1024)

    def forward(self, batch: Dict) -> Dict:
        """Mirrors STNet's batch-dict signature so universal_trainer can call
        it transparently. Returns {"output": List[Tensor]} so downstream code
        can detect the multi-head case without touching shape semantics.
        """
        x = batch["target_img"]
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
        z = self._encode(x)
        return {"output": [head(z) for head in self.heads]}

    def forward_patch_based(self, batch: Dict, device) -> Dict:
        """Universal-trainer entry point (matches STNet baseline)."""
        moved = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        return self.forward(moved)
