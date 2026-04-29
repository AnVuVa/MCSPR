"""
ST-Net backbone.

Original paper: He et al. 2020 "Integrating spatial gene expression
and breast tumour morphology via deep learning"

Architecture:
  - DenseNet121 pretrained on ImageNet
  - Replace final FC layer with Linear(1024, n_genes)
  - Input: (B, 3, 224, 224) individual spot patches
  - Output: (B, n_genes) gene expression predictions

MCSPR integration:
  - Patch-based: uses StratifiedContextSampler (cross-slide batching)
  - Attaches to direct output (no fusion head -- single prediction)
  - universal_trainer._patch_based_step handles this via preds['output']

Implementation notes:
  - TRIPLEX uses ResNet18. ST-Net uses DenseNet121.
    These are different architectures -- do not substitute.
  - We use the same ResNet18 pretrained weights as TRIPLEX for
    the feature extractor in EGN (per TRIPLEX paper Table G.6).
    ST-Net uses its own DenseNet121 -- do not change this.
  - Spot image size: 224x224 (consistent with TRIPLEX protocol,
    not original 112x112 from He et al. 2020 -- but larger image
    is strictly better and this is the field standard now)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict


class STNet(nn.Module):
    """
    ST-Net: DenseNet121 fine-tuned for spatial gene expression prediction.

    Args:
        n_genes:          Number of target genes (250 or 300 HMHVG)
        pretrained:       Use ImageNet pretrained weights (default True)
        dropout:          Dropout before prediction head (default 0.0)
                          He et al. original uses no dropout.
    """

    def __init__(
        self,
        n_genes: int = 250,
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        # DenseNet121 backbone
        base = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        # Remove the original classifier -- keep features only
        # DenseNet121 features output: (B, 1024) after adaptive avg pool
        self.features = base.features          # ConvNet stem + dense blocks
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Prediction head
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.predictor = nn.Linear(1024, n_genes)
        self.n_genes = n_genes

    def forward(self, batch: Dict) -> Dict:
        """
        Args:
            batch: dict from STDataset with key 'target_img': (B, 3, 224, 224)

        Returns:
            dict with key 'output': (B, n_genes)

        Note: returns dict with 'output' key so universal_trainer can
        retrieve Y_hat via preds.get('fusion', preds.get('output'))
        The MCSPR loss attaches to 'output' for ST-Net.
        """
        x = batch["target_img"]

        # Ensure input is on same device as model
        device = self.predictor.weight.device
        if x.device != device:
            x = x.to(device)

        x = self.features(x)           # (B, 1024, 7, 7)
        x = self.avg_pool(x)           # (B, 1024, 1, 1)
        x = self.flatten(x)            # (B, 1024)
        x = self.dropout(x)
        y = self.predictor(x)          # (B, n_genes)
        return {"output": y}

    def forward_patch_based(self, batch: Dict, device: torch.device) -> Dict:
        """
        Called by universal_trainer._patch_based_step.
        Moves batch tensors to device, runs forward.
        """
        moved = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        return self.forward(moved)
