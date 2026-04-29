import torch
import torch.nn as nn

from src.models.resnet import ResNet18Features


class TargetEncoder(nn.Module):
    """TEM — Target spot encoder.

    Input: target spot image (B, 3, 224, 224)
    ResNet18 (trainable) -> (B, 512, 7, 7) -> reshape to (B, 49, 512)
    Average pool -> (B, 512) -> FC -> (B, n_genes)
    """

    def __init__(self, n_genes=250, pretrained_path=None):
        super().__init__()
        self.backbone = ResNet18Features(pretrained_path=pretrained_path)
        self.predictor = nn.Linear(512, n_genes)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        feat_map = self.backbone(x)  # (B, 512, 7, 7)
        B, C, H, W = feat_map.shape
        tokens = feat_map.reshape(B, C, H * W).permute(0, 2, 1)  # (B, 49, 512)
        pooled = tokens.mean(dim=1)  # (B, 512)
        pred = self.predictor(pooled)  # (B, n_genes)
        return tokens, pred
