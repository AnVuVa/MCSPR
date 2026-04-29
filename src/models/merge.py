"""MERGE model architectures (ported from MERGE/utils/model.py).

Pure baseline; no MCSPR-related code lives here. MCSPR loss attaches in
src/experiments/merge_mcspr/ and operates on this model's outputs.

Components:
  ResnetMLP      - SimCLR-style ResNet18 backbone + 2-layer projection MLP.
                   Trainable. Loaded with the TCGA pretrained weights at
                   pretrained/model-low-v1.pth (path from CNN config).
  CNN_Predictor  - ResnetMLP + dropout + Linear(256 -> num_genes).
                   This is the patch-level CNN trained in stage 1.
  GATNet         - 4-layer GAT GNN (256 -> 448*8 -> 384*8 -> 256*8 -> num_genes).
                   Edges are constructed externally (spatial + hierarchical
                   clusters); see src/data/merge_graph.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.utils import dropout_edge


class ResnetMLP(nn.Module):
    """SimCLR-style backbone — ResNet18 (instance-norm) + Linear+ReLU+Linear(256).

    The 256-dim output is what the GNN consumes. The pretrained checkpoint is
    expected to match this exact architecture (the original MERGE code uses
    pretrained/model-low-v1.pth shipped with the upstream repo).
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
        # Drop the original Linear classifier; keep the conv tower.
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features  # 512
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        # Output dim hardcoded to 256 because that's what the upstream
        # pretrained weights expect.
        self.l2 = nn.Linear(num_ftrs, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return x


class CNN_Predictor(nn.Module):
    """Stage-1 CNN: ResnetMLP backbone + dropout + Linear(256 -> num_genes)."""

    def __init__(self, num_genes: int, config: dict):
        super().__init__()
        device = config["device"]
        self.module = ResnetMLP().to(device)

        # Load the upstream SimCLR pretrained backbone weights into self
        # (state_dict keys are prefixed with "module."). Honour both new
        # MCSPR-style "model" key and old MERGE-style flat config.
        cnn_cfg = config.get("CNN", {})
        pre_path = cnn_cfg.get("pretrained_path") or config.get("pretrained_path")
        if pre_path:
            self.load_state_dict(
                torch.load(pre_path, map_location=device, weights_only=False),
            )

        self.dropout = nn.Dropout(p=cnn_cfg.get("dropout", 0.2)).to(device)
        num_features = self.module.l2.out_features  # 256
        self.fc = nn.Linear(num_features, num_genes).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.module(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc(x)


class GATNet(nn.Module):
    """4-layer GAT — operates on (n_spots, 256) features and edge_index."""

    def __init__(self, num_genes: int, num_heads: int = 8, drop_edge: float = 0.2):
        super().__init__()
        dim1, dim2, dim3 = 448, 384, 256
        self.drop_edge = drop_edge
        self.nn1 = GATConv(256, dim1, num_heads)
        self.layer_norm1 = LayerNorm(dim1 * num_heads)
        self.nn2 = GATConv(dim1 * num_heads, dim2, num_heads)
        self.layer_norm2 = LayerNorm(dim2 * num_heads)
        self.nn3 = GATConv(dim2 * num_heads, dim3, num_heads)
        self.layer_norm3 = LayerNorm(dim3 * num_heads)
        self.nn4 = GATConv(dim3 * num_heads, num_genes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = dropout_edge(
            edge_index, p=self.drop_edge, training=self.training,
        )
        x = F.relu(self.nn1(x, edge_index))
        x = self.layer_norm1(x)
        x = F.relu(self.nn2(x, edge_index))
        x = self.layer_norm2(x)
        x = F.relu(self.nn3(x, edge_index))
        x = self.layer_norm3(x)
        x = self.nn4(x, edge_index)
        return x
