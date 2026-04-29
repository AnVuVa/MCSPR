import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Features(nn.Module):
    """ResNet18 with global average pooling and FC removed.

    Outputs spatial feature map: (B, 512, 7, 7) for 224x224 input.
    Used as target spot encoder backbone.
    """

    def __init__(self, pretrained_path=None):
        super().__init__()
        base = models.resnet18(pretrained=False)
        # Remove avgpool and fc
        self.features = nn.Sequential(*list(base.children())[:-2])

        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            self.features.load_state_dict(state, strict=False)

    def forward(self, x):
        return self.features(x)  # (B, 512, 7, 7)
