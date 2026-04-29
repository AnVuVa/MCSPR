"""Ciga et al. (2021) SimCLR ResNet18, self-supervised on multi-organ histology.

The checkpoint at `TRIPLEX/weights/cigar/tenpercent_resnet18.ckpt` is a
PyTorch-Lightning SimCLR snapshot. Its `state_dict` keys are all prefixed
`model.resnet.` and include a two-layer projection head (`fc.1.*`, `fc.3.*`)
that does not exist in torchvision's stock ResNet18. We strip the prefix,
drop the head, and return a ResNet18 whose `fc` is `nn.Identity` — i.e. a
512-d global-pooled feature extractor matching the ImageNet path it replaces.
"""

import sys
import types

import torch
import torch.nn as nn
import torchvision.models as tv_models


def _install_lightning_stub() -> None:
    """Lightning checkpoints pickle `ModelCheckpoint` instances inside
    `callbacks`. If pytorch_lightning isn't installed, `torch.load` fails
    at unpickling. Stubbing the referenced classes lets us read the dict."""
    if "pytorch_lightning" in sys.modules:
        return
    pl = types.ModuleType("pytorch_lightning")
    pl_cb = types.ModuleType("pytorch_lightning.callbacks")
    pl_mc = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")

    class _Stub:
        def __init__(self, *a, **kw):
            pass

        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {})

    pl_mc.ModelCheckpoint = _Stub
    pl_cb.ModelCheckpoint = _Stub
    sys.modules["pytorch_lightning"] = pl
    sys.modules["pytorch_lightning.callbacks"] = pl_cb
    sys.modules["pytorch_lightning.callbacks.model_checkpoint"] = pl_mc


def load_ciga_resnet18(ckpt_path: str, device: str = "cpu") -> nn.Module:
    _install_lightning_stub()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    prefix = "model.resnet."
    backbone_sd = {
        k[len(prefix):]: v
        for k, v in sd.items()
        if k.startswith(prefix) and not k.startswith(prefix + "fc.")
    }
    if not backbone_sd:
        raise RuntimeError(
            f"No backbone keys with prefix '{prefix}' found in {ckpt_path}"
        )

    model = tv_models.resnet18(weights=None)
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)

    non_fc_missing = [k for k in missing if not k.startswith("fc.")]
    if non_fc_missing:
        raise RuntimeError(
            f"Ciga checkpoint missing non-fc backbone keys: {non_fc_missing[:5]}"
        )
    if unexpected:
        raise RuntimeError(
            f"Ciga checkpoint had unexpected keys: {unexpected[:5]}"
        )

    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device).eval()

    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224, device=device))
    assert out.shape == (1, 512), f"Expected (1, 512), got {tuple(out.shape)}"

    return model
