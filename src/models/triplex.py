import torch
import torch.nn as nn

from src.models.target_encoder import TargetEncoder
from src.models.neighbor_encoder import NeighborEncoder
from src.models.global_encoder import GlobalEncoder
from src.models.fusion_layer import FusionLayer
from src.models.resnet import ResNet18Features


class TRIPLEX(nn.Module):
    """Full TRIPLEX model — CVPR 2024.

    Forward pass:
    1. Target: target_img -> TargetEncoder -> z_ta, pred_ta
    2. Global: global_features + coords -> GlobalEncoder -> z_gl_all, pred_gl_all
    3. Neighbor: neighbor features -> NeighborEncoder -> z_ne, pred_ne
    4. Extract target spots' global tokens
    5. Fuse: FusionLayer(z_gl, z_ta, z_ne) -> z_ftn, pred_fusion
    """

    def __init__(self, config, n_genes=250):
        super().__init__()
        mc = config["model"]
        d_model = mc.get("d_model", 512)

        # Pretrained path for ResNet18
        pretrained_path = mc.get("pretrained_path", None)

        # Target encoder — trainable ResNet18
        self.target_encoder = TargetEncoder(
            n_genes=n_genes, pretrained_path=pretrained_path
        )

        # Frozen ResNet18 for extracting neighbor features at runtime
        self.frozen_resnet = ResNet18Features(pretrained_path=pretrained_path)
        for p in self.frozen_resnet.parameters():
            p.requires_grad = False

        # Neighbor encoder
        ne = mc.get("neighbor_encoder", {})
        self.neighbor_encoder = NeighborEncoder(
            n_genes=n_genes,
            depth=ne.get("depth", 3),
            num_heads=ne.get("num_heads", 16),
            mlp_ratio=ne.get("mlp_ratio", 1),
            dropout=ne.get("dropout", 0.3),
            d_model=d_model,
        )

        # Global encoder
        ge = mc.get("global_encoder", {})
        self.global_encoder = GlobalEncoder(
            n_genes=n_genes,
            depth=ge.get("depth", 3),
            num_heads=ge.get("num_heads", 16),
            mlp_ratio=ge.get("mlp_ratio", 4),
            dropout=ge.get("dropout", 0.3),
            d_model=d_model,
        )

        # Fusion layer
        fl = mc.get("fusion_layer", {})
        self.fusion_layer = FusionLayer(
            n_genes=n_genes,
            d_model=d_model,
            num_heads=fl.get("num_heads", 4),
            mlp_ratio=fl.get("mlp_ratio", 4),
            depth=fl.get("depth", 1),
            dropout=fl.get("dropout", 0.2),
        )

    def _extract_neighbor_features(self, neighbor_imgs):
        """Extract features from neighbor images using frozen ResNet18.

        If input is (B, 25, 3, 224, 224): apply frozen ResNet to get (B, 25, 512).
        If input is already (B, 25, 512): pass through directly.
        """
        if neighbor_imgs.dim() == 5:
            B, K, C, H, W = neighbor_imgs.shape
            imgs_flat = neighbor_imgs.reshape(B * K, C, H, W)
            # Chunk the frozen pass so 3200-image tensors don't blow VRAM.
            chunk = 256
            feats_list = []
            with torch.no_grad():
                for i in range(0, imgs_flat.shape[0], chunk):
                    fm = self.frozen_resnet(imgs_flat[i:i + chunk])
                    feats_list.append(fm.mean(dim=[2, 3]))
            feats = torch.cat(feats_list, dim=0)  # (B*K, 512)
            return feats.reshape(B, K, -1)  # (B, 25, 512)
        else:
            # Already pre-extracted (B, 25, 512)
            return neighbor_imgs

    def forward(self, batch):
        """
        batch keys:
          target_img:       (B, 3, 224, 224)
          neighbor_imgs:    (B, 25, 3, 224, 224) or pre-extracted (B, 25, 512)
          global_features:  (N, 512)   ALL spots in this slide
          spot_coords:      (N, 2)     ALL spot coords in this slide
          target_spot_idx:  (B,) int   indices into N for the B target spots

        Returns:
          preds: dict with 'target', 'neighbor', 'global', 'fusion'
          tokens: dict with 'z_ta', 'z_ne', 'z_gl', 'z_ftn'
        """
        target_img = batch["target_img"]
        neighbor_imgs = batch["neighbor_imgs"]
        global_features = batch["global_features"]
        spot_coords = batch["spot_coords"]
        target_spot_idx = batch["target_spot_idx"]

        # 1. Target encoder (trainable ResNet18)
        z_ta, pred_ta = self.target_encoder(target_img)  # (B,49,d), (B,m)

        # 2. Global encoder (all spots in slide)
        z_gl_all, pred_gl_all = self.global_encoder(
            global_features, spot_coords
        )  # (N,d), (N,m)

        # 3. Neighbor encoder
        neighbor_feats = self._extract_neighbor_features(neighbor_imgs)
        z_ne, pred_ne = self.neighbor_encoder(neighbor_feats)  # (B,25,d), (B,m)

        # 4. Extract target spots' global tokens
        z_gl = z_gl_all[target_spot_idx]  # (B, d)
        pred_gl = pred_gl_all[target_spot_idx]  # (B, m)

        # 5. Fusion
        z_ftn, pred_fusion = self.fusion_layer(
            z_gl, z_ta, z_ne
        )  # (B,d), (B,m)

        preds = {
            "target": pred_ta,
            "neighbor": pred_ne,
            "global": pred_gl,
            "fusion": pred_fusion,
            # Spec v2 amendment A7: MCSPR/loss attachment is
            # gene-expression space R^300; 'output' is the canonical
            # spec key, aliased to 'fusion' (the final (N,300) head).
            "output": pred_fusion,
        }
        tokens = {
            "z_ta": z_ta,
            "z_ne": z_ne,
            "z_gl": z_gl,
            "z_ftn": z_ftn,
        }
        return preds, tokens
