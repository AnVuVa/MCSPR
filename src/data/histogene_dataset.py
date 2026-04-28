"""
HisToGeneDataset — whole-slide loader for graph-based models.

Each __getitem__ returns the complete set of spots from ONE slide.
HisToGene's spatial transformer requires all spots simultaneously
to compute positional attention over the spatial graph.

Key difference from STDataset (patch-based):
  - STDataset: __getitem__ returns ONE spot
  - HisToGeneDataset: __getitem__ returns ALL spots in ONE slide

Context weights are loaded per slide. MCSPR accumulates cross-slide
via EMA — it receives whatever context diversity exists on each slide.
Individual slides are tissue-homogeneous; diversity accumulates across
the training set via the EMA buffers.

Data source: MERGE-format preprocessed data (counts_spcs, wsi, tissue_positions)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional
import torchvision.transforms as T


class HisToGeneDataset(Dataset):
    """
    Whole-slide dataset for HisToGene and other graph-based ST models.

    Args:
        data_dir:         Path to MERGE-format data directory
        sample_names:     List of sample IDs for this split
        n_genes:          Number of genes (250 for top-250, 300 for HMHVG)
        patch_size:       Pixel size of each spot patch (default 224)
        augment:          Apply random flip/rotation augmentation (train only)
        context_dir:      Path to precomputed context weight .npy files
                          Shape per file: (N_spots, T)
        max_spots:        Hard cap per slide for memory safety (default 1024)
                          Slides exceeding this are randomly subsampled.
                          Set to None for no cap.
    """

    def __init__(
        self,
        data_dir: str,
        sample_names: List[str],
        n_genes: int = 250,
        patch_size: int = 224,
        augment: bool = False,
        context_dir: Optional[str] = None,
        max_spots: Optional[int] = 1024,
    ):
        self.data_dir = Path(data_dir)
        self.sample_names = sample_names
        self.n_genes = n_genes
        self.patch_size = patch_size
        self.augment = augment
        self.context_dir = Path(context_dir) if context_dir else None
        self.max_spots = max_spots
        # Option D: 300-gene SVG panel swaps counts_spcs -> counts_svg and
        # features -> features_svg. Matches STDataset behavior.
        self._counts_dirname = "counts_svg" if n_genes == 300 else "counts_spcs"
        self._features_dirname = "features_svg" if n_genes == 300 else "features"

        # Image normalization (ImageNet stats — standard for pathology models)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        if augment:
            self.aug = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomApply(
                    [T.Lambda(lambda x: torch.rot90(x, 1, [1, 2]))], p=0.5
                ),
            ])
        else:
            self.aug = None

        # Load all slides at init — cache expression + coords in RAM
        self._slides: List[Dict] = []
        self._load_all_slides()

    def _load_all_slides(self):
        """Load expression, coordinates, and context weights for all slides."""
        gene_order = None

        for sample in self.sample_names:
            expr_path = self.data_dir / self._counts_dirname / f"{sample}.npy"
            coord_path = self.data_dir / "tissue_positions" / f"{sample}.csv"
            feat_path = self.data_dir / self._features_dirname / f"{sample}.csv"
            wsi_path = self.data_dir / "wsi" / f"{sample}.jpg"

            if not expr_path.exists():
                print(f"  WARN: {expr_path} missing, skipping {sample}")
                continue

            Y = np.load(expr_path).astype(np.float32)
            coords = pd.read_csv(coord_path, index_col=0)
            genes = pd.read_csv(feat_path, header=None)[0].tolist()

            if gene_order is None:
                gene_order = genes
            elif genes != gene_order:
                raise ValueError(
                    f"Gene order mismatch in {sample}. "
                    "All samples must share identical gene ordering."
                )

            N = Y.shape[0]
            if N < 128:
                print(f"  SKIP: {sample} has {N} spots < 128 minimum")
                continue

            spot_idx = np.arange(N)
            if self.max_spots is not None and N > self.max_spots:
                spot_idx = np.random.choice(N, self.max_spots, replace=False)
                spot_idx = np.sort(spot_idx)
                Y = Y[spot_idx]
                coords = coords.iloc[spot_idx]

            grid_coords = coords[["array_row", "array_col"]].values.astype(
                np.float32
            )
            grid_min = grid_coords.min(0)
            grid_max = grid_coords.max(0)
            grid_range = np.maximum(grid_max - grid_min, 1.0)
            grid_coords_norm = (grid_coords - grid_min) / grid_range

            ctx_w = None
            if self.context_dir is not None:
                ctx_path = self.context_dir / f"{sample}.npy"
                if ctx_path.exists():
                    ctx_w = np.load(ctx_path).astype(np.float32)
                    if self.max_spots is not None and len(spot_idx) < len(ctx_w):
                        ctx_w = ctx_w[spot_idx]

            self._slides.append({
                "sample": sample,
                "Y": Y,
                "grid_coords": grid_coords,
                "grid_norm": grid_coords_norm,
                "ctx_weights": ctx_w,
                "wsi_path": wsi_path if wsi_path.exists() else None,
                "n_spots": len(Y),
            })

        print(
            f"HisToGeneDataset: loaded {len(self._slides)} slides "
            f"({sum(s['n_spots'] for s in self._slides)} total spots)"
        )

    def __len__(self) -> int:
        return len(self._slides)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns ALL spots from slide idx.

        Return dict:
          expression:     (N, m)   float32 — log1p normalized
          grid_coords:    (N, 2)   float32 — raw grid [row, col]
          grid_norm:      (N, 2)   float32 — normalized [0,1]
          context_weights:(N, T)   float32 — soft context membership
          patches:        (N, 3, patch_size, patch_size) — if WSI available
          sample_name:    str
          n_spots:        int
        """
        slide = self._slides[idx]

        out = {
            "expression": torch.from_numpy(slide["Y"]),
            "grid_coords": torch.from_numpy(slide["grid_coords"]),
            "grid_norm": torch.from_numpy(slide["grid_norm"]),
            "sample_name": slide["sample"],
            "n_spots": slide["n_spots"],
        }

        if slide["ctx_weights"] is not None:
            out["context_weights"] = torch.from_numpy(slide["ctx_weights"])
        else:
            T_ctx = 6
            out["context_weights"] = (
                torch.ones(slide["n_spots"], T_ctx) / T_ctx
            )

        if slide["wsi_path"] is not None:
            patches = self._extract_patches(slide)
            if self.aug is not None:
                patches = torch.stack([self.aug(p) for p in patches])
            out["patches"] = self.normalize(patches)
        else:
            out["patches"] = torch.zeros(
                slide["n_spots"], 3, self.patch_size, self.patch_size
            )

        return out

    def _extract_patches(self, slide: Dict) -> torch.Tensor:
        """Extract patch_size x patch_size crops centered on each spot."""
        wsi = Image.open(slide["wsi_path"]).convert("RGB")
        w, h = wsi.size
        coords = slide["grid_coords"]

        row_min, row_max = coords[:, 0].min(), coords[:, 0].max()
        col_min, col_max = coords[:, 1].min(), coords[:, 1].max()
        n_rows = max(row_max - row_min + 1, 1)
        n_cols = max(col_max - col_min + 1, 1)
        px_per_row = h / (n_rows + 1)
        px_per_col = w / (n_cols + 1)
        half = self.patch_size // 2

        patches = []
        for i in range(slide["n_spots"]):
            px = int((coords[i, 1] - col_min + 0.5) * px_per_col)
            py = int((coords[i, 0] - row_min + 0.5) * px_per_row)
            left = max(0, px - half)
            top = max(0, py - half)
            right = min(w, px + half)
            bottom = min(h, py + half)
            patch = wsi.crop((left, top, right, bottom))
            patch = patch.resize((self.patch_size, self.patch_size))
            t = torch.from_numpy(
                np.array(patch, dtype=np.float32).transpose(2, 0, 1) / 255.0
            )
            patches.append(t)

        return torch.stack(patches)
