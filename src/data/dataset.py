import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random

# HER2ST WSIs are ~92 MP; raise PIL's default DoS-safety cap so we don't
# flood logs with DecompressionBombWarning on every header read.
Image.MAX_IMAGE_PIXELS = None


class STDataset(Dataset):
    """Dataset for MERGE-format spatial transcriptomics data.

    Loads counts_spcs (SPCS-smoothed, log1p) data exclusively.
    Raises ValueError if accidentally loading from wrong directory.

    global_features and spot_coords are for the ENTIRE sample, not just
    one spot. The global encoder needs all spots in a WSI simultaneously.
    """

    def __init__(
        self,
        data_dir,
        dataset,
        sample_names,
        n_genes=250,
        neighbor_size=5,
        augment=False,
        n_contexts=6,
        context_dir=None,
        global_feat_dir=None,
        skip_unused=False,
        counts_subdir=None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.sample_names = sample_names
        self.n_genes = n_genes
        self.neighbor_size = neighbor_size
        self.augment = augment
        self.n_contexts = n_contexts
        # STNet-only fast path: target_img + expression only.
        # Skips the 25× neighbor PIL crops and global/spot_coords tensors
        # that STNet's forward never reads. See loaders.build_stnet_loaders.
        self.skip_unused = skip_unused

        # Option D: 300-gene SVG panel (counts_svg) instead of 250 MERGE.
        # Training targets are indexed in svg_genes.json order to match
        # NMF artifacts (M, M_pinv shape (300, 15)).
        # `counts_subdir` lets the caller override the default — needed for
        # Phase 2 fair-protocol runs that train on counts_svg_spcs but
        # evaluate against counts_svg.
        if counts_subdir is not None:
            counts_dirname = counts_subdir
        else:
            counts_dirname = "counts_svg" if n_genes == 300 else "counts_spcs"
        features_dirname = "features_svg" if n_genes == 300 else "features"
        self._counts_dirname = counts_dirname
        self._features_dirname = features_dirname
        counts_path = self.data_dir / counts_dirname
        if not counts_path.exists():
            raise ValueError(
                f"{counts_dirname} directory not found at {counts_path}. "
                "Run scripts/build_counts_svg.py first." if n_genes == 300
                else "SPCS smoothed data is required."
            )

        # Storage
        self.expressions = {}
        self.coords = {}           # grid (array_row, array_col)
        self.pixel_coords = {}     # pixel (cy, cx) centers per spot
        self.wsi_size = {}         # (W, H) per sample
        self.grid_pitch = {}       # (px_r, px_c) pixels per grid unit
        self.gene_names = None
        self.barcodes = {}
        self.wsi_images = {}
        self.global_features = {}
        self.context_weights = {}
        self.items = []
        # Precomputed target patches: (N, 224, 224, 3) uint8 memmap per sample.
        # Eliminates PIL from the target-patch hot path; neighbors stay PIL.
        self._patch_cache = {}

        # Resolve context and global feature directories
        self.context_dir = Path(context_dir) if context_dir else None
        self.global_feat_dir = (
            Path(global_feat_dir)
            if global_feat_dir
            else self.data_dir / "global_features"
        )

        self._load_all_samples()

    def _load_all_samples(self):
        reference_genes = None

        for s_idx, s_name in enumerate(self.sample_names):
            # Load expression counts (counts_svg for 300-SVG, else counts_spcs)
            expr_path = self.data_dir / self._counts_dirname / f"{s_name}.npy"
            Y_s = np.load(str(expr_path)).astype(np.float32)  # (N_s, n_genes)
            self.expressions[s_idx] = Y_s

            # Load tissue positions (MERGE/Visium schema).
            # Columns: idx, in_tissue, array_row, array_col,
            #          pxl_col_in_fullres, pxl_row_in_fullres.
            # We store GRID coords (array_row, array_col) in self.coords and
            # compute pixel centers separately using the full WSI size below.
            pos_path = self.data_dir / "tissue_positions" / f"{s_name}.csv"
            import pandas as _pd  # local import; dataset.py doesn't import pd globally
            _pos_df = _pd.read_csv(str(pos_path), index_col=0)
            coords_s = _pos_df[["array_row", "array_col"]].values.astype(
                np.float32
            )
            self.coords[s_idx] = coords_s  # (N_s, 2) grid

            # Load gene names (features_svg for 300-SVG, else features)
            feat_path = self.data_dir / self._features_dirname / f"{s_name}.csv"
            with open(str(feat_path), "r") as f:
                genes = [line.strip() for line in f if line.strip()]
            if reference_genes is None:
                reference_genes = genes
            elif genes != reference_genes:
                raise ValueError(
                    f"Gene order mismatch for sample {s_name}. "
                    f"Expected {reference_genes[:5]}..., "
                    f"got {genes[:5]}..."
                )

            # Load barcodes
            bc_path = self.data_dir / "barcodes" / f"{s_name}.csv"
            with open(str(bc_path), "r") as f:
                barcodes = [line.strip() for line in f if line.strip()]
            self.barcodes[s_idx] = barcodes

            # Lazy-load WSI
            wsi_path = self.data_dir / "wsi" / f"{s_name}.jpg"
            if wsi_path.exists():
                self.wsi_images[s_name] = wsi_path  # store path, load lazily
                # Read size only (PIL lazy header read; no decode).
                # Match the MERGE precompute pixel formula:
                #   px_r = H / (n_r + 1); px_c = W / (n_c + 1)
                #   cy = (row - row_min + 0.5) * px_r
                #   cx = (col - col_min + 0.5) * px_c
                with Image.open(str(wsi_path)) as _wsi:
                    W_img, H_img = _wsi.size
                rows = coords_s[:, 0].astype(int)
                cols = coords_s[:, 1].astype(int)
                n_r = int(rows.max() - rows.min() + 1)
                n_c = int(cols.max() - cols.min() + 1)
                px_r = H_img / (n_r + 1)
                px_c = W_img / (n_c + 1)
                cy = (rows - rows.min() + 0.5) * px_r
                cx = (cols - cols.min() + 0.5) * px_c
                self.wsi_size[s_name] = (W_img, H_img)
                self.grid_pitch[s_name] = (float(px_r), float(px_c))
                self.pixel_coords[s_idx] = np.stack(
                    [cy, cx], axis=1
                ).astype(np.float32)
            else:
                self.wsi_images[s_name] = None
                self.pixel_coords[s_idx] = None

            # Load precomputed global features if available
            gf_path = self.global_feat_dir / f"{s_name}.npy"
            if gf_path.exists():
                self.global_features[s_idx] = np.load(str(gf_path)).astype(
                    np.float32
                )

            # Load context weights if available
            if self.context_dir is not None:
                cw_path = self.context_dir / f"{s_name}.npy"
                if cw_path.exists():
                    self.context_weights[s_idx] = np.load(str(cw_path)).astype(
                        np.float32
                    )

            # Optional precomputed target-patch cache (N, 224, 224, 3) uint8.
            wsi224_path = self.data_dir / "wsi224" / f"{s_name}.npy"
            if wsi224_path.exists():
                self._patch_cache[s_idx] = np.load(
                    str(wsi224_path), mmap_mode="r"
                )

            # Build flat index
            N_s = Y_s.shape[0]
            for spot_idx in range(N_s):
                self.items.append((s_idx, spot_idx))

        self.gene_names = reference_genes

    def _load_wsi(self, sample_name):
        path = self.wsi_images.get(sample_name)
        if path is None:
            return None
        # Per-slide LRU cache (size 1). SlideBatchSampler guarantees all spots
        # in a batch come from one slide, so loading + converting the 92-MP
        # WSI exactly once per slide instead of per spot is the difference
        # between hours and days.
        cache = getattr(self, "_wsi_cache", None)
        if cache is None:
            cache = {}
            self._wsi_cache = cache
        if sample_name in cache:
            return cache[sample_name]
        img = Image.open(str(path)).convert("RGB")
        cache.clear()
        cache[sample_name] = img
        return img

    def _extract_patch(self, wsi, cy, cx, crop_size=224):
        """Extract a crop_size x crop_size patch centered at pixel (cy, cx)."""
        if wsi is None:
            return torch.zeros(3, crop_size, crop_size)

        w, h = wsi.size  # PIL: (width, height)
        cy = int(round(cy))
        cx = int(round(cx))
        half = crop_size // 2

        y1, y2 = cy - half, cy + half
        x1, x2 = cx - half, cx + half

        # Clamp to image bounds (entire crop outside → return zeros)
        y1c = max(0, min(h, y1))
        y2c = max(0, min(h, y2))
        x1c = max(0, min(w, x1))
        x2c = max(0, min(w, x2))

        result = np.zeros((crop_size, crop_size, 3), dtype=np.float32)
        if y2c > y1c and x2c > x1c:
            crop = wsi.crop((x1c, y1c, x2c, y2c))
            crop_arr = np.array(crop, dtype=np.float32) / 255.0
            pad_top = y1c - y1
            pad_left = x1c - x1
            ch = crop_arr.shape[0]
            cw_px = crop_arr.shape[1]
            result[pad_top : pad_top + ch, pad_left : pad_left + cw_px] = crop_arr

        return torch.from_numpy(result).permute(2, 0, 1)  # (3, H, W)

    def _extract_neighbor_patches(
        self, wsi, cy, cx, px_r, px_c, crop_size=224
    ):
        """5x5 = 25 neighbor patches stepped by the per-slide grid pitch."""
        patches = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                ncy = cy + dr * px_r
                ncx = cx + dc * px_c
                patches.append(self._extract_patch(wsi, ncy, ncx, crop_size))
        return torch.stack(patches)  # (25, 3, 224, 224)

    def _apply_augmentation(self, target_img, neighbor_imgs):
        """Apply SAME random augmentation to target + all neighbor patches."""
        # Random horizontal flip
        if random.random() > 0.5:
            target_img = torch.flip(target_img, [2])
            neighbor_imgs = torch.flip(neighbor_imgs, [3])

        # Random vertical flip
        if random.random() > 0.5:
            target_img = torch.flip(target_img, [1])
            neighbor_imgs = torch.flip(neighbor_imgs, [2])

        # Random 90/180/270 rotation
        k = random.randint(0, 3)
        if k > 0:
            target_img = torch.rot90(target_img, k, [1, 2])
            neighbor_imgs = torch.rot90(neighbor_imgs, k, [2, 3])

        return target_img, neighbor_imgs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s_idx, spot_idx = self.items[idx]
        s_name = self.sample_names[s_idx]
        coords = self.coords[s_idx]

        # Fast path for STNet: target patch only, no neighbors, no globals.
        if self.skip_unused:
            cached = self._patch_cache.get(s_idx)
            if cached is not None:
                arr = np.asarray(cached[spot_idx], dtype=np.float32) / 255.0
                target_img = torch.from_numpy(arr).permute(2, 0, 1)
            else:
                # Fallback only if wsi224 cache missing (shouldn't happen).
                wsi = self._load_wsi(s_name)
                if wsi is not None and self.pixel_coords.get(s_idx) is not None:
                    cy, cx = self.pixel_coords[s_idx][spot_idx]
                else:
                    cy, cx = 0.0, 0.0
                target_img = self._extract_patch(wsi, cy, cx)

            if self.augment:
                if random.random() > 0.5:
                    target_img = torch.flip(target_img, [2])
                if random.random() > 0.5:
                    target_img = torch.flip(target_img, [1])
                k = random.randint(0, 3)
                if k > 0:
                    target_img = torch.rot90(target_img, k, [1, 2])

            expression = torch.from_numpy(
                self.expressions[s_idx][spot_idx]
            ).float()

            return {
                "target_img": target_img,
                "expression": expression,
                "sample_idx": s_idx,
                "spot_idx": spot_idx,
            }

        wsi = self._load_wsi(s_name)
        if wsi is not None and self.pixel_coords.get(s_idx) is not None:
            cy, cx = self.pixel_coords[s_idx][spot_idx]
            px_r, px_c = self.grid_pitch[s_name]
        else:
            # Fallback: no WSI available — zero tensors downstream.
            cy, cx, px_r, px_c = 0.0, 0.0, 224.0, 224.0

        # Target patch: use wsi224 mmap cache when present; else PIL crop.
        cached = self._patch_cache.get(s_idx)
        if cached is not None:
            arr = np.asarray(cached[spot_idx], dtype=np.float32) / 255.0
            target_img = torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)
        else:
            target_img = self._extract_patch(wsi, cy, cx)
        neighbor_imgs = self._extract_neighbor_patches(
            wsi, cy, cx, px_r, px_c
        )

        # Augmentation
        if self.augment:
            target_img, neighbor_imgs = self._apply_augmentation(
                target_img, neighbor_imgs
            )

        # Expression
        expression = torch.from_numpy(
            self.expressions[s_idx][spot_idx]
        ).float()

        # Global features (entire sample)
        if s_idx in self.global_features:
            global_feats = torch.from_numpy(
                self.global_features[s_idx]
            ).float()
        else:
            global_feats = torch.zeros(coords.shape[0], 512)

        # Spot coords (entire sample)
        spot_coords = torch.from_numpy(coords).float()

        # Context weights (this spot)
        if s_idx in self.context_weights:
            ctx_w = torch.from_numpy(
                self.context_weights[s_idx][spot_idx]
            ).float()
        else:
            ctx_w = torch.ones(self.n_contexts) / self.n_contexts

        return {
            "target_img": target_img,
            "neighbor_imgs": neighbor_imgs,
            "global_features": global_feats,
            "spot_coords": spot_coords,
            "expression": expression,
            "context_weights": ctx_w,
            "sample_idx": s_idx,
            "spot_idx": spot_idx,
        }
