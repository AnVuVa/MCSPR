"""MERGE data pipeline (ported from MERGE/utils/data.py).

Stage 1 (CNN): per-patch dataset (LazyGeneDataset). Each item is a single
spot's 256x256 patch + its (num_genes,) label vector.
Stage 2 (GNN): per-slide dataset (GraphDataset, in src/data/merge_graph.py).

Reads the MCSPR 4-fold patient-paired manifest by default, OR a MERGE-format
manifest if pointed at one. Counts directory is configurable; default
"counts_svg" for the 300-SVG canonical panel that matches our other baselines.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import PIL
import numpy as np
import pandas as pd
import skimage.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = None


class LazyGeneDataset(Dataset):
    """Patch-level dataset for CNN training.

    Items are (file_path, local_idx, label) — patches are read from the
    cached per-slide tensor file via memory-mapped torch.load on demand.
    Only spots with non-zero count totals are materialised (matches
    upstream MERGE's nonzero filter).
    """

    def __init__(self, patch_files, counts, nonzeros, transform=None):
        self.items = []
        self.transform = transform
        for pf, c, nz in zip(patch_files, counts, nonzeros):
            n_patches = int(nz.sum())
            nonzero_counts = c[nz]
            for local_idx in range(n_patches):
                self.items.append((pf, local_idx, nonzero_counts[local_idx]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        pf, local_idx, label = self.items[idx]
        patches = torch.load(pf, map_location="cpu", mmap=True)
        image = patches[local_idx]
        sample = {"image": image, "label": np.array([label]).squeeze(0)}
        if self.transform is not None:
            sample["image"] = self.transform(sample["image"])
        return sample


def _extract_and_save_patches(
    data_path: str, slide: str, tissue_positions: pd.DataFrame, cache_dir: str,
) -> str:
    """Extract 256x256 RGB patches from the WSI at each in-tissue spot and
    save as a single per-slide tensor for memory-mapped lazy loading."""
    cache_path = os.path.join(cache_dir, f"{slide}.pt")
    if os.path.exists(cache_path):
        return cache_path

    wsi_path = None
    for ext in ("tif", "tiff", "svs", "jpg"):
        candidate = f"{data_path}/wsi/{slide}.{ext}"
        if os.path.exists(candidate):
            wsi_path = candidate
            break
    if wsi_path is None:
        raise FileNotFoundError(f"No WSI file found for slide {slide}")

    wsi = skimage.io.imread(wsi_path)
    patches = []
    x_coords = tissue_positions["pxl_col_in_fullres"].values
    y_coords = tissue_positions["pxl_row_in_fullres"].values

    for x, y in zip(x_coords, y_coords):
        x, y = round(x), round(y)
        # Clamp so the 256x256 crop stays inside the WSI.
        if x < 128:
            x = 128
        if y < 128:
            y = 128
        if x > wsi.shape[0] - 128:
            x = wsi.shape[0] - 128
        if y > wsi.shape[1] - 128:
            y = wsi.shape[1] - 128
        patch = wsi[y - 128:y + 128, x - 128:x + 128, :3]
        patches.append(patch)

    del wsi
    patches = torch.tensor(
        np.array(patches).transpose([0, 3, 1, 2]), dtype=torch.float,
    )
    torch.save(patches, cache_path)
    del patches
    return cache_path


def _resolve_fold_split(splits_path: str, fold_idx: int):
    """Accept both MCSPR 4-fold manifest format and MERGE legacy format."""
    with open(splits_path) as f:
        all_splits = json.load(f)
    if "folds" in all_splits:
        fold_split = all_splits["folds"][fold_idx]
        return fold_split["train_slides"], fold_split["val_slides"]
    fold_key = f"fold_{fold_idx}"
    fold_split = all_splits[fold_key]
    train = [s.split("/")[-1] for s in fold_split["train"]]
    val = [s.split("/")[-1] for s in fold_split["val"]]
    return train, val


def preprocess_data(config: dict):
    """Returns (data, image_datasets, dataloaders, dataset_sizes).

    `data` collects per-slide barcodes / spotnum / counts / nonzero mask /
    tissue_positions / cached patch path / patch_embeddings (filled later by
    `generate_features`). The CNN trains on `image_datasets["train|val"]`;
    the GNN consumes `data` once embeddings are populated.
    """
    data_path = config["Data"]["path"]
    counts_dir = config["Data"].get("counts_dir", "counts_svg")
    splits_path = config["Data"]["splits"]
    fold_idx = config["Data"]["fold"]

    train_slides, val_slides = _resolve_fold_split(splits_path, fold_idx)
    all_slides = sorted(train_slides + val_slides)

    data = {
        "barcodes": [], "spotnum": [], "counts": [], "nonzero": [],
        "tissue_positions": [], "patch_files": [], "patch_embeddings": [],
        "num_genes": config["Data"]["num_genes"],
        "slides": np.array(all_slides),
    }
    data["train_slides"] = np.isin(data["slides"], np.array(train_slides))
    data["val_slides"] = np.isin(data["slides"], np.array(val_slides))

    cache_dir = os.path.join(data_path, "patches_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    for slide in tqdm(data["slides"], desc="Loading slides"):
        barcodes = pd.read_csv(
            f"{data_path}/barcodes/{slide}.csv", header=None,
        )[0].values
        data["barcodes"].append(barcodes)
        data["spotnum"].append(len(barcodes))

        tp = pd.read_csv(f"{data_path}/tissue_positions/{slide}.csv", index_col=0)
        tp = tp[tp["in_tissue"] == 1]
        data["tissue_positions"].append(tp)

        counts = np.load(f"{data_path}/{counts_dir}/{slide}.npy")
        data["nonzero"].append(counts.sum(axis=1) > 0)
        data["counts"].append(counts)

        cache_path = _extract_and_save_patches(data_path, slide, tp, cache_dir)
        data["patch_files"].append(cache_path)

    # Standard ImageNet normalization. The training transform omits horizontal
    # flip on purpose (matches upstream MERGE; flip lives on the val side).
    train_tfm = torch.nn.Sequential(
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    )
    val_tfm = torch.nn.Sequential(
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    )

    train_pf, val_pf, train_c, val_c, train_nz, val_nz = [], [], [], [], [], []
    for i in range(len(data["slides"])):
        if data["train_slides"][i]:
            train_pf.append(data["patch_files"][i])
            train_c.append(data["counts"][i])
            train_nz.append(data["nonzero"][i])
        if data["val_slides"][i]:
            val_pf.append(data["patch_files"][i])
            val_c.append(data["counts"][i])
            val_nz.append(data["nonzero"][i])

    image_datasets = {
        "train": LazyGeneDataset(train_pf, train_c, train_nz, train_tfm),
        "val": LazyGeneDataset(val_pf, val_c, val_nz, val_tfm),
    }
    dataset_sizes = {k: len(v) for k, v in image_datasets.items()}
    bs = config["CNN"]["batch_size"]
    dataloaders = {
        k: torch.utils.data.DataLoader(
            image_datasets[k], batch_size=bs, shuffle=True, num_workers=4,
        )
        for k in ("train", "val")
    }
    return data, image_datasets, dataloaders, dataset_sizes
