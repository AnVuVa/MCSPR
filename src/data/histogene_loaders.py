"""
HisToGene patient-LOPCV data loaders.

Key difference from patch-based loaders:
  - Batch size = number of SLIDES, not number of spots
  - Each element of the batch is a complete WSI (variable N spots)
  - Custom collate_fn handles variable-length spot sequences
  - DataLoader batch_size=1 is the default (one slide per step)
    For multi-slide batching, use batch_size>1 with the provided collate_fn.

Variable-length handling strategy:
  Option A (used here): batch_size=1, no padding needed.
    HisToGene processes one slide at a time. Simple, correct, no memory waste.
  Option B (alternative): Pad to max N in batch with attention mask.
    Only needed if GPU utilization is poor with batch_size=1.
    Implement if profiling shows <50% GPU utilization.

The EMA in MCSPRLoss accumulates across sequential single-slide forward
passes. After K training slides, all T contexts are populated regardless
of individual slide homogeneity.
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from src.data.histogene_dataset import HisToGeneDataset
# Single source of truth for patient-paired LOPCV splits. Previously this
# module shipped its own build_lopcv_folds that didn't accept n_folds and
# defaulted to 8 single-patient folds for her2st — incompatible with the
# 4-fold patient-paired split used by STNet/TRIPLEX (spec v2).
from src.data.loaders import build_lopcv_folds, get_patient_id  # noqa: F401


def whole_slide_collate_fn(batch: List[Dict]) -> List[Dict]:
    """
    Collate function for whole-slide batches.

    Returns the batch as a list of dicts (NOT stacked tensors).
    Each dict contains tensors for one complete slide.

    Why not stack? Variable spot counts per slide make stacking impossible
    without padding. For batch_size=1 this is trivially a list of length 1.
    For batch_size>1, the training loop iterates over the list.
    """
    return batch


def build_histogene_loaders(
    data_dir: str,
    dataset: str,
    fold_idx: int,
    config: dict,
    context_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for HisToGene.

    Args:
        data_dir:    Path to MERGE-format data directory
        dataset:     'her2st', 'stnet', or 'scc'
        fold_idx:    Which LOPCV fold to use
        config:      Dict with keys: n_genes, patch_size, max_spots, seed
        context_dir: Path to precomputed context weight files

    Returns:
        (train_loader, val_loader)
    """
    torch.manual_seed(config.get("seed", 2021))

    feature_dir = Path(data_dir) / "features"
    sample_names = sorted([f.stem for f in feature_dir.glob("*.csv")])
    if not sample_names:
        raise FileNotFoundError(f"No feature files found in {feature_dir}")

    folds = build_lopcv_folds(sample_names, dataset, n_folds=config.get("n_folds"))
    train_samples, val_samples = folds[fold_idx]
    print(
        f"Fold {fold_idx}: {len(train_samples)} train slides, "
        f"{len(val_samples)} val slides"
    )

    train_max_spots = config.get("max_spots", 1024)
    # val can often fit more since no backward pass (no stored activations).
    # Default val cap = 2x train cap; None = uncapped (legacy behavior).
    val_max_spots = config.get(
        "val_max_spots",
        train_max_spots * 2 if train_max_spots else None,
    )

    train_ds = HisToGeneDataset(
        data_dir=data_dir,
        sample_names=train_samples,
        n_genes=config.get("n_genes", 250),
        patch_size=config.get("patch_size", 224),
        augment=True,
        context_dir=context_dir,
        max_spots=train_max_spots,
    )
    val_ds = HisToGeneDataset(
        data_dir=data_dir,
        sample_names=val_samples,
        n_genes=config.get("n_genes", 250),
        patch_size=config.get("patch_size", 224),
        augment=False,
        context_dir=context_dir,
        max_spots=val_max_spots,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        collate_fn=whole_slide_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        collate_fn=whole_slide_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
