import re
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, BatchSampler, RandomSampler, SequentialSampler
from typing import Dict, List, Tuple, Iterator

from src.data.dataset import STDataset


class SlideBatchSampler(Sampler):
    """Yields batches of flat indices where every batch is drawn from a single
    slide. TRIPLEX's global encoder attends over ALL spots of one slide, so
    mixing slides in a batch both breaks semantics and blows VRAM.

    For each slide, partitions its spot indices into chunks of `batch_size`.
    `drop_last` drops partial chunks; otherwise keeps them. Shuffles within
    each slide and across the list of chunks so batches are randomized while
    remaining slide-pure.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True,
                 drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        slide_to_flat: Dict[int, List[int]] = {}
        for flat_i, (s_idx, _spot_idx) in enumerate(dataset.items):
            slide_to_flat.setdefault(s_idx, []).append(flat_i)
        self.slide_to_flat = slide_to_flat

    def __iter__(self) -> Iterator[List[int]]:
        chunks: List[List[int]] = []
        for s_idx, flat_indices in self.slide_to_flat.items():
            idxs = list(flat_indices)
            if self.shuffle:
                random.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                chunk = idxs[i:i + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                chunks.append(chunk)
        if self.shuffle:
            random.shuffle(chunks)
        for c in chunks:
            yield c

    def __len__(self) -> int:
        total = 0
        for flat_indices in self.slide_to_flat.values():
            n = len(flat_indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


def get_patient_id(sample_name: str, dataset: str) -> str:
    """Extract patient ID from sample name.

    her2st: 'her2st_A1' -> 'A'
    stnet:  'stnet_BC23270_1' -> 'BC23270'
    scc:    'scc_P1_1' -> 'P1'
    """
    if dataset == "her2st":
        # MERGE format uses bare names: 'A1', 'B2', ... Patient = leading letter.
        # Legacy format: 'her2st_A1' -> 'A'.
        match = re.match(r"(?:her2st_)?([A-Za-z]+)\d*", sample_name)
        if match:
            return match.group(1).upper()
        raise ValueError(f"Cannot parse patient from her2st sample: {sample_name}")

    elif dataset == "stnet":
        # Extract patient ID: 'stnet_BC23270_1' -> 'BC23270'
        parts = sample_name.split("_")
        if len(parts) >= 2:
            return parts[1]
        raise ValueError(f"Cannot parse patient from stnet sample: {sample_name}")

    elif dataset == "scc":
        # Extract patient prefix: 'scc_P1_1' -> 'P1'
        parts = sample_name.split("_")
        if len(parts) >= 2:
            return parts[1]
        raise ValueError(f"Cannot parse patient from scc sample: {sample_name}")

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def build_lopcv_folds(
    sample_names: List[str],
    dataset: str,
    n_folds: int = None,
) -> List[Tuple[List[str], List[str]]]:
    """Build patient-stratified CV folds.

    Spec v2 (MCSPR_FINAL_LOCKED_V2.md line 63): HER2ST uses 4-fold LOPCV
    with 2 patients per fold (alphabetical pairing):
        Fold 0: A, B   (test, reported)
        Fold 1: C, D   (lambda-selection fold — never reported as test)
        Fold 2: E, F   (test, reported)
        Fold 3: G, H   (test, reported)

    Args:
      sample_names: list of sample identifiers (e.g., 'A1', 'B2', ...)
      dataset:      'her2st' | 'stnet' | 'scc'
      n_folds:      explicit fold count. For her2st with n_folds=4, uses
                    alphabetical 2-patient pairing. For stnet default=8.
                    When None and dataset==her2st, preserves legacy LOPCV
                    (8 one-patient folds).
    """
    # Group samples by patient
    patient_to_samples: Dict[str, List[str]] = {}
    for s in sample_names:
        pid = get_patient_id(s, dataset)
        patient_to_samples.setdefault(pid, []).append(s)

    patients = sorted(patient_to_samples.keys())

    if dataset == "stnet":
        # 8-fold patient-stratified CV: group patients into 8 folds
        k = n_folds if n_folds is not None else 8
        patient_folds: List[List[str]] = [[] for _ in range(k)]
        for i, pid in enumerate(patients):
            patient_folds[i % k].append(pid)

        folds = []
        for fold_idx in range(k):
            test_patients = set(patient_folds[fold_idx])
            test_samples = []
            train_samples = []
            for pid, samples in patient_to_samples.items():
                if pid in test_patients:
                    test_samples.extend(samples)
                else:
                    train_samples.extend(samples)
            folds.append((sorted(train_samples), sorted(test_samples)))
    elif dataset == "her2st" and n_folds is not None:
        # Spec v2: group patients into n_folds contiguous-alphabetical groups
        # (8 patients / 4 folds = 2 patients per fold → A,B | C,D | E,F | G,H)
        n_patients = len(patients)
        if n_patients % n_folds != 0:
            raise ValueError(
                f"her2st has {n_patients} patients which does not divide "
                f"evenly into {n_folds} folds."
            )
        per_fold = n_patients // n_folds
        patient_folds = [
            patients[i * per_fold : (i + 1) * per_fold]
            for i in range(n_folds)
        ]
        folds = []
        for fold_idx in range(n_folds):
            test_patients = set(patient_folds[fold_idx])
            test_samples = []
            train_samples = []
            for pid, samples in patient_to_samples.items():
                if pid in test_patients:
                    test_samples.extend(samples)
                else:
                    train_samples.extend(samples)
            folds.append((sorted(train_samples), sorted(test_samples)))
    else:
        # LOPCV: one fold per patient
        folds = []
        for pid in patients:
            test_samples = patient_to_samples[pid]
            train_samples = []
            for other_pid, samples in patient_to_samples.items():
                if other_pid != pid:
                    train_samples.extend(samples)
            folds.append((sorted(train_samples), sorted(test_samples)))

    return folds


def slide_collate_fn(batch):
    """Custom collate that groups items by sample_idx.

    The global encoder needs ALL spots from a slide simultaneously.
    When a batch spans multiple slides, we concatenate each slide's
    global_features / spot_coords along axis 0 and offset the per-item
    target_spot_idx so indexing z_gl_all[target_spot_idx] remains valid.
    """
    by_sample: Dict[int, List] = {}
    for item in batch:
        sid = item["sample_idx"]
        by_sample.setdefault(sid, []).append(item)

    all_target_imgs = []
    all_neighbor_imgs = []
    all_expressions = []
    all_context_weights = []
    all_spot_indices = []
    all_sample_indices = []
    all_global_features = []
    all_spot_coords = []

    sample_keys = sorted(by_sample.keys())
    offset = 0
    for sid in sample_keys:
        items = by_sample[sid]
        first_item = items[0]
        gf = first_item["global_features"]  # (N_s, 512)
        sc = first_item["spot_coords"]      # (N_s, 2)
        n_s = gf.shape[0]
        all_global_features.append(gf)
        all_spot_coords.append(sc)

        for item in items:
            all_target_imgs.append(item["target_img"])
            all_neighbor_imgs.append(item["neighbor_imgs"])
            all_expressions.append(item["expression"])
            all_context_weights.append(item["context_weights"])
            all_spot_indices.append(item["spot_idx"] + offset)
            all_sample_indices.append(item["sample_idx"])

        offset += n_s

    collated = {
        "target_img": torch.stack(all_target_imgs),
        "neighbor_imgs": torch.stack(all_neighbor_imgs),
        "global_features": torch.cat(all_global_features, dim=0),
        "spot_coords": torch.cat(all_spot_coords, dim=0),
        "expression": torch.stack(all_expressions),
        "context_weights": torch.stack(all_context_weights),
        "target_spot_idx": torch.tensor(all_spot_indices, dtype=torch.long),
        "sample_idx": torch.tensor(all_sample_indices, dtype=torch.long),
    }
    return collated


def build_loaders(
    data_dir: str,
    dataset: str,
    fold_idx: int,
    config: dict,
    sample_names: List[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders for a given fold.

    Train: augment=True, custom collate for slide-level batching.
    Val: augment=False, batch_size=1 (one sample at a time).
    """
    if sample_names is None:
        # Discover sample names from barcodes directory
        from pathlib import Path

        bc_dir = Path(data_dir) / "barcodes"
        sample_names = sorted(
            [p.stem for p in bc_dir.glob("*.csv")]
        )

    folds = build_lopcv_folds(sample_names, dataset, n_folds=config.get("n_folds"))
    train_samples, val_samples = folds[fold_idx]

    tc = config.get("training", {})
    mc = config.get("mcspr", {})
    n_contexts = mc.get("n_contexts", 6)

    # Resolve context directory for this fold
    from pathlib import Path

    base = Path(data_dir)
    ctx_dir = base / "context_weights" / f"fold_{fold_idx}"
    if not ctx_dir.exists():
        ctx_dir = None

    gf_dir = base / "global_features"
    if not gf_dir.exists():
        gf_dir = None

    train_ds = STDataset(
        data_dir=data_dir,
        dataset=dataset,
        sample_names=train_samples,
        n_genes=config.get("n_genes", 250),
        augment=True,
        n_contexts=n_contexts,
        context_dir=str(ctx_dir) if ctx_dir else None,
        global_feat_dir=str(gf_dir) if gf_dir else None,
    )

    val_ds = STDataset(
        data_dir=data_dir,
        dataset=dataset,
        sample_names=val_samples,
        n_genes=config.get("n_genes", 250),
        augment=False,
        n_contexts=n_contexts,
        context_dir=str(ctx_dir) if ctx_dir else None,
        global_feat_dir=str(gf_dir) if gf_dir else None,
    )

    pin_memory = bool(tc.get("pin_memory", False))
    train_batch_sampler = SlideBatchSampler(
        train_ds,
        batch_size=tc.get("batch_size", 128),
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=slide_collate_fn,
        num_workers=tc.get("num_workers", 4),
        pin_memory=pin_memory,
    )

    val_batch_sampler = SlideBatchSampler(
        val_ds,
        batch_size=tc.get("batch_size", 128),
        shuffle=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_batch_sampler,
        collate_fn=slide_collate_fn,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def build_triplex_hydra_loaders(
    data_dir: str,
    dataset: str,
    fold_idx: int,
    config: dict,
    sample_names: List[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Dedicated loader builder for HydraTRIPLEX.

    Functionally identical to build_loaders today (same target/neighbor/global
    features and 300-gene labels) but isolated for explicit per-baseline scope:
    if HydraTRIPLEX ever needs custom collation (e.g. precomputed module masks)
    those changes land here, not in build_loaders.
    """
    return build_loaders(
        data_dir, dataset, fold_idx, config, sample_names,
    )


def build_stnet_hydra_loaders(
    data_dir: str,
    dataset: str,
    fold_idx: int,
    config: dict,
    sample_names: List[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Dedicated loader builder for HydraSTNet.

    Functionally identical to build_stnet_loaders today (same per-spot patches,
    same labels) but kept as a separate symbol for explicit isolation between
    baselines (per project convention: each model owns its own model file,
    runner, and loader builder so future divergence is local).
    """
    return build_stnet_loaders(
        data_dir, dataset, fold_idx, config, sample_names,
    )


def build_stnet_loaders(
    data_dir: str,
    dataset: str,
    fold_idx: int,
    config: dict,
    sample_names: List[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Fast STNet-only loader path.

    STNet's forward reads only `target_img`. This builder skips the 25×
    per-spot neighbor PIL crops, global_features, and spot_coords tensors
    that the shared path in build_loaders produces for TRIPLEX. All 36
    HER2ST slides have a wsi224/*.npy mmap cache, so target extraction is
    O(memmap read), not O(PIL.Image.crop).

    Train uses a random cross-slide sampler (appropriate for a per-patch
    model; matches ST-Net paper). Val keeps SlideBatchSampler so each
    validation batch comes from a single slide, preserving the current
    per-slide PCC semantics in universal_trainer._evaluate.
    """
    if sample_names is None:
        from pathlib import Path

        bc_dir = Path(data_dir) / "barcodes"
        sample_names = sorted([p.stem for p in bc_dir.glob("*.csv")])

    n_folds = config.get("n_folds")
    folds = build_lopcv_folds(sample_names, dataset, n_folds=n_folds)
    train_samples, val_samples = folds[fold_idx]

    tc = config.get("training", {})
    mc = config.get("mcspr", {})
    n_contexts = mc.get("n_contexts", 6)

    from pathlib import Path

    base = Path(data_dir)
    ctx_dir = base / "context_weights" / f"fold_{fold_idx}"
    if not ctx_dir.exists():
        ctx_dir = None

    gf_dir = base / "global_features"
    if not gf_dir.exists():
        gf_dir = None

    # Phase 2 fair-protocol hook: training-target source (e.g. counts_svg_spcs)
    # may differ from the eval-target source (counts_svg) — read both from
    # config.training and pass through to STDataset's counts_subdir override.
    train_counts_subdir = tc.get("train_counts_subdir")
    val_counts_subdir = tc.get("val_counts_subdir")

    train_ds = STDataset(
        data_dir=data_dir,
        dataset=dataset,
        sample_names=train_samples,
        n_genes=config.get("n_genes", 250),
        augment=True,
        n_contexts=n_contexts,
        context_dir=str(ctx_dir) if ctx_dir else None,
        global_feat_dir=str(gf_dir) if gf_dir else None,
        skip_unused=True,
        counts_subdir=train_counts_subdir,
    )

    val_ds = STDataset(
        data_dir=data_dir,
        dataset=dataset,
        sample_names=val_samples,
        n_genes=config.get("n_genes", 250),
        augment=False,
        n_contexts=n_contexts,
        context_dir=str(ctx_dir) if ctx_dir else None,
        global_feat_dir=str(gf_dir) if gf_dir else None,
        counts_subdir=val_counts_subdir,
        skip_unused=True,
    )

    pin_memory = bool(tc.get("pin_memory", False))
    # DenseNet121 activations balloon at bs=256 on 16GB (near-OOM allocator
    # thrashing → throughput collapse). ST-Net paper uses bs≈64.
    batch_size = tc.get("stnet_batch_size", 64)
    num_workers = tc.get("stnet_num_workers", 2)

    # Train: cross-slide random batches (standard for per-patch model).
    # Multi-worker is safe — each worker reads wsi224/*.npy via mmap.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    # Val: one-or-more batches per slide, preserves per-slide PCC semantics.
    val_batch_sampler = SlideBatchSampler(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_batch_sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
