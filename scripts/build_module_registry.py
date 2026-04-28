"""Build the gene-module registry consumed by HydraSTNet.

For each LOPCV fold we cluster the 300 SVG genes into K modules using KMeans
on the gene-by-spot expression matrix (each gene's vector across train spots
of that fold). Output: modules_fold{F}.json with the schema HydraSTNet's
runner expects:

  {
    "fold": F,
    "K": 7,
    "n_genes": 300,
    "gene_names_full": [...300 names in canonical counts_svg column order...],
    "gene_to_module":  {gene_name: module_id, ...},
    "module_to_genes": {"0": [gene names...], "1": [...], ..., "K-1": [...]},
    "module_to_indices": {"0": [col indices in y...], ...},
    "module_sizes":  [m_0, m_1, ..., m_{K-1}],   # sum == 300
    "kmeans": {"seed": 2021, "n_init": 10, "max_iter": 300},
    "train_slides": [...],
    "n_train_spots": int,
    "sha256": "<hash of the registry contents excluding the sha256 field>"
  }

Usage:
  python scripts/build_module_registry.py --fold 0 --K 7
  python scripts/build_module_registry.py --all   # folds 0..3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import build_lopcv_folds  # canonical patient-paired folds


DATA_DIR = ROOT / "data" / "her2st"
COUNTS_DIR = DATA_DIR / "counts_svg"
FEATURES_DIR = DATA_DIR / "features_svg"
OUT_DIR_DEFAULT = ROOT / "results" / "ablation" / "kmeans_y_elbow"


def _load_gene_names() -> list[str]:
    """Canonical gene order = column order of counts_svg/*.npy.

    All features_svg/*.csv files are byte-identical and equal to that order
    (verified at dataset boundary in src/data/dataset.py).
    """
    any_csv = next(FEATURES_DIR.glob("*.csv"))
    return any_csv.read_text().splitlines()


def _load_train_y(train_slides: list[str]) -> np.ndarray:
    """Concatenate (n_spots, 300) log1p expression across train slides."""
    arrays = []
    for s in train_slides:
        path = COUNTS_DIR / f"{s}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        arrays.append(np.load(str(path)).astype(np.float32))
    return np.concatenate(arrays, axis=0)


def _hash_registry(registry: dict) -> str:
    """Stable SHA-256 over registry content with the sha256 field excluded."""
    payload = {k: v for k, v in registry.items() if k != "sha256"}
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def build_registry(fold: int, K: int, seed: int = 2021,
                   out_dir: Path = OUT_DIR_DEFAULT) -> Path:
    bc_dir = DATA_DIR / "barcodes"
    sample_names = sorted([p.stem for p in bc_dir.glob("*.csv")])
    folds = build_lopcv_folds(sample_names, "her2st", n_folds=4)
    train_slides, _val_slides = folds[fold]

    gene_names = _load_gene_names()
    n_genes = len(gene_names)
    if n_genes != 300:
        raise ValueError(f"Expected 300 SVG genes, got {n_genes}")

    Y = _load_train_y(train_slides)  # (n_train_spots, 300)
    if Y.shape[1] != n_genes:
        raise ValueError(
            f"Counts column count {Y.shape[1]} != gene name count {n_genes}"
        )

    # Cluster the 300 genes by their expression profile across train spots.
    # KMeans expects (n_samples, n_features) — here n_samples=300 genes,
    # n_features=n_train_spots.
    X = Y.T  # (300, n_train_spots)
    km = KMeans(
        n_clusters=K, random_state=seed, n_init=10, max_iter=300,
    )
    labels = km.fit_predict(X)
    if len(set(labels.tolist())) != K:
        raise RuntimeError(
            f"KMeans produced {len(set(labels.tolist()))} non-empty clusters "
            f"but K={K} requested. Try a different seed."
        )

    module_to_indices: dict[str, list[int]] = {str(k): [] for k in range(K)}
    module_to_genes: dict[str, list[str]] = {str(k): [] for k in range(K)}
    gene_to_module: dict[str, int] = {}
    for idx, lbl in enumerate(labels.tolist()):
        module_to_indices[str(lbl)].append(idx)
        module_to_genes[str(lbl)].append(gene_names[idx])
        gene_to_module[gene_names[idx]] = lbl

    # Stable: sort indices within each module so iteration is deterministic.
    for k in range(K):
        module_to_indices[str(k)] = sorted(module_to_indices[str(k)])
        module_to_genes[str(k)] = [
            gene_names[i] for i in module_to_indices[str(k)]
        ]

    module_sizes = [len(module_to_indices[str(k)]) for k in range(K)]
    if sum(module_sizes) != n_genes:
        raise RuntimeError(
            f"Module size sum {sum(module_sizes)} != n_genes {n_genes}"
        )

    registry = {
        "fold": fold,
        "K": K,
        "n_genes": n_genes,
        "gene_names_full": gene_names,
        "gene_to_module": gene_to_module,
        "module_to_genes": module_to_genes,
        "module_to_indices": module_to_indices,
        "module_sizes": module_sizes,
        "kmeans": {"seed": seed, "n_init": 10, "max_iter": 300},
        "train_slides": train_slides,
        "n_train_spots": int(Y.shape[0]),
    }
    registry["sha256"] = _hash_registry(registry)

    out = out_dir / f"fold_{fold}" / f"modules_fold{fold}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(registry, f, indent=2)
    print(
        f"Fold {fold}: K={K} sizes={module_sizes} sum={sum(module_sizes)} "
        f"n_train_spots={Y.shape[0]} sha256={registry['sha256'][:12]}..."
    )
    print(f"  → {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all", action="store_true",
                        help="Build registries for folds 0..3.")
    parser.add_argument("--K", type=int, default=7,
                        help="Number of modules.")
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    folds = list(range(4)) if args.all else [args.fold]
    for f in folds:
        build_registry(f, K=args.K, seed=args.seed)


if __name__ == "__main__":
    main()
