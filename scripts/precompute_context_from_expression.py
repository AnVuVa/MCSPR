"""Precompute context clusters from expression data (fast path).

Uses KMeans on counts_spcs expression vectors instead of ResNet18 features.
For the context profiling gate, the question is whether context distributions
are sufficiently balanced — expression-based clustering is a valid proxy.
"""

import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data/her2st")
N_CONTEXTS = 6
SEED = 2021


def get_patient_id(name):
    return name[0].upper()


def main():
    bc_dir = DATA_DIR / "barcodes"
    sample_names = sorted([p.stem for p in bc_dir.glob("*.csv")])
    print(f"Dataset: her2st, {len(sample_names)} samples")

    # Build LOPCV folds
    patient_map = defaultdict(list)
    for s in sample_names:
        patient_map[get_patient_id(s)].append(s)
    patients = sorted(patient_map.keys())

    for fold_idx, held in enumerate(patients):
        split_id = f"fold_{fold_idx}"
        train_samples = [s for p in patients if p != held
                         for s in patient_map[p]]

        print(f"\nFold {fold_idx}: {len(train_samples)} train, "
              f"held-out patient {held}")

        # Load training expression data
        train_exprs = []
        for s in train_samples:
            Y = np.load(str(DATA_DIR / "counts_spcs" / f"{s}.npy"))
            train_exprs.append(Y)
        all_train = np.concatenate(train_exprs, axis=0).astype(np.float32)

        # KMeans on expression
        km = KMeans(n_clusters=N_CONTEXTS, random_state=SEED, n_init=10)
        km.fit(all_train)
        centroids = km.cluster_centers_

        # Save centroids
        cent_dir = DATA_DIR / "kmeans_centroids"
        cent_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(cent_dir / f"{split_id}.npy"), centroids)

        # Compute labels and soft weights for ALL samples
        label_dir = DATA_DIR / "context_labels" / split_id
        weight_dir = DATA_DIR / "context_weights" / split_id
        label_dir.mkdir(parents=True, exist_ok=True)
        weight_dir.mkdir(parents=True, exist_ok=True)

        for s in sample_names:
            Y = np.load(str(DATA_DIR / "counts_spcs" / f"{s}.npy")).astype(np.float32)
            labels = km.predict(Y)
            np.save(str(label_dir / f"{s}.npy"), labels.astype(np.int32))

            # Soft weights via softmax of negative distances
            dists = np.linalg.norm(
                Y[:, None, :] - centroids[None, :, :], axis=2
            )
            neg_dists = -dists
            exp_d = np.exp(neg_dists - neg_dists.max(axis=1, keepdims=True))
            weights = exp_d / exp_d.sum(axis=1, keepdims=True)
            np.save(str(weight_dir / f"{s}.npy"), weights.astype(np.float32))

        print(f"  Saved context labels and weights for {len(sample_names)} samples")

    # Also do STNet if available
    stnet_dir = Path("data/stnet")
    if (stnet_dir / "counts_spcs").exists():
        stnet_samples = sorted([p.stem for p in (stnet_dir / "barcodes").glob("*.csv")])
        print(f"\n=== STNet: {len(stnet_samples)} samples ===")

        # Load all expression data for single-fold clustering
        all_exprs = []
        for s in stnet_samples:
            Y = np.load(str(stnet_dir / "counts_spcs" / f"{s}.npy")).astype(np.float32)
            all_exprs.append(Y)
        all_data = np.concatenate(all_exprs, axis=0)
        print(f"Total spots: {all_data.shape[0]}")

        km = KMeans(n_clusters=N_CONTEXTS, random_state=SEED, n_init=10)
        km.fit(all_data)

        weight_dir = stnet_dir / "context_weights" / "fold_0"
        label_dir = stnet_dir / "context_labels" / "fold_0"
        weight_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        centroids = km.cluster_centers_
        for s in stnet_samples:
            Y = np.load(str(stnet_dir / "counts_spcs" / f"{s}.npy")).astype(np.float32)
            labels = km.predict(Y)
            np.save(str(label_dir / f"{s}.npy"), labels.astype(np.int32))

            dists = np.linalg.norm(Y[:, None, :] - centroids[None, :, :], axis=2)
            neg_dists = -dists
            exp_d = np.exp(neg_dists - neg_dists.max(axis=1, keepdims=True))
            weights = exp_d / exp_d.sum(axis=1, keepdims=True)
            np.save(str(weight_dir / f"{s}.npy"), weights.astype(np.float32))

        print(f"  Saved context weights for {len(stnet_samples)} STNet samples")

    print("\nDone.")


if __name__ == "__main__":
    main()
