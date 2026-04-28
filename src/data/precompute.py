"""Offline preprocessing: global features, context clusters, NMF, C_prior.

Run ONCE before training. All artifacts are saved to data/{dataset}/.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple

# Ciga et al. 2021 SimCLR ResNet18 — self-supervised on multi-organ histology.
# Same encoder family TRIPLEX (Section G.1) uses. Biologically valid for tissue.
CIGA_CKPT_PATH = (
    "/mnt/d/docker_machine/anvuva/TRIPLEX/weights/cigar/tenpercent_resnet18.ckpt"
)


def precompute_global_features(data_dir: str, dataset: str, device: str = "cuda"):
    """Extract ResNet18 512-dim features for all spot patches.

    Uses pretrained ResNet18 (MERGE weights or torchvision pretrained).
    Saves to data/{dataset}/global_features/{sample}.npy shape (N, 512).
    """
    from PIL import Image
    from src.models.ciga_encoder import load_ciga_resnet18

    base = Path(data_dir)
    out_dir = base / "global_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    resnet = load_ciga_resnet18(CIGA_CKPT_PATH, device=device)
    print(f"Loaded Ciga SimCLR ResNet18 from {CIGA_CKPT_PATH}")

    # Discover samples
    bc_dir = base / "barcodes"
    sample_names = sorted([p.stem for p in bc_dir.glob("*.csv")])

    for s_name in sample_names:
        out_path = out_dir / f"{s_name}.npy"
        if out_path.exists():
            print(f"  {s_name}: already exists, skipping")
            continue

        # Load WSI and coordinates
        wsi_path = base / "wsi" / f"{s_name}.jpg"
        pos_path = base / "tissue_positions" / f"{s_name}.csv"

        if not wsi_path.exists():
            print(f"  {s_name}: WSI not found, skipping")
            continue

        wsi = Image.open(str(wsi_path)).convert("RGB")
        w, h = wsi.size

        pos_data = np.genfromtxt(
            str(pos_path), delimiter=",", skip_header=1, dtype=float
        )
        if pos_data.ndim == 1:
            pos_data = pos_data.reshape(1, -1)
        coords = pos_data[:, -2:]

        grid_rows = int(coords[:, 0].max()) + 1
        grid_cols = int(coords[:, 1].max()) + 1
        patch_h = h / grid_rows
        patch_w = w / grid_cols

        N = coords.shape[0]
        features = np.zeros((N, 512), dtype=np.float32)

        # Process in batches
        batch_size = 64
        patches_list = []

        for i in range(N):
            row, col = coords[i]
            cy = int(row * patch_h)
            cx = int(col * patch_w)
            half = 112

            y1 = max(0, cy - half)
            y2 = min(h, cy + half)
            x1 = max(0, cx - half)
            x2 = min(w, cx + half)

            crop = wsi.crop((x1, y1, x2, y2))
            crop_arr = np.array(crop.resize((224, 224)), dtype=np.float32) / 255.0
            patches_list.append(crop_arr)

            if len(patches_list) == batch_size or i == N - 1:
                batch_arr = np.stack(patches_list)
                batch_t = (
                    torch.from_numpy(batch_arr).permute(0, 3, 1, 2).to(device)
                )
                with torch.no_grad():
                    feats = resnet(batch_t).cpu().numpy()

                start = i - len(patches_list) + 1
                features[start : i + 1] = feats
                patches_list = []

        np.save(str(out_path), features)
        print(f"  {s_name}: saved {N} features")

    print("Global feature extraction complete.")


def precompute_context_clusters(
    data_dir: str,
    dataset: str,
    train_samples: List[str],
    split_id: str,
    n_contexts: int = 6,
):
    """Fit KMeans on training global features and compute soft context weights.

    IMPORTANT: Only uses training samples for fitting.
    Saves labels and weights for ALL available samples (train uses fit,
    test uses transform).

    Saves:
      context_labels/{split_id}/{sample}.npy  shape (N,) int
      context_weights/{split_id}/{sample}.npy shape (N, T) float32
      kmeans_centroids/{split_id}.npy         shape (T, 512)
    """
    from sklearn.cluster import KMeans

    base = Path(data_dir)
    gf_dir = base / "global_features"

    # Load training features
    train_feats = []
    for s_name in train_samples:
        path = gf_dir / f"{s_name}.npy"
        if path.exists():
            train_feats.append(np.load(str(path)))
    if not train_feats:
        raise ValueError("No global features found for training samples")

    all_train = np.concatenate(train_feats, axis=0)
    print(f"Fitting KMeans with {all_train.shape[0]} spots, {n_contexts} clusters")

    km = KMeans(n_clusters=n_contexts, random_state=2021, n_init=10)
    km.fit(all_train)
    centroids = km.cluster_centers_  # (T, 512)

    # Save centroids
    cent_dir = base / "kmeans_centroids"
    cent_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(cent_dir / f"{split_id}.npy"), centroids.astype(np.float32))

    # Compute labels and soft weights for all samples
    label_dir = base / "context_labels" / split_id
    weight_dir = base / "context_weights" / split_id
    label_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)

    all_samples = sorted([p.stem for p in gf_dir.glob("*.npy")])
    for s_name in all_samples:
        feats = np.load(str(gf_dir / f"{s_name}.npy"))
        # Hard labels
        labels = km.predict(feats)
        np.save(str(label_dir / f"{s_name}.npy"), labels.astype(np.int32))

        # Soft weights via softmax of negative distances to centroids
        dists = np.linalg.norm(
            feats[:, None, :] - centroids[None, :, :], axis=2
        )  # (N, T)
        neg_dists = -dists
        # Softmax
        exp_d = np.exp(neg_dists - neg_dists.max(axis=1, keepdims=True))
        weights = exp_d / exp_d.sum(axis=1, keepdims=True)
        np.save(str(weight_dir / f"{s_name}.npy"), weights.astype(np.float32))

    print(f"Context clusters saved to {split_id}")


def precompute_nmf_and_prior(
    data_dir: str,
    dataset: str,
    split_id: str,
    train_samples: List[str],
    n_modules: int = 15,
    n_contexts: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit NMF on 2000-gene panel and compute C_prior; training data ONLY.

    Spec v2 pipeline (admin directive 2026-04-22):
      0. Load raw UMI (N, 11870) + gene names + SVG names
      1. Build guaranteed 2000-gene panel (SVG first 300 + top 1700 HVG)
      2. Fit NMF on log1p(CPM) of 2000-panel; hard R²≥0.35 gate inside fit_nmf
      3. Tikhonov pseudo-inverse on SVG subset rows (stable for near-zero cols)
      4. C_prior in 2000-gene latent space (full-panel projection)

    Saves:
      nmf/{split_id}/panel_genes.npy     shape (2000,) str
      nmf/{split_id}/svg_in_panel.npy    shape (300,) int — SVG indices in panel
      nmf/{split_id}/M_full.npy          shape (2000, B)
      nmf/{split_id}/M_pinv_full.npy     shape (B, 2000)
      nmf/{split_id}/M_pinv.npy          shape (B, 300) — Tikhonov SVG projection
      nmf/{split_id}/C_prior.npy         shape (T, B, B)
      nmf/{split_id}/r2.txt              float
    """
    from mcspr.prior import (
        fit_nmf,
        compute_context_priors,
        build_nmf_panel,
        compute_svg_projection_matrix,
        load_gene_names,
    )

    base = Path(data_dir)

    # ── 0. Load gene name lists ───────────────────────────────────────────
    umi_gene_names, svg_gene_names = load_gene_names(base)

    # ── Load raw UMI + context weights for training fold ──────────────────
    umi_parts: List[np.ndarray] = []
    W_parts: List[np.ndarray] = []
    for s_name in train_samples:
        umi_path = base / "umi_counts" / f"{s_name}.npy"
        assert umi_path.exists(), f"Missing UMI counts: {umi_path}"
        umi_parts.append(np.load(str(umi_path)))

        w_path = base / "context_weights" / split_id / f"{s_name}.npy"
        if w_path.exists():
            W_parts.append(np.load(str(w_path)))

    raw_umi_train = np.concatenate(umi_parts, axis=0)  # (N, 11870) float32

    if W_parts:
        W_train = np.concatenate(W_parts, axis=0)
    else:
        W_train = (
            np.ones((raw_umi_train.shape[0], n_contexts)) / n_contexts
        )

    nmf_dir = base / "nmf" / split_id
    nmf_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Guaranteed 2000-gene panel (SVG ∪ top-1700 HVG) ────────────────
    panel_indices, svg_in_panel, panel_genes = build_nmf_panel(
        svg_gene_names, umi_gene_names, raw_umi_train,
        n_hvg=1700, seed=2021,
    )
    np.save(str(nmf_dir / "panel_genes.npy"), panel_genes)
    np.save(str(nmf_dir / "svg_in_panel.npy"), svg_in_panel)

    # ── 2. Log-normalize 2000-gene panel, fit NMF ─────────────────────────
    Y_2000 = raw_umi_train[:, panel_indices].astype(np.float32)
    row_sums = Y_2000.sum(axis=1, keepdims=True) + 1e-8
    Y_norm = np.log1p(Y_2000 / row_sums)  # (N, 2000) log-CPM-ish
    Y_nn = np.clip(Y_norm, 0, None)

    print(
        f"NMF fitting: {Y_nn.shape[0]} spots, "
        f"{n_modules} components, 2000-gene panel"
    )
    M_full, M_pinv_full, r2 = fit_nmf(Y_nn, n_components=n_modules)
    print(f"  R² = {r2:.4f} (PASS, >= 0.35)")
    np.save(str(nmf_dir / "M_full.npy"), M_full.astype(np.float32))
    np.save(
        str(nmf_dir / "M_pinv_full.npy"), M_pinv_full.astype(np.float32)
    )

    # ── 3. Tikhonov SVG-subset pseudo-inverse (inference-time projection) ──
    M_pinv_svg = compute_svg_projection_matrix(
        M_full, svg_in_panel, lambda_ridge=1e-3, kappa_max=100.0
    )
    np.save(str(nmf_dir / "M_pinv.npy"), M_pinv_svg)  # (B, 300)

    # ── 4. C_prior from full 2000-gene latent projection ──────────────────
    C_prior = compute_context_priors(
        Y_norm, M_pinv_full, W_train, n_contexts
    )
    print(f"  C_prior computed, {n_contexts} contexts")
    np.save(str(nmf_dir / "C_prior.npy"), C_prior.astype(np.float32))

    with open(nmf_dir / "r2.txt", "w") as _f:
        _f.write(f"{float(r2):.6f}\n")

    # Return signature kept for back-compat (M = M_full, M_pinv = M_pinv_svg)
    return M_full, M_pinv_svg, C_prior


def precompute_all(data_dir: str, dataset: str, config: dict):
    """Run all preprocessing for a dataset.

    1. Global features (all samples, once)
    2. For each LOPCV fold: context clusters + NMF/C_prior (train samples only)
    """
    from src.data.loaders import build_lopcv_folds

    base = Path(data_dir)
    mc = config.get("mcspr", {})
    n_modules = mc.get("n_modules", 15)
    n_contexts = mc.get("n_contexts", 6)

    # Discover samples
    bc_dir = base / "barcodes"
    sample_names = sorted([p.stem for p in bc_dir.glob("*.csv")])
    print(f"Dataset: {dataset}, {len(sample_names)} samples")

    # Step 1: Global features (all samples)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n=== Step 1: Global features ===")
    precompute_global_features(data_dir, dataset, device)

    # Step 2: Per-fold context clusters and NMF
    # Spec v2: n_folds passed through config (HER2ST: 4, pairs of 2 patients)
    n_folds_cfg = config.get("n_folds")
    folds = build_lopcv_folds(sample_names, dataset, n_folds=n_folds_cfg)
    r2_table = []

    for fold_idx, (train_samples, test_samples) in enumerate(folds):
        split_id = f"fold_{fold_idx}"
        print(f"\n=== Fold {fold_idx}: {len(train_samples)} train, "
              f"{len(test_samples)} test ===")

        precompute_context_clusters(
            data_dir, dataset, train_samples, split_id, n_contexts
        )
        _, _, _ = precompute_nmf_and_prior(
            data_dir, dataset, split_id, train_samples, n_modules, n_contexts
        )

        # Fold-local per-gene variance on counts_svg (300-gene panel) for
        # NormalizedMSELoss. Same train_samples as NMF — zero test leakage.
        Y_train_svg_parts = []
        for s_name in train_samples:
            svg_path = base / "counts_svg" / f"{s_name}.npy"
            if svg_path.exists():
                Y_train_svg_parts.append(
                    np.load(str(svg_path)).astype(np.float32)
                )
        if Y_train_svg_parts:
            Y_train_svg = np.concatenate(Y_train_svg_parts, axis=0)
            gene_var = Y_train_svg.var(axis=0) + 1e-8
            nmf_dir = base / "nmf" / split_id
            nmf_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                str(nmf_dir / "gene_var.npy"),
                gene_var.astype(np.float32),
            )
            print(
                f"  gene_var.npy  shape={gene_var.shape}  "
                f"min={gene_var.min():.4e}  max={gene_var.max():.4e}"
            )

    print("\n=== Precomputation complete ===")


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="MCSPR Phase 1 precompute: global features, KMeans, "
        "NMF, C_prior, gene_var. Spec v2 (MCSPR_FINAL_LOCKED_V2.md File 2)."
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Defaults to data/<dataset>/",
    )
    parser.add_argument("--n_modules", required=True, type=int)
    parser.add_argument("--n_folds", type=int, default=4)
    parser.add_argument("--n_contexts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    data_dir = args.data_dir or f"data/{args.dataset}"
    config = {
        "mcspr": {
            "n_modules": args.n_modules,
            "n_contexts": args.n_contexts,
        },
        "seed": args.seed,
        "n_folds": args.n_folds,
    }

    print(
        f"[precompute CLI] dataset={args.dataset} data_dir={data_dir} "
        f"n_modules={args.n_modules} n_folds={args.n_folds} "
        f"n_contexts={args.n_contexts} seed={args.seed}"
    )
    precompute_all(data_dir, args.dataset, config)
