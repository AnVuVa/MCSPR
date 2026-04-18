"""
Offline precompute -- zero data leakage.
All computation uses training-fold data only.
Test slides are never loaded or touched.

Output structure:
  data/her2st/
    global_features/          {sample}.npy  (N, 512) ResNet18 patch features
    context_weights/fold_{i}/ {sample}.npy  (N, 6)   soft KMeans context weights
    context_labels/fold_{i}/  {sample}.npy  (N,)     hard context assignments
    nmf/fold_{i}/
      M.npy           (m, B)  NMF loading matrix
      M_pinv.npy      (B, m)  pseudoinverse for module projection
      C_prior.npy     (T, B, B) per-context correlation prior
      r2.txt          reconstruction R^2
      stats.json      per-context diagnostics
"""

import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.special import softmax
from sklearn.preprocessing import normalize

# -- mcspr package --
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcspr.prior.construction import fit_nmf, compute_context_priors
from src.models.ciga_encoder import load_ciga_resnet18

# -- Config --
DATA_DIR    = Path("data/her2st")
RAW_DIR     = DATA_DIR / "umi_counts"
LOG_DIR     = DATA_DIR / "counts_spcs"
WSI_DIR     = DATA_DIR / "wsi"
SVG_PATH    = Path("results/pre_training_gates/gene_selection/svg_genes.json")
T           = 6      # context clusters
B           = 15     # NMF modules
SEED        = 2021
PATCH_SIZE  = 224
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# Ciga et al. 2021 SimCLR ResNet18 — self-supervised on multi-organ histology.
# Same encoder family TRIPLEX (Section G.1) uses. Biologically valid for tissue.
CIGA_CKPT_PATH = (
    "/mnt/d/docker_machine/anvuva/TRIPLEX/weights/cigar/tenpercent_resnet18.ckpt"
)

with open(SVG_PATH) as f:
    svg_genes = json.load(f)
print(f"SVG gene set: {len(svg_genes)} genes")

# -- Patient-LOPCV folds --
feature_dir  = DATA_DIR / "features"
sample_names = sorted([f.stem for f in feature_dir.glob("*.csv")])
patient_map  = defaultdict(list)
for s in sample_names:
    pid = s[0].upper()
    patient_map[pid].append(s)
patients = sorted(patient_map.keys())

folds = []
for held in patients:
    train = [s for p in patients if p != held for s in patient_map[p]]
    test  = patient_map[held]
    folds.append((train, test))
print(f"LOPCV: {len(folds)} folds, {len(patients)} patients")

# -- Step 1: Extract global features (ResNet18, runs ONCE for all samples) --
print("\n-- Step 1: ResNet18 global feature extraction --")

import torchvision.models as tv_models
import torchvision.transforms as T_img
from PIL import Image

feat_dir = DATA_DIR / "global_features"
feat_dir.mkdir(parents=True, exist_ok=True)

# Ciga SimCLR ResNet18 — histology-pretrained, same encoder family as TRIPLEX.
base = load_ciga_resnet18(CIGA_CKPT_PATH, device=DEVICE)
print(f"  Loaded Ciga SimCLR ResNet18 from {CIGA_CKPT_PATH}")
transform = T_img.Compose([
    T_img.Resize((PATCH_SIZE, PATCH_SIZE)),
    T_img.ToTensor(),
    T_img.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

for sample in sample_names:
    out_path = feat_dir / f"{sample}.npy"
    if out_path.exists():
        print(f"  {sample}: cached")
        continue

    wsi_path = WSI_DIR / f"{sample}.jpg"
    if not wsi_path.exists():
        wsi_path = WSI_DIR / f"{sample}.tif"
    coord_path = DATA_DIR / "tissue_positions" / f"{sample}.csv"

    if not wsi_path.exists() or not coord_path.exists():
        print(f"  {sample}: SKIP (missing WSI or coords)")
        continue

    coords = pd.read_csv(coord_path, index_col=0)
    wsi    = Image.open(wsi_path).convert("RGB")
    W_img, H_img = wsi.size
    N      = len(coords)

    rows  = coords["array_row"].values.astype(int)
    cols  = coords["array_col"].values.astype(int)
    n_r   = rows.max() - rows.min() + 1
    n_c   = cols.max() - cols.min() + 1
    px_r  = H_img / (n_r + 1)
    px_c  = W_img / (n_c + 1)
    half  = PATCH_SIZE // 2

    feats = []
    batch_imgs = []
    batch_size = 64

    for i in range(N):
        px = int((cols[i] - cols.min() + 0.5) * px_c)
        py = int((rows[i] - rows.min() + 0.5) * px_r)
        crop = wsi.crop((max(0, px - half), max(0, py - half),
                         min(W_img, px + half), min(H_img, py + half)))
        crop = crop.resize((PATCH_SIZE, PATCH_SIZE))
        batch_imgs.append(transform(crop))

        if len(batch_imgs) == batch_size or i == N - 1:
            with torch.no_grad():
                batch_t = torch.stack(batch_imgs).to(DEVICE)
                f = base(batch_t).cpu().numpy()   # (batch, 512)
            feats.append(f)
            batch_imgs = []

    features = np.concatenate(feats, axis=0)   # (N, 512)
    np.save(out_path, features.astype(np.float32))
    print(f"  {sample}: {N} spots -> {features.shape}")

print("Global features done.")

# -- Steps 2+3: Per-fold context clustering + NMF --
for fold_idx, (train_samples, test_samples) in enumerate(folds):
    held_patient = patients[fold_idx]
    print(f"\n-- Fold {fold_idx} (held: {held_patient}) --")
    print(f"   Train: {len(train_samples)} | Test: {len(test_samples)}")

    ctx_dir = DATA_DIR / "context_weights" / f"fold_{fold_idx}"
    lbl_dir = DATA_DIR / "context_labels"  / f"fold_{fold_idx}"
    nmf_dir = DATA_DIR / "nmf"             / f"fold_{fold_idx}"
    for d in [ctx_dir, lbl_dir, nmf_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # -- 2a. Stack training global features --
    train_feats_list = []
    train_sample_map = []   # (n_spots, sample_name)
    for s in train_samples:
        p = feat_dir / f"{s}.npy"
        if not p.exists():
            print(f"   WARN: no features for {s}, skipping")
            continue
        f = np.load(p)    # (N, 512)
        train_feats_list.append(f)
        train_sample_map.append((f.shape[0], s))

    if not train_feats_list:
        print(f"   ERROR: no training features for fold {fold_idx}")
        continue

    F_train = np.concatenate(train_feats_list, axis=0)   # (N_total, 512)
    F_norm  = normalize(F_train, norm="l2")
    print(f"   Training features: {F_train.shape}")

    # -- 2b. KMeans T=6 on training features --
    np.random.seed(SEED)
    km = KMeans(n_clusters=T, random_state=SEED, n_init=10, max_iter=300)
    km.fit(F_norm)
    centroids = km.cluster_centers_   # (T, 512)

    # -- 2c. Compute soft weights for ALL samples (train + test) --
    # Test samples get context weights too -- needed at evaluation time
    # But KMeans was fitted ONLY on training features (no leakage)
    for s in sample_names:
        feat_p = feat_dir / f"{s}.npy"
        if not feat_p.exists():
            continue
        f      = normalize(np.load(feat_p), norm="l2")   # (N, 512)
        dists  = np.linalg.norm(
            f[:, None, :] - centroids[None, :, :], axis=2
        )   # (N, T)
        # Soft weights via softmax of negative distances (temperature=0.5)
        weights = softmax(-dists / 0.5, axis=1)   # (N, T)
        labels  = dists.argmin(axis=1)             # (N,) hard assignments

        np.save(ctx_dir / f"{s}.npy", weights.astype(np.float32))
        np.save(lbl_dir / f"{s}.npy", labels.astype(np.int32))

    print(f"   Context weights saved for all {len(sample_names)} samples")

    # -- 3. NMF on SVG gene expression (training only) --
    gene_names_path = DATA_DIR / "features_full" / "gene_names.json"
    with open(gene_names_path) as f_:
        all_genes = json.load(f_)
    gene_idx_map = {g: i for i, g in enumerate(all_genes)}
    svg_idx      = np.array([gene_idx_map[g] for g in svg_genes
                             if g in gene_idx_map])
    print(f"   SVG genes found in transcriptome: {len(svg_idx)}/{len(svg_genes)}")

    # Stack log-normalized expression for training slides, SVG genes only
    Y_train_list = []
    W_train_list = []   # context weights for NMF spots
    L_train_list = []   # context labels

    for s in train_samples:
        raw_p = RAW_DIR / f"{s}.npy"
        ctx_p = ctx_dir / f"{s}.npy"
        lbl_p = lbl_dir / f"{s}.npy"
        if not raw_p.exists() or not ctx_p.exists():
            continue
        Y_raw = np.load(raw_p).astype(np.float32)
        lib   = Y_raw.sum(axis=1, keepdims=True)
        Y_log = np.log1p(Y_raw / (lib + 1e-8) * 1e4)
        Y_svg = Y_log[:, svg_idx]   # (N, 300)
        ctx_w = np.load(ctx_p)      # (N, T)
        ctx_l = np.load(lbl_p)      # (N,)
        Y_train_list.append(Y_svg)
        W_train_list.append(ctx_w)
        L_train_list.append(ctx_l)

    Y_train = np.concatenate(Y_train_list, axis=0)   # (N_total, 300)
    W_train = np.concatenate(W_train_list, axis=0)   # (N_total, T)
    L_train = np.concatenate(L_train_list, axis=0)   # (N_total,)
    print(f"   Training expression matrix: {Y_train.shape}")

    # Fit NMF — inline with max_iter=2000 (bypasses fit_nmf's hardcoded 500).
    # Matches mcspr.prior.fit_nmf semantics (init, pinv, r2) exactly.
    np.random.seed(SEED)
    from sklearn.decomposition import NMF as _NMF
    _nmf = _NMF(
        n_components=B,
        init="nndsvd",
        max_iter=2000,
        random_state=SEED,
    )
    Z_nmf = _nmf.fit_transform(Y_train)
    M = _nmf.components_.T
    M_pinv = np.linalg.pinv(M)
    _Y_rec = Z_nmf @ M.T
    _ss_res = np.sum((Y_train - _Y_rec) ** 2)
    _ss_tot = np.sum((Y_train - Y_train.mean(axis=0)) ** 2)
    r2 = float(1.0 - _ss_res / _ss_tot)
    print(f"   NMF R2: {r2:.4f}  ({'OK' if r2 > 0.60 else 'WARN: below 0.60'})")

    # Compute context priors
    C_prior, stats = compute_context_priors(
        Y_train, M_pinv, L_train, W_train, n_contexts=T
    )
    print(f"   C_prior shape: {C_prior.shape}")
    for t in range(T):
        n_t = float(W_train[:, t].sum())
        print(f"   Context {t}: eff_n={n_t:.0f}  "
              f"diag_mean={np.diag(C_prior[t]).mean():.4f}")

    # Save
    np.save(nmf_dir / "M.npy",      M.astype(np.float32))
    np.save(nmf_dir / "M_pinv.npy", M_pinv.astype(np.float32))
    np.save(nmf_dir / "C_prior.npy", C_prior.astype(np.float32))
    with open(nmf_dir / "r2.txt", "w") as f_:
        f_.write(f"{r2:.6f}\n")
    with open(nmf_dir / "stats.json", "w") as f_:
        json.dump({
            "fold": fold_idx, "held_patient": held_patient,
            "n_train_spots": int(Y_train.shape[0]),
            "n_svg_genes": len(svg_idx), "n_modules": B,
            "nmf_r2": float(r2),
            "context_eff_n": [float(W_train[:, t].sum()) for t in range(T)],
        }, f_, indent=2)
    print(f"   Saved to {nmf_dir}")

print("\n=== Precompute complete ===")
print("Next step: run validate_prior.py for Gate 2")
print("Then: run select_lambda.py on Fold 1")
