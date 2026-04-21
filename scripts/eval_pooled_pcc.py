"""Computes pooled PCC metrics (PCC-10, PCC-50, PCC-200, RVD, Q1_MSE) on
all completed baseline + MCSPR checkpoint pairs.

Pooled convention:    pool all test spots across slides, compute per-gene PCC
                      over the full pool. Matches TRIPLEX paper / lambda-
                      selection eval convention.
Per-slide convention: per-gene PCC within each slide, then average.

Both are reported for comparison.
"""
import sys, os, json, glob
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, '.')


def compute_pcc_pooled(y_hat_all, y_true_all):
    """y_hat_all, y_true_all: (N_spots_total, n_genes)"""
    n_genes = y_hat_all.shape[1]
    pccs = []
    for j in range(n_genes):
        if y_true_all[:, j].std() < 1e-8:
            continue
        r, _ = pearsonr(y_hat_all[:, j], y_true_all[:, j])
        if not np.isnan(r):
            pccs.append(r)
    pccs = np.array(sorted(pccs, reverse=True))
    return {
        'pcc_m':   float(pccs.mean())    if len(pccs) > 0 else 0,
        'pcc_10':  float(pccs[:10].mean())  if len(pccs) >= 10 else 0,
        'pcc_50':  float(pccs[:50].mean())  if len(pccs) >= 50 else 0,
        'pcc_200': float(pccs[:200].mean()) if len(pccs) >= 200 else 0,
        'n_genes': len(pccs),
    }


def compute_per_slide_pcc(per_slide_preds):
    """per_slide_preds: list of (y_hat, y_true) tuples per slide."""
    slide_pccs = []
    for y_hat, y_true in per_slide_preds:
        gene_pccs = []
        for j in range(y_hat.shape[1]):
            if y_true[:, j].std() < 1e-8:
                continue
            r, _ = pearsonr(y_hat[:, j], y_true[:, j])
            if not np.isnan(r):
                gene_pccs.append(r)
        if gene_pccs:
            slide_pccs.append(np.mean(gene_pccs))
    return float(np.mean(slide_pccs)) if slide_pccs else 0


def compute_rvd(y_hat_all, y_true_all):
    """RVD = mean_j [(var_pred_j - var_true_j)^2 / var_true_j^2]"""
    var_pred = y_hat_all.var(axis=0)
    var_true = y_true_all.var(axis=0)
    mask = var_true > 1e-8
    if mask.sum() == 0:
        return float('inf')
    rvd = ((var_pred[mask] - var_true[mask]) ** 2 / var_true[mask] ** 2).mean()
    return float(rvd)


def compute_q1_mse(y_hat_all, y_true_all):
    """Q1 MSE: MSE of bottom 25% genes by per-gene MSE (low-error tail)."""
    mse_per_gene = ((y_hat_all - y_true_all) ** 2).mean(axis=0)
    q25 = np.percentile(mse_per_gene, 25)
    return float(mse_per_gene[mse_per_gene <= q25].mean())


def load_cache(fold_dir):
    """Load test_predictions.npz from fold_dir. Returns (y_hat, y_true,
    per_slide_list) or None if absent. Per-slide grouping uses sample_idx
    (TRIPLEX evaluator convention); falls back to batch_sizes if
    sample_idx is absent (older caches)."""
    cache = os.path.join(fold_dir, 'test_predictions.npz')
    if not os.path.exists(cache):
        return None
    data = np.load(cache)
    y_hat = data['y_hat']
    y_true = data['y_true']
    if 'sample_idx' in data.files:
        sid = data['sample_idx']
        per_slide = []
        for s in np.unique(sid):
            mask = sid == s
            per_slide.append((y_hat[mask], y_true[mask]))
    elif 'batch_sizes' in data.files:
        sizes = data['batch_sizes']
        splits = np.cumsum(sizes)[:-1]
        per_slide = list(zip(np.split(y_hat, splits, axis=0),
                             np.split(y_true, splits, axis=0)))
    elif 'slide_sizes' in data.files:   # legacy
        sizes = data['slide_sizes']
        splits = np.cumsum(sizes)[:-1]
        per_slide = list(zip(np.split(y_hat, splits, axis=0),
                             np.split(y_true, splits, axis=0)))
    else:
        per_slide = [(y_hat, y_true)]
    return y_hat, y_true, per_slide


def main():
    import yaml
    with open('configs/her2st.yaml') as f:
        cfg = yaml.safe_load(f)

    results = {}

    checkpoint_patterns = [
        ('triplex', 'baseline', 'results/triplex/her2st/fold_*/best_model.pt'),
        ('triplex', 'mcspr',    'results/triplex_mcspr/her2st/fold_*/best_model.pt'),
        ('stnet',   'baseline', 'results/stnet/her2st/fold_*/best_model.pt'),
        ('stnet',   'mcspr',    'results/stnet_mcspr/her2st/fold_*/best_model.pt'),
    ]

    for arch, variant, pattern in checkpoint_patterns:
        ckpts = sorted(glob.glob(pattern))
        if not ckpts:
            print(f"  SKIP {arch} {variant}: no checkpoints found (tried {pattern})")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating {arch} {variant}: {len(ckpts)} folds")

        all_y_hat, all_y_true = [], []
        per_slide_across = []
        fold_results = []

        for ckpt_path in ckpts:
            fold_dir = os.path.dirname(ckpt_path)
            fold_id = os.path.basename(fold_dir)

            cache = load_cache(fold_dir)
            if cache is None:
                print(f"  {fold_id}: no prediction cache, skipping")
                print(f"    Run: python scripts/run_inference.py --config configs/her2st.yaml")
                continue

            y_hat, y_true, per_slide = cache
            all_y_hat.append(y_hat)
            all_y_true.append(y_true)
            per_slide_across.extend(per_slide)

            fold_pooled = compute_pcc_pooled(y_hat, y_true)
            fold_per_slide = compute_per_slide_pcc(per_slide)
            fold_rvd = compute_rvd(y_hat, y_true)

            fold_results.append({
                'fold': fold_id,
                'n_slides': len(per_slide),
                'pcc_m_per_slide': fold_per_slide,
                **{f'{k}_pooled': v for k, v in fold_pooled.items()},
                'rvd': fold_rvd,
            })
            print(f"  {fold_id} (slides={len(per_slide)}): per-slide PCC={fold_per_slide:.4f}, "
                  f"pooled PCC-50={fold_pooled['pcc_50']:.4f}, RVD={fold_rvd:.4f}")

        if all_y_hat:
            pooled_hat = np.vstack(all_y_hat)
            pooled_true = np.vstack(all_y_true)
            overall = compute_pcc_pooled(pooled_hat, pooled_true)
            overall_rvd = compute_rvd(pooled_hat, pooled_true)
            overall_q1mse = compute_q1_mse(pooled_hat, pooled_true)
            overall_per_slide = compute_per_slide_pcc(per_slide_across)

            print(f"\n  AGGREGATE ({len(all_y_hat)} folds, {len(per_slide_across)} slides):")
            print(f"    Per-slide PCC(M) = {overall_per_slide:.4f}")
            print(f"    Pooled PCC(M)    = {overall['pcc_m']:.4f}")
            print(f"    Pooled PCC-10    = {overall['pcc_10']:.4f}")
            print(f"    Pooled PCC-50    = {overall['pcc_50']:.4f}")
            print(f"    Pooled PCC-200   = {overall['pcc_200']:.4f}")
            print(f"    RVD              = {overall_rvd:.4f}")
            print(f"    Q1 MSE           = {overall_q1mse:.4f}")

            results[f'{arch}_{variant}'] = {
                'folds': fold_results,
                'aggregate_pooled': overall,
                'aggregate_per_slide_pcc_m': overall_per_slide,
                'aggregate_rvd': overall_rvd,
                'aggregate_q1mse': overall_q1mse,
                'n_folds': len(all_y_hat),
                'n_slides': len(per_slide_across),
            }

    # ── Delta computation ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("DELTA TABLE (MCSPR − baseline, pooled PCC):")
    print(f"{'=' * 60}")
    for arch in ['stnet', 'triplex']:
        base_key = f'{arch}_baseline'
        mcspr_key = f'{arch}_mcspr'
        if base_key in results and mcspr_key in results:
            # Restrict to folds that have BOTH baseline and MCSPR caches for
            # apples-to-apples comparison.
            base_folds = {r['fold']: r for r in results[base_key]['folds']}
            mcspr_folds = {r['fold']: r for r in results[mcspr_key]['folds']}
            shared = sorted(set(base_folds) & set(mcspr_folds))
            if not shared:
                continue

            shared_base_yh, shared_base_yt = [], []
            shared_mcspr_yh, shared_mcspr_yt = [], []
            for f in shared:
                # Re-load caches to build shared-fold pools.
                arch_base_dir = f"results/{arch}/her2st/{f}"
                arch_mc_dir = f"results/{arch}_mcspr/her2st/{f}"
                b = load_cache(arch_base_dir)
                m = load_cache(arch_mc_dir)
                if b and m:
                    shared_base_yh.append(b[0]); shared_base_yt.append(b[1])
                    shared_mcspr_yh.append(m[0]); shared_mcspr_yt.append(m[1])

            if not shared_base_yh:
                continue
            bP = compute_pcc_pooled(np.vstack(shared_base_yh), np.vstack(shared_base_yt))
            mP = compute_pcc_pooled(np.vstack(shared_mcspr_yh), np.vstack(shared_mcspr_yt))
            bR = compute_rvd(np.vstack(shared_base_yh), np.vstack(shared_base_yt))
            mR = compute_rvd(np.vstack(shared_mcspr_yh), np.vstack(shared_mcspr_yt))
            bQ = compute_q1_mse(np.vstack(shared_base_yh), np.vstack(shared_base_yt))
            mQ = compute_q1_mse(np.vstack(shared_mcspr_yh), np.vstack(shared_mcspr_yt))

            d_m = mP['pcc_m'] - bP['pcc_m']
            d_10 = mP['pcc_10'] - bP['pcc_10']
            d_50 = mP['pcc_50'] - bP['pcc_50']
            d_200 = mP['pcc_200'] - bP['pcc_200']
            d_rvd = mR - bR
            d_q1 = mQ - bQ

            print(f"  {arch.upper()}  (shared folds: {len(shared)} — {', '.join(shared)})")
            print(f"    Δ PCC(M)  = {d_m:+.4f}  {'✓' if d_m > 0 else '✗'}")
            print(f"    Δ PCC-10  = {d_10:+.4f}  {'✓' if d_10 > 0 else '✗'}")
            print(f"    Δ PCC-50  = {d_50:+.4f}  {'✓' if d_50 > 0 else '✗'}")
            print(f"    Δ PCC-200 = {d_200:+.4f}  {'✓' if d_200 > 0 else '✗'}")
            print(f"    Δ RVD     = {d_rvd:+.4f}  {'✓ (improves)' if d_rvd < 0 else '✗ (worsens)'}")
            print(f"    Δ Q1_MSE  = {d_q1:+.4f}  {'✓ (improves)' if d_q1 < 0 else '✗ (worsens)'}")

            results[f'{arch}_delta'] = {
                'shared_folds': shared,
                'pcc_m': d_m, 'pcc_10': d_10, 'pcc_50': d_50, 'pcc_200': d_200,
                'rvd': d_rvd, 'q1_mse': d_q1,
                'baseline_pcc_m': bP['pcc_m'],
                'mcspr_pcc_m': mP['pcc_m'],
            }

    os.makedirs('logs', exist_ok=True)
    with open('logs/pooled_pcc_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results: logs/pooled_pcc_results.json")


if __name__ == "__main__":
    main()
