"""KMeans elbow analysis on training labels y = log1p expression (SVG-300).

Loads per-fold training labels from data/her2st/counts_svg/, concatenates
across train slides, and fits KMeans for K=1..K_MAX. Reports inertia,
inertia-reduction ratio, Delta inertia, silhouette, Calinski-Harabasz,
Davies-Bouldin, and picks K* via the kneedle algorithm.

Fold 1 is reserved for lambda selection (see project memory), so the
default fold is 0. Use --fold all to sweep 0,2-7.

Outputs:
  results/ablation/kmeans_y_elbow/fold_{F}/kmeans_elbow.csv
  results/ablation/kmeans_y_elbow/fold_{F}/kmeans_elbow.json
  results/ablation/kmeans_y_elbow/fold_{F}/kmeans_elbow.png   (if matplotlib)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "her2st"
SPLITS_DIR = DATA_DIR / "splits" / "splits"
COUNTS_DIR = DATA_DIR / "counts_svg"
SEED = 2021


def load_fold_train_y(fold: int) -> tuple[np.ndarray, list[str]]:
    split_csv = SPLITS_DIR / f"train_{fold}.csv"
    df = pd.read_csv(split_csv)
    samples = df["sample_id"].astype(str).tolist()
    ys = []
    for s in samples:
        path = COUNTS_DIR / f"{s}.npy"
        y = np.load(path).astype(np.float32)  # log1p(SVG-300)
        ys.append(y)
    Y = np.concatenate(ys, axis=0)
    return Y, samples


def kneedle_knee(k_values: np.ndarray, inertia: np.ndarray) -> int:
    """Pick the knee as the K with max perpendicular distance from the line
    connecting (K_min, inertia_min-point) to (K_max, inertia_max-point).

    The inertia curve is monotonically non-increasing; the knee is where
    diminishing returns begin.
    """
    x = k_values.astype(np.float64)
    y = inertia.astype(np.float64)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    # Line from first to last normalized point
    x1, y1 = x_norm[0], y_norm[0]
    x2, y2 = x_norm[-1], y_norm[-1]
    num = np.abs((y2 - y1) * x_norm - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    dist = num / den
    return int(k_values[int(np.argmax(dist))])


def run_one_fold(
    fold: int,
    k_max: int,
    silhouette_sample_size: int,
    out_root: Path,
    seed: int,
) -> dict:
    out_dir = out_root / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    Y, samples = load_fold_train_y(fold)
    N, D = Y.shape
    print(f"[fold {fold}] train spots N={N}, genes D={D}, slides={len(samples)}")

    rng = np.random.default_rng(seed)
    if silhouette_sample_size and N > silhouette_sample_size:
        sil_idx = rng.choice(N, size=silhouette_sample_size, replace=False)
    else:
        sil_idx = np.arange(N)

    rows = []
    inertias = []
    ks = list(range(1, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        labels = km.fit_predict(Y)
        inertia = float(km.inertia_)
        inertias.append(inertia)

        if k >= 2:
            sil = float(silhouette_score(Y[sil_idx], labels[sil_idx]))
            ch = float(calinski_harabasz_score(Y, labels))
            db = float(davies_bouldin_score(Y, labels))
        else:
            sil = float("nan")
            ch = float("nan")
            db = float("nan")

        rows.append({
            "K": k,
            "inertia": inertia,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "n_iter": int(km.n_iter_),
        })
        print(f"  K={k:>2d}  inertia={inertia:,.2f}  "
              f"sil={sil:.4f}  CH={ch:,.1f}  DB={db:.4f}")

    inertias = np.asarray(inertias)
    ks_arr = np.asarray(ks)
    inertia0 = inertias[0]
    reduction = 1.0 - inertias / inertia0
    delta = np.concatenate([[0.0], -np.diff(inertias)])  # positive drops
    rel_drop = delta / (inertia0 + 1e-12)

    for i, r in enumerate(rows):
        r["inertia_reduction_ratio"] = float(reduction[i])
        r["delta_inertia"] = float(delta[i])
        r["rel_drop"] = float(rel_drop[i])

    df = pd.DataFrame(rows)
    csv_path = out_dir / "kmeans_elbow.csv"
    df.to_csv(csv_path, index=False)

    k_knee = kneedle_knee(ks_arr, inertias)
    valid_sil = df.dropna(subset=["silhouette"])
    k_sil = int(valid_sil.loc[valid_sil["silhouette"].idxmax(), "K"])
    k_ch = int(valid_sil.loc[valid_sil["calinski_harabasz"].idxmax(), "K"])
    k_db = int(valid_sil.loc[valid_sil["davies_bouldin"].idxmin(), "K"])

    summary = {
        "fold": fold,
        "n_train_spots": int(N),
        "n_genes": int(D),
        "n_train_slides": len(samples),
        "train_slides": samples,
        "k_range": [1, k_max],
        "seed": int(seed),
        "K_star_elbow_kneedle": int(k_knee),
        "K_star_silhouette_max": k_sil,
        "K_star_calinski_harabasz_max": k_ch,
        "K_star_davies_bouldin_min": k_db,
    }
    json_path = out_dir / "kmeans_elbow.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(ks_arr, inertias, "o-")
        axes[0].axvline(k_knee, color="red", linestyle="--",
                        label=f"knee K*={k_knee}")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("Inertia (WCSS)")
        axes[0].set_title(f"Elbow — fold {fold}")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        sil_arr = df["silhouette"].values
        axes[1].plot(ks_arr, sil_arr, "o-", color="green")
        axes[1].axvline(k_sil, color="red", linestyle="--",
                        label=f"max silhouette K*={k_sil}")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("Silhouette")
        axes[1].set_title(f"Silhouette — fold {fold}")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "kmeans_elbow.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] plotting skipped: {e}")

    print(f"[fold {fold}] K*_elbow={k_knee}  K*_silhouette={k_sil}  "
          f"K*_CH={k_ch}  K*_DB={k_db}")
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")

    return summary


def format_table(df: pd.DataFrame, k_knee: int, k_sil: int) -> str:
    header = (
        f"{'K':>3}  {'inertia':>14}  {'reduction':>9}  {'rel_drop':>9}  "
        f"{'silhouette':>10}  {'CH':>12}  {'DB':>7}  {'note':>6}"
    )
    lines = [header, "-" * len(header)]
    for _, r in df.iterrows():
        k = int(r["K"])
        note = ""
        if k == k_knee:
            note = "*elbow"
        elif k == k_sil:
            note = "*sil"
        ch = "-" if np.isnan(r["calinski_harabasz"]) else f"{r['calinski_harabasz']:,.0f}"
        db = "-" if np.isnan(r["davies_bouldin"]) else f"{r['davies_bouldin']:.3f}"
        sil = "-" if np.isnan(r["silhouette"]) else f"{r['silhouette']:.4f}"
        lines.append(
            f"{k:>3d}  {r['inertia']:>14,.2f}  "
            f"{r['inertia_reduction_ratio']:>9.4f}  "
            f"{r['rel_drop']:>9.4f}  {sil:>10}  {ch:>12}  {db:>7}  {note:>6}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0",
                    help="fold index (0..7) or 'all' for 0,2..7 "
                         "(fold 1 reserved for lambda selection)")
    ap.add_argument("--k_max", type=int, default=30)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--silhouette_sample_size", type=int, default=20000,
                    help="subsample for silhouette to cap O(N^2) cost; "
                         "set 0 to use all spots")
    ap.add_argument("--out_root",
                    default=str(ROOT / "results" / "ablation"
                                / "kmeans_y_elbow"))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.fold.lower() == "all":
        folds = [0, 2, 3, 4, 5, 6, 7]
    else:
        folds = [int(args.fold)]
        if folds[0] == 1:
            print("[warn] fold 1 is reserved for lambda selection — "
                  "per project memory. Proceeding anyway since you asked.")

    summaries = []
    for f in folds:
        s = run_one_fold(
            fold=f,
            k_max=args.k_max,
            silhouette_sample_size=args.silhouette_sample_size,
            out_root=out_root,
            seed=args.seed,
        )
        summaries.append(s)

        df = pd.read_csv(out_root / f"fold_{f}" / "kmeans_elbow.csv")
        print("\n" + format_table(df, s["K_star_elbow_kneedle"],
                                  s["K_star_silhouette_max"]))
        print()

    if len(summaries) > 1:
        combined = {
            "folds": summaries,
            "K_star_elbow_counts": {
                str(k): int(sum(1 for s in summaries
                                if s["K_star_elbow_kneedle"] == k))
                for k in sorted({s["K_star_elbow_kneedle"] for s in summaries})
            },
        }
        with open(out_root / "summary_all_folds.json", "w") as f:
            json.dump(combined, f, indent=2)
        print(f"wrote {out_root / 'summary_all_folds.json'}")


if __name__ == "__main__":
    main()
