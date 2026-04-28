"""Combined elbow plot across folds 0-3 for the KMeans-on-y ablation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "results" / "ablation" / "kmeans_y_elbow"
FOLDS = [0, 1, 2, 3]
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def main():
    dfs = {}
    summaries = {}
    for f in FOLDS:
        dfs[f] = pd.read_csv(RES_DIR / f"fold_{f}" / "kmeans_elbow.csv")
        with open(RES_DIR / f"fold_{f}" / "kmeans_elbow.json") as fh:
            summaries[f] = json.load(fh)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Raw inertia
    ax = axes[0, 0]
    for f, c in zip(FOLDS, COLORS):
        df = dfs[f]
        ax.plot(df["K"], df["inertia"], "o-", color=c, ms=4,
                label=f"fold {f}  (K*={summaries[f]['K_star_elbow_kneedle']})")
        k_star = summaries[f]["K_star_elbow_kneedle"]
        y_star = df.loc[df["K"] == k_star, "inertia"].values[0]
        ax.scatter([k_star], [y_star], color=c, s=110, marker="*",
                   edgecolor="black", zorder=5)
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("(a) Elbow — inertia vs K")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) Normalized inertia-reduction ratio (0..1)
    ax = axes[0, 1]
    for f, c in zip(FOLDS, COLORS):
        df = dfs[f]
        ax.plot(df["K"], df["inertia_reduction_ratio"], "o-", color=c, ms=4,
                label=f"fold {f}")
        k_star = summaries[f]["K_star_elbow_kneedle"]
        y_star = df.loc[df["K"] == k_star, "inertia_reduction_ratio"].values[0]
        ax.scatter([k_star], [y_star], color=c, s=110, marker="*",
                   edgecolor="black", zorder=5)
    ax.set_xlabel("K")
    ax.set_ylabel("1 - inertia / inertia(K=1)")
    ax.set_title("(b) Normalized inertia reduction")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) Silhouette
    ax = axes[1, 0]
    for f, c in zip(FOLDS, COLORS):
        df = dfs[f]
        ax.plot(df["K"], df["silhouette"], "o-", color=c, ms=4,
                label=f"fold {f}  (argmax={summaries[f]['K_star_silhouette_max']})")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette")
    ax.set_title("(c) Silhouette vs K")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (d) Relative drop per step (the "diminishing returns" view)
    ax = axes[1, 1]
    for f, c in zip(FOLDS, COLORS):
        df = dfs[f]
        ax.plot(df["K"], df["rel_drop"], "o-", color=c, ms=4,
                label=f"fold {f}")
        k_star = summaries[f]["K_star_elbow_kneedle"]
        ax.axvline(k_star, color=c, linestyle=":", alpha=0.5)
    ax.set_xlabel("K")
    ax.set_ylabel("Relative drop = -Δinertia / inertia(K=1)")
    ax.set_title("(d) Marginal inertia drop")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")

    fig.suptitle(
        "K-Means on training labels y = log1p(SVG-300) — HER2ST folds 0-3",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out_png = RES_DIR / "kmeans_elbow_folds_0_1_2_3.png"
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")

    summary = {
        "folds": FOLDS,
        "per_fold": {
            str(f): {
                "K_star_elbow_kneedle": summaries[f]["K_star_elbow_kneedle"],
                "K_star_silhouette_max": summaries[f]["K_star_silhouette_max"],
                "K_star_calinski_harabasz_max":
                    summaries[f]["K_star_calinski_harabasz_max"],
                "K_star_davies_bouldin_min":
                    summaries[f]["K_star_davies_bouldin_min"],
                "n_train_spots": summaries[f]["n_train_spots"],
            }
            for f in FOLDS
        },
        "K_star_elbow_median": int(np.median([
            summaries[f]["K_star_elbow_kneedle"] for f in FOLDS
        ])),
    }
    with open(RES_DIR / "summary_folds_0_1_2_3.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
