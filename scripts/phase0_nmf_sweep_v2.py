"""Phase 0 v2 — NMF R² sweep on ACTUAL 4-fold training data (spec v2).

Uses build_lopcv_folds(n_folds=4) patient pairing to get each fold's true
training slide list, then fits NMF across a grid of n_modules. Reports
R² per fold per n_modules. Smallest n achieving R² ≥ 0.60 on ALL 4 folds
is the valid choice.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import NMF

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loaders import build_lopcv_folds, get_patient_id


def load_Y(slides, data_dir):
    parts = []
    for s in slides:
        p = Path(data_dir) / "counts_svg" / f"{s}.npy"
        if p.exists():
            parts.append(np.load(p).astype(np.float32))
        else:
            print(f"  MISSING counts_svg for {s}")
    return np.clip(np.concatenate(parts, axis=0), 0, None)


def fit_r2(Y, n):
    ss_tot = ((Y - Y.mean()) ** 2).sum()
    model = NMF(
        n_components=n, init="nndsvda", random_state=2021,
        max_iter=500, tol=1e-4,
    )
    W = model.fit_transform(Y)
    return 1.0 - ((Y - W @ model.components_) ** 2).sum() / ss_tot


def main():
    data_dir = "data/her2st"
    bc_dir = Path(data_dir) / "barcodes"
    samples = sorted([p.stem for p in bc_dir.glob("*.csv")])
    folds = build_lopcv_folds(samples, "her2st", n_folds=4)

    print("Spec v2 4-fold grouping:")
    for fi, (train, test) in enumerate(folds):
        test_pats = sorted({get_patient_id(s, "her2st") for s in test})
        train_pats = sorted({get_patient_id(s, "her2st") for s in train})
        print(
            f"  Fold {fi}: train_patients={train_pats}  "
            f"test_patients={test_pats}  ({len(train)} train slides)"
        )

    grid = [6, 10, 15, 20, 25, 30, 40, 50]
    all_results = {}
    for fi, (train, _test) in enumerate(folds):
        Y = load_Y(train, data_dir)
        print(f"\n=== Fold {fi} — Y_train shape={Y.shape}  "
              f"min={Y.min():.4f} max={Y.max():.4f} ===")
        per_fold = {}
        for n in grid:
            r2 = float(fit_r2(Y, n))
            flag = "PASS" if r2 >= 0.60 else "fail"
            print(f"  n_modules={n:3d}  R²={r2:.4f}  {flag}", flush=True)
            per_fold[n] = r2
        all_results[f"fold_{fi}"] = per_fold

    # Smallest n_modules achieving R² >= 0.60 on ALL folds
    valid_n = [
        n for n in grid
        if all(all_results[f"fold_{fi}"][n] >= 0.60 for fi in range(len(folds)))
    ]
    print("\n" + "=" * 60)
    if valid_n:
        best = min(valid_n)
        worst_r2 = min(all_results[f"fold_{fi}"][best] for fi in range(len(folds)))
        print(
            f"SMALLEST n_modules passing on ALL 4 folds: n={best} "
            f"(worst-fold R²={worst_r2:.4f})"
        )
    else:
        print("NO n_modules in grid passes ALL 4 folds at R² >= 0.60")
        for n in grid:
            min_r2 = min(all_results[f"fold_{fi}"][n] for fi in range(len(folds)))
            print(f"  n={n}: min-across-folds R²={min_r2:.4f}")

    Path("results/v2").mkdir(parents=True, exist_ok=True)
    with open("results/v2/phase0_nmf_sweep_v2.json", "w") as f:
        json.dump(
            {
                "grid": grid,
                "per_fold_r2": all_results,
                "smallest_passing_all_folds": (
                    valid_n[0] if valid_n else None
                ),
            },
            f,
            indent=2,
        )
    print("Saved: results/v2/phase0_nmf_sweep_v2.json")


if __name__ == "__main__":
    main()
