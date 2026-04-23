"""Phase 0 NMF R² sweep per MCSPR_FINAL_LOCKED_V2.md.

Finds smallest n_modules achieving R² >= 0.60 on fold 0 counts_svg.
CPU-only sklearn NMF. No dependencies on MCSPR package.
"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import NMF


def main():
    fold = 0
    train_csv = Path("data/her2st/splits/splits/train_0.csv")
    if not train_csv.exists():
        # Fallback without inner 'splits/' just in case
        train_csv = Path("data/her2st/splits/train_0.csv")
    train_slides = [
        s.strip()
        for s in open(train_csv).read().splitlines()[1:]
        if s.strip()
    ]
    print(f"train_csv: {train_csv}")
    print(f"train_slides ({len(train_slides)}): {train_slides}")

    Y_parts = []
    for s in train_slides:
        p = Path(f"data/her2st/counts_svg/{s}.npy")
        if p.exists():
            Y_parts.append(np.load(p).astype(np.float32))
        else:
            print(f"  MISSING counts_svg for {s}")
    Y = np.concatenate(Y_parts, axis=0)
    Y_nn = np.clip(Y, 0, None)
    print(
        f"Training matrix: {Y_nn.shape}  "
        f"min={Y_nn.min():.4f} max={Y_nn.max():.4f} "
        f"mean={Y_nn.mean():.4f}"
    )

    results = []
    ss_tot = ((Y_nn - Y_nn.mean()) ** 2).sum()
    for n in [6, 10, 15, 20, 25, 30, 40, 50]:
        model = NMF(
            n_components=n,
            init="nndsvda",
            random_state=2021,
            max_iter=500,
            tol=1e-4,
        )
        W = model.fit_transform(Y_nn)
        ss_res = ((Y_nn - W @ model.components_) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot
        flag = "PASS" if r2 >= 0.60 else "fail"
        print(
            f"n_modules={n:3d}  R²={r2:.4f}  iter={model.n_iter_:3d}  {flag}",
            flush=True,
        )
        results.append((n, float(r2)))

    passing = [(n, r2) for n, r2 in results if r2 >= 0.60]
    print()
    if passing:
        best_n, best_r2 = passing[0]
        print(f"RECOMMENDED: n_modules={best_n}  R²={best_r2:.4f}")
    else:
        print("FAIL: no n_modules in sweep reached R²=0.60")
        print(
            f"Maximum: n={results[-1][0]}  R²={results[-1][1]:.4f}"
        )

    import json
    Path("results/v2").mkdir(parents=True, exist_ok=True)
    with open("results/v2/phase0_nmf_sweep.json", "w") as f:
        json.dump(
            {"sweep": results, "recommended": passing[0] if passing else None},
            f,
            indent=2,
        )
    print("Saved: results/v2/phase0_nmf_sweep.json")


if __name__ == "__main__":
    main()
