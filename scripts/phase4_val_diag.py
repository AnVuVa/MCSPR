"""
Phase 4 validation diagnostic (read-only on saved ckpts).

Re-evaluates the λ=0, 0.01, 0.05, 0.1, 0.5 checkpoints from Phase 4 on the
TRUE fold-1 held-out val patients (C,D), not the internal 20% split used
during selection. Determines whether the Q1 MSE drift observed at λ=0.05/0.1
is a training artifact (internal val) or a generalization failure (real val).

Run on GPU 1; the expanded 9-value sweep is live on GPU 0.
"""
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import STDataset
from src.data.loaders import build_lopcv_folds, slide_collate_fn, SlideBatchSampler
from src.models.triplex import TRIPLEX
from src.training.universal_trainer import _evaluate
from torch.utils.data import DataLoader


CONFIG = "configs/her2st.yaml"
FOLD = 1
RESULTS_DIR = Path("results/lambda_selection/her2st")
LAMBDAS = [0.0, 0.01, 0.05, 0.1, 0.5]


def main():
    with open(CONFIG) as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    data_dir = config["data_dir"]
    n_contexts = config.get("mcspr", {}).get("n_contexts", 6)
    n_genes = config.get("n_genes", 300)
    batch_size = config.get("training", {}).get("batch_size", 256)

    sample_names = sorted([p.stem for p in (Path(data_dir) / "barcodes").glob("*.csv")])
    folds = build_lopcv_folds(sample_names, dataset, n_folds=config.get("n_folds"))
    _, fold_val = folds[FOLD]
    print(f"Fold {FOLD} held-out val patients: {sorted(set(s[0] for s in fold_val))}")
    print(f"Fold {FOLD} held-out val slides  : {fold_val}")

    ctx_dir = Path(data_dir) / "context_weights" / f"fold_{FOLD}"
    gf_dir = Path(data_dir) / "global_features"

    val_ds = STDataset(
        data_dir=data_dir,
        dataset=dataset,
        sample_names=fold_val,
        n_genes=n_genes,
        augment=False,
        n_contexts=n_contexts,
        context_dir=str(ctx_dir) if ctx_dir.exists() else None,
        global_feat_dir=str(gf_dir) if gf_dir.exists() else None,
    )
    sampler = SlideBatchSampler(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(
        val_ds,
        batch_sampler=sampler,
        collate_fn=slide_collate_fn,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  (CUDA_VISIBLE_DEVICES restricts this process to one GPU)")

    rows = []
    for lam in LAMBDAS:
        ckpt_path = RESULTS_DIR / f"lambda_{lam}" / f"fold_{FOLD}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  λ={lam}: MISSING {ckpt_path}  (skip)")
            continue
        print(f"  λ={lam}: loading {ckpt_path.name}...")
        model = TRIPLEX(config, n_genes=n_genes)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model = model.to(device)
        metrics = _evaluate(model, "triplex", val_loader, device)
        row = {
            "lambda": lam,
            "pcc_m": float(metrics["pcc_m"]),
            "q1_mse": float(metrics.get("q1_mse", 0.0)),
            "mse": float(metrics.get("mse", 0.0)),
            "rvd": float(metrics.get("rvd", 0.0)),
        }
        rows.append(row)
        print(f"    → PCC(M)={row['pcc_m']:.4f}  Q1_MSE={row['q1_mse']:.4f}  "
              f"MSE={row['mse']:.4f}  RVD={row['rvd']:.1f}")
        del model, ckpt
        torch.cuda.empty_cache()

    # Print clean table
    if rows:
        baseline_q1 = next((r["q1_mse"] for r in rows if r["lambda"] == 0.0), None)
        baseline_pcc = next((r["pcc_m"] for r in rows if r["lambda"] == 0.0), None)
        print("\n" + "=" * 80)
        print(f"{'λ':>6} {'PCC(M)':>10} {'ΔPCC':>10} {'Q1_MSE':>10} "
              f"{'ΔQ1%':>10} {'RVD':>10}")
        print("-" * 80)
        for r in rows:
            dpcc = r["pcc_m"] - baseline_pcc if baseline_pcc is not None else float("nan")
            dq1 = ((r["q1_mse"] - baseline_q1) / baseline_q1 * 100.0
                   if baseline_q1 else float("nan"))
            print(f"{r['lambda']:>6} {r['pcc_m']:>10.4f} {dpcc:>+10.4f} "
                  f"{r['q1_mse']:>10.4f} {dq1:>+10.2f} {r['rvd']:>10.1f}")
        print("=" * 80)

    out_path = RESULTS_DIR / "fold1_val_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
