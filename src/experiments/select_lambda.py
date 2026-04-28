"""
Lambda selection for MCSPR -- anti-leakage protocol.

Rules (from locked blueprint):
  1. Use Fold 1 training data ONLY (never test data)
  2. Split Fold 1 training data 80/20 (internal train / internal val)
  3. Sweep lambda in {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0}
  4. Select lambda that maximizes Delta_PCC on internal val set
     WITHOUT triggering representation drift (Q1 MSE > 5% baseline increase)
  5. Save selected lambda to results/lambda_selection/{dataset}/selected_lambda.json
  6. This lambda is FROZEN for all architectures and all folds

Run ONCE per dataset. Never re-run to get a better result.

Usage:
    python src/experiments/select_lambda.py --config configs/her2st.yaml
"""

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import STDataset
from src.data.loaders import build_lopcv_folds, slide_collate_fn
from src.models.triplex import TRIPLEX
from src.training.universal_trainer import train_one_fold, _evaluate
from torch.utils.data import DataLoader


LAMBDA_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
FOLD_FOR_SELECTION = 1        # Fold 1 used for lambda selection (fixed)
INTERNAL_VAL_FRACTION = 0.20  # 80/20 split within Fold 1 training data
DRIFT_THRESHOLD = 0.05        # Max allowed Q1 MSE increase relative to baseline


def set_seed(seed: int = 2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def split_internal(
    train_samples: List[str],
    val_fraction: float = 0.20,
    seed: int = 2021,
) -> Tuple[List[str], List[str]]:
    """Split train samples into internal 80/20."""
    rng = np.random.RandomState(seed)
    n_val = max(1, int(len(train_samples) * val_fraction))
    idx = rng.permutation(len(train_samples))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return ([train_samples[i] for i in train_idx],
            [train_samples[i] for i in val_idx])


def build_internal_loaders(
    data_dir: str,
    dataset: str,
    train_samples: List[str],
    val_samples: List[str],
    config: dict,
    fold_idx: int,
) -> Tuple[DataLoader, DataLoader]:
    """Build DataLoaders from explicit train/val sample lists."""
    tc = config.get("training", {})
    mc = config.get("mcspr", {})
    n_contexts = mc.get("n_contexts", 6)

    base = Path(data_dir)
    ctx_dir = base / "context_weights" / f"fold_{fold_idx}"
    if not ctx_dir.exists():
        ctx_dir = None

    gf_dir = base / "global_features"
    if not gf_dir.exists():
        gf_dir = None

    train_ds = STDataset(
        data_dir=data_dir,
        dataset=dataset,
        sample_names=train_samples,
        n_genes=config.get("n_genes", 250),
        augment=True,
        n_contexts=n_contexts,
        context_dir=str(ctx_dir) if ctx_dir else None,
        global_feat_dir=str(gf_dir) if gf_dir else None,
    )
    val_ds = STDataset(
        data_dir=data_dir,
        dataset=dataset,
        sample_names=val_samples,
        n_genes=config.get("n_genes", 250),
        augment=False,
        n_contexts=n_contexts,
        context_dir=str(ctx_dir) if ctx_dir else None,
        global_feat_dir=str(gf_dir) if gf_dir else None,
    )

    from src.data.loaders import SlideBatchSampler
    pin_memory = bool(tc.get("pin_memory", False))
    train_batch_sampler = SlideBatchSampler(
        train_ds,
        batch_size=tc.get("batch_size", 128),
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=slide_collate_fn,
        num_workers=tc.get("num_workers", 4),
        pin_memory=pin_memory,
    )
    val_batch_sampler = SlideBatchSampler(
        val_ds,
        batch_size=tc.get("batch_size", 128),
        shuffle=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_batch_sampler,
        collate_fn=slide_collate_fn,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def evaluate_lambda(
    lambda_val: float,
    config: dict,
    fold_1_train_samples: List[str],
    baseline_q1_mse: float,
    data_dir: str,
    dataset: str,
    mcspr_artifacts: dict,
    output_dir: Path,
) -> Dict:
    """
    Train one short run (20 epochs max) with given lambda on internal 80%,
    evaluate on internal 20%.
    Returns dict with pcc_m, q1_mse, drift_triggered.
    """
    set_seed(2021)

    internal_train, internal_val = split_internal(fold_1_train_samples)

    ckpt_path = output_dir / f"lambda_{lambda_val}" / f"fold_{FOLD_FOR_SELECTION}" / "best_model.pt"

    # Resume-friendly: if a best_model.pt from a prior run exists, skip training
    # and only load + evaluate. Baseline (~100 min) and any prior lambda sweep
    # survive crashes this way.
    if ckpt_path.exists():
        print(f"  Found existing checkpoint for lambda={lambda_val} — "
              f"loading, skipping training.")
        _, val_loader = build_internal_loaders(
            data_dir=data_dir,
            dataset=dataset,
            train_samples=internal_train,
            val_samples=internal_val,
            config=config,
            fold_idx=FOLD_FOR_SELECTION,
        )
        train_loader = None
    else:
        train_loader, val_loader = build_internal_loaders(
            data_dir=data_dir,
            dataset=dataset,
            train_samples=internal_train,
            val_samples=internal_val,
            config=config,
            fold_idx=FOLD_FOR_SELECTION,
        )

        # Spec v2: sweep config must carry top-level keys (data_dir, dataset,
        # n_genes, n_folds, use_normalized_mse) so train_one_fold can locate
        # gene_var.npy and construct NormalizedMSELoss. Previously only
        # training + mcspr subsections were merged — caused KeyError on
        # data_dir lookup.
        sweep_config = {
            k: v for k, v in config.items()
            if k not in ("training", "mcspr", "model")
        }
        sweep_config.update(config.get("training", {}))
        sweep_config.update(config.get("mcspr", {}))
        sweep_config["lambda_max"] = lambda_val
        sweep_config["max_epochs"] = 20   # Short run for selection only
        sweep_config["early_stopping_patience"] = 10

        model = TRIPLEX(config, n_genes=config.get("n_genes", 250))

        # fold_idx = FOLD_FOR_SELECTION (1) so gene_var.npy is loaded from
        # nmf/fold_1/ — matches the fold the selection runs on.
        fold_result = train_one_fold(
            model=model,
            model_type="triplex",
            train_loader=train_loader,
            val_loader=val_loader,
            config=sweep_config,
            fold_idx=FOLD_FOR_SELECTION,
            output_dir=output_dir / f"lambda_{lambda_val}",
            mcspr_artifacts=mcspr_artifacts if lambda_val > 0 else None,
            dry_run=False,
        )

    # Load best checkpoint and evaluate
    model = TRIPLEX(config, n_genes=config.get("n_genes", 250))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    metrics = _evaluate(model, "triplex", val_loader, device)

    # Drift check: Q1 MSE must not increase > 5% above baseline
    drift_triggered = False
    if baseline_q1_mse > 0:
        q1_increase = (metrics["q1_mse"] - baseline_q1_mse) / baseline_q1_mse
        drift_triggered = q1_increase > DRIFT_THRESHOLD
        if drift_triggered:
            print(f"  lambda={lambda_val}: DRIFT triggered "
                  f"(Q1 MSE +{q1_increase:.1%} above baseline)")

    result = {
        "lambda": lambda_val,
        "pcc_m": metrics["pcc_m"],
        "q1_mse": metrics.get("q1_mse", 0.0),
        "rvd": metrics.get("rvd", 0.0),
        "drift_triggered": drift_triggered,
    }

    # Persist per-lambda result so parallel subset runs and the --finalize
    # aggregator can read individual sweep outcomes from disk.
    per_lambda_json = output_dir / f"lambda_{lambda_val}" / "result.json"
    per_lambda_json.parent.mkdir(parents=True, exist_ok=True)
    with open(per_lambda_json, "w") as f:
        json.dump(result, f, indent=2)

    # Explicit cleanup so WSI PIL caches, mmap refs, and CUDA tensors are
    # released before the next lambda iteration spins up a new dataset+model.
    # Without this, WSL's OOM killer can SIGKILL the process between sweeps.
    del model, train_loader, val_loader, ckpt, metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="MCSPR lambda selection (anti-leakage protocol)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--architecture", type=str, default="triplex",
        choices=["triplex"],
        help="Architecture to run selection on. Spec v2 locks to triplex; "
             "selected λ is frozen for all architectures.",
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Verify config and precomputed artifacts, "
                             "then exit without training.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override config training.num_workers. "
                             "Use 0 to avoid WSL OOM-kill from multi-proc "
                             "DataLoader workers holding WSI objects.")
    parser.add_argument("--lambda_subset", type=str, default=None,
                        help="Comma-separated list of lambda values to sweep "
                             "(e.g. '0.01,0.05'). When set, the script runs "
                             "ONLY these values and does not write "
                             "selected_lambda.json — used for GPU-parallel "
                             "sweeps. Baseline (lambda=0) still runs if its "
                             "cached metrics file (baseline.json) is absent.")
    parser.add_argument("--finalize", action="store_true",
                        help="Aggregate all per-lambda result.json files "
                             "under results/lambda_selection/{dataset}/ and "
                             "write selected_lambda.json. No training. Run "
                             "after all --lambda_subset processes complete.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.num_workers is not None:
        config.setdefault("training", {})["num_workers"] = args.num_workers
        print(f"num_workers override: {args.num_workers}")

    dataset = config["dataset"]
    data_dir = config["data_dir"]
    out_dir = Path(f"results/lambda_selection/{dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already selected -- do not re-run
    result_path = out_dir / "selected_lambda.json"
    if result_path.exists():
        with open(result_path) as f:
            existing = json.load(f)
        print(f"Lambda already selected: {existing['selected_lambda']}")
        print(f"Delete results/lambda_selection/{dataset}/ to re-run.")
        print("WARNING: Re-running invalidates the anti-leakage protocol.")
        return

    seed = config.get("training", {}).get("seed", 2021)
    set_seed(seed)

    # Discover samples
    bc_dir = Path(data_dir) / "barcodes"
    if not bc_dir.exists():
        print(f"ERROR: {bc_dir} not found.")
        return
    sample_names = sorted([p.stem for p in bc_dir.glob("*.csv")])

    # Spec v2: pass n_folds from config so her2st → 4-fold LOPCV
    folds = build_lopcv_folds(
        sample_names, dataset, n_folds=config.get("n_folds")
    )

    # Use Fold 1 (index 1) training samples for selection
    fold_1_train, fold_1_val = folds[FOLD_FOR_SELECTION]
    print(f"Lambda selection using Fold {FOLD_FOR_SELECTION}: "
          f"{len(fold_1_train)} train samples")
    print(f"Internal split: 80% train / 20% val")
    print(f"Lambda grid: {LAMBDA_GRID}")

    # Load NMF artifacts for Fold 1
    nmf_dir = Path(data_dir) / "nmf" / f"fold_{FOLD_FOR_SELECTION}"
    if not nmf_dir.exists():
        raise FileNotFoundError(
            f"NMF artifacts not found at {nmf_dir}. "
            f"Run precompute_all() first."
        )
    mc = config.get("mcspr", {})
    mcspr_artifacts = {
        "M_pinv": np.load(nmf_dir / "M_pinv.npy"),
        "C_prior": np.load(nmf_dir / "C_prior.npy"),
        "n_modules": mc.get("n_modules", 15),
        "n_contexts": mc.get("n_contexts", 6),
    }

    if args.dry_run:
        print("\n--- dry-run: config + artifacts OK ---")
        print(f"  M_pinv:  {mcspr_artifacts['M_pinv'].shape}")
        print(f"  C_prior: {mcspr_artifacts['C_prior'].shape}")
        print(f"  Fold {FOLD_FOR_SELECTION}: {len(fold_1_train)} train samples")
        print(f"  Lambda grid: {LAMBDA_GRID}")
        print("dry-run complete, exiting.")
        return

    baseline_json = out_dir / "baseline.json"

    # --finalize mode: read all per-lambda result.json files (plus baseline.json)
    # and write selected_lambda.json. No training.
    if args.finalize:
        if not baseline_json.exists():
            raise FileNotFoundError(
                f"baseline.json missing at {baseline_json}. "
                f"Run a non-finalize pass first to establish baseline."
            )
        with open(baseline_json) as f:
            baseline_cache = json.load(f)
        baseline_pcc = baseline_cache["pcc_m"]
        baseline_q1_mse = baseline_cache["q1_mse"]

        # Scan all lambda_*/result.json dirs so lambda=0 (baseline as a
        # candidate) and any future grid extensions are picked up without
        # requiring code edits.
        sweep_results = []
        found = []
        for d in sorted(out_dir.glob("lambda_*")):
            per = d / "result.json"
            if not per.exists():
                continue
            with open(per) as f:
                r = json.load(f)
            r["delta_pcc"] = r["pcc_m"] - baseline_pcc
            sweep_results.append(r)
            found.append(r["lambda"])
        missing = [l for l in LAMBDA_GRID if l not in found]
        if missing:
            raise FileNotFoundError(
                f"Missing per-lambda result.json for: {missing}. "
                f"Run --lambda_subset for those values first."
            )
        print(f"Finalize: candidates = {sorted(found)}")
        # Fall through to selection block below with sweep_results populated.
    else:
        # Baseline: run once, cache metrics to baseline.json for subset runs.
        if baseline_json.exists():
            with open(baseline_json) as f:
                baseline_cache = json.load(f)
            baseline_pcc = baseline_cache["pcc_m"]
            baseline_q1_mse = baseline_cache["q1_mse"]
            print(f"  Baseline cached: PCC(M)={baseline_pcc:.4f}  "
                  f"Q1_MSE={baseline_q1_mse:.4f}")
        else:
            print("\nStep 0: Establishing baseline (lambda=0, no MCSPR)...")
            baseline_result = evaluate_lambda(
                lambda_val=0.0,
                config=config,
                fold_1_train_samples=fold_1_train,
                baseline_q1_mse=0.0,
                data_dir=data_dir,
                dataset=dataset,
                mcspr_artifacts=None,
                output_dir=out_dir,
            )
            baseline_pcc = baseline_result["pcc_m"]
            baseline_q1_mse = baseline_result["q1_mse"]
            with open(baseline_json, "w") as f:
                json.dump({"pcc_m": baseline_pcc, "q1_mse": baseline_q1_mse}, f)
            print(f"  Baseline: PCC(M)={baseline_pcc:.4f}  "
                  f"Q1_MSE={baseline_q1_mse:.4f}")

        # Determine which lambda values this process handles
        if args.lambda_subset:
            lambdas_to_run = [float(x) for x in args.lambda_subset.split(",")]
            print(f"\nSubset mode: running {lambdas_to_run} only "
                  f"(no final aggregation).")
        else:
            lambdas_to_run = list(LAMBDA_GRID)

        sweep_results = []
        for lam in lambdas_to_run:
            print(f"\nSweeping lambda={lam}...")
            result = evaluate_lambda(
                lambda_val=lam,
                config=config,
                fold_1_train_samples=fold_1_train,
                baseline_q1_mse=baseline_q1_mse,
                data_dir=data_dir,
                dataset=dataset,
                mcspr_artifacts=mcspr_artifacts,
                output_dir=out_dir,
            )
            delta_pcc = result["pcc_m"] - baseline_pcc
            result["delta_pcc"] = delta_pcc
            sweep_results.append(result)
            print(f"  lambda={lam}: PCC(M)={result['pcc_m']:.4f} "
                  f"delta_PCC={delta_pcc:+.4f} "
                  f"drift={'YES' if result['drift_triggered'] else 'no'}")

        # Subset mode exits without writing selected_lambda.json.
        if args.lambda_subset:
            print(f"\nSubset complete. Per-lambda results saved under "
                  f"{out_dir}. Run `--finalize` after all subsets finish.")
            return

    # Step 2: Select best lambda that does not trigger drift
    valid = [r for r in sweep_results if not r["drift_triggered"]]
    if not valid:
        print("\nWARNING: All lambda values triggered drift. "
              "Selecting smallest lambda as fallback.")
        selected = sweep_results[0]  # lambda=0.01 (smallest)
    else:
        selected = max(valid, key=lambda r: r["delta_pcc"])

    print(f"\n{'=' * 50}")
    print(f"SELECTED LAMBDA: {selected['lambda']}")
    print(f"  Delta PCC: {selected['delta_pcc']:+.4f}")
    print(f"  Drift: {selected['drift_triggered']}")
    print(f"  This lambda is NOW FROZEN for all architectures and all folds.")
    print(f"{'=' * 50}")

    final = {
        "selected_lambda": selected["lambda"],
        "selection_fold": FOLD_FOR_SELECTION,
        "internal_val_frac": INTERNAL_VAL_FRACTION,
        "baseline_pcc_m": baseline_pcc,
        "baseline_q1_mse": baseline_q1_mse,
        "sweep_results": sweep_results,
        "drift_threshold": DRIFT_THRESHOLD,
        "proxy_architecture": "triplex",
        "frozen_for_architectures": ["stnet", "histogene", "triplex"],
        "frozen_for_all_folds": True,
    }
    with open(result_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
