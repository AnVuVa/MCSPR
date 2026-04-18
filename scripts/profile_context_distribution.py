"""Context distribution profiler — HisToGene MCSPR compatibility gate.

Determines whether within-slide context stratification is viable for
graph-based architectures (HisToGene) or whether cross-slide EMA
moment tracking must be derived.

Gate: >= 80% of training slides must have >= T-1 active contexts
      where active = effective_n = sum_i(w_it) >= k_min.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path("results/pre_training_gates/context_profiling")
T          = 6
K_MIN      = 30.0


def get_patient_id(sample_name: str, dataset: str) -> str:
    """Extract patient ID from MERGE bare sample names.

    her2st: A1 -> A, B2 -> B (letter prefix before digit)
    stnet:  BC23270_1 -> BC23270 (prefix before last underscore+digit)
    scc:    P1_1 -> P1 (prefix before last underscore+digit)
    """
    if dataset == "her2st":
        return sample_name[0].upper()
    elif dataset == "stnet":
        # STNet: patient is everything up to the last _N suffix
        parts = sample_name.rsplit("_", 1)
        return parts[0] if len(parts) > 1 else sample_name
    elif dataset == "scc":
        parts = sample_name.rsplit("_", 1)
        return parts[0] if len(parts) > 1 else sample_name
    raise ValueError(f"Unknown dataset: {dataset}")


def build_lopcv_folds(sample_names, dataset) -> List[Tuple]:
    patient_map = defaultdict(list)
    for s in sample_names:
        patient_map[get_patient_id(s, dataset)].append(s)
    patients = sorted(patient_map.keys())

    if dataset == "stnet":
        # 8-fold patient-stratified
        n_folds = 8
        fold_size = max(1, len(patients) // n_folds)
        folds = []
        for i in range(n_folds):
            test_pats  = patients[i * fold_size:(i + 1) * fold_size]
            train_pats = [p for p in patients if p not in test_pats]
            folds.append(
                ([s for p in train_pats for s in patient_map[p]],
                 [s for p in test_pats  for s in patient_map[p]])
            )
    else:
        # LOPCV
        folds = []
        for held in patients:
            train = [s for p in patients if p != held
                     for s in patient_map[p]]
            folds.append((train, patient_map[held]))
    return folds


def profile_slide(weights: np.ndarray, k_min: float) -> Dict:
    N, T_actual = weights.shape
    eff_n    = weights.sum(axis=0)
    active   = eff_n >= k_min
    n_active = int(active.sum())
    return {
        "n_spots":          N,
        "effective_n":      eff_n.tolist(),
        "n_active":         n_active,
        "passes":           n_active >= (T_actual - 1),
        "dominant_frac":    float(eff_n.max() / N),
        "rarest_eff_n":     float(eff_n.min()),
        "active_contexts":  np.where(active)[0].tolist(),
        "dropped_contexts": np.where(~active)[0].tolist(),
    }


def run_dataset(data_dir: Path, dataset: str, context_base_dir: Path) -> Dict:
    # Check for fold-specific subdirectories
    fold_dirs = sorted(context_base_dir.glob("fold_*"))
    if fold_dirs:
        # Use fold-aware profiling
        sample_names_all = sorted([
            f.stem for f in fold_dirs[0].glob("*.npy")
        ])
    else:
        # Flat directory
        weight_files = sorted(context_base_dir.glob("*.npy"))
        if not weight_files:
            return {"error": f"No context weight files in {context_base_dir}"}
        sample_names_all = [f.stem for f in weight_files]
        fold_dirs = [context_base_dir]

    folds = build_lopcv_folds(sample_names_all, dataset)

    all_stats = []
    for fold_idx, (train_samples, _) in enumerate(folds):
        # Use fold-specific weights if available
        if len(fold_dirs) > 1 and fold_idx < len(fold_dirs):
            ctx_dir = fold_dirs[fold_idx]
        elif len(fold_dirs) == 1:
            ctx_dir = fold_dirs[0]
        else:
            ctx_dir = context_base_dir / f"fold_{fold_idx}"

        for sample in train_samples:
            wf = ctx_dir / f"{sample}.npy"
            if not wf.exists():
                continue
            w = np.load(wf)
            if w.shape[1] != T:
                print(f"  WARN: {sample} T={w.shape[1]} != {T}, skipping")
                continue
            stat = profile_slide(w, K_MIN)
            stat["sample"] = sample
            stat["fold"]   = fold_idx
            all_stats.append(stat)

    if not all_stats:
        return {"error": "no valid slides"}

    passes        = [s["passes"] for s in all_stats]
    pass_rate     = np.mean(passes)
    n_active_list = [s["n_active"] for s in all_stats]
    dom_fracs     = [s["dominant_frac"] for s in all_stats]
    rarest        = [s["rarest_eff_n"] for s in all_stats]

    gate = pass_rate >= 0.80

    print(f"\n-- {dataset.upper()} --")
    print(f"  Slides profiled:       {len(all_stats)}")
    print(f"  Pass rate (>= T-1=5):  {pass_rate:.1%}  "
          f"[{'PASS' if gate else 'FAIL'}]")
    print(f"  Mean active contexts:  {np.mean(n_active_list):.2f}")
    print(f"  Mean dominant frac:    {np.mean(dom_fracs):.3f}")
    print(f"  Slides > 80% single:   "
          f"{sum(d > 0.80 for d in dom_fracs)} "
          f"({sum(d > 0.80 for d in dom_fracs) / len(dom_fracs):.1%})")
    print(f"  Rarest eff_n mean:     {np.mean(rarest):.1f}")
    print(f"  Rarest eff_n 10th pct: {np.percentile(rarest, 10):.1f}")

    print(f"\n  Context starvation distribution:")
    ctr = Counter(n_active_list)
    for n_act in sorted(ctr):
        print(f"    {n_act}/{T} active: {ctr[n_act]} slides "
              f"({ctr[n_act] / len(all_stats):.1%})")

    return {
        "dataset":              dataset,
        "n_slides":             len(all_stats),
        "pass_rate":            float(pass_rate),
        "gate_pass":            bool(gate),
        "mean_active_contexts": float(np.mean(n_active_list)),
        "mean_dominant_frac":   float(np.mean(dom_fracs)),
        "p10_rarest_eff_n":     float(np.percentile(rarest, 10)),
        "frac_gt80_single":     float(sum(d > 0.80 for d in dom_fracs)
                                      / len(dom_fracs)),
        "decision": (
            "within_slide_stratification_viable"
            if gate else
            "cross_slide_EMA_gradient_required"
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for dataset in ["her2st", "stnet"]:
        data_dir    = Path(f"data/{dataset}")
        context_dir = data_dir / "context_weights"

        if not context_dir.exists():
            print(f"WARN: {context_dir} not found. "
                  f"Run precompute_context_clusters() first.")
            results[dataset] = {"error": "context_weights_not_found"}
            continue

        results[dataset] = run_dataset(data_dir, dataset, context_dir)

    # Final decision
    print(f"\n{'='*60}")
    print(f"GRAPH MODEL COMPATIBILITY GATE (T={T}, k_min={K_MIN})")
    print(f"{'='*60}")

    all_pass = all(r.get("gate_pass", False) for r in results.values()
                   if "error" not in r)

    for ds, r in results.items():
        if "error" in r:
            print(f"  {ds}: ERROR -- {r['error']}")
        else:
            status = "PASS" if r["gate_pass"] else "FAIL"
            print(f"  {ds}: {status} (pass_rate={r['pass_rate']:.1%})")
            print(f"    Decision: {r['decision']}")

    print(f"\nOverall gate: {'PASS' if all_pass else 'FAIL'}")
    if all_pass:
        print("  HisToGene+MCSPR proceeds with within-slide stratification.")
        print("  Cross-slide EMA gradient formulation is NOT required.")
    else:
        print("  Cross-slide EMA gradient formulation is REQUIRED")
        print("  before HisToGene+MCSPR can be implemented.")

    with open(OUTPUT_DIR / "context_profile_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
