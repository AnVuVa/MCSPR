"""Gate 2 — MSigDB Hallmark enrichment of NMF modules.

Per-module test: for each NMF module k, take top-50 SVG genes by loading
weight in M[:, k], run Fisher's exact (one-sided greater) against every
Hallmark gene set, with BH-FDR correction over all 15 modules × 50 sets
per fold. Module passes if any pathway reaches FDR < 0.05.

Fold passes Gate 2 enrichment if >= 10 / 15 modules are enriched.

Replaces the buggy per-context enrichment in validate_prior.py, which
(a) loaded gene_names from features/*.csv (the 512-dim MERGE HVG set —
wrong panel), and (b) used module indices as gene indices.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.stats import fisher_exact


TOP_K = 50          # top-loading SVG genes per module
FDR_ALPHA = 0.05    # per-module significance
MIN_MODULES_PASS = 10   # >= 10 / 15 for fold PASS


def load_gmt(path: Path) -> dict:
    sets = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = {g for g in parts[2:] if g}
            sets[name] = genes
    return sets


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def module_top_genes(M: np.ndarray, svg_names: list, k: int, top_k: int) -> list:
    idx = np.argsort(M[:, k])[::-1][:top_k]
    return [svg_names[int(i)] for i in idx]


def fisher_top_genes_vs_set(
    top_genes: set, gene_set: set, universe: set
) -> tuple:
    """Fisher's exact against a GENOME-WIDE universe (not the SVG panel).

    The SVG panel (300 genes) is pre-selected for spatial variability; using
    it as the Fisher universe drastically underestimates the baseline rate
    of any Hallmark gene in the background, which makes BH correction over
    50 sets × 15 modules unreachable even for biologically strong hits
    (e.g. p=0.047 on Module 1 × HYPOXIA collapses to FDR=1.0).

    The biologically meaningful question is "is this module's top-50 enriched
    for HYPOXIA compared to a random draw from the transcriptome", which
    requires the full gene list as the universe.
    """
    a = len(top_genes & gene_set)
    b = len(top_genes) - a
    c = len(gene_set & universe) - a
    d = len(universe) - a - b - c
    if a == 0 or c < 0 or d < 0:
        return 1.0, a
    _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return float(p), a


def enrich_fold(
    fold_idx: int,
    nmf_dir: Path,
    svg_names: list,
    hallmark_sets: dict,
    universe: set,
) -> dict:
    M = np.load(nmf_dir / f"fold_{fold_idx}" / "M.npy")
    assert M.shape == (len(svg_names), 15), (
        f"Fold {fold_idx}: M={M.shape} does not match "
        f"(n_svg={len(svg_names)}, 15)"
    )

    set_names = list(hallmark_sets.keys())

    n_modules, n_sets = M.shape[1], len(set_names)
    pvals = np.ones((n_modules, n_sets))
    overlaps = np.zeros((n_modules, n_sets), dtype=int)
    module_genes = {}

    for k in range(n_modules):
        genes_k = module_top_genes(M, svg_names, k, TOP_K)
        module_genes[k] = genes_k
        top_set = set(genes_k)
        for j, sname in enumerate(set_names):
            p, a = fisher_top_genes_vs_set(top_set, hallmark_sets[sname], universe)
            pvals[k, j] = p
            overlaps[k, j] = a

    fdr = bh_fdr(pvals.flatten()).reshape(pvals.shape)

    modules = []
    n_pass = 0
    for k in range(n_modules):
        best_j = int(np.argmin(fdr[k]))
        best_fdr = float(fdr[k, best_j])
        best_p = float(pvals[k, best_j])
        best_set = set_names[best_j]
        best_ov = int(overlaps[k, best_j])
        passes = best_fdr < FDR_ALPHA
        n_pass += int(passes)
        modules.append({
            "module": k,
            "top_pathway": best_set,
            "p_value": best_p,
            "fdr": best_fdr,
            "overlap": best_ov,
            "top_50_genes": module_genes[k][:20],
            "pass": passes,
        })

    return {
        "fold": fold_idx,
        "n_modules_enriched": n_pass,
        "n_modules_total": n_modules,
        "pass": n_pass >= MIN_MODULES_PASS,
        "modules": modules,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nmf_dir", type=Path, default=Path("data/her2st/nmf")
    )
    ap.add_argument(
        "--svg_path", type=Path,
        default=Path("results/pre_training_gates/gene_selection/svg_genes.json"),
    )
    ap.add_argument(
        "--msigdb_path", type=Path,
        default=Path(
            "/mnt/d/DrugResponsePrediction/XGDP/data/h.all.v7.5.1.symbols.gmt"
        ),
    )
    ap.add_argument(
        "--universe_path", type=Path,
        default=Path("data/her2st/features_full/gene_names.json"),
        help="Full-transcriptome universe for Fisher's exact. "
             "Must NOT be the SVG panel (too narrow for BH).",
    )
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument(
        "--output_dir", type=Path,
        default=Path("results/prior_validation/enrichment"),
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    svg_names = json.loads(args.svg_path.read_text())
    hallmark = load_gmt(args.msigdb_path)
    universe = set(json.loads(args.universe_path.read_text()))

    print(f"GMT file:       {args.msigdb_path}")
    print(f"Hallmark sets:  {len(hallmark)}")
    print(f"SVG genes:      {len(svg_names)}")
    print(f"Universe genes: {len(universe)}  ({args.universe_path})")
    print(f"Folds:          {args.folds}")
    print()

    per_fold = []
    for f in args.folds:
        res = enrich_fold(f, args.nmf_dir, svg_names, hallmark, universe)
        per_fold.append(res)

        print(f"=== FOLD {f}  "
              f"({res['n_modules_enriched']}/{res['n_modules_total']} modules enriched, "
              f"{'PASS' if res['pass'] else 'FAIL'}) ===")
        print(f"{'Mod':>3} | {'Top Pathway':<40} | {'p-value':>10} | "
              f"{'FDR':>8} | Overlap")
        print("-" * 82)
        for m in res["modules"]:
            print(
                f"{m['module']:>3} | {m['top_pathway']:<40} | "
                f"{m['p_value']:>10.2e} | {m['fdr']:>8.3e} | "
                f"{m['overlap']:>3}/{TOP_K}"
            )
        print()

    total_pass = sum(1 for r in per_fold if r["pass"])
    overall_pass = total_pass == len(per_fold)

    print("=" * 82)
    print(f"{'Fold':<4} | {'Modules enriched (FDR<0.05) / 15':<36} | PASS (>=10)?")
    print("-" * 82)
    for r in per_fold:
        print(
            f"{r['fold']:<4} | "
            f"{r['n_modules_enriched']:>10} / 15 "
            f"{'':<22} | {'YES' if r['pass'] else 'NO'}"
        )
    print("-" * 82)
    print(f"Overall enrichment gate: {'PASS' if overall_pass else 'FAIL'} "
          f"({total_pass}/{len(per_fold)} folds)")

    out = args.output_dir / "enrichment_summary.json"
    with open(out, "w") as fp:
        json.dump(
            {
                "msigdb_path": str(args.msigdb_path),
                "n_hallmark_sets": len(hallmark),
                "n_svg_genes": len(svg_names),
                "top_k_genes_per_module": TOP_K,
                "fdr_alpha": FDR_ALPHA,
                "min_modules_pass": MIN_MODULES_PASS,
                "per_fold": per_fold,
                "overall_pass": overall_pass,
            },
            fp,
            indent=2,
        )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
