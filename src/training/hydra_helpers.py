"""Helpers for HydraSTNet runner: registry verification, label slicing,
per-head and full-300 result IO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def load_registry(path: str | Path) -> dict:
    """Load and SHA-256-verify the module registry."""
    path = Path(path)
    with open(path) as f:
        registry = json.load(f)
    declared = registry.get("sha256")
    payload = {k: v for k, v in registry.items() if k != "sha256"}
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    if declared != actual:
        raise ValueError(
            f"Registry SHA-256 mismatch at {path}\n"
            f"  declared: {declared}\n  actual:   {actual}"
        )
    return registry


def verify_modules(
    dataset, registry: dict,
    fold_idx: int | None = None,
    train_loader=None,
) -> List[List[int]]:
    """CP2: ensure dataset gene order matches the registry exactly and that
    the module partition is a true partition of [0, n_genes). Returns
    idx_list where idx_list[k] is the column indices in y for head k.

    Stronger guards (added after compaction):
      - fold_idx: if provided, asserts registry["fold"] == fold_idx so a
        wrong-fold registry doesn't silently get used (gene names match
        across folds; only the module partition differs, so the cheaper
        `gene_names == gene_names_full` check would NOT catch this).
      - train_loader: if provided, asserts the loader's underlying train
        slides match registry["train_slides"]. Catches a registry that was
        built from a different fold's training-slide list (e.g. someone
        regenerated splits without rebuilding registries).
      - The returned idx_list is checked against module_to_genes by name:
        for each module k and position j, gene_names_full[idx_list[k][j]]
        == module_to_genes[k][j]. This catches a registry whose
        module_to_indices and module_to_genes were generated independently
        (or got out of sync after manual editing).
    """
    if fold_idx is not None and registry.get("fold") != fold_idx:
        raise ValueError(
            f"Registry fold mismatch: registry['fold']={registry.get('fold')} "
            f"but runner fold_idx={fold_idx}. The wrong fold's registry would "
            f"give correct gene names but the wrong module partition — refusing."
        )

    if dataset.gene_names != registry["gene_names_full"]:
        for i, (a, b) in enumerate(
            zip(dataset.gene_names, registry["gene_names_full"])
        ):
            if a != b:
                raise ValueError(
                    f"Gene order mismatch at column {i}: dataset={a!r} "
                    f"registry={b!r}"
                )
        raise ValueError(
            f"Gene name list lengths differ: dataset={len(dataset.gene_names)} "
            f"registry={len(registry['gene_names_full'])}"
        )

    if train_loader is not None and "train_slides" in registry:
        # Pull the slide-name set the loader is actually feeding the model.
        # STDataset stores sample_names as a list; the order doesn't matter
        # for correctness here, only the set membership does.
        loader_slides = sorted(getattr(train_loader.dataset, "sample_names",
                                       getattr(train_loader.dataset, "slides",
                                               [])))
        registry_slides = sorted(registry["train_slides"])
        if loader_slides and loader_slides != registry_slides:
            extra_in_loader = sorted(set(loader_slides) - set(registry_slides))
            extra_in_registry = sorted(
                set(registry_slides) - set(loader_slides)
            )
            raise ValueError(
                "Train slide set drift between loader and registry. The "
                "registry was built from a different training set than what "
                "the dataloader is feeding the model. "
                f"Only-in-loader: {extra_in_loader[:5]} (n="
                f"{len(extra_in_loader)}) "
                f"Only-in-registry: {extra_in_registry[:5]} (n="
                f"{len(extra_in_registry)}) "
                "Rebuild the registry: "
                f"python scripts/build_module_registry.py --fold {fold_idx}"
            )

    K = registry["K"]
    n_genes = registry["n_genes"]
    gene_names_full = registry["gene_names_full"]
    idx_list: List[List[int]] = []
    seen = set()
    for k in range(K):
        idx = sorted(int(i) for i in registry["module_to_indices"][str(k)])
        if any(i < 0 or i >= n_genes for i in idx):
            raise ValueError(f"Module {k} has out-of-range index")
        for i in idx:
            if i in seen:
                raise ValueError(f"Gene index {i} appears in multiple modules")
            seen.add(i)
        # Cross-check: column index → gene name must match module_to_genes
        names_by_index = [gene_names_full[i] for i in idx]
        names_recorded = sorted(registry["module_to_genes"][str(k)])
        if sorted(names_by_index) != names_recorded:
            raise ValueError(
                f"Module {k}: module_to_indices and module_to_genes disagree.\n"
                f"  by index → name: {names_by_index[:5]}...\n"
                f"  by name (sorted): {names_recorded[:5]}..."
            )
        idx_list.append(idx)
    if len(seen) != n_genes:
        missing = sorted(set(range(n_genes)) - seen)
        raise ValueError(
            f"Module partition missing {len(missing)} genes "
            f"(first few: {missing[:10]})"
        )
    return idx_list


def assert_first_batch_slicing(
    preds: List[torch.Tensor], y_true: torch.Tensor, idx_list: List[List[int]],
) -> None:
    """CP2 step C: verify the labels we're sending to head k really are the
    columns idx_list[k] of the batch's expression vector.

    We assert shapes match; value-level slice correctness is enforced by
    construction in train_step (we never look up labels by gene name in the
    hot loop). This guard catches a future regression where someone changes
    the slice semantics.
    """
    K = len(idx_list)
    if len(preds) != K:
        raise AssertionError(f"Got {len(preds)} preds, expected {K}")
    for k in range(K):
        expected = (preds[k].shape[0], len(idx_list[k]))
        target = y_true[:, idx_list[k]]
        if preds[k].shape != expected:
            raise AssertionError(
                f"Head {k} pred shape {tuple(preds[k].shape)} != "
                f"expected {expected}"
            )
        if target.shape != expected:
            raise AssertionError(
                f"Head {k} label slice shape {tuple(target.shape)} != "
                f"expected {expected}"
            )


def per_head_loss(
    preds: List[torch.Tensor],
    y_true: torch.Tensor,
    idx_list: List[List[int]],
    loss_fn=None,
    return_per_head: bool = False,
):
    """Weighted-by-module-size mean of per-head MSEs.

    Mathematically identical to a single F.mse_loss on the full 300-gene
    prediction reassembled from the per-head outputs:

        total = (Σ_k m_k · MSE_k) / Σ_k m_k
              = (Σ_all_elements (y_hat - y_true)²) / (B · n_genes)
              = F.mse_loss(full_300_preds, y_true)

    Why weighted, not sum:
      - Each GENE contributes one unit of weight (not each HEAD), so big
        modules don't get under-weighted just for being one head; small
        modules don't dominate via the unweighted sum-of-means.
      - Total scale matches baseline STNet's loss (~0.5–1.5 over training)
        so the two are directly comparable.
      - Per-gene gradient magnitude ≡ baseline STNet — head k's params
        only see gradient from genes in module k (head_k is decoupled);
        the shared backbone receives the same per-gene gradient as in
        baseline STNet.

    Args:
        return_per_head: when True, returns (total, per_head_mse_list).
            head_losses[k] is the per-head MSE (mean over module's genes
            and batch's spots) — for logging only; the training signal is
            the weighted total.
    """
    if loss_fn is None:
        loss_fn = torch.nn.functional.mse_loss
    head_losses: List[torch.Tensor] = []
    sse_total = 0.0  # weighted accumulator (mse_k · n_elements_k = SSE_k)
    n_elements = 0
    for k in range(len(idx_list)):
        mse_k = loss_fn(preds[k], y_true[:, idx_list[k]])
        head_losses.append(mse_k)
        sse_total = sse_total + mse_k * preds[k].numel()
        n_elements += preds[k].numel()
    total = sse_total / n_elements
    if return_per_head:
        return total, head_losses
    return total


def reassemble_full_preds(
    preds_list: List[torch.Tensor], idx_list: List[List[int]], n_genes: int,
) -> torch.Tensor:
    """Stitch K head outputs into a single (B, n_genes) tensor in canonical
    column order so existing per-slide PCC machinery can consume it."""
    out = torch.empty(
        preds_list[0].shape[0], n_genes,
        dtype=preds_list[0].dtype, device=preds_list[0].device,
    )
    for k, idx in enumerate(idx_list):
        out[:, idx] = preds_list[k]
    return out


def save_head_results(
    *,
    pcc_per_gene: Dict[str, float],
    module_id: int,
    registry: dict,
    fold: int,
    backbone: str,
    path: str | Path,
    extra: Dict | None = None,
) -> None:
    """CP3: per-head result file with gene metadata + registry hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gene_names = registry["module_to_genes"][str(module_id)]
    pcc_vals = [pcc_per_gene[g] for g in gene_names]
    payload = {
        "backbone": backbone,
        "fold": fold,
        "module_id": module_id,
        "registry_hash": registry["sha256"],
        "gene_names": gene_names,
        "pcc_per_gene": pcc_per_gene,
        "pcc_module_mean": float(np.nanmean(pcc_vals)) if pcc_vals else 0.0,
        "pcc_module_std": float(np.nanstd(pcc_vals)) if pcc_vals else 0.0,
        "n_genes_in_module": len(gene_names),
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_full_results(
    *,
    pcc_per_gene: Dict[str, float],
    registry: dict,
    fold: int,
    backbone: str,
    path: str | Path,
    extra: Dict | None = None,
) -> None:
    """CP3: full-300 result file (used for direct comparison vs baseline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    full_names = registry["gene_names_full"]
    if set(pcc_per_gene.keys()) != set(full_names):
        missing = set(full_names) - set(pcc_per_gene.keys())
        extra_keys = set(pcc_per_gene.keys()) - set(full_names)
        raise ValueError(
            f"pcc_per_gene keys do not cover all 300 genes; "
            f"missing={len(missing)} extra={len(extra_keys)}"
        )
    pcc_vals = [pcc_per_gene[g] for g in full_names]
    payload = {
        "backbone": backbone,
        "fold": fold,
        "registry_hash": registry["sha256"],
        "n_genes": len(full_names),
        "pcc_per_gene": pcc_per_gene,
        "pcc_full_mean": float(np.nanmean(pcc_vals)),
        "pcc_full_std": float(np.nanstd(pcc_vals)),
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
