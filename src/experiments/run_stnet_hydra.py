"""Train HydraSTNet (DenseNet121 backbone + 7 MLP heads, one per gene module).

Surgical change from the STNet baseline: same backbone, same loader, same
LOPCV split, same canonical metric. Only the head changes — from a single
Linear(1024 → 300) to K parallel HydraHead MLPs whose outputs cover the
300 SVG genes via the registry partition.

Usage:
    python src/experiments/run_stnet_hydra.py \
        --config configs/her2st.yaml \
        --registry results/ablation/kmeans_y_elbow/fold_{F}/modules_fold{F}.json \
        --fold {F}                                 # default: -1 -> all folds
        [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loaders import build_stnet_hydra_loaders
from src.models.stnet_hydra import HydraSTNet
from src.training.hydra_helpers import (
    assert_first_batch_slicing,
    load_registry,
    per_head_loss,
    reassemble_full_preds,
    save_full_results,
    save_head_results,
    verify_modules,
)


def set_seed(seed: int = 2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _registry_path_for_fold(template: str, fold: int) -> Path:
    """`--registry` accepts {F} or {fold} as the per-fold placeholder."""
    s = template.replace("{F}", str(fold)).replace("{fold}", str(fold))
    return Path(s)


def _per_gene_pcc(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n = y_hat.shape[1]
    out = np.zeros(n, dtype=np.float64)
    for j in range(n):
        if np.std(y_hat[:, j]) < 1e-10 or np.std(y_true[:, j]) < 1e-10:
            out[j] = 0.0
        else:
            r, _ = pearsonr(y_hat[:, j], y_true[:, j])
            out[j] = r if not np.isnan(r) else 0.0
    return out


def _evaluate_full_300(
    model: HydraSTNet, val_loader, idx_list, device,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Collect val predictions/truth grouped by sample_idx, then return
    {sample_idx: (n_spots, 300) arrays} for predictions and ground truth."""
    model.eval()
    n_genes = sum(len(idx) for idx in idx_list)
    slide_preds: dict[int, list[np.ndarray]] = {}
    slide_trues: dict[int, list[np.ndarray]] = {}
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            preds = model(batch)["output"]
            full = reassemble_full_preds(preds, idx_list, n_genes).cpu().numpy()
            y_true = batch["expression"].cpu().numpy()
            sids = batch["sample_idx"].cpu().numpy()
            for i in range(full.shape[0]):
                sid = int(sids[i])
                slide_preds.setdefault(sid, []).append(full[i])
                slide_trues.setdefault(sid, []).append(y_true[i])
    return (
        {sid: np.stack(v) for sid, v in slide_preds.items()},
        {sid: np.stack(v) for sid, v in slide_trues.items()},
    )


def _train_fold(
    *,
    fold_idx: int,
    config: dict,
    registry: dict,
    output_dir: Path,
    device: torch.device,
    dry_run: bool = False,
) -> None:
    data_dir = config["data_dir"]
    dataset = config["dataset"]
    seed = config.get("training", {}).get("seed", 2021)
    tc = config.get("training", {})

    set_seed(seed)
    train_loader, val_loader = build_stnet_hydra_loaders(
        data_dir, dataset, fold_idx, config,
    )

    # CP2 (gene-order match + partition validity + fold-mismatch + slide drift)
    # — happens BEFORE model build so a wrong-fold registry can't waste compute.
    idx_list = verify_modules(
        train_loader.dataset, registry,
        fold_idx=fold_idx, train_loader=train_loader,
    )
    module_sizes = registry["module_sizes"]
    if [len(idx) for idx in idx_list] != module_sizes:
        raise AssertionError(
            f"idx_list sizes {[len(i) for i in idx_list]} != registry "
            f"module_sizes {module_sizes}"
        )
    assert sum(module_sizes) == registry["n_genes"], (
        "module_sizes must sum to n_genes — registry corrupted"
    )

    model = HydraSTNet(
        module_sizes=module_sizes,
        d_hidden=config.get("model", {}).get("hydra_d_hidden", 128),
        pretrained=True,
    ).to(device)
    print(
        f"Fold {fold_idx} | HydraSTNet | K={registry['K']} sizes={module_sizes} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tc.get("lr", 1e-4),
        weight_decay=tc.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=tc.get("step_size", 50),
        gamma=tc.get("gamma", 0.9),
    )
    scaler = torch.amp.GradScaler("cuda")

    max_epochs = tc.get("max_epochs", 200)
    patience = tc.get("early_stopping_patience", 20)
    best_pcc = -float("inf")
    patience_counter = 0
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    training_log = []
    first_batch_checked = False

    K = len(idx_list)
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_per_head_loss = [0.0] * K
        epoch_start = time.time()
        n_batches = 0
        last_print = epoch_start

        for batch_i, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                preds = model(
                    {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
                )["output"]
                y_true = batch["expression"].to(device)

                if not first_batch_checked:
                    # CP2 step C: shape & slicing assertion (one-shot).
                    assert_first_batch_slicing(preds, y_true, idx_list)
                    first_batch_checked = True
                    print(
                        f"  CP2 OK: K={len(preds)} heads, "
                        f"label slices match expected shapes"
                    )

                loss, head_losses = per_head_loss(
                    preds, y_true, idx_list, F.mse_loss,
                    return_per_head=True,
                )
                if dry_run:
                    per_h = " ".join(
                        f"h{k}={head_losses[k].item():.3f}"
                        for k in range(len(idx_list))
                    )
                    print(
                        f"  Dry run total_loss={loss.item():.4f} ({per_h})"
                    )
                    return

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            for k in range(len(idx_list)):
                epoch_per_head_loss[k] += float(head_losses[k].item())
            n_batches += 1
            if epoch < 3:
                now = time.time()
                if now - last_print >= 20.0 or batch_i == 0:
                    print(
                        f"    E{epoch:03d} batch {batch_i:4d} "
                        f"elapsed={now - epoch_start:6.1f}s "
                        f"loss={loss.item():.4f}",
                        flush=True,
                    )
                    last_print = now

        scheduler.step()
        wall = time.time() - epoch_start

        # Validation — full-300 PCC averaged per slide (matches canonical_eval)
        # AND per-head PCC for visibility (each head's mean PCC over its module).
        slide_preds, slide_trues = _evaluate_full_300(
            model, val_loader, idx_list, device,
        )
        slide_pcc_m = []
        per_slide_gene_pccs = []
        for sid in sorted(slide_preds.keys()):
            gene_pccs = _per_gene_pcc(slide_preds[sid], slide_trues[sid])
            slide_pcc_m.append(float(np.nanmean(gene_pccs)))
            per_slide_gene_pccs.append(gene_pccs)
        val_pcc = float(np.mean(slide_pcc_m)) if slide_pcc_m else 0.0
        # Per-head val PCC (mean over module's genes, averaged across slides)
        if per_slide_gene_pccs:
            mean_gene_pccs = np.nanmean(np.stack(per_slide_gene_pccs), axis=0)
            val_pcc_per_head = [
                float(np.nanmean(mean_gene_pccs[np.array(idx_list[k], dtype=int)]))
                for k in range(K)
            ]
        else:
            val_pcc_per_head = [0.0] * K

        improved = val_pcc > best_pcc
        if improved:
            best_pcc = val_pcc
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "val_pcc_m": val_pcc, "registry_hash": registry["sha256"]},
                fold_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        per_head_train = [
            epoch_per_head_loss[k] / max(n_batches, 1) for k in range(K)
        ]
        training_log.append({
            "epoch": epoch,
            "train_loss_total": epoch_loss / max(n_batches, 1),
            "train_loss_per_head": per_head_train,
            "val_pcc_m": val_pcc,
            "val_pcc_per_head": val_pcc_per_head,
            "wall_s": wall,
        })
        if epoch % 10 == 0 or epoch < 5 or improved:
            head_train_str = " ".join(
                f"h{k}={per_head_train[k]:.3f}" for k in range(K)
            )
            head_val_str = " ".join(
                f"h{k}={val_pcc_per_head[k]:+.3f}" for k in range(K)
            )
            print(
                f"  E{epoch:03d} | loss={epoch_loss/max(n_batches,1):.4f} "
                f"val_pcc={val_pcc:.4f} | "
                f"patience={patience_counter}/{patience} wall={wall:.1f}s "
                f"{'(best)' if improved else ''}",
                flush=True,
            )
            print(f"        train MSE/head: {head_train_str}", flush=True)
            print(f"        val PCC/head:   {head_val_str}", flush=True)

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    with open(fold_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Reload best checkpoint for final evaluation
    ckpt = torch.load(
        str(fold_dir / "best_model.pt"), map_location="cpu", weights_only=False,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)

    slide_preds, slide_trues = _evaluate_full_300(
        model, val_loader, idx_list, device,
    )

    # === Canonical per-gene PCC (matches scripts/canonical_eval.py) ===
    # For each slide compute per-gene PCC via scipy.stats.pearsonr; then
    # average each gene's PCC across slides. This is exactly the
    # mean_gene_pccs that canonical_eval.py uses (same scipy formula, same
    # NaN-safe std-guard). Reporting from this matrix makes Hydra and the
    # baseline fair to compare.
    sids_sorted = sorted(slide_preds.keys())
    per_slide_gene_pccs = np.stack([
        _per_gene_pcc(slide_preds[sid], slide_trues[sid])
        for sid in sids_sorted
    ], axis=0)  # (n_slides, 300)
    gene_pccs = np.nanmean(per_slide_gene_pccs, axis=0)  # (300,)
    pcc_per_gene_full = {
        registry["gene_names_full"][i]: float(gene_pccs[i])
        for i in range(registry["n_genes"])
    }
    # Also per-slide PCC(M) (used for early stopping above) — recompute here
    # for completeness so the saved summary is self-contained.
    slide_pcc_m_final = [
        float(np.nanmean(per_slide_gene_pccs[s])) for s in range(len(sids_sorted))
    ]
    pcc_m_per_slide_mean = float(np.mean(slide_pcc_m_final))

    # === Per-fold module-level breakdown (LOG to stdout AND save to JSON) ===
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx} — HydraSTNet per-module breakdown")
    print(f"{'='*70}")
    print(f"Metric: per-slide scipy.stats.pearsonr → mean across slides "
          f"(n_val_slides={len(sids_sorted)})")
    print(f"{'mod':<5}{'n_genes':<10}{'mean_PCC':<12}{'std':<10}genes")

    module_breakdown = []
    weighted_numerator = 0.0
    total_genes = 0
    for k in range(registry["K"]):
        gene_names_k = registry["module_to_genes"][str(k)]
        idx_k = registry["module_to_indices"][str(k)]
        pcc_k = gene_pccs[np.array(idx_k, dtype=int)]
        m_k = len(gene_names_k)
        mean_k = float(np.nanmean(pcc_k))
        std_k = float(np.nanstd(pcc_k))
        weighted_numerator += mean_k * m_k
        total_genes += m_k
        # Show the genes (truncated for readability if very long)
        gene_preview = ", ".join(gene_names_k[:6])
        if m_k > 6:
            gene_preview += f", ... (+{m_k - 6} more)"
        print(f"{k:<5}{m_k:<10}{mean_k:<12.4f}{std_k:<10.4f}{gene_preview}")
        module_breakdown.append({
            "module_id": k,
            "n_genes": m_k,
            "gene_names": gene_names_k,
            "pcc_per_gene": {g: float(pcc_per_gene_full[g])
                             for g in gene_names_k},
            "module_mean_pcc": mean_k,
            "module_std_pcc": std_k,
        })

    weighted_full_pcc = weighted_numerator / total_genes if total_genes else 0.0
    # Identity check: weighted-by-size mean of module means == unweighted
    # mean of all per-gene PCCs. We compute both and assert equality so this
    # invariant can't drift silently.
    unweighted_full_pcc = float(np.nanmean(gene_pccs))
    assert abs(weighted_full_pcc - unweighted_full_pcc) < 1e-6, (
        f"weighted ({weighted_full_pcc}) != unweighted ({unweighted_full_pcc})"
    )

    print(f"{'-'*70}")
    print(f"FINAL  | weighted PCC across all modules (Σ m_k·μ_k / Σ m_k) = "
          f"{weighted_full_pcc:.4f}")
    print(f"       | per-slide PCC(M) mean (canonical) = "
          f"{pcc_m_per_slide_mean:.4f}")
    print(f"{'='*70}\n", flush=True)

    # Save the breakdown alongside the per-head JSONs for downstream tooling
    breakdown_path = fold_dir / "module_breakdown.json"
    with open(breakdown_path, "w") as f:
        json.dump({
            "fold": fold_idx,
            "backbone": "stnet_hydra",
            "registry_hash": registry["sha256"],
            "n_val_slides": len(sids_sorted),
            "metric": "per-slide scipy.stats.pearsonr, averaged across slides",
            "modules": module_breakdown,
            "weighted_full_pcc": weighted_full_pcc,
            "pcc_m_per_slide_mean": pcc_m_per_slide_mean,
            "best_val_pcc_m_during_training": best_pcc,
        }, f, indent=2)

    # CP3: per-head + full result files (now using canonical per-slide PCC)
    for k in range(registry["K"]):
        pcc_module = {
            g: pcc_per_gene_full[g] for g in registry["module_to_genes"][str(k)]
        }
        save_head_results(
            pcc_per_gene=pcc_module,
            module_id=k,
            registry=registry,
            fold=fold_idx,
            backbone="stnet_hydra",
            path=fold_dir / f"head_{k}.json",
        )

    save_full_results(
        pcc_per_gene=pcc_per_gene_full,
        registry=registry,
        fold=fold_idx,
        backbone="stnet_hydra",
        path=fold_dir / "full.json",
        extra={
            "best_val_pcc_m_per_slide_mean": best_pcc,
            "pcc_m_per_slide_mean_final": pcc_m_per_slide_mean,
            "weighted_full_pcc": weighted_full_pcc,
        },
    )
    print(
        f"Fold {fold_idx} done. best_val_pcc(per-slide mean during training)="
        f"{best_pcc:.4f} | final per-slide PCC(M)={pcc_m_per_slide_mean:.4f} "
        f"| weighted_full_pcc={weighted_full_pcc:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train HydraSTNet")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--registry", type=str,
        default="results/ablation/kmeans_y_elbow/fold_{F}/modules_fold{F}.json",
        help="Path to per-fold module registry. Use {F} as fold placeholder.",
    )
    parser.add_argument("--fold", type=int, default=-1,
                        help="Run a single fold; -1 = all folds.")
    parser.add_argument("--output_dir", type=str,
                        default="results/baselines/stnet_hydra")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    n_folds = config.get("n_folds", 4)
    fold_range = [args.fold] if args.fold >= 0 else list(range(n_folds))
    if args.fold >= 0 and not (0 <= args.fold < n_folds):
        raise ValueError(f"--fold {args.fold} out of range [0, {n_folds})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_idx in fold_range:
        registry_path = _registry_path_for_fold(args.registry, fold_idx)
        if not registry_path.exists():
            raise FileNotFoundError(
                f"Module registry not found: {registry_path}\n"
                f"Build it first: python scripts/build_module_registry.py "
                f"--fold {fold_idx} --K 7"
            )
        registry = load_registry(registry_path)  # CP1: SHA-256 verified here
        print(
            f"\n=== Fold {fold_idx} | registry={registry_path} | "
            f"sha256={registry['sha256'][:12]}... ==="
        )

        # Resume guard — skip if fold already finished.
        full_path = output_dir / f"fold_{fold_idx}" / "full.json"
        if full_path.exists() and not args.dry_run:
            print(f"Fold {fold_idx}: full.json exists — skipping.")
            continue

        _train_fold(
            fold_idx=fold_idx,
            config=config,
            registry=registry,
            output_dir=output_dir,
            device=device,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
