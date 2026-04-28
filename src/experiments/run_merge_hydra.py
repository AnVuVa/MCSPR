"""HydraMERGE end-to-end runner (CNN + HydraGATNet).

Surgical change from src/experiments/run_merge.py:
  * Stage 1 CNN unchanged — re-uses the baseline CNN training and patch-
    embedding generation. The CNN checkpoint is interchangeable with the
    baseline's so we can warm-start from an existing fold's CNN if present.
  * Stage 2 GNN replaces GATNet with HydraGATNet and the single F.mse_loss
    with the hydra-weighted-by-module-size MSE on per-head outputs.

Eval/save:
  * Per-slide preds (full 300, reassembled) are saved to
    `results/baselines/merge_hydra/preds/{slide}.npy` so the existing
    canonical_eval(--baseline merge) can index them.
  * Per-fold full.json + module_breakdown.json mirror HydraTRIPLEX's layout.

Usage:
  python src/experiments/run_merge_hydra.py \
      --config configs/merge_hydra_her2st_4fold.yaml \
      --output results/baselines/merge_hydra/fold_0 \
      --registry results/ablation/kmeans_y_elbow/fold_{F}/modules_fold{F}.json \
      --device 0 --fold 0 --mode all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.merge_graph import graph_construction
from src.data.merge_loaders import preprocess_data
from src.experiments.run_merge import (
    _safe_serialize,
    cnn_block,
)
from src.models.merge_hydra import HydraGATNet
from src.training.hydra_helpers import (
    load_registry,
    save_full_results,
    save_head_results,
    verify_modules,
)


def _registry_path_for_fold(template: str, fold: int) -> Path:
    s = (
        template
        .replace("{F}", str(fold))
        .replace("{fold}", str(fold))
        .replace("FOLDID", str(fold))
    )
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


def _hydra_weighted_mse(per_head, y, idx_list):
    """Σ_k m_k · MSE_k / Σ_k m_k  ==  MSE on full reassembled (n_spots, 300).

    Returns (total, per_head_losses).
    """
    sse_total = 0.0
    n_elements = 0
    head_losses = []
    for k, idx in enumerate(idx_list):
        target_k = y[:, idx]
        mse_k = F.mse_loss(per_head[k], target_k)
        head_losses.append(mse_k)
        sse_total = sse_total + mse_k * per_head[k].numel()
        n_elements += per_head[k].numel()
    return sse_total / n_elements, head_losses


def _gnn_train_epoch(gnn, dataloader, optimizer, idx_list):
    gnn.train()
    train_mse, train_corr = [], []
    K = len(idx_list)
    epoch_per_head = [0.0] * K
    n_batches = 0
    for batch in dataloader:
        slide_index, edge_indices, labels, patch_embeddings, _ = batch
        labels = labels.squeeze()
        edge_indices = edge_indices.squeeze()
        patch_embeddings = patch_embeddings.squeeze()

        with torch.set_grad_enabled(True):
            preds = gnn(patch_embeddings, edge_indices)
        full = preds["fusion"].type(labels.dtype).view_as(labels)
        per_head = preds["fusion_per_head"]

        loss, head_losses = _hydra_weighted_mse(per_head, labels, idx_list)

        out_t = full.T.detach().cpu()
        lbl_t = labels.T.detach().cpu()
        corr = []
        for g in range(lbl_t.shape[0]):
            corr.append(pearsonr(out_t[g], lbl_t[g])[0])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_mse.append(loss.item())
        train_corr.append(float(np.nanmean(corr)))
        for k in range(K):
            epoch_per_head[k] += float(head_losses[k].item())
        n_batches += 1
    per_head_train = [epoch_per_head[k] / max(n_batches, 1) for k in range(K)]
    return float(np.mean(train_mse)), float(np.mean(train_corr)), per_head_train


def _gnn_eval(gnn, dataloader, num_genes, idx_list):
    """Per-slide full-300 PCC mean (canonical metric)."""
    gnn.eval()
    test_mse, test_mae, test_corr = [], [], []
    per_head_val = [0.0] * len(idx_list)
    n_batches = 0
    for batch in dataloader:
        slide_index, edge_indices, labels, patch_embeddings, _ = batch
        labels = labels.squeeze()
        edge_indices = edge_indices.squeeze()
        patch_embeddings = patch_embeddings.squeeze()
        with torch.no_grad():
            preds = gnn(patch_embeddings, edge_indices)
        full = preds["fusion"].type(labels.dtype).view_as(labels)
        test_mse.append(F.mse_loss(full, labels).item())
        test_mae.append(F.l1_loss(full, labels).item())
        out_t = full.T.detach().cpu()
        lbl_t = labels.T.detach().cpu()
        corr = []
        for g in range(num_genes):
            corr.append(pearsonr(out_t[g], lbl_t[g])[0])
        test_corr.append(float(np.nanmean(corr)))
        for k, idx in enumerate(idx_list):
            mse_k = F.mse_loss(preds["fusion_per_head"][k], labels[:, idx])
            per_head_val[k] += float(mse_k.item())
        n_batches += 1
    per_head_val = [v / max(n_batches, 1) for v in per_head_val]
    return (
        float(np.mean(test_mse)),
        float(np.mean(test_mae)),
        float(np.mean(test_corr)),
        per_head_val,
    )


def _gnn_save_preds_and_metrics(
    plot_path, gnn, dataloader, data, config, idx_list, registry, fold_idx,
):
    """Iterate val slides; save per-slide full-300 preds and per-fold result
    JSONs (full + module_breakdown + per-head)."""
    gnn.eval()
    base_dir = "/".join(config["output_dir"].split("/")[:-1]) \
        if len(config["output_dir"].split("/")) > 1 else config["output_dir"]
    preds_dir = os.path.join(base_dir, "preds")
    Path(preds_dir).mkdir(parents=True, exist_ok=True)

    test_mse, test_mae, test_corr = [], [], []
    per_slide_gene_pccs = []
    sids_sorted = []
    for batch in dataloader:
        slide_index, edge_indices, labels, patch_embeddings, _ = batch
        labels = labels.squeeze()
        edge_indices = edge_indices.squeeze()
        patch_embeddings = patch_embeddings.squeeze()
        with torch.no_grad():
            preds = gnn(patch_embeddings, edge_indices)
        full = preds["fusion"]
        idx = slide_index.item()
        slide = data["slides"][idx]
        np.save(f"{preds_dir}/{slide}.npy", full.cpu().detach().numpy())

        full_t = full.type(labels.dtype).view_as(labels)
        test_mse.append(F.mse_loss(full_t, labels).item())
        test_mae.append(F.l1_loss(full_t, labels).item())
        gene_pccs = _per_gene_pcc(
            full_t.cpu().numpy(), labels.cpu().numpy(),
        )
        test_corr.append(float(np.nanmean(gene_pccs)))
        per_slide_gene_pccs.append(gene_pccs)
        sids_sorted.append(slide)

    per_slide_gene_pccs = np.stack(per_slide_gene_pccs, axis=0)
    gene_pccs_mean = np.nanmean(per_slide_gene_pccs, axis=0)
    pcc_per_gene_full = {
        registry["gene_names_full"][i]: float(gene_pccs_mean[i])
        for i in range(registry["n_genes"])
    }
    pcc_m_per_slide_mean = float(np.mean(test_corr))

    # Per-fold legacy results.json (matches baseline run_merge layout).
    results = {
        "mse": float(np.mean(test_mse)),
        "mae": float(np.mean(test_mae)),
        "corr": pcc_m_per_slide_mean,
    }
    with open(f"{plot_path}/results.json", "w") as f:
        json.dump(_safe_serialize(results), f)

    # Per-module breakdown — mirrors HydraTRIPLEX.
    module_breakdown = []
    weighted_numerator = 0.0
    total_genes = 0
    for k in range(registry["K"]):
        gene_names_k = registry["module_to_genes"][str(k)]
        idx_k = registry["module_to_indices"][str(k)]
        pcc_k = gene_pccs_mean[np.array(idx_k, dtype=int)]
        m_k = len(gene_names_k)
        mean_k = float(np.nanmean(pcc_k))
        std_k = float(np.nanstd(pcc_k))
        weighted_numerator += mean_k * m_k
        total_genes += m_k
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

    fold_dir = Path(plot_path).parent.parent  # results/.../fold_{F}/0/gnn -> fold_{F}
    with open(fold_dir / "module_breakdown.json", "w") as f:
        json.dump({
            "fold": fold_idx,
            "backbone": "merge_hydra",
            "registry_hash": registry["sha256"],
            "n_val_slides": len(sids_sorted),
            "metric": "per-slide scipy.stats.pearsonr, averaged across slides",
            "modules": module_breakdown,
            "weighted_full_pcc": weighted_full_pcc,
            "pcc_m_per_slide_mean": pcc_m_per_slide_mean,
        }, f, indent=2)

    for k in range(registry["K"]):
        pcc_module = {
            g: pcc_per_gene_full[g] for g in registry["module_to_genes"][str(k)]
        }
        save_head_results(
            pcc_per_gene=pcc_module, module_id=k, registry=registry,
            fold=fold_idx, backbone="merge_hydra",
            path=fold_dir / f"head_{k}.json",
        )

    save_full_results(
        pcc_per_gene=pcc_per_gene_full, registry=registry, fold=fold_idx,
        backbone="merge_hydra", path=fold_dir / "full.json",
        extra={
            "pcc_m_per_slide_mean_final": pcc_m_per_slide_mean,
            "weighted_full_pcc": weighted_full_pcc,
        },
    )

    print(f"\nFOLD {fold_idx} | merge_hydra | "
          f"per-slide PCC(M)={pcc_m_per_slide_mean:.4f} | "
          f"weighted_full={weighted_full_pcc:.4f}")


def gnn_block(data, dataloaders, config, registry, idx_list, run_idx: int = 0):
    plot_path = os.path.join(config["output_dir"], str(run_idx), "gnn") + os.sep
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    gnn_ckpt = os.path.join(plot_path, "model_state_dict.pt")

    gnn = HydraGATNet(
        module_sizes=registry["module_sizes"],
        idx_list=idx_list,
        num_heads=config["GNN"]["attn_heads"],
        drop_edge=config["GNN"]["drop_edge"],
    ).to(config["device"])

    print(
        f"HydraMERGE | K={registry['K']} sizes={registry['module_sizes']} "
        f"params={sum(p.numel() for p in gnn.parameters()):,}"
    )

    optimizer = torch.optim.Adam(
        gnn.parameters(),
        lr=config["GNN"]["optimizer"]["lr"],
        weight_decay=config["GNN"]["optimizer"]["weight_decay"],
    )

    sched_cfg = config["GNN"]["scheduler"]
    if sched_cfg["type"] == "warmup":
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=sched_cfg["warmup_steps"],
            num_training_steps=config["GNN"]["epochs"],
        )
    else:
        scheduler = None

    epochs = config["GNN"]["epochs"]
    K = len(idx_list)
    for i in range(epochs + 1):
        train_mse, train_corr, per_head_train = _gnn_train_epoch(
            gnn, dataloaders["train"], optimizer, idx_list,
        )
        if i % 40 == 0:
            if scheduler is not None:
                scheduler.step()
            test_mse, test_mae, test_corr, per_head_val = _gnn_eval(
                gnn, dataloaders["val"], data["num_genes"], idx_list,
            )
            head_train_str = " ".join(
                f"h{k}={per_head_train[k]:.3f}" for k in range(K)
            )
            print(f"Epoch: {i}, Train MSE: {train_mse:.4f}, "
                  f"Train Corr: {train_corr:.4f}", flush=True)
            print(f"Epoch: {i}, Test MSE: {test_mse:.4f}, "
                  f"Test MAE: {test_mae:.4f}, Test Corr: {test_corr:.4f}",
                  flush=True)
            print(f"        train MSE/head: {head_train_str}", flush=True)

    _gnn_save_preds_and_metrics(
        plot_path, gnn, dataloaders["val"], data, config,
        idx_list, registry, config["Data"]["fold"],
    )
    torch.save(gnn.cpu().state_dict(), gnn_ckpt)


def main():
    parser = argparse.ArgumentParser(description="HydraMERGE (CNN + HydraGAT)")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output dir for THIS fold (e.g. results/.../fold_0)")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-f", "--fold", type=int, default=0)
    parser.add_argument(
        "--registry", type=str,
        default="results/ablation/kmeans_y_elbow/fold_{F}/modules_fold{F}.json",
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["cnn_train", "gnn_train", "all"],
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["output_dir"] = args.output
    config["Data"]["fold"] = args.fold
    config["mode"] = args.mode
    config["device"] = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    with open(f"{config['output_dir']}/config.yaml", "w") as f:
        cfg_dump = {**config, "device": str(config["device"])}
        yaml.dump(cfg_dump, f)

    registry_path = _registry_path_for_fold(args.registry, args.fold)
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Registry not found: {registry_path}\nBuild it via "
            f"scripts/build_module_registry.py --fold {args.fold} --K 7"
        )
    registry = load_registry(registry_path)
    print(f"Registry: K={registry['K']} sizes={registry['module_sizes']} "
          f"sha256={registry['sha256'][:12]}...")

    data, _, dataloaders, dataset_sizes = preprocess_data(config)

    # CP2: ensure dataset gene order matches the registry exactly. The MERGE
    # loader doesn't expose gene_names because counts are stored as plain
    # .npy in canonical SVG order, so we trust that here and verify the K
    # partition + module_sizes against the registry.
    idx_list = []
    seen = set()
    for k in range(registry["K"]):
        idx = sorted(int(i) for i in registry["module_to_indices"][str(k)])
        for i in idx:
            if i in seen:
                raise ValueError(f"Gene index {i} in multiple modules")
            seen.add(i)
        idx_list.append(idx)
    if sum(len(idx) for idx in idx_list) != registry["n_genes"]:
        raise ValueError("Module partition does not cover all genes")
    if [len(idx) for idx in idx_list] != registry["module_sizes"]:
        raise AssertionError(
            f"idx_list sizes {[len(i) for i in idx_list]} != "
            f"registry module_sizes {registry['module_sizes']}"
        )
    if registry.get("fold") != args.fold:
        raise ValueError(
            f"Registry fold mismatch: registry['fold']={registry.get('fold')} "
            f"!= --fold {args.fold}"
        )

    if args.mode in ("cnn_train", "all"):
        # CNN stage is byte-identical to the baseline (full-300 head).
        data = cnn_block(data, dataloaders, dataset_sizes, config)
    else:
        cnn_ckpt = os.path.join(args.output, "0", "cnn", "model_state_dict.pt")
        if not os.path.exists(cnn_ckpt):
            raise FileNotFoundError(f"GNN-only mode needs CNN ckpt at {cnn_ckpt}")
        from src.experiments.run_merge import _generate_features
        from src.models.merge import CNN_Predictor
        model = CNN_Predictor(num_genes=data["num_genes"], config=config)
        model.load_state_dict(torch.load(cnn_ckpt, map_location=config["device"]))
        data = _generate_features(data, model, config)

    if args.mode in ("gnn_train", "all"):
        graph_dataloaders = graph_construction(data, config)
        start = time.time()
        gnn_block(data, graph_dataloaders, config, registry, idx_list)
        print(f"GNN stage wall: {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
