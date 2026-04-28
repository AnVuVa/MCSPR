"""MERGE end-to-end runner (CNN -> features -> GNN -> per-slide preds).

Pure baseline: no MCSPR loss path. The MCSPR-extended runner lives in
src/experiments/merge_mcspr/run_merge_mcspr.py and reuses this module's
preprocess + model + graph wiring while overriding the GNN train step.

Bug fixes vs upstream MERGE:
  * All json.dump calls go through `recursively_serialize` so np.float32
    metrics no longer crash the wrapper at the end of training.
  * `gnn_test_save` here writes preds AND model checkpoint, then the
    metrics file. Previous order risked losing the checkpoint when metrics
    serialization failed.

Usage:
  python src/experiments/run_merge.py \
      --config configs/merge_her2st_4fold.yaml --fold 0 --device 0 --mode all
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
import torch.optim as optim
import yaml
from scipy.stats import pearsonr
from torch.optim import lr_scheduler
from torch_geometric.utils import dropout_edge  # noqa: F401  (registered for GAT)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.merge_graph import graph_construction
from src.data.merge_loaders import preprocess_data
from src.models.merge import CNN_Predictor, GATNet


# ============================================================================
# Small JSON-safe utility (kept here to avoid a one-liner import file).
# ============================================================================
def _safe_serialize(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, list):
        return [_safe_serialize(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    return obj


# ============================================================================
# CNN stage
# ============================================================================
def _train_cnn(dataloaders, model, criterion, optimizer, scheduler,
               dataset_sizes, num_epochs, device):
    since = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}\n----------")
        model.train()
        running_mse, running_mae = 0.0, 0.0
        preds, ground_truths = [], []
        for batch in dataloaders["train"]:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            squeezed_labels = labels.squeeze()
            if squeezed_labels.dim() == 1:
                squeezed_labels = squeezed_labels.unsqueeze(0)
            ground_truths.append(squeezed_labels)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs).type(labels.dtype).view_as(labels)
                squeezed_outputs = outputs.squeeze()
                if squeezed_outputs.dim() == 1:
                    squeezed_outputs = squeezed_outputs.unsqueeze(0)
                preds.append(squeezed_outputs)
                mse = criterion(outputs, labels)
                mae = F.l1_loss(outputs, labels)
                loss = mse + 1.0 * mae
                loss.backward()
                optimizer.step()
            running_mse += mse.item() * inputs.size(0)
            running_mae += mae.item() * inputs.size(0)

        preds = torch.cat(preds, dim=0)
        ground_truths = torch.cat(ground_truths, dim=0)
        r = []
        for g in range(ground_truths.shape[1]):
            r.append(pearsonr(
                preds[:, g].cpu().detach(), ground_truths[:, g].cpu().detach(),
            )[0])
        scheduler.step()
        epoch_mse = running_mse / dataset_sizes["train"]
        epoch_mae = running_mae / dataset_sizes["train"]
        epoch_corr = float(np.mean(r))
        print(f"Training MSE: {epoch_mse:.4f}, MAE: {epoch_mae:.4f}, "
              f"Corr: {epoch_corr:.4f}")

    # Final val pass
    model.eval()
    running_mse, running_mae = 0.0, 0.0
    preds, ground_truths = [], []
    for batch in dataloaders["val"]:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        squeezed_labels = labels.squeeze()
        if squeezed_labels.dim() == 1:
            squeezed_labels = squeezed_labels.unsqueeze(0)
        ground_truths.append(squeezed_labels)
        with torch.set_grad_enabled(False):
            outputs = model(inputs).type(labels.dtype).view_as(labels)
            squeezed_outputs = outputs.squeeze()
            if squeezed_outputs.dim() == 1:
                squeezed_outputs = squeezed_outputs.unsqueeze(0)
            preds.append(squeezed_outputs)
            mse = F.l1_loss(outputs, labels)
            mae = F.l1_loss(outputs, labels)
        running_mse += mse.item() * inputs.size(0)
        running_mae += mae.item() * inputs.size(0)
    preds = torch.cat(preds, dim=0)
    ground_truths = torch.cat(ground_truths, dim=0)
    r = []
    for g in range(ground_truths.shape[1]):
        r.append(pearsonr(
            preds[:, g].cpu().detach(), ground_truths[:, g].cpu().detach(),
        )[0])
    elapsed = time.time() - since
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    val_mse = running_mse / dataset_sizes["val"]
    val_mae = running_mae / dataset_sizes["val"]
    val_corr = float(np.mean(r))
    print(f"Val MSE: {val_mse:.4f}\nVal MAE: {val_mae:.4f}\nVal Corr: {val_corr:.4f}")
    return model, val_mse, val_mae, val_corr


def _generate_features(data, model, config):
    """Run the (frozen) CNN with its FC head removed to produce 256-d
    embeddings per spot, stored in data['patch_embeddings']."""
    device = config["device"]
    model.to(device).eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    data["patch_embeddings"] = []
    with torch.no_grad():
        for i in tqdm(range(len(data["slides"])), desc="Generating features"):
            patches = torch.load(data["patch_files"][i], map_location="cpu")
            features = []
            for patch in patches:
                patch = patch.to(device)
                features.append(feature_extractor(patch).detach().cpu().numpy())
            data["patch_embeddings"].append(np.array(features))
            del patches
    return data


def cnn_block(data, dataloaders, dataset_sizes, config, run_idx: int = 0):
    """Train CNN, save checkpoint + metrics, then generate per-spot features."""
    plot_path = os.path.join(config["output_dir"], str(run_idx), "cnn") + os.sep
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    cnn_ckpt = os.path.join(plot_path, "model_state_dict.pt")

    if os.path.exists(cnn_ckpt) and config.get("resume_cnn", True):
        print(f"[CNN] Found checkpoint at {cnn_ckpt} — skipping CNN training")
        model = CNN_Predictor(num_genes=data["num_genes"], config=config)
        model.load_state_dict(torch.load(cnn_ckpt, map_location=config["device"]))
    else:
        model = CNN_Predictor(num_genes=data["num_genes"], config=config)
        criterion = F.mse_loss
        cnn_cfg = config["CNN"]
        optimizer = optim.Adam(
            model.parameters(),
            lr=cnn_cfg["optimizer"]["lr"],
            weight_decay=cnn_cfg["optimizer"]["weight_decay"],
        )
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cnn_cfg["scheduler"]["step_size"],
            gamma=cnn_cfg["scheduler"]["gamma"],
        )
        model, val_mse, val_mae, val_corr = _train_cnn(
            dataloaders, model, criterion, optimizer, scheduler,
            dataset_sizes, cnn_cfg["epochs"], device=config["device"],
        )
        torch.save(model.cpu().state_dict(), cnn_ckpt)
        with open(os.path.join(plot_path, "metrics.json"), "w") as f:
            json.dump(
                _safe_serialize({"mse": val_mse, "mae": val_mae, "corr": val_corr}),
                f,
            )

    return _generate_features(data, model, config)


# ============================================================================
# GNN stage
# ============================================================================
def _gnn_train_epoch(gnn, dataloader, optimizer, config):
    """Pure baseline: MSE loss only (Loss.l1 / Loss.grad / Loss.mcspr ignored
    here — those live in the merge_mcspr corner)."""
    gnn.train()
    train_mse, train_corr = [], []
    for batch in dataloader:
        slide_index, edge_indices, labels, patch_embeddings, positions = batch
        labels = labels.squeeze()
        edge_indices = edge_indices.squeeze()
        patch_embeddings = patch_embeddings.squeeze()

        with torch.set_grad_enabled(True):
            output = gnn(patch_embeddings, edge_indices)
        output = output.type(labels.dtype).view_as(labels)

        loss = F.mse_loss(output, labels)
        # Per-gene PCC for monitoring.
        out_t = output.T.detach().cpu()
        lbl_t = labels.T.detach().cpu()
        corr = []
        for g in range(lbl_t.shape[0]):
            corr.append(pearsonr(out_t[g], lbl_t[g])[0])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_mse.append(loss.item())
        train_corr.append(float(np.nanmean(corr)))
    return float(np.mean(train_mse)), float(np.mean(train_corr))


def _gnn_test(gnn, dataloader, num_genes):
    gnn.eval()
    test_mse, test_mae, test_corr = [], [], []
    for batch in dataloader:
        slide_index, edge_indices, labels, patch_embeddings, positions = batch
        labels = labels.squeeze()
        edge_indices = edge_indices.squeeze()
        patch_embeddings = patch_embeddings.squeeze()
        with torch.no_grad():
            output = gnn(patch_embeddings, edge_indices)
        output = output.type(labels.dtype).view_as(labels)
        test_mse.append(F.mse_loss(output, labels).item())
        test_mae.append(F.l1_loss(output, labels).item())
        out_t = output.T.detach().cpu()
        lbl_t = labels.T.detach().cpu()
        corr = []
        for g in range(num_genes):
            corr.append(pearsonr(out_t[g], lbl_t[g])[0])
        test_corr.append(float(np.nanmean(corr)))
    return float(np.mean(test_mse)), float(np.mean(test_mae)), float(np.mean(test_corr))


def _gnn_save_preds_and_metrics(plot_path, gnn, dataloader, data, config):
    """Iterate val slides, save per-slide preds (with zero-row backfill so
    canonical_eval --baseline merge can index by spot), then write metrics."""
    gnn.eval()
    base_dir = "/".join(config["output_dir"].split("/")[:-1]) \
        if len(config["output_dir"].split("/")) > 1 else config["output_dir"]
    preds_dir = os.path.join(base_dir, "preds")
    Path(preds_dir).mkdir(parents=True, exist_ok=True)

    test_mse, test_mae, test_corr = [], [], []
    for batch in dataloader:
        slide_index, edge_indices, labels, patch_embeddings, _ = batch
        labels = labels.squeeze()
        edge_indices = edge_indices.squeeze()
        patch_embeddings = patch_embeddings.squeeze()
        with torch.no_grad():
            output = gnn(patch_embeddings, edge_indices)
        idx = slide_index.item()
        slide = data["slides"][idx]
        np.save(f"{preds_dir}/{slide}.npy", output.cpu().detach().numpy())

        output = output.type(labels.dtype).view_as(labels)
        test_mse.append(F.mse_loss(output, labels).item())
        test_mae.append(F.l1_loss(output, labels).item())
        out_t = output.T.detach().cpu()
        lbl_t = labels.T.detach().cpu()
        per_gene_corr = []
        for g in range(data["num_genes"]):
            per_gene_corr.append(pearsonr(out_t[g], lbl_t[g])[0])
        test_corr.append(float(np.nanmean(per_gene_corr)))

    results = {
        "mse": float(np.mean(test_mse)),
        "mae": float(np.mean(test_mae)),
        "corr": float(np.mean(test_corr)),
    }
    with open(f"{plot_path}/results.json", "w") as f:
        json.dump(_safe_serialize(results), f)


def gnn_block(data, dataloaders, config, run_idx: int = 0):
    plot_path = os.path.join(config["output_dir"], str(run_idx), "gnn") + os.sep
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    gnn_ckpt = os.path.join(plot_path, "model_state_dict.pt")

    gnn = GATNet(
        num_genes=data["num_genes"],
        num_heads=config["GNN"]["attn_heads"],
        drop_edge=config["GNN"]["drop_edge"],
    ).to(config["device"])

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
    for i in range(epochs + 1):
        train_mse, train_corr = _gnn_train_epoch(gnn, dataloaders["train"], optimizer, config)
        if i % 40 == 0:
            if scheduler is not None:
                scheduler.step()
            test_mse, test_mae, test_corr = _gnn_test(
                gnn, dataloaders["val"], data["num_genes"],
            )
            print(f"Epoch: {i}, Train MSE: {train_mse:.4f}, Train Corr: {train_corr:.4f}")
            print(f"Epoch: {i}, Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}, "
                  f"Test Corr: {test_corr:.4f}")

    # Save preds + results.json BEFORE the model checkpoint so a serialization
    # failure can't cost the model. (Both are JSON-safe via _safe_serialize.)
    _gnn_save_preds_and_metrics(plot_path, gnn, dataloaders["val"], data, config)
    torch.save(gnn.cpu().state_dict(), gnn_ckpt)


# ============================================================================
# Entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="MERGE baseline (CNN+GNN)")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output dir for THIS fold (e.g. results/.../fold_0)")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-f", "--fold", type=int, default=0)
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["cnn_train", "gnn_train", "all"],
        help="cnn_train: stage 1 only. gnn_train: stage 2 only "
             "(requires existing CNN checkpoint). all: run both stages.",
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
        # Stringify torch.device for round-trip cleanliness.
        cfg_dump = {**config, "device": str(config["device"])}
        yaml.dump(cfg_dump, f)

    data, _, dataloaders, dataset_sizes = preprocess_data(config)

    if args.mode in ("cnn_train", "all"):
        data = cnn_block(data, dataloaders, dataset_sizes, config)
    else:
        # gnn_train only: still need features → load CNN ckpt and generate
        cnn_ckpt = os.path.join(args.output, "0", "cnn", "model_state_dict.pt")
        if not os.path.exists(cnn_ckpt):
            raise FileNotFoundError(f"GNN-only mode needs CNN ckpt at {cnn_ckpt}")
        model = CNN_Predictor(num_genes=data["num_genes"], config=config)
        model.load_state_dict(torch.load(cnn_ckpt, map_location=config["device"]))
        data = _generate_features(data, model, config)

    if args.mode in ("gnn_train", "all"):
        graph_dataloaders = graph_construction(data, config)
        gnn_block(data, graph_dataloaders, config)


if __name__ == "__main__":
    main()
