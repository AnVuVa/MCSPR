"""MERGE graph construction (ported from MERGE/utils/graph.py).

For each slide:
  1. Build spatial 8-NN graph over the in-tissue spot pixel coordinates.
  2. (Optional, default ON) Build a hierarchical graph by adding edges:
     - First, KMeans on (x, y) -> spatial clusters; connect each cluster's
       centroid spot to the rest of its cluster, and the centroids to each
       other.
     - Then, KMeans on patch embeddings -> feature clusters; do the same,
       skipping spots that are already cluster centroids in the spatial pass.

Each per-slide adjacency is converted to a torch_geometric edge_index. The
resulting GraphDataset yields (slide_idx, edge_index, labels, embeddings,
positions) tuples, one per slide.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import Dataset
from torch_geometric.utils import from_scipy_sparse_matrix
from tqdm import tqdm


class GraphDataset(Dataset):
    def __init__(self, adj, data, config, train: bool = True):
        self.slide_indices = []
        self.edge_indices = []
        self.labels = []
        self.patch_embeddings = []
        self.positions = []

        for idx, _ in enumerate(data["slides"]):
            if train and not data["train_slides"][idx]:
                continue
            if (not train) and data["train_slides"][idx]:
                continue

            ei = from_scipy_sparse_matrix(sp.coo_matrix(adj[idx]))[0]
            ei = ei.to(config["device"])
            lbl = torch.tensor(data["counts"][idx]).to(config["device"])
            emb = torch.tensor(data["patch_embeddings"][idx]).to(config["device"])
            tp = data["tissue_positions"][idx].reset_index()
            pos = torch.tensor(
                tp[["array_row", "array_col"]].values, dtype=torch.long,
            ).to(config["device"])

            self.slide_indices.append(idx)
            self.edge_indices.append(ei)
            self.labels.append(lbl)
            self.patch_embeddings.append(emb)
            self.positions.append(pos)

    def __len__(self) -> int:
        return len(self.edge_indices)

    def __getitem__(self, idx):
        return (
            self.slide_indices[idx],
            self.edge_indices[idx],
            self.labels[idx],
            self.patch_embeddings[idx],
            self.positions[idx],
        )


def _update_adj(adj, cluster_labels, patch_embeddings, old_labels=None):
    """For each cluster: pick the spot whose embedding is closest to the
    cluster centroid, connect every other cluster spot to it, then connect
    all per-cluster centroid spots to each other. Mark centroid spots by
    flipping the sign of their cluster label so a later feature-cluster pass
    can skip them via `old_labels`."""
    unique_cluster_labels = np.unique(cluster_labels)
    centroid_spots = []

    for cluster_label in unique_cluster_labels:
        cluster_spots = np.where(cluster_labels == cluster_label)[0]
        if unique_cluster_labels.shape[0] == 1:
            nearest_spot_idx = np.random.randint(0, len(cluster_spots))
            nearest_spot = cluster_spots[nearest_spot_idx]
        cluster_centroid = patch_embeddings[cluster_spots].mean(axis=0)
        nearest_spot_idx = np.argmin(np.linalg.norm(
            patch_embeddings[cluster_spots] - cluster_centroid, axis=1,
        ))
        nearest_spot = cluster_spots[nearest_spot_idx]

        for j in range(len(cluster_spots)):
            if cluster_spots[j] != nearest_spot:
                adj[cluster_spots[j], nearest_spot] = 1
                adj[nearest_spot, cluster_spots[j]] = 1

        centroid_spots.append(nearest_spot_idx)
        cluster_labels[cluster_spots[nearest_spot_idx]] *= -1
        if cluster_labels[cluster_spots[nearest_spot_idx]] == 0:
            cluster_labels[cluster_spots[nearest_spot_idx]] = -(
                len(unique_cluster_labels)
            )

    if old_labels is not None:
        for j, old_label in enumerate(old_labels):
            if old_label < 0:
                centroid_spots.append(j)

    centroid_spots = list(set(centroid_spots))
    for j in range(len(centroid_spots)):
        for k in range(j + 1, len(centroid_spots)):
            adj[centroid_spots[j], centroid_spots[k]] = 1
            adj[centroid_spots[k], centroid_spots[j]] = 1

    return adj, cluster_labels


def _build_one_hop_graph(data) -> list:
    adj = [torch.zeros(n, n) for n in data["spotnum"]]
    for slide_idx, _ in enumerate(data["slides"]):
        # Note (matches upstream): kneighbors_graph is built using
        # ['pxl_col_in_fullres', 'pxl_col_in_fullres'] — the column appears
        # twice. Faithful to upstream MERGE.
        coords = data["tissue_positions"][slide_idx].reset_index()[
            ["pxl_col_in_fullres", "pxl_col_in_fullres"]
        ].to_numpy()
        tmp_adj = kneighbors_graph(coords, mode="connectivity", n_neighbors=8).toarray()
        tmp_adj = (tmp_adj + tmp_adj.T) > 0
        adj[slide_idx] = torch.tensor(tmp_adj)
        adj[slide_idx].fill_diagonal_(1)
    return adj


def _build_hierarchical_graph(data, config, adj):
    feature_cluster_path = os.path.join(config["output_dir"], "clusters", "feature")
    spatial_cluster_path = os.path.join(config["output_dir"], "clusters", "spatial")
    Path(feature_cluster_path).mkdir(parents=True, exist_ok=True)
    Path(spatial_cluster_path).mkdir(parents=True, exist_ok=True)

    n_spatial = config["GNN"]["clusters"]["spatial"]
    n_feature = config["GNN"]["clusters"]["feature"]

    for i in tqdm(range(len(data["slides"])), desc="Hierarchical graph"):
        emb = np.array(data["patch_embeddings"][i])
        tp = data["tissue_positions"][i].reset_index()
        coords = np.stack(
            [tp["pxl_col_in_fullres"].values, tp["pxl_row_in_fullres"].values],
            axis=1,
        )

        spatial_clusterer = KMeans(n_clusters=n_spatial, max_iter=1000, n_init=10)
        spatial_clusterer.fit(coords)
        spatial_labels = spatial_clusterer.predict(coords)
        adj[i], spatial_labels = _update_adj(adj[i], spatial_labels, emb, None)

        pd.DataFrame({"cluster_labels": spatial_labels}).to_csv(
            f"{spatial_cluster_path}/{data['slides'][i]}.csv", index=False,
        )
        spatial_keep = spatial_labels.copy()

        feature_clusterer = KMeans(n_clusters=n_feature, max_iter=1000, n_init=10)
        feature_clusterer.fit(emb)
        feature_labels = feature_clusterer.predict(emb)
        adj[i], feature_labels = _update_adj(
            adj[i], feature_labels, emb, spatial_keep,
        )
        pd.DataFrame({"cluster_labels": feature_labels}).to_csv(
            f"{feature_cluster_path}/{data['slides'][i]}.csv", index=False,
        )
    return adj


def graph_construction(data, config):
    """Builds spatial + (optionally) hierarchical edges per slide and wraps
    them in train/val DataLoaders, batch_size=1 (one slide per step)."""
    print("Building the spatial graph...")
    adj = _build_one_hop_graph(data)
    print("Building the spatial graph done.")

    if config["GNN"].get("hierarchical", True):
        print("Building the hierarchical graph...")
        adj = _build_hierarchical_graph(data, config, adj)
        print("Building the hierarchical graph done.")

    train_dataset = GraphDataset(adj, data, config, train=True)
    val_dataset = GraphDataset(adj, data, config, train=False)
    return {
        "train": torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=0,
        ),
        "val": torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=0,
        ),
    }
