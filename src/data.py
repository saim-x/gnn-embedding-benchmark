"""
How Embeddings Shape Graph Neural Networks: Classical vs Quantum-Oriented Node Representations — Data

Paper: https://arxiv.org/abs/2604.15273v1
Implements: Dataset loading and preprocessing for the benchmark protocol.

Section references:
  §A. Experimental Setup — datasets and stratified 80/10/10 split
  §I. INTRODUCTION — base node representation uses degree one-hot + Laplacian PE (x || pe)

Data format expected in {data_dir}/{dataset_name}.pt:
  A list of dicts, each containing:
    - "adj": torch.Tensor of shape (n, n), binary/dense adjacency
    - "label": int class id
    - "node_features": optional torch.Tensor of shape (n, f_raw)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, TypedDict

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import build_degree_onehot, laplacian_positional_encoding, stratified_split_indices


class GraphSample(TypedDict):
    node_features: torch.Tensor
    adj: torch.Tensor
    label: torch.Tensor


class BenchmarkGraphDataset(Dataset):
    """Graph dataset wrapper for paper-style benchmark splits."""

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        split: str,
        split_train: float,
        split_val: float,
        split_test: float,
        seed: int,
        pe_k: int,
        use_degree_onehot: bool,
        degree_onehot_dim: int,
        max_graphs: int | None = None,
    ) -> None:
        self.split = split
        self.pe_k = pe_k
        self.use_degree_onehot = use_degree_onehot
        self.degree_onehot_dim = degree_onehot_dim

        dataset_path = Path(data_dir) / f"{dataset_name}.pt"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}\n"
                "Create this file as a list of graph dicts with keys: adj, label, optional node_features."
            )

        # Dataset files are user-supplied graph lists, not state dict checkpoints.
        raw_data = torch.load(dataset_path, map_location="cpu", weights_only=False)
        if not isinstance(raw_data, list) or len(raw_data) == 0:
            raise ValueError(f"Expected non-empty list in {dataset_path}")
        if max_graphs is not None:
            raw_data = raw_data[:max_graphs]

        labels = torch.tensor([int(item["label"]) for item in raw_data], dtype=torch.long)  # (num_graphs,)
        train_idx, val_idx, test_idx = stratified_split_indices(
            labels=labels,
            train_ratio=split_train,
            val_ratio=split_val,
            test_ratio=split_test,
            seed=seed,
        )

        split_to_indices = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }
        if split not in split_to_indices:
            raise ValueError(f"split must be one of train/val/test, got: {split}")

        selected_indices = split_to_indices[split]
        self.samples = [raw_data[i] for i in selected_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphSample:
        sample = self.samples[idx]
        adj = sample["adj"].float()  # (n, n)
        if adj.dim() != 2 or adj.size(0) != adj.size(1):
            raise ValueError("adj must be a square tensor with shape (n, n)")
        n = adj.size(0)

        if "node_features" in sample and sample["node_features"] is not None:
            base_x = sample["node_features"].float()  # (n, f_raw)
        else:
            if not self.use_degree_onehot:
                raise ValueError("node_features missing and degree one-hot disabled.")
            base_x = build_degree_onehot(adj=adj, max_degree_bin=self.degree_onehot_dim)  # (n, degree_bins)

        # §I. INTRODUCTION — concatenate base features and Laplacian PE: (x || pe)
        pe = laplacian_positional_encoding(adj=adj, k=self.pe_k)  # (n, pe_k)
        node_features = torch.cat([base_x, pe], dim=-1)  # (n, f_raw) + (n, pe_k) -> (n, f_total)

        out: GraphSample = {
            "node_features": node_features,
            "adj": adj,
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
        }
        return out


def collate_graph_batch(batch: List[GraphSample]) -> Dict[str, torch.Tensor]:
    """Pad variable-size graphs into a dense batch."""
    batch_size = len(batch)
    max_nodes = max(item["adj"].size(0) for item in batch)
    feature_dim = batch[0]["node_features"].size(-1)

    node_features = torch.zeros(batch_size, max_nodes, feature_dim, dtype=torch.float32)  # (b, n_max, f)
    adjacency = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.float32)  # (b, n_max, n_max)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)  # (b, n_max)
    labels = torch.zeros(batch_size, dtype=torch.long)  # (b,)

    for i, item in enumerate(batch):
        n = item["adj"].size(0)
        node_features[i, :n] = item["node_features"]  # (n, f) -> (b, n_max, f) slice
        adjacency[i, :n, :n] = item["adj"]  # (n, n) -> (b, n_max, n_max) slice
        node_mask[i, :n] = True  # valid nodes
        labels[i] = item["label"]

    return {
        "node_features": node_features,
        "adj": adjacency,
        "node_mask": node_mask,
        "labels": labels,
    }


def build_dataloader(config: dict, split: str) -> DataLoader:
    """Build DataLoader from base.yaml config."""
    data_cfg = config["data"]
    train_cfg = config["training"]

    max_graphs = data_cfg.get("qm9_max_graphs") if data_cfg.get("dataset_name", "").upper() == "QM9" else None
    dataset = BenchmarkGraphDataset(
        data_dir=data_cfg["data_dir"],
        dataset_name=data_cfg["dataset_name"],
        split=split,
        split_train=float(data_cfg["split_train"]),
        split_val=float(data_cfg["split_val"]),
        split_test=float(data_cfg["split_test"]),
        seed=int(train_cfg["seed"]),
        pe_k=int(config["model"]["pe_k"]),
        use_degree_onehot=bool(data_cfg["use_degree_onehot"]),
        degree_onehot_dim=int(data_cfg["degree_onehot_dim"]),
        max_graphs=max_graphs,
    )

    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=(split == "train"),
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate_graph_batch,
        drop_last=False,
    )

