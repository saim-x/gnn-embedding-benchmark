"""
How Embeddings Shape Graph Neural Networks: Classical vs Quantum-Oriented Node Representations — Utilities

Paper: https://arxiv.org/abs/2604.15273v1
Implements: Shared helpers for graph preprocessing, pooling, and reproducibility.

Section references:
  §I. INTRODUCTION — base node representation (x || pe) with degree features + Laplacian PE
  §A. Experimental Setup — PE dimension k=8, split seed=7
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """§A. Experimental Setup — seed control for deterministic splits and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_self_loops(adj: torch.Tensor) -> torch.Tensor:
    """Add identity self-loops.

    Args:
        adj: Adjacency tensor — shape: (batch, n, n)
    """
    eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype).unsqueeze(0)  # (n, n) -> (1, n, n)
    adj_with_loops = torch.clamp(adj + eye, max=1.0)  # (batch, n, n) + (1, n, n) -> (batch, n, n)
    return adj_with_loops


def normalize_adjacency(adj: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Symmetric adjacency normalization D^{-1/2} A D^{-1/2}.

    Args:
        adj: Dense adjacency matrix — shape: (batch, n, n)
    """
    adj_loop = add_self_loops(adj)  # (batch, n, n) -> (batch, n, n)
    deg = adj_loop.sum(dim=-1)  # (batch, n, n) -> (batch, n)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)  # (batch, n) -> (batch, n)
    left = deg_inv_sqrt.unsqueeze(-1)  # (batch, n) -> (batch, n, 1)
    right = deg_inv_sqrt.unsqueeze(-2)  # (batch, n) -> (batch, 1, n)
    norm_adj = left * adj_loop * right  # (batch, n, 1)*(batch, n, n)*(batch, 1, n) -> (batch, n, n)
    return norm_adj


def masked_mean_pool(node_states: torch.Tensor, node_mask: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Global mean pooling with masking.

    Args:
        node_states: Node embeddings — shape: (batch, n, d)
        node_mask: Node-valid mask — shape: (batch, n)
    """
    mask = node_mask.unsqueeze(-1).to(node_states.dtype)  # (batch, n) -> (batch, n, 1)
    summed = (node_states * mask).sum(dim=1)  # (batch, n, d) -> (batch, d)
    counts = mask.sum(dim=1).clamp_min(eps)  # (batch, n, 1) -> (batch, 1)
    pooled = summed / counts  # (batch, d) / (batch, 1) -> (batch, d)
    return pooled


def build_degree_onehot(adj: torch.Tensor, max_degree_bin: int) -> torch.Tensor:
    """§I. INTRODUCTION — degree-derived structural feature vector.

    Args:
        adj: Adjacency matrix for one graph — shape: (n, n)
        max_degree_bin: Size of one-hot feature vector.
    """
    degree = adj.sum(dim=-1).long()  # (n, n) -> (n,)
    clipped = torch.clamp(degree, min=0, max=max_degree_bin - 1)  # (n,) -> (n,)
    onehot = torch.nn.functional.one_hot(clipped, num_classes=max_degree_bin).float()  # (n,) -> (n, max_degree_bin)
    return onehot


def laplacian_positional_encoding(adj: torch.Tensor, k: int, eps: float = 1.0e-8) -> torch.Tensor:
    """§I. INTRODUCTION, §A. Experimental Setup — Laplacian eigenvector PE.

    Args:
        adj: Adjacency matrix for one graph — shape: (n, n)
        k: Number of non-trivial eigenvectors to use.
    """
    n = adj.size(0)
    deg = adj.sum(dim=-1)  # (n, n) -> (n,)
    inv_sqrt_deg = torch.pow(deg + eps, -0.5)  # (n,) -> (n,)
    d_inv_sqrt = torch.diag(inv_sqrt_deg)  # (n,) -> (n, n)
    identity = torch.eye(n, device=adj.device, dtype=adj.dtype)  # (n, n)
    lap = identity - (d_inv_sqrt @ adj @ d_inv_sqrt)  # (n, n) -> (n, n)
    evals, evecs = torch.linalg.eigh(lap)  # (n, n) -> (n,), (n, n)
    usable = min(k, max(n - 1, 0))
    if usable == 0:
        return torch.zeros(n, k, device=adj.device, dtype=adj.dtype)  # -> (n, k)
    pe = evecs[:, 1 : usable + 1]  # (n, n) -> (n, usable)
    if usable < k:
        pad = torch.zeros(n, k - usable, device=adj.device, dtype=adj.dtype)  # -> (n, k-usable)
        pe = torch.cat([pe, pad], dim=-1)  # (n, usable)+(n, k-usable) -> (n, k)
    return pe


def stratified_split_indices(
    labels: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """§A. Experimental Setup — stratified 80/10/10 split with fixed seed.

    Args:
        labels: Class labels — shape: (num_graphs,)
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for cls in np.unique(labels_np):
        cls_indices = np.where(labels_np == cls)[0]
        rng.shuffle(cls_indices)
        n_total = len(cls_indices)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        n_train = min(n_train, n_total)
        n_val = min(n_val, n_total - n_train)
        n_test_start = n_train + n_val

        train_idx.extend(cls_indices[:n_train].tolist())
        val_idx.extend(cls_indices[n_train:n_test_start].tolist())
        test_idx.extend(cls_indices[n_test_start:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx

