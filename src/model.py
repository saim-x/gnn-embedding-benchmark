"""
How Embeddings Shape Graph Neural Networks: Classical vs Quantum-Oriented Node Representations — Model

Paper: https://arxiv.org/abs/2604.15273v1
Authors: Nouhaila Innan, Antonello Rosato, Alberto Marchisio, Muhammad Shafique
Year: 2026

Implements: Unified graph-classification model with interchangeable embedding modules under a fixed GIN backbone.

Section references:
  §I. INTRODUCTION — embedding families and benchmark intent
  §B. Graph classifier and training protocol — shared GIN classifier
  §A. Experimental Setup — fixed protocol hyperparameters
  Algorithm 1 — QuOp
  Algorithm 2 — QWalkVec
  Algorithm 3 — QPE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from src.utils import masked_mean_pool, normalize_adjacency


@dataclass
class ModelConfig:
    """All model hyperparameters.

    Values are from the paper unless marked [UNSPECIFIED].
    """

    # §A. Experimental Setup (Table I)
    embedding_dim: int = 32
    pe_k: int = 8
    gin_layers: int = 3
    gin_hidden_dim: int = 64
    mlp_head_hidden_dim: int = 64
    dropout: float = 0.2

    # [UNSPECIFIED] Default executable embedding choice.
    embedding_kind: str = "qwalkvec_trainable"

    # §A. Experimental Setup (Table I) — QWalkVec
    qwalk_steps: int = 32
    qwalk_wp: float = 0.5
    qwalk_wq: float = 4.0

    # §A. Experimental Setup (Table I) — QPE
    qpe_times: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    qpe_anchors: int = 8

    # [UNSPECIFIED] Algorithm 1 requires these values but paper does not fix them numerically.
    quop_hop_radius: int = 1
    quop_qubits: int = 5

    # [UNSPECIFIED] Projection MLP width for trainable variants.
    projection_hidden_dim: int = 64

    # [UNSPECIFIED] Dataset-dependent output classes.
    num_classes: int = 2

    # §A. Experimental Setup — reproducibility seed.
    seed: int = 7


class FixedProjection(nn.Module):
    """Shared fixed/trainable linear projection block."""

    def __init__(self, in_dim: int, out_dim: int, seed: int, trainable: bool) -> None:
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(seed)
        weight = torch.randn(in_dim, out_dim, generator=generator) / max(in_dim, 1) ** 0.5
        if trainable:
            self.weight = nn.Parameter(weight)  # (in_dim, out_dim)
        else:
            self.register_buffer("weight", weight)  # (in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight  # (batch, n, in_dim) @ (in_dim, out_dim) -> (batch, n, out_dim)


class FixedEmbedding(nn.Module):
    """§I. INTRODUCTION — fixed random projection baseline ("Fixed")."""

    def __init__(self, config: ModelConfig, input_dim: int) -> None:
        super().__init__()
        self.proj = FixedProjection(
            in_dim=input_dim,
            out_dim=config.embedding_dim,
            seed=config.seed,
            trainable=False,
        )

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        _ = adj, node_mask
        z = self.proj(node_features)  # (batch, n, input_dim) -> (batch, n, embedding_dim)
        return z


class MLPEmbedding(nn.Module):
    """§I. INTRODUCTION — trainable MLP baseline ("MLP")."""

    def __init__(self, config: ModelConfig, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projection_hidden_dim, config.embedding_dim),
        )

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        _ = adj, node_mask
        z = self.net(node_features)  # (batch, n, input_dim) -> (batch, n, embedding_dim)
        return z


class AngleVQCPlaceholder(nn.Module):
    """§I. INTRODUCTION, Fig. 2 — circuit-defined Angle-VQC placeholder.

    [UNSPECIFIED] The paper does not provide full executable circuit hyperparameters
    (exact q, L_q, and all gate details) in parsed text. This module keeps interface parity.
    """

    def __init__(self, config: ModelConfig, input_dim: int) -> None:
        super().__init__()
        self.angle_dim = max(2, min(input_dim, 2 ** config.quop_qubits))
        self.angle_proj = nn.Linear(input_dim, self.angle_dim)
        self.out_proj = nn.Linear(2 * self.angle_dim, config.embedding_dim)

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        _ = adj, node_mask
        angles = self.angle_proj(node_features)  # (batch, n, input_dim) -> (batch, n, angle_dim)
        sin_part = torch.sin(angles)  # (batch, n, angle_dim) -> (batch, n, angle_dim)
        cos_part = torch.cos(angles)  # (batch, n, angle_dim) -> (batch, n, angle_dim)
        stacked = torch.cat([sin_part, cos_part], dim=-1)  # -> (batch, n, 2*angle_dim)
        z = self.out_proj(stacked)  # -> (batch, n, embedding_dim)
        return z


class QuOpEmbedding(nn.Module):
    """Algorithm 1 — QuOp local operator-based node embedding."""

    def __init__(self, config: ModelConfig, input_dim: int, trainable_projection: bool) -> None:
        super().__init__()
        self.hop_radius = config.quop_hop_radius
        self.qubits = config.quop_qubits
        self.state_dim = 2 ** self.qubits
        self.summary_dim = 3 * self.state_dim
        self.fused_dim = input_dim + self.summary_dim
        self.trainable_projection = trainable_projection
        self.summary_norm = nn.LayerNorm(self.summary_dim, elementwise_affine=trainable_projection)
        if trainable_projection:
            self.proj = nn.Sequential(
                nn.Linear(self.fused_dim, config.projection_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projection_hidden_dim, config.embedding_dim),
            )
        else:
            self.fixed_proj = FixedProjection(
                in_dim=self.fused_dim,
                out_dim=config.embedding_dim,
                seed=config.seed + 11,
                trainable=False,
            )

    def _ego_nodes(self, local_adj: torch.Tensor, center_idx: int) -> torch.Tensor:
        n = local_adj.size(0)
        visited = torch.zeros(n, dtype=torch.bool, device=local_adj.device)  # (n,)
        frontier = torch.zeros(n, dtype=torch.bool, device=local_adj.device)  # (n,)
        visited[center_idx] = True
        frontier[center_idx] = True
        for _ in range(self.hop_radius):
            reachability = (local_adj @ frontier.float()) > 0  # (n, n) @ (n,) -> (n,)
            frontier = reachability & (~visited)  # (n,) & (n,) -> (n,)
            visited = visited | frontier  # (n,) | (n,) -> (n,)
        indices = torch.nonzero(visited, as_tuple=False).squeeze(-1)  # (n,) -> (k,)
        return indices

    def _single_graph_descriptors(self, local_adj: torch.Tensor) -> torch.Tensor:
        n = local_adj.size(0)
        descriptors = torch.zeros(n, self.summary_dim, device=local_adj.device, dtype=torch.float32)  # (n, s)

        for center in range(n):
            ego_nodes = self._ego_nodes(local_adj, center)  # (k,)
            if ego_nodes.numel() == 0:
                continue
            sub_adj = local_adj.index_select(0, ego_nodes).index_select(1, ego_nodes)  # (n,n)->(k,k)
            h_size = sub_adj.size(0)

            h_matrix = torch.zeros(self.state_dim, self.state_dim, device=local_adj.device, dtype=torch.float32)  # (s,s)
            copy_size = min(h_size, self.state_dim)
            h_matrix[:copy_size, :copy_size] = sub_adj[:copy_size, :copy_size]  # (copy,copy)->(s,s) block
            h_matrix = 0.5 * (h_matrix + h_matrix.transpose(-1, -2))  # (s,s) -> (s,s)

            h_complex = torch.complex(h_matrix, torch.zeros_like(h_matrix))  # (s,s) -> complex(s,s)
            unitary = torch.matrix_exp((-1j) * h_complex)  # (s,s) -> (s,s)

            basis = torch.zeros(self.state_dim, device=local_adj.device, dtype=torch.complex64)  # (s,)
            center_pos = torch.nonzero(ego_nodes == center, as_tuple=False)
            mapped = int(center_pos[0].item()) if center_pos.numel() > 0 else 0
            mapped = min(mapped, self.state_dim - 1)
            basis[mapped] = 1.0 + 0.0j

            evolved = unitary @ basis  # (s,s) @ (s,) -> (s,)
            probs = torch.abs(evolved).pow(2).real  # (s,) -> (s,)
            summary = torch.cat([probs, evolved.real, evolved.imag], dim=0)  # (s)+(s)+(s) -> (3s,)
            descriptors[center] = summary.float()  # (3s,) -> (3s,)

        return descriptors

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_nodes, _ = adj.shape
        summaries = torch.zeros(
            batch_size,
            max_nodes,
            self.summary_dim,
            device=adj.device,
            dtype=torch.float32,
        )  # (batch, n, summary_dim)

        for b in range(batch_size):
            n_valid = int(node_mask[b].sum().item())
            if n_valid == 0:
                continue
            local_adj = adj[b, :n_valid, :n_valid]  # (batch, n, n) -> (n_valid, n_valid)
            local_summary = self._single_graph_descriptors(local_adj)  # -> (n_valid, summary_dim)
            summaries[b, :n_valid] = local_summary  # -> fill valid nodes

        summaries = self.summary_norm(summaries)  # (batch, n, summary_dim) -> (batch, n, summary_dim)
        fused = torch.cat([node_features, summaries], dim=-1)  # (batch, n, f)+(batch, n, s)->(batch, n, f+s)
        if self.trainable_projection:
            z = self.proj(fused)  # (batch, n, f+s) -> (batch, n, embedding_dim)
        else:
            z = self.fixed_proj(fused)  # (batch, n, f+s) -> (batch, n, embedding_dim)

        mask = node_mask.unsqueeze(-1).to(z.dtype)  # (batch, n) -> (batch, n, 1)
        z = z * mask  # (batch, n, d) * (batch, n, 1) -> (batch, n, d)
        return z


class QWalkVecEmbedding(nn.Module):
    """Algorithm 2 — walk-derived node descriptors."""

    def __init__(self, config: ModelConfig, input_dim: int, trainable_projection: bool) -> None:
        super().__init__()
        self.steps = config.qwalk_steps
        self.wp = config.qwalk_wp
        self.wq = config.qwalk_wq
        self.trainable_projection = trainable_projection
        self.summary_dim = self.steps
        self.fused_dim = input_dim + self.summary_dim
        self.summary_norm = nn.LayerNorm(self.summary_dim, elementwise_affine=trainable_projection)
        if trainable_projection:
            self.proj = nn.Sequential(
                nn.Linear(self.fused_dim, config.projection_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projection_hidden_dim, config.embedding_dim),
            )
        else:
            self.fixed_proj = FixedProjection(
                in_dim=self.fused_dim,
                out_dim=config.embedding_dim,
                seed=config.seed + 17,
                trainable=False,
            )

    def _single_graph_descriptors(self, local_adj: torch.Tensor) -> torch.Tensor:
        n = local_adj.size(0)
        identity = torch.eye(n, device=local_adj.device, dtype=local_adj.dtype)  # (n, n)
        # [UNSPECIFIED] Algorithm 2 omits explicit self-loop handling. Using self-loops for stable transitions.
        adj_loop = torch.clamp(local_adj + identity, max=1.0)  # (n, n) + (n, n) -> (n, n)
        row_sum = adj_loop.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)  # (n, n) -> (n, 1)
        transition = adj_loop / row_sum  # (n, n) / (n, 1) -> (n, n)
        walk_logits = self.wp * identity + self.wq * transition  # (n, n) + (n, n) -> (n, n)
        walk_operator = torch.softmax(walk_logits, dim=-1)  # (n, n) -> (n, n)

        # Approximate node visitation dynamics using a global walk state.
        current = torch.full((n,), 1.0 / max(n, 1), device=local_adj.device, dtype=local_adj.dtype)  # (n,)
        descriptors = torch.zeros(n, self.steps, device=local_adj.device, dtype=torch.float32)  # (n, T)
        for t in range(self.steps):
            current = current @ walk_operator  # (n,) @ (n, n) -> (n,)
            descriptors[:, t] = current  # (n,) -> node visitation probability at step t
        return descriptors

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_nodes, _ = adj.shape
        summaries = torch.zeros(
            batch_size,
            max_nodes,
            self.summary_dim,
            device=adj.device,
            dtype=torch.float32,
        )  # (batch, n, T)

        for b in range(batch_size):
            n_valid = int(node_mask[b].sum().item())
            if n_valid == 0:
                continue
            local_adj = adj[b, :n_valid, :n_valid]  # (batch, n, n) -> (n_valid, n_valid)
            local_summary = self._single_graph_descriptors(local_adj)  # -> (n_valid, T)
            summaries[b, :n_valid] = local_summary  # -> fill valid nodes

        summaries = self.summary_norm(summaries)  # (batch, n, T) -> (batch, n, T)
        fused = torch.cat([node_features, summaries], dim=-1)  # (batch, n, f)+(batch, n, T)->(batch, n, f+T)
        if self.trainable_projection:
            z = self.proj(fused)  # (batch, n, f+T) -> (batch, n, d)
        else:
            z = self.fixed_proj(fused)  # (batch, n, f+T) -> (batch, n, d)

        mask = node_mask.unsqueeze(-1).to(z.dtype)  # (batch, n) -> (batch, n, 1)
        z = z * mask  # (batch, n, d) * (batch, n, 1) -> (batch, n, d)
        return z


class QPEEmbedding(nn.Module):
    """Algorithm 3 — quantum positional encoding via spectral evolution."""

    def __init__(self, config: ModelConfig, input_dim: int) -> None:
        super().__init__()
        self.times = [float(t) for t in config.qpe_times]
        self.anchors = config.qpe_anchors
        self.summary_dim = len(self.times) * self.anchors
        self.fused_dim = input_dim + self.summary_dim
        self.summary_norm = nn.LayerNorm(self.summary_dim, elementwise_affine=False)
        self.fixed_proj = FixedProjection(
            in_dim=self.fused_dim,
            out_dim=config.embedding_dim,
            seed=config.seed + 23,
            trainable=False,
        )

    def _single_graph_descriptors(self, local_adj: torch.Tensor) -> torch.Tensor:
        n = local_adj.size(0)
        identity = torch.eye(n, device=local_adj.device, dtype=local_adj.dtype)  # (n, n)
        degree = local_adj.sum(dim=-1)  # (n, n) -> (n,)
        inv_sqrt_degree = torch.pow(degree + 1.0e-8, -0.5)  # (n,) -> (n,)
        d_inv_sqrt = torch.diag(inv_sqrt_degree)  # (n,) -> (n, n)
        h_operator = identity - (d_inv_sqrt @ local_adj @ d_inv_sqrt)  # (n, n) -> (n, n)

        evals, evecs = torch.linalg.eigh(h_operator)  # (n, n) -> (n,), (n, n)
        anchor_count = min(self.anchors, n)
        # [UNSPECIFIED] Anchor selection rule is not specified. Using top-degree anchors.
        anchor_indices = torch.topk(degree, k=anchor_count, largest=True).indices  # (n,) -> (anchor_count,)

        descriptors = torch.zeros(n, self.summary_dim, device=local_adj.device, dtype=torch.float32)  # (n, A*T)
        evecs_complex = evecs.to(torch.complex64)  # (n, n) -> complex(n, n)
        evecs_t_complex = evecs.transpose(-1, -2).to(torch.complex64)  # (n, n) -> complex(n, n)

        for t_idx, t in enumerate(self.times):
            phase = torch.exp(torch.complex(torch.zeros_like(evals), -evals * t))  # (n,) -> complex(n,)
            evolution = (evecs_complex * phase.unsqueeze(0)) @ evecs_t_complex  # (n, n) -> complex(n, n)
            for a_idx in range(anchor_count):
                anchor = int(anchor_indices[a_idx].item())
                col = t_idx * self.anchors + a_idx
                descriptors[:, col] = evolution[:, anchor].real.float()  # complex(n,) -> real(n,)

        return descriptors

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_nodes, _ = adj.shape
        summaries = torch.zeros(
            batch_size,
            max_nodes,
            self.summary_dim,
            device=adj.device,
            dtype=torch.float32,
        )  # (batch, n, A*T)

        for b in range(batch_size):
            n_valid = int(node_mask[b].sum().item())
            if n_valid == 0:
                continue
            local_adj = adj[b, :n_valid, :n_valid]  # (batch, n, n) -> (n_valid, n_valid)
            local_summary = self._single_graph_descriptors(local_adj)  # -> (n_valid, A*T)
            summaries[b, :n_valid] = local_summary

        summaries = self.summary_norm(summaries)  # (batch, n, A*T) -> (batch, n, A*T)
        fused = torch.cat([node_features, summaries], dim=-1)  # (batch, n, f)+(batch, n, A*T)->(batch, n, f+A*T)
        z = self.fixed_proj(fused)  # (batch, n, f+A*T) -> (batch, n, d)
        mask = node_mask.unsqueeze(-1).to(z.dtype)  # (batch, n) -> (batch, n, 1)
        z = z * mask  # (batch, n, d) * (batch, n, 1) -> (batch, n, d)
        return z


def build_embedding_module(config: ModelConfig, input_dim: int) -> nn.Module:
    """Factory for embedding modules used in the benchmark."""
    kind = config.embedding_kind.lower()
    if kind == "fixed":
        return FixedEmbedding(config=config, input_dim=input_dim)
    if kind == "mlp":
        return MLPEmbedding(config=config, input_dim=input_dim)
    if kind == "angle_vqc":
        return AngleVQCPlaceholder(config=config, input_dim=input_dim)
    if kind == "quop":
        return QuOpEmbedding(config=config, input_dim=input_dim, trainable_projection=False)
    if kind == "quop_trainable":
        return QuOpEmbedding(config=config, input_dim=input_dim, trainable_projection=True)
    if kind == "qwalkvec":
        return QWalkVecEmbedding(config=config, input_dim=input_dim, trainable_projection=False)
    if kind == "qwalkvec_trainable":
        return QWalkVecEmbedding(config=config, input_dim=input_dim, trainable_projection=True)
    if kind == "qpe":
        return QPEEmbedding(config=config, input_dim=input_dim)
    raise ValueError(f"Unknown embedding_kind: {config.embedding_kind}")


class GINLayer(nn.Module):
    """§B. Graph classifier and training protocol — GIN message passing layer."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(0.0))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        norm_adj = normalize_adjacency(adj)  # (batch, n, n) -> (batch, n, n)
        agg = torch.matmul(norm_adj, x)  # (batch, n, n) @ (batch, n, h) -> (batch, n, h)
        updated = (1.0 + self.eps) * x + agg  # (batch, n, h) + (batch, n, h) -> (batch, n, h)
        out = self.mlp(updated)  # (batch, n, h) -> (batch, n, h)
        mask = node_mask.unsqueeze(-1).to(out.dtype)  # (batch, n) -> (batch, n, 1)
        out = out * mask  # (batch, n, h) * (batch, n, 1) -> (batch, n, h)
        return out


class GraphEmbeddingBenchmarkModel(nn.Module):
    """Unified benchmark model with fixed GIN backbone and pluggable embedding stage.

    §B. Graph classifier and training protocol — same backbone across embeddings.
    """

    def __init__(self, config: ModelConfig, input_dim: int) -> None:
        super().__init__()
        self.config = config
        self.embedding = build_embedding_module(config=config, input_dim=input_dim)
        self.input_proj = nn.Linear(config.embedding_dim, config.gin_hidden_dim)
        self.layers = nn.ModuleList([GINLayer(config.gin_hidden_dim) for _ in range(config.gin_layers)])
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Sequential(
            nn.Linear(config.gin_hidden_dim, config.mlp_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_head_hidden_dim, config.num_classes),
        )

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: Node inputs — shape: (batch, n, input_dim)
            adj: Dense adjacency — shape: (batch, n, n)
            node_mask: Valid-node mask — shape: (batch, n)
        Returns:
            Graph logits — shape: (batch, num_classes)
        """
        z = self.embedding(node_features, adj, node_mask)  # (batch, n, input_dim) -> (batch, n, d)
        h = self.input_proj(z)  # (batch, n, d) -> (batch, n, hidden)
        h = h * node_mask.unsqueeze(-1).to(h.dtype)  # (batch, n, hidden) * (batch, n, 1) -> (batch, n, hidden)
        for layer in self.layers:
            h = layer(h, adj, node_mask)  # (batch, n, hidden) -> (batch, n, hidden)
        graph_state = masked_mean_pool(h, node_mask)  # (batch, n, hidden) -> (batch, hidden)
        graph_state = self.dropout(graph_state)  # (batch, hidden) -> (batch, hidden)
        logits = self.head(graph_state)  # (batch, hidden) -> (batch, num_classes)
        return logits

