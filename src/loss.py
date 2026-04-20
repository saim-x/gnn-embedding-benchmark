"""
How Embeddings Shape Graph Neural Networks: Classical vs Quantum-Oriented Node Representations — Loss

Paper: https://arxiv.org/abs/2604.15273v1
Implements: Cross-entropy objective used in the shared graph-classification protocol.

Section references:
  §B. Graph classifier and training protocol — "parameters are optimized by minimizing cross-entropy on the training split"
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphClassificationLoss(nn.Module):
    """Cross-entropy loss for graph-level classification logits."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute scalar training loss.

        Args:
            logits: Model outputs — shape: (batch, num_classes)
            targets: Class indices — shape: (batch,)
        """
        loss = self.criterion(logits, targets)  # (batch, C) + (batch,) -> scalar
        return loss

