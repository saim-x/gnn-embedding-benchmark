"""
How Embeddings Shape Graph Neural Networks: Classical vs Quantum-Oriented Node Representations — Evaluation

Paper: https://arxiv.org/abs/2604.15273v1
Implements: Accuracy, Macro-F1, Macro Precision, Macro Recall.

Section references:
  §B. Graph classifier and training protocol — reports Accuracy, Macro-F1, Macro Precision/Recall
"""

from __future__ import annotations

from typing import Dict

import torch


def _confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    conf = torch.zeros(num_classes, num_classes, dtype=torch.float32, device=predictions.device)  # (C, C)
    for t, p in zip(targets, predictions):
        conf[int(t.item()), int(p.item())] += 1.0
    return conf


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    correct = (predictions == targets).float().mean()  # (b,) == (b,) -> scalar
    return float(correct.item())


def compute_macro_precision_recall_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1.0e-8,
) -> Dict[str, float]:
    conf = _confusion_matrix(predictions=predictions, targets=targets, num_classes=num_classes)  # -> (C, C)
    tp = torch.diag(conf)  # (C, C) -> (C,)
    fp = conf.sum(dim=0) - tp  # (C, C) -> (C,)
    fn = conf.sum(dim=1) - tp  # (C, C) -> (C,)

    precision = tp / (tp + fp + eps)  # (C,) -> (C,)
    recall = tp / (tp + fn + eps)  # (C,) -> (C,)
    f1 = 2.0 * precision * recall / (precision + recall + eps)  # (C,) -> (C,)

    return {
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
    }


def compute_all_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int | None = None) -> Dict[str, float]:
    """Compute all paper-reported metrics.

    Args:
        logits: Model outputs — shape: (batch, num_classes)
        targets: Ground-truth labels — shape: (batch,)
    """
    predictions = torch.argmax(logits, dim=-1)  # (batch, C) -> (batch,)
    classes = num_classes if num_classes is not None else int(logits.size(-1))
    acc = compute_accuracy(predictions=predictions, targets=targets)
    macro_metrics = compute_macro_precision_recall_f1(
        predictions=predictions,
        targets=targets,
        num_classes=classes,
    )
    return {
        "accuracy": acc,
        "macro_f1": macro_metrics["macro_f1"],
        "macro_precision": macro_metrics["macro_precision"],
        "macro_recall": macro_metrics["macro_recall"],
    }

