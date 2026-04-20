"""
How Embeddings Shape Graph Neural Networks: Classical vs Quantum-Oriented Node Representations — Training

Paper: https://arxiv.org/abs/2604.15273v1
Implements: Fixed-protocol training loop with early stopping on validation Macro-F1.

Section references:
  §B. Graph classifier and training protocol — cross-entropy objective
  §A. Experimental Setup — Adam, lr=1e-3, batch size 16, max epochs 30, early stopping patience 7
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml

from src.data import build_dataloader
from src.evaluate import compute_all_metrics
from src.loss import GraphClassificationLoss
from src.model import GraphEmbeddingBenchmarkModel, ModelConfig
from src.utils import set_seed


def load_config(config_path: str = "configs/base.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model_config(config: dict) -> ModelConfig:
    model_cfg = config["model"]
    return ModelConfig(
        embedding_dim=int(model_cfg["embedding_dim"]),
        pe_k=int(model_cfg["pe_k"]),
        gin_layers=int(model_cfg["gin_layers"]),
        gin_hidden_dim=int(model_cfg["gin_hidden_dim"]),
        mlp_head_hidden_dim=int(model_cfg["mlp_head_hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        embedding_kind=str(model_cfg["embedding_kind"]),
        qwalk_steps=int(model_cfg["qwalk_steps"]),
        qwalk_wp=float(model_cfg["qwalk_wp"]),
        qwalk_wq=float(model_cfg["qwalk_wq"]),
        qpe_times=[float(x) for x in model_cfg["qpe_times"]],
        qpe_anchors=int(model_cfg["qpe_anchors"]),
        quop_hop_radius=int(model_cfg["quop_hop_radius"]),
        quop_qubits=int(model_cfg["quop_qubits"]),
        projection_hidden_dim=int(model_cfg["projection_hidden_dim"]),
        num_classes=int(model_cfg["num_classes"]),
        seed=int(config["training"]["seed"]),
    )


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    optimizer_name = str(train_cfg["optimizer"]).lower()
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    if optimizer_name == "adam":
        # §A. Experimental Setup — Adam optimizer with stated lr and weight decay.
        # [PARTIALLY_SPECIFIED] Adam betas/eps are not explicitly listed in the paper.
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(float(x) for x in train_cfg["adam_betas"]),
            eps=float(train_cfg["adam_eps"]),
        )
    raise ValueError(f"Unsupported optimizer in this minimal implementation: {optimizer_name}")


def _step_batch(
    model: GraphEmbeddingBenchmarkModel,
    batch: Dict[str, torch.Tensor],
    loss_fn: GraphClassificationLoss,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_features = batch["node_features"].to(device)  # (b, n, f)
    adj = batch["adj"].to(device)  # (b, n, n)
    node_mask = batch["node_mask"].to(device)  # (b, n)
    labels = batch["labels"].to(device)  # (b,)

    logits = model(node_features=node_features, adj=adj, node_mask=node_mask)  # (b, C)
    loss = loss_fn(logits=logits, targets=labels)  # (b, C)+(b,) -> scalar
    return loss, logits, labels


def evaluate_model(
    model: GraphEmbeddingBenchmarkModel,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: GraphClassificationLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    logits_all = []
    targets_all = []

    with torch.no_grad():
        for batch in dataloader:
            loss, logits, labels = _step_batch(model=model, batch=batch, loss_fn=loss_fn, device=device)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            logits_all.append(logits.cpu())
            targets_all.append(labels.cpu())

    if total_count == 0:
        raise RuntimeError("Evaluation dataloader is empty.")
    logits_cat = torch.cat(logits_all, dim=0)  # [(b_i, C)] -> (N, C)
    targets_cat = torch.cat(targets_all, dim=0)  # [(b_i,)] -> (N,)
    metrics = compute_all_metrics(logits=logits_cat, targets=targets_cat)
    metrics["loss"] = total_loss / total_count
    return metrics


def train(config_path: str = "configs/base.yaml") -> Dict[str, float]:
    config = load_config(config_path)
    set_seed(int(config["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_dataloader(config=config, split="train")
    val_loader = build_dataloader(config=config, split="val")
    test_loader = build_dataloader(config=config, split="test")

    first_batch = next(iter(train_loader), None)
    if first_batch is None:
        raise RuntimeError("Training dataloader is empty.")
    input_dim = int(first_batch["node_features"].size(-1))

    model_cfg = build_model_config(config=config)
    model = GraphEmbeddingBenchmarkModel(config=model_cfg, input_dim=input_dim).to(device)
    optimizer = build_optimizer(model=model, config=config)
    loss_fn = GraphClassificationLoss()

    best_metric = float("-inf")
    best_state = None
    patience = int(config["training"]["early_stopping_patience"])
    patience_counter = 0
    metric_name = str(config["training"]["early_stopping_metric"])
    max_epochs = int(config["training"]["max_epochs"])
    grad_clip = config["training"].get("gradient_clip_norm")

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = _step_batch(model=model, batch=batch, loss_fn=loss_fn, device=device)
            loss.backward()
            if grad_clip is not None:
                # [UNSPECIFIED] Gradient clipping is not reported; controlled by config.
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        val_metrics = evaluate_model(model=model, dataloader=val_loader, loss_fn=loss_fn, device=device)
        current = float(val_metrics[metric_name])
        if current > best_metric:
            best_metric = current
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        print(f"Epoch {epoch + 1}/{max_epochs} | val_{metric_name}: {current:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model=model, dataloader=test_loader, loss_fn=loss_fn, device=device)
    print(
        "Test metrics | "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}, "
        f"macro_precision={test_metrics['macro_precision']:.4f}, "
        f"macro_recall={test_metrics['macro_recall']:.4f}"
    )

    output_path = Path("best_model.pt")
    torch.save(model.state_dict(), output_path)
    return test_metrics


if __name__ == "__main__":
    train()

