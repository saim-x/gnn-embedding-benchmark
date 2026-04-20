"""
Release benchmark runner for:
How Embeddings Shape Graph Neural Networks: Classical vs Quantum-Oriented Node Representations

This script executes an end-to-end multi-embedding benchmark on a synthetic dataset
and writes publication-ready figures for README usage.
"""

from __future__ import annotations

import copy
import csv
import statistics
import tempfile
from collections import defaultdict
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from src.data import build_dataloader
from src.evaluate import compute_all_metrics
from src.loss import GraphClassificationLoss
from src.model import GraphEmbeddingBenchmarkModel
from src.train import build_model_config, train
from src.utils import set_seed


@contextmanager
def _pushd(path: Path):
    import os

    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def _erdos_renyi_graph(n_nodes: int, edge_prob: float) -> torch.Tensor:
    upper = (torch.rand(n_nodes, n_nodes) < edge_prob).float().triu(diagonal=1)
    return upper + upper.transpose(0, 1)


def _build_synthetic_dataset(path: Path, num_graphs: int, seed: int) -> None:
    set_seed(seed)
    graphs: List[Dict[str, torch.Tensor | int]] = []
    for i in range(num_graphs):
        label = i % 2
        n_nodes = int(torch.randint(low=14, high=22, size=(1,)).item())
        edge_prob = 0.24 if label == 0 else 0.30
        adj = _erdos_renyi_graph(n_nodes=n_nodes, edge_prob=edge_prob)
        perm = torch.randperm(n_nodes)
        adj = adj.index_select(0, perm).index_select(1, perm)
        graphs.append({"adj": adj, "label": label})
    torch.save(graphs, path)


def _save_metrics_csv(rows: List[Dict[str, float | str]], path: Path, fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )


def _save_svg(path: Path, body: str, width: int, height: int) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_svg_header(width, height))
        handle.write(body)
        handle.write("</svg>")


def _render_grouped_bar_svg(
    path: Path,
    title: str,
    labels: List[str],
    left_values: List[float],
    right_values: List[float],
    left_name: str,
    right_name: str,
) -> None:
    width, height = 1400, 700
    left_margin, right_margin, top_margin, bottom_margin = 90, 40, 90, 180
    chart_w = width - left_margin - right_margin
    chart_h = height - top_margin - bottom_margin
    n = len(labels)
    group_w = chart_w / max(n, 1)
    bar_w = group_w * 0.33

    body: List[str] = [
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="45" text-anchor="middle" font-size="28" font-family="Arial">{title}</text>',
        f'<text x="{left_margin + 10}" y="{top_margin - 20}" font-size="16" font-family="Arial" fill="#1f77b4">{left_name}</text>',
        f'<text x="{left_margin + 170}" y="{top_margin - 20}" font-size="16" font-family="Arial" fill="#ff7f0e">{right_name}</text>',
        f'<line x1="{left_margin}" y1="{top_margin + chart_h}" x2="{left_margin + chart_w}" y2="{top_margin + chart_h}" stroke="#444" stroke-width="2"/>',
        f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin + chart_h}" stroke="#444" stroke-width="2"/>',
    ]

    for y_tick in range(6):
        y_val = y_tick / 5
        y = top_margin + chart_h - y_val * chart_h
        body.append(
            f'<line x1="{left_margin}" y1="{y:.2f}" x2="{left_margin + chart_w}" y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{left_margin - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="12" font-family="Arial">{y_val:.1f}</text>'
        )

    for idx, label in enumerate(labels):
        x0 = left_margin + idx * group_w + group_w * 0.17
        left_h = max(0.0, min(1.0, left_values[idx])) * chart_h
        right_h = max(0.0, min(1.0, right_values[idx])) * chart_h
        left_y = top_margin + chart_h - left_h
        right_y = top_margin + chart_h - right_h

        body.append(
            f'<rect x="{x0:.2f}" y="{left_y:.2f}" width="{bar_w:.2f}" height="{left_h:.2f}" fill="#1f77b4"/>'
        )
        body.append(
            f'<rect x="{x0 + bar_w + group_w * 0.1:.2f}" y="{right_y:.2f}" width="{bar_w:.2f}" height="{right_h:.2f}" fill="#ff7f0e"/>'
        )
        body.append(
            f'<text x="{x0 + group_w * 0.32:.2f}" y="{top_margin + chart_h + 24}" text-anchor="middle" '
            f'font-size="12" transform="rotate(35 {x0 + group_w * 0.32:.2f} {top_margin + chart_h + 24})" font-family="Arial">{label}</text>'
        )

    _save_svg(path=path, body="".join(body), width=width, height=height)


def _render_scatter_svg(path: Path, title: str, x_vals: List[float], y_vals: List[float], labels: List[str]) -> None:
    width, height = 1000, 700
    left_margin, right_margin, top_margin, bottom_margin = 90, 80, 90, 90
    chart_w = width - left_margin - right_margin
    chart_h = height - top_margin - bottom_margin
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = 0.0, 1.0
    x_span = max(1.0e-8, x_max - x_min)

    body: List[str] = [
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="45" text-anchor="middle" font-size="28" font-family="Arial">{title}</text>',
        f'<line x1="{left_margin}" y1="{top_margin + chart_h}" x2="{left_margin + chart_w}" y2="{top_margin + chart_h}" stroke="#444" stroke-width="2"/>',
        f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin + chart_h}" stroke="#444" stroke-width="2"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="16" font-family="Arial">Test loss</text>',
        f'<text x="24" y="{height / 2}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 24 {height / 2})">Macro-F1</text>',
    ]

    for y_tick in range(6):
        y_val = y_tick / 5
        y = top_margin + chart_h - y_val * chart_h
        body.append(
            f'<line x1="{left_margin}" y1="{y:.2f}" x2="{left_margin + chart_w}" y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{left_margin - 10}" y="{y + 5:.2f}" text-anchor="end" font-size="12" font-family="Arial">{y_val:.1f}</text>'
        )

    for idx, x in enumerate(x_vals):
        y = y_vals[idx]
        px = left_margin + ((x - x_min) / x_span) * chart_w
        py = top_margin + chart_h - ((y - y_min) / (y_max - y_min + 1.0e-8)) * chart_h
        body.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6" fill="#2ca02c"/>')
        body.append(
            f'<text x="{px + 8:.2f}" y="{py - 8:.2f}" font-size="12" font-family="Arial">{labels[idx]}</text>'
        )

    _save_svg(path=path, body="".join(body), width=width, height=height)


def _save_figures(rows: List[Dict[str, float | str]], out_dir: Path) -> None:
    embeddings = [str(r["embedding"]) for r in rows]
    accuracy = [float(r["accuracy"]) for r in rows]
    macro_f1 = [float(r["macro_f1"]) for r in rows]
    macro_precision = [float(r["macro_precision"]) for r in rows]
    macro_recall = [float(r["macro_recall"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]

    _render_grouped_bar_svg(
        path=out_dir / "accuracy_macro_f1.svg",
        title="Synthetic Benchmark: Accuracy vs Macro-F1",
        labels=embeddings,
        left_values=accuracy,
        right_values=macro_f1,
        left_name="Accuracy",
        right_name="Macro-F1",
    )
    _render_grouped_bar_svg(
        path=out_dir / "precision_recall.svg",
        title="Synthetic Benchmark: Macro Precision vs Macro Recall",
        labels=embeddings,
        left_values=macro_precision,
        right_values=macro_recall,
        left_name="Macro Precision",
        right_name="Macro Recall",
    )
    _render_scatter_svg(
        path=out_dir / "loss_vs_macro_f1.svg",
        title="Synthetic Benchmark: Loss vs Macro-F1",
        x_vals=loss,
        y_vals=macro_f1,
        labels=embeddings,
    )


def _evaluate_saved_model(config: dict, run_dir: Path) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = GraphClassificationLoss()
    num_classes = int(config["model"]["num_classes"])

    train_loader = build_dataloader(config=config, split="train")
    first_batch = next(iter(train_loader), None)
    if first_batch is None:
        raise RuntimeError("Training dataloader is empty in release benchmark.")
    input_dim = int(first_batch["node_features"].size(-1))

    model_cfg = build_model_config(config=config)
    model = GraphEmbeddingBenchmarkModel(config=model_cfg, input_dim=input_dim).to(device)
    state_path = run_dir / "best_model.pt"
    state_dict = torch.load(state_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    test_loader = build_dataloader(config=config, split="test")
    logits_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in test_loader:
            node_features = batch["node_features"].to(device)
            adj = batch["adj"].to(device)
            node_mask = batch["node_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(node_features=node_features, adj=adj, node_mask=node_mask)
            loss = loss_fn(logits=logits, targets=labels)
            bsz = int(labels.size(0))
            total_loss += float(loss.item()) * bsz
            total_count += bsz
            logits_all.append(logits.cpu())
            targets_all.append(labels.cpu())

    if total_count == 0:
        raise RuntimeError("Test dataloader is empty in release benchmark.")

    logits_cat = torch.cat(logits_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)
    preds = torch.argmax(logits_cat, dim=-1)
    metrics = compute_all_metrics(logits=logits_cat, targets=targets_cat, num_classes=num_classes)

    pred_counts = torch.bincount(preds, minlength=num_classes).float()
    target_counts = torch.bincount(targets_cat, minlength=num_classes).float()
    n = float(total_count)

    majority_pred_rate = float(pred_counts.max().item() / n)
    positive_rate = float(pred_counts[1].item() / n) if num_classes > 1 else 0.0
    target_positive_rate = float(target_counts[1].item() / n) if num_classes > 1 else 0.0
    unique_pred_classes = float(int((pred_counts > 0).sum().item()))
    collapse_flag = 1.0 if unique_pred_classes <= 1.0 else 0.0

    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "loss": float(total_loss / n),
        "pred_majority_rate": majority_pred_rate,
        "pred_class1_rate": positive_rate,
        "target_class1_rate": target_positive_rate,
        "unique_pred_classes": unique_pred_classes,
        "collapse_flag": collapse_flag,
    }


def _aggregate_rows(rows: List[Dict[str, float | str]], embeddings: List[str]) -> List[Dict[str, float | str]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        embedding = str(row["embedding"])
        for key, value in row.items():
            if key in {"embedding", "seed"}:
                continue
            grouped[embedding][key].append(float(value))

    out: List[Dict[str, float | str]] = []
    for embedding in embeddings:
        metrics = grouped[embedding]
        summary: Dict[str, float | str] = {"embedding": embedding}
        for metric_name, values in metrics.items():
            summary[metric_name] = float(statistics.mean(values))
            summary[f"{metric_name}_std"] = float(statistics.pstdev(values) if len(values) > 1 else 0.0)
        out.append(summary)
    return out


def run_release_benchmark(
    output_dir: str = "results/release_verification",
    epochs: int = 14,
    patience: int = 5,
    num_graphs: int = 300,
    seeds: List[int] | None = None,
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_values = seeds if seeds is not None else [7, 13, 29]

    embeddings = [
        "fixed",
        "mlp",
        "angle_vqc",
        "quop",
        "quop_trainable",
        "qwalkvec",
        "qwalkvec_trainable",
        "qpe",
    ]

    with open(repo_root / "configs" / "base.yaml", "r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)

    detailed_rows: List[Dict[str, float | str]] = []
    with tempfile.TemporaryDirectory(prefix="release_benchmark_") as tmp:
        tmp_dir = Path(tmp)
        for seed in seed_values:
            dataset_name = f"SYNTHETIC_RELEASE_{seed}"
            dataset_path = tmp_dir / f"{dataset_name}.pt"
            _build_synthetic_dataset(path=dataset_path, num_graphs=num_graphs, seed=seed)

            seed_cfg_base = copy.deepcopy(base_cfg)
            seed_cfg_base["data"]["data_dir"] = str(tmp_dir)
            seed_cfg_base["data"]["dataset_name"] = dataset_name
            seed_cfg_base["data"]["num_workers"] = 0
            seed_cfg_base["model"]["num_classes"] = 2
            seed_cfg_base["training"]["max_epochs"] = int(epochs)
            seed_cfg_base["training"]["early_stopping_patience"] = int(patience)
            seed_cfg_base["training"]["batch_size"] = 16
            seed_cfg_base["training"]["seed"] = int(seed)

            for embedding in embeddings:
                cfg = copy.deepcopy(seed_cfg_base)
                cfg["model"]["embedding_kind"] = embedding
                cfg_path = tmp_dir / f"config_{embedding}_{seed}.yaml"
                with open(cfg_path, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(cfg, handle, sort_keys=False)

                with _pushd(tmp_dir):
                    with redirect_stdout(StringIO()):
                        train(str(cfg_path))
                    metrics = _evaluate_saved_model(config=cfg, run_dir=tmp_dir)

                detailed_rows.append(
                    {
                        "seed": float(seed),
                        "embedding": embedding,
                        **metrics,
                    }
                )

    summary_rows = _aggregate_rows(rows=detailed_rows, embeddings=embeddings)
    summary_fieldnames = [
        "embedding",
        "accuracy",
        "accuracy_std",
        "macro_f1",
        "macro_f1_std",
        "macro_precision",
        "macro_precision_std",
        "macro_recall",
        "macro_recall_std",
        "loss",
        "loss_std",
        "pred_majority_rate",
        "pred_majority_rate_std",
        "pred_class1_rate",
        "pred_class1_rate_std",
        "target_class1_rate",
        "target_class1_rate_std",
        "unique_pred_classes",
        "unique_pred_classes_std",
        "collapse_flag",
        "collapse_flag_std",
    ]
    detailed_fieldnames = [
        "seed",
        "embedding",
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "loss",
        "pred_majority_rate",
        "pred_class1_rate",
        "target_class1_rate",
        "unique_pred_classes",
        "collapse_flag",
    ]
    _save_metrics_csv(rows=summary_rows, path=out_dir / "metrics.csv", fieldnames=summary_fieldnames)
    _save_metrics_csv(rows=detailed_rows, path=out_dir / "metrics_detailed.csv", fieldnames=detailed_fieldnames)
    _save_figures(rows=summary_rows, out_dir=out_dir)
    return out_dir


if __name__ == "__main__":
    out = run_release_benchmark()
    print(f"Release benchmark artifacts written to: {out}")

