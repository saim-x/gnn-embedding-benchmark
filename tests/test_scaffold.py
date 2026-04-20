from __future__ import annotations

import copy
import io
import random
import shutil
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import torch
import yaml

from src.data import build_dataloader
from src.evaluate import compute_all_metrics
from src.model import GraphEmbeddingBenchmarkModel
from src.train import build_model_config, train


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    try:
        import os

        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class TestScaffold(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="scaffold_test_"))
        cls.dataset_path = cls.temp_dir / "SMOKE.pt"

        random.seed(7)
        torch.manual_seed(7)

        graphs = []
        for i in range(40):
            n = random.randint(6, 9)
            p = 0.25 if i % 2 == 0 else 0.35
            upper = (torch.rand(n, n) < p).float().triu(diagonal=1)
            adj = upper + upper.transpose(0, 1)
            graphs.append({"adj": adj, "label": i % 2})

        torch.save(graphs, cls.dataset_path)

        with open(cls.repo_root / "configs" / "base.yaml", "r", encoding="utf-8") as f:
            cls.base_cfg = yaml.safe_load(f)

        cls.base_cfg["data"]["data_dir"] = str(cls.temp_dir)
        cls.base_cfg["data"]["dataset_name"] = "SMOKE"
        cls.base_cfg["data"]["num_workers"] = 0
        cls.base_cfg["model"]["num_classes"] = 2
        cls.base_cfg["training"]["batch_size"] = 8
        cls.base_cfg["training"]["max_epochs"] = 1
        cls.base_cfg["training"]["early_stopping_patience"] = 1

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_macro_metrics_use_logit_class_count(self) -> None:
        logits = torch.tensor([[4.0, -4.0], [3.0, -3.0]], dtype=torch.float32)
        targets = torch.tensor([0, 0], dtype=torch.long)
        metrics = compute_all_metrics(logits=logits, targets=targets)
        self.assertAlmostEqual(metrics["macro_f1"], 0.5, places=6)

    def test_all_embedding_variants_forward(self) -> None:
        cfg = copy.deepcopy(self.base_cfg)
        loader = build_dataloader(config=cfg, split="train")
        first_batch = next(iter(loader))
        input_dim = int(first_batch["node_features"].size(-1))
        variants = [
            "fixed",
            "mlp",
            "angle_vqc",
            "quop",
            "quop_trainable",
            "qwalkvec",
            "qwalkvec_trainable",
            "qpe",
        ]

        for kind in variants:
            cfg["model"]["embedding_kind"] = kind
            model_cfg = build_model_config(cfg)
            model = GraphEmbeddingBenchmarkModel(config=model_cfg, input_dim=input_dim)
            with torch.no_grad():
                logits = model(
                    node_features=first_batch["node_features"],
                    adj=first_batch["adj"],
                    node_mask=first_batch["node_mask"],
                )
            self.assertEqual(tuple(logits.shape), (first_batch["labels"].shape[0], 2))

    def test_quantum_embeddings_depend_on_base_features(self) -> None:
        cfg = copy.deepcopy(self.base_cfg)
        loader = build_dataloader(config=cfg, split="train")
        batch = next(iter(loader))
        input_dim = int(batch["node_features"].size(-1))
        variants = ["quop", "quop_trainable", "qwalkvec", "qwalkvec_trainable", "qpe"]

        node_features_a = batch["node_features"].clone()
        node_features_b = batch["node_features"].clone() + 0.5

        for kind in variants:
            cfg["model"]["embedding_kind"] = kind
            model_cfg = build_model_config(cfg)
            model = GraphEmbeddingBenchmarkModel(config=model_cfg, input_dim=input_dim)
            with torch.no_grad():
                logits_a = model(
                    node_features=node_features_a,
                    adj=batch["adj"],
                    node_mask=batch["node_mask"],
                )
                logits_b = model(
                    node_features=node_features_b,
                    adj=batch["adj"],
                    node_mask=batch["node_mask"],
                )
            self.assertFalse(torch.allclose(logits_a, logits_b), msg=f"{kind} ignored base node features")

    def test_train_smoke(self) -> None:
        cfg = copy.deepcopy(self.base_cfg)
        cfg["model"]["embedding_kind"] = "qwalkvec_trainable"
        config_path = self.temp_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        with _pushd(self.temp_dir):
            with redirect_stdout(io.StringIO()):
                metrics = train(str(config_path))

        for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "loss"):
            self.assertIn(key, metrics)
            self.assertTrue(isinstance(metrics[key], float))


if __name__ == "__main__":
    unittest.main()

