#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


TEST_FILE = Path(__file__).resolve()
ALIGNMENT_DIR = TEST_FILE.parent
if not (ALIGNMENT_DIR / "10_export_node_embeddings_v6_3.py").exists():
    ALIGNMENT_DIR = TEST_FILE.parents[1]
EXPORTER = ALIGNMENT_DIR / "10_export_node_embeddings_v6_3.py"
VERIFIER = ALIGNMENT_DIR / "11_verify_node_embedding_export_v6_3.py"


def load_exporter_module():
    spec = importlib.util.spec_from_file_location("node_export_v6_3", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExporterStaticContractTest(unittest.TestCase):
    def test_exporter_and_verifier_parse(self):
        for path in (EXPORTER, VERIFIER):
            ast.parse(path.read_text(encoding="utf-8"), filename=path.name)

    def test_formal_contract_is_explicit(self):
        source = EXPORTER.read_text(encoding="utf-8")
        self.assertIn("stage3_node_base_train_graph_encoder_v1", source)
        self.assertIn("stage5_unified_node_base_three_loss_encoder_v6_5", source)
        self.assertIn("validation_selected", source)
        self.assertIn("node_embedding_index.json", source)
        self.assertIn("embedding_ref", source)
        self.assertIn("torch.inference_mode", source)
        self.assertNotIn("low_embeddings + high_embeddings", source)

    def test_nested_node_schema_adapter(self):
        module = load_exporter_module()
        row = {
            "payload": {
                "node": {
                    "node_ref": "sample-x::TypeSem::t1",
                    "metadata": {"sample_hash": "sample-x", "semantic_type": "TypeSem"},
                }
            }
        }
        resolved = module.resolve_node_metadata(row)
        self.assertEqual(resolved["node_id"], "sample-x::TypeSem::t1")
        self.assertEqual(resolved["sample_id"], "sample-x")
        self.assertEqual(resolved["node_type"], "TypeSem")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for tensor export integration")
class TinyFormalExportIntegrationTest(unittest.TestCase):
    def test_tiny_formal_export_and_verification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            feature_path = root / "features.pt"
            checkpoint_path = root / "alignment_model_best.pt"
            metrics_path = root / "alignment_metrics.json"
            nodes_path = root / "nodes.jsonl"
            out = root / "embedding"

            node_ids = [
                "sample-a::InstrSem::i1",
                "sample-a::TypeSem::t1",
                "sample-b::APISem::a1",
            ]
            features = torch.tensor(
                [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.4, 0.3, 1.0]],
                dtype=torch.float32,
            )
            torch.save({
                "features": features,
                "node_ids": node_ids,
                "node_index": {node_id: i for i, node_id in enumerate(node_ids)},
                "feature_dim": 3,
                "feature_schema": "tiny-test",
            }, feature_path)

            node_base = nn.Sequential(
                nn.Linear(3, 5),
                nn.LayerNorm(5),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(5, 2),
            )
            state = {"node_base." + key: value for key, value in node_base.state_dict().items()}
            torch.save({
                "model_state_dict": state,
                "objective_revision": "multi_positive_target_attention_v2_train_graph_firewall_v3",
                "selected_by": "validation_mrr",
                "best_epoch": 2,
                "best_validation_mrr": 0.75,
            }, checkpoint_path)
            metrics_path.write_text(json.dumps({
                "dataset": {
                    "graph_split_audit": {
                        "policy": "train_samples_only_v3",
                        "passed": True,
                        "positive_edge_leakage_count": 0,
                        "negative_node_leakage_count": 0,
                    }
                },
                "model": {
                    "objective_revision": "multi_positive_target_attention_v2_train_graph_firewall_v3"
                },
                "evaluation_audit": {
                    "passed": True,
                    "evaluation_count": 1,
                    "test_evaluations_during_training": 0,
                    "checkpoint_selection_split": "validation",
                    "test_labels_used_for_selection": False,
                },
            }), encoding="utf-8")

            nodes_path.write_text(
                "".join(
                    json.dumps({
                        "attrs": {
                            "node_id": node_id,
                            "sample_id": node_id.split("::", 1)[0],
                            "node_type": node_id.split("::")[-2],
                        }
                    }) + "\n"
                    for node_id in node_ids
                ),
                encoding="utf-8",
            )

            subprocess.run([
                sys.executable,
                str(EXPORTER),
                "--checkpoint", str(checkpoint_path),
                "--stage3-metrics", str(metrics_path),
                "--node-features", str(feature_path),
                "--nodes", str(nodes_path),
                "--out-dir", str(out),
                "--device", "cpu",
                "--batch-size", "2",
                "--input-hash-policy", "sha256",
                "--require-node-metadata-coverage",
            ], check=True, capture_output=True, text=True)

            report = out / "verify.json"
            subprocess.run([
                sys.executable,
                str(VERIFIER),
                "--embeddings", str(out / "node_embeddings.pt"),
                "--index", str(out / "node_embedding_index.json"),
                "--manifest", str(out / "node_embedding_export_manifest.json"),
                "--report", str(report),
                "--require-formal",
            ], check=True, capture_output=True, text=True)

            verification = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(verification["passed"])
            self.assertEqual(verification["tensor"]["rows"], 3)
            self.assertEqual(verification["tensor"]["dimension"], 2)


if __name__ == "__main__":
    unittest.main()
