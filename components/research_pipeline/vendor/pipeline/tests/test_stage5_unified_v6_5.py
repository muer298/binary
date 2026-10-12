#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for the unified Stage5 objective and split firewall."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ALIGNMENT_DIR = HERE if (HERE / "07_train_alignment_stage5_full.py").is_file() else HERE.parent
STAGE5_PATH = ALIGNMENT_DIR / "07_train_alignment_stage5_full.py"
EXPORTER_PATH = ALIGNMENT_DIR / "10_export_node_embeddings_v6_3.py"


def load_source() -> str:
    return STAGE5_PATH.read_text(encoding="utf-8")


class Stage5StaticContractTest(unittest.TestCase):
    def test_unified_three_loss_and_export_contract_are_present(self):
        source = load_source()
        required = (
            'OBJECTIVE_REVISION = "stage5_unified_three_loss_train_graph_v6_5"',
            "MultiPositiveEvidenceObjective",
            "TargetGroupedBatchSampler",
            "self.node_base",
            "low_base = self.enc_node",
            "high_base = self.enc_node",
            "constraint_margin_loss",
            "edge_reconstruction_loss",
            "scheduled_constraint_weight",
            "sinusoidal_position_encoding",
            "constraint_warmup_epochs",
            "constraint_ramp_epochs",
            "train_samples_only_v6_5",
            "SingleUseEvaluationGuard",
            '"checkpoint_selection_split": "validation"',
            '"test_labels_used_for_selection": False',
        )
        for marker in required:
            self.assertIn(marker, source, marker)

    def test_formal_exporter_accepts_the_unified_checkpoint(self):
        source = EXPORTER_PATH.read_text(encoding="utf-8")
        self.assertIn("stage5_unified_three_loss_train_graph_v6_5", source)
        self.assertIn("train_samples_only_v6_5", source)


try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for Stage5 gradient tests")
class Stage5RuntimeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(ALIGNMENT_DIR) not in sys.path:
            sys.path.insert(0, str(ALIGNMENT_DIR))
        spec = importlib.util.spec_from_file_location("stage5_v6_5", STAGE5_PATH)
        cls.stage5 = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.stage5)

    def test_train_graph_contains_train_edges_and_negative_pool_only(self):
        module = self.stage5
        node_ids = [
            "trainA::InstrSem::i1",
            "trainA::TypeSem::t1",
            "validA::InstrSem::i1",
            "validA::TypeSem::t1",
            "testA::InstrSem::i1",
            "testA::TypeSem::t1",
        ]
        feature_payload = {"node_ids": node_ids, "node_index": {v: i for i, v in enumerate(node_ids)}}
        node_info = {
            value: {
                "sample_id": value.split("::", 1)[0],
                "node_type": value.split("::")[1],
            }
            for value in node_ids
        }
        graph = module.GraphStore()
        for start in (0, 2, 4):
            graph.add(start, start + 1, "constraint:type", "constraint")
        split = {"trainA": "train", "validA": "validation", "testA": "test"}
        filtered = module.build_train_loss_graph(graph, feature_payload, node_info, split)
        self.assertEqual(len(filtered.edges), 1)
        self.assertEqual(filtered.edges[0]["sample_id"], "trainA")
        self.assertEqual(set(filtered.train_node_samples.values()), {"trainA"})
        self.assertEqual(filtered.audit["positive_edge_leakage_count"], 0)
        self.assertEqual(filtered.audit["negative_node_leakage_count"], 0)

    def test_evidence_first_constraint_schedule(self):
        schedule = self.stage5.scheduled_constraint_weight
        self.assertEqual(schedule(1, 8, 0.0, 0.5, 2, 4), 0.0)
        self.assertEqual(schedule(2, 8, 0.0, 0.5, 2, 4), 0.0)
        self.assertAlmostEqual(schedule(4, 8, 0.0, 0.5, 2, 4), 0.25)
        self.assertAlmostEqual(schedule(6, 8, 0.0, 0.5, 2, 4), 0.5)

    def test_instruction_position_signal_is_not_constant(self):
        encoding = self.stage5.sinusoidal_position_encoding(
            3, 8, torch.device("cpu"), torch.float32
        )
        self.assertEqual(tuple(encoding.shape), (3, 8))
        self.assertFalse(torch.allclose(encoding[0], encoding[1]))

    def test_evidence_loss_backpropagates_into_exportable_node_base(self):
        module = self.stage5
        torch.manual_seed(7)
        model = module.Stage5FullModel(
            input_dim=4,
            hidden_dim=8,
            embed_dim=4,
            etype_dim=2,
            rel_count=1,
            rel_dim=4,
            nhead=2,
            transformer_layers=1,
            gnn_layers=1,
            dropout=0.0,
            attention_hidden_dim=8,
            attention_dropout=0.0,
        )
        features = torch.randn(4, 4)
        records = [
            {
                "instr_idx": [0], "low_idx": [0], "high_idx": [1],
                "low_edges": [], "high_edges": [], "etype_idx": 0,
                "strength": 0.9, "target_group_id": "target-A",
            },
            {
                "instr_idx": [2], "low_idx": [2], "high_idx": [3],
                "low_edges": [], "high_edges": [], "etype_idx": 1,
                "strength": 0.8, "target_group_id": "target-B",
            },
        ]
        low, high, _ = model.encode_batch(features, records, torch.device("cpu"))
        loss, _, _ = model.evidence_objective(
            low,
            high,
            torch.tensor([0, 1]),
            torch.tensor([0.9, 0.8]),
            ["target-A", "target-B"],
            torch.ones(len(module.EVIDENCE_TYPES)),
        )
        loss.backward()
        gradients = [parameter.grad for parameter in model.node_base.parameters()]
        self.assertTrue(all(value is not None for value in gradients))
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
