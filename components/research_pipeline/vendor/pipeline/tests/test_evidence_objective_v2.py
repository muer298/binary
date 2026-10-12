#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False


TEST_FILE = Path(__file__).resolve()
ALIGNMENT_DIR = TEST_FILE.parent
if not (ALIGNMENT_DIR / "evidence_objective_v2.py").exists():
    ALIGNMENT_DIR = TEST_FILE.parents[1]
if str(ALIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ALIGNMENT_DIR))

if TORCH_AVAILABLE:
    from evidence_objective_v2 import (  # noqa: E402
        MultiPositiveEvidenceObjective,
        TargetEvidenceAttention,
        TargetGroupedBatchSampler,
        build_positive_mask,
        multi_positive_info_nce,
        resolve_evidence_record,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for evidence-objective tests")
class EvidenceSchemaCompatibilityTest(unittest.TestCase):
    def test_nested_schema_is_adapted_without_renaming_source_data(self):
        record = {
            "payload": {
                "edge": {
                    "src": {"node_id": "sampleA::InstrSem::i1"},
                    "dst": {"node_id": "sampleA::LayoutSem::l1"},
                    "relation_type": "fixed-offset",
                    "confidence": 0.8,
                }
            },
            "neighborhood": {
                "low_side": {"nodes": ["sampleA::InstrSem::i1", "sampleA::DFGSem::d1"]},
                "high_side": {"node_ids": ["sampleA::LayoutSem::l1"]},
            },
        }
        resolved = resolve_evidence_record(record, row_index=7)
        self.assertEqual(resolved["source_node"], "sampleA::InstrSem::i1")
        self.assertEqual(resolved["target_node"], "sampleA::LayoutSem::l1")
        self.assertEqual(resolved["evidence_type"], "fixed_offset")
        self.assertAlmostEqual(resolved["strength"], 0.8)
        self.assertEqual(resolved["sample_id"], "sampleA")
        self.assertEqual(resolved["target_group_source"], "target_node")
        self.assertIn("sampleA::DFGSem::d1", resolved["low_node_ids"])

    def test_missing_targets_are_never_merged(self):
        first = resolve_evidence_record({"anchor_id": "a", "evidence_type": "def_use"}, 0)
        second = resolve_evidence_record({"anchor_id": "a", "evidence_type": "def_use"}, 1)
        self.assertNotEqual(first["target_group_id"], second["target_group_id"])
        self.assertEqual(first["target_group_source"], "row_unique_fallback")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for evidence-objective tests")
class MultiPositiveLossTest(unittest.TestCase):
    def test_positive_mask_contains_all_same_target_pairs(self):
        mask = build_positive_mask(["A", "A", "B"])
        expected = torch.tensor(
            [[True, True, False], [True, True, False], [False, False, True]]
        )
        self.assertTrue(torch.equal(mask.cpu(), expected))

    def test_valid_alternative_evidence_is_not_a_false_negative(self):
        low = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), dim=-1)
        # The two A-target rows are deliberately swapped.  A diagonal-only
        # loss treats both correct A alternatives as errors.
        high = F.normalize(torch.tensor([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]), dim=-1)
        logits = low @ high.t() / 0.1
        labels = torch.arange(3)
        diagonal_loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        )
        multi_loss, stats = multi_positive_info_nce(
            low,
            high,
            ["A", "A", "B"],
            temperature=0.1,
        )
        self.assertLess(float(multi_loss), float(diagonal_loss))
        self.assertEqual(stats["non_diagonal_positive_pairs"], 2)
        self.assertTrue(stats["has_negative_targets"])

    def test_full_objective_has_finite_gradients(self):
        torch.manual_seed(3)
        support = torch.randn(4, 8, requires_grad=True)
        target = torch.randn(4, 8, requires_grad=True)
        objective = MultiPositiveEvidenceObjective(
            embed_dim=8,
            evidence_type_count=5,
            evidence_type_dim=4,
            attention_hidden_dim=12,
            attention_dropout=0.0,
        )
        loss, diagnostics, _ = objective(
            support,
            target,
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([0.9, 0.7, 0.8, 0.6]),
            ["A", "A", "B", "C"],
            torch.ones(5),
            temperature=0.2,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(support.grad).all())
        self.assertTrue(torch.isfinite(target.grad).all())
        self.assertEqual(diagnostics["target_groups_in_batch"], 3)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for evidence-objective tests")
class TargetEvidenceAttentionTest(unittest.TestCase):
    def test_greater_strength_has_greater_attention_for_equal_evidence(self):
        attention = TargetEvidenceAttention(
            embed_dim=4,
            evidence_type_count=5,
            evidence_type_dim=2,
            hidden_dim=6,
            dropout=0.0,
            strength_log_scale=1.0,
            attention_temperature=1.0,
        )
        for parameter in attention.parameters():
            torch.nn.init.zeros_(parameter)
        attention.eval()

        support = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        target = support.clone()
        result = attention(
            support,
            target,
            torch.tensor([0, 0]),
            torch.tensor([0.2, 0.8]),
            ["same", "same"],
        )
        weights = result["row_attention_weights"]
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertAlmostEqual(float(weights[1]), 0.8, places=5)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for evidence-objective tests")
class TargetGroupedSamplerTest(unittest.TestCase):
    def test_complete_target_groups_are_not_split(self):
        targets = ["A", "A", "A", "B", "B", "C"]
        sampler = TargetGroupedBatchSampler(
            targets,
            batch_size=4,
            shuffle=False,
            max_group_size=0,
        )
        batches = list(iter(sampler))
        batch_for_index = {}
        for batch_number, batch in enumerate(batches):
            for index in batch:
                batch_for_index[index] = batch_number
        for target in set(targets):
            memberships = {
                batch_for_index[index]
                for index, value in enumerate(targets)
                if value == target
            }
            self.assertEqual(len(memberships), 1)
        self.assertEqual(sampler.audit()["split_target_groups"], 0)


if __name__ == "__main__":
    unittest.main()
