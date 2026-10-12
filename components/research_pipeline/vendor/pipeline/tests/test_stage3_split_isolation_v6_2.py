#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import ast
import sys
import tempfile
import unittest
from pathlib import Path


TEST_FILE = Path(__file__).resolve()
ALIGNMENT_DIR = TEST_FILE.parent
if not (ALIGNMENT_DIR / "split_isolation_v3.py").exists():
    ALIGNMENT_DIR = TEST_FILE.parents[1]
if str(ALIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ALIGNMENT_DIR))

from split_isolation_v3 import (  # noqa: E402
    SingleUseEvaluationGuard,
    load_split_manifest,
    partition_pairs_by_sample,
    resolve_edge_sample,
)


class FrozenSplitCompatibilityTest(unittest.TestCase):
    def test_nested_alias_manifest_and_partition(self):
        rows = [
            {"metadata": {"sample_hash": "s1", "subset": "training"}, "group_id": "g1"},
            {"sample_id": "s2", "split": "val", "group_id": "g2"},
            {"binary_id": "s3", "partition": "testing", "group_id": "g3"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "split.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            assignments, audit = load_split_manifest(str(path))
            self.assertEqual(
                assignments,
                {"s1": "train", "s2": "validation", "s3": "test"},
            )
            self.assertEqual(audit["group_overlap_count"], 0)

            pairs = [
                {"sample_id": "s1", "target_group_id": "s1::T"},
                {"sample_id": "s2", "target_group_id": "s2::T"},
                {"sample_id": "s3", "target_group_id": "s3::T"},
            ]
            partitions, mapping, split_audit = partition_pairs_by_sample(
                pairs,
                seed=7,
                train_ratio=0.7,
                validation_ratio=0.15,
                split_manifest=str(path),
            )
            self.assertEqual(partitions, {"train": [0], "validation": [1], "test": [2]})
            self.assertEqual(mapping, assignments)
            self.assertEqual(split_audit["split_source"], "frozen_manifest")
            self.assertEqual(split_audit["sample_overlap_count"], 0)

    def test_conflicting_group_assignments_fail(self):
        rows = [
            {"sample_id": "s1", "split": "train", "group_id": "same"},
            {"sample_id": "s2", "split": "test", "group_id": "same"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Group split leakage"):
                load_split_manifest(str(path))


class EdgeSampleResolutionTest(unittest.TestCase):
    def test_actual_usg_aliases_resolve_from_node_info(self):
        edge = {
            "payload": {
                "edge": {
                    "src": {"node_id": "s1::InstrSem::i"},
                    "dst": {"node_id": "s1::TypeSem::t"},
                }
            }
        }
        info = {
            "s1::InstrSem::i": {"sample_id": "s1"},
            "s1::TypeSem::t": {"sample_hash": "s1"},
        }
        resolved = resolve_edge_sample(edge, info)
        self.assertTrue(resolved["valid"])
        self.assertEqual(resolved["sample_id"], "s1")
        self.assertEqual(resolved["source_node"], "s1::InstrSem::i")
        self.assertEqual(resolved["target_node"], "s1::TypeSem::t")

    def test_cross_sample_edge_is_rejected(self):
        edge = {
            "source_node": "s1::InstrSem::i",
            "target_node": "s2::TypeSem::t",
        }
        resolved = resolve_edge_sample(edge)
        self.assertFalse(resolved["valid"])
        self.assertIn("cross_sample_endpoints", resolved["conflicts"])


class TestEvaluationPolicyTest(unittest.TestCase):
    def test_test_token_is_single_use(self):
        guard = SingleUseEvaluationGuard("test")
        guard.claim()
        self.assertEqual(guard.audit()["evaluation_count"], 1)
        with self.assertRaisesRegex(RuntimeError, "already executed"):
            guard.claim()


class Stage3SourceFirewallStaticTest(unittest.TestCase):
    def test_stage3_has_no_all_node_negative_fallback(self):
        source = (ALIGNMENT_DIR / "07_train_alignment_stage5_full.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("train_samples_only_v6_5", source)
        self.assertIn("SingleUseEvaluationGuard", source)
        self.assertIn("negative_node_leakage_count", source)
        self.assertNotIn("pool = list(range(node_count))", source)
        self.assertNotIn("chosen = random.randrange(node_count)", source)

    def test_test_split_has_exactly_one_eval_call(self):
        source = (ALIGNMENT_DIR / "07_train_alignment_stage5_full.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        test_calls = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "eval_model"
            ):
                if (
                    len(node.args) >= 4
                    and isinstance(node.args[3], ast.Name)
                    and node.args[3].id == "te"
                ):
                    test_calls.append(node)
        self.assertEqual(len(test_calls), 1)


if __name__ == "__main__":
    unittest.main()
