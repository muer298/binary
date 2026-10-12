#!/usr/bin/env python3
from __future__ import annotations

import unittest

from layout_reranker_v6_7_2 import DEFAULT_CONFIG
from module_specialized_v6_6 import GraphContext, score_candidate


class LayoutModuleIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.context = GraphContext()
        self.context.nodes = {
            "q": {
                "node_id": "q",
                "node_type": "InstrSem",
                "attrs": {"observed_field_offsets": [0, 4, 8, 12]},
            },
            "correct": {
                "node_id": "correct",
                "node_type": "LayoutSem",
                "attrs": {"field_offsets": [0, 4, 8, 12], "layout_kind": "struct"},
            },
            "wrong": {
                "node_id": "wrong",
                "node_type": "LayoutSem",
                "attrs": {"field_offsets": [0, 8, 16, 24], "layout_kind": "struct"},
            },
            "type": {
                "node_id": "type",
                "node_type": "TypeSem",
                "attrs": {"type_category": "pointer", "width": 64},
            },
        }

    def candidate(self, target):
        return {
            "target_id": target,
            "target_semantic_type": "LayoutSem",
            "confidence": 0.70,
            "representation_score": 0.70,
            "direct_support_score": 0.5,
            "constraint_consistency_score": 0.5,
            "evidence_coverage": 0.5,
        }

    def test_layout_score_uses_query_node_and_candidate_node(self):
        query = {"query_id": "q", "sample_id": "s"}
        correct, correct_trace = score_candidate(
            "LayoutSem", query, self.candidate("correct"), self.context, {}, DEFAULT_CONFIG,
        )
        wrong, wrong_trace = score_candidate(
            "LayoutSem", query, self.candidate("wrong"), self.context, {}, DEFAULT_CONFIG,
        )
        self.assertGreater(correct, wrong)
        self.assertTrue(correct_trace["layout_pairwise_trace"]["usable"])
        self.assertEqual(
            correct_trace["feature_schema"],
            "specialised_v6_7_3_layout_evidence_bridge",
        )

    def test_non_layout_module_is_unchanged_by_layout_config(self):
        query = {"query_id": "q", "sample_id": "s"}
        candidate = {
            "target_id": "type",
            "target_semantic_type": "TypeSem",
            "confidence": 0.7,
            "representation_score": 0.7,
            "direct_support_score": 0.5,
            "constraint_consistency_score": 0.5,
            "evidence_coverage": 0.5,
        }
        first, _ = score_candidate("TypeSem", query, candidate, self.context, {}, DEFAULT_CONFIG)
        disabled = dict(DEFAULT_CONFIG, enabled=False, pairwise_blend=0.0)
        second, _ = score_candidate("TypeSem", query, candidate, self.context, {}, disabled)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
