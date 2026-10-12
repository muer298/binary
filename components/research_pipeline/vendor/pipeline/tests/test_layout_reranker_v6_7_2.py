#!/usr/bin/env python3
from __future__ import annotations

import unittest

from layout_reranker_v6_7_2 import DEFAULT_CONFIG, score_layout_pair


def score(query_offsets, candidate_offsets, **target_attrs):
    query_node = {"attrs": {"observed_field_offsets": query_offsets}}
    target = {"attrs": {"field_offsets": candidate_offsets, **target_attrs}}
    return score_layout_pair(
        query_record={"query_id": "q"},
        query_node=query_node,
        candidate={"target_id": "c", "candidate_source": "same_sample_embedding"},
        target_node=target,
        query_edges=[],
        target_edges=[],
        pair_edges=[],
        legacy_score=0.5,
        config=DEFAULT_CONFIG,
    )


class LayoutRerankerTests(unittest.TestCase):
    def test_exact_layout_beats_wrong_stride(self):
        correct, correct_trace = score([0, 4, 8, 12], [0, 4, 8, 12])
        wrong, wrong_trace = score([0, 4, 8, 12], [0, 8, 16, 24])
        self.assertGreater(correct, wrong)
        self.assertEqual(correct_trace["features"]["observed_coverage"], 1.0)
        self.assertLess(wrong_trace["features"]["observed_coverage"], 1.0)

    def test_observed_subset_accepts_candidate_extra_fields(self):
        compatible, _ = score([4, 8, 12], [0, 4, 8, 12, 16])
        incompatible, _ = score([4, 8, 12], [0, 16, 32, 48])
        self.assertGreater(compatible, incompatible)

    def test_small_disassembly_error_uses_tolerance(self):
        close, close_trace = score([0, 8, 16], [0, 9, 17])
        far, _ = score([0, 8, 16], [0, 24, 48])
        self.assertGreater(close, far)
        self.assertGreater(close_trace["features"]["tolerant_f1"], close_trace["features"]["exact_f1"])

    def test_missing_query_offsets_is_exact_legacy_fallback(self):
        value, trace = score([], [0, 4, 8])
        self.assertEqual(value, 0.5)
        self.assertFalse(trace["usable"])
        self.assertEqual(trace["fallback_reason"], "query_offsets_missing")

    def test_inferred_only_offsets_are_penalised(self):
        clean, _ = score([0, 4, 8], [0, 4, 8])
        inferred, inferred_trace = score(
            [0, 4, 8],
            [0, 4, 8],
            inferred_field_offsets=[0, 4, 8],
        )
        self.assertGreater(clean, inferred)
        self.assertGreater(inferred_trace["features"]["invalid_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()

