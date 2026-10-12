import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layout_reranker_v6_7_2 import (  # noqa: E402
    DEFAULT_CONFIG,
    EVIDENCE_BRIDGE_REVISION,
    IMPLEMENTATION_REVISION,
    score_layout_pair,
)


class NestedEvidenceBridgeTests(unittest.TestCase):
    def score(self, query_access):
        return score_layout_pair(
            query_record={
                "raw": {
                    "effects": {
                        "memory_accesses": [query_access],
                    },
                },
            },
            query_node={},
            candidate={},
            target_node={
                "raw": {
                    "fields": [
                        {"offset": 0, "byte_width": 4},
                        {"offset": "0x8", "byte_width": 4},
                    ],
                },
            },
            query_edges=[],
            target_edges=[],
            pair_edges=[],
            legacy_score=0.25,
            config=DEFAULT_CONFIG,
        )

    def test_nested_memory_and_field_lists_are_read(self):
        _, trace = self.score({
            "offset": "0x8",
            "byte_width": 4,
            "is_fixed": True,
            "access": "load",
        })

        self.assertTrue(trace["usable"])
        self.assertEqual(trace["query_offsets"], [8])
        self.assertEqual(trace["candidate_offsets"], [0, 8])
        self.assertEqual(trace["query_widths"], [4])
        self.assertEqual(trace["candidate_widths"], [4])
        self.assertIn(
            "query_row.memory_accesses[].offset",
            trace["query_offset_sources"],
        )
        self.assertIn(
            "target_node.fields[].offset",
            trace["candidate_offset_sources"],
        )
        self.assertEqual(trace["implementation_revision"], IMPLEMENTATION_REVISION)
        self.assertEqual(
            trace["layout_evidence_bridge_revision"],
            EVIDENCE_BRIDGE_REVISION,
        )
        self.assertFalse(trace["gold_used"])
        self.assertFalse(trace["test_used"])

    def test_explicitly_dynamic_access_is_not_promoted(self):
        score, trace = self.score({
            "offset": 8,
            "byte_width": 4,
            "is_fixed": False,
        })

        self.assertFalse(trace["usable"])
        self.assertEqual(trace["fallback_reason"], "query_offsets_missing")
        self.assertEqual(score, 0.25)


if __name__ == "__main__":
    unittest.main()
