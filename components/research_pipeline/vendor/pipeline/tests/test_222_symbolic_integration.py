#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def locate(filename: str) -> Path:
    flat = SRC / filename
    if flat.is_file():
        return flat
    matches = [
        path for path in ROOT.rglob(filename)
        if "98_历史版本与原始归档" not in path.parts
        and "tests" not in path.parts
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected one active %s, found %s" % (filename, matches)
        )
    return matches[0]


def subprocess_env():
    dependency_dir = str(locate("module_common_v5_3.py").parent)
    existing = os.environ.get("PYTHONPATH", "")
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            item for item in (dependency_dir, existing) if item
        ),
    }


class ReasoningIntegrationTest(unittest.TestCase):
    def test_main_loop_runs_without_embeddings_in_fixed_candidate_ablation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = root / "validated.jsonl"
            edges = root / "edges.jsonl"
            index = root / "index.json"
            output = root / "reasoning.jsonl"
            summary = root / "summary.json"

            row = {
                "query_id": "q1",
                "sample_id": "s1",
                "evidence_type": "api_call",
                "active_semantic_type": "APISem",
                "status": "confirmed_hard",
                "confidence": 1.0,
                "candidates": [
                    {
                        "target_id": "api1",
                        "target_semantic_type": "APISem",
                        "candidate_status": "confirmed_hard",
                        "selected_in_semantic_group": True,
                        "validated_score": 1.0,
                        "confidence": 1.0,
                        "representation_score": 1.0,
                        "evidence_coverage": 1.0,
                        "api_source": "symbol_table",
                        "hard_api_eligible": True,
                    },
                    {
                        "target_id": "type1",
                        "target_semantic_type": "TypeSem",
                        "candidate_status": "hypothesis",
                        "validated_score": 0.60,
                        "confidence": 0.60,
                        "representation_score": 0.60,
                        "direct_support_score": 0.80,
                        "constraint_consistency_score": 0.0,
                        "evidence_coverage": 0.35,
                        "component_observed": {
                            "direct_support": True,
                            "constraint_consistency": False,
                            "cross_object_collaboration": False,
                        },
                        "representation_prior_weight": 0.25,
                    },
                    {
                        "target_id": "type2",
                        "target_semantic_type": "TypeSem",
                        "candidate_status": "hypothesis",
                        "validated_score": 0.40,
                        "confidence": 0.40,
                        "representation_score": 0.40,
                        "direct_support_score": 0.20,
                        "constraint_consistency_score": 0.0,
                        "evidence_coverage": 0.35,
                        "component_observed": {
                            "direct_support": True,
                            "constraint_consistency": False,
                            "cross_object_collaboration": False,
                        },
                        "representation_prior_weight": 0.25,
                    },
                ],
                "semantic_decisions": {
                    "APISem": {
                        "semantic_type": "APISem",
                        "status": "confirmed_hard",
                        "confidence": 1.0,
                        "best_target_id": "api1",
                        "confirmation_source": "symbol_table",
                        "confirmation_level": 3,
                        "propagation_weight": 1.0,
                        "locked": True,
                    },
                    "TypeSem": {
                        "semantic_type": "TypeSem",
                        "status": "hypothesis",
                        "confidence": 0.60,
                        "best_target_id": "type1",
                        "confirmation_source": "insufficient_unique_evidence",
                        "confirmation_level": 1,
                        "propagation_weight": 0.0,
                        "locked": False,
                    },
                },
            }
            validated.write_text(json.dumps(row) + "\n", encoding="utf-8")
            edges.write_text("\n".join((
                json.dumps({
                    "source": "api1", "target": "type1",
                    "edge_kind": "constraint", "constraint_level": 3,
                    "satisfaction": 1.0, "weight": 1.0,
                }),
                json.dumps({
                    "source": "api1", "target": "type2",
                    "edge_kind": "constraint", "constraint_level": 3,
                    "satisfaction": 0.0, "weight": 1.0,
                }),
            )) + "\n", encoding="utf-8")
            index.write_text(json.dumps([
                {"node_id": "q1", "node_type": "InstrSem", "sample_id": "s1"},
                {"node_id": "api1", "node_type": "APISem", "sample_id": "s1"},
                {"node_id": "type1", "node_type": "TypeSem", "sample_id": "s1"},
                {"node_id": "type2", "node_type": "TypeSem", "sample_id": "s1"},
            ]), encoding="utf-8")

            subprocess.run([
                sys.executable,
                str(locate("10_iterative_reasoning_v5_6.py")),
                "--validated", str(validated),
                "--edges", str(edges),
                "--index", str(index),
                "--no-iterative-retrieval",
                "--max-iterations", "3",
                "--global-threshold", "1.0",
                "--out", str(output),
                "--summary-out", str(summary),
            ], check=True, capture_output=True, text=True, env=subprocess_env())
            report = json.loads(summary.read_text(encoding="utf-8"))
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                report["implementation_revision"],
                "v5_6_irreversible_confirmation_transactional_raw_sglobal",
            )
            self.assertGreater(report["total_score_increases"], 0)
            self.assertEqual(
                result["semantic_decisions"]["TypeSem"]["best_target_id"],
                "type1",
            )
            rejected = next(
                value for value in result["candidates"]
                if value["target_id"] == "type2"
            )
            self.assertEqual(rejected["candidate_status"], "rejected")


if __name__ == "__main__":
    unittest.main()
