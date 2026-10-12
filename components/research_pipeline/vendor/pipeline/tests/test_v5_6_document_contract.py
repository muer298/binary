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


def locate(name):
    matches = [p for p in ROOT.rglob(name) if "98_历史版本与原始归档" not in p.parts and "tests" not in p.parts]
    if len(matches) != 1:
        raise FileNotFoundError((name, matches))
    return matches[0]


def env():
    dependency = str(locate("module_common_v5_3.py").parent)
    return {**os.environ, "PYTHONPATH": dependency}


class DocumentContractTest(unittest.TestCase):
    def test_v56_actual_consistency_and_confirmation_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = root / "validated.jsonl"
            edges = root / "edges.jsonl"
            index = root / "index.json"
            out = root / "out.jsonl"
            summary = root / "summary.json"
            row = {
                "query_id": "q", "sample_id": "s", "active_semantic_type": "APISem",
                "status": "confirmed_soft", "confidence": 0.9,
                "candidates": [{
                    "target_id": "api", "target_semantic_type": "APISem",
                    "candidate_status": "confirmed_soft", "validated_score": 0.9,
                    "confidence": 0.9, "representation_score": 0.9,
                    "evidence_coverage": 1.0, "selected_in_semantic_group": True,
                }],
                "semantic_decisions": {"APISem": {
                    "status": "confirmed_soft", "confidence": 0.9,
                    "best_target_id": "api", "locked": True,
                    "propagation_weight": 0.9,
                }},
            }
            validated.write_text(json.dumps(row) + "\n", encoding="utf-8")
            edges.write_text("", encoding="utf-8")
            index.write_text(json.dumps([
                {"node_id": "q", "node_type": "InstrSem", "sample_id": "s"},
                {"node_id": "api", "node_type": "APISem", "sample_id": "s"},
            ]), encoding="utf-8")
            subprocess.run([
                sys.executable, str(locate("10_iterative_reasoning_v5_6.py")),
                "--validated", str(validated), "--edges", str(edges),
                "--index", str(index), "--no-iterative-retrieval",
                "--max-iterations", "2", "--out", str(out),
                "--summary-out", str(summary),
            ], check=True, capture_output=True, text=True, env=env())
            report = json.loads(summary.read_text(encoding="utf-8"))
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["implementation_revision"], "v5_6_irreversible_confirmation_transactional_raw_sglobal")
            self.assertTrue(report["confirmed_states_irreversible"])
            self.assertTrue(report["raw_monotonic_transaction_enabled"])
            self.assertFalse(report["reported_global_consistency_is_monotonic_envelope"])
            self.assertEqual(result["semantic_decisions"]["APISem"]["status"], "confirmed_soft")


if __name__ == "__main__":
    unittest.main()

