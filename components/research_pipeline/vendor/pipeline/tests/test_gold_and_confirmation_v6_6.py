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


def write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class GoldAndConfirmationTest(unittest.TestCase):
    def test_split_guard_rejects_test_gold_in_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / "test.jsonl"
            write(gold, [{
                "query_id": "q", "target_id": "t",
                "target_semantic_type": "TypeSem", "split": "test",
            }])
            result = subprocess.run([
                sys.executable, str(locate("20_validate_gold_split_contract_v6_6.py")),
                "--gold", str(gold), "--expected-split", "validation",
                "--purpose", "calibration",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_gold_readiness_and_object_level_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes = root / "nodes.jsonl"
            index = root / "index.jsonl"
            reasoning = root / "reasoning.jsonl"
            gold_paths = {}
            node_rows = []
            gold_rows = []
            candidates = []
            decisions = {}
            for number, semantic_type in enumerate(("APISem", "TypeSem", "LayoutSem", "InterfaceSem")):
                query = "q%d" % number
                target = "t%d" % number
                node_rows.extend((
                    {"node_id": query, "node_type": "InstrSem", "sample_id": "s"},
                    {"node_id": target, "node_type": semantic_type, "sample_id": "s"},
                ))
                candidates.append({
                    "target_id": target, "target_semantic_type": semantic_type,
                    "confidence": 0.9, "validated_score": 0.9,
                    "evidence_coverage": 0.9, "candidate_status": "confirmed_soft",
                    "component_observed": {"direct_support": True, "constraint_consistency": True, "cross_object_collaboration": True},
                })
                decisions[semantic_type] = {"status": "confirmed_soft", "best_target_id": target, "confidence": 0.9}
                for split in ("train", "validation", "test"):
                    split_query = "%s-%s" % (split, query)
                    split_target = "%s-%s" % (split, target)
                    node_rows.extend((
                        {"node_id": split_query, "node_type": "InstrSem", "sample_id": "s"},
                        {"node_id": split_target, "node_type": semantic_type, "sample_id": "s"},
                    ))
                    gold_rows.append({
                        "query_id": split_query, "target_id": split_target,
                        "target_semantic_type": semantic_type, "sample_id": "s",
                        "group_id": "%s-%s" % (split, semantic_type), "split": split,
                        "gold_strength": "strict_independent",
                    })
            write(nodes, node_rows)
            write(index, [{"sample_id": "s"}])
            # One row may carry four semantic object decisions.
            write(reasoning, [{"query_id": "q0", "sample_id": "s", "candidates": candidates, "semantic_decisions": decisions}])
            for split in ("train", "validation", "test"):
                path = root / (split + ".jsonl")
                write(path, [row for row in gold_rows if row["split"] == split])
                gold_paths[split] = path

            # The synthetic Gold uses different queries per module, while the
            # reasoning row has q0; omit candidate coverage in this structural test.
            subprocess.run([
                sys.executable, str(locate("19_verify_gold_readiness_v6_6.py")),
                "--gold", str(gold_paths["train"]), "--gold", str(gold_paths["validation"]),
                "--gold", str(gold_paths["test"]), "--nodes", str(nodes),
                "--usg-index", str(index), "--expected-samples", "1",
                "--out", str(root / "ready.json"),
            ], check=True, capture_output=True, text=True)
            ready = json.loads((root / "ready.json").read_text(encoding="utf-8"))
            self.assertTrue(ready["dataset_ready"])

            dependency = str(locate("module_common_v5_3.py").parent)
            subprocess.run([
                sys.executable, str(locate("15_audit_confirmation_coverage_v6_6.py")),
                "--input", str(reasoning), "--out", str(root / "coverage.json"),
            ], check=True, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": dependency})
            coverage = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
            self.assertEqual(coverage["semantic_object_groups"], 4)
            self.assertEqual(coverage["confirmed_groups"], 4)


if __name__ == "__main__":
    unittest.main()
