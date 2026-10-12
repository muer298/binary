#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "24_calibrate_layout_reranker_v6_7_2.py"


class LayoutCalibrationTests(unittest.TestCase):
    def test_validation_selects_pairwise_reranker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module_result = root / "layout.jsonl"
            gold = root / "gold.jsonl"
            out = root / "config.json"
            rows = []
            gold_rows = []
            for index in range(120):
                query_id = "q%d" % index
                candidates = []
                for target_id, baseline, pairwise in (
                    ("wrong%d" % index, 0.9, 0.1),
                    ("gold%d" % index, 0.8, 0.9),
                ):
                    candidates.append({
                        "target_id": target_id,
                        "confidence_before_223": baseline,
                        "module_feature_trace_223_v6_6": {
                            "rule_score": baseline,
                            "layout_pairwise_trace": {
                                "legacy_score": baseline,
                                "pairwise_score": pairwise,
                                "usable": True,
                            },
                        },
                    })
                rows.append({
                    "query_id": query_id,
                    "semantic_type": "LayoutSem",
                    "candidate_scores": candidates,
                })
                gold_rows.append({
                    "query_id": query_id,
                    "target_id": "gold%d" % index,
                    "target_semantic_type": "LayoutSem",
                    "split": "validation",
                })
            module_result.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            gold.write_text(
                "".join(json.dumps(row) + "\n" for row in gold_rows),
                encoding="utf-8",
            )
            result = subprocess.run([
                sys.executable,
                str(SCRIPT),
                "--layout-module-result", str(module_result),
                "--gold-validation", str(gold),
                "--out", str(out),
            ], check=True, capture_output=True, text=True)
            config = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(config["selected_on_split"], "validation")
            self.assertFalse(config["test_used_for_selection"])
            self.assertGreater(config["pairwise_blend"], 0.0)
            self.assertEqual(
                config["selected_metrics_before_collaboration"]["all"]["recall@1"],
                1.0,
            )
            self.assertIn("VALIDATION CALIBRATION: PASSED", result.stdout)

    def test_test_gold_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module_result = root / "layout.jsonl"
            gold = root / "gold.jsonl"
            module_result.write_text(
                json.dumps({
                    "query_id": "q",
                    "semantic_type": "LayoutSem",
                    "candidate_scores": [],
                }) + "\n",
                encoding="utf-8",
            )
            gold.write_text(json.dumps({
                "query_id": "q",
                "target_id": "g",
                "target_semantic_type": "LayoutSem",
                "split": "test",
            }) + "\n", encoding="utf-8")
            result = subprocess.run([
                sys.executable,
                str(SCRIPT),
                "--layout-module-result", str(module_result),
                "--gold-validation", str(gold),
                "--out", str(root / "config.json"),
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected validation", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
