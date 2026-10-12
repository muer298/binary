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
    matches = [
        path for path in ROOT.rglob(name)
        if "98_历史版本与原始归档" not in path.parts
        and "tests" not in path.parts
    ]
    if len(matches) != 1:
        raise FileNotFoundError((name, matches))
    return matches[0]


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class ConfirmationFailClosedTest(unittest.TestCase):
    def test_no_safe_layout_threshold_disables_soft_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = root / "validated.jsonl"
            gold = root / "gold_validation.jsonl"
            policy = root / "policy.json"
            report = root / "report.json"
            validated_rows = []
            gold_rows = []
            for index in range(24):
                query_id = "q%d" % index
                validated_rows.append({
                    "query_id": query_id,
                    "candidates": [
                        {
                            "target_id": "wrong%d" % index,
                            "target_semantic_type": "LayoutSem",
                            "validated_score": 0.99,
                            "evidence_coverage": 1.0,
                        },
                        {
                            "target_id": "gold%d" % index,
                            "target_semantic_type": "LayoutSem",
                            "validated_score": 0.50,
                            "evidence_coverage": 1.0,
                        },
                    ],
                })
                gold_rows.append({
                    "query_id": query_id,
                    "target_id": "gold%d" % index,
                    "target_semantic_type": "LayoutSem",
                    "split": "validation",
                })
            write_jsonl(validated, validated_rows)
            write_jsonl(gold, gold_rows)
            dependency = str(locate("module_common_v5_3.py").parent)
            subprocess.run([
                sys.executable,
                str(locate("14_calibrate_confirmation_policy_v5_4.py")),
                "--validated", str(validated),
                "--gold-jsonl", str(gold),
                "--out-policy", str(policy),
                "--out-report", str(report),
                "--min-confirmed", "5",
            ], check=True, capture_output=True, text=True, env={
                **os.environ, "PYTHONPATH": dependency,
            })
            payload = json.loads(policy.read_text(encoding="utf-8"))
            self.assertFalse(
                payload["policies"]["LayoutSem"][
                    "soft_confirmation_enabled"
                ]
            )
            audit = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(
                audit["by_module"]["LayoutSem"]["selection"],
                "fail_closed_no_grid_point_met_precision_floor",
            )

            check = subprocess.run([
                sys.executable, "-c",
                (
                    "from module_common_v5_3 import apply_confirm_policy,"
                    "determine_status;"
                    "apply_confirm_policy(%r);"
                    "r=determine_status('LayoutSem',{},0.99,0.99,"
                    "evidence_coverage=1.0);"
                    "assert r['status']=='hypothesis',r;"
                    "assert r['confirmation_source']=="
                    "'soft_confirmation_disabled_by_validation',r"
                ) % str(policy),
            ], capture_output=True, text=True, env={
                **os.environ, "PYTHONPATH": dependency,
            })
            self.assertEqual(check.returncode, 0, check.stderr)


if __name__ == "__main__":
    unittest.main()
