#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from layout_reranker_v6_7_2 import DEFAULT_CONFIG, load_layout_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "25_verify_layout_rerank_v6_7_2.py"
MODULES = ("APISem", "TypeSem", "LayoutSem", "InterfaceSem")


def evaluation(layout_r1):
    by_module = {}
    for module in MODULES:
        r1 = layout_r1 if module == "LayoutSem" else 0.8
        by_module[module] = {
            "after_collaboration": {
                "n": 10,
                "recall@1": r1,
                "recall@5": 1.0,
                "mrr": (r1 + 1.0) / 2.0,
            }
        }
    return {
        "records": 10,
        "gold_queries_matched": 10,
        "gold_labels_matched": 10,
        "pass_structural_checks": True,
        "ranking_by_module": by_module,
        "ranking_overall": {
            "after_collaboration": {
                "recall@1": 0.8 + max(0.0, layout_r1 - 0.1) * 0.01,
                "mrr": 0.9 + max(0.0, layout_r1 - 0.1) * 0.01,
            }
        },
    }


class PromotionGateTests(unittest.TestCase):
    def test_improved_guarded_run_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(json.dumps(DEFAULT_CONFIG))
            config.update({
                "selected_on_split": "validation",
                "test_used_for_selection": False,
                "pairwise_blend": 0.75,
                "recommended_module_weight": 0.45,
            })
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            _, _, digest = load_layout_config(str(config_path))

            values = {
                "baseline.json": evaluation(0.10),
                "candidate.json": evaluation(0.30),
                "collaboration.json": {
                    "confirmation_policy_fail_closed_modules": ["APISem", "LayoutSem"],
                    "configured_direction_weights": {"TypeSem->InterfaceSem": 0.0},
                    "module_fusion_weights": {"LayoutSem": 0.45},
                },
                "four.json": {
                    "layout_reranker_revision": "v6_7_3_layout_evidence_bridge_reranker",
                    "layout_reranker_config_sha256": digest,
                    "layout_gold_used_during_inference": False,
                    "gold_used_during_inference": False,
                },
            }
            for name, value in values.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(SCRIPT),
                "--baseline-evaluation", str(root / "baseline.json"),
                "--candidate-evaluation", str(root / "candidate.json"),
                "--candidate-collaboration-summary", str(root / "collaboration.json"),
                "--candidate-four-module-summary", str(root / "four.json"),
                "--layout-config", str(config_path),
                "--out", str(root / "report.json"),
            ], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PROMOTION GATE: PASSED", result.stdout)


if __name__ == "__main__":
    unittest.main()
