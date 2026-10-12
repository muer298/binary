#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = ("APISem", "TypeSem", "LayoutSem", "InterfaceSem")


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


def runtime_env():
    directories = {
        str(locate("module_common_223_v5_3.py").parent),
        str(locate("module_specialized_v6_6.py").parent),
    }
    return {**os.environ, "PYTHONPATH": os.pathsep.join(sorted(directories))}


def load_script(name):
    path = locate(name)
    for dependency in (
        locate("module_common_223_v5_3.py").parent,
        locate("module_specialized_v6_6.py").parent,
    ):
        if str(dependency) not in sys.path:
            sys.path.insert(0, str(dependency))
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ValidationGuardTest(unittest.TestCase):
    def test_validation_calibrated_module_and_direction_controls(self):
        collaboration = load_script(
            "20_iterative_module_collaboration_v6_6.py"
        )
        module_weights = collaboration.parse_module_weights(
            "APISem=0.2,InterfaceSem=0", 0.25
        )
        self.assertEqual(module_weights["APISem"], 0.2)
        self.assertEqual(module_weights["TypeSem"], 0.25)
        self.assertEqual(module_weights["InterfaceSem"], 0.0)
        direction_weights = collaboration.parse_direction_weights(
            "APISem->InterfaceSem=0,TypeSem->InterfaceSem=0"
        )
        self.assertEqual(
            direction_weights[("APISem", "InterfaceSem")], 0.0
        )
        self.assertFalse(collaboration.direction_enabled(
            ("APISem", "InterfaceSem"), "no_interface_incoming"
        ))
        self.assertFalse(collaboration.direction_enabled(
            ("TypeSem", "InterfaceSem"), "no_interface_incoming"
        ))

    def test_head_blend_is_selected_on_validation(self):
        trainer = load_script("18_train_specialised_module_heads_v6_6.py")
        # Rule scores rank the wrong candidate first; the learned head can
        # reverse it, so Validation must select the head instead of a constant.
        group = (
            [[1.0], [0.0]],
            [1, 0],
            ["gold", "wrong"],
            [0.1, 0.9],
        )
        result = trainer.train_one(
            [group] * 8,
            [group] * 4,
            dimension=1,
            epochs=30,
            learning_rate=0.1,
            seed=7,
            blend_grid=(0.0, 1.0),
        )
        self.assertEqual(result["blend_weight"], 1.0)
        self.assertEqual(
            result["blend_selection"],
            "per_module_validation_grid_search",
        )

    def test_formal_overall_is_query_module_micro_not_cross_module_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.jsonl"
            final_path = root / "final.jsonl"
            gold_path = root / "gold.jsonl"
            summary_path = root / "summary.json"
            output_path = root / "evaluation.json"
            candidates = [
                {
                    "target_id": "api-wrong",
                    "target_semantic_type": "APISem",
                    "confidence": 0.99,
                    "score_after_module_fusion_before_collaboration_223_v5_3": 0.99,
                    "score_223_v5_3": 0.99,
                },
                {
                    "target_id": "type-gold",
                    "target_semantic_type": "TypeSem",
                    "confidence": 0.80,
                    "score_after_module_fusion_before_collaboration_223_v5_3": 0.80,
                    "score_223_v5_3": 0.80,
                },
                {
                    "target_id": "type-wrong",
                    "target_semantic_type": "TypeSem",
                    "confidence": 0.70,
                    "score_after_module_fusion_before_collaboration_223_v5_3": 0.70,
                    "score_223_v5_3": 0.70,
                },
            ]
            row = {
                "query_id": "q",
                "candidates": candidates,
                "semantic_decisions": {
                    "APISem": {
                        "status": "hypothesis", "locked": False,
                        "propagation_weight": 0.0,
                    },
                    "TypeSem": {
                        "status": "hypothesis", "locked": False,
                        "propagation_weight": 0.0,
                    },
                },
            }
            write_jsonl(baseline_path, [row])
            write_jsonl(final_path, [row])
            write_jsonl(gold_path, [{
                "query_id": "q",
                "target_id": "type-gold",
                "target_semantic_type": "TypeSem",
                "split": "validation",
            }])
            summary_path.write_text(json.dumps({
                "ablation": "full",
                "locked_decision_violations": 0,
                "collaboration_edges": 1,
                "iterations_run": 1,
                "directions": {
                    "TypeSem->LayoutSem": 0.8,
                    "LayoutSem->TypeSem": 0.8,
                    "InterfaceSem->TypeSem": 0.45,
                },
                "signed_constraint_feedback": True,
                "suppressed_confirmed_source_count": 0,
                "consistency_trace": [
                    {"raw_s_global": 0.5}, {"raw_s_global": 0.5}
                ],
            }), encoding="utf-8")
            subprocess.run([
                sys.executable, str(locate("21_evaluate_223_v6_6.py")),
                "--baseline", str(baseline_path),
                "--input", str(final_path),
                "--collaboration-summary", str(summary_path),
                "--gold-jsonl", str(gold_path),
                "--out", str(output_path),
            ], check=True, capture_output=True, text=True, env=runtime_env())
            evaluation = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(evaluation["formal_ranking_unit"], "(query_id, semantic_type)")
            self.assertEqual(
                evaluation["ranking_overall"]["after_collaboration"]["n"], 1
            )
            self.assertEqual(
                evaluation["ranking_overall"]["after_collaboration"]["recall@1"],
                1.0,
            )
            self.assertEqual(
                evaluation["ranking_cross_module_flat_pool_diagnostic"]
                ["after_collaboration"]["recall@1"],
                0.0,
            )

    def test_per_module_regression_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation_path = root / "evaluation.json"
            summary_path = root / "summary.json"
            ranking_by_module = {}
            confirmation_by_module = {}
            for module in MODULES:
                final_r1 = 0.70 if module == "InterfaceSem" else 1.0
                ranking_by_module[module] = {
                    "before_223": {
                        "available": True, "recall@1": 1.0, "mrr": 1.0,
                    },
                    "after_collaboration": {
                        "available": True, "recall@1": final_r1,
                        "mrr": final_r1,
                    },
                }
                confirmation_by_module[module] = {
                    "confirmed_combined": {
                        "confirmed": 0,
                        "evaluable_confirmed": 0,
                        "precision": None,
                        "coverage_gold_objects": 0.0,
                    }
                }
            evaluation_path.write_text(json.dumps({
                "implementation_revision": "v6_7_validation_guarded",
                "records": 4,
                "pass_structural_checks": True,
                "checks": {"synthetic": True},
                "ablation": "full",
                "gold_queries_matched": 4,
                "ranking": {"available": True},
                "ranking_overall": {
                    "before_223": {"recall@1": 1.0, "mrr": 1.0},
                    "after_collaboration": {"recall@1": 0.95, "mrr": 0.95},
                },
                "ranking_by_module": ranking_by_module,
                "confirmation_quality": {"by_module": confirmation_by_module},
            }), encoding="utf-8")
            summary_path.write_text(json.dumps({
                "implementation_revision": "v6_7_validation_guarded",
                "ablation": "full",
                "locked_decision_violations": 0,
                "specialised_module_scores_consumed": True,
                "confirmed_input_decisions_irreversible": True,
            }), encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(locate("22_verify_223_v6_6.py")),
                "--evaluation", str(evaluation_path),
                "--summary", str(summary_path),
                "--require-ranking-nondegrading",
                "--require-module-ranking-nondegrading",
                "--confirmation-precision-floors",
                "APISem=0.95,TypeSem=0.95,LayoutSem=0.95,InterfaceSem=0.95",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("InterfaceSem recall@1 degraded", result.stdout)
            self.assertIn("VERIFY 2.2.3 RESULT: FAILED", result.stdout)

            payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
            payload["ranking_overall"]["after_collaboration"] = {
                "recall@1": 1.0, "mrr": 1.0,
            }
            payload["ranking_by_module"]["InterfaceSem"][
                "after_collaboration"
            ] = {"available": True, "recall@1": 1.0, "mrr": 1.0}
            evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
            passed = subprocess.run([
                sys.executable, str(locate("22_verify_223_v6_6.py")),
                "--evaluation", str(evaluation_path),
                "--summary", str(summary_path),
                "--require-ranking-nondegrading",
                "--require-module-ranking-nondegrading",
                "--confirmation-precision-floors",
                "APISem=0.95,TypeSem=0.95,LayoutSem=0.95,InterfaceSem=0.95",
            ], capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            self.assertIn("VERIFY 2.2.3 RESULT: PASSED", passed.stdout)


if __name__ == "__main__":
    unittest.main()
