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
    dependency_dir = str(locate("module_common_223_v5_3.py").parent)
    existing = os.environ.get("PYTHONPATH", "")
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            item for item in (dependency_dir, existing) if item
        ),
    }
VARIANTS = (
    "full",
    "no_api_hard_seed",
    "no_type_layout",
    "no_interface_feedback",
    "no_collaboration",
)


def write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def candidate(target_id: str, semantic_type: str, score: float):
    return {
        "target_id": target_id,
        "target_semantic_type": semantic_type,
        "confidence": score,
        "validated_score": score,
        "representation_score": score,
        "direct_support_score": score,
        "constraint_consistency_score": score,
        "evidence_coverage": 0.6,
        "candidate_status": "hypothesis",
    }


class AblationIntegrationTest(unittest.TestCase):
    def test_all_variants_pass_structural_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reasoning = root / "reasoning.jsonl"
            edges = root / "edges.jsonl"
            gold = root / "gold.jsonl"
            modules = root / "modules"
            modules.mkdir()

            candidates = []
            for semantic_type, prefix, high in (
                ("APISem", "api", 0.95),
                ("TypeSem", "type", 0.68),
                ("LayoutSem", "layout", 0.66),
                ("InterfaceSem", "interface", 0.67),
            ):
                candidates.extend((
                    candidate(prefix + "1", semantic_type, high),
                    candidate(prefix + "2", semantic_type, 0.20),
                ))
            candidates[0]["candidate_status"] = "confirmed_hard"
            candidates[0]["api_source"] = "symbol_table"
            candidates[0]["hard_api_eligible"] = True
            row = {
                "query_id": "q1",
                "sample_id": "s1",
                "active_semantic_type": "APISem",
                "status": "confirmed_hard",
                "confidence": 1.0,
                "candidates": candidates,
                "semantic_decisions": {
                    "APISem": {
                        "semantic_type": "APISem",
                        "status": "confirmed_hard",
                        "confidence": 1.0,
                        "best_target_id": "api1",
                        "confirmation_source": "symbol_table",
                        "propagation_weight": 1.0,
                        "locked": True,
                    },
                    **{
                        semantic_type: {
                            "semantic_type": semantic_type,
                            "status": "hypothesis",
                            "confidence": 0.6,
                            "best_target_id": prefix + "1",
                            "propagation_weight": 0.0,
                            "locked": False,
                        }
                        for semantic_type, prefix in (
                            ("TypeSem", "type"),
                            ("LayoutSem", "layout"),
                            ("InterfaceSem", "interface"),
                        )
                    },
                },
            }
            write_jsonl(reasoning, [row])
            write_jsonl(edges, [
                {"source": "api1", "target": "type1", "edge_kind": "constraint", "weight": 1.0},
                {"source": "api1", "target": "interface1", "edge_kind": "constraint", "weight": 1.0},
                {"source": "type1", "target": "layout1", "edge_kind": "constraint", "weight": 1.0},
                {"source": "layout1", "target": "type1", "edge_kind": "constraint", "weight": 1.0},
                {"source": "type1", "target": "interface1", "edge_kind": "constraint", "weight": 1.0},
                {"source": "interface1", "target": "type1", "edge_kind": "constraint", "weight": 1.0},
            ])
            write_jsonl(gold, [
                {"query_id": "q1", "target_id": "api1", "semantic_type": "APISem"},
                {"query_id": "q1", "target_id": "type1", "semantic_type": "TypeSem"},
                {"query_id": "q1", "target_id": "layout1", "semantic_type": "LayoutSem"},
                {"query_id": "q1", "target_id": "interface1", "semantic_type": "InterfaceSem"},
            ])

            subprocess.run([
                sys.executable,
                str(locate("19_run_four_module_inference_v6_6.py")),
                "--reasoning", str(reasoning),
                "--out-dir", str(modules),
                "--summary-out", str(root / "module_summary.json"),
            ], check=True, capture_output=True, text=True, env=subprocess_env())
            module_args = []
            for module in (
                "apiinfer", "typeinfer", "layoutinfer", "interfaceinfer"
            ):
                module_args.extend((
                    "--module-result",
                    str(modules / (module + "_v6_6.jsonl")),
                ))

            for variant in VARIANTS:
                out = root / variant
                out.mkdir()
                result = out / "result.jsonl"
                summary = out / "summary.json"
                evaluation = out / "evaluation.json"
                subprocess.run([
                    sys.executable,
                    str(locate("20_iterative_module_collaboration_v6_6.py")),
                    "--reasoning", str(reasoning),
                    "--edges", str(edges),
                    *module_args,
                    "--ablation", variant,
                    "--out", str(result),
                    "--summary-out", str(summary),
                ], check=True, capture_output=True, text=True, env=subprocess_env())
                subprocess.run([
                    sys.executable,
                    str(locate("21_evaluate_223_v6_6.py")),
                    "--baseline", str(reasoning),
                    "--input", str(result),
                    "--collaboration-summary", str(summary),
                    "--gold-jsonl", str(gold),
                    "--ablation", variant,
                    "--out", str(evaluation),
                ], check=True, capture_output=True, text=True, env=subprocess_env())
                report = json.loads(evaluation.read_text(encoding="utf-8"))
                self.assertTrue(
                    report["pass_structural_checks"],
                    (variant, report["checks"]),
                )
                self.assertTrue(
                    report["ranking_by_module"]["TypeSem"]
                    ["after_collaboration"]["available"]
                )

                # Match the filenames consumed by the final suite summarizer.
                (out / "evaluation_v6_6.json").write_text(
                    evaluation.read_text(encoding="utf-8"), encoding="utf-8"
                )
                (out / "collaboration_summary_v6_6.json").write_text(
                    summary.read_text(encoding="utf-8"), encoding="utf-8"
                )

            summary_json = root / "ablation_summary.json"
            summary_csv = root / "ablation_summary.csv"
            subprocess.run([
                sys.executable,
                str(locate("23_summarize_223_ablations_v6_6.py")),
                "--suite-dir", str(root),
                "--json-out", str(summary_json),
                "--csv-out", str(summary_csv),
            ], check=True, capture_output=True, text=True, env=subprocess_env())
            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["rows"]), 25)
            self.assertTrue(summary_csv.is_file())


if __name__ == "__main__":
    unittest.main()
