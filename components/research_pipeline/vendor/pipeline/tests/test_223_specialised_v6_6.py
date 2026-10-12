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
    dirs = {str(locate("module_common_223_v5_3.py").parent), str(locate("module_specialized_v6_6.py").parent)}
    return {**os.environ, "PYTHONPATH": os.pathsep.join(sorted(dirs))}


class SpecialisedModulesTest(unittest.TestCase):
    def test_module_runner_uses_graph_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reasoning = root / "reasoning.jsonl"
            nodes = root / "nodes.jsonl"
            edges = root / "edges.jsonl"
            out = root / "modules"
            summary = root / "summary.json"
            candidates = []
            for semantic_type, target in (("APISem", "a"), ("TypeSem", "t"), ("LayoutSem", "l"), ("InterfaceSem", "i")):
                candidates.append({
                    "target_id": target, "target_semantic_type": semantic_type,
                    "confidence": 0.6, "representation_score": 0.6,
                    "direct_support_score": 0.5, "constraint_consistency_score": 0.5,
                    "evidence_coverage": 0.5, "candidate_status": "hypothesis",
                })
            reasoning.write_text(json.dumps({"query_id": "q", "sample_id": "s", "candidates": candidates, "semantic_decisions": {}}) + "\n", encoding="utf-8")
            node_rows = [
                {"node_id": "q", "node_type": "InstrSem", "sample_id": "s", "attrs": {"opcode": "load"}},
                {"node_id": "a", "node_type": "APISem", "sample_id": "s", "attrs": {"api_source": "symbol_table", "semantic_template": {"api_intent": "allocation"}}},
                {"node_id": "t", "node_type": "TypeSem", "sample_id": "s", "attrs": {"type_category": "pointer", "width": 64}},
                {"node_id": "l", "node_type": "LayoutSem", "sample_id": "s", "attrs": {"offsets": [0, 8], "layout_kind": "struct"}},
                {"node_id": "i", "node_type": "InterfaceSem", "sample_id": "s", "attrs": {"calling_convention": "sysv", "param_roles": ["buffer", "size"]}},
            ]
            nodes.write_text("".join(json.dumps(row) + "\n" for row in node_rows), encoding="utf-8")
            edge_rows = [
                {"source": "q", "target": target, "edge_kind": kind, "weight": 1.0}
                for target, kind in (("a", "api_call"), ("t", "deref_chain"), ("l", "fixed_offset"), ("i", "def_use"))
            ]
            edges.write_text("".join(json.dumps(row) + "\n" for row in edge_rows), encoding="utf-8")
            subprocess.run([
                sys.executable, str(locate("19_run_four_module_inference_v6_6.py")),
                "--reasoning", str(reasoning), "--nodes", str(nodes), "--edges", str(edges),
                "--out-dir", str(out), "--summary-out", str(summary),
            ], check=True, capture_output=True, text=True, env=env())
            report = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(report["loaded_node_metadata"], 5)
            self.assertEqual(set(report["module_counts"]), {"APIInfer", "TypeInfer", "LayoutInfer", "InterfaceInfer"})
            api = json.loads((out / "apiinfer_v6_6.jsonl").read_text(encoding="utf-8"))
            trace = api["candidate_scores"][0]["module_feature_trace_223_v6_6"]
            self.assertTrue(trace["target_node_metadata_available"])
            self.assertTrue(trace["frozen_221_representation_reused"])


if __name__ == "__main__":
    unittest.main()

