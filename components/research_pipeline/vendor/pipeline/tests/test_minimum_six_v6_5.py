#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The six minimum behavioural tests required by the 2.2.1.4--2.2.1.6 guide."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # The deployment/server environment is expected to provide PyTorch.
    torch = None
    TORCH_AVAILABLE = False


HERE = Path(__file__).resolve().parent
RESOLVER_DIR = HERE if (HERE / "conflict_resolver_v6_5.py").is_file() else HERE.parent
SECTION_ROOT = RESOLVER_DIR.parent
ALIGNMENT_DIR = SECTION_ROOT / "2.2.1.4_非对称约束投影对齐"
BINDING_DIR = SECTION_ROOT / "2.2.1.5_对象级表示绑定与证据可追溯性保证"
if not ALIGNMENT_DIR.is_dir():
    ALIGNMENT_DIR = RESOLVER_DIR
if not BINDING_DIR.is_dir():
    BINDING_DIR = RESOLVER_DIR
for source_dir in (RESOLVER_DIR, ALIGNMENT_DIR, BINDING_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from binding_store_v6_4 import append_event, connect_store, trace_node  # noqa: E402

if TORCH_AVAILABLE:
    from conflict_resolver_v6_5 import (  # noqa: E402
        LEVEL_DETERMINISTIC,
        LEVEL_HIGH_CONFIDENCE,
        LEVEL_STATISTICAL,
        ConflictResolver,
        ConstraintSpec,
        make_reverse_vector_constraints,
    )
    from evidence_objective_v2 import TargetEvidenceAttention  # noqa: E402


def constraint(
    constraint_id: str,
    node_id: str,
    level: int,
    *,
    weight: float = 1.0,
    allowed=(),
    candidate_bias=None,
    vector=None,
    threshold: float = 0.0,
) -> ConstraintSpec:
    return ConstraintSpec(
        constraint_id=constraint_id,
        node_id=node_id,
        level=level,
        weight=weight,
        predicate_name="selected_in_allowed" if allowed else "always_true",
        projection_name="towards_constraint",
        allowed_candidates=tuple(allowed),
        candidate_bias=dict(candidate_bias or {}),
        constraint_vector=vector,
        vector_threshold=threshold,
        hard_source="deterministic_rule" if level == LEVEL_DETERMINISTIC else "",
        hard_source_verified=level == LEVEL_DETERMINISTIC,
        raw_hash=hashlib.sha256(constraint_id.encode("utf-8")).hexdigest(),
        raw={"constraint_id": constraint_id},
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for the six resolver tests")
class MinimumSixV65Test(unittest.TestCase):
    def test_01_asymmetric_target_aggregation_and_reverse_checks(self):
        evidence_records = [
            {"evidence_type": "fixed_offset", "source_node": "instr-1", "target_node": "layout-L"},
            {"evidence_type": "fixed_offset", "source_node": "instr-2", "target_node": "layout-L"},
            {"evidence_type": "fixed_offset", "source_node": "instr-3", "target_node": "layout-L"},
        ]
        self.assertEqual({row["evidence_type"] for row in evidence_records}, {"fixed_offset"})
        self.assertEqual({row["target_node"] for row in evidence_records}, {"layout-L"})
        attention = TargetEvidenceAttention(
            embed_dim=4,
            evidence_type_count=4,
            evidence_type_dim=2,
            hidden_dim=8,
            dropout=0.0,
            strength_log_scale=1.0,
            attention_temperature=1.0,
        )
        for parameter in attention.parameters():
            torch.nn.init.zeros_(parameter)
        support = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
        ])
        aggregate = attention(
            support,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
            # All three rows are the same fixed_offset evidence type; only
            # source vectors and strengths differ.
            torch.tensor([1, 1, 1]),
            torch.tensor([0.7, 0.8, 0.9]),
            ["layout-L", "layout-L", "layout-L"],
        )
        self.assertEqual(tuple(aggregate["support"].shape), (1, 4))
        self.assertEqual(aggregate["target_group_ids"], ["layout-L"])
        reverse = make_reverse_vector_constraints(
            "layout-L",
            [row["source_node"] for row in evidence_records],
            aggregate["target"][0],
        )
        self.assertEqual(len(reverse), 3)
        self.assertEqual({item.node_id for item in reverse}, {"instr-1", "instr-2", "instr-3"})

    def test_02_evidence_strength_controls_attention_weight(self):
        attention = TargetEvidenceAttention(
            embed_dim=4,
            evidence_type_count=2,
            evidence_type_dim=2,
            hidden_dim=6,
            dropout=0.0,
            strength_log_scale=1.0,
            attention_temperature=1.0,
        )
        for parameter in attention.parameters():
            torch.nn.init.zeros_(parameter)
        vector = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        aggregate = attention(
            vector,
            vector,
            torch.tensor([0, 0]),
            torch.tensor([0.2, 0.8]),
            ["same-target", "same-target"],
        )
        weights = aggregate["row_attention_weights"]
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_03_binding_is_immutable_event_is_append_only_and_trace_reaches_instruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "node_embedding_index.json"
            embeddings_path = root / "node_embeddings.pt"
            embedding_manifest = root / "node_embedding_export_manifest.json"
            nodes_path = root / "nodes.jsonl"
            edges_path = root / "edges.jsonl"
            store_path = root / "binding.sqlite3"
            manifest_path = root / "binding_manifest.json"
            instr = "sampleA::InstrSem::i1"
            type_node = "sampleA::TypeSem::t1"
            index_rows = [
                {"index": 0, "node_id": instr, "node_type": "InstrSem", "sample_id": "sampleA"},
                {"index": 1, "node_id": type_node, "node_type": "TypeSem", "sample_id": "sampleA"},
            ]
            index_path.write_text(json.dumps(index_rows), encoding="utf-8")
            embeddings_path.write_bytes(b"minimum-six-placeholder")

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            embedding_manifest.write_text(json.dumps({
                "schema": "task2_node_embedding_export_v6_3",
                "formal_ready": False,
                "outputs": {
                    "index": {"sha256": digest(index_path)},
                    "embeddings": {"sha256": digest(embeddings_path)},
                },
                "inputs": {"checkpoint": {"path": "best.pt", "sha256": "a" * 64}},
                "checkpoint": {"objective_revision": "minimum_six_test"},
            }), encoding="utf-8")
            nodes_path.write_text(
                json.dumps({"node_id": instr, "node_type": "InstrSem", "sample_id": "sampleA"}) + "\n"
                + json.dumps({"node_id": type_node, "node_type": "TypeSem", "sample_id": "sampleA"}) + "\n",
                encoding="utf-8",
            )
            edges_path.write_text(json.dumps({
                "edge_id": "ev-instr-type",
                "edge_kind": "evidence",
                "evidence_type": "def_use",
                "source_node": instr,
                "target_node": type_node,
                "strength": 0.9,
            }) + "\n", encoding="utf-8")
            builder = BINDING_DIR / "12_build_embedding_binding_records.py"
            subprocess.run([
                sys.executable,
                str(builder),
                "--embedding-index", str(index_path),
                "--embedding-manifest", str(embedding_manifest),
                "--embeddings", str(embeddings_path),
                "--nodes", str(nodes_path),
                "--edges", str(edges_path),
                "--store", str(store_path),
                "--manifest", str(manifest_path),
            ], check=True, capture_output=True, text=True)
            connection = connect_store(store_path)
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute(
                    "UPDATE bindings SET initial_state='rejected' WHERE node_id=?", (type_node,)
                )
            connection.rollback()
            event = append_event(connection, {
                "node_id": type_node,
                "step": 1,
                "reason": "mask_project_blend_resolution",
                "new_state": "confirmed_soft",
                "new_confidence": 0.88,
                "new_embedding_ref": "resolved_embeddings.pt#row=1",
                "evidence_edge_ids": ["ev-instr-type"],
                "payload": {"resolution_level": 2, "iteration": 1},
                "created_at": "2026-01-01T00:00:00Z",
            })
            self.assertEqual(event["sequence"], 1)
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute("DELETE FROM update_events WHERE event_id=?", (event["event_id"],))
            connection.rollback()
            trace = trace_node(connection, type_node, max_depth=2)
            self.assertEqual(trace["update_timeline"][0]["reason"], "mask_project_blend_resolution")
            self.assertIn(instr, [item["to_node"] for item in trace["recursive_lineage"]])
            connection.close()

    def test_04_hard_pointer_mask_wins_over_statistical_integer_bias(self):
        node_id = "sampleA::TypeSem::t1"
        result = ConflictResolver(max_iterations=4).resolve(
            node_id,
            {"int64": 0.55, "char *": 0.45},
            torch.tensor([0.0, 1.0]),
            [
                constraint("hard-pointer", node_id, LEVEL_DETERMINISTIC, allowed=("ptr",)),
                constraint(
                    "soft-integer",
                    node_id,
                    LEVEL_STATISTICAL,
                    weight=1.0,
                    candidate_bias={"int": 1000.0},
                    vector=torch.tensor([1.0, 0.0]),
                ),
            ],
        )
        self.assertEqual(result["selected_candidate"], "char *")
        self.assertEqual(result["state"], "confirmed_hard")
        self.assertEqual(result["final_candidates"]["int64"], 0.0)
        self.assertTrue(result["checks"]["lower_levels_did_not_unmask_hard_candidates"])

    def test_05_all_hard_candidates_masked_creates_explicit_hypothesis_without_nan(self):
        node_id = "sampleA::TypeSem::t2"
        result = ConflictResolver(max_iterations=3).resolve(
            node_id,
            {"int32": 0.6, "char *": 0.4},
            torch.tensor([1.0, 0.0]),
            [
                constraint("hard-pointer", node_id, LEVEL_DETERMINISTIC, allowed=("ptr",)),
                constraint("hard-float", node_id, LEVEL_DETERMINISTIC, allowed=("float",)),
            ],
        )
        self.assertTrue(result["unresolved_hard_conflict"])
        self.assertEqual(result["state"], "hypothesis")
        self.assertTrue(result["new_hypotheses"])
        self.assertTrue(result["checks"]["probabilities_finite"])
        self.assertTrue(result["checks"]["probabilities_sum_to_one"])

    def test_06_mutually_conflicting_projections_stop_at_max_iterations_unresolved(self):
        node_id = "sampleA::LayoutSem::l1"
        result = ConflictResolver(max_iterations=3, epsilon=1e-8).resolve(
            node_id,
            {"layout_a": 0.5, "layout_b": 0.5},
            torch.tensor([0.0, 1.0]),
            [
                constraint(
                    "project-positive",
                    node_id,
                    LEVEL_HIGH_CONFIDENCE,
                    vector=torch.tensor([1.0, 0.0]),
                    threshold=0.99,
                ),
                constraint(
                    "project-negative",
                    node_id,
                    LEVEL_HIGH_CONFIDENCE,
                    vector=torch.tensor([-1.0, 0.0]),
                    threshold=0.99,
                ),
            ],
        )
        self.assertTrue(result["projection_incompatibilities"])
        self.assertEqual(result["stop_reason"], "max_iterations")
        self.assertEqual(result["iterations_run"], 3)
        self.assertGreater(result["active_conflict_count"], 0)
        self.assertEqual(result["state"], "hypothesis")


if __name__ == "__main__":
    unittest.main(verbosity=2)
