#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
# In the bundle this test lives in ``2.2.1.5/tests``; the deployment script
# intentionally flattens it beside the active sources in ``task2_exp/src``.
SRC = HERE if (HERE / "binding_store_v6_4.py").is_file() else HERE.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from binding_store_v6_4 import (  # noqa: E402
    BindingStoreError, ConcurrencyError, append_event, connect_store,
    rebuild_heads, trace_node,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BindingStoreV64Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.index = self.root / "node_embedding_index.json"
        self.embeddings = self.root / "node_embeddings.pt"
        self.embedding_manifest = self.root / "node_embedding_export_manifest.json"
        self.nodes = self.root / "nodes.jsonl"
        self.edges = self.root / "edges.jsonl"
        self.store = self.root / "binding.sqlite3"
        self.manifest = self.root / "binding_manifest.json"
        self.report = self.root / "verify.json"

        self.instr = "sampleA::InstrSem::i1"
        self.type_node = "sampleA::TypeSem::t1"
        self.layout = "sampleA::LayoutSem::l1"
        rows = [
            {"index": 0, "node_id": self.instr, "node_type": "InstrSem", "sample_id": "sampleA", "embedding_ref": "node_embeddings.pt#row=0"},
            {"index": 1, "node_id": self.type_node, "node_type": "TypeSem", "sample_id": "sampleA", "embedding_ref": "node_embeddings.pt#row=1"},
            {"index": 2, "node_id": self.layout, "node_type": "LayoutSem", "sample_id": "sampleA", "embedding_ref": "node_embeddings.pt#row=2"},
        ]
        self.index.write_text(json.dumps(rows), encoding="utf-8")
        self.embeddings.write_bytes(b"test tensor placeholder")
        self.embedding_manifest.write_text(json.dumps({
            "schema": "task2_node_embedding_export_v6_3",
            "formal_ready": False,
            "outputs": {
                "index": {"sha256": sha256(self.index)},
                "embeddings": {"sha256": sha256(self.embeddings)},
            },
            "inputs": {"checkpoint": {"path": "best.pt", "sha256": "a" * 64}},
            "checkpoint": {"objective_revision": "test_objective"},
        }), encoding="utf-8")
        node_rows = [
            {"record": {"node_id": self.instr, "node_type": "InstrSem", "sample_id": "sampleA"}},
            {"payload": {"node_ref": self.type_node, "semantic_type": "TypeSem", "sample_hash": "sampleA"}},
            {"node_id": self.layout, "node_type": "LayoutSem", "sample_id": "sampleA"},
        ]
        self.nodes.write_text("".join(json.dumps(x) + "\n" for x in node_rows), encoding="utf-8")
        edge_rows = [
            {"edge_id": "ev1", "edge_kind": "evidence", "evidence_type": "def_use", "src": self.instr, "dst": self.type_node, "strength": 0.9},
            {"edge": {"edge_id": "c1", "edge_kind": "constraint", "constraint_type": "type_layout", "source_node": self.type_node, "target_node": self.layout, "confidence": 1.0}},
        ]
        self.edges.write_text("".join(json.dumps(x) + "\n" for x in edge_rows), encoding="utf-8")
        subprocess.run([
            sys.executable, str(SRC / "12_build_embedding_binding_records.py"),
            "--embedding-index", str(self.index),
            "--embedding-manifest", str(self.embedding_manifest),
            "--embeddings", str(self.embeddings),
            "--nodes", str(self.nodes),
            "--edges", str(self.edges),
            "--store", str(self.store),
            "--manifest", str(self.manifest),
        ], check=True, capture_output=True, text=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_node_level_genesis_and_immutable_trigger(self):
        connection = connect_store(self.store)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM bindings").fetchone()[0], 3)
        source = connection.execute(
            "SELECT source_index_row,source_node_line,source_index_record_hash,"
            "source_node_record_hash FROM bindings WHERE node_id=?", (self.type_node,)
        ).fetchone()
        self.assertEqual(source["source_index_row"], 1)
        self.assertEqual(source["source_node_line"], 2)
        self.assertEqual(len(source["source_index_record_hash"]), 64)
        self.assertEqual(len(source["source_node_record_hash"]), 64)
        with self.assertRaises(sqlite3.DatabaseError):
            connection.execute(
                "UPDATE bindings SET node_type='X' WHERE node_id=?", (self.type_node,)
            )
        connection.rollback()
        with self.assertRaises(sqlite3.DatabaseError):
            connection.execute(
                "UPDATE metadata SET value_json='null' WHERE key='nodes_path'"
            )
        connection.rollback()
        connection.close()

    def test_append_only_event_and_stale_write_guard(self):
        connection = connect_store(self.store)
        first = append_event(connection, {
            "node_id": self.type_node,
            "step": 0,
            "reason": "evidence_projection",
            "new_state": "confirmed_soft",
            "new_confidence": 0.81,
            "loss": 0.13,
            "evidence_edge_ids": ["ev1"],
            "constraint_edge_ids": ["c1"],
            # A resolver may discover a new provenance edge while producing
            # the update.  The edge, its node references, the event and the
            # derived head must be committed by the same transaction.
            "new_edges": [{
                "edge_kind": "constraint",
                "constraint_type": "derived_api_type_rule",
                "source_node": self.instr,
                "target_node": self.type_node,
                "confidence": 0.88,
                "source_path": "resolver:test",
            }],
            "created_at": "2026-01-01T00:00:00Z",
        })
        second = append_event(connection, {
            "node_id": self.type_node,
            "step": 1,
            "reason": "confidence_refinement",
            "expected_old_state": "confirmed_soft",
            "expected_old_confidence": 0.81,
            "new_confidence": 0.84,
            "created_at": "2026-01-01T00:00:01Z",
        })
        self.assertEqual(first["sequence"], 1)
        self.assertAlmostEqual(first["payload"]["loss"], 0.13)
        self.assertEqual(len(first["new_edge_ids"]), 1)
        self.assertEqual(second["sequence"], 2)
        self.assertEqual(second["prev_event_hash"], first["event_hash"])
        derived_edge_id = first["new_edge_ids"][0]
        derived = connection.execute(
            "SELECT edge_class,source_node,target_node FROM edge_catalog WHERE edge_id=?",
            (derived_edge_id,),
        ).fetchone()
        self.assertEqual(derived["edge_class"], "constraint")
        self.assertEqual(derived["source_node"], self.instr)
        self.assertEqual(derived["target_node"], self.type_node)
        self.assertIsNotNone(connection.execute(
            "SELECT 1 FROM event_edge_refs WHERE event_id=? AND edge_id=?",
            (first["event_id"], derived_edge_id),
        ).fetchone())
        with self.assertRaises(ConcurrencyError):
            append_event(connection, {
                "node_id": self.type_node,
                "step": 2,
                "reason": "stale_writer",
                "expected_old_confidence": 0.10,
                "new_confidence": 0.9,
            })
        with self.assertRaises(sqlite3.DatabaseError):
            connection.execute(
                "DELETE FROM update_events WHERE event_id=?", (first["event_id"],)
            )
        connection.rollback()
        connection.close()

    def test_failed_event_rolls_back_new_provenance_edge(self):
        connection = connect_store(self.store)
        raw_edge = {
            "edge_id": "derived-will-rollback",
            "edge_kind": "evidence",
            "evidence_type": "derived_support",
            "source_node": self.instr,
            "target_node": self.type_node,
        }
        # Detection happens after the new edge INSERT.  The conflicting class
        # therefore exercises rollback of edge + refs + event + head.
        with self.assertRaises(BindingStoreError):
            append_event(connection, {
                "node_id": self.type_node,
                "step": 0,
                "reason": "invalid_mixed_class_event_with_edge",
                "new_state": "confirmed_soft",
                "new_confidence": 0.8,
                "constraint_edge_ids": ["derived-will-rollback"],
                "new_edges": [raw_edge],
            })
        self.assertIsNone(connection.execute(
            "SELECT 1 FROM edge_catalog WHERE edge_id='derived-will-rollback'"
        ).fetchone())
        connection.close()

    def test_trace_returns_original_instruction_and_timeline(self):
        connection = connect_store(self.store)
        append_event(connection, {
            "node_id": self.type_node,
            "step": 0,
            "reason": "def_use_support",
            "new_state": "confirmed_soft",
            "new_confidence": 0.8,
            "evidence_edge_ids": ["ev1"],
            "created_at": "2026-01-01T00:00:00Z",
        })
        trace = trace_node(connection, self.type_node, max_depth=2)
        self.assertEqual(trace["binding_record"]["node_id"], self.type_node)
        self.assertEqual(trace["source_artifacts"]["nodes_path"], str(self.nodes.resolve()))
        self.assertEqual(trace["update_timeline"][0]["reason"], "def_use_support")
        source_ids = [
            edge.get("source_node_binding", {}).get("node_id")
            for edge in trace["evidence_edges"]
        ]
        self.assertIn(self.instr, source_ids)
        lineage_ids = [item["to_node"] for item in trace["recursive_lineage"]]
        self.assertIn(self.instr, lineage_ids)
        connection.close()

    def test_head_cache_is_rebuildable(self):
        connection = connect_store(self.store)
        append_event(connection, {
            "node_id": self.layout,
            "step": 0,
            "reason": "constraint_support",
            "new_state": "confirmed_soft",
            "new_confidence": 0.76,
            "constraint_edge_ids": ["c1"],
            "created_at": "2026-01-01T00:00:00Z",
        })
        self.assertEqual(rebuild_heads(connection), 1)
        head = connection.execute(
            "SELECT state,confidence,event_count FROM binding_heads WHERE node_id=?",
            (self.layout,),
        ).fetchone()
        self.assertEqual(head["state"], "confirmed_soft")
        self.assertAlmostEqual(head["confidence"], 0.76)
        self.assertEqual(head["event_count"], 1)
        connection.close()

    def test_batch_dry_run_simulates_dependencies_without_persisting(self):
        events_path = self.root / "events.jsonl"
        dry_report = self.root / "dry_report.json"
        events = [
            {
                "node_id": self.type_node,
                "step": 0,
                "reason": "first_provisional_event",
                "expected_old_state": "unresolved",
                "new_state": "confirmed_soft",
                "new_confidence": 0.80,
                "evidence_edge_ids": ["ev1"],
            },
            {
                "node_id": self.type_node,
                "step": 1,
                "reason": "dependent_provisional_event",
                "expected_old_state": "confirmed_soft",
                "expected_old_confidence": 0.80,
                "new_confidence": 0.83,
            },
        ]
        events_path.write_text(
            "".join(json.dumps(item) + "\n" for item in events), encoding="utf-8"
        )
        subprocess.run([
            sys.executable, str(SRC / "13_append_binding_events_v6_4.py"),
            "--store", str(self.store),
            "--events", str(events_path),
            "--report", str(dry_report),
            "--dry-run",
        ], check=True, capture_output=True, text=True)
        report = json.loads(dry_report.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertEqual(report["counts"]["validated"], 2)
        self.assertFalse(report["store_changed"])
        connection = connect_store(self.store)
        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM update_events").fetchone()[0], 0
        )
        head = connection.execute(
            "SELECT state,confidence,event_count FROM binding_heads WHERE node_id=?",
            (self.type_node,),
        ).fetchone()
        self.assertEqual(head["state"], "unresolved")
        self.assertEqual(head["confidence"], 0.0)
        self.assertEqual(head["event_count"], 0)
        connection.close()

    def test_strict_verifier(self):
        connection = connect_store(self.store)
        append_event(connection, {
            "node_id": self.layout,
            "step": 0,
            "reason": "verify_derived_edge_hash",
            "new_state": "confirmed_soft",
            "new_confidence": 0.77,
            "new_edges": [{
                "edge_id": "verify-derived-constraint",
                "edge_kind": "constraint",
                "constraint_type": "type_layout",
                "source_node": self.type_node,
                "target_node": self.layout,
                "confidence": 0.92,
            }],
            "created_at": "2026-01-01T00:00:00Z",
        })
        connection.close()
        subprocess.run([
            sys.executable, str(SRC / "15_verify_binding_store_v6_4.py"),
            "--store", str(self.store),
            "--manifest", str(self.manifest),
            "--report", str(self.report),
            "--trace-node-id", self.type_node,
        ], check=True, capture_output=True, text=True)
        report = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["append_only_triggers_present"])
        self.assertTrue(report["checks"]["event_hash_chains_valid"])
        self.assertTrue(report["checks"]["event_edge_classes_consistent"])


if __name__ == "__main__":
    unittest.main()
