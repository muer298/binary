#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict structural and event-chain verification for BindingStore v6.4."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

from binding_store_v6_4 import (
    EVENT_SCHEMA, SCHEMA, connect_store, get_metadata, recompute_event_hash,
    sha256_file, table_count, trace_node,
)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent),
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def load_manifest(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError("Manifest root must be an object")
    return value


def trigger_names(connection: sqlite3.Connection) -> set:
    return set(
        row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )
    )


def test_binding_immutable(connection: sqlite3.Connection) -> bool:
    row = connection.execute("SELECT node_id FROM bindings LIMIT 1").fetchone()
    if row is None:
        return False
    try:
        connection.execute("BEGIN")
        connection.execute(
            "UPDATE bindings SET node_type=node_type WHERE node_id=?", (row["node_id"],)
        )
    except sqlite3.DatabaseError:
        connection.rollback()
        return True
    connection.rollback()
    return False


def verify_event_chains(connection: sqlite3.Connection) -> Dict[str, Any]:
    errors = []
    checked = 0
    by_node: Counter = Counter()
    current = {}
    for row in connection.execute(
        "SELECT * FROM update_events ORDER BY node_id,sequence"
    ):
        node_id = str(row["node_id"])
        if node_id not in current:
            binding = connection.execute(
                "SELECT embedding_ref,initial_state,initial_confidence FROM bindings WHERE node_id=?",
                (node_id,),
            ).fetchone()
            if binding is None:
                errors.append("event references unknown node %s" % node_id)
                continue
            current[node_id] = {
                "sequence": 0,
                "embedding_ref": str(binding["embedding_ref"]),
                "state": str(binding["initial_state"]),
                "confidence": float(binding["initial_confidence"]),
                "hash": "",
                "step": -1,
            }
        head = current[node_id]
        refs = [dict(ref) for ref in connection.execute(
            "SELECT edge_id,edge_class,role FROM event_edge_refs "
            "WHERE event_id=? ORDER BY edge_class,edge_id", (row["event_id"],)
        )]
        payload = json.loads(str(row["payload_json"]))
        appended_hashes = payload.get("_appended_edge_hashes", {})
        if not isinstance(appended_hashes, Mapping):
            if len(errors) < 100:
                errors.append("%s has invalid _appended_edge_hashes" % row["event_id"])
            appended_hashes = {}
        referenced_edge_ids = {str(ref["edge_id"]) for ref in refs}
        for edge_id, expected_raw_hash in appended_hashes.items():
            edge = connection.execute(
                "SELECT raw_hash FROM edge_catalog WHERE edge_id=?", (str(edge_id),)
            ).fetchone()
            if edge is None:
                if len(errors) < 100:
                    errors.append(
                        "%s binds missing appended edge %s" % (row["event_id"], edge_id)
                    )
                continue
            if str(edge["raw_hash"]) != str(expected_raw_hash) and len(errors) < 100:
                errors.append(
                    "%s appended edge hash mismatch for %s" % (row["event_id"], edge_id)
                )
            if str(edge_id) not in referenced_edge_ids and len(errors) < 100:
                errors.append(
                    "%s appended edge is absent from event refs: %s"
                    % (row["event_id"], edge_id)
                )
        expected_hash = recompute_event_hash(row, refs)
        checks = {
            "schema": row["event_schema"] == EVENT_SCHEMA,
            "sequence": int(row["sequence"]) == head["sequence"] + 1,
            "prev_hash": str(row["prev_event_hash"]) == head["hash"],
            "event_hash": str(row["event_hash"]) == expected_hash,
            "event_id": str(row["event_id"]) == "event::" + expected_hash[:32],
            "old_embedding": str(row["old_embedding_ref"]) == head["embedding_ref"],
            "old_state": str(row["old_state"]) == head["state"],
            "old_confidence": math.isclose(
                float(row["old_confidence"]), head["confidence"],
                rel_tol=0.0, abs_tol=1e-12,
            ),
            "step_monotonic": int(row["step"]) >= head["step"],
            "new_confidence_finite": math.isfinite(float(row["new_confidence"]))
            and 0.0 <= float(row["new_confidence"]) <= 1.0,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed and len(errors) < 100:
            errors.append("%s failed %s" % (row["event_id"], failed))
        head.update({
            "sequence": int(row["sequence"]),
            "embedding_ref": str(row["new_embedding_ref"]),
            "state": str(row["new_state"]),
            "confidence": float(row["new_confidence"]),
            "hash": str(row["event_hash"]),
            "step": int(row["step"]),
            "event_id": str(row["event_id"]),
        })
        checked += 1
        by_node[node_id] += 1

    for node_id, expected in current.items():
        actual = connection.execute(
            "SELECT * FROM binding_heads WHERE node_id=?", (node_id,)
        ).fetchone()
        if actual is None:
            errors.append("missing head for %s" % node_id)
            continue
        final_checks = (
            int(actual["event_count"]) == expected["sequence"],
            str(actual["latest_event_id"]) == expected.get("event_id", ""),
            str(actual["embedding_ref"]) == expected["embedding_ref"],
            str(actual["state"]) == expected["state"],
            math.isclose(float(actual["confidence"]), expected["confidence"], rel_tol=0.0, abs_tol=1e-12),
            str(actual["latest_event_hash"]) == expected["hash"],
        )
        if not all(final_checks) and len(errors) < 100:
            errors.append("derived head mismatch for %s" % node_id)

    zero_head_mismatches = int(connection.execute(
        """
        SELECT COUNT(*) FROM binding_heads h JOIN bindings b ON b.node_id=h.node_id
        WHERE h.event_count=0 AND (
          h.latest_event_id IS NOT NULL OR h.embedding_ref<>b.embedding_ref OR
          h.state<>b.initial_state OR ABS(h.confidence-b.initial_confidence)>1e-12 OR
          h.latest_event_hash<>'' OR h.last_step<>-1
        )
        """
    ).fetchone()[0])
    if zero_head_mismatches:
        errors.append("%d zero-event heads differ from genesis" % zero_head_mismatches)
    return {
        "events_checked": checked,
        "nodes_with_events": len(by_node),
        "zero_event_head_mismatches": zero_head_mismatches,
        "errors": errors,
        "passed": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-formal", action="store_true")
    parser.add_argument("--trace-node-id", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    store = Path(args.store).resolve()
    manifest_path = Path(args.manifest).resolve()
    report_path = Path(args.report).resolve()
    for path in (store, manifest_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    if report_path.exists() and not args.overwrite:
        raise FileExistsError("Refusing to overwrite report: %s" % report_path)

    manifest = load_manifest(manifest_path)
    connection = connect_store(store)
    metadata = get_metadata(connection)
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    foreign_keys = [tuple(row) for row in connection.execute("PRAGMA foreign_key_check")]
    counts = {
        table: table_count(connection, table) for table in (
            "metadata", "bindings", "edge_catalog", "binding_edge_refs",
            "update_events", "event_edge_refs", "binding_heads",
        )
    }
    class_counts = {
        row["edge_class"]: int(row["n"]) for row in connection.execute(
            "SELECT edge_class,COUNT(*) AS n FROM edge_catalog GROUP BY edge_class"
        )
    }
    binding_ref_class_mismatches = int(connection.execute(
        """
        SELECT COUNT(*) FROM binding_edge_refs r
        JOIN edge_catalog e ON e.edge_id=r.edge_id
        WHERE r.edge_class<>e.edge_class
        """
    ).fetchone()[0])
    event_ref_class_mismatches = int(connection.execute(
        """
        SELECT COUNT(*) FROM event_edge_refs r
        JOIN edge_catalog e ON e.edge_id=r.edge_id
        WHERE r.edge_class<>e.edge_class
        """
    ).fetchone()[0])
    invalid_binding_source_refs = int(connection.execute(
        """
        SELECT COUNT(*) FROM bindings
        WHERE source_index_row<0 OR source_node_line<1
          OR LENGTH(source_index_record_hash)<>64
          OR LENGTH(source_node_record_hash)<>64
          OR LENGTH(source_record_hash)<>64
        """
    ).fetchone()[0])
    expected_triggers = set()
    for table in (
        "metadata", "bindings", "edge_catalog", "binding_edge_refs",
        "update_events", "event_edge_refs",
    ):
        expected_triggers.add("no_update_" + table)
        expected_triggers.add("no_delete_" + table)
    triggers = trigger_names(connection)
    append_only_triggers_present = expected_triggers.issubset(triggers)
    immutable_runtime = test_binding_immutable(connection)
    event_audit = verify_event_chains(connection)
    trace_audit: Dict[str, Any] = {"requested": False, "passed": True}
    if args.trace_node_id:
        try:
            trace = trace_node(connection, args.trace_node_id, max_depth=1)
            trace_audit = {
                "requested": True, "passed": True,
                "node_id": args.trace_node_id, "counts": trace["counts"],
                "recursive_nodes_visited": trace.get("recursive_nodes_visited", 1),
            }
        except Exception as exc:
            trace_audit = {
                "requested": True, "passed": False,
                "node_id": args.trace_node_id, "error": str(exc),
            }
    connection.close()

    current_store_sha = sha256_file(store)
    genesis_store_sha = str((manifest.get("store") or {}).get("sha256") or "")
    # The DB hash is expected to change after valid append-only events.  It is a
    # strict creation-snapshot check only while the event log is empty.
    creation_snapshot_hash_valid = (
        current_store_sha == genesis_store_sha if counts["update_events"] == 0 else None
    )
    checks = {
        "schema_valid": metadata.get("schema") == SCHEMA and manifest.get("schema") == SCHEMA,
        "model_version_matches_manifest": metadata.get("model_version") == (
            manifest.get("versions") or {}
        ).get("model_version"),
        "graph_version_matches_manifest": metadata.get("graph_version") == (
            manifest.get("versions") or {}
        ).get("graph_version"),
        "embedding_index_hash_matches_manifest": metadata.get("embedding_index_sha256") == (
            (manifest.get("inputs") or {}).get("embedding_index") or {}
        ).get("sha256"),
        "embedding_tensor_hash_matches_manifest": metadata.get("embeddings_sha256") == (
            (manifest.get("inputs") or {}).get("embeddings") or {}
        ).get("sha256"),
        "sqlite_integrity": integrity == "ok",
        "foreign_keys_valid": not foreign_keys,
        "one_head_per_binding": counts["bindings"] > 0 and counts["bindings"] == counts["binding_heads"],
        "manifest_binding_count_matches": counts["bindings"] == int(
            (manifest.get("binding_records") or {}).get("count", -1)
        ),
        "binding_source_references_valid": invalid_binding_source_refs == 0,
        "provenance_catalog_present": counts["edge_catalog"] > 0 and counts["binding_edge_refs"] > 0,
        "binding_edge_classes_consistent": binding_ref_class_mismatches == 0,
        "event_edge_classes_consistent": event_ref_class_mismatches == 0,
        "append_only_triggers_present": append_only_triggers_present,
        "immutable_binding_runtime_guard": immutable_runtime,
        "event_hash_chains_valid": event_audit["passed"],
        "trace_query_valid": trace_audit["passed"],
        "creation_snapshot_hash_valid": creation_snapshot_hash_valid is not False,
    }
    if args.require_formal:
        checks["formal_metadata"] = bool(metadata.get("formal_ready"))
        checks["formal_manifest"] = bool(manifest.get("formal_ready"))
    passed = all(checks.values())
    result = {
        "schema": "task2_binding_store_verification_v6_4",
        "passed": passed,
        "formal_required": bool(args.require_formal),
        "store": str(store),
        "manifest": str(manifest_path),
        "metadata": metadata,
        "counts": counts,
        "edge_class_counts": class_counts,
        "edge_reference_audit": {
            "binding_ref_class_mismatches": binding_ref_class_mismatches,
            "event_ref_class_mismatches": event_ref_class_mismatches,
        },
        "invalid_binding_source_references": invalid_binding_source_refs,
        "checks": checks,
        "integrity_check": integrity,
        "foreign_key_violations": foreign_keys[:100],
        "expected_triggers": sorted(expected_triggers),
        "missing_triggers": sorted(expected_triggers.difference(triggers)),
        "event_audit": event_audit,
        "trace_audit": trace_audit,
        "store_hash": {
            "creation_snapshot_sha256": genesis_store_sha,
            "current_sha256": current_store_sha,
            "events_exist": counts["update_events"] > 0,
            "note": "Hash changes after valid append-only events; event chains then become the history integrity contract.",
        },
    }
    atomic_json(report_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("ERROR: BindingStore verification failed; inspect %s" % report_path)


if __name__ == "__main__":
    main()
