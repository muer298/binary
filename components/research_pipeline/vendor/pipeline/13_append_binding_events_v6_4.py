#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append validated UpdateEvents to a node-level BindingStore.

Each input JSONL row is committed independently with one atomic transaction:
optional derived provenance-edge inserts, event-edge references, the
``update_events`` insert and the derived ``binding_heads`` update.  No USG or
embedding source artifact is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping

from binding_store_v6_4 import (
    BindingStoreError, append_event, connect_store, get_metadata, iter_jsonl,
    sha256_file, utc_now,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--expected-model-version", default="")
    parser.add_argument("--expected-graph-version", default="")
    parser.add_argument("--overwrite-report", action="store_true")
    args = parser.parse_args()

    store = Path(args.store).resolve()
    events = Path(args.events).resolve()
    report = Path(args.report).resolve()
    for path in (store, events):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError("Missing or empty input: %s" % path)
    if report.exists() and not args.overwrite_report:
        raise FileExistsError("Refusing to overwrite report: %s" % report)

    before_sha = sha256_file(store)
    connection = connect_store(store)
    metadata = get_metadata(connection)
    if args.expected_model_version and metadata.get("model_version") != args.expected_model_version:
        raise BindingStoreError(
            "model_version mismatch: expected=%r actual=%r"
            % (args.expected_model_version, metadata.get("model_version"))
        )
    if args.expected_graph_version and metadata.get("graph_version") != args.expected_graph_version:
        raise BindingStoreError(
            "graph_version mismatch: expected=%r actual=%r"
            % (args.expected_graph_version, metadata.get("graph_version"))
        )

    counts: Counter = Counter()
    appended_examples = []
    failures = []
    if args.dry_run:
        # Simulate the complete ordered batch in one outer transaction.  This
        # lets event N+1 see the provisional head from event N while the final
        # ROLLBACK guarantees that no event/new edge/head is persisted.
        connection.execute("BEGIN IMMEDIATE")
    for line_number, row in iter_jsonl(events):
        if args.max_events > 0 and counts["read"] >= args.max_events:
            break
        counts["read"] += 1
        savepoint = "dry_event_%d" % line_number
        try:
            if args.dry_run:
                connection.execute("SAVEPOINT " + savepoint)
                result = append_event(
                    connection, row, dry_run=False, manage_transaction=False
                )
                connection.execute("RELEASE SAVEPOINT " + savepoint)
                result["dry_run"] = True
            else:
                result = append_event(connection, row)
            counts["validated" if args.dry_run else "appended"] += 1
            if len(appended_examples) < 20:
                appended_examples.append({
                    "line": line_number,
                    "event_id": result["event_id"],
                    "node_id": result["node_id"],
                    "sequence": result["sequence"],
                    "event_hash": result["event_hash"],
                    "new_edge_ids": result.get("new_edge_ids", []),
                })
        except Exception as exc:
            if args.dry_run:
                connection.execute("ROLLBACK TO SAVEPOINT " + savepoint)
                connection.execute("RELEASE SAVEPOINT " + savepoint)
            counts["failed"] += 1
            if len(failures) < 50:
                failures.append({
                    "line": line_number,
                    "node_id": str(row.get("node_id") or row.get("node_ref") or ""),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
            if not args.allow_errors:
                break
    if args.dry_run:
        connection.rollback()
    connection.close()
    after_sha = sha256_file(store)
    result = {
        "schema": "task2_binding_event_append_report_v6_4",
        "store": str(store),
        "events": str(events),
        "dry_run": bool(args.dry_run),
        "counts": dict(counts),
        "appended_examples": appended_examples,
        "failures": failures,
        "store_sha256_before": before_sha,
        "store_sha256_after": after_sha,
        "store_changed": before_sha != after_sha,
        "metadata_versions": {
            "model_version": metadata.get("model_version"),
            "graph_version": metadata.get("graph_version"),
        },
        "created_at": utc_now(),
        "passed": counts["failed"] == 0,
    }
    atomic_json(report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("ERROR: one or more UpdateEvents were rejected; inspect %s" % report)


if __name__ == "__main__":
    main()
