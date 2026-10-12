#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the formal node-level BindingStore (2.2.1.5, v6.4).

The historical filename is retained so an existing pipeline cannot silently
continue producing anchor-level pseudo-bindings.  The previous implementation
is preserved as ``12_build_anchor_binding_records_legacy.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from binding_store_v6_4 import (
    SCHEMA, SCHEMA_VERSION, BindingStoreError, canonical_json, connect_store,
    initialize_schema, iter_jsonl, load_index, resolve_edge,
    resolve_node_metadata, set_metadata, sha256_file, sha256_text,
    utc_now, validate_confidence, validate_state,
)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError("Missing or empty %s: %s" % (label, path))


def metadata_fingerprint(path: Path) -> str:
    stat = path.stat()
    return "metadata:size=%d;mtime_ns=%d" % (stat.st_size, stat.st_mtime_ns)


def checkpoint_build_batch(connection: sqlite3.Connection) -> None:
    """Bound WAL growth while an unpublished temporary store is built."""
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.execute("BEGIN IMMEDIATE")


def cleanup_sqlite_artifacts(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            candidate.unlink()


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise BindingStoreError("Expected JSON object in %s" % path)
    return value


def manifest_output(manifest: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        return {}
    value = outputs.get(key)
    return value if isinstance(value, Mapping) else {}


def derive_versions(
    manifest: Mapping[str, Any], model_version: str, graph_version: str,
    edges_sha256: str,
) -> Tuple[str, str, str]:
    checkpoint = manifest.get("checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    inputs = manifest.get("inputs")
    inputs = inputs if isinstance(inputs, Mapping) else {}
    checkpoint_input = inputs.get("checkpoint")
    checkpoint_input = checkpoint_input if isinstance(checkpoint_input, Mapping) else {}
    checkpoint_sha = str(checkpoint_input.get("sha256") or "")
    objective = str(checkpoint.get("objective_revision") or "unknown_objective")
    selected_model = model_version.strip() or (
        "%s@%s" % (objective, checkpoint_sha[:16] or "unhashed")
    )
    if graph_version.strip():
        selected_graph = graph_version.strip()
    elif len(edges_sha256) == 64 and all(ch in "0123456789abcdef" for ch in edges_sha256.lower()):
        selected_graph = "sha256:%s" % edges_sha256
    else:
        selected_graph = "smoke:%s" % edges_sha256
    checkpoint_ref = str(checkpoint_input.get("path") or "")
    if checkpoint_sha:
        checkpoint_ref = "%s#sha256=%s" % (checkpoint_ref, checkpoint_sha)
    return selected_model, selected_graph, checkpoint_ref


def normalise_index_rows(
    rows: List[Mapping[str, Any]], embeddings_name: str, max_nodes: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if max_nodes > 0:
        rows = rows[:max_nodes]
    result: List[Dict[str, Any]] = []
    seen = set()
    errors: List[str] = []
    index_values: List[int] = []
    for position, row in enumerate(rows):
        metadata = resolve_node_metadata(row)
        node_id = metadata["node_id"]
        if not node_id:
            errors.append("row %d has no node_id" % position)
            continue
        if node_id in seen:
            errors.append("duplicate node_id %s" % node_id)
            continue
        seen.add(node_id)
        raw_index = row.get("index", row.get("row", row.get("embedding_index", position)))
        try:
            row_index = int(raw_index)
        except (TypeError, ValueError):
            errors.append("invalid index for %s: %r" % (node_id, raw_index))
            continue
        index_values.append(row_index)
        embedding_ref = str(row.get("embedding_ref") or "").strip()
        if not embedding_ref:
            embedding_ref = "%s#row=%d" % (embeddings_name, row_index)
        result.append({
            "node_id": node_id,
            "sample_id": metadata["sample_id"],
            "node_type": metadata["node_type"],
            "layer": metadata["layer"],
            "raw_id": metadata["raw_id"],
            "index": row_index,
            "embedding_ref": embedding_ref,
            "raw_index_record": dict(row),
        })
    if errors:
        raise BindingStoreError("Invalid embedding index: %s" % "; ".join(errors[:20]))
    if index_values != list(range(len(result))):
        raise BindingStoreError(
            "Embedding index must be ordered and contiguous from zero; first=%r"
            % index_values[:20]
        )
    return result, {
        "rows": len(result),
        "unique_node_ids": len(seen),
        "contiguous_zero_based": True,
    }


def scan_node_metadata(
    path: Path, wanted: set, formal_scan: bool,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    found: Dict[str, Dict[str, Any]] = {}
    duplicate = empty = extra = rows_scanned = 0
    for line_number, row in iter_jsonl(path):
        rows_scanned += 1
        metadata = resolve_node_metadata(row)
        node_id = metadata["node_id"]
        if not node_id:
            empty += 1
            continue
        if node_id not in wanted:
            extra += 1
        elif node_id in found:
            duplicate += 1
        else:
            metadata["_source_node_line"] = line_number
            metadata["_source_node_record_hash"] = sha256_text(canonical_json(row))
            found[node_id] = metadata
        if not formal_scan and len(found) == len(wanted):
            break
    missing = sorted(wanted.difference(found))
    return found, {
        "rows_scanned": rows_scanned,
        "matched": len(found),
        "missing": len(missing),
        "missing_examples": missing[:20],
        "duplicates": duplicate,
        "empty_node_ids": empty,
        "extra_rows_seen": extra,
        "coverage": (len(found) / len(wanted)) if wanted else 1.0,
        "full_file_scanned": formal_scan,
    }


def insert_bindings(
    connection: sqlite3.Connection, rows: List[Dict[str, Any]],
    node_metadata: Mapping[str, Mapping[str, Any]], model_version: str,
    graph_version: str, checkpoint_ref: str, initial_state: str,
    initial_confidence: float, created_at: str, commit_every: int,
) -> Counter:
    counts: Counter = Counter()
    connection.execute("BEGIN IMMEDIATE")
    try:
        for row in rows:
            node_id = row["node_id"]
            source_meta = node_metadata.get(node_id, {})
            merged = dict(row)
            for key in ("sample_id", "node_type", "layer", "raw_id"):
                if source_meta.get(key):
                    merged[key] = source_meta[key]
            index_record_hash = sha256_text(canonical_json(row["raw_index_record"]))
            node_record_hash = str(source_meta.get("_source_node_record_hash") or "")
            source_node_line = int(source_meta.get("_source_node_line") or 0)
            if source_node_line < 1 or not node_record_hash:
                raise BindingStoreError(
                    "Missing immutable USG source reference for node %s" % node_id
                )
            record_hash = sha256_text(canonical_json({
                "index_record_hash": index_record_hash,
                "node_record_hash": node_record_hash,
                "source_node_line": source_node_line,
                "embedding_ref": row["embedding_ref"],
            }))
            binding_id = "binding::" + sha256_text(node_id)[:32]
            connection.execute(
                """
                INSERT INTO bindings(
                  node_id,binding_id,node_type,sample_id,layer,raw_id,embedding_ref,
                  initial_state,initial_confidence,schema_version,model_version,
                  graph_version,checkpoint_ref,source_index_row,source_index_record_hash,
                  source_node_line,source_node_record_hash,source_record_hash,created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    node_id, binding_id, merged.get("node_type", "unknown"),
                    merged.get("sample_id", ""), merged.get("layer", ""),
                    merged.get("raw_id", ""), row["embedding_ref"], initial_state,
                    initial_confidence, SCHEMA_VERSION, model_version, graph_version,
                    checkpoint_ref, row["index"], index_record_hash,
                    source_node_line, node_record_hash, record_hash, created_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO binding_heads(
                  node_id,latest_event_id,event_count,embedding_ref,state,confidence,
                  last_step,latest_event_hash
                ) VALUES (?,NULL,0,?,?,?,-1,'')
                """,
                (node_id, row["embedding_ref"], initial_state, initial_confidence),
            )
            counts["bindings"] += 1
            counts["type::" + str(merged.get("node_type", "unknown"))] += 1
            if counts["bindings"] % commit_every == 0:
                checkpoint_build_batch(connection)
                counts["batches_committed"] += 1
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return counts


def insert_edges(
    connection: sqlite3.Connection, path: Path, binding_ids: set,
    include_relations: bool, max_edges: int, commit_every: int,
) -> Dict[str, Any]:
    counts: Counter = Counter()
    bad_examples: List[Dict[str, Any]] = []
    connection.execute("BEGIN IMMEDIATE")
    try:
        for line_number, raw in iter_jsonl(path):
            if max_edges > 0 and counts["rows_scanned"] >= max_edges:
                break
            counts["rows_scanned"] += 1
            edge = resolve_edge(raw, line_number=line_number)
            edge_class = edge["edge_class"]
            counts["class::" + edge_class] += 1
            if edge_class == "relation" and not include_relations:
                counts["relations_skipped"] += 1
                continue
            source, target = edge["source_node"], edge["target_node"]
            if not source or not target:
                counts["missing_endpoint"] += 1
                if len(bad_examples) < 20:
                    bad_examples.append({
                        "line": line_number, "edge_id": edge["edge_id"],
                        "source": source, "target": target,
                    })
                continue
            existing = connection.execute(
                "SELECT raw_hash FROM edge_catalog WHERE edge_id=?", (edge["edge_id"],)
            ).fetchone()
            if existing is not None:
                if str(existing["raw_hash"]) != edge["raw_hash"]:
                    raise BindingStoreError(
                        "Conflicting duplicate edge_id %s at line %d"
                        % (edge["edge_id"], line_number)
                    )
                counts["duplicate_edges_identical"] += 1
            else:
                connection.execute(
                    """
                    INSERT INTO edge_catalog(
                      edge_id,edge_class,edge_kind,edge_type,source_node,target_node,
                      strength,source_path,raw_hash,source_line
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        edge["edge_id"], edge_class, edge["edge_kind"],
                        edge["edge_type"], source, target, edge["strength"],
                        edge["source_path"], edge["raw_hash"], line_number,
                    ),
                )
                counts["edges_inserted"] += 1
            for node_id, role in ((source, "source"), (target, "target")):
                if node_id not in binding_ids:
                    counts["endpoint_without_binding"] += 1
                    continue
                connection.execute(
                    "INSERT OR IGNORE INTO binding_edge_refs(node_id,edge_id,edge_class,role) "
                    "VALUES (?,?,?,?)",
                    (node_id, edge["edge_id"], edge_class, role),
                )
                counts["binding_refs_attempted"] += 1
            if counts["rows_scanned"] % commit_every == 0:
                checkpoint_build_batch(connection)
                counts["batches_committed"] += 1
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    refs = int(connection.execute("SELECT COUNT(*) FROM binding_edge_refs").fetchone()[0])
    return {
        "counts": dict(counts),
        "binding_edge_refs": refs,
        "bad_edge_examples": bad_examples,
        "full_file_scanned": max_edges == 0,
        "commit_every": commit_every,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-index", required=True)
    parser.add_argument("--embedding-manifest", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-version", default="")
    parser.add_argument("--graph-version", default="")
    parser.add_argument("--initial-state", default="unresolved")
    parser.add_argument("--initial-confidence", type=float, default=0.0)
    parser.add_argument("--include-relation-edges", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--max-edges", type=int, default=0)
    parser.add_argument("--require-formal", action="store_true")
    parser.add_argument("--commit-every", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    index_path = Path(args.embedding_index).resolve()
    embedding_manifest_path = Path(args.embedding_manifest).resolve()
    embeddings_path = Path(args.embeddings).resolve()
    nodes_path = Path(args.nodes).resolve()
    edges_path = Path(args.edges).resolve()
    store_path = Path(args.store).resolve()
    output_manifest_path = Path(args.manifest).resolve()
    for path, label in (
        (index_path, "embedding index"),
        (embedding_manifest_path, "embedding manifest"),
        (embeddings_path, "embedding tensor"),
        (nodes_path, "USG nodes"),
        (edges_path, "USG edges"),
    ):
        require_file(path, label)
    existing = [path for path in (store_path, output_manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: %s"
            % ", ".join(str(path) for path in existing)
        )
    if args.require_formal and (args.max_nodes > 0 or args.max_edges > 0):
        raise BindingStoreError("Formal build cannot use --max-nodes/--max-edges")
    if args.commit_every < 1000:
        raise BindingStoreError("--commit-every must be at least 1000")

    source_manifest = load_json(embedding_manifest_path)
    if args.require_formal and not bool(source_manifest.get("formal_ready")):
        raise BindingStoreError("Formal BindingStore requires formal_ready embedding export")
    index_sha = sha256_file(index_path)
    embeddings_sha = sha256_file(embeddings_path)
    manifest_index = manifest_output(source_manifest, "index")
    manifest_embeddings = manifest_output(source_manifest, "embeddings")
    if manifest_index.get("sha256") and str(manifest_index["sha256"]) != index_sha:
        raise BindingStoreError("Embedding index SHA-256 does not match v6.3 manifest")
    if manifest_embeddings.get("sha256") and str(manifest_embeddings["sha256"]) != embeddings_sha:
        raise BindingStoreError("Embedding tensor SHA-256 does not match v6.3 manifest")

    index_rows, index_audit = normalise_index_rows(
        load_index(index_path), embeddings_path.name, max(0, args.max_nodes)
    )
    if not index_rows:
        raise BindingStoreError("Embedding index produced zero BindingRecords")
    binding_ids = set(row["node_id"] for row in index_rows)
    node_metadata, node_audit = scan_node_metadata(
        nodes_path, binding_ids, formal_scan=(args.max_nodes == 0),
    )
    if node_audit["missing"] or node_audit["duplicates"]:
        raise BindingStoreError(
            "USG node/index mismatch: missing=%d duplicate=%d"
            % (node_audit["missing"], node_audit["duplicates"])
        )

    # Formal builds use content hashes.  Smoke avoids a full pre-scan of the
    # very large graph files and records an explicit non-formal metadata
    # fingerprint instead; its output can therefore never pass --require-formal.
    full_graph_hashes = args.max_nodes == 0 and args.max_edges == 0
    edges_sha = sha256_file(edges_path) if full_graph_hashes else ""
    nodes_sha = sha256_file(nodes_path) if full_graph_hashes else ""
    edges_fingerprint = edges_sha or metadata_fingerprint(edges_path)
    nodes_fingerprint = nodes_sha or metadata_fingerprint(nodes_path)
    model_version, graph_version, checkpoint_ref = derive_versions(
        source_manifest, args.model_version, args.graph_version, edges_fingerprint,
    )
    initial_state = validate_state(args.initial_state)
    initial_confidence = validate_confidence(args.initial_confidence)
    created_at = utc_now()

    store_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=store_path.name + ".", suffix=".tmp", dir=str(store_path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = connect_store(temporary)
        initialize_schema(connection)
        for key, value in (
            ("schema", SCHEMA), ("schema_version", SCHEMA_VERSION),
            ("model_version", model_version),
            ("graph_version", graph_version), ("checkpoint_ref", checkpoint_ref),
            ("embedding_index_sha256", index_sha),
            ("embeddings_sha256", embeddings_sha), ("nodes_sha256", nodes_sha),
            ("edges_sha256", edges_sha), ("nodes_fingerprint", nodes_fingerprint),
            ("edges_fingerprint", edges_fingerprint), ("created_at", created_at),
            ("embedding_index_path", str(index_path)),
            ("embeddings_path", str(embeddings_path)),
            ("nodes_path", str(nodes_path)), ("edges_path", str(edges_path)),
            ("embedding_manifest_path", str(embedding_manifest_path)),
        ):
            set_metadata(connection, key, value)
        connection.commit()
        binding_counts = insert_bindings(
            connection, index_rows, node_metadata, model_version, graph_version,
            checkpoint_ref, initial_state, initial_confidence, created_at,
            args.commit_every,
        )
        edge_audit = insert_edges(
            connection, edges_path, binding_ids, args.include_relation_edges,
            max(0, args.max_edges), args.commit_every,
        )
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise BindingStoreError(
                "Foreign-key violations: %r" % [tuple(x) for x in foreign_keys[:20]]
            )
        formal_ready = bool(
            source_manifest.get("formal_ready") and args.max_nodes == 0
            and args.max_edges == 0 and node_audit["coverage"] == 1.0
            and node_audit["extra_rows_seen"] == 0
            and node_audit["empty_node_ids"] == 0
        )
        set_metadata(connection, "formal_ready", formal_ready)
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("VACUUM")
        connection.commit()
        connection.close()
        connection = None
        os.replace(str(temporary), str(store_path))
        cleanup_sqlite_artifacts(temporary)
    except Exception:
        if connection is not None:
            connection.close()
        cleanup_sqlite_artifacts(temporary)
        raise

    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "formal_ready": formal_ready,
        "store": {
            "path": str(store_path), "sha256": sha256_file(store_path),
            "size": store_path.stat().st_size,
            "format": "SQLite3 event-sourced BindingStore",
        },
        "inputs": {
            "embedding_index": {"path": str(index_path), "sha256": index_sha},
            "embedding_manifest": {
                "path": str(embedding_manifest_path),
                "sha256": sha256_file(embedding_manifest_path),
                "formal_ready": bool(source_manifest.get("formal_ready")),
            },
            "embeddings": {"path": str(embeddings_path), "sha256": embeddings_sha},
            "nodes": {
                "path": str(nodes_path), "sha256": nodes_sha or None,
                "fingerprint": nodes_fingerprint,
            },
            "edges": {
                "path": str(edges_path), "sha256": edges_sha or None,
                "fingerprint": edges_fingerprint,
            },
        },
        "versions": {
            "model_version": model_version, "graph_version": graph_version,
            "checkpoint_ref": checkpoint_ref,
        },
        "binding_records": {
            "count": binding_counts["bindings"],
            "node_type_counts": {
                key.replace("type::", ""): value for key, value in binding_counts.items()
                if key.startswith("type::")
            },
            "initial_state": initial_state,
            "initial_confidence": initial_confidence,
        },
        "index_audit": index_audit,
        "node_audit": node_audit,
        "edge_audit": edge_audit,
        "build_io_policy": {
            "temporary_store_then_atomic_publish": True,
            "commit_every": args.commit_every,
            "binding_batches_committed": binding_counts.get("batches_committed", 0),
            "wal_checkpoint_policy": "TRUNCATE after each build batch",
        },
        "contracts": {
            "binding_granularity": "one immutable genesis BindingRecord per node_id",
            "source_reference": "node JSONL line + raw hash; embedding index row + raw hash",
            "update_history": "append-only update_events with per-node SHA-256 chain",
            "current_state": "binding_heads is a rebuildable transactional cache",
            "embedding_storage": "embedding_ref only; tensor is never copied or rewritten",
            "legacy_anchor_records_formal": False,
            "legacy_2_2_1_6_writeback_enabled": False,
        },
        "checks": {
            "embedding_hashes_match_manifest": True,
            "index_contiguous": index_audit["contiguous_zero_based"],
            "node_metadata_complete": node_audit["coverage"] == 1.0,
            "binding_count_matches_index": binding_counts["bindings"] == index_audit["rows"],
            "foreign_keys_valid": True,
        },
        "created_at": created_at,
    }
    atomic_write_json(output_manifest_path, result)
    print(json.dumps({
        "schema": SCHEMA, "formal_ready": formal_ready,
        "store": str(store_path), "manifest": str(output_manifest_path),
        "bindings": binding_counts["bindings"],
        "edges": edge_audit["counts"].get("edges_inserted", 0),
        "binding_edge_refs": edge_audit["binding_edge_refs"],
    }, ensure_ascii=False, indent=2))
    if args.require_formal and not formal_ready:
        raise BindingStoreError("BindingStore completed but is not formal-ready")


if __name__ == "__main__":
    main()
