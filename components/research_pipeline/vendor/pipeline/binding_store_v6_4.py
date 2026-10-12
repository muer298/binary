#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transactional node-level BindingStore for Research Content 2, section 2.2.1.5.

The authoritative history is event sourced:

* ``bindings`` contains one immutable genesis BindingRecord per USG node;
* ``edge_catalog`` and ``binding_edge_refs`` contain immutable provenance;
* ``update_events`` and ``event_edge_refs`` are append-only;
* ``binding_heads`` is a transactionally maintained, rebuildable cache.

The implementation deliberately does not rewrite USG nodes or the formal embedding
table.  ``embedding_ref`` is a stable reference into ``node_embeddings.pt``.  A
future 2.2.1.6 resolver must append an UpdateEvent through :func:`append_event`;
direct SQL UPDATE/DELETE on authoritative tables is blocked by database triggers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA = "task2_node_binding_store_v6_4"
SCHEMA_VERSION = "6.4"
EVENT_SCHEMA = "task2_binding_update_event_v6_4"
TRACE_SCHEMA = "task2_binding_trace_v6_4"

NODE_ID_KEYS = (
    "node_id", "node_ref", "id", "uid", "global_id", "semantic_node_id",
)
SAMPLE_ID_KEYS = (
    "sample_id", "sample_hash", "binary_id", "binary_hash", "program_id",
    "sha256",
)
NODE_TYPE_KEYS = ("node_type", "semantic_type", "node_kind", "kind", "type")
LAYER_KEYS = ("layer", "semantic_layer", "level")
RAW_ID_KEYS = ("raw_id", "local_id", "object_id")
SOURCE_KEYS = (
    "source_node", "source", "src", "from_node", "from", "u",
    "source_ref", "source_id", "low_node", "low_ref",
)
TARGET_KEYS = (
    "target_node", "target", "dst", "to_node", "to", "v",
    "target_ref", "target_id", "high_node", "high_ref", "destination",
)
EDGE_ID_KEYS = ("edge_id", "relation_id", "id", "uid")
EDGE_KIND_KEYS = ("edge_kind", "edge_family", "family", "category", "kind")
EDGE_TYPE_KEYS = (
    "edge_type", "relation", "relation_type", "constraint_type",
    "evidence_type", "predicate", "type",
)
STRENGTH_KEYS = ("strength", "weight", "confidence", "score", "probability")
SOURCE_PATH_KEYS = ("source_path", "json_path", "origin_path", "path")
NESTED_KEYS = (
    "attrs", "attributes", "metadata", "meta", "context", "payload",
    "node", "edge", "record", "data",
)

ALLOWED_STATES = {
    "unresolved", "hypothesis", "bound", "confirmed_hard", "confirmed_soft",
    "rejected", "conflict", "masked", "projected", "blended",
}


class BindingStoreError(RuntimeError):
    """Base error for contract violations."""


class ConcurrencyError(BindingStoreError):
    """Raised when an event was built from a stale binding head."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _iter_dicts(value: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return
    queue: List[Mapping[str, Any]] = [value]
    seen = set()
    while queue:
        current = queue.pop(0)
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        yield current
        for key in NESTED_KEYS:
            child = current.get(key)
            if isinstance(child, Mapping):
                queue.append(child)


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        for key in NODE_ID_KEYS + ("name", "value"):
            nested = value.get(key)
            if nested is not None and not isinstance(nested, (Mapping, list, tuple, set)):
                return str(nested).strip()
        return ""
    if isinstance(value, (list, tuple, set)):
        return ""
    return str(value).strip()


def resolve_alias(record: Mapping[str, Any], keys: Sequence[str]) -> Tuple[str, str]:
    containers = list(_iter_dicts(record))
    for key in keys:
        for container in containers:
            if key in container:
                value = _scalar(container.get(key))
                if value:
                    return value, key
    return "", ""


def resolve_float(record: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    containers = list(_iter_dicts(record))
    for key in keys:
        for container in containers:
            if key not in container:
                continue
            try:
                value = float(container.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None


def infer_sample_id(node_id: str) -> str:
    return str(node_id or "").split("::", 1)[0] if "::" in str(node_id or "") else ""


def infer_node_type(node_id: str) -> str:
    value = str(node_id or "")
    parts = value.split("::")
    if len(parts) >= 3 and parts[-2]:
        return parts[-2]
    for candidate in (
        "InstrSem", "BlockSem", "CFGSem", "DFGSem", "TypeSem",
        "InterfaceSem", "APISem", "LayoutSem",
    ):
        if candidate in value:
            return candidate
    return "unknown"


def resolve_node_metadata(record: Mapping[str, Any]) -> Dict[str, str]:
    node_id, node_id_source = resolve_alias(record, NODE_ID_KEYS)
    sample_id, sample_id_source = resolve_alias(record, SAMPLE_ID_KEYS)
    node_type, node_type_source = resolve_alias(record, NODE_TYPE_KEYS)
    layer, layer_source = resolve_alias(record, LAYER_KEYS)
    raw_id, raw_id_source = resolve_alias(record, RAW_ID_KEYS)
    if not sample_id:
        sample_id = infer_sample_id(node_id)
        sample_id_source = "node_id_prefix" if sample_id else ""
    if not node_type:
        node_type = infer_node_type(node_id)
        node_type_source = "node_id_token" if node_type != "unknown" else ""
    return {
        "node_id": node_id,
        "sample_id": sample_id,
        "node_type": node_type or "unknown",
        "layer": layer,
        "raw_id": raw_id,
        "node_id_source": node_id_source,
        "sample_id_source": sample_id_source,
        "node_type_source": node_type_source,
        "layer_source": layer_source,
        "raw_id_source": raw_id_source,
    }


def _edge_text(record: Mapping[str, Any]) -> str:
    values: List[str] = []
    for keys in (EDGE_KIND_KEYS, EDGE_TYPE_KEYS):
        value, _ = resolve_alias(record, keys)
        if value:
            values.append(value.lower())
    return "|".join(values)


def classify_edge(record: Mapping[str, Any]) -> str:
    text = _edge_text(record)
    if any(token in text for token in (
        "constraint", "predicate", "abi", "calling_convention", "layout_rule",
        "type_layout", "compatible", "consistency", "must_", "requires",
    )):
        return "constraint"
    if any(token in text for token in (
        "evidence", "fixed_offset", "deref", "def_use", "api_call",
        "call_instruction", "paramflow", "parameter_flow", "data_flow",
        "control_flow", "observed", "supports",
    )):
        return "evidence"
    return "relation"


def resolve_edge(record: Mapping[str, Any], line_number: int = 0) -> Dict[str, Any]:
    source, source_key = resolve_alias(record, SOURCE_KEYS)
    target, target_key = resolve_alias(record, TARGET_KEYS)
    edge_id, edge_id_source = resolve_alias(record, EDGE_ID_KEYS)
    edge_kind, edge_kind_source = resolve_alias(record, EDGE_KIND_KEYS)
    edge_type, edge_type_source = resolve_alias(record, EDGE_TYPE_KEYS)
    source_path, source_path_source = resolve_alias(record, SOURCE_PATH_KEYS)
    strength = resolve_float(record, STRENGTH_KEYS)
    raw_hash = sha256_text(canonical_json(record))
    if not edge_id:
        edge_id = "edge::" + raw_hash[:32]
        edge_id_source = "generated_sha256"
    return {
        "edge_id": edge_id,
        "edge_class": classify_edge(record),
        "edge_kind": edge_kind,
        "edge_type": edge_type,
        "source_node": source,
        "target_node": target,
        "strength": strength,
        "source_path": source_path,
        "raw_hash": raw_hash,
        "line_number": int(line_number),
        "schema_sources": {
            "edge_id": edge_id_source,
            "edge_kind": edge_kind_source,
            "edge_type": edge_type_source,
            "source_node": source_key,
            "target_node": target_key,
            "source_path": source_path_source,
        },
    }


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Mapping[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BindingStoreError(
                    "Invalid JSONL at %s:%d: %s" % (path, line_number, exc)
                ) from exc
            if not isinstance(value, Mapping):
                raise BindingStoreError(
                    "Expected a JSON object at %s:%d" % (path, line_number)
                )
            yield line_number, value


def load_index(path: Path) -> List[Mapping[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [row for _, row in iter_jsonl(path)]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping):
        rows = None
        for key in ("records", "nodes", "index", "items", "data"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            # Compatibility with {node_id: row_or_index} maps.
            rows = []
            for node_id, candidate in value.items():
                if isinstance(candidate, Mapping):
                    row = dict(candidate)
                    row.setdefault("node_id", node_id)
                else:
                    row = {"node_id": node_id, "index": candidate}
                rows.append(row)
    else:
        raise BindingStoreError("Unsupported embedding index root: %s" % type(value))
    if not all(isinstance(row, Mapping) for row in rows):
        raise BindingStoreError("Every embedding index row must be an object")
    return list(rows)


def validate_state(state: Any) -> str:
    value = str(state or "").strip()
    if not value:
        raise BindingStoreError("Binding state cannot be empty")
    if value not in ALLOWED_STATES and not re.match(r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$", value):
        raise BindingStoreError("Invalid binding state: %s" % value)
    return value


def validate_confidence(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BindingStoreError("Confidence must be numeric: %r" % (value,)) from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise BindingStoreError("Confidence must be finite and in [0, 1]: %r" % result)
    return result


def connect_store(path: Path, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path).resolve()
    if read_only:
        connection = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    else:
        connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    if not read_only:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
          key TEXT PRIMARY KEY,
          value_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bindings (
          node_id TEXT PRIMARY KEY,
          binding_id TEXT NOT NULL UNIQUE,
          node_type TEXT NOT NULL,
          sample_id TEXT NOT NULL,
          layer TEXT NOT NULL,
          raw_id TEXT NOT NULL,
          embedding_ref TEXT NOT NULL,
          initial_state TEXT NOT NULL,
          initial_confidence REAL NOT NULL CHECK(initial_confidence >= 0 AND initial_confidence <= 1),
          schema_version TEXT NOT NULL,
          model_version TEXT NOT NULL,
          graph_version TEXT NOT NULL,
          checkpoint_ref TEXT NOT NULL,
          source_index_row INTEGER NOT NULL,
          source_index_record_hash TEXT NOT NULL,
          source_node_line INTEGER NOT NULL CHECK(source_node_line >= 1),
          source_node_record_hash TEXT NOT NULL,
          source_record_hash TEXT NOT NULL,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS edge_catalog (
          edge_id TEXT PRIMARY KEY,
          edge_class TEXT NOT NULL CHECK(edge_class IN ('evidence','constraint','relation')),
          edge_kind TEXT NOT NULL,
          edge_type TEXT NOT NULL,
          source_node TEXT NOT NULL,
          target_node TEXT NOT NULL,
          strength REAL,
          source_path TEXT NOT NULL,
          raw_hash TEXT NOT NULL,
          source_line INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS binding_edge_refs (
          node_id TEXT NOT NULL,
          edge_id TEXT NOT NULL,
          edge_class TEXT NOT NULL,
          role TEXT NOT NULL,
          PRIMARY KEY(node_id, edge_id, role),
          FOREIGN KEY(node_id) REFERENCES bindings(node_id),
          FOREIGN KEY(edge_id) REFERENCES edge_catalog(edge_id)
        );

        CREATE TABLE IF NOT EXISTS update_events (
          event_id TEXT PRIMARY KEY,
          event_schema TEXT NOT NULL,
          node_id TEXT NOT NULL,
          sequence INTEGER NOT NULL CHECK(sequence >= 1),
          step INTEGER NOT NULL CHECK(step >= 0),
          reason TEXT NOT NULL,
          old_embedding_ref TEXT NOT NULL,
          new_embedding_ref TEXT NOT NULL,
          old_state TEXT NOT NULL,
          new_state TEXT NOT NULL,
          old_confidence REAL NOT NULL CHECK(old_confidence >= 0 AND old_confidence <= 1),
          new_confidence REAL NOT NULL CHECK(new_confidence >= 0 AND new_confidence <= 1),
          checkpoint_ref TEXT NOT NULL,
          model_version TEXT NOT NULL,
          graph_version TEXT NOT NULL,
          prev_event_hash TEXT NOT NULL,
          event_hash TEXT NOT NULL UNIQUE,
          payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          emitted_by TEXT NOT NULL,
          UNIQUE(node_id, sequence),
          FOREIGN KEY(node_id) REFERENCES bindings(node_id)
        );

        CREATE TABLE IF NOT EXISTS event_edge_refs (
          event_id TEXT NOT NULL,
          edge_id TEXT NOT NULL,
          edge_class TEXT NOT NULL,
          role TEXT NOT NULL,
          PRIMARY KEY(event_id, edge_id, role),
          FOREIGN KEY(event_id) REFERENCES update_events(event_id),
          FOREIGN KEY(edge_id) REFERENCES edge_catalog(edge_id)
        );

        CREATE TABLE IF NOT EXISTS binding_heads (
          node_id TEXT PRIMARY KEY,
          latest_event_id TEXT,
          event_count INTEGER NOT NULL CHECK(event_count >= 0),
          embedding_ref TEXT NOT NULL,
          state TEXT NOT NULL,
          confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
          last_step INTEGER NOT NULL CHECK(last_step >= -1),
          latest_event_hash TEXT NOT NULL,
          FOREIGN KEY(node_id) REFERENCES bindings(node_id),
          FOREIGN KEY(latest_event_id) REFERENCES update_events(event_id)
        );

        CREATE INDEX IF NOT EXISTS idx_bindings_sample ON bindings(sample_id);
        CREATE INDEX IF NOT EXISTS idx_bindings_type ON bindings(node_type);
        CREATE INDEX IF NOT EXISTS idx_edges_source ON edge_catalog(source_node);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON edge_catalog(target_node);
        CREATE INDEX IF NOT EXISTS idx_binding_refs_node_class ON binding_edge_refs(node_id, edge_class);
        CREATE INDEX IF NOT EXISTS idx_events_node_sequence ON update_events(node_id, sequence);
        CREATE INDEX IF NOT EXISTS idx_event_refs_event_class ON event_edge_refs(event_id, edge_class);
        """
    )
    _install_immutable_triggers(connection)


def _install_immutable_triggers(connection: sqlite3.Connection) -> None:
    for table in (
        "metadata", "bindings", "edge_catalog", "binding_edge_refs",
        "update_events", "event_edge_refs",
    ):
        connection.execute(
            "CREATE TRIGGER IF NOT EXISTS no_update_%s BEFORE UPDATE ON %s "
            "BEGIN SELECT RAISE(ABORT, '%s is append-only'); END" % (table, table, table)
        )
        connection.execute(
            "CREATE TRIGGER IF NOT EXISTS no_delete_%s BEFORE DELETE ON %s "
            "BEGIN SELECT RAISE(ABORT, '%s is append-only'); END" % (table, table, table)
        )


def set_metadata(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
        (str(key), canonical_json(value)),
    )


def get_metadata(connection: sqlite3.Connection) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for row in connection.execute("SELECT key, value_json FROM metadata ORDER BY key"):
        result[row["key"]] = json.loads(row["value_json"])
    return result


def _row_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


def _normalise_edge_ids(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        raise BindingStoreError("Edge IDs must be a string or list")
    return sorted(set(str(item).strip() for item in values if str(item).strip()))


def normalize_event_input(record: Mapping[str, Any]) -> Dict[str, Any]:
    node_id, _ = resolve_alias(record, NODE_ID_KEYS)
    payload_value = record.get("payload")
    payload = dict(payload_value) if isinstance(payload_value, Mapping) else {}
    # Preserve the training/inference objective context named by the design
    # document without forcing one historical schema.  These values are part
    # of the event hash and become directly visible in trace output.
    for source_key, payload_key in (
        ("loss", "loss"), ("loss_value", "loss"),
        ("losses", "losses"), ("objective", "objective"),
        ("metrics", "metrics"),
    ):
        if source_key in record and payload_key not in payload:
            payload[payload_key] = record.get(source_key)
    def first(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in record and record.get(key) is not None:
                return record.get(key)
        return default
    new_edges_value = first("new_edges", "derived_edges", "provenance_edges", default=[])
    if new_edges_value is None:
        new_edges_value = []
    if not isinstance(new_edges_value, list) or not all(
        isinstance(item, Mapping) for item in new_edges_value
    ):
        raise BindingStoreError("new_edges/derived_edges must be a list of objects")
    return {
        "node_id": node_id,
        "step": int(first("step", "iteration", "round", default=0)),
        "reason": str(first("reason", "update_reason", "cause", default="unspecified")).strip(),
        "new_embedding_ref": str(first("new_embedding_ref", "embedding_ref", default="")).strip(),
        "new_state": str(first("new_state", "state", "status", default="")).strip(),
        "new_confidence": first("new_confidence", "confidence", "score", default=None),
        "expected_old_embedding_ref": str(first("expected_old_embedding_ref", "old_embedding_ref", default="")).strip(),
        "expected_old_state": str(first("expected_old_state", "old_state", default="")).strip(),
        "expected_old_confidence": first("expected_old_confidence", "old_confidence", default=None),
        "checkpoint_ref": str(first("checkpoint_ref", "checkpoint", default="")).strip(),
        "model_version": str(first("model_version", default="")).strip(),
        "graph_version": str(first("graph_version", default="")).strip(),
        "evidence_edge_ids": _normalise_edge_ids(first(
            "evidence_edge_ids", "evidence_ids", "supporting_evidence", default=[]
        )),
        "constraint_edge_ids": _normalise_edge_ids(first(
            "constraint_edge_ids", "constraint_ids", "applied_constraints", default=[]
        )),
        "new_edges": [dict(item) for item in new_edges_value],
        "payload": payload,
        "created_at": str(first("created_at", "timestamp", default="")).strip(),
        "emitted_by": str(first("emitted_by", "producer", default="binding_event_cli_v6_4")).strip(),
    }


def _event_hash_material(
    node_id: str,
    sequence: int,
    step: int,
    reason: str,
    old_embedding_ref: str,
    new_embedding_ref: str,
    old_state: str,
    new_state: str,
    old_confidence: float,
    new_confidence: float,
    checkpoint_ref: str,
    model_version: str,
    graph_version: str,
    prev_event_hash: str,
    evidence_edge_ids: Sequence[str],
    constraint_edge_ids: Sequence[str],
    payload: Mapping[str, Any],
    created_at: str,
    emitted_by: str,
) -> Dict[str, Any]:
    return {
        "event_schema": EVENT_SCHEMA,
        "node_id": node_id,
        "sequence": int(sequence),
        "step": int(step),
        "reason": reason,
        "old_embedding_ref": old_embedding_ref,
        "new_embedding_ref": new_embedding_ref,
        "old_state": old_state,
        "new_state": new_state,
        "old_confidence": old_confidence,
        "new_confidence": new_confidence,
        "checkpoint_ref": checkpoint_ref,
        "model_version": model_version,
        "graph_version": graph_version,
        "prev_event_hash": prev_event_hash,
        "evidence_edge_ids": list(evidence_edge_ids),
        "constraint_edge_ids": list(constraint_edge_ids),
        "payload": dict(payload),
        "created_at": created_at,
        "emitted_by": emitted_by,
    }


def append_event(
    connection: sqlite3.Connection,
    event: Mapping[str, Any],
    dry_run: bool = False,
    manage_transaction: bool = True,
) -> Dict[str, Any]:
    """Append one event and update the derived head.

    Normal callers keep ``manage_transaction=True`` and receive one atomic
    transaction per event.  The batch CLI uses ``manage_transaction=False``
    only inside its own outer dry-run transaction so dependent events can be
    simulated in order and then rolled back together.
    """
    if dry_run and not manage_transaction:
        raise BindingStoreError(
            "dry_run and manage_transaction=False cannot be combined"
        )
    normalized = normalize_event_input(event)
    node_id = normalized["node_id"]
    if not node_id:
        raise BindingStoreError("UpdateEvent is missing node_id")
    reason = normalized["reason"]
    if not reason:
        raise BindingStoreError("UpdateEvent reason cannot be empty")

    if manage_transaction:
        connection.execute("BEGIN IMMEDIATE")
    try:
        binding = connection.execute(
            "SELECT * FROM bindings WHERE node_id=?", (node_id,)
        ).fetchone()
        head = connection.execute(
            "SELECT * FROM binding_heads WHERE node_id=?", (node_id,)
        ).fetchone()
        if binding is None or head is None:
            raise BindingStoreError("Unknown binding node_id: %s" % node_id)

        old_embedding_ref = str(head["embedding_ref"])
        old_state = str(head["state"])
        old_confidence = float(head["confidence"])
        for expected, actual, label in (
            (normalized["expected_old_embedding_ref"], old_embedding_ref, "embedding_ref"),
            (normalized["expected_old_state"], old_state, "state"),
        ):
            if expected and str(expected) != str(actual):
                raise ConcurrencyError(
                    "Stale expected old %s for %s: expected=%r actual=%r"
                    % (label, node_id, expected, actual)
                )
        if normalized["expected_old_confidence"] is not None:
            expected_confidence = validate_confidence(normalized["expected_old_confidence"])
            if not math.isclose(expected_confidence, old_confidence, rel_tol=0.0, abs_tol=1e-12):
                raise ConcurrencyError(
                    "Stale expected old confidence for %s: expected=%r actual=%r"
                    % (node_id, expected_confidence, old_confidence)
                )

        new_embedding_ref = normalized["new_embedding_ref"] or old_embedding_ref
        new_state = validate_state(normalized["new_state"] or old_state)
        new_confidence = (
            old_confidence if normalized["new_confidence"] is None
            else validate_confidence(normalized["new_confidence"])
        )
        step = int(normalized["step"])
        if step < int(head["last_step"]):
            raise BindingStoreError(
                "Event step cannot move backwards for %s: %d < %d"
                % (node_id, step, int(head["last_step"]))
            )

        evidence_ids = list(normalized["evidence_edge_ids"])
        constraint_ids = list(normalized["constraint_edge_ids"])
        payload = dict(normalized["payload"])
        new_edge_ids: List[str] = []
        new_edge_hashes: Dict[str, str] = {}
        for position, raw_edge in enumerate(normalized["new_edges"], start=1):
            resolved = resolve_edge(raw_edge, line_number=0)
            if resolved["edge_class"] not in {"evidence", "constraint"}:
                raise BindingStoreError(
                    "UpdateEvent new edge must be evidence/constraint, got %s"
                    % resolved["edge_class"]
                )
            if not resolved["source_node"] or not resolved["target_node"]:
                raise BindingStoreError(
                    "UpdateEvent new edge %d has an unresolved endpoint" % position
                )
            if node_id not in {resolved["source_node"], resolved["target_node"]}:
                raise BindingStoreError(
                    "UpdateEvent new edge %s is not incident to updated node %s"
                    % (resolved["edge_id"], node_id)
                )
            existing_edge = connection.execute(
                "SELECT raw_hash,edge_class FROM edge_catalog WHERE edge_id=?",
                (resolved["edge_id"],),
            ).fetchone()
            if existing_edge is not None:
                if str(existing_edge["raw_hash"]) != resolved["raw_hash"]:
                    raise BindingStoreError(
                        "Conflicting append-only edge_id: %s" % resolved["edge_id"]
                    )
                if str(existing_edge["edge_class"]) != resolved["edge_class"]:
                    raise BindingStoreError(
                        "Existing edge class mismatch: %s" % resolved["edge_id"]
                    )
            else:
                connection.execute(
                    """
                    INSERT INTO edge_catalog(
                      edge_id,edge_class,edge_kind,edge_type,source_node,target_node,
                      strength,source_path,raw_hash,source_line
                    ) VALUES (?,?,?,?,?,?,?,?,?,0)
                    """,
                    (
                        resolved["edge_id"], resolved["edge_class"],
                        resolved["edge_kind"], resolved["edge_type"],
                        resolved["source_node"], resolved["target_node"],
                        resolved["strength"], resolved["source_path"],
                        resolved["raw_hash"],
                    ),
                )
            for endpoint, role in (
                (resolved["source_node"], "source"),
                (resolved["target_node"], "target"),
            ):
                if connection.execute(
                    "SELECT 1 FROM bindings WHERE node_id=?", (endpoint,)
                ).fetchone() is not None:
                    connection.execute(
                        "INSERT OR IGNORE INTO binding_edge_refs(node_id,edge_id,edge_class,role) "
                        "VALUES (?,?,?,?)",
                        (endpoint, resolved["edge_id"], resolved["edge_class"], role),
                    )
            new_edge_ids.append(resolved["edge_id"])
            new_edge_hashes[resolved["edge_id"]] = resolved["raw_hash"]
            if resolved["edge_class"] == "evidence":
                evidence_ids.append(resolved["edge_id"])
            else:
                constraint_ids.append(resolved["edge_id"])
        evidence_ids = sorted(set(evidence_ids))
        constraint_ids = sorted(set(constraint_ids))
        if new_edge_hashes:
            payload["_appended_edge_hashes"] = new_edge_hashes
        overlap = set(evidence_ids).intersection(constraint_ids)
        if overlap:
            raise BindingStoreError(
                "The same edge cannot be both evidence and constraint in one event: %s"
                % sorted(overlap)[:5]
            )
        expected_classes: List[Tuple[str, str]] = [
            (edge_id, "evidence") for edge_id in evidence_ids
        ] + [
            (edge_id, "constraint") for edge_id in constraint_ids
        ]
        for edge_id, expected_class in expected_classes:
            row = connection.execute(
                "SELECT edge_class FROM edge_catalog WHERE edge_id=?", (edge_id,)
            ).fetchone()
            if row is None:
                raise BindingStoreError("UpdateEvent cites unknown edge_id: %s" % edge_id)
            if str(row["edge_class"]) != expected_class:
                raise BindingStoreError(
                    "UpdateEvent edge class mismatch for %s: expected=%s actual=%s"
                    % (edge_id, expected_class, row["edge_class"])
                )

        sequence = int(head["event_count"]) + 1
        prev_hash = str(head["latest_event_hash"] or "")
        checkpoint_ref = normalized["checkpoint_ref"] or str(binding["checkpoint_ref"])
        model_version = normalized["model_version"] or str(binding["model_version"])
        graph_version = normalized["graph_version"] or str(binding["graph_version"])
        created_at = normalized["created_at"] or utc_now()
        emitted_by = normalized["emitted_by"] or "binding_event_cli_v6_4"
        material = _event_hash_material(
            node_id=node_id,
            sequence=sequence,
            step=step,
            reason=reason,
            old_embedding_ref=old_embedding_ref,
            new_embedding_ref=new_embedding_ref,
            old_state=old_state,
            new_state=new_state,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            checkpoint_ref=checkpoint_ref,
            model_version=model_version,
            graph_version=graph_version,
            prev_event_hash=prev_hash,
            evidence_edge_ids=evidence_ids,
            constraint_edge_ids=constraint_ids,
            payload=payload,
            created_at=created_at,
            emitted_by=emitted_by,
        )
        event_hash = sha256_text(canonical_json(material))
        event_id = "event::" + event_hash[:32]
        result = dict(material)
        result["event_id"] = event_id
        result["event_hash"] = event_hash
        result["new_edge_ids"] = new_edge_ids

        if dry_run:
            connection.rollback()
            result["dry_run"] = True
            return result

        connection.execute(
            """
            INSERT INTO update_events(
              event_id,event_schema,node_id,sequence,step,reason,
              old_embedding_ref,new_embedding_ref,old_state,new_state,
              old_confidence,new_confidence,checkpoint_ref,model_version,
              graph_version,prev_event_hash,event_hash,payload_json,created_at,emitted_by
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event_id, EVENT_SCHEMA, node_id, sequence, step, reason,
                old_embedding_ref, new_embedding_ref, old_state, new_state,
                old_confidence, new_confidence, checkpoint_ref, model_version,
                graph_version, prev_hash, event_hash,
                canonical_json(payload), created_at, emitted_by,
            ),
        )
        for edge_id, edge_class in expected_classes:
            connection.execute(
                "INSERT INTO event_edge_refs(event_id,edge_id,edge_class,role) VALUES (?,?,?,?)",
                (event_id, edge_id, edge_class, "update_support"),
            )
        connection.execute(
            """
            UPDATE binding_heads SET latest_event_id=?, event_count=?, embedding_ref=?,
              state=?, confidence=?, last_step=?, latest_event_hash=? WHERE node_id=?
            """,
            (
                event_id, sequence, new_embedding_ref, new_state,
                new_confidence, step, event_hash, node_id,
            ),
        )
        if manage_transaction:
            connection.commit()
        return result
    except Exception:
        if manage_transaction:
            connection.rollback()
        raise


def _edge_with_nodes(connection: sqlite3.Connection, edge_id: str) -> Dict[str, Any]:
    row = connection.execute(
        "SELECT * FROM edge_catalog WHERE edge_id=?", (edge_id,)
    ).fetchone()
    if row is None:
        return {"edge_id": edge_id, "missing": True}
    edge = dict(row)
    for label in ("source_node", "target_node"):
        node_id = edge.get(label, "")
        node = connection.execute(
            "SELECT node_id,node_type,sample_id,layer,raw_id,embedding_ref,"
            "source_index_row,source_node_line,source_node_record_hash "
            "FROM bindings WHERE node_id=?", (node_id,)
        ).fetchone()
        edge[label + "_binding"] = _row_dict(node)
    return edge


def trace_node(
    connection: sqlite3.Connection,
    node_id: str,
    max_depth: int = 0,
) -> Dict[str, Any]:
    """Return BindingRecord, evidence/constraint sources and chronological history."""
    binding = connection.execute(
        "SELECT * FROM bindings WHERE node_id=?", (node_id,)
    ).fetchone()
    if binding is None:
        raise BindingStoreError("Unknown binding node_id: %s" % node_id)
    head = connection.execute(
        "SELECT * FROM binding_heads WHERE node_id=?", (node_id,)
    ).fetchone()
    refs = list(connection.execute(
        "SELECT edge_id,edge_class,role FROM binding_edge_refs "
        "WHERE node_id=? ORDER BY edge_class,edge_id,role", (node_id,)
    ))
    events = []
    for row in connection.execute(
        "SELECT * FROM update_events WHERE node_id=? ORDER BY sequence", (node_id,)
    ):
        event = dict(row)
        event["payload"] = json.loads(event.pop("payload_json"))
        event_refs = list(connection.execute(
            "SELECT edge_id,edge_class,role FROM event_edge_refs "
            "WHERE event_id=? ORDER BY edge_class,edge_id", (row["event_id"],)
        ))
        event["edge_refs"] = [dict(ref) for ref in event_refs]
        events.append(event)

    evidence_ids = sorted(set(
        str(ref["edge_id"]) for ref in refs if ref["edge_class"] == "evidence"
    ))
    constraint_ids = sorted(set(
        str(ref["edge_id"]) for ref in refs if ref["edge_class"] == "constraint"
    ))
    for event in events:
        for ref in event["edge_refs"]:
            if ref["edge_class"] == "evidence":
                evidence_ids.append(ref["edge_id"])
            elif ref["edge_class"] == "constraint":
                constraint_ids.append(ref["edge_id"])
    evidence_ids = sorted(set(evidence_ids))
    constraint_ids = sorted(set(constraint_ids))
    metadata = get_metadata(connection)

    trace: Dict[str, Any] = {
        "schema": TRACE_SCHEMA,
        "node_id": node_id,
        "binding_record": dict(binding),
        "current_head": dict(head) if head is not None else None,
        "source_artifacts": {
            "embedding_index_path": metadata.get("embedding_index_path", ""),
            "embeddings_path": metadata.get("embeddings_path", ""),
            "nodes_path": metadata.get("nodes_path", ""),
            "edges_path": metadata.get("edges_path", ""),
            "embedding_manifest_path": metadata.get("embedding_manifest_path", ""),
            "embedding_index_sha256": metadata.get("embedding_index_sha256", ""),
            "embeddings_sha256": metadata.get("embeddings_sha256", ""),
            "nodes_sha256": metadata.get("nodes_sha256", ""),
            "edges_sha256": metadata.get("edges_sha256", ""),
        },
        "evidence_edges": [_edge_with_nodes(connection, edge_id) for edge_id in evidence_ids],
        "constraint_edges": [_edge_with_nodes(connection, edge_id) for edge_id in constraint_ids],
        "update_timeline": events,
        "counts": {
            "evidence_edges": len(evidence_ids),
            "constraint_edges": len(constraint_ids),
            "update_events": len(events),
        },
    }
    if max_depth > 0:
        visited = {node_id}
        frontier = [node_id]
        lineage = []
        for depth in range(1, max_depth + 1):
            next_frontier: List[str] = []
            for current in frontier:
                current_refs = connection.execute(
                    """
                    SELECT e.* FROM edge_catalog e
                    JOIN binding_edge_refs r ON r.edge_id=e.edge_id
                    WHERE r.node_id=? AND r.edge_class IN ('evidence','constraint')
                    ORDER BY e.edge_id
                    """, (current,)
                ).fetchall()
                for edge_row in current_refs:
                    edge = dict(edge_row)
                    candidates = []
                    if edge["target_node"] == current and edge["source_node"]:
                        candidates.append(edge["source_node"])
                    elif edge["edge_class"] == "constraint":
                        if edge["source_node"] == current and edge["target_node"]:
                            candidates.append(edge["target_node"])
                    for candidate in candidates:
                        item = connection.execute(
                            "SELECT node_id,node_type,sample_id,embedding_ref FROM bindings WHERE node_id=?",
                            (candidate,),
                        ).fetchone()
                        lineage.append({
                            "depth": depth,
                            "from_node": current,
                            "via_edge_id": edge["edge_id"],
                            "via_edge_class": edge["edge_class"],
                            "to_node": candidate,
                            "binding": _row_dict(item),
                            "cycle_or_duplicate": candidate in visited,
                        })
                        if item is not None and candidate not in visited:
                            visited.add(candidate)
                            next_frontier.append(candidate)
            frontier = next_frontier
            if not frontier:
                break
        trace["recursive_lineage"] = lineage
        trace["recursive_nodes_visited"] = len(visited)
    return trace


def recompute_event_hash(event_row: Mapping[str, Any], edge_refs: Sequence[Mapping[str, Any]]) -> str:
    evidence_ids = sorted(
        str(ref["edge_id"]) for ref in edge_refs if ref["edge_class"] == "evidence"
    )
    constraint_ids = sorted(
        str(ref["edge_id"]) for ref in edge_refs if ref["edge_class"] == "constraint"
    )
    payload_json = event_row["payload_json"]
    payload = json.loads(payload_json) if isinstance(payload_json, str) else dict(payload_json)
    material = _event_hash_material(
        node_id=str(event_row["node_id"]),
        sequence=int(event_row["sequence"]),
        step=int(event_row["step"]),
        reason=str(event_row["reason"]),
        old_embedding_ref=str(event_row["old_embedding_ref"]),
        new_embedding_ref=str(event_row["new_embedding_ref"]),
        old_state=str(event_row["old_state"]),
        new_state=str(event_row["new_state"]),
        old_confidence=float(event_row["old_confidence"]),
        new_confidence=float(event_row["new_confidence"]),
        checkpoint_ref=str(event_row["checkpoint_ref"]),
        model_version=str(event_row["model_version"]),
        graph_version=str(event_row["graph_version"]),
        prev_event_hash=str(event_row["prev_event_hash"]),
        evidence_edge_ids=evidence_ids,
        constraint_edge_ids=constraint_ids,
        payload=payload,
        created_at=str(event_row["created_at"]),
        emitted_by=str(event_row["emitted_by"]),
    )
    return sha256_text(canonical_json(material))


def rebuild_heads(connection: sqlite3.Connection) -> int:
    """Recreate the derived head cache from immutable genesis + event history."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute("DELETE FROM binding_heads")
        connection.execute(
            """
            INSERT INTO binding_heads(
              node_id,latest_event_id,event_count,embedding_ref,state,confidence,
              last_step,latest_event_hash
            )
            SELECT node_id,NULL,0,embedding_ref,initial_state,initial_confidence,-1,''
            FROM bindings
            """
        )
        event_count = 0
        for event in connection.execute(
            "SELECT * FROM update_events ORDER BY node_id,sequence"
        ).fetchall():
            head = connection.execute(
                "SELECT * FROM binding_heads WHERE node_id=?", (event["node_id"],)
            ).fetchone()
            if head is None:
                raise BindingStoreError("Event references missing binding: %s" % event["node_id"])
            if int(event["sequence"]) != int(head["event_count"]) + 1:
                raise BindingStoreError("Non-contiguous event sequence for %s" % event["node_id"])
            if str(event["prev_event_hash"]) != str(head["latest_event_hash"]):
                raise BindingStoreError("Broken event hash chain for %s" % event["node_id"])
            if str(event["old_embedding_ref"]) != str(head["embedding_ref"]):
                raise BindingStoreError("Broken embedding transition for %s" % event["node_id"])
            if str(event["old_state"]) != str(head["state"]):
                raise BindingStoreError("Broken state transition for %s" % event["node_id"])
            if not math.isclose(
                float(event["old_confidence"]), float(head["confidence"]),
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise BindingStoreError("Broken confidence transition for %s" % event["node_id"])
            connection.execute(
                """
                UPDATE binding_heads SET latest_event_id=?,event_count=?,embedding_ref=?,
                  state=?,confidence=?,last_step=?,latest_event_hash=? WHERE node_id=?
                """,
                (
                    event["event_id"], event["sequence"], event["new_embedding_ref"],
                    event["new_state"], event["new_confidence"], event["step"],
                    event["event_hash"], event["node_id"],
                ),
            )
            event_count += 1
        connection.commit()
        return event_count
    except Exception:
        connection.rollback()
        raise


def table_count(connection: sqlite3.Connection, table: str) -> int:
    if table not in {
        "metadata", "bindings", "edge_catalog", "binding_edge_refs",
        "update_events", "event_edge_refs", "binding_heads",
    }:
        raise ValueError("Unsupported table: %s" % table)
    return int(connection.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0])
