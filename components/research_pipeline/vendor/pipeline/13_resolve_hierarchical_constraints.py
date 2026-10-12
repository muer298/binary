#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formal 2.2.1.6 mask/project/blend batch entry point (v6.5).

The program is intentionally sidecar-first.  It never rewrites USG JSONL or
the immutable genesis BindingRecord.  It writes a new embedding artifact,
resolution/audit JSONL and append-only UpdateEvent drafts.  BindingStore
mutation happens only with the explicit ``--apply-events`` switch and is one
transaction for the entire batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - executed on the GPU server
    torch = None


HERE = Path(__file__).resolve().parent
BINDING_DIR = HERE.parent / "2.2.1.5_对象级表示绑定与证据可追溯性保证"
if not BINDING_DIR.is_dir():
    BINDING_DIR = HERE
for source_dir in (HERE, BINDING_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from binding_store_v6_4 import (  # noqa: E402
    append_event,
    connect_store,
    iter_jsonl,
    load_index,
    resolve_edge,
    resolve_node_metadata,
)
from conflict_resolver_v6_5 import (  # noqa: E402
    HIGH_NODE_TYPES,
    ConflictResolver,
    ConstraintCompiler,
    ConstraintSpec,
    extract_candidate_scores,
    extract_confidence,
    extract_state,
    first_value,
    infer_node_type,
    normalize_node_type,
    serializable_result,
)


SCHEMA = "task2_mask_project_blend_batch_v6_5"
OUTPUT_EMBEDDINGS = "node_embeddings_resolved_v6_5.pt"
OUTPUT_INDEX = "node_embedding_index_resolved_v6_5.json"
OUTPUT_RESULTS = "resolution_results_v6_5.jsonl"
OUTPUT_CONFLICTS = "active_conflicts_v6_5.jsonl"
OUTPUT_EVENTS = "binding_update_events_v6_5.jsonl"
OUTPUT_SUMMARY = "resolution_summary_v6_5.json"
OUTPUT_MANIFEST = "resolution_manifest_v6_5.json"


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for conflict resolution")


def json_default(value: Any) -> Any:
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Not JSON serializable: %s" % type(value).__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default, allow_nan=False) + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    os.close(descriptor)
    temporary_path = Path(temporary)
    count = 0
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=json_default,
                    allow_nan=False,
                ) + "\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return count


def atomic_torch_save(value: Any, path: Path) -> None:
    require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def torch_load(path: Path) -> Any:
    require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def embedding_tensor(payload: Any) -> "torch.Tensor":
    if torch.is_tensor(payload):
        value = payload
    elif isinstance(payload, Mapping):
        value = None
        for key in ("embeddings", "node_embeddings", "tensor", "x", "data"):
            candidate = payload.get(key)
            if torch.is_tensor(candidate):
                value = candidate
                break
        if value is None:
            raise RuntimeError("Embedding payload has no tensor key")
    else:
        raise RuntimeError("Unsupported embedding payload type: %s" % type(payload).__name__)
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise RuntimeError("Embeddings must be a non-empty rank-2 tensor")
    value = value.detach().cpu().float().contiguous()
    if not torch.isfinite(value).all():
        raise RuntimeError("Input embeddings contain NaN/Inf")
    return value


def resolved_embedding_payload(original: Any, tensor: "torch.Tensor", source: Path) -> Any:
    if torch.is_tensor(original):
        return tensor
    output = dict(original) if isinstance(original, Mapping) else {}
    output["embeddings"] = tensor
    output["node_embeddings"] = tensor
    output["num_nodes"] = int(tensor.shape[0])
    output["embedding_dim"] = int(tensor.shape[1])
    output["resolution_schema"] = SCHEMA
    output["resolved_from"] = str(source.resolve())
    return output


def index_position(row: Mapping[str, Any], fallback: int) -> int:
    value, _ = first_value(row, ("index", "row", "row_index", "embedding_index"), fallback)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Non-integer embedding index at row %d: %r" % (fallback, value)) from exc


def index_node_id(row: Mapping[str, Any]) -> str:
    value, _ = first_value(row, ("node_id", "node_ref", "id", "uid", "semantic_node_id"), "")
    return str(value or "").strip()


def prepare_index(
    rows: Sequence[Mapping[str, Any]],
    row_count: int,
    output_embeddings_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Counter]:
    if len(rows) != row_count:
        raise RuntimeError("Index rows %d != embedding rows %d" % (len(rows), row_count))
    by_node: Dict[str, int] = {}
    output: List[Optional[Dict[str, Any]]] = [None] * row_count
    node_types: Counter = Counter()
    for fallback, source_row in enumerate(rows):
        row = dict(source_row)
        position = index_position(row, fallback)
        if position < 0 or position >= row_count or output[position] is not None:
            raise RuntimeError("Duplicate/out-of-range embedding index: %d" % position)
        node_id = index_node_id(row)
        if not node_id or node_id in by_node:
            raise RuntimeError("Missing or duplicate index node_id: %r" % node_id)
        node_type_value, _ = first_value(
            row, ("node_type", "semantic_type", "node_kind", "kind", "type"), ""
        )
        node_type = normalize_node_type(node_type_value or infer_node_type(node_id))
        row["index"] = position
        row["node_id"] = node_id
        row["node_type"] = node_type
        row["embedding_ref"] = "%s#row=%d" % (output_embeddings_name, position)
        output[position] = row
        by_node[node_id] = position
        node_types[node_type] += 1
    if any(row is None for row in output):
        raise RuntimeError("Index does not cover every embedding row")
    return [row for row in output if row is not None], by_node, node_types


class EmbeddingLookup(Mapping[str, "torch.Tensor"]):
    """Lazy node->tensor-row adapter; avoids one Tensor object per node."""

    def __init__(self, tensor: "torch.Tensor", by_node: Mapping[str, int]) -> None:
        self.tensor = tensor
        self.by_node = by_node

    def __getitem__(self, key: str) -> "torch.Tensor":
        return self.tensor[self.by_node[key]]

    def __iter__(self):
        return iter(self.by_node)

    def __len__(self) -> int:
        return len(self.by_node)

    def get(self, key: str, default=None):
        position = self.by_node.get(key)
        return self.tensor[position] if position is not None else default


def compact_node_record(raw: Mapping[str, Any], node_id: str) -> Dict[str, Any]:
    metadata = resolve_node_metadata(raw)
    compact: Dict[str, Any] = {
        "node_id": node_id,
        "node_type": normalize_node_type(metadata.get("node_type") or infer_node_type(node_id)),
        "sample_id": metadata.get("sample_id") or node_id.split("::", 1)[0],
    }
    scores = extract_candidate_scores(raw)
    if scores:
        compact["candidate_scores"] = scores
    for destination, aliases in (
        ("status", ("status", "state", "candidate_status", "semantic_status")),
        ("confidence", ("confidence", "semantic_confidence", "prediction_confidence")),
        ("api_name", ("api_name", "api_target", "function_name", "name")),
        ("api_source", ("api_source", "confirmation_source", "match_type", "source_kind", "provenance_source")),
        ("api_role", ("api_role", "parameter_role", "argument_role", "role", "io_role")),
        ("param_idx", ("param_idx", "parameter_index", "arg_index", "argument_index")),
        ("rank", ("rank", "candidate_rank")),
        ("reciprocal_rank", ("reciprocal_rank", "rr", "mrr_contribution")),
    ):
        value, path = first_value(raw, aliases, None)
        if value not in (None, "", [], {}):
            compact[destination] = value
            compact[destination + "_schema_path"] = path
    return compact


def load_sidecar_records(path: Optional[Path]) -> Iterable[Mapping[str, Any]]:
    if path is None:
        return []
    if path.suffix.lower() == ".jsonl":
        return [row for _, row in iter_jsonl(path)]
    root = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(root, list):
        return [row for row in root if isinstance(row, Mapping)]
    if isinstance(root, Mapping):
        for key in ("records", "nodes", "lineage", "items", "data"):
            value = root.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, Mapping)]
        rows = []
        for key, value in root.items():
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("node_id", key)
                rows.append(row)
        return rows
    raise RuntimeError("Unsupported provenance sidecar root")


def merge_sidecar(nodes: MutableMapping[str, Dict[str, Any]], path: Optional[Path]) -> Dict[str, Any]:
    counts: Counter = Counter()
    examples = []
    for raw in load_sidecar_records(path):
        node_id, _ = first_value(
            raw, ("node_id", "api_node_id", "target_id", "node_ref", "semantic_node_id"), ""
        )
        node_id = str(node_id or "")
        if not node_id:
            counts["missing_node_id"] += 1
            continue
        current = nodes.get(node_id)
        if current is None:
            counts["not_in_loaded_nodes"] += 1
            continue
        source, path_used = first_value(
            raw, ("api_source", "confirmation_source", "match_type", "source_kind", "provenance_source"), ""
        )
        api_name, name_path = first_value(raw, ("api_name", "api_target", "symbol_name", "name"), "")
        if source:
            current["api_source"] = source
            current["api_source_schema_path"] = "sidecar." + path_used
        if api_name and not current.get("api_name"):
            current["api_name"] = api_name
            current["api_name_schema_path"] = "sidecar." + name_path
        counts["merged"] += 1
        if len(examples) < 20:
            examples.append(node_id)
    return {"counts": dict(counts), "examples": examples}


def append_bounded(
    grouped: MutableMapping[str, List[ConstraintSpec]],
    item: ConstraintSpec,
    limit: int,
    overflow: Counter,
) -> None:
    bucket = grouped[item.node_id]
    bucket.append(item)
    if limit > 0 and len(bucket) > limit:
        bucket.sort(key=lambda value: (-value.level, -value.weight, value.constraint_id))
        dropped = bucket.pop()
        overflow["constraints_dropped"] += 1
        overflow["level_%d_dropped" % dropped.level] += 1


def make_event(
    result: Mapping[str, Any],
    constraints: Sequence[ConstraintSpec],
    row: int,
    embeddings_name: str,
) -> Dict[str, Any]:
    evidence_ids = sorted(set(
        item.source_edge_id for item in constraints
        if item.source_edge_id and item.source_edge_class == "evidence"
    ))
    constraint_ids = sorted(set(
        item.source_edge_id for item in constraints
        if item.source_edge_id and item.source_edge_class == "constraint"
    ))
    return {
        "node_id": result["node_id"],
        "step": int(result["iterations_run"]),
        "reason": "mask_project_blend_resolution_v6_5",
        "new_embedding_ref": "%s#row=%d" % (embeddings_name, row),
        "new_state": result["state"],
        "new_confidence": result["confidence"],
        "evidence_edge_ids": evidence_ids,
        "constraint_edge_ids": constraint_ids,
        "emitted_by": "13_resolve_hierarchical_constraints.py:v6.5",
        "payload": {
            "resolution_schema": result["schema"],
            "selected_candidate": result["selected_candidate"],
            "initial_candidates": result["initial_candidates"],
            "final_candidates": result["final_candidates"],
            "constraint_ids": result["constraint_ids"],
            "constraint_levels": result["constraint_levels"],
            "active_conflict_ids": result["active_conflict_ids"],
            "unresolved_hard_conflict": result["unresolved_hard_conflict"],
            "new_hypotheses": result["new_hypotheses"],
            "embedding_delta": result["embedding_delta"],
            "iterations_run": result["iterations_run"],
            "stop_reason": result["stop_reason"],
            "iteration_trace": result["iteration_trace"],
            "checks": result["checks"],
        },
    }


def apply_events_atomically(store_path: Path, events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    connection = connect_store(store_path)
    appended = []
    connection.execute("BEGIN IMMEDIATE")
    try:
        for event in events:
            appended.append(append_event(connection, event, manage_transaction=False))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {
        "applied": True,
        "events": len(appended),
        "first_event_id": appended[0]["event_id"] if appended else "",
        "last_event_id": appended[-1]["event_id"] if appended else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--api-provenance", type=Path)
    parser.add_argument("--binding-store", type=Path)
    parser.add_argument("--apply-events", action="store_true")
    parser.add_argument("--node-types", default="TypeSem,InterfaceSem,APISem,LayoutSem")
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--max-edge-rows", type=int, default=0)
    parser.add_argument("--max-constraints-per-node", type=int, default=64)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--stable-rounds", type=int, default=2)
    parser.add_argument("--confirmation-threshold", type=float, default=0.85)
    parser.add_argument("--max-statistical-alpha", type=float, default=0.35)
    parser.add_argument("--high-strength-threshold", type=float, default=0.8)
    parser.add_argument("--vector-threshold", type=float, default=0.7)
    parser.add_argument("--require-model-rank1", action="store_true")
    parser.add_argument("--min-model-rr", type=float)
    parser.add_argument("--fail-on-unsupported", action="store_true")
    parser.add_argument("--allow-node-errors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_torch()
    for path in (args.nodes, args.edges, args.embeddings, args.index):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.api_provenance is not None and not args.api_provenance.is_file():
        raise FileNotFoundError(args.api_provenance)
    if args.apply_events and args.binding_store is None:
        raise RuntimeError("--apply-events requires --binding-store")
    if args.binding_store is not None and not args.binding_store.is_file():
        raise FileNotFoundError(args.binding_store)

    output_paths = [
        args.out_dir / name for name in (
            OUTPUT_EMBEDDINGS, OUTPUT_INDEX, OUTPUT_RESULTS, OUTPUT_CONFLICTS,
            OUTPUT_EVENTS, OUTPUT_SUMMARY, OUTPUT_MANIFEST,
        )
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: %s" % existing[:5])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    original_payload = torch_load(args.embeddings)
    source_embeddings = embedding_tensor(original_payload)
    corrected_embeddings = source_embeddings.clone()
    source_index = load_index(args.index)
    index_rows, by_node, index_type_counts = prepare_index(
        source_index, int(source_embeddings.shape[0]), OUTPUT_EMBEDDINGS
    )
    selected_types = set(
        normalize_node_type(item.strip()) for item in args.node_types.split(",") if item.strip()
    )
    if not selected_types:
        selected_types = set(HIGH_NODE_TYPES)
    selected_ids = []
    for row in index_rows:
        if normalize_node_type(row.get("node_type")) not in selected_types:
            continue
        selected_ids.append(row["node_id"])
        if args.max_nodes > 0 and len(selected_ids) >= args.max_nodes:
            break
    selected_set = set(selected_ids)
    if not selected_ids:
        raise RuntimeError("No nodes matched --node-types/--max-nodes")

    node_records: Dict[str, Dict[str, Any]] = {}
    node_scan: Counter = Counter()
    for line_number, raw in iter_jsonl(args.nodes):
        metadata = resolve_node_metadata(raw)
        node_id = metadata["node_id"]
        node_type = normalize_node_type(metadata["node_type"] or infer_node_type(node_id))
        if node_id in selected_set or node_type == "APISem":
            compact = compact_node_record(raw, node_id)
            previous = node_records.get(node_id)
            if previous is not None and previous != compact:
                raise RuntimeError("Conflicting duplicate node record: %s" % node_id)
            node_records[node_id] = compact
            node_scan["retained"] += 1
            node_scan["retained_" + node_type] += 1
        node_scan["rows"] += 1
        if args.progress_every > 0 and line_number % args.progress_every == 0:
            print(json.dumps({"phase": "scan_nodes", "rows": line_number, "retained": len(node_records)}), flush=True)
    sidecar_audit = merge_sidecar(node_records, args.api_provenance)
    embeddings_by_node = EmbeddingLookup(source_embeddings, by_node)
    compiler = ConstraintCompiler(
        node_records,
        embeddings_by_node,
        high_strength_threshold=args.high_strength_threshold,
        default_vector_threshold=args.vector_threshold,
        require_model_rank1=args.require_model_rank1,
        min_model_rr=args.min_model_rr,
    )
    constraints_by_node: Dict[str, List[ConstraintSpec]] = defaultdict(list)
    overflow: Counter = Counter()
    edge_scan: Counter = Counter()
    for line_number, raw in iter_jsonl(args.edges):
        if args.max_edge_rows > 0 and line_number > args.max_edge_rows:
            break
        resolved = resolve_edge(raw, line_number)
        source = resolved["source_node"]
        target = resolved["target_node"]
        edge_scan["rows"] += 1
        if source not in selected_set and target not in selected_set:
            edge_scan["not_incident_to_selected"] += 1
            continue
        for spec in compiler.compile_edge(raw, resolved):
            if spec.node_id in selected_set:
                append_bounded(
                    constraints_by_node,
                    spec,
                    args.max_constraints_per_node,
                    overflow,
                )
                edge_scan["compiled_for_selected"] += 1
        if args.progress_every > 0 and line_number % args.progress_every == 0:
            print(json.dumps({
                "phase": "scan_edges",
                "rows": line_number,
                "nodes_with_constraints": len(constraints_by_node),
            }), flush=True)
    for spec in compiler.compile_api_self_constraints(selected_ids):
        append_bounded(
            constraints_by_node,
            spec,
            args.max_constraints_per_node,
            overflow,
        )

    resolver = ConflictResolver(
        max_iterations=args.max_iterations,
        epsilon=args.epsilon,
        stable_rounds=args.stable_rounds,
        confirmation_threshold=args.confirmation_threshold,
        max_statistical_alpha=args.max_statistical_alpha,
        fail_on_unsupported=args.fail_on_unsupported,
    )
    results = []
    conflicts = []
    events = []
    outcomes: Counter = Counter()
    errors = []
    for ordinal, node_id in enumerate(selected_ids, start=1):
        record = node_records.get(node_id, {"node_id": node_id, "node_type": infer_node_type(node_id)})
        scores = extract_candidate_scores(record)
        if not scores:
            outcomes["missing_candidates"] += 1
            continue
        row = by_node[node_id]
        node_constraints = constraints_by_node.get(node_id, [])
        try:
            result = resolver.resolve(
                node_id,
                scores,
                source_embeddings[row],
                node_constraints,
                initial_state=extract_state(record),
                initial_confidence=extract_confidence(record, scores),
            )
        except Exception as exc:
            error = {"node_id": node_id, "error_type": type(exc).__name__, "error": str(exc)}
            errors.append(error)
            outcomes["node_errors"] += 1
            if not args.allow_node_errors:
                raise RuntimeError("Node resolution failed: %s" % error) from exc
            continue
        corrected_embeddings[row].copy_(result["embedding"].to(corrected_embeddings.dtype))
        public_result = serializable_result(result)
        public_result["embedding_row"] = row
        public_result["constraint_records"] = [item.as_dict() for item in node_constraints]
        results.append(public_result)
        outcomes["resolved"] += 1
        outcomes["state_" + result["state"]] += 1
        outcomes["stop_" + result["stop_reason"]] += 1
        if result["active_conflict_count"] or result["unresolved_hard_conflict"]:
            conflicts.append({
                "node_id": node_id,
                "state": result["state"],
                "stop_reason": result["stop_reason"],
                "active_conflict_ids": result["active_conflict_ids"],
                "unresolved_hard_conflict": result["unresolved_hard_conflict"],
                "new_hypotheses": result["new_hypotheses"],
                "projection_incompatibilities": result["projection_incompatibilities"],
            })
        changed = (
            result["embedding_delta"] > args.epsilon
            or result["state"] != extract_state(record)
            or result["final_candidates"] != result["initial_candidates"]
        )
        if changed:
            events.append(make_event(result, node_constraints, row, OUTPUT_EMBEDDINGS))
        if args.progress_every > 0 and ordinal % max(1, args.progress_every // 10) == 0:
            print(json.dumps({"phase": "resolve", "nodes": ordinal, "resolved": len(results)}), flush=True)

    if not torch.isfinite(corrected_embeddings).all():
        raise RuntimeError("Resolved embedding table contains NaN/Inf")
    result_path = args.out_dir / OUTPUT_RESULTS
    conflict_path = args.out_dir / OUTPUT_CONFLICTS
    event_path = args.out_dir / OUTPUT_EVENTS
    index_path = args.out_dir / OUTPUT_INDEX
    embeddings_path = args.out_dir / OUTPUT_EMBEDDINGS
    atomic_write_jsonl(result_path, results)
    atomic_write_jsonl(conflict_path, conflicts)
    atomic_write_jsonl(event_path, events)
    atomic_write_json(index_path, index_rows)
    atomic_torch_save(
        resolved_embedding_payload(original_payload, corrected_embeddings, args.embeddings),
        embeddings_path,
    )
    binding_audit = {"applied": False, "events": 0}
    if args.apply_events:
        binding_audit = apply_events_atomically(args.binding_store, events)

    checks = {
        "input_output_row_count_equal": int(corrected_embeddings.shape[0]) == len(index_rows),
        "all_embeddings_finite": bool(torch.isfinite(corrected_embeddings).all()),
        "at_least_one_node_resolved": bool(results),
        "every_result_has_finite_probabilities": all(
            result["checks"]["probabilities_finite"] for result in results
        ),
        "every_result_respects_hard_mask": all(
            result["checks"]["lower_levels_did_not_unmask_hard_candidates"] for result in results
        ),
        "every_unresolved_hard_has_hypothesis": all(
            result["checks"]["unresolved_hard_has_hypothesis"] for result in results
        ),
        "no_node_errors": not errors,
        "binding_store_mutation_was_explicit": (not binding_audit["applied"]) or args.apply_events,
    }
    summary = {
        "schema": SCHEMA,
        "inputs": {
            "nodes": str(args.nodes.resolve()),
            "edges": str(args.edges.resolve()),
            "embeddings": str(args.embeddings.resolve()),
            "index": str(args.index.resolve()),
            "api_provenance": str(args.api_provenance.resolve()) if args.api_provenance else "",
            "binding_store": str(args.binding_store.resolve()) if args.binding_store else "",
        },
        "configuration": vars(args),
        "embedding_shape": list(corrected_embeddings.shape),
        "index_type_counts": dict(index_type_counts),
        "selected_nodes": len(selected_ids),
        "loaded_node_records": len(node_records),
        "nodes_with_constraints": len(constraints_by_node),
        "constraint_count": sum(len(value) for value in constraints_by_node.values()),
        "node_scan": dict(node_scan),
        "edge_scan": dict(edge_scan),
        "compiler_audit": dict(compiler.audit),
        "compiler_skipped_examples": compiler.skipped[:100],
        "constraint_overflow": dict(overflow),
        "sidecar_audit": sidecar_audit,
        "outcomes": dict(outcomes),
        "results": len(results),
        "active_conflict_records": len(conflicts),
        "event_drafts": len(events),
        "binding_store": binding_audit,
        "node_errors": errors[:100],
        "checks": checks,
        "pass_structural_checks": all(checks.values()),
    }
    # Path objects in configuration are normalized by json_default.
    atomic_write_json(args.out_dir / OUTPUT_SUMMARY, summary)
    manifest = {
        "schema": SCHEMA,
        "formal_ready": bool(summary["pass_structural_checks"]),
        "source_fingerprints": {
            "embeddings_sha256": sha256_file(args.embeddings),
            "index_sha256": sha256_file(args.index),
        },
        "outputs": {
            path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (embeddings_path, index_path, result_path, conflict_path, event_path)
        },
        "checks": checks,
        "binding_store": binding_audit,
    }
    atomic_write_json(args.out_dir / OUTPUT_MANIFEST, manifest)
    print(json.dumps({
        "done": True,
        "out_dir": str(args.out_dir.resolve()),
        "selected_nodes": len(selected_ids),
        "results": len(results),
        "event_drafts": len(events),
        "active_conflicts": len(conflicts),
        "pass_structural_checks": summary["pass_structural_checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
