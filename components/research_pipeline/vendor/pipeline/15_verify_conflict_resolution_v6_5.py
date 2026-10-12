#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict structural verifier for a v6.5 mask/project/blend output directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


SCHEMA = "task2_mask_project_blend_verification_v6_5"
HARD_API_SOURCES = {"import_table", "symbol_table", "exact_signature"}
REQUIRED_RESULT_CHECKS = (
    "embedding_finite",
    "probabilities_finite",
    "probabilities_sum_to_one",
    "lower_levels_did_not_unmask_hard_candidates",
    "unresolved_hard_has_hypothesis",
    "max_iteration_respected",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise RuntimeError("Expected object at %s:%d" % (path, line_number))
            yield value


def extract_tensor(payload: Any):
    if torch is None:
        raise RuntimeError("PyTorch is required by this verifier")
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, Mapping):
        for key in ("embeddings", "node_embeddings", "tensor", "x", "data"):
            value = payload.get(key)
            if torch.is_tensor(value):
                return value
    raise RuntimeError("Resolved embedding payload has no tensor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-results", type=int, default=1)
    parser.add_argument("--binding-store", type=Path)
    parser.add_argument("--require-events-applied", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = {
        "embeddings": args.out_dir / "node_embeddings_resolved_v6_5.pt",
        "index": args.out_dir / "node_embedding_index_resolved_v6_5.json",
        "results": args.out_dir / "resolution_results_v6_5.jsonl",
        "conflicts": args.out_dir / "active_conflicts_v6_5.jsonl",
        "events": args.out_dir / "binding_update_events_v6_5.jsonl",
        "summary": args.out_dir / "resolution_summary_v6_5.json",
        "manifest": args.out_dir / "resolution_manifest_v6_5.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing v6.5 outputs: %s" % missing)
    summary = read_json(required["summary"])
    manifest = read_json(required["manifest"])
    index = read_json(required["index"])
    results = list(read_jsonl(required["results"]))
    conflicts = list(read_jsonl(required["conflicts"]))
    events = list(read_jsonl(required["events"]))
    try:
        payload = torch.load(required["embeddings"], map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(required["embeddings"], map_location="cpu")
    tensor = extract_tensor(payload).detach().cpu()

    result_ids = [str(row.get("node_id") or "") for row in results]
    event_ids = [str(row.get("node_id") or "") for row in events]
    states = Counter(str(row.get("state") or "") for row in results)
    stops = Counter(str(row.get("stop_reason") or "") for row in results)
    level3_api_bad = []
    invalid_result_checks = []
    unresolved_without_hypothesis = []
    incompatible_false_convergence = []
    for row in results:
        checks = row.get("checks") if isinstance(row.get("checks"), Mapping) else {}
        missing_or_false = [
            key for key in REQUIRED_RESULT_CHECKS if not bool(checks.get(key))
        ]
        if missing_or_false:
            invalid_result_checks.append({
                "node_id": row.get("node_id"),
                "missing_or_false": missing_or_false,
            })
        if row.get("unresolved_hard_conflict") and not row.get("new_hypotheses"):
            unresolved_without_hypothesis.append(row.get("node_id"))
        if row.get("projection_incompatibilities") and row.get("stop_reason") != "max_iterations":
            incompatible_false_convergence.append(row.get("node_id"))
        for constraint in row.get("constraint_records") or []:
            if not isinstance(constraint, Mapping):
                continue
            if int(constraint.get("level") or 0) != 3:
                continue
            if str(constraint.get("origin") or "").startswith("api_"):
                source = str(constraint.get("hard_source") or "")
                if source not in HARD_API_SOURCES or not constraint.get("hard_source_verified"):
                    level3_api_bad.append({
                        "node_id": row.get("node_id"),
                        "constraint_id": constraint.get("constraint_id"),
                        "hard_source": source,
                    })

    hashes_ok = True
    hash_errors = []
    for filename, descriptor in (manifest.get("outputs") or {}).items():
        path = args.out_dir / filename
        expected = str((descriptor or {}).get("sha256") or "")
        actual = sha256_file(path) if path.is_file() else ""
        if expected != actual:
            hashes_ok = False
            hash_errors.append({"file": filename, "expected": expected, "actual": actual})

    binding_check = {"requested": args.require_events_applied, "available": False, "events_found": 0}
    if args.binding_store is not None:
        if not args.binding_store.is_file():
            raise FileNotFoundError(args.binding_store)
        connection = sqlite3.connect(str(args.binding_store))
        count = connection.execute(
            "SELECT COUNT(*) FROM update_events WHERE emitted_by=?",
            ("13_resolve_hierarchical_constraints.py:v6.5",),
        ).fetchone()[0]
        connection.close()
        binding_check = {"requested": args.require_events_applied, "available": True, "events_found": count}

    checks = {
        "summary_schema_valid": summary.get("schema") == "task2_mask_project_blend_batch_v6_5",
        "summary_passed": bool(summary.get("pass_structural_checks")),
        "manifest_formal_ready": bool(manifest.get("formal_ready")),
        "manifest_hashes_match": hashes_ok,
        "index_is_list": isinstance(index, list),
        "tensor_rank_two": tensor.ndim == 2,
        "tensor_rows_match_index": tensor.ndim == 2 and int(tensor.shape[0]) == len(index),
        "tensor_is_finite": bool(torch.isfinite(tensor).all()),
        "enough_results": len(results) >= args.min_results,
        "result_ids_nonempty_unique": all(result_ids) and len(result_ids) == len(set(result_ids)),
        "event_ids_unique": len(event_ids) == len(set(event_ids)),
        "result_checks_pass": not invalid_result_checks,
        "unresolved_hard_is_explicit": not unresolved_without_hypothesis,
        "incompatible_projection_did_not_false_converge": not incompatible_false_convergence,
        "api_level3_sources_are_verified": not level3_api_bad,
        "conflict_count_matches_summary": len(conflicts) == int(summary.get("active_conflict_records") or 0),
        "event_count_matches_summary": len(events) == int(summary.get("event_drafts") or 0),
        "node_errors_absent": not summary.get("node_errors"),
        "events_applied_if_required": (
            not args.require_events_applied
            or (binding_check["available"] and binding_check["events_found"] >= len(events))
        ),
    }
    report = {
        "schema": SCHEMA,
        "out_dir": str(args.out_dir.resolve()),
        "records": {
            "index": len(index) if isinstance(index, list) else 0,
            "embedding_rows": int(tensor.shape[0]) if tensor.ndim == 2 else 0,
            "results": len(results),
            "conflicts": len(conflicts),
            "event_drafts": len(events),
        },
        "state_counts": dict(states),
        "stop_reason_counts": dict(stops),
        "binding_store": binding_check,
        "errors": {
            "hashes": hash_errors,
            "invalid_result_checks": invalid_result_checks[:100],
            "unresolved_without_hypothesis": unresolved_without_hypothesis[:100],
            "incompatible_false_convergence": incompatible_false_convergence[:100],
            "invalid_api_hard_sources": level3_api_bad[:100],
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    report_path = args.report or args.out_dir / "verification_v6_5.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("VERIFY 2.2.1.6 v6.5 RESULT: FAILED")
    print("VERIFY 2.2.1.6 v6.5 RESULT: PASSED")


if __name__ == "__main__":
    main()
