#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end Gold readiness gate: provenance, split isolation and coverage."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Set, Tuple


SEMANTIC_TYPES = ("APISem", "TypeSem", "LayoutSem", "InterfaceSem")
VALID_SPLITS = ("train", "validation", "test")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Non-object row at %s:%d" % (path, number))
            yield value


def index_samples(path: str) -> Set[str]:
    result = set()
    for row in read_jsonl(path):
        value = str(row.get("sample_id") or row.get("id") or "")
        if value:
            result.add(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", action="append", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--reasoning")
    parser.add_argument("--usg-index")
    parser.add_argument("--expected-samples", type=int, default=0)
    parser.add_argument("--min-labels-per-module-split", type=int, default=1)
    parser.add_argument("--min-candidate-coverage", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: %s" % out)

    nodes = {}
    for row in read_jsonl(args.nodes):
        node_id = str(row.get("node_id") or "")
        if node_id:
            nodes[node_id] = (
                str(row.get("sample_id") or ""),
                str(row.get("node_type") or ""),
            )

    labels = []
    duplicate_keys = Counter()
    group_splits = defaultdict(set)
    counts = Counter()
    module_split = Counter()
    errors = Counter()
    for path in args.gold:
        for row in read_jsonl(path):
            labels.append(row)
            query_id = str(row.get("query_id") or "")
            target_id = str(row.get("target_id") or "")
            semantic_type = str(row.get("target_semantic_type") or row.get("semantic_type") or "")
            sample_id = str(row.get("sample_id") or "")
            split = str(row.get("split") or "").lower()
            group_id = str(row.get("group_id") or sample_id)
            duplicate_keys[(query_id, target_id, semantic_type)] += 1
            group_splits[group_id].add(split)
            module_split[(semantic_type, split)] += 1
            counts["labels"] += 1
            if semantic_type not in SEMANTIC_TYPES:
                errors["invalid_semantic_type"] += 1
            if split not in VALID_SPLITS:
                errors["invalid_or_missing_split"] += 1
            if str(row.get("gold_strength") or "") != "strict_independent":
                errors["non_strict_or_missing_gold_strength"] += 1
            query = nodes.get(query_id)
            target = nodes.get(target_id)
            if query is None:
                errors["query_node_missing"] += 1
            if target is None:
                errors["target_node_missing"] += 1
            if query and sample_id and query[0] != sample_id:
                errors["query_sample_mismatch"] += 1
            if target and sample_id and target[0] != sample_id:
                errors["target_sample_mismatch"] += 1
            if target and target[1] != semantic_type:
                errors["target_type_mismatch"] += 1

    duplicate_count = sum(value - 1 for value in duplicate_keys.values() if value > 1)
    overlap = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    missing_cells = {
        "%s/%s" % (module, split): args.min_labels_per_module_split - module_split[(module, split)]
        for module in SEMANTIC_TYPES
        for split in VALID_SPLITS
        if module_split[(module, split)] < args.min_labels_per_module_split
    }

    gold_map = defaultdict(set)
    for row in labels:
        gold_map[(str(row.get("query_id") or ""), str(row.get("target_semantic_type") or row.get("semantic_type") or ""))].add(str(row.get("target_id") or ""))
    matched = present = 0
    coverage_by_module = defaultdict(Counter)
    if args.reasoning:
        for row in read_jsonl(args.reasoning):
            query_id = str(row.get("query_id") or "")
            candidates = defaultdict(set)
            for candidate in row.get("candidates", []) or []:
                semantic_type = str(candidate.get("target_semantic_type") or "")
                candidates[semantic_type].add(str(candidate.get("target_id") or ""))
            for semantic_type in SEMANTIC_TYPES:
                targets = gold_map.get((query_id, semantic_type))
                if not targets:
                    continue
                matched += 1
                hit = bool(targets & candidates.get(semantic_type, set()))
                present += int(hit)
                coverage_by_module[semantic_type]["matched"] += 1
                coverage_by_module[semantic_type]["present"] += int(hit)
    candidate_coverage = present / matched if matched else None

    sample_report = None
    if args.usg_index:
        samples = index_samples(args.usg_index)
        sample_report = {
            "unique_samples": len(samples),
            "expected_samples": args.expected_samples or None,
            "complete": not args.expected_samples or len(samples) == args.expected_samples,
        }
    structural_ready = (
        counts["labels"] > 0 and not errors and duplicate_count == 0
        and not overlap and not missing_cells
    )
    data_ready = structural_ready
    if sample_report is not None:
        data_ready = data_ready and bool(sample_report["complete"])
    if args.reasoning and candidate_coverage is not None:
        data_ready = data_ready and candidate_coverage >= args.min_candidate_coverage

    report = {
        "schema": "task2_gold_readiness_v6_6",
        "code_path_complete": True,
        "dataset_ready": data_ready,
        "structural_gold_ready": structural_ready,
        "labels": counts["labels"],
        "labels_by_module_split": {
            "%s/%s" % key: value for key, value in sorted(module_split.items())
        },
        "missing_module_split_cells": missing_cells,
        "duplicate_labels": duplicate_count,
        "group_split_overlap_count": len(overlap),
        "group_split_overlap_examples": dict(list(overlap.items())[:20]),
        "errors": dict(errors),
        "candidate_coverage": {
            "available": args.reasoning is not None,
            "matched_gold_groups": matched,
            "gold_target_present": present,
            "coverage": candidate_coverage,
            "minimum_required": args.min_candidate_coverage,
            "by_module": {
                module: {
                    "matched": values["matched"],
                    "present": values["present"],
                    "coverage": values["present"] / values["matched"] if values["matched"] else None,
                }
                for module, values in coverage_by_module.items()
            },
        },
        "sample_index": sample_report,
        "split_contract": {
            "train": "parameter learning only",
            "validation": "threshold/hyperparameter selection only",
            "test": "frozen final evaluation exactly once",
        },
        "note": (
            "code_path_complete and dataset_ready are deliberately separate; "
            "missing recovered samples cannot be repaired by changing thresholds."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not data_ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

