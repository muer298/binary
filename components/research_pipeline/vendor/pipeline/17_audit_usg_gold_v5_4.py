#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit strict, independently produced USG Gold before any model use."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Set, Tuple


SEMANTIC_TYPES = ("APISem", "TypeSem", "LayoutSem", "InterfaceSem")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-labels-per-module", type=int, default=1)
    parser.add_argument("--require-all-splits", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: %s" % out)
    out.parent.mkdir(parents=True, exist_ok=True)

    nodes: Dict[str, Tuple[str, str]] = {}
    for row in read_jsonl(args.nodes):
        node_id = str(row.get("node_id") or "")
        if node_id:
            nodes[node_id] = (
                str(row.get("sample_id") or ""),
                str(row.get("node_type") or ""),
            )

    records = list(read_jsonl(args.gold))
    counts = Counter()
    modules = Counter()
    splits = Counter()
    strengths = Counter()
    duplicates = Counter()
    group_splits: Dict[str, Set[str]] = defaultdict(set)
    errors = Counter()
    examples = []
    for row in records:
        query_id = str(row.get("query_id") or "")
        target_id = str(row.get("target_id") or "")
        semantic_type = str(row.get("target_semantic_type") or "")
        sample_id = str(row.get("sample_id") or "")
        split = str(row.get("split") or "")
        group_id = str(row.get("group_id") or "")
        key = (query_id, target_id, semantic_type)
        duplicates[key] += 1
        modules[semantic_type] += 1
        splits[split] += 1
        strengths[str(row.get("gold_strength") or "missing")] += 1
        group_splits[group_id].add(split)
        for name, node_id, expected_type in (
            ("query", query_id, None),
            ("target", target_id, semantic_type),
        ):
            metadata = nodes.get(node_id)
            if metadata is None:
                errors["%s_node_missing" % name] += 1
                continue
            node_sample, node_type = metadata
            if node_sample != sample_id:
                errors["%s_sample_mismatch" % name] += 1
            if expected_type and node_type != expected_type:
                errors["target_semantic_type_mismatch"] += 1
        if semantic_type not in SEMANTIC_TYPES:
            errors["unknown_semantic_type"] += 1
        if split not in ("train", "validation", "test"):
            errors["invalid_split"] += 1
        if str(row.get("gold_strength") or "") != "strict_independent":
            errors["non_strict_gold"] += 1
        counts["labels"] += 1

    duplicate_labels = sum(value - 1 for value in duplicates.values() if value > 1)
    overlaps = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    missing_modules = {
        module: max(0, args.min_labels_per_module - modules[module])
        for module in SEMANTIC_TYPES
        if modules[module] < args.min_labels_per_module
    }
    missing_splits = sorted(
        set(("train", "validation", "test")) - set(splits)
    )
    ready = (
        counts["labels"] > 0
        and not errors
        and duplicate_labels == 0
        and not overlaps
        and not missing_modules
        and (not args.require_all_splits or not missing_splits)
    )
    report = {
        "schema": "research2_usg_gold_validation_v1",
        "gold": args.gold,
        "nodes": args.nodes,
        "labels": counts["labels"],
        "labels_by_module": dict(modules),
        "labels_by_split": dict(splits),
        "gold_strength_counts": dict(strengths),
        "duplicate_label_count": duplicate_labels,
        "group_overlap_count": len(overlaps),
        "group_overlap_examples": dict(list(overlaps.items())[:20]),
        "missing_module_labels": missing_modules,
        "missing_splits": missing_splits,
        "errors": dict(errors),
        "gold_ready": ready,
        "examples": examples,
    }
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
