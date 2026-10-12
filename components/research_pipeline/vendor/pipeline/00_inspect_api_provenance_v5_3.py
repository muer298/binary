#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect whether complete USG APISem nodes retain recognition provenance."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Dict, List

from module_common_v5_3 import (
    HARD_API_SOURCES,
    api_source_evidence_from_objects,
    node_id,
    normalize_semantic_type,
    read_jsonl,
    unwrap_node_record,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    api_nodes = 0
    source_counts: Counter = Counter()
    path_counts: Counter = Counter()
    hard_examples: List[Dict[str, Any]] = []
    unknown_examples: List[Dict[str, Any]] = []

    for raw in read_jsonl(args.nodes):
        row = unwrap_node_record(raw)
        semantic_type = normalize_semantic_type(
            row.get("node_type") or row.get("semantic_type") or ""
        )
        if semantic_type != "APISem":
            continue
        api_nodes += 1
        evidence = api_source_evidence_from_objects(
            row,
            object_labels=("usg_node",),
        )
        source = str(evidence.get("source") or "unknown")
        source_counts[source] += 1
        if source != "unknown":
            path_counts[
                "%s:%s"
                % (source, str(evidence.get("path") or "<root>"))
            ] += 1
        example = {
            "node_id": node_id(row),
            "source": source,
            "origin": evidence.get("origin"),
            "path": evidence.get("path"),
            "raw_value": evidence.get("raw_value"),
            "top_level_keys": sorted(map(str, row.keys())),
        }
        if (
            source in HARD_API_SOURCES
            and len(hard_examples) < max(0, args.examples)
        ):
            hard_examples.append(example)
        elif (
            source == "unknown"
            and len(unknown_examples) < max(0, args.examples)
        ):
            unknown_examples.append(example)

    hard_count = sum(source_counts[source] for source in HARD_API_SOURCES)
    report = {
        "schema": "task2_222_api_provenance_inspection_v5_3_1",
        "nodes": args.nodes,
        "apisem_nodes": api_nodes,
        "recognized_source_counts": dict(source_counts),
        "recognized_source_paths": dict(path_counts),
        "hard_source_nodes": hard_count,
        "hard_source_coverage": hard_count / max(api_nodes, 1),
        "hard_examples": hard_examples,
        "unknown_examples": unknown_examples,
        "note": (
            "A hard-source node becomes a confirmed_hard seed only when a "
            "direct query-to-APISem evidence edge uniquely selects it."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
