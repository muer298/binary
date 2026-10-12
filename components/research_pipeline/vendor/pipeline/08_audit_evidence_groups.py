#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit real neighborhood JSONL before multi-positive training.

This command is intentionally read-only.  It shows which compatibility paths
were used and whether the data actually contains multi-evidence targets.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from evidence_objective_v2 import make_target_group_stats, resolve_evidence_record


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit("invalid JSON at line %d: %s" % (line_number, exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neighborhoods", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument(
        "--max-row-fallback-rate",
        type=float,
        default=1.0,
        help="fail if unresolved target groups exceed this rate; 1.0 only reports",
    )
    parser.add_argument("--require-multi-positive-targets", action="store_true")
    args = parser.parse_args()

    evidence_types = Counter()
    target_types = Counter()
    group_sources = Counter()
    compatibility_paths = Counter()
    group_members = defaultdict(list)
    fallback_examples = []
    records = 0

    for line_number, raw in read_jsonl(args.neighborhoods):
        if args.max_records and records >= args.max_records:
            break
        resolved = resolve_evidence_record(raw, row_index=records)
        records += 1
        evidence_types[resolved["evidence_type"]] += 1
        target_types[resolved["target_type"]] += 1
        group_sources[resolved["target_group_source"]] += 1
        group_members[resolved["target_group_id"]].append({
            "line": line_number,
            "anchor_id": resolved["anchor_id"],
            "evidence_type": resolved["evidence_type"],
            "strength": resolved["strength"],
        })
        for field, path in resolved["compatibility_paths"].items():
            compatibility_paths["%s:%s" % (field, path or "missing")] += 1
        if (
            resolved["target_group_source"] == "row_unique_fallback"
            and len(fallback_examples) < args.examples
        ):
            fallback_examples.append({
                "line": line_number,
                "anchor_id": resolved["anchor_id"],
                "source_node": resolved["source_node"],
                "target_group_id": resolved["target_group_id"],
                "compatibility_paths": resolved["compatibility_paths"],
            })

    group_stats = make_target_group_stats([
        target for target, members in group_members.items() for _ in members
    ])
    fallback_count = group_sources.get("row_unique_fallback", 0)
    fallback_rate = fallback_count / max(1, records)
    largest_groups = sorted(
        (
            {
                "target_group_id": target,
                "evidence_count": len(members),
                "evidence_types": dict(Counter(row["evidence_type"] for row in members)),
                "members": members[:args.examples],
            }
            for target, members in group_members.items()
        ),
        key=lambda row: (-row["evidence_count"], row["target_group_id"]),
    )[:args.examples]

    checks = {
        "has_records": records > 0,
        "row_fallback_rate_within_limit": fallback_rate <= args.max_row_fallback_rate,
        "has_multi_positive_targets": group_stats["multi_evidence_targets"] > 0,
    }
    ready = checks["has_records"] and checks["row_fallback_rate_within_limit"]
    if args.require_multi_positive_targets:
        ready = ready and checks["has_multi_positive_targets"]

    report = {
        "schema": "task2_evidence_group_audit_v2",
        "objective_revision": "multi_positive_target_attention_v2",
        "input": str(Path(args.neighborhoods).resolve()),
        "records": records,
        "evidence_type_counts": dict(evidence_types),
        "target_type_counts": dict(target_types),
        "target_group_source_counts": dict(group_sources),
        "target_group_stats": group_stats,
        "row_unique_fallback_rate": fallback_rate,
        "schema_compatibility_paths": dict(compatibility_paths),
        "largest_target_groups": largest_groups,
        "fallback_examples": fallback_examples,
        "checks": checks,
        "ready": ready,
        "interpretation": {
            "row_unique_fallback": "target identity unavailable; records stay independent to avoid false positives",
            "single_high_node": "target identity recovered conservatively from exactly one high-side node",
            "target_node": "explicit target identity supplied by the input record",
        },
    }

    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("AUDIT RESULT:", "PASSED" if ready else "FAILED")
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
