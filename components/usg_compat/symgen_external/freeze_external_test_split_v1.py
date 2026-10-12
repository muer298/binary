#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 sanitized inference manifest 与 USG 的交集全部冻结为 external Test。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--usg-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    report_path = Path(args.report)
    for path in (output, report_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.pair_manifest, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        forbidden = {
            field for field in fields
            if field in {"full_path", "full_sha256", "source_relative_path"}
            or field.startswith("gold_")
        }
        if forbidden:
            raise SystemExit(
                "ERROR: external Test split must use sanitized inference manifest; "
                "forbidden fields=%s" % sorted(forbidden)
            )
        pairs = {
            str(row.get("sample_id") or ""): row
            for row in reader
            if row.get("status") == "ok"
        }
    indexed = {}
    with open(args.usg_index, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                indexed[str(row.get("sample_id") or "")] = row
    common = sorted(set(pairs) & set(indexed))
    missing = sorted(set(pairs) - set(indexed))
    unexpected = sorted(set(indexed) - set(pairs))
    if not common or missing or unexpected:
        raise SystemExit(
            "ERROR: pair/index mismatch common=%d missing=%d unexpected=%d"
            % (len(common), len(missing), len(unexpected))
        )

    rows = []
    for sample_id in common:
        pair = pairs[sample_id]
        rows.append({
            "sample_id": sample_id,
            "group_id": "symgen::" + str(pair.get("project") or "unknown_project"),
            "group_source": "frozen_pair_manifest.project",
            "project": str(pair.get("project") or "unknown_project"),
            "optimization": str(pair.get("optimization") or "unknown"),
            "architecture": str(pair.get("architecture") or "x86_64"),
            "split": "test",
            "split_seed": "external_dataset_all_test_no_selection",
            "dataset": "SymGen",
        })
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema": "symgen_external_test_split_v1",
        "samples": len(rows),
        "split_counts": {"test": len(rows)},
        "train_samples": 0,
        "validation_samples": 0,
        "gold_used_for_assignment": False,
        "metric_used_for_assignment": False,
        "selection_policy": "all successfully paired and indexed samples",
        "passed": bool(rows),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("SYMGEN EXTERNAL TEST SPLIT: FROZEN")


if __name__ == "__main__":
    main()
