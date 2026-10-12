#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed split guard for Gold consumed by training or evaluation.

The guard deliberately rejects a missing ``split`` field.  A file that does
not carry an auditable split cannot silently enter either parameter learning
or the one-shot Test evaluation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


VALID_SPLITS = ("train", "validation", "test")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Non-object Gold row at %s:%d" % (path, line_number))
            yield value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", action="append", required=True)
    parser.add_argument("--expected-split", choices=VALID_SPLITS, required=True)
    parser.add_argument("--purpose", choices=("train", "calibration", "final_test"), required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.purpose == "train" and args.expected_split != "train":
        raise SystemExit("ERROR: training may consume Train Gold only")
    if args.purpose == "calibration" and args.expected_split != "validation":
        raise SystemExit("ERROR: calibration may consume Validation Gold only")
    if args.purpose == "final_test" and args.expected_split != "test":
        raise SystemExit("ERROR: final evaluation may consume Test Gold only")

    counts = Counter()
    errors = []
    seen = set()
    for path in args.gold:
        if not Path(path).is_file():
            errors.append("missing file: %s" % path)
            continue
        for row_number, row in enumerate(read_jsonl(path), 1):
            counts["rows"] += 1
            split = str(row.get("split") or "").strip().lower()
            counts["split/%s" % (split or "missing")] += 1
            if split != args.expected_split:
                errors.append(
                    "%s:%d expected split=%s, observed=%s"
                    % (path, row_number, args.expected_split, split or "<missing>")
                )
            key = (
                str(row.get("query_id") or ""),
                str(row.get("target_id") or ""),
                str(row.get("target_semantic_type") or row.get("semantic_type") or ""),
            )
            if not all(key):
                errors.append("%s:%d missing query/target/semantic type" % (path, row_number))
            if key in seen:
                errors.append("%s:%d duplicate Gold triple %r" % (path, row_number, key))
            seen.add(key)

    if not counts["rows"]:
        errors.append("Gold input contains no records")
    report = {
        "schema": "task2_gold_split_contract_v6_6",
        "passed": not errors,
        "purpose": args.purpose,
        "expected_split": args.expected_split,
        "files": list(args.gold),
        "records": counts["rows"],
        "split_counts": {
            key.split("/", 1)[1]: value
            for key, value in sorted(counts.items()) if key.startswith("split/")
        },
        "errors": errors[:50],
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
