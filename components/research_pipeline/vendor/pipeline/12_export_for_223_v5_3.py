#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a backward-compatible reasoning scope for the existing 2.2.3 code."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from module_common_v5_3 import (
    CONFIRMED_STATUSES,
    ensure_output_path,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = ensure_output_path(args.out, overwrite=args.overwrite)
    counts: Counter = Counter()
    records = 0
    with output.open("w", encoding="utf-8") as writer:
        for row in read_jsonl(args.input):
            result = dict(row)
            status_v5_3 = str(row.get("status") or "hypothesis")
            if status_v5_3 in CONFIRMED_STATUSES:
                legacy_status = "confirmed"
            else:
                legacy_status = status_v5_3
            result.update(
                {
                    "schema": "task2_222_for_223_v5_3",
                    "status_v5_3": status_v5_3,
                    "status": legacy_status,
                    "status_locked_v5_3": (
                        bool(row.get("locked"))
                        or status_v5_3 == "rejected"
                    ),
                    "confirmation_level_v5_3": row.get(
                        "confirmation_level"
                    ),
                    "propagation_weight_v5_3": row.get(
                        "propagation_weight"
                    ),
                }
            )
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            counts[status_v5_3] += 1
            records += 1

    print(
        json.dumps(
            {
                "done": True,
                "schema": "task2_222_export_for_223_summary_v5_3",
                "input": args.input,
                "output": str(output),
                "records": records,
                "source_status_counts": dict(counts),
                "compatibility_rule": (
                    "confirmed_hard/confirmed_soft -> confirmed; "
                    "all v5.3 fields retained"
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
