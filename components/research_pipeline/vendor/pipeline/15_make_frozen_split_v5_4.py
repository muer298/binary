#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a dataset-independent, group-aware frozen Train/Validation/Test split.

Splitting happens before gold conversion and never looks at labels.  Multiple
compiler/optimization variants should share a project/program group so they
cannot leak across Train and Test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def normalize_program_name(value: Any) -> str:
    text = Path(str(value or "")).name.lower()
    text = re.sub(r"\.(stripped|unstripped|full|debug)$", "", text)
    text = re.sub(r"(?:^|[-_.])(o[0-3s]|x86_64|x86|arm64|aarch64|arm)(?:[-_.]|$)", "_", text)
    text = re.sub(r"[-_.]+", "_", text).strip("_")
    return text


def group_for(row: Mapping[str, Any], explicit_field: str) -> tuple[str, str]:
    if explicit_field:
        value = str(row.get(explicit_field) or "").strip()
        if value:
            return value, "explicit:%s" % explicit_field
    for key in (
        "project_id", "project", "repo", "repository", "package",
        "source_project", "program_name",
    ):
        value = str(row.get(key) or "").strip()
        if value:
            normalized = normalize_program_name(value)
            if normalized:
                return normalized, key
    full_path = str(row.get("full_path") or "")
    if full_path:
        normalized = normalize_program_name(full_path)
        if normalized:
            return normalized, "full_path_basename"
    sample_id = str(row.get("sample_id") or "")
    return sample_id, "sample_id_fallback"


def assignment(group_id: str, seed: str, train: float, validation: float) -> str:
    digest = hashlib.sha256((seed + "\0" + group_id).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2 ** 64)
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usg-index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--group-field", default="")
    parser.add_argument("--seed", default="research2-frozen-split-v1")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-formal",
        action="store_true",
        help="Fail when any sample falls back to sample_id grouping.",
    )
    args = parser.parse_args()
    if abs(args.train_ratio + args.validation_ratio + args.test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1")

    out = Path(args.out)
    report_path = Path(args.report)
    for path in (out, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError("Refusing to overwrite: %s" % path)
        path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(args.usg_index))
    group_splits: Dict[str, str] = {}
    group_sources: Counter = Counter()
    group_members: Dict[str, list[str]] = defaultdict(list)
    output = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        group_id, source = group_for(row, args.group_field)
        split = group_splits.setdefault(
            group_id,
            assignment(
                group_id, args.seed, args.train_ratio, args.validation_ratio
            ),
        )
        group_sources[source] += 1
        group_members[group_id].append(sample_id)
        output.append({
            "sample_id": sample_id,
            "group_id": group_id,
            "group_source": source,
            "split": split,
            "split_seed": args.seed,
        })

    with out.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    sample_counts = Counter(row["split"] for row in output)
    group_counts = Counter(group_splits.values())
    fallback_samples = sum(
        1 for row in output if row["group_source"] == "sample_id_fallback"
    )
    leakage = {
        group_id: sorted({
            row["split"] for row in output if row["group_id"] == group_id
        })
        for group_id in group_members
    }
    leakage = {key: value for key, value in leakage.items() if len(value) > 1}
    report = {
        "schema": "research2_frozen_split_v1",
        "usg_index": args.usg_index,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "samples": len(output),
        "groups": len(group_splits),
        "sample_counts": dict(sample_counts),
        "group_counts": dict(group_counts),
        "group_source_counts": dict(group_sources),
        "sample_id_fallback_count": fallback_samples,
        "group_overlap_count": len(leakage),
        "group_overlap_examples": dict(list(leakage.items())[:20]),
        "formal_split_ready": len(leakage) == 0 and fallback_samples == 0,
        "warning": (
            "Some samples use sample_id as their own group. Provide project/program "
            "metadata or --group-field before a formal experiment."
            if fallback_samples else None
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_formal and not report["formal_split_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
