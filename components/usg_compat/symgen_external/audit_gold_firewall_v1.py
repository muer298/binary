#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计 Ghidra 是否只分析 stripped ELF，并检查 Gold/推理目录隔离。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def canonical(path: str) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def overlaps(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--inference-manifest", required=True)
    parser.add_argument("--semantic-json-root", required=True)
    parser.add_argument("--gold-root", required=True)
    parser.add_argument("--inference-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = Path(args.report)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.pair_manifest, encoding="utf-8", newline="") as stream:
        pairs = {str(row["sample_id"]): row for row in csv.DictReader(stream)}
    with open(args.inference_manifest, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        inference_fields = tuple(reader.fieldnames or ())
        inference = {str(row["sample_id"]): row for row in reader}
    semantic_root = Path(args.semantic_json_root).resolve()
    errors = []
    forbidden_fields = sorted(
        field for field in inference_fields
        if field in {"full_path", "full_sha256", "source_relative_path"}
        or field.startswith("gold_")
    )
    if forbidden_fields:
        errors.append({
            "error": "inference_manifest_contains_gold_fields",
            "fields": forbidden_fields,
        })
    if set(inference) != set(pairs):
        errors.append({
            "error": "pair_and_inference_manifest_sample_mismatch",
            "missing_in_inference": sorted(set(pairs) - set(inference))[:20],
            "unexpected_in_inference": sorted(set(inference) - set(pairs))[:20],
        })
    checked = 0
    for sample_id, pair in pairs.items():
        meta_path = semantic_root / (sample_id + ".meta.json")
        if not meta_path.is_file():
            errors.append({"sample_id": sample_id, "error": "missing_meta"})
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        analyzed = canonical(str(meta.get("binary") or ""))
        stripped = canonical(str(pair.get("stripped_path") or ""))
        full = canonical(str(pair.get("full_path") or ""))
        inference_row = inference.get(sample_id, {})
        recorded_row = meta.get("manifest_row") or {}
        if analyzed != stripped:
            errors.append({"sample_id": sample_id, "error": "not_stripped_input"})
        if analyzed == full:
            errors.append({"sample_id": sample_id, "error": "full_used_for_ghidra"})
        if canonical(str(inference_row.get("stripped_path") or "")) != stripped:
            errors.append({
                "sample_id": sample_id,
                "error": "sanitized_manifest_stripped_path_mismatch",
            })
        leaked_fields = sorted(
            key for key in recorded_row
            if key in {"full_path", "full_sha256", "source_relative_path"}
            or str(key).startswith("gold_")
        )
        if leaked_fields:
            errors.append({
                "sample_id": sample_id,
                "error": "ghidra_meta_contains_gold_fields",
                "fields": leaked_fields,
            })
        checked += 1
    gold_root = Path(args.gold_root).resolve()
    inference_root = Path(args.inference_root).resolve()
    if overlaps(gold_root, inference_root):
        errors.append({"error": "gold_and_inference_roots_overlap"})
    report = {
        "schema": "symgen_gold_firewall_audit_v1",
        "pair_rows": len(pairs),
        "inference_manifest_rows": len(inference),
        "inference_manifest_fields": list(inference_fields),
        "inference_manifest_contains_gold_fields": bool(forbidden_fields),
        "ghidra_meta_checked": checked,
        "gold_root": str(gold_root),
        "inference_root": str(inference_root),
        "errors": errors[:100],
        "error_count": len(errors),
        "passed": bool(pairs) and checked == len(pairs) and not errors,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
    print("SYMGEN GOLD/INFERENCE FIREWALL: PASSED")


if __name__ == "__main__":
    main()
