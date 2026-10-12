#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train/Validation/Test isolation helpers for Stage3 alignment.

USG files used by this project can follow several compatible
schemas.  This module therefore adapts field aliases at the boundary instead
of requiring the source JSONL to be rewritten.  Its two invariants are:

1. every sample has exactly one frozen split assignment;
2. the Test evaluation token can be claimed exactly once.

No PyTorch dependency is used here so the split firewall can be audited and
unit-tested before a GPU training job starts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SAMPLE_ID_ALIASES = (
    "sample_id",
    "sample_hash",
    "binary_id",
    "binary_hash",
    "sha256",
)
SPLIT_ALIASES = ("split", "subset", "partition", "dataset_split", "set")
SOURCE_ALIASES = (
    "source_node",
    "source_ref",
    "from_node",
    "src_node",
    "source_id",
    "src",
    "source",
)
TARGET_ALIASES = (
    "target_node",
    "target_ref",
    "to_node",
    "dst_node",
    "target_id",
    "dst",
    "target",
)
NESTED_MAPPING_KEYS = (
    "attrs",
    "attributes",
    "metadata",
    "meta",
    "context",
    "payload",
    "edge",
    "record",
)


def _scalar(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("node_id", "id", "ref", "value", "name"):
            candidate = value.get(key)
            if candidate is not None and not isinstance(candidate, (Mapping, list, tuple)):
                text = str(candidate).strip()
                if text:
                    return text
        return ""
    if isinstance(value, (list, tuple)):
        return ""
    return str(value or "").strip()


def _mapping_layers(record: Mapping[str, Any]) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    """Yield bounded nested mappings in deterministic precedence order."""

    yielded = set()

    def visit(path: str, value: Mapping[str, Any], depth: int) -> Iterable[Tuple[str, Mapping[str, Any]]]:
        marker = id(value)
        if marker in yielded:
            return
        yielded.add(marker)
        yield path, value
        if depth >= 3:
            return
        for key in NESTED_MAPPING_KEYS:
            nested = value.get(key)
            if isinstance(nested, Mapping):
                for item in visit("%s.%s" % (path, key), nested, depth + 1):
                    yield item

    for item in visit("record", record, 0):
        yield item


def find_alias(record: Mapping[str, Any], aliases: Sequence[str]) -> Tuple[str, str]:
    for layer_path, layer in _mapping_layers(record):
        for key in aliases:
            if key not in layer:
                continue
            value = _scalar(layer.get(key))
            if value:
                return value, "%s.%s" % (layer_path, key)
    return "", ""


def infer_sample_from_node_id(node_id: Any) -> str:
    text = _scalar(node_id)
    if "::" in text:
        return text.split("::", 1)[0].strip()
    return ""


def normalize_split_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "train": "train",
        "training": "train",
        "tr": "train",
        "validation": "validation",
        "valid": "validation",
        "val": "validation",
        "dev": "validation",
        "development": "validation",
        "test": "test",
        "testing": "test",
        "te": "test",
    }
    return aliases.get(text, "")


def resolve_node_sample_id(
    node_id: Any,
    node_info: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[str, str]:
    nid = _scalar(node_id)
    info = (node_info or {}).get(nid, {})
    if isinstance(info, Mapping):
        value, path = find_alias(info, SAMPLE_ID_ALIASES)
        if value:
            return value, "node_info.%s" % path
    inferred = infer_sample_from_node_id(nid)
    if inferred:
        return inferred, "node_id_prefix"
    return "", "unresolved"


def resolve_edge_endpoints(edge: Mapping[str, Any]) -> Dict[str, str]:
    source, source_path = find_alias(edge, SOURCE_ALIASES)
    target, target_path = find_alias(edge, TARGET_ALIASES)
    return {
        "source_node": source,
        "target_node": target,
        "source_path": source_path,
        "target_path": target_path,
    }


def resolve_edge_sample(
    edge: Mapping[str, Any],
    node_info: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    endpoints = resolve_edge_endpoints(edge)
    source_sample, source_sample_path = resolve_node_sample_id(
        endpoints["source_node"], node_info
    )
    target_sample, target_sample_path = resolve_node_sample_id(
        endpoints["target_node"], node_info
    )
    explicit_sample, explicit_path = find_alias(edge, SAMPLE_ID_ALIASES)

    conflicts = []
    if source_sample and target_sample and source_sample != target_sample:
        conflicts.append("cross_sample_endpoints")
    endpoint_sample = source_sample or target_sample
    if explicit_sample and endpoint_sample and explicit_sample != endpoint_sample:
        conflicts.append("explicit_endpoint_sample_mismatch")

    sample_id = explicit_sample or endpoint_sample
    if explicit_sample:
        sample_source = explicit_path
    elif source_sample:
        sample_source = "source:%s" % source_sample_path
    elif target_sample:
        sample_source = "target:%s" % target_sample_path
    else:
        sample_source = "unresolved"

    return {
        **endpoints,
        "sample_id": sample_id,
        "sample_source": sample_source,
        "source_sample_id": source_sample,
        "target_sample_id": target_sample,
        "conflicts": conflicts,
        "valid": bool(sample_id) and not conflicts,
    }


def _iter_manifest_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter=delimiter):
                yield row
        return

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, Mapping):
                    yield row
            return
        if isinstance(payload, Mapping):
            for key in ("records", "rows", "samples", "assignments", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    for row in value:
                        if isinstance(row, Mapping):
                            yield row
                    return
            # Also accept a direct {sample_id: split} mapping.
            if all(not isinstance(value, (Mapping, list, tuple)) for value in payload.values()):
                for sample_id, split in payload.items():
                    yield {"sample_id": sample_id, "split": split}
                return
        raise ValueError("Unsupported split JSON schema: %s" % path)

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("Split JSONL row %d is not an object" % line_number)
            yield row


def load_split_manifest(path: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("Split manifest does not exist: %s" % source)

    assignments: Dict[str, str] = {}
    split_counts: Counter = Counter()
    sample_field_paths: Counter = Counter()
    split_field_paths: Counter = Counter()
    group_splits: Dict[str, set] = defaultdict(set)
    rows = 0
    duplicate_rows = 0

    for row in _iter_manifest_rows(source):
        rows += 1
        sample_id, sample_path = find_alias(row, SAMPLE_ID_ALIASES)
        split_raw, split_path = find_alias(row, SPLIT_ALIASES)
        split = normalize_split_name(split_raw)
        if not sample_id or not split:
            raise ValueError(
                "Split manifest row %d lacks a recognized sample_id/split: %r"
                % (rows, dict(row))
            )
        if sample_id in assignments:
            duplicate_rows += 1
            if assignments[sample_id] != split:
                raise ValueError(
                    "Conflicting split assignment for %s: %s versus %s"
                    % (sample_id, assignments[sample_id], split)
                )
        else:
            assignments[sample_id] = split
            split_counts[split] += 1
        sample_field_paths[sample_path] += 1
        split_field_paths[split_path] += 1
        group_id = _scalar(row.get("group_id") or row.get("program_group") or row.get("project_group"))
        if group_id:
            group_splits[group_id].add(split)

    group_overlap = {
        group_id: sorted(values)
        for group_id, values in group_splits.items()
        if len(values) > 1
    }
    if group_overlap:
        example = next(iter(group_overlap.items()))
        raise ValueError("Group split leakage in manifest: %r" % (example,))
    if not assignments:
        raise ValueError("Split manifest is empty: %s" % source)

    audit = {
        "path": str(source.resolve()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": rows,
        "unique_samples": len(assignments),
        "duplicate_rows_same_assignment": duplicate_rows,
        "split_sample_counts": dict(split_counts),
        "sample_field_paths": dict(sample_field_paths),
        "split_field_paths": dict(split_field_paths),
        "group_count": len(group_splits),
        "group_overlap_count": 0,
    }
    return assignments, audit


def _assert_pair_sample_ids(pairs: Sequence[Mapping[str, Any]]) -> None:
    missing = [index for index, pair in enumerate(pairs) if not _scalar(pair.get("sample_id"))]
    if missing:
        raise ValueError(
            "%d evidence pairs have no sample_id; first rows: %s"
            % (len(missing), missing[:10])
        )


def partition_pairs_by_sample(
    pairs: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    split_manifest: str = "",
    missing_manifest_policy: str = "error",
    require_nonempty: bool = True,
) -> Tuple[Dict[str, list], Dict[str, str], Dict[str, Any]]:
    """Partition pair indexes by sample without looking at labels."""

    _assert_pair_sample_ids(pairs)
    if missing_manifest_policy not in ("error", "drop"):
        raise ValueError("missing_manifest_policy must be error or drop")
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("train/validation ratios must leave a positive Test ratio")

    by_sample: Dict[str, list] = defaultdict(list)
    for index, pair in enumerate(pairs):
        by_sample[_scalar(pair.get("sample_id"))].append(index)

    manifest_audit: Optional[Dict[str, Any]] = None
    dropped_samples = []
    if split_manifest:
        assignments, manifest_audit = load_split_manifest(split_manifest)
        missing = sorted(set(by_sample) - set(assignments))
        if missing and missing_manifest_policy == "error":
            raise ValueError(
                "%d pair samples are absent from the frozen split manifest; examples: %s"
                % (len(missing), missing[:10])
            )
        dropped_samples = missing
    else:
        samples = sorted(by_sample)
        random.Random(seed).shuffle(samples)
        train_end = int(len(samples) * train_ratio)
        validation_end = int(len(samples) * (train_ratio + validation_ratio))
        assignments = {}
        for position, sample_id in enumerate(samples):
            if position < train_end:
                assignments[sample_id] = "train"
            elif position < validation_end:
                assignments[sample_id] = "validation"
            else:
                assignments[sample_id] = "test"

    partitions = {"train": [], "validation": [], "test": []}
    pair_sample_counts: Counter = Counter()
    for sample_id, indexes in by_sample.items():
        split = assignments.get(sample_id)
        if not split:
            continue
        partitions[split].extend(indexes)
        pair_sample_counts[split] += 1

    if require_nonempty:
        empty = [split for split, indexes in partitions.items() if not indexes]
        if empty:
            raise ValueError(
                "Empty pair split(s) %s. Use more samples or a complete split manifest."
                % ", ".join(empty)
            )

    sample_sets = {
        split: {sample_id for sample_id, value in assignments.items() if value == split}
        for split in partitions
    }
    overlap = (
        (sample_sets["train"] & sample_sets["validation"])
        | (sample_sets["train"] & sample_sets["test"])
        | (sample_sets["validation"] & sample_sets["test"])
    )
    if overlap:
        raise AssertionError("Sample split overlap: %s" % sorted(overlap)[:10])

    target_splits: Dict[str, set] = defaultdict(set)
    for split, indexes in partitions.items():
        for index in indexes:
            group_id = _scalar(pairs[index].get("target_group_id"))
            if group_id:
                target_splits[group_id].add(split)
    target_overlap = {
        group_id: sorted(values)
        for group_id, values in target_splits.items()
        if len(values) > 1
    }
    if target_overlap:
        example = next(iter(target_overlap.items()))
        raise ValueError("Target group crosses data splits: %r" % (example,))

    audit = {
        "split_mode": "sample",
        "split_source": "frozen_manifest" if split_manifest else "seeded_sample_shuffle",
        "seed": seed,
        "pair_counts": {key: len(value) for key, value in partitions.items()},
        "pair_sample_counts": dict(pair_sample_counts),
        "assignment_sample_counts": {
            key: len(value) for key, value in sample_sets.items()
        },
        "sample_overlap_count": 0,
        "target_group_overlap_count": 0,
        "missing_manifest_sample_count": len(dropped_samples),
        "missing_manifest_sample_examples": dropped_samples[:20],
        "manifest": manifest_audit,
    }
    return partitions, assignments, audit


class SingleUseEvaluationGuard:
    """Fail closed if a protected evaluation split is requested twice."""

    def __init__(self, split_name: str = "test") -> None:
        self.split_name = split_name
        self.count = 0

    def claim(self) -> None:
        if self.count:
            raise RuntimeError(
                "%s evaluation was already executed; Test is single-use"
                % self.split_name
            )
        self.count += 1

    def audit(self) -> Dict[str, Any]:
        return {
            "protected_split": self.split_name,
            "evaluation_count": self.count,
            "single_use_enforced": True,
        }
