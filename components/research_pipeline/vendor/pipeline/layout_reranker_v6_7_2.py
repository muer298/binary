#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gold-free, schema-tolerant LayoutSem pairwise reranker for Task 2.2.3.

The v6.6 LayoutSem scorer only counted whether offset-like attributes existed.
This module compares the offsets observed for a query with the offsets declared
by each candidate.  The v6.7.3 evidence bridge also understands the raw
``effects.memory_accesses[].offset`` and ``fields[].offset`` schemas so a
bounded nested list cannot silently hide usable evidence.  Gold labels are
never accepted by this module.  Validation
may select the blend/fusion hyperparameters through the separate calibration
script, and the resulting JSON must then be frozen before Test.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CONFIG_SCHEMA = "task2_layout_reranker_config_v6_7_2"
IMPLEMENTATION_REVISION = "v6_7_3_layout_evidence_bridge_reranker"
EVIDENCE_BRIDGE_REVISION = "v6_7_3_nested_memory_access_bridge"

OBSERVED_OFFSET_KEYS = (
    "observed_field_offsets",
    "field_offsets_observed",
    "eval_field_offsets",
    "observed_offsets",
    "accessed_offsets",
    "memory_offsets",
    "access_offsets",
)
GENERIC_OFFSET_KEYS = (
    "field_offsets",
    "offsets",
    "offset_set",
)
INFERRED_OFFSET_KEYS = (
    "inferred_field_offsets",
    "field_offsets_inferred",
    "inferred_offsets",
)
WIDTH_KEYS = (
    "access_widths",
    "field_widths",
    "widths",
    "access_width",
    "field_width",
    "byte_width",
    "byte_size",
    "size_bytes",
)
ALIGNMENT_KEYS = ("alignment", "align", "byte_alignment")
SIZE_KEYS = ("total_size", "layout_size", "object_size", "byte_size", "size_bytes")
LAYOUT_KIND_KEYS = ("layout_kind", "kind", "structure_kind", "aggregate_kind")

NESTED_KEYS = (
    "attrs",
    "effects",
    "metadata",
    "candidate_metadata",
    "payload",
    "raw",
    "layout",
    "memory_layout",
    "propagated_constraints",
)

MEMORY_ACCESS_CONTAINER_KEYS = (
    "memory_accesses",
    "memory_access_patterns",
)
FIELD_CONTAINER_KEYS = (
    "fields",
    "members",
    "layout_fields",
    "field_layouts",
)
SINGULAR_OFFSET_KEYS = (
    "offset",
    "byte_offset",
    "displacement",
    "disp",
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "schema": CONFIG_SCHEMA,
    "implementation_revision": IMPLEMENTATION_REVISION,
    "enabled": True,
    "selected_on_split": "default_not_selected",
    "test_used_for_selection": False,
    "pairwise_blend": 0.75,
    "recommended_module_weight": 0.45,
    "max_abs_offset": 65536,
    "tolerances": [0, 1, 2, 4],
    "tolerance_weights": [0.55, 0.15, 0.15, 0.15],
    "invalid_penalty_weight": 0.12,
    "positive_weights": {
        "observed_coverage": 0.22,
        "candidate_precision": 0.06,
        "exact_f1": 0.13,
        "tolerant_f1": 0.09,
        "offset_jaccard": 0.08,
        "normalized_offset_f1": 0.08,
        "delta_compatibility": 0.10,
        "stride_compatibility": 0.05,
        "width_compatibility": 0.04,
        "alignment_compatibility": 0.03,
        "structural_compatibility": 0.03,
        "provenance_specificity": 0.04,
        "evidence_quality": 0.05,
    },
}


def clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalise_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config.update({key: value for key, value in raw.items() if key != "positive_weights"})
    if isinstance(raw.get("positive_weights"), Mapping):
        config["positive_weights"].update(raw["positive_weights"])

    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("Unsupported LayoutSem config schema: %r" % config.get("schema"))
    if config.get("test_used_for_selection") is not False:
        raise ValueError("Refusing LayoutSem config selected with Test")
    if str(config.get("selected_on_split") or "").lower() == "test":
        raise ValueError("Refusing LayoutSem config selected on Test")

    for name in ("pairwise_blend", "recommended_module_weight", "invalid_penalty_weight"):
        value = float(config[name])
        if not 0.0 <= value <= 1.0:
            raise ValueError("%s must be in [0,1]" % name)
        config[name] = value

    max_abs_offset = int(config["max_abs_offset"])
    if max_abs_offset <= 0:
        raise ValueError("max_abs_offset must be positive")
    config["max_abs_offset"] = max_abs_offset

    tolerances = [int(value) for value in config.get("tolerances") or []]
    tolerance_weights = [float(value) for value in config.get("tolerance_weights") or []]
    if not tolerances or len(tolerances) != len(tolerance_weights):
        raise ValueError("tolerances and tolerance_weights must be non-empty and aligned")
    if any(value < 0 for value in tolerances) or any(value < 0.0 for value in tolerance_weights):
        raise ValueError("tolerances and tolerance_weights must be non-negative")
    if sum(tolerance_weights) <= 0.0:
        raise ValueError("tolerance_weights must have positive mass")
    config["tolerances"] = tolerances
    config["tolerance_weights"] = tolerance_weights

    positive = config.get("positive_weights")
    if not isinstance(positive, Mapping) or not positive:
        raise ValueError("positive_weights must be a non-empty object")
    positive = {str(key): float(value) for key, value in positive.items()}
    if any(value < 0.0 for value in positive.values()) or sum(positive.values()) <= 0.0:
        raise ValueError("positive_weights must be non-negative with positive mass")
    config["positive_weights"] = positive
    return config


def load_layout_config(path: Optional[str]) -> Tuple[Dict[str, Any], Optional[str], str]:
    if path:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError("LayoutSem reranker config does not exist: %s" % config_path)
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("LayoutSem reranker config must be a JSON object")
        config = _normalise_config(raw)
        resolved = str(config_path.resolve())
    else:
        config = _normalise_config(DEFAULT_CONFIG)
        resolved = None
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return config, resolved, digest


def _maps(record: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    seen = set()
    stack: List[Mapping[str, Any]] = [record]
    while stack:
        current = stack.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        yield current
        for key in NESTED_KEYS:
            child = current.get(key)
            if isinstance(child, Mapping):
                stack.append(child)


def _numbers(value: Any) -> List[float]:
    result: List[float] = []
    if value is None or isinstance(value, bool):
        return result
    if isinstance(value, (int, float)):
        number = _finite_float(value)
        return [] if number is None else [number]
    if isinstance(value, Mapping):
        for key in ("offset", "value", "width", "size", "bytes", "alignment"):
            if key in value:
                result.extend(_numbers(value[key]))
        if result:
            return result
        for item in value.values():
            if isinstance(item, (int, float)):
                result.extend(_numbers(item))
        return result
    if isinstance(value, (list, tuple, set)):
        for item in value:
            result.extend(_numbers(item))
        return result
    text = str(value).strip().lower()
    if not text:
        return result
    for token in re.findall(r"[-+]?0x[0-9a-f]+|[-+]?\d+(?:\.\d+)?", text):
        try:
            result.append(float(int(token, 0)) if "x" in token else float(token))
        except (TypeError, ValueError):
            continue
    return result


def _collect(record: Mapping[str, Any], keys: Sequence[str]) -> Tuple[List[float], List[str]]:
    values: List[float] = []
    sources: List[str] = []
    for mapping in _maps(record):
        for key in keys:
            if key not in mapping:
                continue
            extracted = _numbers(mapping.get(key))
            if extracted:
                values.extend(extracted)
                sources.append(key)
    return values, sorted(set(sources))


def _explicit_false(value: Any) -> bool:
    if value is False or value == 0:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {
            "false", "no", "off", "0", "dynamic", "unknown",
        }
    return False


def _container_mappings(
    record: Mapping[str, Any],
    container_keys: Sequence[str],
) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    """Yield dictionaries stored under known nested list containers."""
    seen = set()
    for mapping in _maps(record):
        for key in container_keys:
            value = mapping.get(key)
            items: Iterable[Any]
            if isinstance(value, Mapping):
                items = (value,)
            elif isinstance(value, (list, tuple)):
                items = value
            else:
                continue
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                identity = id(item)
                if identity in seen:
                    continue
                seen.add(identity)
                yield key, item


def _collect_container_numbers(
    record: Mapping[str, Any],
    container_keys: Sequence[str],
    value_keys: Sequence[str],
    require_fixed: bool = False,
) -> Tuple[List[float], List[str]]:
    values: List[float] = []
    sources: List[str] = []
    for container_key, item in _container_mappings(
        record, container_keys,
    ):
        if require_fixed and _explicit_false(item.get("is_fixed")):
            continue
        for key in value_keys:
            if key not in item:
                continue
            extracted = _numbers(item.get(key))
            if extracted:
                values.extend(extracted)
                sources.append("%s[].%s" % (container_key, key))
                break
    return values, sorted(set(sources))


def _memory_access_offsets(
    record: Mapping[str, Any],
) -> Tuple[List[float], List[str]]:
    return _collect_container_numbers(
        record,
        MEMORY_ACCESS_CONTAINER_KEYS,
        SINGULAR_OFFSET_KEYS,
        require_fixed=True,
    )


def _field_member_offsets(
    record: Mapping[str, Any],
) -> Tuple[List[float], List[str]]:
    return _collect_container_numbers(
        record,
        FIELD_CONTAINER_KEYS,
        SINGULAR_OFFSET_KEYS,
        require_fixed=False,
    )


def _memory_access_widths(
    record: Mapping[str, Any],
) -> Tuple[List[float], List[str]]:
    return _collect_container_numbers(
        record,
        MEMORY_ACCESS_CONTAINER_KEYS,
        WIDTH_KEYS,
        require_fixed=True,
    )


def _field_member_widths(
    record: Mapping[str, Any],
) -> Tuple[List[float], List[str]]:
    return _collect_container_numbers(
        record,
        FIELD_CONTAINER_KEYS,
        WIDTH_KEYS,
        require_fixed=False,
    )


def _edge_text(edge: Mapping[str, Any]) -> str:
    attrs = edge.get("attrs") if isinstance(edge.get("attrs"), Mapping) else {}
    parts = [
        edge.get("edge_kind"), edge.get("edge_type"), edge.get("evidence_type"),
        edge.get("constraint_type"), edge.get("source_path"),
        attrs.get("edge_kind"), attrs.get("edge_type"), attrs.get("evidence_type"),
        attrs.get("constraint_type"), attrs.get("source_path"),
    ]
    return " ".join(str(part or "").lower() for part in parts)


def _edge_offsets(edges: Sequence[Mapping[str, Any]]) -> Tuple[List[float], List[str]]:
    values: List[float] = []
    sources: List[str] = []
    for edge in edges:
        text = _edge_text(edge)
        if not any(token in text for token in (
            "offset", "field", "memory", "deref", "layout", "access", "stride",
        )):
            continue
        attrs = edge.get("attrs") if isinstance(edge.get("attrs"), Mapping) else {}
        for record, prefix in ((edge, "edge"), (attrs, "edge.attrs")):
            extracted, keys = _collect(record, OBSERVED_OFFSET_KEYS + GENERIC_OFFSET_KEYS)
            if extracted:
                values.extend(extracted)
                sources.extend("%s.%s" % (prefix, key) for key in keys)
    return values, sorted(set(sources))


def _normalise_offsets(values: Sequence[float], maximum: int) -> List[int]:
    result = set()
    for value in values:
        number = _finite_float(value)
        if number is None:
            continue
        integer = int(round(number))
        if abs(number - integer) > 1e-6:
            continue
        if 0 <= integer <= maximum:
            result.add(integer)
    return sorted(result)


def _normalise_positive(values: Sequence[float], maximum: int = 65536) -> List[int]:
    result = set()
    for value in values:
        number = _finite_float(value)
        if number is None:
            continue
        integer = int(round(number))
        if abs(number - integer) <= 1e-6 and 0 < integer <= maximum:
            result.add(integer)
    return sorted(result)


def _greedy_match(left: Sequence[int], right: Sequence[int], tolerance: int) -> int:
    pairs: List[Tuple[int, int, int]] = []
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            distance = abs(left_value - right_value)
            if distance <= tolerance:
                pairs.append((distance, left_index, right_index))
    pairs.sort()
    used_left = set()
    used_right = set()
    matched = 0
    for _, left_index, right_index in pairs:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matched += 1
    return matched


def _metrics(left: Sequence[int], right: Sequence[int], tolerance: int = 0) -> Dict[str, float]:
    if not left or not right:
        return {"matches": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0, "jaccard": 0.0}
    matched = _greedy_match(left, right, tolerance)
    recall = matched / len(left)
    precision = matched / len(right)
    f1 = 0.0 if recall + precision <= 0.0 else 2.0 * recall * precision / (recall + precision)
    union = len(left) + len(right) - matched
    return {
        "matches": float(matched),
        "recall": clamp01(recall),
        "precision": clamp01(precision),
        "f1": clamp01(f1),
        "jaccard": clamp01(matched / union if union else 1.0),
    }


def _normalised(values: Sequence[int]) -> List[int]:
    if not values:
        return []
    base = min(values)
    return [value - base for value in values]


def _adjacent_deltas(values: Sequence[int]) -> List[int]:
    ordered = sorted(set(values))
    return [right - left for left, right in zip(ordered, ordered[1:]) if right > left]


def _stride(values: Sequence[int]) -> int:
    deltas = _adjacent_deltas(values)
    return reduce(gcd, deltas) if deltas else 0


def _ratio_compatibility(left: int, right: int, neutral: float = 0.5) -> float:
    if left <= 0 or right <= 0:
        return neutral
    return clamp01(min(left, right) / max(left, right))


def _first_scalar(record: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    values, _ = _collect(record, keys)
    return values[0] if values else None


def _layout_kind(record: Mapping[str, Any]) -> str:
    for mapping in _maps(record):
        for key in LAYOUT_KIND_KEYS:
            value = mapping.get(key)
            if value not in (None, ""):
                return str(value).strip().lower()
    return ""


def _kind_compatibility(query_kind: str, candidate_kind: str, candidate_offsets: Sequence[int]) -> float:
    if query_kind and candidate_kind:
        if query_kind == candidate_kind:
            return 1.0
        if ("struct" in query_kind and "struct" in candidate_kind) or (
            "array" in query_kind and "array" in candidate_kind
        ):
            return 0.9
        return 0.1
    deltas = _adjacent_deltas(candidate_offsets)
    regular = bool(deltas) and len(set(deltas)) == 1
    if query_kind:
        if "array" in query_kind:
            return 0.85 if regular else 0.35
        if "struct" in query_kind:
            return 0.65 if not regular else 0.50
    return 0.5


def _provenance(candidate: Mapping[str, Any], pair_edges: Sequence[Mapping[str, Any]]) -> float:
    tokens: List[str] = []
    for key in ("candidate_source", "direct_evidence_types", "direct_edge_kinds", "direct_edge_families"):
        value = candidate.get(key)
        if isinstance(value, (list, tuple, set)):
            tokens.extend(str(item).lower() for item in value)
        elif value not in (None, ""):
            tokens.append(str(value).lower())
    tokens.extend(_edge_text(edge) for edge in pair_edges)
    text = " ".join(tokens)
    if any(token in text for token in ("fixed_offset", "field_access", "memory_access")):
        return 1.0
    if any(token in text for token in ("deref", "same_base", "stride")):
        return 0.80
    if any(token in text for token in ("def_use", "constraint", "direct")):
        return 0.55
    return 0.20 if int(candidate.get("direct_edge_count", 0) or 0) > 0 else 0.0


def _inferred_offsets(record: Mapping[str, Any], maximum: int) -> List[int]:
    values, _ = _collect(record, INFERRED_OFFSET_KEYS)
    return _normalise_offsets(values, maximum)


def score_layout_pair(
    query_record: Mapping[str, Any],
    query_node: Mapping[str, Any],
    candidate: Mapping[str, Any],
    target_node: Mapping[str, Any],
    query_edges: Sequence[Mapping[str, Any]],
    target_edges: Sequence[Mapping[str, Any]],
    pair_edges: Sequence[Mapping[str, Any]],
    legacy_score: float,
    config: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Return blended LayoutSem score plus a complete audit trace."""
    config = _normalise_config(config)
    maximum = int(config["max_abs_offset"])

    query_values: List[float] = []
    query_sources: List[str] = []
    for record, prefix in ((query_record, "query_row"), (query_node, "query_node")):
        values, keys = _collect(record, OBSERVED_OFFSET_KEYS + GENERIC_OFFSET_KEYS)
        query_values.extend(values)
        query_sources.extend("%s.%s" % (prefix, key) for key in keys)
        values, keys = _memory_access_offsets(record)
        query_values.extend(values)
        query_sources.extend("%s.%s" % (prefix, key) for key in keys)
    edge_values, edge_sources = _edge_offsets(query_edges)
    if edge_values:
        query_values.extend(edge_values)
        query_sources.extend("query.%s" % source for source in edge_sources)

    # Pair-edge offsets are a lower-confidence fallback.  They are useful when
    # the query itself is an evidence stub, but are never mixed in when the
    # query node already carries explicit offsets.
    pair_values, pair_sources = _edge_offsets(pair_edges)
    if not query_values and pair_values:
        query_values.extend(pair_values)
        query_sources.extend("pair.%s" % source for source in pair_sources)

    candidate_values: List[float] = []
    candidate_sources: List[str] = []
    for record, prefix in ((candidate, "candidate"), (target_node, "target_node")):
        values, keys = _collect(
            record,
            OBSERVED_OFFSET_KEYS + GENERIC_OFFSET_KEYS + INFERRED_OFFSET_KEYS,
        )
        candidate_values.extend(values)
        candidate_sources.extend("%s.%s" % (prefix, key) for key in keys)
        values, keys = _field_member_offsets(record)
        candidate_values.extend(values)
        candidate_sources.extend("%s.%s" % (prefix, key) for key in keys)
    target_edge_values, target_edge_sources = _edge_offsets(target_edges)
    if not candidate_values and target_edge_values:
        candidate_values.extend(target_edge_values)
        candidate_sources.extend("target.%s" % source for source in target_edge_sources)

    observed = _normalise_offsets(query_values, maximum)
    declared = _normalise_offsets(candidate_values, maximum)
    inferred = _inferred_offsets(target_node, maximum)
    usable = bool(observed and declared and config.get("enabled", True))

    exact = _metrics(observed, declared, 0)
    tolerance_scores = []
    for tolerance, weight in zip(config["tolerances"], config["tolerance_weights"]):
        tolerance_scores.append((float(weight), _metrics(observed, declared, int(tolerance))["f1"]))
    tolerance_mass = sum(weight for weight, _ in tolerance_scores)
    tolerant_f1 = (
        sum(weight * value for weight, value in tolerance_scores) / tolerance_mass
        if tolerance_mass > 0.0 else 0.0
    )
    normalised_f1 = _metrics(_normalised(observed), _normalised(declared), 0)["f1"]
    delta_compatibility = _metrics(_adjacent_deltas(observed), _adjacent_deltas(declared), 0)["f1"]

    query_width_values: List[float] = []
    candidate_width_values: List[float] = []
    for record in (query_record, query_node):
        values, _ = _collect(record, WIDTH_KEYS)
        query_width_values.extend(values)
        values, _ = _memory_access_widths(record)
        query_width_values.extend(values)
    for record in (candidate, target_node):
        values, _ = _collect(record, WIDTH_KEYS)
        candidate_width_values.extend(values)
        values, _ = _field_member_widths(record)
        candidate_width_values.extend(values)
    query_widths = _normalise_positive(query_width_values)
    candidate_widths = _normalise_positive(candidate_width_values)
    width_compatibility = (
        _metrics(query_widths, candidate_widths, 0)["f1"]
        if query_widths and candidate_widths else 0.5
    )

    query_alignment = _first_scalar(query_node, ALIGNMENT_KEYS)
    candidate_alignment = _first_scalar(target_node, ALIGNMENT_KEYS)
    if query_alignment is None:
        query_alignment = float(_stride(observed) or 0)
    if candidate_alignment is None:
        candidate_alignment = float(_stride(declared) or 0)
    alignment_compatibility = _ratio_compatibility(
        int(query_alignment or 0), int(candidate_alignment or 0), neutral=0.5,
    )
    stride_compatibility = _ratio_compatibility(_stride(observed), _stride(declared), neutral=0.5)

    query_kind = _layout_kind(query_node) or _layout_kind(query_record)
    candidate_kind = _layout_kind(target_node) or _layout_kind(candidate)
    structural_compatibility = _kind_compatibility(query_kind, candidate_kind, declared)

    query_quality = 1.0 if any(source.startswith("query_node.observed") or "field_offsets_observed" in source for source in query_sources) else 0.85
    if query_sources and all(source.startswith("pair.") for source in query_sources):
        query_quality = 0.60
    candidate_quality = 1.0 if any("observed_field_offsets" in source or "field_offsets_observed" in source for source in candidate_sources) else 0.85
    evidence_quality = clamp01(0.5 * query_quality + 0.5 * candidate_quality) if usable else 0.0

    inferred_ratio = len(set(inferred) & set(declared)) / len(declared) if declared else 0.0
    declared_size = _first_scalar(target_node, SIZE_KEYS)
    out_of_bounds = 0.0
    if declared_size is not None and declared_size > 0 and declared:
        out_of_bounds = sum(1 for value in declared if value >= declared_size) / len(declared)
    invalid_penalty = clamp01(
        (1.0 if observed and not declared else 0.0)
        + 0.35 * inferred_ratio
        + 0.65 * out_of_bounds
    )

    features = {
        "observed_coverage": exact["recall"],
        "candidate_precision": exact["precision"],
        "exact_f1": exact["f1"],
        "tolerant_f1": clamp01(tolerant_f1),
        "offset_jaccard": exact["jaccard"],
        "normalized_offset_f1": normalised_f1,
        "delta_compatibility": delta_compatibility,
        "stride_compatibility": stride_compatibility,
        "width_compatibility": width_compatibility,
        "alignment_compatibility": alignment_compatibility,
        "structural_compatibility": structural_compatibility,
        "provenance_specificity": _provenance(candidate, pair_edges),
        "evidence_quality": evidence_quality,
        "invalid_penalty": invalid_penalty,
    }

    positive_weights = config["positive_weights"]
    positive_mass = sum(float(value) for value in positive_weights.values())
    pairwise_score = clamp01(
        sum(float(weight) * clamp01(features.get(name, 0.0)) for name, weight in positive_weights.items())
        / max(1e-12, positive_mass)
        - float(config["invalid_penalty_weight"]) * invalid_penalty
    )
    blend = float(config["pairwise_blend"]) if usable else 0.0
    blended = clamp01((1.0 - blend) * clamp01(legacy_score) + blend * pairwise_score)

    trace = {
        "schema": "task2_layout_pairwise_trace_v6_7_3",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "layout_evidence_bridge_revision": EVIDENCE_BRIDGE_REVISION,
        "usable": usable,
        "fallback_reason": (
            None if usable else
            "reranker_disabled" if not config.get("enabled", True) else
            "query_offsets_missing" if not observed else
            "candidate_offsets_missing"
        ),
        "query_offsets": observed,
        "candidate_offsets": declared,
        "candidate_inferred_offsets": inferred,
        "query_offset_sources": sorted(set(query_sources)),
        "candidate_offset_sources": sorted(set(candidate_sources)),
        "query_widths": query_widths,
        "candidate_widths": candidate_widths,
        "query_layout_kind": query_kind,
        "candidate_layout_kind": candidate_kind,
        "features": features,
        "legacy_score": clamp01(legacy_score),
        "pairwise_score": pairwise_score,
        "pairwise_blend_applied": blend,
        "blended_score": blended,
        "gold_used": False,
        "test_used": False,
    }
    return blended, trace
