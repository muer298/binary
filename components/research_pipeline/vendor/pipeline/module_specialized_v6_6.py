#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Document-aligned specialised inference features for Task 2.2.3.

The module intentionally reuses the frozen 2.2.1 representation score.  It
does not train another graph encoder.  Each high-level module contributes an
auditable, schema-tolerant feature vector and may optionally load a small
logistic calibration head trained on Train and selected on Validation only.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from module_common_223_v5_3 import (
    HARD_API_SOURCES, clamp01, edge_endpoints, edge_weight, flattened_text,
    normalize_api_source, normalize_semantic_type, noisy_or, read_jsonl,
)
from layout_reranker_v6_7_2 import (
    DEFAULT_CONFIG as DEFAULT_LAYOUT_CONFIG,
    IMPLEMENTATION_REVISION as LAYOUT_IMPLEMENTATION_REVISION,
    score_layout_pair,
)


FEATURE_NAMES = {
    "APISem": (
        "representation", "direct_support", "constraint_consistency",
        "evidence_coverage", "trusted_provenance", "soft_signature",
        "template_completeness", "behavior_evidence", "def_use_evidence",
    ),
    "TypeSem": (
        "representation", "direct_support", "constraint_consistency",
        "evidence_coverage", "hierarchy_evidence", "width_evidence",
        "instruction_rule", "knn_support", "api_interface_constraint",
        "layout_compatibility",
    ),
    "LayoutSem": (
        "representation", "direct_support", "constraint_consistency",
        "evidence_coverage", "offset_evidence", "width_stride_evidence",
        "struct_array_evidence", "alias_cross_function", "type_feedback",
        "nesting_valid",
    ),
    "InterfaceSem": (
        "representation", "direct_support", "constraint_consistency",
        "evidence_coverage", "calling_convention", "entry_def_use",
        "parameter_location", "return_location", "role_evidence",
        "callsite_consistency", "type_feedback",
    ),
}

RULE_WEIGHTS = {
    "APISem": (0.28, 0.12, 0.08, 0.07, 0.18, 0.07, 0.08, 0.06, 0.06),
    "TypeSem": (0.30, 0.10, 0.10, 0.06, 0.08, 0.07, 0.08, 0.05, 0.09, 0.07),
    "LayoutSem": (0.30, 0.09, 0.11, 0.06, 0.10, 0.08, 0.08, 0.06, 0.08, 0.04),
    "InterfaceSem": (0.28, 0.09, 0.09, 0.06, 0.08, 0.09, 0.07, 0.06, 0.07, 0.06, 0.05),
}


def _nested_maps(value: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    yield value
    for key in ("attrs", "metadata", "candidate_metadata", "payload", "raw"):
        child = value.get(key)
        if isinstance(child, Mapping):
            yield child


def first_value(*records: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for mapping in _nested_maps(record):
            for key in keys:
                value = mapping.get(key)
                if value not in (None, "", [], {}):
                    return value
    return None


def token_signal(text: str, tokens: Sequence[str]) -> float:
    hits = sum(1 for token in tokens if token in text)
    return clamp01(hits / max(1.0, min(3.0, float(len(tokens)))))


def count_signal(value: Any, scale: float = 4.0) -> float:
    if isinstance(value, Mapping):
        return clamp01(len(value) / scale)
    if isinstance(value, (list, tuple, set)):
        return clamp01(len(value) / scale)
    try:
        return clamp01(float(value) / scale)
    except (TypeError, ValueError):
        return 0.0


def sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def dot(weights: Sequence[float], values: Sequence[float]) -> float:
    return sum(float(weight) * float(value) for weight, value in zip(weights, values))


class GraphContext:
    """Memory-bounded context for only IDs referenced by reasoning rows."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.incident: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    @classmethod
    def load(
        cls,
        nodes_path: Optional[str],
        edges_path: Optional[str],
        wanted_ids: Sequence[str],
    ) -> "GraphContext":
        result = cls()
        wanted = set(str(value) for value in wanted_ids if value)
        if nodes_path:
            for row in read_jsonl(nodes_path):
                node_id = str(row.get("node_id") or row.get("id") or "")
                if node_id in wanted:
                    result.nodes[node_id] = row
        if edges_path:
            for edge in read_jsonl(edges_path):
                source, target = edge_endpoints(edge)
                if source in wanted or target in wanted:
                    compact = dict(edge)
                    if source in wanted:
                        result.incident[source].append(compact)
                    if target in wanted and target != source:
                        result.incident[target].append(compact)
        return result

    def edges_between(
        self,
        source_id: str,
        target_id: str,
    ) -> List[Dict[str, Any]]:
        """Return unique graph edges joining one query/candidate pair."""
        result: List[Dict[str, Any]] = []
        seen = set()
        for edge in self.incident.get(source_id, []):
            source, target = edge_endpoints(edge)
            if {source, target} != {source_id, target_id}:
                continue
            identity = str(
                edge.get("edge_id")
                or edge.get("id")
                or json.dumps(edge, sort_keys=True, ensure_ascii=False)
            )
            if identity in seen:
                continue
            seen.add(identity)
            result.append(edge)
        return result


def collect_wanted_ids(reasoning_path: str) -> List[str]:
    wanted = set()
    for row in read_jsonl(reasoning_path):
        query_id = str(row.get("query_id") or "")
        if query_id:
            wanted.add(query_id)
        for candidate in row.get("candidates", []) or []:
            target_id = str(candidate.get("target_id") or "")
            if target_id:
                wanted.add(target_id)
    return sorted(wanted)


def _base(candidate: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    representation = clamp01(candidate.get(
        "representation_score",
        candidate.get("confidence", candidate.get("validated_score", 0.0)),
    ))
    return (
        representation,
        clamp01(candidate.get("direct_support_score", candidate.get("direct_edge_weight", 0.0))),
        clamp01(candidate.get("constraint_consistency_score", 0.0)),
        clamp01(candidate.get("evidence_coverage", 0.0)),
    )


def _edge_text(edges: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(flattened_text(edge).lower() for edge in edges[:128])


def specialised_features(
    semantic_type: str,
    query: Mapping[str, Any],
    candidate: Mapping[str, Any],
    target: Mapping[str, Any],
    incident_edges: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    semantic_type = normalize_semantic_type(semantic_type)
    names = FEATURE_NAMES[semantic_type]
    base = list(_base(candidate))
    text = " ".join((
        flattened_text(query).lower(),
        flattened_text(candidate).lower(),
        flattened_text(target).lower(),
        _edge_text(incident_edges),
    ))
    edge_text = _edge_text(incident_edges)

    if semantic_type == "APISem":
        source = normalize_api_source(first_value(
            candidate, target,
            keys=("api_source", "match_source", "recognition_source", "resolution_source"),
        ))
        hard = 1.0 if source in HARD_API_SOURCES else 0.0
        soft = 1.0 if source in ("flirt", "byte_signature", "behavior_match") else 0.0
        template = token_signal(text, ("semantic_template", "api_intent", "param_roles", "precondition", "postcondition", "side_effect"))
        behavior = token_signal(text, ("behavior", "side_effect", "allocation", "io", "string", "crypto"))
        def_use = token_signal(edge_text, ("def_use", "parameter", "return", "api_call"))
        values = base + [hard, soft, template, behavior, def_use]
    elif semantic_type == "TypeSem":
        hierarchy = token_signal(text, ("base_type", "pointer", "pointee", "array", "struct", "function_type", "type_category"))
        width = token_signal(text, ("access_width", "bit_width", "width", "size", "bytes", "bits"))
        instruction = token_signal(text, ("load", "store", "deref", "sign_extend", "zero_extend", "float", "bitmask", "compare"))
        knn = clamp01(first_value(candidate, keys=("knn_score", "neighbor_score", "retrieval_score")) or 0.0)
        api_interface = token_signal(edge_text, ("api_to_type", "interface_to_type", "parameter_type", "return_type"))
        layout = token_signal(edge_text, ("layout_to_type", "field_type", "type_compatible"))
        values = base + [hierarchy, width, instruction, knn, api_interface, layout]
    elif semantic_type == "LayoutSem":
        offsets = first_value(candidate, target, keys=("offsets", "field_offsets", "observed_field_offsets", "inferred_field_offsets", "offset_set"))
        offset_evidence = max(count_signal(offsets), token_signal(text, ("fixed_offset", "offset")))
        width_stride = token_signal(text, ("access_width", "width", "stride", "alignment", "padding"))
        struct_array = token_signal(text, ("struct", "array", "field", "element"))
        alias_cross = token_signal(text, ("alias", "base_object", "cross_function", "support_sites"))
        type_feedback = token_signal(edge_text, ("type_to_layout", "layout_to_type", "field_type", "type_compatible"))
        depth = first_value(candidate, target, keys=("nesting_depth", "depth"))
        try:
            nesting_valid = 1.0 if depth is None or int(depth) <= 3 else 0.0
        except (TypeError, ValueError):
            nesting_valid = 0.0
        values = base + [offset_evidence, width_stride, struct_array, alias_cross, type_feedback, nesting_valid]
    else:
        calling = token_signal(text, ("calling_convention", "abi", "sysv", "cdecl", "aapcs"))
        entry_def_use = token_signal(edge_text, ("def_use", "read_before_def", "entry"))
        parameter = token_signal(text, ("argument_locs", "param_count", "param_types", "reg:a", "reg:r", "reg:x", "stack"))
        returns = token_signal(text, ("return_locs", "return_type", "return_register", "retval"))
        roles = token_signal(text, ("param_roles", "buffer", "size", "length", "fd", "output_ptr", "callback"))
        callsite = token_signal(text, ("callsite", "cross_callsite", "call_site", "caller"))
        type_feedback = token_signal(edge_text, ("type_to_interface", "interface_to_type", "parameter_type", "return_type"))
        values = base + [calling, entry_def_use, parameter, returns, roles, callsite, type_feedback]

    return {name: clamp01(value) for name, value in zip(names, values)}


def load_heads(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") not in (
        "task2_223_specialised_heads_v6_6",
        "task2_223_specialised_heads_v6_7",
        "task2_223_specialised_heads_v6_7_2",
    ):
        raise ValueError("Unsupported head checkpoint schema: %r" % payload.get("schema"))
    if payload.get("test_used_for_training_or_selection"):
        raise ValueError("Refusing a checkpoint that used Test for training/selection")
    return dict(payload.get("modules") or {})


def score_candidate(
    semantic_type: str,
    query: Mapping[str, Any],
    candidate: Mapping[str, Any],
    context: GraphContext,
    heads: Mapping[str, Mapping[str, Any]],
    layout_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    query_id = str(query.get("query_id") or query.get("node_id") or "")
    target_id = str(candidate.get("target_id") or "")
    target = context.nodes.get(target_id, {})
    target_edges = context.incident.get(target_id, [])
    features = specialised_features(
        semantic_type, query, candidate, target, target_edges,
    )
    names = FEATURE_NAMES[semantic_type]
    values = [features[name] for name in names]
    weights = RULE_WEIGHTS[semantic_type]
    rule_score = clamp01(dot(weights, values) / max(1e-12, sum(weights)))

    head = heads.get(semantic_type)
    head_score = None
    if isinstance(head, Mapping):
        checkpoint_names = tuple(head.get("feature_names") or ())
        if checkpoint_names != names:
            raise ValueError("Feature schema mismatch for %s" % semantic_type)
        head_score = sigmoid(float(head.get("bias", 0.0)) + dot(head.get("weights", ()), values))
        blend = clamp01(head.get("blend_weight", 0.35))
        score = clamp01((1.0 - blend) * rule_score + blend * head_score)
    else:
        score = rule_score

    layout_trace = None
    if semantic_type == "LayoutSem":
        score, layout_trace = score_layout_pair(
            query_record=query,
            query_node=context.nodes.get(query_id, {}),
            candidate=candidate,
            target_node=target,
            query_edges=context.incident.get(query_id, []),
            target_edges=target_edges,
            pair_edges=context.edges_between(query_id, target_id),
            legacy_score=score,
            config=(layout_config or DEFAULT_LAYOUT_CONFIG),
        )

    return score, {
        "semantic_type": semantic_type,
        "feature_schema": (
            "specialised_v6_7_3_layout_evidence_bridge"
            if semantic_type == "LayoutSem"
            else "specialised_v6_6"
        ),
        "features": features,
        "rule_score": rule_score,
        "learned_head_score": head_score,
        "learned_head_loaded": head_score is not None,
        "frozen_221_representation_reused": True,
        "target_node_metadata_available": bool(target),
        "incident_edge_count": len(context.incident.get(target_id, [])),
        "layout_reranker_revision": (
            LAYOUT_IMPLEMENTATION_REVISION
            if semantic_type == "LayoutSem" else None
        ),
        "layout_pairwise_trace": layout_trace,
    }


def rows_by_query(path: str) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("query_id") or ""): row
        for row in read_jsonl(path)
        if row.get("query_id")
    }
