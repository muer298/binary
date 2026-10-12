#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared policies and helpers for Task 2.2.2 v5.3.

The implementation follows the three-level conflict policy described in
Research Content 2:

* level 3: deterministic constraints, immutable hard confirmations;
* level 2: high-confidence evidence, weighted soft confirmations;
* level 1: statistical hypotheses, kept as multiple candidates.

Python 3.8+.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SEMANTIC_TYPES: Tuple[str, ...] = (
    "APISem",
    "TypeSem",
    "LayoutSem",
    "InterfaceSem",
)

EVIDENCE_TO_SEMANTIC: Dict[str, str] = {
    "api_call": "APISem",
    "deref_chain": "TypeSem",
    "fixed_offset": "LayoutSem",
    "def_use": "InterfaceSem",
}

SEMANTIC_TO_MODULE: Dict[str, str] = {
    "APISem": "APIInfer",
    "TypeSem": "TypeInfer",
    "LayoutSem": "LayoutInfer",
    "InterfaceSem": "InterfaceInfer",
}

HARD_API_SOURCES: Tuple[str, ...] = (
    "import_table",
    "symbol_table",
    "exact_signature",
)

SOFT_API_SOURCES: Tuple[str, ...] = (
    "flirt",
    "byte_signature",
    "behavior_match",
)

CONFIRM_POLICY: Dict[str, Dict[str, Any]] = {
    "APISem": {
        "soft_confirmation_enabled": True,
        "hard_sources": list(HARD_API_SOURCES),
        "soft_sources": list(SOFT_API_SOURCES),
        "hard_confidence": 1.0,
        "min_evidence": 0.80,
        "soft_threshold": 0.80,
        "gap_threshold": 0.15,
        "min_evidence_coverage": 0.35,
    },
    "TypeSem": {
        "soft_confirmation_enabled": True,
        "min_evidence": 0.70,
        "gap_threshold": 0.15,
        "min_evidence_coverage": 0.35,
    },
    "LayoutSem": {
        "soft_confirmation_enabled": True,
        "min_evidence": 0.75,
        "gap_threshold": 0.20,
        "min_evidence_coverage": 0.35,
    },
    "InterfaceSem": {
        "soft_confirmation_enabled": True,
        "min_evidence": 0.60,
        "gap_threshold": 0.15,
        "min_evidence_coverage": 0.35,
    },
}

CALIBRATABLE_POLICY_KEYS: Tuple[str, ...] = (
    "min_evidence",
    "gap_threshold",
    "min_evidence_coverage",
)

CALIBRATABLE_POLICY_BOOLEAN_KEYS: Tuple[str, ...] = (
    "soft_confirmation_enabled",
)


def load_confirm_policy(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load a validation-calibrated policy without weakening hard rules.

    The three soft-confirmation gates and a fail-closed enable switch are
    configurable.  Deterministic APISem sources and hard confidence stay
    compiled into the implementation, so a malformed calibration file cannot
    turn a statistical API match into a hard seed.
    """
    if not path:
        return {name: dict(value) for name, value in CONFIRM_POLICY.items()}
    payload = load_json(path)
    candidate = payload.get("policies", payload.get("policy", payload))
    if not isinstance(candidate, Mapping):
        raise ValueError("Confirmation policy must be a JSON object")

    merged = {name: dict(value) for name, value in CONFIRM_POLICY.items()}
    for semantic_type, overrides in candidate.items():
        normalized = normalize_semantic_type(semantic_type)
        if normalized not in SEMANTIC_TYPES or not isinstance(overrides, Mapping):
            continue
        for key in CALIBRATABLE_POLICY_KEYS:
            if key not in overrides:
                continue
            value = float(overrides[key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "%s.%s must be in [0, 1]" % (normalized, key)
                )
            merged[normalized][key] = value
        for key in CALIBRATABLE_POLICY_BOOLEAN_KEYS:
            if key not in overrides:
                continue
            value = overrides[key]
            if not isinstance(value, bool):
                raise ValueError(
                    "%s.%s must be a JSON boolean" % (normalized, key)
                )
            merged[normalized][key] = value
        if normalized == "APISem":
            merged[normalized]["soft_threshold"] = merged[normalized][
                "min_evidence"
            ]
    return merged


def apply_confirm_policy(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Apply a policy file process-wide and return the active policy."""
    active = load_confirm_policy(path)
    CONFIRM_POLICY.clear()
    CONFIRM_POLICY.update(active)
    return active

EVIDENCE_COMPONENT_WEIGHTS: Dict[str, float] = {
    "direct_support": 0.35,
    "constraint_consistency": 0.45,
    "cross_object_collaboration": 0.20,
}

CONFIRMED_STATUSES = frozenset(("confirmed_hard", "confirmed_soft"))
ALL_STATUSES = frozenset(
    ("confirmed_hard", "confirmed_soft", "hypothesis", "rejected")
)

# The compact ``node_index.json`` produced by the embedding stage normally
# keeps only node_id/node_type/sample_id.  API recognition provenance therefore
# has to be recovered from the complete USG node record (or from the direct
# evidence edge).  These keys intentionally describe provenance rather than an
# API name: a known name alone is not sufficient for hard confirmation.
API_SOURCE_KEYS: Tuple[str, ...] = (
    "api_source",
    "match_type",
    "match_source",
    "recognition_source",
    "resolution_source",
    "symbol_source",
    "signature_source",
    "source_kind",
    "candidate_source",
    "provenance",
    "resolution_method",
    "match_method",
    "recognizer",
    "resolver",
    "origin",
    "origin_type",
    "source_type",
    "table_source",
    "symbol_table_source",
    "signature_match_type",
)

API_SOURCE_GENERIC_KEYS: Tuple[str, ...] = (
    "source",
    "kind",
    "type",
    "table",
    "section",
    "provider",
    "method",
)

API_SOURCE_FLAG_MAP: Dict[str, str] = {
    "is_import": "import_table",
    "is_imported": "import_table",
    "from_import_table": "import_table",
    "import_table_match": "import_table",
    "resolved_from_import": "import_table",
    "resolved_by_import": "import_table",
    "is_plt": "import_table",
    "from_plt": "import_table",
    "plt_resolved": "import_table",
    "is_got": "import_table",
    "from_got": "import_table",
    "got_resolved": "import_table",
    "is_dynamic_symbol": "import_table",
    "is_undefined_symbol": "import_table",
    "from_dynamic_symbol_table": "import_table",
    "from_symbol_table": "symbol_table",
    "symbol_table_match": "symbol_table",
    "resolved_from_symbol": "symbol_table",
    "resolved_by_symbol": "symbol_table",
    "from_symtab": "symbol_table",
    "symtab_match": "symbol_table",
    "is_symbol": "symbol_table",
    "is_exact_signature": "exact_signature",
    "exact_signature_match": "exact_signature",
    "signature_exact_match": "exact_signature",
    "resolved_by_exact_signature": "exact_signature",
}

API_SOURCE_PRESENCE_MAP: Dict[str, str] = {
    "import_name": "import_table",
    "imported_name": "import_table",
    "import_symbol": "import_table",
    "import_entry": "import_table",
    "plt_name": "import_table",
    "plt_symbol": "import_table",
    "got_name": "import_table",
    "got_symbol": "import_table",
    "dynamic_symbol_name": "import_table",
    "undefined_symbol_name": "import_table",
    "symtab_name": "symbol_table",
    "symbol_table_name": "symbol_table",
    "exact_signature_id": "exact_signature",
    "exact_signature_name": "exact_signature",
    "exact_signature_result": "exact_signature",
}


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Read dictionaries from JSONL and report the exact malformed line."""
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    "Invalid JSONL %s:%d: %s" % (path, line_number, exc)
                )
            if isinstance(row, dict):
                yield row


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def ensure_output_path(path: str, overwrite: bool = False) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output: %s. "
            "Use a new output directory or pass --overwrite." % output
        )
    return output


def write_json(path: str, value: Any, overwrite: bool = False) -> None:
    output = ensure_output_path(path, overwrite=overwrite)
    temporary = output.with_name(output.name + ".tmp.%d" % os.getpid())
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(output)


def clamp01(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def cosine_to_probability(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    return clamp01((number + 1.0) / 2.0)


def noisy_or(values: Sequence[float]) -> float:
    """Aggregate independent positive supports without exceeding one."""
    remaining = 1.0
    found = False
    for value in values:
        found = True
        remaining *= 1.0 - clamp01(value)
    return 1.0 - remaining if found else 0.0


def weighted_average(
    values_and_weights: Sequence[Tuple[float, float]],
    default: float = 0.0,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values_and_weights:
        positive_weight = max(0.0, float(weight))
        numerator += clamp01(value) * positive_weight
        denominator += positive_weight
    if denominator <= 0:
        return clamp01(default)
    return clamp01(numerator / denominator)


def node_id(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("node_id") or value.get("id")
    return str(value).strip() if value is not None else ""


def endpoints(edge: Mapping[str, Any]) -> Tuple[str, str]:
    source = node_id(
        edge.get("source_node")
        or edge.get("source")
        or edge.get("src")
        or edge.get("from")
    )
    target = node_id(
        edge.get("target_node")
        or edge.get("target")
        or edge.get("dst")
        or edge.get("to")
    )
    return source, target


def edge_attrs(edge: Mapping[str, Any]) -> Mapping[str, Any]:
    value = edge.get("attrs")
    return value if isinstance(value, dict) else {}


def first_present(
    objects: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> Any:
    for obj in objects:
        for key in keys:
            value = obj.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def edge_weight(edge: Mapping[str, Any]) -> float:
    attrs = edge_attrs(edge)
    value = first_present(
        (edge, attrs),
        ("weight", "strength", "confidence", "score", "support_strength"),
    )
    return clamp01(0.5 if value is None else value)


def normalize_semantic_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "")
    aliases = {
        "api": "APISem",
        "apisem": "APISem",
        "apiinfer": "APISem",
        "type": "TypeSem",
        "typesem": "TypeSem",
        "typeinfer": "TypeSem",
        "layout": "LayoutSem",
        "layoutsem": "LayoutSem",
        "layoutinfer": "LayoutSem",
        "interface": "InterfaceSem",
        "interfacesem": "InterfaceSem",
        "interfaceinfer": "InterfaceSem",
    }
    return aliases.get(text, str(value or "").strip())


def normalize_evidence_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "api": "api_call",
        "call": "api_call",
        "api_semantic": "api_call",
        "deref": "deref_chain",
        "dereference": "deref_chain",
        "pointer_chain": "deref_chain",
        "offset": "fixed_offset",
        "field_offset": "fixed_offset",
        "fixedoffset": "fixed_offset",
        "data_flow": "def_use",
        "defuse": "def_use",
        "use_def": "def_use",
    }
    if text in EVIDENCE_TO_SEMANTIC:
        return text
    if text in aliases:
        return aliases[text]
    for known in EVIDENCE_TO_SEMANTIC:
        if known in text:
            return known
    if "deref" in text or "pointer_chain" in text:
        return "deref_chain"
    if "offset" in text or "displacement" in text:
        return "fixed_offset"
    if "callee" in text or "api_name" in text or "callsite" in text:
        return "api_call"
    if "def_use" in text or "data_flow" in text:
        return "def_use"
    return "unknown"


def evidence_type_from_objects(*objects: Mapping[str, Any]) -> str:
    keys = (
        "evidence_type",
        "anchor_type",
        "relation_family",
        "evidence_relation",
        "evidence_name",
    )
    containers: List[Mapping[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        containers.append(obj)
        for container_key in ("query_context", "source_context", "attrs", "meta"):
            nested = obj.get(container_key)
            if isinstance(nested, Mapping):
                containers.append(nested)
    for obj in containers:
        for key in keys:
            if key in obj:
                normalized = normalize_evidence_type(obj.get(key))
                if normalized != "unknown":
                    return normalized
    return "unknown"


def edge_family(edge: Mapping[str, Any]) -> str:
    attrs = edge_attrs(edge)
    pieces = []
    for key in (
        "edge_kind",
        "relation_family",
        "relation_type",
        "edge_type",
        "constraint_type",
        "evidence_type",
    ):
        for obj in (edge, attrs):
            value = obj.get(key)
            if value not in (None, ""):
                pieces.append(str(value).lower())
    text = "|".join(pieces)
    if "constraint" in text or "compatible" in text or "requires" in text:
        return "constraint"
    if "evidence" in text or normalize_evidence_type(text) != "unknown":
        return "evidence"
    return "unknown"


def edge_kind(edge: Mapping[str, Any]) -> str:
    attrs = edge_attrs(edge)
    values = []
    for key in (
        "edge_kind",
        "relation_family",
        "relation_type",
        "edge_type",
        "constraint_type",
        "evidence_type",
    ):
        for obj in (edge, attrs):
            value = obj.get(key)
            if value not in (None, ""):
                values.append(str(value))
    return "|".join(values)


def edge_satisfaction(edge: Mapping[str, Any]) -> Optional[float]:
    attrs = edge_attrs(edge)
    value = first_present(
        (edge, attrs),
        (
            "satisfaction",
            "satisfy",
            "satisfaction_score",
            "constraint_satisfaction",
            "compatibility",
            "compatibility_score",
        ),
    )
    if value is not None:
        return clamp01(value)
    compatible = first_present((edge, attrs), ("compatible", "is_compatible"))
    if isinstance(compatible, bool):
        return 1.0 if compatible else 0.0
    violated = first_present((edge, attrs), ("violated", "is_violation"))
    if isinstance(violated, bool):
        return 0.0 if violated else 1.0
    return None


def constraint_level(edge: Mapping[str, Any], source_sem_type: str = "") -> int:
    attrs = edge_attrs(edge)
    value = first_present(
        (edge, attrs),
        ("constraint_level", "confirmation_level", "priority_level", "level"),
    )
    if value is not None:
        try:
            number = int(value)
            if number in (1, 2, 3):
                return number
        except Exception:
            pass
    text = edge_kind(edge).lower()
    if "hard" in text or "deterministic" in text:
        return 3
    if "high_confidence" in text or "high-confidence" in text:
        return 2
    if normalize_semantic_type(source_sem_type) == "APISem":
        return 3
    if normalize_semantic_type(source_sem_type) in ("TypeSem", "LayoutSem"):
        return 2
    return 1


def normalize_api_source(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        return "unknown"
    exact_aliases = {
        "import": "import_table",
        "imports": "import_table",
        "imported": "import_table",
        "import_name": "import_table",
        "import_entry": "import_table",
        "imported_function": "import_table",
        "elf_import": "import_table",
        "importtable": "import_table",
        "got": "import_table",
        "got_plt": "import_table",
        "plt": "import_table",
        "plt_stub": "import_table",
        "dynamic_import": "import_table",
        "dynamic_symbol": "import_table",
        "dynamic_symbol_table": "import_table",
        "dynsym": "import_table",
        "undefined_symbol": "import_table",
        "undefined_dynamic_symbol": "import_table",
        "symbol": "symbol_table",
        "symbols": "symbol_table",
        "symboltable": "symbol_table",
        "symtab": "symbol_table",
        "static_symbol": "symbol_table",
        "static_symbol_table": "symbol_table",
        "exact": "exact_signature",
        "exact_match": "exact_signature",
        "exact_name": "exact_signature",
        "signature_exact": "exact_signature",
        "exact_signature_match": "exact_signature",
        "signature_database_exact": "exact_signature",
        "flirt_signature": "flirt",
        "byte_pattern": "byte_signature",
        "byte_sequence": "byte_signature",
        "behavior": "behavior_match",
        "behaviour": "behavior_match",
        "semantic_behavior": "behavior_match",
    }
    if text in exact_aliases:
        return exact_aliases[text]
    if text in HARD_API_SOURCES or text in SOFT_API_SOURCES:
        return text
    if (
        "import" in text
        or text in ("got", "plt", "got_plt", "dynsym")
        or "dynamic_symbol" in text
        or "undefined_symbol" in text
    ):
        return "import_table"
    if "symbol" in text or "symtab" in text:
        return "symbol_table"
    if "exact" in text and ("signature" in text or "match" in text):
        return "exact_signature"
    if "flirt" in text:
        return "flirt"
    if "byte" in text and ("signature" in text or "pattern" in text):
        return "byte_signature"
    if "behav" in text:
        return "behavior_match"
    return "unknown"


def _truthy_provenance_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "present",
        "matched",
        "resolved",
    )


def _mapping_is_edge_like(value: Mapping[str, Any]) -> bool:
    has_source = any(
        key in value for key in ("source", "source_node", "src", "from")
    )
    has_target = any(
        key in value for key in ("target", "target_node", "dst", "to")
    )
    return has_source and has_target


def _normalize_field_key(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def _iter_nested_mappings(
    value: Any,
    path: str = "",
    depth: int = 0,
    max_depth: int = 5,
) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    """Yield nested mappings with paths while bounding hostile/huge records."""
    if depth > max_depth:
        return
    if isinstance(value, Mapping):
        yield path, value
        for key, nested in value.items():
            nested_path = "%s.%s" % (path, key) if path else str(key)
            if isinstance(nested, Mapping):
                for item in _iter_nested_mappings(
                    nested,
                    nested_path,
                    depth + 1,
                    max_depth,
                ):
                    yield item
            elif isinstance(nested, (list, tuple)):
                for index, item_value in enumerate(nested[:64]):
                    if not isinstance(item_value, Mapping):
                        continue
                    item_path = "%s[%d]" % (nested_path, index)
                    for item in _iter_nested_mappings(
                        item_value,
                        item_path,
                        depth + 1,
                        max_depth,
                    ):
                        yield item


def api_source_evidence_from_objects(
    *objects: Mapping[str, Any],
    object_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Return the first explicit API recognition provenance with audit data.

    Values are considered only under provenance-like keys or explicit boolean
    flags.  API names and embedding-neighbour status are deliberately ignored,
    so this helper cannot manufacture a deterministic seed from a known name.
    """
    labels = list(object_labels or [])
    for object_index, root in enumerate(objects):
        if not isinstance(root, Mapping):
            continue
        label = (
            labels[object_index]
            if object_index < len(labels)
            else "object_%d" % object_index
        )
        for path, mapping in _iter_nested_mappings(root):
            edge_like = _mapping_is_edge_like(mapping)
            normalized_items = {
                _normalize_field_key(raw_key): (str(raw_key), raw_value)
                for raw_key, raw_value in mapping.items()
            }

            for key in API_SOURCE_KEYS:
                if key not in normalized_items:
                    continue
                actual_key, raw = normalized_items[key]
                normalized = normalize_api_source(raw)
                if normalized != "unknown":
                    return {
                        "source": normalized,
                        "origin": label,
                        "path": (
                            "%s.%s" % (path, actual_key)
                        ).strip("."),
                        "raw_value": str(raw),
                        "signal_kind": "provenance_field",
                    }

            for key, source in API_SOURCE_FLAG_MAP.items():
                if key not in normalized_items:
                    continue
                actual_key, raw = normalized_items[key]
                if _truthy_provenance_flag(
                    raw
                ):
                    return {
                        "source": source,
                        "origin": label,
                        "path": (
                            "%s.%s" % (path, actual_key)
                        ).strip("."),
                        "raw_value": str(raw),
                        "signal_kind": "deterministic_flag",
                    }

            for key, source in API_SOURCE_PRESENCE_MAP.items():
                if key not in normalized_items:
                    continue
                actual_key, raw = normalized_items[key]
                if raw in (None, "", [], {}, False):
                    continue
                return {
                    "source": source,
                    "origin": label,
                    "path": (
                        "%s.%s" % (path, actual_key)
                    ).strip("."),
                    "raw_value": str(raw),
                    "signal_kind": "deterministic_presence_field",
                }

            for key in API_SOURCE_GENERIC_KEYS:
                if key not in normalized_items:
                    continue
                actual_key, raw = normalized_items[key]
                # ``source`` on an edge is the endpoint node id, not
                # provenance. Other generic keys may still carry explicit
                # values such as ".plt", ".dynsym" or "ImportTable".
                if key == "source" and edge_like:
                    continue
                if isinstance(raw, (Mapping, list, tuple)):
                    continue
                normalized = normalize_api_source(raw)
                if normalized != "unknown":
                    return {
                        "source": normalized,
                        "origin": label,
                        "path": (
                            "%s.%s" % (path, actual_key)
                        ).strip("."),
                        "raw_value": str(raw),
                        "signal_kind": "generic_provenance_field",
                    }

    return {
        "source": "unknown",
        "origin": "none",
        "path": "",
        "raw_value": "",
        "signal_kind": "none",
    }


def api_source_from_objects(*objects: Mapping[str, Any]) -> str:
    return str(api_source_evidence_from_objects(*objects)["source"])


def unwrap_node_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize common JSONL wrappers while retaining outer metadata."""
    if not isinstance(row, Mapping):
        return {}
    for key in ("node", "record", "item", "data"):
        nested = row.get(key)
        if isinstance(nested, Mapping) and node_id(nested):
            merged = dict(row)
            merged.update(nested)
            return merged
    return dict(row)


def load_node_metadata(
    path: Optional[str],
    wanted_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load selected complete USG node records from a JSONL file."""
    if not path:
        return {}
    wanted = (
        {str(value) for value in wanted_ids if str(value)}
        if wanted_ids is not None
        else None
    )
    result: Dict[str, Dict[str, Any]] = {}
    for raw in read_jsonl(path):
        row = unwrap_node_record(raw)
        identity = node_id(row)
        if not identity or (wanted is not None and identity not in wanted):
            continue
        if identity in result:
            merged = dict(result[identity])
            merged.update(row)
            result[identity] = merged
        else:
            result[identity] = row
    return result


def is_hard_api_source(source: Any) -> bool:
    return normalize_api_source(source) in HARD_API_SOURCES


def is_soft_api_source(source: Any) -> bool:
    return normalize_api_source(source) in SOFT_API_SOURCES


def evidence_score(
    representation_score: float,
    direct_support_values: Sequence[float],
    constraint_values: Sequence[float],
    collaboration_values: Sequence[float],
    prior_weight: float = 0.25,
) -> Dict[str, Any]:
    """Calculate the auditable three-part evidence score.

    The alpha weights are applied only to observed evidence components. The
    normalized evidence result is then calibrated with the neural
    representation prior. A candidate supported only by its embedding receives
    half of the representation probability and cannot pass the default gates.
    """
    representation = clamp01(representation_score)
    direct = noisy_or(direct_support_values)
    constraint = weighted_average(
        [(value, 1.0) for value in constraint_values]
    )
    collaboration = noisy_or(collaboration_values)

    observed = {
        "direct_support": bool(direct_support_values),
        "constraint_consistency": bool(constraint_values),
        "cross_object_collaboration": bool(collaboration_values),
    }
    values = {
        "direct_support": direct,
        "constraint_consistency": constraint,
        "cross_object_collaboration": collaboration,
    }
    active = [
        (
            values[name],
            EVIDENCE_COMPONENT_WEIGHTS[name],
        )
        for name in EVIDENCE_COMPONENT_WEIGHTS
        if observed[name]
    ]
    coverage = sum(
        EVIDENCE_COMPONENT_WEIGHTS[name]
        for name in EVIDENCE_COMPONENT_WEIGHTS
        if observed[name]
    )

    if active:
        core = weighted_average(active)
        prior = clamp01(prior_weight)
        fused = (1.0 - prior) * core + prior * representation
    else:
        core = 0.0
        fused = 0.5 * representation

    return {
        "representation_score": representation,
        "direct_support_score": direct,
        "constraint_consistency_score": constraint,
        "cross_object_collaboration_score": collaboration,
        "evidence_core_score": clamp01(core),
        "evidence_coverage": clamp01(coverage),
        "validated_score": clamp01(fused),
        "component_observed": observed,
        "component_weights": dict(EVIDENCE_COMPONENT_WEIGHTS),
        "representation_prior_weight": clamp01(prior_weight),
    }


def determine_status(
    sem_type: str,
    candidate: Optional[Mapping[str, Any]],
    evidence_score_value: float,
    gap: float,
    source: Optional[str] = None,
    evidence_coverage: float = 1.0,
    hard_conflict: bool = False,
) -> Dict[str, Any]:
    """Apply semantic-type-specific confirmation policy."""
    semantic_type = normalize_semantic_type(sem_type)
    if semantic_type not in CONFIRM_POLICY:
        return {
            "status": "hypothesis",
            "confidence": clamp01(evidence_score_value),
            "level": 1,
            "confirmation_level": 1,
            "confirmation_source": "unknown_semantic_type",
            "propagation_weight": 0.0,
            "locked": False,
        }

    policy = CONFIRM_POLICY[semantic_type]
    normalized_source = normalize_api_source(
        source
        or (
            api_source_from_objects(candidate)
            if isinstance(candidate, Mapping)
            else "unknown"
        )
    )
    score = clamp01(evidence_score_value)
    score_gap = max(0.0, float(gap))
    coverage = clamp01(evidence_coverage)

    if hard_conflict:
        return {
            "status": "hypothesis",
            "confidence": score,
            "level": 1,
            "confirmation_level": 1,
            "confirmation_source": "hard_constraint_conflict",
            "propagation_weight": 0.0,
            "locked": False,
        }

    if (
        semantic_type == "APISem"
        and normalized_source in policy["hard_sources"]
    ):
        return {
            "status": "confirmed_hard",
            "confidence": float(policy["hard_confidence"]),
            "level": 3,
            "confirmation_level": 3,
            "confirmation_source": normalized_source,
            "propagation_weight": 1.0,
            "locked": True,
        }

    source_allows_soft_confirmation = (
        semantic_type != "APISem"
        or normalized_source in policy["soft_sources"]
    )
    passes_soft_gate = (
        bool(policy.get("soft_confirmation_enabled", True))
        and source_allows_soft_confirmation
        and score >= float(policy["min_evidence"])
        and score_gap >= float(policy["gap_threshold"])
        and coverage >= float(policy["min_evidence_coverage"])
    )
    if passes_soft_gate:
        return {
            "status": "confirmed_soft",
            "confidence": score,
            "level": 2,
            "confirmation_level": 2,
            "confirmation_source": (
                normalized_source
                if semantic_type == "APISem"
                and normalized_source != "unknown"
                else "evidence_consensus"
            ),
            "propagation_weight": score,
            "locked": True,
        }

    return {
        "status": "hypothesis",
        "confidence": score,
        "level": 1,
        "confirmation_level": 1,
        "confirmation_source": (
            "insufficient_unique_evidence"
            if bool(policy.get("soft_confirmation_enabled", True))
            else "soft_confirmation_disabled_by_validation"
        ),
        "propagation_weight": 0.0,
        "locked": False,
    }


def propagation_weight(value: Mapping[str, Any]) -> float:
    status = str(value.get("status") or "")
    if status == "confirmed_hard":
        return 1.0
    if status == "confirmed_soft":
        return clamp01(
            value.get("propagation_weight", value.get("confidence", 0.0))
        )
    return 0.0


def semantic_type_for_evidence(evidence_type: Any) -> str:
    return EVIDENCE_TO_SEMANTIC.get(
        normalize_evidence_type(evidence_type),
        "",
    )


def stable_candidate_order(candidates: Sequence[Mapping[str, Any]]) -> List[int]:
    return sorted(
        range(len(candidates)),
        key=lambda index: (
            -clamp01(
                candidates[index].get(
                    "validated_score",
                    candidates[index].get("confidence", 0.0),
                )
            ),
            str(candidates[index].get("target_id") or ""),
            index,
        ),
    )


def select_row_decision(
    semantic_decisions: Mapping[str, Mapping[str, Any]],
    evidence_type: Any,
) -> Tuple[str, Dict[str, Any]]:
    routed_semantic_type = semantic_type_for_evidence(evidence_type)
    if routed_semantic_type and routed_semantic_type in semantic_decisions:
        return routed_semantic_type, dict(
            semantic_decisions[routed_semantic_type]
        )

    if not semantic_decisions:
        return "", {
            "status": "rejected",
            "confidence": 0.0,
            "confirmation_level": 0,
            "confirmation_source": "no_candidates",
            "propagation_weight": 0.0,
            "locked": True,
        }

    semantic_type, decision = max(
        semantic_decisions.items(),
        key=lambda item: (
            propagation_weight(item[1]),
            clamp01(item[1].get("confidence", 0.0)),
            -SEMANTIC_TYPES.index(item[0])
            if item[0] in SEMANTIC_TYPES
            else -len(SEMANTIC_TYPES),
        ),
    )
    result = dict(decision)
    if result.get("status") in CONFIRMED_STATUSES:
        result["confirmation_source"] = (
            str(result.get("confirmation_source") or "")
            + "|unrouted_fallback"
        ).strip("|")
    return semantic_type, result


def summarize_statuses(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    row_counts = {status: 0 for status in ALL_STATUSES}
    semantic_counts: Dict[str, Dict[str, int]] = {
        semantic_type: {status: 0 for status in ALL_STATUSES}
        for semantic_type in SEMANTIC_TYPES
    }
    for row in rows:
        row_status = str(row.get("status") or "hypothesis")
        row_counts.setdefault(row_status, 0)
        row_counts[row_status] += 1
        decisions = row.get("semantic_decisions")
        if not isinstance(decisions, Mapping):
            continue
        for semantic_type, decision in decisions.items():
            normalized = normalize_semantic_type(semantic_type)
            if normalized not in semantic_counts or not isinstance(
                decision, Mapping
            ):
                continue
            status = str(decision.get("status") or "hypothesis")
            semantic_counts[normalized].setdefault(status, 0)
            semantic_counts[normalized][status] += 1
    return {
        "row_status_counts": row_counts,
        "semantic_status_counts": semantic_counts,
    }


__all__ = [
    "ALL_STATUSES",
    "CONFIRMED_STATUSES",
    "CONFIRM_POLICY",
    "CALIBRATABLE_POLICY_KEYS",
    "EVIDENCE_COMPONENT_WEIGHTS",
    "EVIDENCE_TO_SEMANTIC",
    "HARD_API_SOURCES",
    "SEMANTIC_TO_MODULE",
    "SEMANTIC_TYPES",
    "SOFT_API_SOURCES",
    "api_source_evidence_from_objects",
    "api_source_from_objects",
    "clamp01",
    "constraint_level",
    "cosine_to_probability",
    "determine_status",
    "edge_attrs",
    "edge_family",
    "edge_kind",
    "edge_satisfaction",
    "edge_weight",
    "endpoints",
    "ensure_output_path",
    "evidence_score",
    "load_confirm_policy",
    "apply_confirm_policy",
    "evidence_type_from_objects",
    "is_hard_api_source",
    "is_soft_api_source",
    "load_json",
    "load_node_metadata",
    "node_id",
    "noisy_or",
    "normalize_api_source",
    "normalize_evidence_type",
    "normalize_semantic_type",
    "propagation_weight",
    "read_jsonl",
    "select_row_decision",
    "semantic_type_for_evidence",
    "stable_candidate_order",
    "summarize_statuses",
    "unwrap_node_record",
    "weighted_average",
    "write_json",
]
