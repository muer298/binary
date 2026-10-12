#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities and policies for Task 2.2.3 v5.3."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SEMANTIC_TO_MODULE = {
    "APISem": "APIInfer",
    "TypeSem": "TypeInfer",
    "LayoutSem": "LayoutInfer",
    "InterfaceSem": "InterfaceInfer",
}
MODULE_TO_SEMANTIC = {value: key for key, value in SEMANTIC_TO_MODULE.items()}
SEMANTIC_TYPES = tuple(SEMANTIC_TO_MODULE)
MODULES = tuple(MODULE_TO_SEMANTIC)
CONFIRMED = frozenset(("confirmed_hard", "confirmed_soft"))
ALL_STATUSES = frozenset((*CONFIRMED, "hypothesis", "rejected"))

POLICY = {
    "APISem": {"min_evidence": 0.80, "gap": 0.15, "coverage": 0.35},
    "TypeSem": {"min_evidence": 0.70, "gap": 0.15, "coverage": 0.35},
    "LayoutSem": {"min_evidence": 0.75, "gap": 0.20, "coverage": 0.35},
    "InterfaceSem": {"min_evidence": 0.60, "gap": 0.15, "coverage": 0.35},
}

HARD_API_SOURCES = frozenset(("import_table", "symbol_table", "exact_signature"))
SOFT_API_SOURCES = frozenset(("flirt", "byte_signature", "behavior_match"))

# Paper 2.2.3.5: direction-specific collaboration strength.
DIRECTION_WEIGHT = {
    ("APISem", "TypeSem"): 1.00,
    ("APISem", "InterfaceSem"): 1.00,
    ("TypeSem", "LayoutSem"): 0.80,
    ("LayoutSem", "TypeSem"): 0.80,
    ("TypeSem", "InterfaceSem"): 0.65,
    ("InterfaceSem", "TypeSem"): 0.45,
}


def clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def noisy_or(values: Sequence[float]) -> float:
    product = 1.0
    for value in values:
        product *= 1.0 - clamp01(value)
    return clamp01(1.0 - product)


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError("Non-object JSONL row at %s:%d" % (path, line_number))
            yield value


def write_json(path: str, value: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def output_path(path: str, overwrite: bool = False) -> Path:
    result = Path(path)
    if result.exists() and not overwrite:
        raise RuntimeError("Refusing to overwrite: %s" % result)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def normalize_semantic_type(value: Any) -> str:
    text = re.sub(r"[^a-z]", "", str(value or "").lower())
    aliases = {
        "api": "APISem", "apisem": "APISem", "apiinfer": "APISem",
        "type": "TypeSem", "typesem": "TypeSem", "typeinfer": "TypeSem",
        "layout": "LayoutSem", "layoutsem": "LayoutSem", "layoutinfer": "LayoutSem",
        "interface": "InterfaceSem", "interfacesem": "InterfaceSem",
        "interfaceinfer": "InterfaceSem",
    }
    return aliases.get(text, str(value or ""))


def load_confirmation_policy_223(
    path: Optional[str],
) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """Load the Validation-calibrated 2.2.2 confirmation policy for 2.2.3.

    2.2.2 and 2.2.3 use different names for the gap and coverage gates.  This
    adapter maps the calibrated keys explicitly and, most importantly,
    preserves ``soft_confirmation_enabled=False`` as a fail-closed decision.
    """
    active = {
        semantic_type: {
            **values,
            "soft_confirmation_enabled": True,
        }
        for semantic_type, values in POLICY.items()
    }
    if not path:
        return active, None

    policy_path = Path(path)
    if not policy_path.is_file():
        raise FileNotFoundError(
            "Confirmation policy does not exist: %s" % policy_path
        )
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    policies = payload.get("policies", payload.get("policy", payload))
    if not isinstance(policies, Mapping):
        raise ValueError("Confirmation policy must contain a policy object")

    key_map = {
        "min_evidence": "min_evidence",
        "gap_threshold": "gap",
        "min_evidence_coverage": "coverage",
    }
    for raw_name, overrides in policies.items():
        semantic_type = normalize_semantic_type(raw_name)
        if semantic_type not in SEMANTIC_TYPES or not isinstance(
            overrides, Mapping
        ):
            continue
        if "soft_confirmation_enabled" in overrides:
            enabled = overrides["soft_confirmation_enabled"]
            if not isinstance(enabled, bool):
                raise ValueError(
                    "%s.soft_confirmation_enabled must be boolean"
                    % semantic_type
                )
            active[semantic_type]["soft_confirmation_enabled"] = enabled
        for source_key, target_key in key_map.items():
            if source_key not in overrides:
                continue
            value = float(overrides[source_key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "%s.%s must be in [0,1]"
                    % (semantic_type, source_key)
                )
            active[semantic_type][target_key] = value
    return active, str(policy_path.resolve())


def normalize_api_source(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    if "import" in text or "dynsym" in text or text in ("plt", "got"):
        return "import_table"
    if "exact" in text and "signature" in text:
        return "exact_signature"
    if "symbol" in text or "symtab" in text or "external" in text:
        return "symbol_table"
    if "flirt" in text:
        return "flirt"
    if "byte" in text and "signature" in text:
        return "byte_signature"
    if "behavior" in text or "behaviour" in text:
        return "behavior_match"
    return "unknown"


def attrs(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for key in ("attrs", "metadata", "candidate_metadata", "payload", "raw"):
        value = candidate.get(key)
        if isinstance(value, Mapping):
            merged.update(value)
    return merged


def flattened_text(value: Any, depth: int = 0) -> str:
    if depth > 4:
        return ""
    if isinstance(value, Mapping):
        return " ".join(
            "%s %s" % (key, flattened_text(item, depth + 1))
            for key, item in list(value.items())[:80]
        )
    if isinstance(value, (list, tuple)):
        return " ".join(flattened_text(item, depth + 1) for item in value[:80])
    return str(value or "")


def edge_endpoints(edge: Mapping[str, Any]) -> Tuple[str, str]:
    source = edge.get("source_node") or edge.get("source") or edge.get("src")
    target = edge.get("target_node") or edge.get("target") or edge.get("dst")
    return str(source or ""), str(target or "")


def edge_weight(edge: Mapping[str, Any]) -> float:
    metadata = edge.get("attrs") if isinstance(edge.get("attrs"), Mapping) else {}
    for key in ("weight", "strength", "confidence", "score", "evidence_strength"):
        if key in edge:
            return clamp01(edge[key])
        if key in metadata:
            return clamp01(metadata[key])
    return 1.0


def edge_satisfaction(edge: Mapping[str, Any]):
    """Return explicit compatibility in [0,1], or None when unspecified."""
    metadata = edge.get("attrs") if isinstance(edge.get("attrs"), Mapping) else {}
    for key in (
        "satisfaction", "satisfaction_score", "constraint_satisfaction",
        "compatibility", "compatibility_score",
    ):
        if key in edge:
            return clamp01(edge[key])
        if key in metadata:
            return clamp01(metadata[key])
    for key in ("compatible", "is_compatible"):
        value = edge.get(key, metadata.get(key))
        if isinstance(value, bool):
            return 1.0 if value else 0.0
    for key in ("violated", "is_violation"):
        value = edge.get(key, metadata.get(key))
        if isinstance(value, bool):
            return 0.0 if value else 1.0
    return None


def edge_is_constraint(edge: Mapping[str, Any]) -> bool:
    metadata = edge.get("attrs") if isinstance(edge.get("attrs"), Mapping) else {}
    text = " ".join(
        str(value or "") for value in (
            edge.get("edge_kind"), edge.get("edge_type"),
            edge.get("constraint_type"), metadata.get("edge_kind"),
            metadata.get("edge_type"), metadata.get("constraint_type"),
        )
    ).lower()
    return "constraint" in text or any(
        token in text for token in ("api_to_", "layout_to_type", "interface_to_type",
                                    "type_compatible", "parameter_type", "field_type")
    )


def base_candidate_score(candidate: Mapping[str, Any]) -> float:
    return clamp01(candidate.get("confidence", candidate.get("validated_score", 0.0)))


def module_specific_score(semantic_type: str, candidate: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Auditable module score without using gold labels."""
    metadata = attrs(candidate)
    text = flattened_text(metadata).lower()
    direct = clamp01(candidate.get("direct_support_score", candidate.get("direct_edge_weight", 0.0)))
    constraint = clamp01(candidate.get("constraint_consistency_score", 0.0))
    coverage = clamp01(candidate.get("evidence_coverage", 0.0))
    representation = clamp01(candidate.get("representation_score", base_candidate_score(candidate)))
    signals: Dict[str, float] = {
        "representation": representation,
        "direct_support": direct,
        "constraint_consistency": constraint,
        "evidence_coverage": coverage,
    }

    if semantic_type == "APISem":
        source = normalize_api_source(candidate.get("api_source") or metadata.get("api_source"))
        provenance = 1.0 if source in HARD_API_SOURCES else 0.80 if source in SOFT_API_SOURCES else 0.0
        template = 1.0 if any(token in text for token in ("semantic_template", "api_intent", "param_roles", "return")) else 0.0
        signals.update(provenance=provenance, template_completeness=template)
        specific = 0.70 * provenance + 0.30 * template
    elif semantic_type == "TypeSem":
        hierarchy = 1.0 if any(token in text for token in ("pointer", "pointee", "struct", "array", "function_type", "base_type")) else 0.0
        width = 1.0 if any(token in text for token in ("width", "size", "bits", "bytes")) else 0.0
        signals.update(type_hierarchy=hierarchy, width_evidence=width)
        specific = 0.65 * hierarchy + 0.35 * width
    elif semantic_type == "LayoutSem":
        offsets = 1.0 if any(token in text for token in ("offset", "field", "stride", "padding")) else 0.0
        access = 1.0 if any(token in text for token in ("access_width", "alignment", "array", "struct")) else 0.0
        signals.update(offset_pattern=offsets, access_pattern=access)
        specific = 0.65 * offsets + 0.35 * access
    else:
        calling = 1.0 if any(token in text for token in ("calling_convention", "argument_locs", "return_locs", "param_count")) else 0.0
        roles = 1.0 if any(token in text for token in ("param_roles", "param_types", "return_type", "def_use")) else 0.0
        signals.update(calling_convention=calling, parameter_roles=roles)
        specific = 0.60 * calling + 0.40 * roles

    evidence = noisy_or((direct, constraint))
    score = clamp01(0.55 * representation + 0.25 * evidence + 0.20 * specific)
    signals["module_specific"] = specific
    signals["module_score"] = score
    return score, signals


def decision_for_group(
    semantic_type: str,
    candidates: Sequence[Mapping[str, Any]],
    previous: Mapping[str, Any],
    confirmation_policy: Optional[
        Mapping[str, Mapping[str, Any]]
    ] = None,
) -> Dict[str, Any]:
    previous_status = str(previous.get("status") or "hypothesis")
    previous_target = str(previous.get("best_target_id") or "")
    # A confirmed semantic decision is irreversible in the formal mainline.
    # Exploratory revisions must be implemented as an explicit ablation in a
    # separate run, never by silently reopening a confirmed object here.
    if (
        previous_status in CONFIRMED
        or previous_status == "rejected"
        or bool(previous.get("locked_from_222_v5_5"))
        or bool(previous.get("locked_from_222_v5_6"))
    ):
        result = dict(previous)
        result["locked"] = True
        return result

    active = [candidate for candidate in candidates if candidate.get("candidate_status") != "rejected"]
    if not active:
        return {
            "semantic_type": semantic_type, "status": "rejected", "confidence": 0.0,
            "confidence_gap": 0.0, "best_target_id": None, "confirmation_level": 0,
            "confirmation_source": "all_candidates_rejected", "propagation_weight": 0.0,
            "locked": True,
        }
    ordered = sorted(active, key=lambda item: (-clamp01(item.get("score_223_v5_3", 0.0)), str(item.get("target_id") or "")))
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    score = clamp01(best.get("score_223_v5_3", 0.0))
    gap = score - clamp01(second.get("score_223_v5_3", 0.0)) if second else score
    coverage = clamp01(best.get("evidence_coverage", 0.0))
    policy = (
        confirmation_policy.get(semantic_type, POLICY[semantic_type])
        if confirmation_policy is not None
        else POLICY[semantic_type]
    )
    soft_enabled = bool(policy.get("soft_confirmation_enabled", True))
    confirmed = (
        soft_enabled
        and score >= float(policy["min_evidence"])
        and gap >= float(policy["gap"])
        and coverage >= float(policy["coverage"])
    )
    status = "confirmed_soft" if confirmed else "hypothesis"
    return {
        "semantic_type": semantic_type,
        "status": status,
        "confidence": score,
        "confidence_gap": max(0.0, gap),
        "best_target_id": best.get("target_id"),
        "confirmation_level": 2 if confirmed else 1,
        "confirmation_source": (
            "four_module_consensus"
            if confirmed
            else (
                "validation_policy_fail_closed"
                if not soft_enabled
                else "insufficient_four_module_evidence"
            )
        ),
        "soft_confirmation_enabled_223": soft_enabled,
        "propagation_weight": score if confirmed else 0.0,
        "locked": confirmed,
        "candidate_count": len(candidates),
        "previous_best_target_id": previous_target or None,
    }
