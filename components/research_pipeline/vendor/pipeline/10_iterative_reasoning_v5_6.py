#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the document-compliant Task 2.2.2 closed loop (v5.6).

This revision removes two limitations of the earlier v5.3.4 loop:

* embedding feedback is damped but bidirectional, so an unsupported candidate
  may lose confidence instead of every score being forced to increase;
* after joint-space message passing, affected samples are searched again in
  the updated embedding space, so new same-sample candidates can enter the
  hypothesis set under a bounded candidate budget.

All confirmed conclusions are immutable by default, as required by section
2.2.2.5.  Hypotheses can still gain or lose score, be rejected, and receive
new candidates.  Each round is a compact transaction: if the actual raw
S_global would fall, touched rows and embedding vectors are rolled back rather
than hiding the decrease behind a reporting envelope.  The older revisable
soft-confirmation behavior remains available only as an explicit ablation.
No gold label is read by this script.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # Allows symbolic-only execution without embedding I/O.
    torch = None
    F = None

from module_common_v5_3 import (
    CONFIRMED_STATUSES,
    CONFIRM_POLICY,
    SEMANTIC_TYPES,
    clamp01,
    cosine_to_probability,
    constraint_level,
    determine_status,
    edge_family,
    edge_kind,
    edge_satisfaction,
    edge_weight,
    endpoints,
    ensure_output_path,
    evidence_score,
    load_json,
    noisy_or,
    normalize_api_source,
    normalize_semantic_type,
    propagation_weight,
    read_jsonl,
    select_row_decision,
    summarize_statuses,
    write_json,
    apply_confirm_policy,
)


class EmbeddingLookup(Mapping):
    """Zero-copy id lookup that avoids one Python Tensor view per node."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        id_to_index: Mapping[str, int],
    ) -> None:
        self.embeddings = embeddings
        self.id_to_index = id_to_index

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.embeddings[self.id_to_index[key]]

    def __iter__(self):
        return iter(self.id_to_index)

    def __len__(self) -> int:
        return len(self.id_to_index)


def extract_index_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "items", "nodes", "index", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise RuntimeError("Index must be a JSON list or contain records/items/nodes")


def extract_embedding_tensor(payload: Any) -> Tuple[torch.Tensor, Optional[str]]:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required when --embeddings is supplied"
        )
    if isinstance(payload, torch.Tensor):
        return payload, None
    if isinstance(payload, dict):
        for key in (
            "embeddings",
            "node_embeddings",
            "features",
            "vectors",
            "tensor",
        ):
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                return value, key
    raise RuntimeError(
        "Embedding file must contain a Tensor or a dictionary Tensor field"
    )


def candidate_score(candidate: Mapping[str, Any]) -> float:
    return clamp01(
        candidate.get(
            "validated_score",
            candidate.get("confidence", 0.0),
        )
    )


def bounded_damped_update(
    old_value: float,
    proposed_value: float,
    blend: float,
    max_drop: float,
    max_rise: float,
) -> float:
    """Move toward a proposal while bounding both per-round directions."""
    old_score = clamp01(old_value)
    proposal = clamp01(proposed_value)
    damped = old_score + clamp01(blend) * (proposal - old_score)
    lower = old_score - max(0.0, float(max_drop))
    upper = old_score + max(0.0, float(max_rise))
    return clamp01(min(upper, max(lower, damped)))


def signed_message_value(message: Mapping[str, Any]) -> float:
    """Return a signed constraint contribution in [-1, 1]."""
    if "signed_contribution" in message:
        try:
            value = float(message.get("signed_contribution", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        return max(-1.0, min(1.0, value))
    return clamp01(message.get("contribution", 0.0))


def signed_to_constraint_value(value: float) -> Optional[float]:
    """Map signed support to the [0,1] consistency scale used by evidence_score."""
    signed = max(-1.0, min(1.0, float(value)))
    if abs(signed) <= 1e-12:
        return None
    # Zero is neutral, positive evidence moves toward one, and contradiction
    # moves toward zero.  This symmetric map avoids treating a weak positive
    # message as a strong contradiction merely because evidence_score accepts
    # values rather than signed weights.
    return clamp01(0.5 + 0.5 * signed)


def group_candidates(row: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in row.get("candidates", []):
        semantic_type = normalize_semantic_type(
            candidate.get("target_semantic_type") or ""
        )
        if semantic_type in SEMANTIC_TYPES:
            result[semantic_type].append(candidate)
    return result


def find_candidate(
    row: Mapping[str, Any],
    target_id: str,
) -> Optional[Dict[str, Any]]:
    for candidate in row.get("candidates", []):
        if str(candidate.get("target_id") or "") == target_id:
            return candidate
    return None


def recompute_decision(
    semantic_type: str,
    candidates: Sequence[Dict[str, Any]],
    previous: Optional[Mapping[str, Any]],
    soft_revisable: bool = True,
    soft_switch_margin: float = 0.03,
    soft_release_margin: float = 0.04,
    soft_release_gap_margin: float = 0.03,
) -> Dict[str, Any]:
    semantic_type = normalize_semantic_type(semantic_type)
    previous = previous if isinstance(previous, Mapping) else {}
    previous_status = str(previous.get("status") or "")
    previous_target = str(previous.get("best_target_id") or "")
    soft_confirmation_enabled = bool(
        CONFIRM_POLICY.get(semantic_type, {}).get(
            "soft_confirmation_enabled", True
        )
    )

    active = [
        candidate
        for candidate in candidates
        if str(candidate.get("candidate_status") or "") != "rejected"
    ]
    if not active:
        return {
            "semantic_type": semantic_type,
            "status": "rejected",
            "confidence": 0.0,
            "confidence_gap": 0.0,
            "confirmation_level": 0,
            "confirmation_source": "all_candidates_rejected",
            "propagation_weight": 0.0,
            "locked": True,
            "candidate_count": len(candidates),
            "best_target_id": None,
            "hard_seed_conflict": False,
            "policy": CONFIRM_POLICY.get(semantic_type, {}),
        }

    # Deterministic conclusions are immutable.  The legacy mode can also lock
    # soft conclusions, but v5.5 defaults to revisable soft confirmation.
    previous_is_immutable = (
        previous_status == "confirmed_hard"
        or (
            previous_status == "confirmed_soft"
            and not soft_revisable
            and soft_confirmation_enabled
        )
    )
    if previous_is_immutable and previous_target:
        locked_candidate = next(
            (
                candidate
                for candidate in candidates
                if str(candidate.get("target_id") or "") == previous_target
            ),
            None,
        )
        if locked_candidate is not None:
            result = dict(previous)
            result["confidence"] = (
                1.0 if previous_status == "confirmed_hard"
                else candidate_score(locked_candidate)
            )
            result["propagation_weight"] = (
                1.0
                if previous_status == "confirmed_hard"
                else result["confidence"]
            )
            result["locked"] = True
            result["candidate_count"] = len(candidates)
            locked_candidate["candidate_status"] = previous_status
            locked_candidate["selected_in_semantic_group"] = True
            return result

    active.sort(
        key=lambda candidate: (
            -candidate_score(candidate),
            str(candidate.get("target_id") or ""),
        )
    )

    # Hysteresis: a revisable soft winner is not replaced by a numerically
    # indistinguishable neighbour.  If it no longer clears the confirmation
    # gate it can still be released to hypothesis below.
    if (
        soft_revisable
        and soft_confirmation_enabled
        and previous_status == "confirmed_soft"
        and previous_target
    ):
        previous_candidate = next(
            (
                candidate for candidate in active
                if str(candidate.get("target_id") or "") == previous_target
            ),
            None,
        )
        if previous_candidate is not None and active:
            current_top = active[0]
            if (
                current_top is not previous_candidate
                and candidate_score(current_top)
                - candidate_score(previous_candidate)
                < max(0.0, soft_switch_margin)
            ):
                active.remove(previous_candidate)
                active.insert(0, previous_candidate)
    best = active[0]
    second = active[1] if len(active) > 1 else None
    best_score = candidate_score(best)
    second_score = candidate_score(second) if second is not None else 0.0
    gap = best_score - second_score if second is not None else best_score

    hard_candidates = [
        candidate
        for candidate in active
        if bool(candidate.get("hard_api_eligible"))
    ]
    hard_conflict = len(hard_candidates) > 1
    if len(hard_candidates) == 1:
        best = hard_candidates[0]
        best_score = 1.0
        second_score = max(
            (
                candidate_score(candidate)
                for candidate in active
                if candidate is not best
            ),
            default=0.0,
        )
        gap = 1.0 - second_score
        best["validated_score"] = 1.0
        best["confidence"] = 1.0

    source = str(best.get("api_source") or "unknown")
    status_source = source
    if (
        semantic_type == "APISem"
        and normalize_api_source(source)
        in CONFIRM_POLICY["APISem"]["hard_sources"]
        and not bool(best.get("hard_api_eligible"))
    ):
        status_source = "unknown"
    result = determine_status(
        semantic_type,
        best,
        best_score,
        gap,
        source=status_source,
        evidence_coverage=float(best.get("evidence_coverage", 0.0)),
        hard_conflict=hard_conflict,
    )

    # Release hysteresis prevents a soft decision from flapping at an exact
    # threshold while still allowing sustained negative evidence to demote it.
    if (
        soft_revisable
        and soft_confirmation_enabled
        and previous_status == "confirmed_soft"
        and previous_target == str(best.get("target_id") or "")
        and result.get("status") == "hypothesis"
        and not hard_conflict
    ):
        policy = CONFIRM_POLICY.get(semantic_type, {})
        source_allowed = (
            semantic_type != "APISem"
            or normalize_api_source(status_source)
            in set(policy.get("soft_sources", ()))
        )
        coverage = clamp01(best.get("evidence_coverage", 0.0))
        if (
            source_allowed
            and best_score
            >= max(0.0, float(policy.get("min_evidence", 1.0))
                   - max(0.0, soft_release_margin))
            and gap
            >= max(0.0, float(policy.get("gap_threshold", 1.0))
                   - max(0.0, soft_release_gap_margin))
            and coverage >= float(policy.get("min_evidence_coverage", 1.0))
        ):
            result.update({
                "status": "confirmed_soft",
                "confidence": best_score,
                "level": 2,
                "confirmation_level": 2,
                "confirmation_source": "evidence_consensus|hysteresis",
                "propagation_weight": best_score,
                "locked": True,
                "hysteresis_retained": True,
                "hysteresis_min_evidence": max(
                    0.0,
                    float(policy.get("min_evidence", 1.0))
                    - max(0.0, soft_release_margin),
                ),
                "hysteresis_min_gap": max(
                    0.0,
                    float(policy.get("gap_threshold", 1.0))
                    - max(0.0, soft_release_gap_margin),
                ),
            })
    result.update(
        {
            "semantic_type": semantic_type,
            "best_target_id": best.get("target_id"),
            "best_candidate_source": best.get("candidate_source"),
            "confidence_gap": max(0.0, gap),
            "second_score": second_score,
            "candidate_count": len(candidates),
            "hard_seed_count": len(hard_candidates),
            "hard_seed_conflict": hard_conflict,
            "policy": CONFIRM_POLICY.get(semantic_type, {}),
            "soft_confirmation_revisable_during_222": soft_revisable,
        }
    )
    for candidate in candidates:
        if str(candidate.get("candidate_status") or "") == "rejected":
            continue
        selected = str(candidate.get("target_id") or "") == str(
            best.get("target_id") or ""
        )
        candidate["selected_in_semantic_group"] = selected
        candidate["candidate_status"] = (
            str(result["status"]) if selected else "hypothesis"
        )
    return result


def refresh_row_decisions(
    row: Dict[str, Any],
    soft_revisable: bool = True,
    soft_switch_margin: float = 0.03,
    soft_release_margin: float = 0.04,
    soft_release_gap_margin: float = 0.03,
) -> int:
    previous_decisions = row.get("semantic_decisions")
    previous_decisions = (
        previous_decisions
        if isinstance(previous_decisions, Mapping)
        else {}
    )
    new_decisions: Dict[str, Dict[str, Any]] = {}
    changes = 0
    grouped = group_candidates(row)
    for semantic_type in SEMANTIC_TYPES:
        if semantic_type not in grouped:
            continue
        previous = previous_decisions.get(semantic_type, {})
        decision = recompute_decision(
            semantic_type,
            grouped[semantic_type],
            previous,
            soft_revisable=soft_revisable,
            soft_switch_margin=soft_switch_margin,
            soft_release_margin=soft_release_margin,
            soft_release_gap_margin=soft_release_gap_margin,
        )
        new_decisions[semantic_type] = decision
        if (
            str(previous.get("status") or "")
            != str(decision.get("status") or "")
            or str(previous.get("best_target_id") or "")
            != str(decision.get("best_target_id") or "")
        ):
            changes += 1
            if str(previous.get("status") or "") == "confirmed_soft":
                row["soft_confirmation_revisions"] = int(
                    row.get("soft_confirmation_revisions", 0) or 0
                ) + 1

    active_semantic_type, active = select_row_decision(
        new_decisions,
        row.get("evidence_type"),
    )
    active_target = str(active.get("best_target_id") or "")
    best_candidate = find_candidate(row, active_target)
    row["semantic_decisions"] = new_decisions
    row["active_semantic_type"] = active_semantic_type
    row["status"] = str(active.get("status") or "hypothesis")
    row["confidence"] = clamp01(active.get("confidence", 0.0))
    row["confidence_gap"] = max(
        0.0, float(active.get("confidence_gap", 0.0))
    )
    row["confirmation_source"] = active.get("confirmation_source")
    row["confirmation_level"] = int(
        active.get("confirmation_level", 1)
    )
    row["propagation_weight"] = clamp01(
        active.get("propagation_weight", 0.0)
    )
    row["locked"] = bool(active.get("locked", False))
    row["best_candidate"] = best_candidate
    row["preferred_target_id"] = (
        best_candidate.get("target_id")
        if isinstance(best_candidate, Mapping)
        else None
    )
    row["candidates"].sort(
        key=lambda candidate: (
            str(candidate.get("target_id") or "") != active_target,
            -candidate_score(candidate),
            str(candidate.get("target_id") or ""),
        )
    )
    return changes


def build_occurrences(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, List[Tuple[int, Dict[str, Any]]]]:
    occurrences: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(
        list
    )
    for row_index, row in enumerate(rows):
        for candidate in row.get("candidates", []):
            target_id = str(candidate.get("target_id") or "")
            if target_id:
                occurrences[target_id].append((row_index, candidate))
    return occurrences


def confirmed_nodes(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[int]]]:
    states: Dict[str, Dict[str, Any]] = {}
    source_rows: Dict[str, List[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        decisions = row.get("semantic_decisions")
        if not isinstance(decisions, Mapping):
            continue
        for semantic_type, decision in decisions.items():
            if not isinstance(decision, Mapping):
                continue
            weight = propagation_weight(decision)
            target_id = str(decision.get("best_target_id") or "")
            if not target_id or weight <= 0:
                continue
            source_rows[target_id].append(row_index)
            previous = states.get(target_id)
            state = {
                "target_id": target_id,
                "semantic_type": normalize_semantic_type(semantic_type),
                "status": str(decision.get("status") or ""),
                "confidence": clamp01(decision.get("confidence", 0.0)),
                "propagation_weight": weight,
                "confirmation_source": decision.get(
                    "confirmation_source"
                ),
            }
            if (
                previous is None
                or state["propagation_weight"]
                > previous["propagation_weight"]
            ):
                states[target_id] = state
    return states, source_rows


def evidence_component_values(
    candidate: Mapping[str, Any],
    propagation_values: Sequence[float],
) -> Tuple[List[float], List[float], List[float]]:
    observed = candidate.get("component_observed")
    observed = observed if isinstance(observed, Mapping) else {}
    direct = (
        [clamp01(candidate.get("direct_support_score", 0.0))]
        if bool(observed.get("direct_support"))
        else []
    )
    constraint = (
        [clamp01(candidate.get("base_constraint_score", candidate.get(
            "constraint_consistency_score", 0.0
        )))]
        if bool(observed.get("constraint_consistency"))
        else []
    )
    for value in propagation_values:
        converted = signed_to_constraint_value(value)
        if converted is not None:
            constraint.append(converted)
    collaboration = (
        [
            clamp01(
                candidate.get("cross_object_collaboration_score", 0.0)
            )
        ]
        if bool(observed.get("cross_object_collaboration"))
        else []
    )
    return direct, constraint, collaboration


def make_derived_candidate(
    row: Mapping[str, Any],
    target_id: str,
    target_semantic_type: str,
    messages: Sequence[Mapping[str, Any]],
    embedding_by_id: Optional[Mapping[str, torch.Tensor]],
) -> Dict[str, Any]:
    propagation_values = [signed_message_value(message) for message in messages]
    positive_support = [value for value in propagation_values if value > 0.0]
    constraint_values = [
        converted
        for converted in (
            signed_to_constraint_value(value) for value in propagation_values
        )
        if converted is not None
    ]
    propagation_support = noisy_or(positive_support)
    query_id = str(row.get("query_id") or "")
    representation = 0.5 * propagation_support
    cosine = None
    if (
        embedding_by_id is not None
        and query_id in embedding_by_id
        and target_id in embedding_by_id
    ):
        cosine = float(
            embedding_by_id[query_id] @ embedding_by_id[target_id]
        )
        representation = clamp01((cosine + 1.0) / 2.0)

    score = evidence_score(
        representation,
        [],
        constraint_values,
        [],
    )
    return {
        "target_id": target_id,
        "target_semantic_type": target_semantic_type,
        "candidate_source": "cross_layer_derived",
        "same_sample": True,
        "embedding_cosine": cosine,
        "representation_score": score["representation_score"],
        "direct_edge_count": 0,
        "direct_edge_weight": 0.0,
        "direct_support_score": score["direct_support_score"],
        "base_constraint_score": 0.0,
        "constraint_consistency_score": score[
            "constraint_consistency_score"
        ],
        "cross_object_collaboration_score": score[
            "cross_object_collaboration_score"
        ],
        "evidence_core_score": score["evidence_core_score"],
        "evidence_coverage": score["evidence_coverage"],
        "validated_score": score["validated_score"],
        "confidence": score["validated_score"],
        "component_observed": score["component_observed"],
        "component_weights": score["component_weights"],
        "representation_prior_weight": score[
            "representation_prior_weight"
        ],
        "api_source": "unknown",
        "hard_api_eligible": False,
        "candidate_status": "hypothesis",
        "propagation_support": propagation_support,
        "propagation_evidence": [dict(message) for message in messages[:16]],
    }


def update_candidate_from_messages(
    candidate: Dict[str, Any],
    messages: Sequence[Mapping[str, Any]],
    representation_step: float,
    score_blend: float,
    max_score_drop: float,
    max_score_rise: float,
) -> float:
    if (
        str(candidate.get("candidate_status") or "")
        in ("rejected", "confirmed_hard")
        or bool(candidate.get("hard_api_eligible"))
    ):
        return 0.0
    propagation_values = [signed_message_value(message) for message in messages]
    if not any(abs(value) > 1e-12 for value in propagation_values):
        return 0.0
    if "base_constraint_score" not in candidate:
        candidate["base_constraint_score"] = clamp01(
            candidate.get("constraint_consistency_score", 0.0)
        )

    old_representation = clamp01(
        candidate.get(
            "representation_score",
            candidate.get("confidence", 0.0),
        )
    )
    positive_sources = [
        (
            clamp01(message.get("source_confidence", 0.0)),
            max(0.0, signed_message_value(message)),
        )
        for message in messages
        if signed_message_value(message) > 0.0
    ]
    total = sum(weight for _, weight in positive_sources)
    source_mean = (
        sum(value * weight for value, weight in positive_sources) / total
        if total > 0
        else old_representation
    )
    beta = clamp01(
        representation_step
        * noisy_or([value for value in propagation_values if value > 0.0])
    )
    proposed_representation = (
        (1.0 - beta) * old_representation + beta * source_mean
    )
    representation = bounded_damped_update(
        old_representation,
        proposed_representation,
        1.0,
        max_score_drop,
        max_score_rise,
    )

    direct, constraint, collaboration = evidence_component_values(
        candidate,
        propagation_values,
    )
    score = evidence_score(
        representation,
        direct,
        constraint,
        collaboration,
        prior_weight=float(
            candidate.get("representation_prior_weight", 0.25)
        ),
    )
    old_score = candidate_score(candidate)
    proposed_score = float(score["validated_score"])
    propagation_support = noisy_or(
        [value for value in propagation_values if value > 0.0]
    )
    propagation_contradiction = noisy_or(
        [abs(value) for value in propagation_values if value < 0.0]
    )
    if propagation_contradiction > propagation_support:
        # Dominant contradiction is directional: recomputing other evidence
        # must not accidentally turn a negative symbolic message into a score
        # increase.  The outer bounded update still limits the per-round drop.
        net_contradiction = (
            propagation_contradiction - propagation_support
        )
        proposed_score = min(
            proposed_score,
            old_score * (1.0 - net_contradiction),
        )
    new_score = bounded_damped_update(
        old_score,
        proposed_score,
        score_blend,
        max_score_drop,
        max_score_rise,
    )
    candidate.update(score)
    candidate["representation_score"] = representation
    candidate["validated_score"] = new_score
    candidate["confidence"] = new_score
    candidate["propagation_support"] = propagation_support
    candidate["propagation_contradiction"] = propagation_contradiction
    candidate["propagation_evidence"] = [
        dict(message) for message in messages[:16]
    ]
    return new_score - old_score


def score_delta_summary(deltas: Sequence[float]) -> Dict[str, float]:
    increases = [value for value in deltas if value > 1e-12]
    decreases = [-value for value in deltas if value < -1e-12]
    return {
        "rescored_candidates": len(deltas),
        "increased_candidates": len(increases),
        "decreased_candidates": len(decreases),
        "mean_signed_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "mean_abs_delta": (
            sum(abs(value) for value in deltas) / len(deltas)
            if deltas else 0.0
        ),
        "max_increase": max(increases) if increases else 0.0,
        "max_decrease": max(decreases) if decreases else 0.0,
    }


def refresh_candidate_representation_scores(
    rows: Sequence[Dict[str, Any]],
    embedding_by_id: Mapping[str, torch.Tensor],
    blend: float,
    score_blend: float,
    max_score_drop: float,
    max_score_rise: float,
    affected_target_ids: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Feed updated embeddings back with auditable up/down score movement."""
    alpha = clamp01(blend)
    if alpha <= 0:
        return score_delta_summary([])
    deltas: List[float] = []
    for row in rows:
        query_id = str(row.get("query_id") or "")
        if query_id not in embedding_by_id:
            continue
        for candidate in row.get("candidates", []):
            if str(candidate.get("candidate_status") or "") in (
                "rejected", "confirmed_hard"
            ) or bool(candidate.get("hard_api_eligible")):
                continue
            target_id = str(candidate.get("target_id") or "")
            if target_id not in embedding_by_id:
                continue
            if (
                affected_target_ids is not None
                and target_id not in affected_target_ids
                and query_id not in affected_target_ids
            ):
                continue
            cosine = float(
                embedding_by_id[query_id] @ embedding_by_id[target_id]
            )
            representation = clamp01((cosine + 1.0) / 2.0)
            old_representation = clamp01(
                candidate.get("representation_score", 0.0)
            )
            blended_representation = (
                (1.0 - alpha) * old_representation
                + alpha * representation
            )
            propagation = [
                signed_message_value(item)
                for item in candidate.get("propagation_evidence", []) or []
                if isinstance(item, Mapping)
            ]
            direct, constraint, collaboration = evidence_component_values(
                candidate, propagation
            )
            score = evidence_score(
                blended_representation,
                direct,
                constraint,
                collaboration,
                prior_weight=float(
                    candidate.get("representation_prior_weight", 0.25)
                ),
            )
            old_score = candidate_score(candidate)
            new_score = bounded_damped_update(
                old_score,
                float(score["validated_score"]),
                score_blend,
                max_score_drop,
                max_score_rise,
            )
            candidate.update(score)
            candidate["embedding_cosine_after_message_passing"] = cosine
            candidate["representation_score"] = blended_representation
            candidate["validated_score"] = new_score
            candidate["confidence"] = new_score
            candidate["last_embedding_rescore_delta"] = new_score - old_score
            deltas.append(new_score - old_score)
    return score_delta_summary(deltas)


def calculate_global_consistency(
    rows: Sequence[Mapping[str, Any]],
    evidence_edges: Sequence[Mapping[str, Any]],
    constraints: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    pair_scores: Dict[Tuple[str, str], float] = {}
    node_scores: Dict[str, float] = {}
    node_statuses: Dict[str, str] = {}

    for row in rows:
        query_id = str(row.get("query_id") or "")
        for candidate in row.get("candidates", []):
            if str(candidate.get("candidate_status") or "") == "rejected":
                continue
            target_id = str(candidate.get("target_id") or "")
            score = candidate_score(candidate)
            pair_scores[(query_id, target_id)] = max(
                pair_scores.get((query_id, target_id), 0.0),
                score,
            )
            node_scores[target_id] = max(
                node_scores.get(target_id, 0.0),
                score,
            )
        decisions = row.get("semantic_decisions")
        if not isinstance(decisions, Mapping):
            continue
        for decision in decisions.values():
            if not isinstance(decision, Mapping):
                continue
            target_id = str(decision.get("best_target_id") or "")
            if not target_id:
                continue
            node_scores[target_id] = max(
                node_scores.get(target_id, 0.0),
                clamp01(decision.get("confidence", 0.0)),
            )
            if propagation_weight(decision) > 0:
                node_statuses[target_id] = str(
                    decision.get("status") or ""
                )

    evidence_numerator = 0.0
    evidence_denominator = 0.0
    for edge in evidence_edges:
        weight = clamp01(edge.get("weight", 0.0))
        support = pair_scores.get(
            (
                str(edge.get("source") or ""),
                str(edge.get("target") or ""),
            ),
            0.0,
        )
        evidence_numerator += weight * support
        evidence_denominator += weight

    constraint_numerator = 0.0
    constraint_denominator = 0.0
    for constraint in constraints:
        weight = clamp01(constraint.get("weight", 0.0))
        source = str(constraint.get("source") or "")
        target = str(constraint.get("target") or "")
        explicit = constraint.get("explicit_satisfaction")
        if explicit is not None:
            satisfaction = clamp01(explicit)
        elif source in node_statuses:
            satisfaction = node_scores.get(target, 0.0)
        else:
            source_score = node_scores.get(source, 0.0)
            target_score = node_scores.get(target, 0.0)
            satisfaction = min(source_score, target_score)
        constraint_numerator += weight * clamp01(satisfaction)
        constraint_denominator += weight

    denominator = evidence_denominator + constraint_denominator
    raw = (
        (evidence_numerator + constraint_numerator) / denominator
        if denominator > 0
        else 0.0
    )
    return {
        "raw_global_consistency": clamp01(raw),
        "evidence_consistency": (
            evidence_numerator / evidence_denominator
            if evidence_denominator > 0
            else 0.0
        ),
        "constraint_consistency": (
            constraint_numerator / constraint_denominator
            if constraint_denominator > 0
            else 0.0
        ),
        "evidence_weight": evidence_denominator,
        "constraint_weight": constraint_denominator,
    }


def update_embeddings(
    embeddings: torch.Tensor,
    id_to_index: Mapping[str, int],
    messages_by_target: Mapping[str, Sequence[Mapping[str, Any]]],
    step: float,
) -> Tuple[Dict[str, float], Set[str]]:
    if torch is None or F is None:
        raise RuntimeError(
            "PyTorch is required for joint-space embedding updates"
        )
    deltas: List[float] = []
    updates = 0
    updated_target_ids: Set[str] = set()
    old_embeddings = embeddings.clone()
    for target_id, messages in messages_by_target.items():
        target_index = id_to_index.get(target_id)
        if target_index is None:
            continue
        signed_deltas = []
        absolute_weights = []
        target_vector = old_embeddings[target_index]
        for message in messages:
            source_index = id_to_index.get(
                str(message.get("source_target_id") or "")
            )
            signed_weight = signed_message_value(message)
            if source_index is None or abs(signed_weight) <= 1e-12:
                continue
            # Positive constraints attract; negative constraints repel.  The
            # update is computed from the frozen beginning-of-round tensor so
            # row ordering cannot change the result.
            signed_deltas.append(
                signed_weight
                * (old_embeddings[source_index] - target_vector)
            )
            absolute_weights.append(abs(signed_weight))
        if not signed_deltas:
            continue
        mean_delta = torch.stack(signed_deltas, dim=0).sum(dim=0) / max(
            sum(absolute_weights), 1e-12
        )
        beta = clamp01(step * noisy_or(absolute_weights))
        proposed = target_vector + beta * mean_delta
        proposed = F.normalize(proposed.unsqueeze(0), dim=-1).squeeze(0)
        delta = float(
            torch.linalg.vector_norm(
                proposed - old_embeddings[target_index]
            )
        )
        embeddings[target_index] = proposed
        deltas.append(delta)
        updates += 1
        updated_target_ids.add(target_id)
    return {
        "updated_nodes": updates,
        "mean_embedding_delta": (
            sum(deltas) / len(deltas) if deltas else 0.0
        ),
        "max_embedding_delta": max(deltas) if deltas else 0.0,
        "attractive_messages": sum(
            1 for values in messages_by_target.values() for message in values
            if signed_message_value(message) > 0.0
        ),
        "repulsive_messages": sum(
            1 for values in messages_by_target.values() for message in values
            if signed_message_value(message) < 0.0
        ),
    }, updated_target_ids


def resolve_device(requested: str) -> str:
    if torch is None:
        raise RuntimeError("PyTorch is required for iterative retrieval")
    value = str(requested or "auto").lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA retrieval requested but CUDA is unavailable")
    return value


def topk_chunked(
    query_matrix: torch.Tensor,
    target_indices: Sequence[int],
    embeddings: torch.Tensor,
    top_k: int,
    target_batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Memory-bounded cosine retrieval using the current normalized tensor."""
    if not target_indices or top_k <= 0:
        shape = (query_matrix.shape[0], 0)
        return torch.empty(shape), torch.empty(shape, dtype=torch.long)
    keep = min(int(top_k), len(target_indices))
    best_values = torch.empty(
        (query_matrix.shape[0], 0), dtype=query_matrix.dtype, device=device
    )
    best_indices = torch.empty(
        (query_matrix.shape[0], 0), dtype=torch.long, device=device
    )
    for start in range(0, len(target_indices), target_batch_size):
        chunk = list(target_indices[start : start + target_batch_size])
        target_matrix = embeddings[chunk].to(device)
        similarities = query_matrix @ target_matrix.T
        local_keep = min(keep, similarities.shape[1])
        values, locations = torch.topk(similarities, k=local_keep, dim=1)
        global_indices = torch.tensor(
            chunk, dtype=torch.long, device=device
        )[locations]
        combined_values = torch.cat((best_values, values), dim=1)
        combined_indices = torch.cat((best_indices, global_indices), dim=1)
        combined_keep = min(keep, combined_values.shape[1])
        best_values, selected = torch.topk(
            combined_values, k=combined_keep, dim=1
        )
        best_indices = torch.gather(combined_indices, 1, selected)
    return best_values.cpu(), best_indices.cpu()


def make_retrieved_candidate(
    target_id: str,
    semantic_type: str,
    cosine: float,
    iteration: int,
) -> Dict[str, Any]:
    representation = cosine_to_probability(cosine)
    score = evidence_score(representation, [], [], [])
    return {
        "target_id": target_id,
        "target_semantic_type": semantic_type,
        "candidate_source": "iterative_embedding_retrieval",
        "same_sample": True,
        "embedding_cosine": cosine,
        "embedding_cosine_after_message_passing": cosine,
        "representation_score": representation,
        "direct_edge_count": 0,
        "direct_edge_weight": 0.0,
        "direct_support_score": score["direct_support_score"],
        "base_constraint_score": 0.0,
        "constraint_consistency_score": score[
            "constraint_consistency_score"
        ],
        "cross_object_collaboration_score": score[
            "cross_object_collaboration_score"
        ],
        "evidence_core_score": score["evidence_core_score"],
        "evidence_coverage": score["evidence_coverage"],
        "validated_score": score["validated_score"],
        "confidence": score["validated_score"],
        "component_observed": score["component_observed"],
        "component_weights": score["component_weights"],
        "representation_prior_weight": score[
            "representation_prior_weight"
        ],
        "api_source": "unknown",
        "hard_api_eligible": False,
        "candidate_status": "hypothesis",
        "retrieved_iteration": iteration,
        "retrieval_history": [
            {"iteration": iteration, "cosine": cosine}
        ],
        "propagation_evidence": [],
    }


def candidate_is_protected(
    candidate: Mapping[str, Any], protected_targets: Set[str]
) -> bool:
    target_id = str(candidate.get("target_id") or "")
    source = str(candidate.get("candidate_source") or "")
    rejection = str(candidate.get("rejection_reason") or "")
    return (
        target_id in protected_targets
        or bool(candidate.get("hard_api_eligible"))
        or int(candidate.get("direct_edge_count", 0) or 0) > 0
        or source in ("usg_direct_evidence", "cross_layer_derived")
        or str(candidate.get("candidate_status") or "") == "confirmed_hard"
        or "deterministic" in rejection
    )


def reretrieve_candidates(
    rows: Sequence[Dict[str, Any]],
    embeddings: torch.Tensor,
    id_to_index: Mapping[str, int],
    node_id_by_index: Sequence[str],
    sample_rows: Mapping[str, Sequence[int]],
    sample_targets: Mapping[str, Mapping[str, Sequence[int]]],
    affected_samples: Set[str],
    iteration: int,
    top_k_per_type: int,
    max_candidates_per_type: int,
    min_probability: float,
    replacement_margin: float,
    query_batch_size: int,
    target_batch_size: int,
    device: str,
) -> Dict[str, Any]:
    """Search affected samples again and maintain a bounded hypothesis set."""
    counters: Counter = Counter()
    if top_k_per_type <= 0 or not affected_samples:
        return dict(counters)
    for sample_id in sorted(affected_samples):
        row_indices = list(sample_rows.get(sample_id, ()))
        targets = sample_targets.get(sample_id, {})
        if not row_indices or not targets:
            continue
        counters["affected_samples"] += 1
        for semantic_type in SEMANTIC_TYPES:
            target_indices = list(targets.get(semantic_type, ()))
            if not target_indices:
                continue
            eligible_rows = []
            for row_index in row_indices:
                row = rows[row_index]
                decision = (
                    row.get("semantic_decisions", {}).get(semantic_type, {})
                    if isinstance(row.get("semantic_decisions"), Mapping)
                    else {}
                )
                if str(decision.get("status") or "") in (
                    "confirmed_hard", "rejected"
                ):
                    continue
                query_id = str(row.get("query_id") or "")
                if query_id in id_to_index:
                    eligible_rows.append(row_index)
            for start in range(0, len(eligible_rows), query_batch_size):
                batch_rows = eligible_rows[start : start + query_batch_size]
                query_indices = [
                    id_to_index[str(rows[index].get("query_id") or "")]
                    for index in batch_rows
                ]
                query_matrix = embeddings[query_indices].to(device)
                values, indices = topk_chunked(
                    query_matrix,
                    target_indices,
                    embeddings,
                    top_k_per_type,
                    target_batch_size,
                    device,
                )
                for offset, row_index in enumerate(batch_rows):
                    row = rows[row_index]
                    existing = {
                        str(candidate.get("target_id") or ""): candidate
                        for candidate in row.get("candidates", [])
                    }
                    group = [
                        candidate for candidate in row.get("candidates", [])
                        if normalize_semantic_type(
                            candidate.get("target_semantic_type")
                        ) == semantic_type
                    ]
                    protected_targets = {
                        str(decision.get("best_target_id") or "")
                        for decision in (
                            row.get("semantic_decisions", {}).values()
                            if isinstance(row.get("semantic_decisions"), Mapping)
                            else ()
                        )
                        if isinstance(decision, Mapping)
                        and decision.get("best_target_id")
                    }
                    for cosine_tensor, target_tensor in zip(
                        values[offset], indices[offset]
                    ):
                        target_index = int(target_tensor)
                        target_id = node_id_by_index[target_index]
                        if not target_id:
                            continue
                        cosine = float(cosine_tensor)
                        probability = cosine_to_probability(cosine)
                        present = existing.get(target_id)
                        if present is not None:
                            history = present.setdefault("retrieval_history", [])
                            history.append({"iteration": iteration, "cosine": cosine})
                            if len(history) > 16:
                                del history[:-16]
                            present["last_retrieved_iteration"] = iteration
                            counters["existing_candidates_retrieved"] += 1
                            continue
                        if probability < min_probability:
                            counters["below_probability_floor"] += 1
                            continue
                        removable = [
                            candidate for candidate in group
                            if not candidate_is_protected(
                                candidate, protected_targets
                            )
                        ]
                        if len(group) >= max_candidates_per_type and removable:
                            weakest = min(
                                removable,
                                key=lambda item: (
                                    clamp01(item.get("representation_score", 0.0)),
                                    str(item.get("target_id") or ""),
                                ),
                            )
                            weakest_representation = clamp01(
                                weakest.get("representation_score", 0.0)
                            )
                            if probability < (
                                weakest_representation + replacement_margin
                            ):
                                counters["below_replacement_margin"] += 1
                                continue
                        candidate = make_retrieved_candidate(
                            target_id, semantic_type, cosine, iteration
                        )
                        row["candidates"].append(candidate)
                        group.append(candidate)
                        existing[target_id] = candidate
                        counters["new_candidates"] += 1

                    # Enforce the budget only on removable neural candidates;
                    # direct evidence, deterministic decisions and derived
                    # symbolic hypotheses are never discarded.
                    removable = [
                        candidate for candidate in group
                        if not candidate_is_protected(candidate, protected_targets)
                    ]
                    excess = max(0, len(group) - max_candidates_per_type)
                    if excess > 0 and removable:
                        removable.sort(
                            key=lambda item: (
                                candidate_score(item),
                                clamp01(item.get("representation_score", 0.0)),
                                str(item.get("target_id") or ""),
                            )
                        )
                        removed = removable[: min(excess, len(removable))]
                        removed_ids = {
                            str(candidate.get("target_id") or "")
                            for candidate in removed
                        }
                        row["candidates"] = [
                            candidate for candidate in row["candidates"]
                            if str(candidate.get("target_id") or "")
                            not in removed_ids
                        ]
                        audit = row.setdefault("candidate_pruning_audit", [])
                        audit.append({
                            "iteration": iteration,
                            "semantic_type": semantic_type,
                            "target_ids": sorted(removed_ids)[:16],
                        })
                        if len(audit) > 16:
                            del audit[:-16]
                        counters["pruned_candidates"] += len(removed_ids)
                    counters["query_type_searches"] += 1
    return dict(counters)


def save_embedding_payload(
    path: str,
    original_payload: Any,
    tensor_key: Optional[str],
    embeddings: torch.Tensor,
    overwrite: bool,
) -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for joint-space embedding updates"
        )
    output = ensure_output_path(path, overwrite=overwrite)
    if tensor_key is None:
        payload = embeddings
    else:
        payload = copy.copy(original_payload)
        payload[tensor_key] = embeddings
    torch.save(payload, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--embeddings")
    parser.add_argument("--embedding-out")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--global-threshold", type=float, default=0.85)
    parser.add_argument("--stable-rounds", type=int, default=2)
    parser.add_argument("--representation-step", type=float, default=0.10)
    parser.add_argument(
        "--embedding-rescore-blend",
        type=float,
        default=0.25,
        help=(
            "Blend updated joint-space cosine back into unlocked candidate "
            "representation scores after each propagation round."
        ),
    )
    parser.add_argument(
        "--score-feedback-blend",
        type=float,
        default=0.35,
        help="Damped neural/symbolic score update strength per round.",
    )
    parser.add_argument(
        "--max-score-drop-per-round",
        type=float,
        default=0.08,
        help="Maximum confidence decrease caused by one feedback round.",
    )
    parser.add_argument(
        "--max-score-rise-per-round",
        type=float,
        default=0.08,
        help="Maximum confidence increase caused by one feedback round.",
    )
    parser.add_argument(
        "--legacy-monotonic-score-updates",
        action="store_true",
        help=(
            "Ablation/compatibility mode: prevent score decreases and lock "
            "soft confirmations as in the earlier monotonic implementation."
        ),
    )
    parser.set_defaults(lock_soft_confirmations=True)
    parser.add_argument(
        "--lock-soft-confirmations",
        dest="lock_soft_confirmations",
        action="store_true",
        help="Keep confirmed_soft immutable (document-compliant default).",
    )
    parser.add_argument(
        "--allow-soft-confirmation-revisions",
        dest="lock_soft_confirmations",
        action="store_false",
        help=(
            "Experimental ablation only: allow confirmed_soft to switch or "
            "return to hypothesis. This disables the document's irreversible "
            "confirmation premise."
        ),
    )
    parser.set_defaults(enforce_raw_monotonic=True)
    parser.add_argument(
        "--enforce-raw-monotonic",
        dest="enforce_raw_monotonic",
        action="store_true",
        help="Rollback a proposal round when actual raw S_global falls (default).",
    )
    parser.add_argument(
        "--allow-raw-consistency-decrease",
        dest="enforce_raw_monotonic",
        action="store_false",
        help="Diagnostic ablation; permits actual raw S_global decreases.",
    )
    parser.add_argument("--soft-switch-margin", type=float, default=0.03)
    parser.add_argument("--soft-release-margin", type=float, default=0.04)
    parser.add_argument(
        "--soft-release-gap-margin", type=float, default=0.03
    )
    parser.set_defaults(iterative_retrieval=True)
    parser.add_argument(
        "--iterative-retrieval",
        dest="iterative_retrieval",
        action="store_true",
        help="Re-query the updated embedding space after each round (default).",
    )
    parser.add_argument(
        "--no-iterative-retrieval",
        dest="iterative_retrieval",
        action="store_false",
        help="Ablation: only update/re-rank the initial candidate set.",
    )
    parser.add_argument("--retrieval-interval", type=int, default=1)
    parser.add_argument("--retrieval-top-k-per-type", type=int, default=5)
    parser.add_argument(
        "--retrieval-max-candidates-per-type", type=int, default=6
    )
    parser.add_argument(
        "--retrieval-min-probability", type=float, default=0.55
    )
    parser.add_argument(
        "--retrieval-replacement-margin", type=float, default=0.02
    )
    parser.add_argument(
        "--retrieval-query-batch-size", type=int, default=256
    )
    parser.add_argument(
        "--retrieval-target-batch-size", type=int, default=8192
    )
    parser.add_argument(
        "--retrieval-device",
        default="auto",
        help="auto, cpu, cuda, or cuda:N; CUDA_VISIBLE_DEVICES is respected.",
    )
    parser.add_argument(
        "--confirm-policy",
        help="Validation-calibrated soft-confirmation policy JSON.",
    )
    parser.add_argument(
        "--min-global-delta",
        type=float,
        default=1e-6,
        help="Convergence tolerance used together with low state change rate.",
    )
    parser.add_argument(
        "--max-state-change-rate-for-stable",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    active_policy = apply_confirm_policy(args.confirm_policy)

    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive")
    if args.stable_rounds <= 0:
        raise ValueError("--stable-rounds must be positive")
    if not 0.0 <= args.global_threshold <= 1.0:
        raise ValueError("--global-threshold must be in [0, 1]")
    if not 0.0 <= args.representation_step <= 1.0:
        raise ValueError("--representation-step must be in [0, 1]")
    if not 0.0 <= args.embedding_rescore_blend <= 1.0:
        raise ValueError("--embedding-rescore-blend must be in [0, 1]")
    if not 0.0 <= args.score_feedback_blend <= 1.0:
        raise ValueError("--score-feedback-blend must be in [0, 1]")
    for name in (
        "max_score_drop_per_round",
        "max_score_rise_per_round",
        "soft_switch_margin",
        "soft_release_margin",
        "soft_release_gap_margin",
        "retrieval_replacement_margin",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError("--%s must be non-negative" % name.replace("_", "-"))
    if not 0.0 <= args.retrieval_min_probability <= 1.0:
        raise ValueError("--retrieval-min-probability must be in [0, 1]")
    for name in (
        "retrieval_interval",
        "retrieval_top_k_per_type",
        "retrieval_max_candidates_per_type",
        "retrieval_query_batch_size",
        "retrieval_target_batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError("--%s must be positive" % name.replace("_", "-"))
    if (
        args.retrieval_max_candidates_per_type
        < args.retrieval_top_k_per_type
    ):
        raise ValueError(
            "--retrieval-max-candidates-per-type must be >= "
            "--retrieval-top-k-per-type"
        )
    if args.min_global_delta < 0:
        raise ValueError("--min-global-delta must be non-negative")
    if not 0.0 <= args.max_state_change_rate_for_stable <= 1.0:
        raise ValueError(
            "--max-state-change-rate-for-stable must be in [0, 1]"
        )
    if bool(args.embeddings) != bool(args.embedding_out):
        raise RuntimeError(
            "--embeddings and --embedding-out must be supplied together"
        )
    if args.legacy_monotonic_score_updates:
        args.max_score_drop_per_round = 0.0
        args.lock_soft_confirmations = True
        args.iterative_retrieval = False
    if args.iterative_retrieval and not args.embeddings:
        raise RuntimeError(
            "Iterative retrieval requires --embeddings/--embedding-out; "
            "use --no-iterative-retrieval for symbolic-only execution."
        )

    rows = [dict(row) for row in read_jsonl(args.validated)]
    index = extract_index_records(load_json(args.index))
    id_to_index: Dict[str, int] = {}
    sample_by_id: Dict[str, str] = {}
    semantic_type_by_id: Dict[str, str] = {}
    node_id_by_index: List[str] = [""] * len(index)
    sample_targets: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for number, record in enumerate(index):
        identity = str(record.get("node_id") or record.get("id") or "")
        if not identity:
            continue
        node_id_by_index[number] = identity
        id_to_index[identity] = number
        sample_id = str(
            record.get("sample_id")
            or record.get("binary_id")
            or record.get("program_id")
            or ""
        )
        semantic_type = normalize_semantic_type(
            record.get("node_type") or record.get("semantic_type") or ""
        )
        sample_by_id[identity] = sample_id
        semantic_type_by_id[identity] = semantic_type
        if sample_id and semantic_type in SEMANTIC_TYPES:
            sample_targets[sample_id][semantic_type].append(number)

    required_node_ids = {
        str(row.get("query_id") or "")
        for row in rows
        if row.get("query_id")
    }
    required_node_ids.update(
        str(candidate.get("target_id") or "")
        for row in rows
        for candidate in row.get("candidates", [])
        if candidate.get("target_id")
    )
    missing_index_nodes = sorted(required_node_ids - set(id_to_index))
    if missing_index_nodes:
        raise RuntimeError(
            "Reasoning nodes missing from index: %d (first: %s)"
            % (
                len(missing_index_nodes),
                ", ".join(missing_index_nodes[:10]),
            )
        )

    relevant_samples = {
        str(row.get("sample_id") or "") for row in rows
    }
    if "" in relevant_samples:
        raise RuntimeError(
            "sample_id is required on every validated reasoning row"
        )
    sample_rows: Dict[str, List[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        sample_rows[str(row.get("sample_id") or "")].append(row_index)
    candidate_pairs = {
        (
            str(row.get("query_id") or ""),
            str(candidate.get("target_id") or ""),
        )
        for row in rows
        for candidate in row.get("candidates", [])
        if row.get("query_id") and candidate.get("target_id")
    }

    constraints: List[Dict[str, Any]] = []
    adjacency: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    evidence_edges: List[Dict[str, Any]] = []
    edges_seen = 0
    for edge in read_jsonl(args.edges):
        edges_seen += 1
        source, target = endpoints(edge)
        if not source or not target:
            continue
        source_type = semantic_type_by_id.get(source, "")
        target_type = semantic_type_by_id.get(target, "")
        family = edge_family(edge)
        if (source, target) in candidate_pairs and family != "constraint":
            evidence_edges.append(
                {
                    "source": source,
                    "target": target,
                    "weight": edge_weight(edge),
                    "kind": edge_kind(edge),
                }
            )

        high_to_high = (
            source_type in SEMANTIC_TYPES
            and target_type in SEMANTIC_TYPES
        )
        same_sample = (
            sample_by_id.get(source) == sample_by_id.get(target)
            and sample_by_id.get(source) in relevant_samples
        )
        if not high_to_high or not same_sample:
            continue
        if family not in ("constraint", "unknown"):
            continue
        record = {
            "source": source,
            "target": target,
            "source_semantic_type": source_type,
            "target_semantic_type": target_type,
            "sample_id": sample_by_id.get(source, ""),
            "weight": edge_weight(edge),
            "kind": edge_kind(edge),
            "level": constraint_level(edge, source_type),
            "explicit_satisfaction": edge_satisfaction(edge),
        }
        constraints.append(record)
        adjacency[source].append(record)

        if edges_seen % 1000000 == 0:
            print(
                json.dumps(
                    {
                        "phase": "scan_edges",
                        "edges_seen": edges_seen,
                        "constraint_edges": len(constraints),
                        "evidence_edges": len(evidence_edges),
                    }
                ),
                flush=True,
            )

    embedding_payload = None
    embedding_tensor_key = None
    embeddings = None
    embedding_by_id = None
    retrieval_device = "disabled"
    if args.embeddings:
        if torch is None or F is None:
            raise RuntimeError(
                "PyTorch is required when --embeddings is supplied"
            )
        embedding_payload = torch.load(
            args.embeddings,
            map_location="cpu",
        )
        embedding_tensor, embedding_tensor_key = extract_embedding_tensor(
            embedding_payload
        )
        if len(embedding_tensor) != len(index):
            raise RuntimeError(
                "Index/embedding length mismatch: %d != %d"
                % (len(index), len(embedding_tensor))
            )
        embeddings = F.normalize(embedding_tensor.float().clone(), dim=-1)
        embedding_by_id = EmbeddingLookup(embeddings, id_to_index)
        if args.iterative_retrieval:
            retrieval_device = resolve_device(args.retrieval_device)

    for row in rows:
        row["schema"] = "task2_222_iterative_reasoning_v5_3"
        row.setdefault("status_history", [])
        row["status_history"].append(
            {
                "iteration": 0,
                "status": row.get("status"),
                "active_semantic_type": row.get(
                    "active_semantic_type"
                ),
                "confidence": row.get("confidence"),
            }
        )
        for candidate in row.get("candidates", []):
            candidate.setdefault(
                "base_constraint_score",
                clamp01(
                    candidate.get(
                        "constraint_consistency_score",
                        0.0,
                    )
                ),
            )

    initial_consistency = calculate_global_consistency(
        rows,
        evidence_edges,
        constraints,
    )
    monotonic_consistency = initial_consistency[
        "raw_global_consistency"
    ]
    trace: List[Dict[str, Any]] = [
        {
            "iteration": 0,
            **initial_consistency,
            "global_consistency": monotonic_consistency,
            "status_changes": 0,
            "new_derived_candidates": 0,
            "propagation_messages": 0,
            "representation_updates": {
                "updated_nodes": 0,
                "mean_embedding_delta": 0.0,
                "max_embedding_delta": 0.0,
            },
            "message_score_feedback": score_delta_summary([]),
            "embedding_rescore": score_delta_summary([]),
            "iterative_retrieval": {},
            "raw_global_delta": 0.0,
            "state_change_rate": 0.0,
        }
    ]

    stable_count = 0
    stop_reason = "max_iterations"
    hard_constraint_conflicts = 0
    total_derived = 0
    total_retrieved = 0
    total_pruned = 0
    total_score_increases = 0
    total_score_decreases = 0

    for iteration in range(1, args.max_iterations + 1):
        occurrences = build_occurrences(rows)
        node_states, source_rows = confirmed_nodes(rows)
        messages_by_occurrence: Dict[
            Tuple[int, str], List[Dict[str, Any]]
        ] = defaultdict(list)
        messages_by_target: Dict[
            str, List[Dict[str, Any]]
        ] = defaultdict(list)
        derived_requests: Dict[
            Tuple[int, str], List[Dict[str, Any]]
        ] = defaultdict(list)
        iteration_hard_conflicts = 0

        for source_target_id, state in node_states.items():
            for constraint in adjacency.get(source_target_id, []):
                target_id = str(constraint["target"])
                explicit = constraint.get("explicit_satisfaction")
                satisfaction = (
                    clamp01(explicit) if explicit is not None else 1.0
                )
                signed_contribution = (
                    float(state["propagation_weight"])
                    * float(constraint["weight"])
                    * (2.0 * satisfaction - 1.0)
                )
                message = {
                    "iteration": iteration,
                    "source_target_id": source_target_id,
                    "source_semantic_type": state["semantic_type"],
                    "source_status": state["status"],
                    "source_confidence": state["confidence"],
                    "source_propagation_weight": state[
                        "propagation_weight"
                    ],
                    "constraint_kind": constraint["kind"],
                    "constraint_level": constraint["level"],
                    "constraint_weight": constraint["weight"],
                    "constraint_satisfaction": satisfaction,
                    # contribution remains an absolute compatibility field;
                    # signed_contribution is authoritative in v5.5.
                    "contribution": clamp01(abs(signed_contribution)),
                    "signed_contribution": max(
                        -1.0, min(1.0, signed_contribution)
                    ),
                }
                messages_by_target[target_id].append(message)

                hard_rejection = (
                    int(constraint["level"]) == 3
                    and explicit is not None
                    and clamp01(explicit) <= 0.0
                )
                existing = occurrences.get(target_id, [])
                if existing:
                    for row_index, _candidate in existing:
                        if (
                            str(rows[row_index].get("sample_id") or "")
                            == str(constraint.get("sample_id") or "")
                        ):
                            item = dict(message)
                            item["hard_rejection"] = hard_rejection
                            messages_by_occurrence[
                                (row_index, target_id)
                            ].append(item)
                else:
                    for row_index in source_rows.get(
                        source_target_id, []
                    ):
                        if (
                            str(rows[row_index].get("sample_id") or "")
                            == str(constraint.get("sample_id") or "")
                        ):
                            item = dict(message)
                            item["hard_rejection"] = hard_rejection
                            derived_requests[(row_index, target_id)].append(
                                item
                            )

        # Compact transaction journal.  Only rows that can be mutated by a
        # message, a derived candidate, an embedding rescore, or an affected
        # sample re-retrieval are copied.  This avoids duplicating the complete
        # full-run candidate table on every iteration.
        touched_row_indexes: Set[int] = {
            row_index for row_index, _target_id in messages_by_occurrence
        }
        touched_row_indexes.update(
            row_index for row_index, _target_id in derived_requests
        )
        potential_updated_target_ids = {
            target_id for target_id in messages_by_target
            if target_id in id_to_index
        }
        for target_id in potential_updated_target_ids:
            for row_index, _candidate in occurrences.get(target_id, []):
                touched_row_indexes.add(row_index)
        if args.iterative_retrieval:
            potential_affected_samples = {
                sample_by_id.get(target_id, "")
                for target_id in potential_updated_target_ids
                if sample_by_id.get(target_id, "")
            }
            for sample_id in potential_affected_samples:
                touched_row_indexes.update(sample_rows.get(sample_id, []))
        row_journal = {
            row_index: copy.deepcopy(rows[row_index])
            for row_index in touched_row_indexes
        }
        embedding_journal = {}
        if embeddings is not None:
            embedding_journal = {
                id_to_index[target_id]: embeddings[
                    id_to_index[target_id]
                ].detach().clone()
                for target_id in potential_updated_target_ids
            }

        derived_count = 0
        for (row_index, target_id), messages in derived_requests.items():
            row = rows[row_index]
            if find_candidate(row, target_id) is not None:
                continue
            if (
                not any(signed_message_value(message) > 0.0 for message in messages)
                and not any(
                    bool(message.get("hard_rejection")) for message in messages
                )
            ):
                # A contradiction may reduce an existing candidate but should
                # not create a brand-new unsupported hypothesis.
                continue
            target_semantic_type = semantic_type_by_id.get(target_id, "")
            if target_semantic_type not in SEMANTIC_TYPES:
                continue
            candidate = make_derived_candidate(
                row,
                target_id,
                target_semantic_type,
                messages,
                embedding_by_id,
            )
            if any(bool(message.get("hard_rejection")) for message in messages):
                candidate["candidate_status"] = "rejected"
                candidate["rejection_reason"] = (
                    "deterministic_constraint_violation"
                )
                iteration_hard_conflicts += 1
            row["candidates"].append(candidate)
            messages_by_occurrence[(row_index, target_id)].extend(messages)
            derived_count += 1

        message_score_deltas: List[float] = []
        for (row_index, target_id), messages in messages_by_occurrence.items():
            candidate = find_candidate(rows[row_index], target_id)
            if candidate is None:
                continue
            hard_rejection = any(
                bool(message.get("hard_rejection")) for message in messages
            )
            if hard_rejection:
                if str(candidate.get("candidate_status") or "") in (
                    "confirmed_hard",
                    "confirmed_soft",
                ):
                    candidate["hard_constraint_conflict"] = True
                    candidate["hard_constraint_conflict_iteration"] = iteration
                    iteration_hard_conflicts += 1
                else:
                    candidate["candidate_status"] = "rejected"
                    candidate["rejection_reason"] = (
                        "deterministic_constraint_violation"
                    )
            score_delta = update_candidate_from_messages(
                candidate,
                messages,
                args.representation_step,
                args.score_feedback_blend,
                args.max_score_drop_per_round,
                args.max_score_rise_per_round,
            )
            message_score_deltas.append(score_delta)

        message_score_feedback = score_delta_summary(message_score_deltas)

        representation_updates = {
            "updated_nodes": 0,
            "mean_embedding_delta": 0.0,
            "max_embedding_delta": 0.0,
        }
        updated_target_ids: Set[str] = set()
        if embeddings is not None:
            representation_updates, updated_target_ids = update_embeddings(
                embeddings,
                id_to_index,
                messages_by_target,
                args.representation_step,
            )

        iterative_retrieval: Dict[str, Any] = {}
        affected_samples = {
            sample_by_id[target_id]
            for target_id in updated_target_ids
            if sample_by_id.get(target_id)
        }
        if (
            args.iterative_retrieval
            and embeddings is not None
            and iteration % args.retrieval_interval == 0
        ):
            iterative_retrieval = reretrieve_candidates(
                rows,
                embeddings,
                id_to_index,
                node_id_by_index,
                sample_rows,
                sample_targets,
                affected_samples,
                iteration,
                args.retrieval_top_k_per_type,
                args.retrieval_max_candidates_per_type,
                args.retrieval_min_probability,
                args.retrieval_replacement_margin,
                args.retrieval_query_batch_size,
                args.retrieval_target_batch_size,
                retrieval_device,
            )

        embedding_rescore = score_delta_summary([])
        if embedding_by_id is not None:
            embedding_rescore = refresh_candidate_representation_scores(
                rows,
                embedding_by_id,
                args.embedding_rescore_blend,
                args.score_feedback_blend,
                args.max_score_drop_per_round,
                args.max_score_rise_per_round,
                affected_target_ids=(
                    updated_target_ids if updated_target_ids else None
                ),
            )

        status_changes = 0
        for row_index, row in enumerate(rows):
            # An untouched row normally cannot change.  Keep a temporary copy
            # so an unexpected policy-only change is still journaled without
            # retaining copies of every unchanged full-run row.
            temporary_before = None
            if row_index not in row_journal:
                temporary_before = copy.deepcopy(row)
            changed = refresh_row_decisions(
                row,
                soft_revisable=not args.lock_soft_confirmations,
                soft_switch_margin=args.soft_switch_margin,
                soft_release_margin=args.soft_release_margin,
                soft_release_gap_margin=args.soft_release_gap_margin,
            )
            status_changes += changed
            if changed and temporary_before is not None:
                row_journal[row_index] = temporary_before
            row["status_history"].append(
                {
                    "iteration": iteration,
                    "status": row.get("status"),
                    "active_semantic_type": row.get(
                        "active_semantic_type"
                    ),
                    "confidence": row.get("confidence"),
                }
            )

        consistency = calculate_global_consistency(
            rows,
            evidence_edges,
            constraints,
        )
        raw_consistency = consistency["raw_global_consistency"]
        previous_raw_consistency = float(
            trace[-1].get("raw_global_consistency", 0.0)
        )
        raw_global_delta = raw_consistency - previous_raw_consistency
        proposal_audit = {
            "raw_global_consistency": raw_consistency,
            "raw_global_delta": raw_global_delta,
            "status_changes": status_changes,
            "new_derived_candidates": derived_count,
            "iterative_retrieval": dict(iterative_retrieval),
            "representation_updates": dict(representation_updates),
            "message_score_feedback": dict(message_score_feedback),
            "embedding_rescore": dict(embedding_rescore),
            "hard_constraint_conflicts": iteration_hard_conflicts,
        }
        transaction_accepted = True
        rollback_reason = ""
        if (
            args.enforce_raw_monotonic
            and raw_consistency + 1e-12 < previous_raw_consistency
        ):
            transaction_accepted = False
            rollback_reason = "raw_s_global_would_decrease"
            for row_index, snapshot in row_journal.items():
                rows[row_index] = snapshot
            for row_index, row in enumerate(rows):
                if row_index in row_journal:
                    continue
                history = row.get("status_history")
                if (
                    isinstance(history, list)
                    and history
                    and int(history[-1].get("iteration", -1)) == iteration
                ):
                    history.pop()
            if embeddings is not None:
                for embedding_index, snapshot in embedding_journal.items():
                    embeddings[embedding_index].copy_(snapshot)
            consistency = calculate_global_consistency(
                rows,
                evidence_edges,
                constraints,
            )
            raw_consistency = consistency["raw_global_consistency"]
            raw_global_delta = raw_consistency - previous_raw_consistency
            status_changes = 0
            derived_count = 0
            iteration_hard_conflicts = 0
            representation_updates = {
                "updated_nodes": 0,
                "mean_embedding_delta": 0.0,
                "max_embedding_delta": 0.0,
            }
            message_score_feedback = score_delta_summary([])
            embedding_rescore = score_delta_summary([])
            iterative_retrieval = {
                "new_candidates": 0,
                "pruned_candidates": 0,
                "rolled_back": True,
            }
        monotonic_consistency = raw_consistency
        propagation_messages = sum(
            len(values) for values in messages_by_occurrence.values()
        )
        trace.append(
            {
                "iteration": iteration,
                **consistency,
                "global_consistency": monotonic_consistency,
                "monotonic_envelope_applied": False,
                "transaction_accepted": transaction_accepted,
                "rollback_reason": rollback_reason,
                "proposal_audit": proposal_audit,
                "journaled_rows": len(row_journal),
                "journaled_embedding_rows": len(embedding_journal),
                "status_changes": status_changes,
                "new_derived_candidates": derived_count,
                "propagation_messages": propagation_messages,
                "confirmed_source_nodes": len(node_states),
                "hard_constraint_conflicts": iteration_hard_conflicts,
                "representation_updates": representation_updates,
                "message_score_feedback": message_score_feedback,
                "embedding_rescore": embedding_rescore,
                "iterative_retrieval": iterative_retrieval,
                "raw_global_delta": raw_global_delta,
                "state_change_rate": status_changes / max(len(rows), 1),
            }
        )
        print(json.dumps(trace[-1], ensure_ascii=False), flush=True)

        hard_constraint_conflicts += iteration_hard_conflicts
        total_derived += derived_count
        total_retrieved += int(
            iterative_retrieval.get("new_candidates", 0) or 0
        )
        total_pruned += int(
            iterative_retrieval.get("pruned_candidates", 0) or 0
        )
        total_score_increases += int(
            message_score_feedback.get("increased_candidates", 0) or 0
        ) + int(embedding_rescore.get("increased_candidates", 0) or 0)
        total_score_decreases += int(
            message_score_feedback.get("decreased_candidates", 0) or 0
        ) + int(embedding_rescore.get("decreased_candidates", 0) or 0)
        plateau_stable = (
            abs(raw_global_delta) <= args.min_global_delta
            and status_changes / max(len(rows), 1)
            <= args.max_state_change_rate_for_stable
            and derived_count == 0
            and int(iterative_retrieval.get("new_candidates", 0) or 0) == 0
            and int(iterative_retrieval.get("pruned_candidates", 0) or 0) == 0
        )
        if plateau_stable:
            stable_count += 1
        else:
            stable_count = 0

        if monotonic_consistency >= args.global_threshold:
            stop_reason = "global_consistency_reached"
            break
        if stable_count >= args.stable_rounds:
            stop_reason = "state_stable"
            break

    final_iteration = int(trace[-1]["iteration"])
    final_consistency = float(trace[-1]["global_consistency"])
    for row in rows:
        row["implementation_revision"] = (
            "v5_6_irreversible_confirmation_transactional_raw_sglobal"
        )
        row["reasoning_iterations"] = final_iteration
        row["global_consistency"] = final_consistency
        row["reasoning_stop_reason"] = stop_reason

    output = ensure_output_path(args.out, overwrite=args.overwrite)
    with output.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    if embeddings is not None:
        save_embedding_payload(
            args.embedding_out,
            embedding_payload,
            embedding_tensor_key,
            embeddings,
            args.overwrite,
        )

    status_summary = summarize_statuses(rows)
    summary = {
        "done": True,
        "schema": "task2_222_iterative_summary_v5_3",
        "implementation_revision": (
            "v5_6_irreversible_confirmation_transactional_raw_sglobal"
        ),
        "input": args.validated,
        "output": str(output),
        "embedding_output": args.embedding_out,
        "records": len(rows),
        "edges_seen": edges_seen,
        "evidence_edges_used": len(evidence_edges),
        "constraint_edges_used": len(constraints),
        "max_iterations": args.max_iterations,
        "iterations_run": final_iteration,
        "global_threshold": args.global_threshold,
        "stable_rounds_required": args.stable_rounds,
        "min_global_delta": args.min_global_delta,
        "max_state_change_rate_for_stable": (
            args.max_state_change_rate_for_stable
        ),
        "embedding_rescore_blend": args.embedding_rescore_blend,
        "score_feedback_blend": args.score_feedback_blend,
        "max_score_drop_per_round": args.max_score_drop_per_round,
        "max_score_rise_per_round": args.max_score_rise_per_round,
        "soft_confirmations_revisable": not args.lock_soft_confirmations,
        "confirmed_states_irreversible": bool(args.lock_soft_confirmations),
        "raw_monotonic_transaction_enabled": bool(
            args.enforce_raw_monotonic
        ),
        "soft_switch_margin": args.soft_switch_margin,
        "soft_release_margin": args.soft_release_margin,
        "soft_release_gap_margin": args.soft_release_gap_margin,
        "legacy_monotonic_score_updates": (
            args.legacy_monotonic_score_updates
        ),
        "iterative_retrieval_enabled": args.iterative_retrieval,
        "retrieval_device": retrieval_device,
        "retrieval_interval": args.retrieval_interval,
        "retrieval_top_k_per_type": args.retrieval_top_k_per_type,
        "retrieval_max_candidates_per_type": (
            args.retrieval_max_candidates_per_type
        ),
        "retrieval_min_probability": args.retrieval_min_probability,
        "retrieval_replacement_margin": (
            args.retrieval_replacement_margin
        ),
        "confirm_policy_input": str(args.confirm_policy or ""),
        "policies": active_policy,
        "stop_reason": stop_reason,
        "initial_global_consistency": trace[0]["global_consistency"],
        "final_global_consistency": final_consistency,
        "global_consistency_monotonic": all(
            trace[index]["global_consistency"]
            + 1e-12
            >= trace[index - 1]["global_consistency"]
            for index in range(1, len(trace))
        ),
        "raw_consistency_decrease_count": sum(
            int(float(item.get("raw_global_delta", 0.0)) < -1e-12)
            for item in trace
        ),
        "rolled_back_proposal_count": sum(
            int(item.get("transaction_accepted") is False)
            for item in trace
        ),
        "reported_global_consistency_is_monotonic_envelope": False,
        "total_derived_candidates": total_derived,
        "total_retrieved_candidates": total_retrieved,
        "total_pruned_candidates": total_pruned,
        "total_score_increases": total_score_increases,
        "total_score_decreases": total_score_decreases,
        "hard_constraint_conflicts": hard_constraint_conflicts,
        "representation_update_mode": (
            "signed_joint_space_message_passing+bidirectional_rescore+"
            + (
                "iterative_reretrieval"
                if args.iterative_retrieval
                else "fixed_candidate_ablation"
            )
            if embeddings is not None
            else "candidate_score_proxy_only"
        ),
        "neural_symbolic_closed_loop": {
            "neural_state_updated": embeddings is not None,
            "symbolic_messages_signed": True,
            "candidate_scores_bidirectional": (
                args.max_score_drop_per_round > 0.0
            ),
            "soft_decisions_revisable": (
                not args.lock_soft_confirmations
            ),
            "confirmed_states_irreversible": bool(
                args.lock_soft_confirmations
            ),
            "actual_raw_sglobal_transactional": bool(
                args.enforce_raw_monotonic
            ),
            "candidate_set_retrieved_each_round": (
                args.iterative_retrieval
            ),
            "gold_used_during_inference": False,
        },
        "iteration_trace": trace,
        **status_summary,
    }
    write_json(
        args.summary_out,
        summary,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
