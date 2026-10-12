#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation-guarded four-module collaboration and ablations (v6.7)."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from module_common_223_v5_3 import (
    CONFIRMED, DIRECTION_WEIGHT, SEMANTIC_TYPES, clamp01, decision_for_group,
    edge_endpoints, edge_is_constraint, edge_satisfaction, edge_weight,
    load_confirmation_policy_223, noisy_or, normalize_semantic_type,
    output_path, read_jsonl, write_json,
)


Occurrence = Tuple[int, str, int]
CollaborationEdge = Tuple[str, str, float, Tuple[str, str], int]
ABLATIONS = (
    "full",
    "no_api_hard_seed",
    "no_type_layout",
    "no_interface_feedback",
    "no_api_to_interface",
    "no_type_to_interface",
    "no_interface_incoming",
    "no_collaboration",
)


def direction_enabled(
    direction: Tuple[str, str], ablation: str
) -> bool:
    if ablation == "no_collaboration":
        return False
    if ablation == "no_type_layout" and direction in (
        ("TypeSem", "LayoutSem"),
        ("LayoutSem", "TypeSem"),
    ):
        return False
    if ablation == "no_interface_feedback" and direction[0] == "InterfaceSem":
        return False
    if ablation == "no_api_to_interface" and direction == (
        "APISem", "InterfaceSem"
    ):
        return False
    if ablation == "no_type_to_interface" and direction == (
        "TypeSem", "InterfaceSem"
    ):
        return False
    if ablation == "no_interface_incoming" and direction[1] == "InterfaceSem":
        return False
    return True


def parse_module_weights(text: str, fallback: float) -> Dict[str, float]:
    """Parse per-module fusion weights, retaining a scalar fallback."""
    result = {semantic_type: float(fallback) for semantic_type in SEMANTIC_TYPES}
    for raw in (item.strip() for item in str(text or "").split(",")):
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError("Invalid --module-weights item: %s" % raw)
        name, value = raw.split("=", 1)
        semantic_type = normalize_semantic_type(name)
        if semantic_type not in SEMANTIC_TYPES:
            raise ValueError("Unknown module in --module-weights: %s" % name)
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ValueError("Module fusion weights must be in [0,1]")
        result[semantic_type] = number
    return result


def parse_direction_weights(text: str) -> Dict[Tuple[str, str], float]:
    """Parse direction-specific overrides such as A->B=0.25."""
    result = {direction: float(value) for direction, value in DIRECTION_WEIGHT.items()}
    for raw in (item.strip() for item in str(text or "").split(",")):
        if not raw:
            continue
        if "=" not in raw or "->" not in raw:
            raise ValueError(
                "Invalid --direction-weight-overrides item: %s" % raw
            )
        direction_text, value = raw.rsplit("=", 1)
        source_text, target_text = direction_text.split("->", 1)
        direction = (
            normalize_semantic_type(source_text),
            normalize_semantic_type(target_text),
        )
        if direction not in DIRECTION_WEIGHT:
            raise ValueError(
                "Unsupported collaboration direction: %s" % direction_text
            )
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ValueError("Direction weights must be in [0,1]")
        result[direction] = number
    return result


def load_module_scores(paths: List[str]) -> Dict[Tuple[str, str], Dict[str, float]]:
    scores: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for path in paths:
        for row in read_jsonl(path):
            query_id = str(row.get("query_id") or "")
            semantic_type = normalize_semantic_type(row.get("semantic_type"))
            for candidate in row.get("candidate_scores", []) or []:
                target_id = str(candidate.get("target_id") or "")
                if query_id and target_id:
                    scores[(query_id, semantic_type)][target_id] = clamp01(
                        candidate.get(
                            "module_score_223_v6_6",
                            candidate.get("module_score_223_v5_3", 0.0),
                        )
                    )
    return scores


def group(row: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    result = defaultdict(list)
    for candidate in row.get("candidates", []) or []:
        semantic_type = normalize_semantic_type(candidate.get("target_semantic_type"))
        if semantic_type in SEMANTIC_TYPES:
            result[semantic_type].append(candidate)
    return result


def build_occurrences(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Occurrence]], Dict[Occurrence, Dict[str, Any]]]:
    by_target: Dict[str, List[Occurrence]] = defaultdict(list)
    values: Dict[Occurrence, Dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        for semantic_type, candidates in group(row).items():
            for candidate_index, candidate in enumerate(candidates):
                occurrence = (row_index, semantic_type, candidate_index)
                target_id = str(candidate.get("target_id") or "")
                if target_id:
                    by_target[target_id].append(occurrence)
                    values[occurrence] = candidate
    return by_target, values


def confirmed_targets(
    rows: List[Dict[str, Any]], ablation: str = "full"
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        decisions = row.get("semantic_decisions") if isinstance(row.get("semantic_decisions"), Mapping) else {}
        for semantic_type, decision in decisions.items():
            if not isinstance(decision, Mapping) or decision.get("status") not in CONFIRMED:
                continue
            if (
                ablation == "no_api_hard_seed"
                and normalize_semantic_type(semantic_type) == "APISem"
                and decision.get("status") == "confirmed_hard"
            ):
                continue
            target_id = str(decision.get("best_target_id") or "")
            if not target_id:
                continue
            weight = 1.0 if decision.get("status") == "confirmed_hard" else clamp01(decision.get("confidence", 0.0))
            previous = result.get(target_id)
            if previous is None or weight > previous["weight"]:
                result[target_id] = {
                    "semantic_type": normalize_semantic_type(semantic_type),
                    "status": decision.get("status"),
                    "weight": weight,
                }
    return result


def mean_target_score(
    target_id: str,
    by_target: Mapping[str, Sequence[Occurrence]],
    occurrence_values: Mapping[Occurrence, Mapping[str, Any]],
) -> float:
    """Mean belief in a semantic node across all query-local occurrences."""
    values = [
        clamp01(occurrence_values[occurrence].get("score_223_v5_3", 0.0))
        for occurrence in by_target.get(target_id, [])
        if occurrence in occurrence_values
    ]
    return sum(values) / len(values) if values else 0.0


def _empty_consistency_bucket() -> Dict[str, Any]:
    return {
        "eligible_edges": 0,
        "eligible_weight": 0.0,
        "active_edges": 0,
        "active_weight": 0.0,
        "weighted_satisfaction": 0.0,
    }


def _finish_consistency_bucket(bucket: Mapping[str, Any]) -> Dict[str, Any]:
    eligible_edges = int(bucket.get("eligible_edges", 0) or 0)
    eligible_weight = float(bucket.get("eligible_weight", 0.0) or 0.0)
    active_edges = int(bucket.get("active_edges", 0) or 0)
    active_weight = float(bucket.get("active_weight", 0.0) or 0.0)
    weighted_satisfaction = float(
        bucket.get("weighted_satisfaction", 0.0) or 0.0
    )
    return {
        "eligible_edges": eligible_edges,
        "active_edges": active_edges,
        "constraint_coverage": (
            active_edges / eligible_edges if eligible_edges else None
        ),
        "weighted_constraint_coverage": (
            active_weight / eligible_weight if eligible_weight else None
        ),
        "raw_consistency": (
            weighted_satisfaction / active_weight if active_weight else None
        ),
    }


def evidence_snapshot(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Weighted agreement of each module's best candidate with its evidence."""
    total_weight = total_score = 0.0
    by_module = {}
    for semantic_type in SEMANTIC_TYPES:
        module_weight = module_score = 0.0
        groups_seen = 0
        for row in rows:
            candidates = group(row).get(semantic_type, [])
            if not candidates:
                continue
            active = [value for value in candidates if value.get("candidate_status") != "rejected"]
            if not active:
                continue
            best = max(active, key=lambda value: clamp01(value.get("score_223_v5_3", 0.0)))
            coverage = clamp01(best.get("evidence_coverage", 0.0))
            weight = max(0.05, coverage)
            score = clamp01(best.get("score_223_v5_3", 0.0))
            module_weight += weight
            module_score += weight * score
            groups_seen += 1
        value = module_score / module_weight if module_weight else None
        by_module[semantic_type] = {
            "groups": groups_seen,
            "evidence_weight": module_weight,
            "raw_evidence_consistency": value,
        }
        total_weight += module_weight
        total_score += module_score
    return {
        "raw_evidence_consistency": total_score / total_weight if total_weight else None,
        "module_evidence_consistency": by_module,
    }


def consistency_snapshot(
    iteration: int,
    rows: List[Dict[str, Any]],
    collaboration_constraints: Sequence[CollaborationEdge],
    by_target: Mapping[str, Sequence[Occurrence]],
    occurrence_values: Mapping[Occurrence, Mapping[str, Any]],
    ablation: str = "full",
) -> Dict[str, Any]:
    """Compute raw (non-monotonic-envelope) cross-module consistency.

    An edge is active only when its source semantic object is confirmed.  Its
    satisfaction is the mean target belief, weighted by the edge strength and
    by the source propagation weight (hard=1, soft=confidence).
    """
    sources = confirmed_targets(rows, ablation=ablation)
    overall = _empty_consistency_bucket()
    by_direction = {
        "%s->%s" % direction: _empty_consistency_bucket()
        for direction in DIRECTION_WEIGHT
    }
    by_module = {
        semantic_type: _empty_consistency_bucket()
        for semantic_type in SEMANTIC_TYPES
    }

    for (
        source_id, target_id, relation_weight, direction, polarity
    ) in collaboration_constraints:
        direction_name = "%s->%s" % direction
        target_type = direction[1]
        buckets = (overall, by_direction[direction_name], by_module[target_type])
        for bucket in buckets:
            bucket["eligible_edges"] += 1
            bucket["eligible_weight"] += relation_weight

        state = sources.get(source_id)
        if state is None:
            continue
        effective_weight = clamp01(relation_weight * state["weight"])
        if effective_weight <= 0.0:
            continue
        target_belief = mean_target_score(
            target_id, by_target, occurrence_values
        )
        satisfaction = (
            target_belief if polarity >= 0 else 1.0 - target_belief
        )
        for bucket in buckets:
            bucket["active_edges"] += 1
            bucket["active_weight"] += effective_weight
            bucket["weighted_satisfaction"] += (
                effective_weight * satisfaction
            )

    finished_overall = _finish_consistency_bucket(overall)
    evidence = evidence_snapshot(rows)
    evidence_score = evidence["raw_evidence_consistency"]
    constraint_score = finished_overall["raw_consistency"]
    if evidence_score is not None and constraint_score is not None:
        raw_s_global = 0.60 * evidence_score + 0.40 * constraint_score
    elif evidence_score is not None:
        raw_s_global = evidence_score
    else:
        raw_s_global = constraint_score
    return {
        "iteration": iteration,
        "confirmed_sources": len(sources),
        "eligible_constraint_edges": finished_overall["eligible_edges"],
        "active_constraint_edges": finished_overall["active_edges"],
        "constraint_coverage": finished_overall["constraint_coverage"],
        "weighted_constraint_coverage": finished_overall[
            "weighted_constraint_coverage"
        ],
        "raw_s_global": raw_s_global,
        "raw_evidence_consistency": evidence_score,
        "raw_constraint_consistency": constraint_score,
        "s_global_weights": {"evidence": 0.60, "constraint": 0.40},
        "module_evidence_consistency": evidence[
            "module_evidence_consistency"
        ],
        "module_consistency": {
            semantic_type: _finish_consistency_bucket(bucket)
            for semantic_type, bucket in by_module.items()
        },
        "directional_consistency": {
            direction: _finish_consistency_bucket(bucket)
            for direction, bucket in by_direction.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--module-result", action="append", default=[])
    parser.add_argument(
        "--confirm-policy",
        default="",
        help=(
            "Validation-calibrated 2.2.2 confirmation policy. Disabled "
            "modules remain fail-closed when 2.2.3 recomputes decisions."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--stable-rounds", type=int, default=2)
    parser.add_argument("--global-threshold", type=float, default=0.85)
    parser.add_argument("--module-weight", type=float, default=0.25)
    parser.add_argument(
        "--module-weights",
        default="",
        help=(
            "Per-module Validation-calibrated fusion weights, for example "
            "APISem=0.25,TypeSem=0.25,LayoutSem=0.25,InterfaceSem=0"
        ),
    )
    parser.add_argument(
        "--direction-weight-overrides",
        default="",
        help=(
            "Direction-specific weights, for example "
            "APISem->InterfaceSem=0,TypeSem->InterfaceSem=0"
        ),
    )
    parser.add_argument("--propagation-step", type=float, default=0.25)
    parser.add_argument(
        "--max-score-change-per-round", type=float, default=0.08
    )
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_iterations <= 0 or args.stable_rounds <= 0:
        raise ValueError("Iteration counts must be positive")
    if not 0.0 <= args.global_threshold <= 1.0:
        raise ValueError("--global-threshold must be in [0,1]")
    if not 0.0 <= args.max_score_change_per_round <= 1.0:
        raise ValueError("--max-score-change-per-round must be in [0,1]")
    if not 0.0 <= args.module_weight <= 1.0:
        raise ValueError("--module-weight must be in [0,1]")
    module_weights = parse_module_weights(
        args.module_weights, args.module_weight
    )
    direction_weights = parse_direction_weights(
        args.direction_weight_overrides
    )
    confirmation_policy, confirmation_policy_path = (
        load_confirmation_policy_223(args.confirm_policy or None)
    )
    rows = [copy.deepcopy(row) for row in read_jsonl(args.reasoning)]
    module_scores = load_module_scores(args.module_result)

    # Seed scores with 2.2.2 confidence and module-specific recommendations.
    for row in rows:
        query_id = str(row.get("query_id") or "")
        for semantic_type, candidates in group(row).items():
            score_map = module_scores.get((query_id, semantic_type), {})
            for candidate in candidates:
                baseline = clamp01(candidate.get("confidence", candidate.get("validated_score", 0.0)))
                module = score_map.get(str(candidate.get("target_id") or ""), baseline)
                candidate["score_before_223_v5_3"] = baseline
                candidate["module_score_223_v5_3"] = module
                module_weight = module_weights[semantic_type]
                initial_fused_score = clamp01(
                    (1.0 - module_weight) * baseline
                    + module_weight * module
                )
                candidate["module_fusion_weight_223_v6_7"] = module_weight
                candidate[
                    "score_after_module_fusion_before_collaboration_223_v5_3"
                ] = initial_fused_score
                candidate["score_223_v5_3"] = initial_fused_score
                candidate.setdefault("propagation_trace_223_v5_3", [])
                candidate.setdefault("applied_constraint_messages_223_v5_3", [])

    node_type: Dict[str, str] = {}
    by_target, occurrence_values = build_occurrences(rows)
    for target_id, occurrences in by_target.items():
        if occurrences:
            node_type[target_id] = occurrences[0][1]

    adjacency = defaultdict(list)
    collaboration_constraints: List[CollaborationEdge] = []
    edges_seen = collaboration_edges = 0
    filtered_edges = Counter()
    for edge in read_jsonl(args.edges):
        edges_seen += 1
        if not edge_is_constraint(edge):
            continue
        source, target = edge_endpoints(edge)
        source_type, target_type = node_type.get(source), node_type.get(target)
        direction = (source_type, target_type)
        if direction not in DIRECTION_WEIGHT:
            continue
        direction_weight = direction_weights[direction]
        if (
            not direction_enabled(direction, args.ablation)
            or direction_weight <= 0.0
        ):
            filtered_edges["%s->%s" % direction] += 1
            continue
        weight = clamp01(edge_weight(edge) * direction_weight)
        explicit = edge_satisfaction(edge)
        satisfaction = 1.0 if explicit is None else clamp01(explicit)
        signed_weight = weight * (2.0 * satisfaction - 1.0)
        if abs(signed_weight) <= 1e-12:
            filtered_edges["neutral_constraint"] += 1
            continue
        polarity = 1 if signed_weight >= 0.0 else -1
        adjacency[source].append((target, signed_weight, direction))
        collaboration_constraints.append(
            (source, target, abs(signed_weight), direction, polarity)
        )
        collaboration_edges += 1

    initial_locked = {}
    for row_index, row in enumerate(rows):
        decisions = row.get("semantic_decisions") if isinstance(row.get("semantic_decisions"), Mapping) else {}
        for semantic_type, decision in decisions.items():
            if isinstance(decision, Mapping) and (decision.get("status") in CONFIRMED or decision.get("status") == "rejected"):
                if isinstance(decision, dict):
                    decision["locked_from_222_v5_6"] = True
                initial_locked[(row_index, normalize_semantic_type(semantic_type))] = (
                    decision.get("status"), str(decision.get("best_target_id") or "")
                )

    full_initial_sources = confirmed_targets(rows, ablation="full")
    active_initial_sources = confirmed_targets(
        rows, ablation=args.ablation
    )
    suppressed_source_ids = sorted(
        set(full_initial_sources) - set(active_initial_sources)
    )

    initial_consistency = consistency_snapshot(
        0,
        rows,
        collaboration_constraints,
        by_target,
        occurrence_values,
        ablation=args.ablation,
    )
    consistency_trace = [initial_consistency]
    print(json.dumps(initial_consistency, ensure_ascii=False), flush=True)

    trace = []
    stable = 0
    stop_reason = "max_iterations"
    total_score_increases = 0
    total_score_decreases = 0
    for iteration in range(1, args.max_iterations + 1):
        sources = confirmed_targets(rows, ablation=args.ablation)
        messages = defaultdict(list)
        for source_id, state in sources.items():
            for target_id, signed_relation_weight, direction in adjacency.get(source_id, []):
                signed_message_weight = (
                    state["weight"] * signed_relation_weight
                )
                message_weight = clamp01(abs(signed_message_weight))
                if message_weight <= 0.0:
                    continue
                for occurrence in by_target.get(target_id, []):
                    messages[occurrence].append({
                        "source_target_id": source_id,
                        "source_semantic_type": state["semantic_type"],
                        "source_status": state["status"],
                        "direction": "%s->%s" % direction,
                        "weight": message_weight,
                        "signed_weight": max(
                            -1.0, min(1.0, signed_message_weight)
                        ),
                        "polarity": (
                            "support" if signed_message_weight > 0.0
                            else "contradiction"
                        ),
                    })

        score_updates = 0
        score_increases = 0
        score_decreases = 0
        for occurrence, incoming in messages.items():
            candidate = occurrence_values[occurrence]
            already_applied = set(
                str(value)
                for value in candidate.get(
                    "applied_constraint_messages_223_v5_3", []
                )
            )
            novel = []
            novel_keys = []
            for message in incoming:
                key = "%s|%s|%s" % (
                    message["source_target_id"],
                    candidate.get("target_id"),
                    message["direction"] + "|" + message["polarity"],
                )
                if key not in already_applied:
                    novel.append(message)
                    novel_keys.append(key)
                    already_applied.add(key)
            if not novel:
                continue
            support = noisy_or([
                message["weight"] for message in novel
                if message["signed_weight"] > 0.0
            ])
            contradiction = noisy_or([
                message["weight"] for message in novel
                if message["signed_weight"] < 0.0
            ])
            old_score = clamp01(candidate.get("score_223_v5_3", 0.0))
            net = support - contradiction
            beta = clamp01(args.propagation_step * abs(net))
            proposed = (
                old_score + beta * (1.0 - old_score)
                if net >= 0.0
                else old_score * (1.0 - beta)
            )
            lower = old_score - args.max_score_change_per_round
            upper = old_score + args.max_score_change_per_round
            new_score = clamp01(min(upper, max(lower, proposed)))
            candidate["applied_constraint_messages_223_v5_3"] = sorted(
                already_applied
            )
            if abs(new_score - old_score) > 1e-12:
                score_updates += 1
                candidate["score_223_v5_3"] = new_score
                candidate["cross_module_support_223_v5_3"] = support
                candidate["cross_module_contradiction_223_v5_5"] = contradiction
                candidate["propagation_trace_223_v5_3"].extend(novel[:16])
                if new_score > old_score:
                    score_increases += 1
                else:
                    score_decreases += 1

        status_changes = 0
        for row_index, row in enumerate(rows):
            grouped = group(row)
            decisions = row.get("semantic_decisions") if isinstance(row.get("semantic_decisions"), Mapping) else {}
            updated = dict(decisions)
            for semantic_type, candidates in grouped.items():
                previous = decisions.get(semantic_type) if isinstance(decisions.get(semantic_type), Mapping) else {}
                decision = decision_for_group(
                    semantic_type,
                    candidates,
                    previous,
                    confirmation_policy=confirmation_policy,
                )
                if decision.get("status") != previous.get("status") or str(decision.get("best_target_id") or "") != str(previous.get("best_target_id") or ""):
                    status_changes += 1
                updated[semantic_type] = decision
            row["semantic_decisions"] = updated
            active = normalize_semantic_type(row.get("active_semantic_type"))
            if active in updated:
                decision = updated[active]
                row["status"] = decision.get("status", "hypothesis")
                row["confidence"] = decision.get("confidence", 0.0)
                row["preferred_target_id"] = decision.get("best_target_id")

        round_consistency = consistency_snapshot(
            iteration,
            rows,
            collaboration_constraints,
            by_target,
            occurrence_values,
            ablation=args.ablation,
        )
        consistency_trace.append(round_consistency)
        trace.append({
            "iteration": iteration,
            "confirmed_sources": len(sources),
            "confirmed_sources_after": round_consistency["confirmed_sources"],
            "propagation_messages": sum(len(value) for value in messages.values()),
            "candidate_score_updates": score_updates,
            "candidate_score_increases": score_increases,
            "candidate_score_decreases": score_decreases,
            "status_changes": status_changes,
            "raw_s_global": round_consistency["raw_s_global"],
            "raw_evidence_consistency": round_consistency[
                "raw_evidence_consistency"
            ],
            "raw_constraint_consistency": round_consistency[
                "raw_constraint_consistency"
            ],
            "constraint_coverage": round_consistency["constraint_coverage"],
            "weighted_constraint_coverage": round_consistency[
                "weighted_constraint_coverage"
            ],
            "module_consistency": round_consistency["module_consistency"],
            "directional_consistency": round_consistency[
                "directional_consistency"
            ],
        })
        print(json.dumps(trace[-1], ensure_ascii=False), flush=True)
        total_score_increases += score_increases
        total_score_decreases += score_decreases
        if score_updates == 0 and status_changes == 0:
            stable += 1
        else:
            stable = 0
        if (
            round_consistency["raw_s_global"] is not None
            and round_consistency["raw_s_global"] >= args.global_threshold
        ):
            stop_reason = "global_threshold"
            break
        if stable >= args.stable_rounds:
            stop_reason = "state_stable"
            break

    locked_violations = 0
    for (row_index, semantic_type), (status, target_id) in initial_locked.items():
        decision = rows[row_index].get("semantic_decisions", {}).get(semantic_type, {})
        if decision.get("status") != status or str(decision.get("best_target_id") or "") != target_id:
            locked_violations += 1

    output = output_path(args.out, args.overwrite)
    with output.open("w", encoding="utf-8") as writer:
        for row in rows:
            row["schema"] = "task2_223_collaborative_result_v6_7"
            row["implementation_revision_223"] = "v6_7_validation_guarded"
            row["ablation_223_v6_6"] = args.ablation
            row["collaboration_iterations_223_v5_3"] = len(trace)
            row["collaboration_stop_reason_223_v5_3"] = stop_reason
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_consistency = consistency_trace[-1]
    initial_s_global = initial_consistency["raw_s_global"]
    final_s_global = final_consistency["raw_s_global"]
    s_global_delta = (
        final_s_global - initial_s_global
        if initial_s_global is not None and final_s_global is not None
        else None
    )
    summary = {
        "schema": "task2_223_collaboration_summary_v6_7",
        "implementation_revision": "v6_7_validation_guarded",
        "ablation": args.ablation,
        "records": len(rows),
        "edges_seen": edges_seen,
        "collaboration_edges": collaboration_edges,
        "directions": {
            "%s->%s" % key: direction_weights[key]
            for key in DIRECTION_WEIGHT
            if direction_enabled(key, args.ablation)
            and direction_weights[key] > 0.0
        },
        "configured_direction_weights": {
            "%s->%s" % key: value
            for key, value in direction_weights.items()
        },
        "module_fusion_weights": module_weights,
        "confirmation_policy_file": confirmation_policy_path,
        "confirmation_policy": confirmation_policy,
        "confirmation_policy_fail_closed_modules": sorted(
            semantic_type
            for semantic_type, policy in confirmation_policy.items()
            if not bool(policy.get("soft_confirmation_enabled", True))
        ),
        "filtered_collaboration_edges": dict(filtered_edges),
        "suppressed_confirmed_source_count": len(suppressed_source_ids),
        "suppressed_confirmed_source_examples": suppressed_source_ids[:20],
        "signed_constraint_feedback": True,
        "max_score_change_per_round": args.max_score_change_per_round,
        "global_threshold": args.global_threshold,
        "total_candidate_score_increases": total_score_increases,
        "total_candidate_score_decreases": total_score_decreases,
        "iterations_run": len(trace),
        "stop_reason": stop_reason,
        "locked_input_decisions": len(initial_locked),
        "locked_decision_violations": locked_violations,
        "consistency_definition": (
            "raw_s_global = 0.60 * evidence_consistency + 0.40 * "
            "active_constraint_consistency; missing component is omitted; "
            "no monotonic reporting envelope"
        ),
        "specialised_module_scores_consumed": True,
        "confirmed_input_decisions_irreversible": True,
        "initial_raw_s_global": initial_s_global,
        "final_raw_s_global": final_s_global,
        "raw_s_global_delta": s_global_delta,
        "initial_constraint_coverage": initial_consistency[
            "constraint_coverage"
        ],
        "final_constraint_coverage": final_consistency[
            "constraint_coverage"
        ],
        "consistency_trace": consistency_trace,
        "iteration_trace": trace,
    }
    write_json(args.summary_out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
