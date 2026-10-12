#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate hypotheses with type-specific gates and deterministic API seeds."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from module_common_v5_3 import (
    CONFIRM_POLICY,
    HARD_API_SOURCES,
    SEMANTIC_TYPES,
    api_source_evidence_from_objects,
    clamp01,
    determine_status,
    edge_family,
    edge_kind,
    edge_satisfaction,
    edge_weight,
    endpoints,
    ensure_output_path,
    evidence_score,
    evidence_type_from_objects,
    is_hard_api_source,
    load_node_metadata,
    noisy_or,
    normalize_api_source,
    normalize_semantic_type,
    read_jsonl,
    select_row_decision,
    summarize_statuses,
    write_json,
    apply_confirm_policy,
)


Pair = Tuple[str, str]


def context_collaboration_score(
    total_count: int,
    total_weight: float,
    exact_count: int,
    exact_weight: float,
) -> float:
    count = max(0, total_count - exact_count)
    weight = max(0.0, total_weight - exact_weight)
    if count <= 0:
        return 0.0
    mean = weight / count
    breadth = math.log1p(min(count, 16)) / math.log1p(16)
    return clamp01(mean * breadth)


def decide_semantic_group(
    semantic_type: str,
    candidates: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    semantic_type = normalize_semantic_type(semantic_type)
    values = [dict(candidate) for candidate in candidates]
    hard_candidates = [
        candidate
        for candidate in values
        if bool(candidate.get("hard_api_eligible"))
    ]
    hard_conflict = len(hard_candidates) > 1

    if len(hard_candidates) == 1:
        hard_target = str(hard_candidates[0].get("target_id") or "")
        values.sort(
            key=lambda item: (
                str(item.get("target_id") or "") != hard_target,
                -float(item.get("validated_score", 0.0)),
                str(item.get("target_id") or ""),
            )
        )
    else:
        values.sort(
            key=lambda item: (
                -float(item.get("validated_score", 0.0)),
                str(item.get("target_id") or ""),
            )
        )

    if not values:
        return (
            {
                "semantic_type": semantic_type,
                "status": "rejected",
                "confidence": 0.0,
                "confidence_gap": 0.0,
                "confirmation_level": 0,
                "confirmation_source": "no_candidates",
                "propagation_weight": 0.0,
                "locked": True,
                "candidate_count": 0,
                "best_target_id": None,
                "hard_seed_conflict": False,
            },
            values,
        )

    best = values[0]
    second = values[1] if len(values) > 1 else None
    best_score = float(best.get("validated_score", 0.0))
    second_score = (
        float(second.get("validated_score", 0.0))
        if second is not None
        else 0.0
    )
    gap = best_score - second_score if second is not None else best_score

    source = str(best.get("api_source") or "unknown")
    status_source = source
    if (
        semantic_type == "APISem"
        and normalize_api_source(source) in HARD_API_SOURCES
        and not bool(best.get("hard_api_eligible"))
    ):
        status_source = "unknown"
    decision = determine_status(
        semantic_type,
        best,
        best_score,
        gap,
        source=status_source,
        evidence_coverage=float(best.get("evidence_coverage", 0.0)),
        hard_conflict=hard_conflict,
    )
    if len(hard_candidates) == 1:
        decision = determine_status(
            semantic_type,
            best,
            1.0,
            1.0,
            source=source,
            evidence_coverage=1.0,
            hard_conflict=False,
        )
        best["validated_score_before_hard_lock"] = best_score
        best["validated_score"] = 1.0
        best["confidence"] = 1.0
        best_score = 1.0
        gap = 1.0 - second_score if second is not None else 1.0

    decision.update(
        {
            "semantic_type": semantic_type,
            "best_target_id": best.get("target_id"),
            "best_candidate_source": best.get("candidate_source"),
            "confidence": float(decision.get("confidence", best_score)),
            "confidence_gap": max(0.0, gap),
            "second_score": max(0.0, second_score),
            "candidate_count": len(values),
            "hard_seed_count": len(hard_candidates),
            "hard_seed_conflict": hard_conflict,
            "policy": CONFIRM_POLICY.get(semantic_type, {}),
        }
    )

    for candidate in values:
        candidate["candidate_status"] = "hypothesis"
        candidate["selected_in_semantic_group"] = (
            str(candidate.get("target_id") or "")
            == str(best.get("target_id") or "")
        )
    best["candidate_status"] = str(decision["status"])
    if decision["status"] == "confirmed_hard":
        for candidate in values[1:]:
            candidate["candidate_status"] = "rejected"
            candidate["rejection_reason"] = (
                "unique_deterministic_api_constraint"
            )

    return decision, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypotheses", required=True)
    parser.add_argument(
        "--nodes",
        help=(
            "Complete USG node JSONL used to independently recover API "
            "recognition provenance"
        ),
    )
    parser.add_argument(
        "--api-provenance",
        help=(
            "Optional compact APISem provenance sidecar. When supplied, "
            "it is used instead of rescanning the complete nodes JSONL."
        ),
    )
    parser.add_argument("--edges", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    parser.add_argument("--representation-prior-weight", type=float, default=0.25)
    parser.add_argument(
        "--confirm-policy",
        help=(
            "Optional JSON policy calibrated only on a frozen Validation "
            "split. Hard APISem sources cannot be overridden."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    active_policy = apply_confirm_policy(args.confirm_policy)

    rows = list(read_jsonl(args.hypotheses))
    candidate_pairs = set()
    candidate_targets = set()
    candidate_queries = set()
    for row in rows:
        query_id = str(row.get("query_id") or "")
        if query_id:
            candidate_queries.add(query_id)
        for candidate in row.get("candidates", []):
            target_id = str(candidate.get("target_id") or "")
            if query_id and target_id:
                candidate_pairs.add((query_id, target_id))
                candidate_targets.add(target_id)

    provenance_input = args.api_provenance or args.nodes
    complete_node_metadata = load_node_metadata(
        provenance_input,
        wanted_ids=candidate_targets,
    )
    node_api_source_counts: Counter = Counter()
    for target_id, metadata in complete_node_metadata.items():
        source_evidence = api_source_evidence_from_objects(
            metadata,
            object_labels=("usg_node",),
        )
        source = str(source_evidence.get("source") or "unknown")
        if source != "unknown":
            node_api_source_counts[source] += 1

    exact_evidence: Dict[Pair, Dict[str, Any]] = {}
    target_context: Dict[str, List[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    target_constraints: Dict[str, List[float]] = defaultdict(list)
    edges_seen = 0

    for edge in read_jsonl(args.edges):
        edges_seen += 1
        source, target = endpoints(edge)
        if not source or not target:
            continue
        family = edge_family(edge)
        weight = edge_weight(edge)
        pair = (source, target)

        if target in candidate_targets and family != "constraint":
            target_context[target][0] += 1.0
            target_context[target][1] += weight

        if pair in candidate_pairs and family != "constraint":
            record = exact_evidence.setdefault(
                pair,
                {
                    "weights": [],
                    "kinds": [],
                    "evidence_types": [],
                    "api_sources": [],
                    "api_source_evidence": [],
                },
            )
            record["weights"].append(weight)
            record["kinds"].append(edge_kind(edge))
            evidence_type = evidence_type_from_objects(edge)
            if evidence_type != "unknown":
                record["evidence_types"].append(evidence_type)
            source_evidence = api_source_evidence_from_objects(
                edge,
                complete_node_metadata.get(target, {}),
                object_labels=("direct_edge", "usg_node"),
            )
            api_source = str(
                source_evidence.get("source") or "unknown"
            )
            if api_source != "unknown":
                record["api_sources"].append(api_source)
                record["api_source_evidence"].append(source_evidence)

        if family == "constraint":
            satisfaction = edge_satisfaction(edge)
            if satisfaction is not None:
                if target in candidate_targets:
                    target_constraints[target].append(satisfaction)
                if source in candidate_targets:
                    target_constraints[source].append(satisfaction)

        if edges_seen % 1000000 == 0:
            print(
                json.dumps(
                    {
                        "phase": "scan_edges",
                        "edges_seen": edges_seen,
                        "exact_pairs_with_evidence": len(exact_evidence),
                        "targets_with_context": len(target_context),
                    }
                ),
                flush=True,
            )

    output = ensure_output_path(args.out, overwrite=args.overwrite)
    output_rows: List[Dict[str, Any]] = []
    reason_counts: Counter = Counter()
    hard_eligible_source_counts: Counter = Counter()
    hard_confirmed_source_counts: Counter = Counter()
    hard_seed_conflicts = 0

    with output.open("w", encoding="utf-8") as writer:
        for row_number, row in enumerate(rows, 1):
            query_id = str(row.get("query_id") or "")
            scored_candidates: List[Dict[str, Any]] = []

            for candidate in row.get("candidates", []):
                value = dict(candidate)
                target_id = str(value.get("target_id") or "")
                pair_record = exact_evidence.get(
                    (query_id, target_id),
                    {},
                )
                direct_values = list(pair_record.get("weights", []))
                if not direct_values:
                    direct_values = [
                        clamp01(item)
                        for item in value.get("direct_edge_weights", [])
                    ]
                if not direct_values and int(
                    value.get("direct_edge_count", 0)
                ) > 0:
                    direct_values = [
                        clamp01(value.get("direct_edge_weight", 0.0))
                    ]

                exact_count = len(direct_values)
                exact_weight = sum(direct_values)
                total_count = int(
                    target_context.get(target_id, [0.0, 0.0])[0]
                )
                total_weight = float(
                    target_context.get(target_id, [0.0, 0.0])[1]
                )
                collaboration = context_collaboration_score(
                    total_count,
                    total_weight,
                    exact_count,
                    exact_weight,
                )
                collaboration_values = (
                    [collaboration] if collaboration > 0 else []
                )
                constraint_values = list(
                    target_constraints.get(target_id, [])
                )

                score = evidence_score(
                    float(
                        value.get(
                            "representation_score",
                            value.get("confidence", 0.0),
                        )
                    ),
                    direct_values,
                    constraint_values,
                    collaboration_values,
                    prior_weight=args.representation_prior_weight,
                )
                api_sources = [
                    normalize_api_source(item)
                    for item in pair_record.get("api_sources", [])
                    if normalize_api_source(item) != "unknown"
                ]
                source_evidence = (
                    pair_record.get("api_source_evidence", [])[0]
                    if pair_record.get("api_source_evidence")
                    else api_source_evidence_from_objects(
                        complete_node_metadata.get(target_id, {}),
                        object_labels=("usg_node",),
                    )
                )
                api_source = str(
                    source_evidence.get("source") or "unknown"
                )
                if api_source == "unknown" and api_sources:
                    api_source = api_sources[0]
                    source_evidence = {
                        "source": api_source,
                        "origin": "direct_edge",
                        "path": "api_source",
                        "raw_value": api_source,
                        "signal_kind": "normalized_edge_provenance",
                    }

                # Candidate-side provenance may be used only when 08 recorded
                # that it came from a direct edge or the complete USG node.
                # A bare candidate field is not trusted for hard confirmation.
                candidate_origin = str(
                    value.get("api_source_origin") or ""
                )
                if (
                    api_source == "unknown"
                    and candidate_origin in ("direct_edge", "usg_node")
                ):
                    candidate_source = normalize_api_source(
                        value.get("api_source")
                    )
                    if candidate_source != "unknown":
                        api_source = candidate_source
                        source_evidence = {
                            "source": candidate_source,
                            "origin": candidate_origin,
                            "path": str(
                                value.get("api_source_path") or ""
                            ),
                            "raw_value": str(
                                value.get("api_source_raw_value")
                                or value.get("api_source")
                                or ""
                            ),
                            "signal_kind": str(
                                value.get("api_source_signal_kind")
                                or "candidate_audit_copy"
                            ),
                        }
                semantic_type = normalize_semantic_type(
                    value.get("target_semantic_type") or ""
                )
                provenance_origin = str(
                    source_evidence.get("origin") or "none"
                )
                # Hard provenance must be re-established from a direct edge
                # or the complete USG node record. A compact-index API name
                # or an embedding-only candidate is not sufficient.
                direct_count = len(pair_record.get("weights", []))
                hard_api_eligible = (
                    semantic_type == "APISem"
                    and direct_count > 0
                    and is_hard_api_source(api_source)
                    and provenance_origin
                    in ("direct_edge", "usg_node")
                )

                value.update(score)
                value.update(
                    {
                        "confidence_before_validation": float(
                            value.get("confidence", 0.0)
                        ),
                        "confidence": float(score["validated_score"]),
                        "direct_edge_count": direct_count,
                        "direct_edge_weight": noisy_or(direct_values),
                        "direct_edge_kinds": (
                            list(pair_record.get("kinds", []))[:16]
                            or list(value.get("direct_edge_kinds", []))[:16]
                        ),
                        "api_source": api_source,
                        "api_source_origin": provenance_origin,
                        "api_source_path": source_evidence.get(
                            "path", ""
                        ),
                        "api_source_raw_value": source_evidence.get(
                            "raw_value", ""
                        ),
                        "api_source_signal_kind": source_evidence.get(
                            "signal_kind", "none"
                        ),
                        "hard_api_eligible": hard_api_eligible,
                        "validation_trace": {
                            "direct_evidence_count": len(direct_values),
                            "constraint_evidence_count": len(
                                constraint_values
                            ),
                            "collaborating_source_count": max(
                                0, total_count - exact_count
                            ),
                        },
                    }
                )
                scored_candidates.append(value)
                if hard_api_eligible:
                    hard_eligible_source_counts[api_source] += 1

            grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for candidate in scored_candidates:
                semantic_type = normalize_semantic_type(
                    candidate.get("target_semantic_type") or ""
                )
                if semantic_type in SEMANTIC_TYPES:
                    grouped[semantic_type].append(candidate)

            semantic_decisions: Dict[str, Dict[str, Any]] = {}
            decided_candidates: List[Dict[str, Any]] = []
            for semantic_type in SEMANTIC_TYPES:
                if semantic_type not in grouped:
                    continue
                decision, group_values = decide_semantic_group(
                    semantic_type,
                    grouped[semantic_type],
                )
                semantic_decisions[semantic_type] = decision
                decided_candidates.extend(group_values)
                reason_counts[
                    str(decision.get("confirmation_source") or "unknown")
                ] += 1
                hard_seed_conflicts += int(
                    bool(decision.get("hard_seed_conflict"))
                )
                if decision.get("status") == "confirmed_hard":
                    hard_confirmed_source_counts[
                        str(
                            decision.get("confirmation_source")
                            or "unknown"
                        )
                    ] += 1

            active_semantic_type, active_decision = select_row_decision(
                semantic_decisions,
                row.get("evidence_type"),
            )
            active_target_id = str(
                active_decision.get("best_target_id") or ""
            )
            best_candidate = next(
                (
                    candidate
                    for candidate in decided_candidates
                    if str(candidate.get("target_id") or "")
                    == active_target_id
                ),
                None,
            )
            decided_candidates.sort(
                key=lambda item: (
                    str(item.get("target_id") or "") != active_target_id,
                    -float(item.get("validated_score", 0.0)),
                    str(item.get("target_id") or ""),
                )
            )

            output_row = dict(row)
            output_row.update(
                {
                    "schema": "task2_222_evidence_validation_v5_3",
                    "status_before_validation": str(
                        row.get("status") or "hypothesis"
                    ),
                    "status": str(
                        active_decision.get("status") or "hypothesis"
                    ),
                    "confidence": float(
                        active_decision.get("confidence", 0.0)
                    ),
                    "confidence_gap": float(
                        active_decision.get("confidence_gap", 0.0)
                    ),
                    "confirmation_source": active_decision.get(
                        "confirmation_source"
                    ),
                    "confirmation_level": int(
                        active_decision.get("confirmation_level", 1)
                    ),
                    "propagation_weight": float(
                        active_decision.get("propagation_weight", 0.0)
                    ),
                    "locked": bool(active_decision.get("locked", False)),
                    "active_semantic_type": active_semantic_type,
                    "semantic_decisions": semantic_decisions,
                    "best_candidate": best_candidate,
                    "preferred_target_id": (
                        best_candidate.get("target_id")
                        if isinstance(best_candidate, Mapping)
                        else None
                    ),
                    "candidates": decided_candidates,
                }
            )
            writer.write(
                json.dumps(output_row, ensure_ascii=False) + "\n"
            )
            output_rows.append(output_row)

            if row_number % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "validate",
                            "records": row_number,
                            "hard_api_seeds": sum(
                                hard_confirmed_source_counts.values()
                            ),
                        }
                    ),
                    flush=True,
                )

    status_summary = summarize_statuses(output_rows)
    summary = {
        "done": True,
        "schema": "task2_222_evidence_validation_summary_v5_3",
        "output": str(output),
        "records": len(output_rows),
        "edges_seen": edges_seen,
        "exact_pairs_with_evidence": len(exact_evidence),
        "targets_with_context": len(target_context),
        "targets_with_explicit_constraint_satisfaction": len(
            target_constraints
        ),
        "complete_candidate_node_records_loaded": len(
            complete_node_metadata
        ),
        "api_provenance_input": str(provenance_input or ""),
        "api_provenance_sidecar_used": bool(args.api_provenance),
        "api_sources_recovered_from_nodes": dict(
            node_api_source_counts
        ),
        "hard_api_eligible_candidate_sources": dict(
            hard_eligible_source_counts
        ),
        "hard_api_eligible_candidate_count": sum(
            hard_eligible_source_counts.values()
        ),
        "hard_api_seed_sources": dict(hard_confirmed_source_counts),
        "hard_api_seed_count": sum(
            hard_confirmed_source_counts.values()
        ),
        "hard_seed_conflicts": hard_seed_conflicts,
        "confirmation_sources": dict(reason_counts),
        "policies": active_policy,
        "confirm_policy_input": str(args.confirm_policy or ""),
        **status_summary,
    }
    if args.summary_out:
        write_json(
            args.summary_out,
            summary,
            overwrite=args.overwrite,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
