#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Task 2.2.2 v5.3 ranking, confirmation, and convergence."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

try:
    import torch
except ImportError:
    torch = None

from module_common_v5_3 import (
    ALL_STATUSES,
    CONFIRMED_STATUSES,
    CONFIRM_POLICY,
    HARD_API_SOURCES,
    SEMANTIC_TYPES,
    SOFT_API_SOURCES,
    apply_confirm_policy,
    clamp01,
    load_json,
    normalize_api_source,
    normalize_semantic_type,
    read_jsonl,
    summarize_statuses,
    write_json,
)


def extract_records(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list) and (
        not value or isinstance(value[0], dict)
    ):
        return value
    if isinstance(value, dict):
        for key in (
            "records",
            "data",
            "items",
            "anchors",
            "examples",
            "samples",
            "train_records",
            "val_records",
            "test_records",
        ):
            nested = value.get(key)
            if isinstance(nested, list) and (
                not nested or isinstance(nested[0], dict)
            ):
                return nested
        for nested in value.values():
            if isinstance(nested, dict):
                records = extract_records(nested)
                if records:
                    return records
    return []


def load_gold(
    cache_paths: Sequence[str],
    jsonl_paths: Sequence[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    gold: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    source_counts: Counter = Counter()
    duplicate_count = 0

    def add(row: Mapping[str, Any], source_path: str) -> None:
        nonlocal duplicate_count
        query_id = str(
            row.get("source_node")
            or row.get("query_id")
            or row.get("source_id")
            or ""
        )
        target_id = str(
            row.get("target_node")
            or row.get("target_id")
            or row.get("gold_target_id")
            or ""
        )
        semantic_type = normalize_semantic_type(
            row.get("target_semantic_type")
            or row.get("semantic_type")
            or row.get("target_node_type")
            or row.get("module")
            or row.get("evidence_type")
            or ""
        )
        if not query_id or not target_id:
            return
        value = {
            "target_id": target_id,
            "semantic_type": semantic_type if semantic_type in SEMANTIC_TYPES else "",
            "split": row.get("split"),
            "gold_source": row.get("gold_source"),
            "source_path": source_path,
        }
        if any(
            item["target_id"] == target_id
            and item["semantic_type"] == value["semantic_type"]
            for item in gold[query_id]
        ):
            duplicate_count += 1
            return
        gold[query_id].append(value)
        source_counts[source_path] += 1

    for path in cache_paths:
        if not path or not Path(path).exists():
            continue
        if torch is None:
            raise RuntimeError(
                "PyTorch is required to read --gold-cache files; "
                "use --gold-jsonl in a symbolic-only environment"
            )
        payload = torch.load(path, map_location="cpu")
        for row in extract_records(payload):
            add(row, path)
    for path in jsonl_paths:
        for row in read_jsonl(path):
            add(row, path)
    return dict(gold), {
        "source_record_counts": dict(source_counts),
        "unique_queries": len(gold),
        "query_target_labels": sum(len(values) for values in gold.values()),
        "duplicate_labels": duplicate_count,
    }


def ranking_metrics(
    rows: Sequence[Mapping[str, Any]],
    gold: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    totals: Dict[str, Counter] = defaultdict(Counter)
    reciprocal: Dict[str, float] = defaultdict(float)
    confirmation_by_status: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )

    def score(candidate: Mapping[str, Any]) -> float:
        return clamp01(
            candidate.get(
                "validated_score",
                candidate.get("confidence", candidate.get("representation_score", 0.0)),
            )
        )

    for row in rows:
        query_id = str(row.get("query_id") or "")
        labels = list(gold.get(query_id, []))
        if not labels:
            continue
        candidates_by_module: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for candidate in row.get("candidates", []) or []:
            semantic_type = normalize_semantic_type(
                candidate.get("target_semantic_type") or ""
            )
            if semantic_type in SEMANTIC_TYPES:
                candidates_by_module[semantic_type].append(candidate)
        decisions = row.get("semantic_decisions")
        decisions = decisions if isinstance(decisions, Mapping) else {}

        labels_by_module: Dict[str, Set[str]] = defaultdict(set)
        for label in labels:
            semantic_type = normalize_semantic_type(label.get("semantic_type") or "")
            target_id = str(label.get("target_id") or "")
            if semantic_type not in SEMANTIC_TYPES:
                for module, candidates in candidates_by_module.items():
                    if any(str(value.get("target_id") or "") == target_id for value in candidates):
                        semantic_type = module
                        break
            if semantic_type in SEMANTIC_TYPES and target_id:
                labels_by_module[semantic_type].add(target_id)

        for semantic_type, target_ids in labels_by_module.items():
            ordered = sorted(
                candidates_by_module.get(semantic_type, []),
                key=lambda value: (-score(value), str(value.get("target_id") or "")),
            )
            candidate_ids = [str(value.get("target_id") or "") for value in ordered]
            totals[semantic_type]["n"] += 1
            ranks = [candidate_ids.index(target) + 1 for target in target_ids if target in candidate_ids]
            if ranks:
                rank = min(ranks)
                totals[semantic_type]["coverage"] += 1
                totals[semantic_type]["hit1"] += int(rank == 1)
                totals[semantic_type]["hit5"] += int(rank <= 5)
                reciprocal[semantic_type] += 1.0 / rank
            decision = decisions.get(semantic_type, {})
            status = str(decision.get("status") or "hypothesis")
            selected = str(decision.get("best_target_id") or "")
            if status in CONFIRMED_STATUSES:
                totals[semantic_type]["confirmed"] += 1
                correct = int(selected in target_ids)
                totals[semantic_type]["correct_confirmed"] += correct
                confirmation_by_status[semantic_type][status][0] += 1
                confirmation_by_status[semantic_type][status][1] += correct

    def report(modules: Sequence[str]) -> Dict[str, Any]:
        aggregate = Counter()
        rr = 0.0
        status_values: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for module in modules:
            aggregate.update(totals[module])
            rr += reciprocal[module]
            for status, values in confirmation_by_status[module].items():
                status_values[status][0] += values[0]
                status_values[status][1] += values[1]
        n = aggregate["n"]
        confirmed = aggregate["confirmed"]
        return {
            "available": n > 0,
            "n": n,
            "candidate_coverage": aggregate["coverage"] / max(n, 1),
            "recall@1": aggregate["hit1"] / max(n, 1),
            "recall@5": aggregate["hit5"] / max(n, 1),
            "mrr": rr / max(n, 1),
            "confirmed": confirmed,
            "correct_confirmed": aggregate["correct_confirmed"],
            "confirmed_precision": aggregate["correct_confirmed"] / max(confirmed, 1),
            "confirmation_by_status": {
                status: {
                    "confirmed": values[0],
                    "correct": values[1],
                    "precision": values[1] / max(values[0], 1),
                }
                for status, values in status_values.items()
            },
        }

    overall = report(SEMANTIC_TYPES)
    overall["by_module"] = {
        module: report((module,)) for module in SEMANTIC_TYPES
    }
    return overall


def audit_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    invalid_statuses: Counter = Counter()
    hard_errors: Counter = Counter()
    soft_errors: Counter = Counter()
    hypothesis_errors: Counter = Counter()
    semantic_records: Counter = Counter()
    multiple_candidate_groups = 0
    hard_confirmed = 0
    soft_confirmed = 0

    for row in rows:
        row_status = str(row.get("status") or "")
        if row_status not in ALL_STATUSES:
            invalid_statuses[row_status] += 1
        decisions = row.get("semantic_decisions")
        if not isinstance(decisions, Mapping):
            invalid_statuses["missing_semantic_decisions"] += 1
            continue
        candidates_by_type: Dict[str, List[Mapping[str, Any]]] = defaultdict(
            list
        )
        for candidate in row.get("candidates", []):
            semantic_type = normalize_semantic_type(
                candidate.get("target_semantic_type") or ""
            )
            candidates_by_type[semantic_type].append(candidate)

        for semantic_type, decision in decisions.items():
            normalized = normalize_semantic_type(semantic_type)
            if normalized not in SEMANTIC_TYPES:
                invalid_statuses["unknown_semantic_type"] += 1
                continue
            if not isinstance(decision, Mapping):
                invalid_statuses["non_mapping_decision"] += 1
                continue
            semantic_records[normalized] += 1
            if len(candidates_by_type.get(normalized, [])) > 1:
                multiple_candidate_groups += 1
            status = str(decision.get("status") or "")
            if status not in ALL_STATUSES:
                invalid_statuses[status] += 1
                continue

            best_target = str(decision.get("best_target_id") or "")
            best = next(
                (
                    candidate
                    for candidate in candidates_by_type.get(
                        normalized, []
                    )
                    if str(candidate.get("target_id") or "")
                    == best_target
                ),
                None,
            )

            if status == "confirmed_hard":
                hard_confirmed += 1
                source = normalize_api_source(
                    decision.get("confirmation_source")
                )
                if normalized != "APISem":
                    hard_errors["non_api_hard_confirmation"] += 1
                if source not in HARD_API_SOURCES:
                    hard_errors["non_deterministic_source"] += 1
                if abs(float(decision.get("confidence", 0.0)) - 1.0) > 1e-9:
                    hard_errors["confidence_not_one"] += 1
                if int(decision.get("confirmation_level", 0)) != 3:
                    hard_errors["level_not_three"] += 1
                if abs(
                    float(decision.get("propagation_weight", 0.0)) - 1.0
                ) > 1e-9:
                    hard_errors["propagation_weight_not_one"] += 1
                if not bool(decision.get("locked")):
                    hard_errors["not_locked"] += 1
                if not isinstance(best, Mapping) or not bool(
                    best.get("hard_api_eligible")
                ):
                    hard_errors["missing_direct_hard_evidence"] += 1
                if (
                    not isinstance(best, Mapping)
                    or str(best.get("api_source_origin") or "")
                    not in ("direct_edge", "usg_node")
                ):
                    hard_errors["untrusted_provenance_origin"] += 1

            elif status == "confirmed_soft":
                soft_confirmed += 1
                policy = CONFIRM_POLICY[normalized]
                if not bool(policy.get("soft_confirmation_enabled", True)):
                    soft_errors[
                        "soft_confirmation_disabled_by_validation"
                    ] += 1
                score = clamp01(decision.get("confidence", 0.0))
                gap = max(
                    0.0,
                    float(decision.get("confidence_gap", 0.0)),
                )
                coverage = (
                    clamp01(best.get("evidence_coverage", 0.0))
                    if isinstance(best, Mapping)
                    else 0.0
                )
                hysteresis = bool(decision.get("hysteresis_retained"))
                minimum_score = (
                    float(decision.get("hysteresis_min_evidence", 0.0))
                    if hysteresis
                    else float(policy["min_evidence"])
                )
                minimum_gap = (
                    float(decision.get("hysteresis_min_gap", 0.0))
                    if hysteresis
                    else float(policy["gap_threshold"])
                )
                if score + 1e-12 < minimum_score:
                    soft_errors["score_below_policy"] += 1
                if gap + 1e-12 < minimum_gap:
                    soft_errors["gap_below_policy"] += 1
                if coverage + 1e-12 < float(
                    policy["min_evidence_coverage"]
                ):
                    soft_errors["coverage_below_policy"] += 1
                if int(decision.get("confirmation_level", 0)) != 2:
                    soft_errors["level_not_two"] += 1
                if abs(
                    float(decision.get("propagation_weight", 0.0))
                    - score
                ) > 1e-9:
                    soft_errors["propagation_weight_not_confidence"] += 1
                if not bool(decision.get("locked")):
                    soft_errors["not_locked"] += 1
                if (
                    normalized == "APISem"
                    and (
                        not isinstance(best, Mapping)
                        or normalize_api_source(
                            best.get("api_source")
                        )
                        not in SOFT_API_SOURCES
                    )
                ):
                    soft_errors[
                        "api_soft_source_not_allowed"
                    ] += 1

            elif status == "hypothesis":
                if float(decision.get("propagation_weight", 0.0)) > 1e-12:
                    hypothesis_errors[
                        "hypothesis_has_propagation_weight"
                    ] += 1

    return {
        "invalid_statuses": dict(invalid_statuses),
        "hard_confirmation_errors": dict(hard_errors),
        "soft_confirmation_errors": dict(soft_errors),
        "hypothesis_errors": dict(hypothesis_errors),
        "hard_confirmed": hard_confirmed,
        "soft_confirmed": soft_confirmed,
        "semantic_decision_records": dict(semantic_records),
        "multiple_candidate_groups_preserved": multiple_candidate_groups,
        "schema_valid": (
            not invalid_statuses
            and not hard_errors
            and not soft_errors
            and not hypothesis_errors
        ),
    }


def convergence_audit(summary: Mapping[str, Any]) -> Dict[str, Any]:
    trace = summary.get("iteration_trace")
    trace = trace if isinstance(trace, list) else []
    monotonic = all(
        float(trace[index].get("global_consistency", 0.0)) + 1e-12
        >= float(trace[index - 1].get("global_consistency", 0.0))
        for index in range(1, len(trace))
    )
    closed_loop = summary.get("neural_symbolic_closed_loop")
    closed_loop = closed_loop if isinstance(closed_loop, Mapping) else {}
    trace_audited = bool(trace) and all(
        isinstance(item, Mapping)
        and "raw_global_delta" in item
        and "message_score_feedback" in item
        and "embedding_rescore" in item
        and "iterative_retrieval" in item
        for item in trace[1:]
    )
    return {
        "summary_available": bool(summary),
        "iterations": int(summary.get("iterations_run", 0) or 0),
        "stop_reason": summary.get("stop_reason"),
        "initial_global_consistency": float(
            summary.get("initial_global_consistency", 0.0) or 0.0
        ),
        "final_global_consistency": float(
            summary.get("final_global_consistency", 0.0) or 0.0
        ),
        "global_consistency_monotonic": monotonic,
        "raw_consistency_decrease_count": int(
            summary.get("raw_consistency_decrease_count", 0) or 0
        ),
        "hard_constraint_conflicts": int(
            summary.get("hard_constraint_conflicts", 0) or 0
        ),
        "implementation_revision": summary.get("implementation_revision"),
        "closed_loop": dict(closed_loop),
        "trace_audited": trace_audited,
        "max_score_drop_per_round": float(
            summary.get("max_score_drop_per_round", 0.0) or 0.0
        ),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--gold-cache", action="append", default=[])
    parser.add_argument("--gold-jsonl", action="append", default=[])
    parser.add_argument(
        "--confirm-policy",
        help="The same Validation-calibrated policy used by reasoning.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    active_policy = apply_confirm_policy(args.confirm_policy)

    rows = list(read_jsonl(args.input))
    summary = load_json(args.summary) if args.summary else {}
    gold, gold_audit = load_gold(args.gold_cache, args.gold_jsonl)

    status_summary = summarize_statuses(rows)
    audit = audit_rows(rows)
    ranking = ranking_metrics(rows, gold)
    convergence = convergence_audit(summary)

    warnings = []
    if audit["hard_confirmed"] == 0:
        warnings.append(
            "No deterministic APISem seed was found. Check whether import/"
            "symbol/exact-signature provenance is present on direct USG edges."
        )
    if not gold:
        warnings.append(
            "No gold cache/JSONL was supplied; ranking and confirmation "
            "precision cannot be evaluated."
        )
    if convergence["raw_consistency_decrease_count"] > 0:
        warnings.append(
            "Raw S_global decreased in at least one round. v5.5 permits "
            "audited non-monotonic raw feedback; the reported compatibility "
            "envelope remains non-decreasing."
        )
    if convergence["hard_constraint_conflicts"] > 0:
        warnings.append(
            "Deterministic constraint conflicts were detected and preserved "
            "for audit; inspect the affected candidates before full-scale use."
        )

    checks = {
        "schema_and_policy_valid": audit["schema_valid"],
        "global_consistency_monotonic": convergence[
            "global_consistency_monotonic"
        ],
        "raw_feedback_changes_audited": convergence["trace_audited"],
        "reasoning_revision_compatible": (
            convergence["implementation_revision"] in {
                "v5_5_bidirectional_feedback_and_iterative_retrieval",
                "v5_6_irreversible_confirmation_transactional_raw_sglobal",
            }
        ),
        "closed_loop_configuration_recorded": all(
            name in convergence["closed_loop"]
            for name in (
                "neural_state_updated",
                "symbolic_messages_signed",
                "candidate_scores_bidirectional",
                "soft_decisions_revisable",
                "candidate_set_retrieved_each_round",
                "gold_used_during_inference",
            )
        ),
        "no_hard_constraint_conflict": (
            convergence["hard_constraint_conflicts"] == 0
        ),
        "multi_candidate_hypotheses_preserved": (
            audit["multiple_candidate_groups_preserved"] > 0
        ),
        "hard_seed_provenance_valid": not audit[
            "hard_confirmation_errors"
        ],
    }
    result = {
        "schema": "task2_222_evaluation_v5_5",
        "input": args.input,
        "records": len(rows),
        "gold_queries": len(gold),
        "gold_audit": gold_audit,
        "confirm_policy_input": str(args.confirm_policy or ""),
        "policies": active_policy,
        **status_summary,
        "audit": audit,
        "ranking": ranking,
        "convergence": convergence,
        "checks": checks,
        "pass_structural_checks": all(checks.values()),
        "warnings": warnings,
    }
    write_json(args.out, result, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
