#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation-guarded structural/ranking evaluation for Task 2.2.3.

The formal overall ranking unit is one ``(query_id, semantic_type)`` object.
Scores from different modules are never compared against each other.  The old
cross-module flattened pool is retained under an explicitly diagnostic key so
historical runs remain explainable without contaminating the formal metric.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

try:
    import torch
except ImportError:  # JSONL-only evaluation remains available.
    torch = None

from module_common_223_v5_3 import (
    ALL_STATUSES,
    SEMANTIC_TYPES,
    normalize_semantic_type,
    read_jsonl,
    write_json,
)


RANKING_STAGES = (
    "before_223",
    "after_module_heads",
    "after_collaboration",
)
CONFIRMATION_STATUSES = ("confirmed_hard", "confirmed_soft")


def extract_records(value: Any) -> List[Dict[str, Any]]:
    """Extract a list of dict records from common torch-cache layouts."""
    if isinstance(value, (list, tuple)) and (
        not value or isinstance(value[0], Mapping)
    ):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        for key in (
            "records", "data", "items", "anchors", "examples", "samples",
            "train_records", "val_records", "test_records",
        ):
            nested = value.get(key)
            if isinstance(nested, (list, tuple)) and (
                not nested or isinstance(nested[0], Mapping)
            ):
                return [dict(item) for item in nested if isinstance(item, Mapping)]
        for nested in value.values():
            if isinstance(nested, Mapping):
                records = extract_records(nested)
                if records:
                    return records
    return []


def semantic_type_hint(value: Any) -> str:
    normalized = normalize_semantic_type(value)
    if normalized in SEMANTIC_TYPES:
        return normalized
    text = re.sub(r"[^a-z]+", "_", str(value or "").lower())
    for token, semantic_type in (
        ("interface", "InterfaceSem"),
        ("layout", "LayoutSem"),
        ("api", "APISem"),
        ("type", "TypeSem"),
    ):
        if token in text:
            return semantic_type
    return ""


def gold_item(row: Mapping[str, Any], source_path: str) -> Tuple[str, Dict[str, Any]]:
    query_id = str(
        row.get("query_id") or row.get("source_node")
        or row.get("source_id") or ""
    )
    target_id = str(
        row.get("target_id") or row.get("target_node")
        or row.get("gold_target_id") or ""
    )
    semantic_type = ""
    for key in (
        "target_semantic_type", "semantic_type", "target_node_type",
        "module", "evidence_type",
    ):
        semantic_type = semantic_type_hint(row.get(key))
        if semantic_type:
            break
    if not semantic_type:
        semantic_type = semantic_type_hint(target_id)
    return query_id, {
        "target_id": target_id,
        "semantic_type": semantic_type or None,
        "source_path": source_path,
    }


def load_gold(
    cache_paths: Sequence[str], jsonl_paths: Sequence[str]
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    gold: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    source_records = Counter()
    duplicate_same = 0
    semantic_type_conflicts = []

    def add(row: Mapping[str, Any], source_path: str) -> None:
        nonlocal duplicate_same
        query_id, item = gold_item(row, source_path)
        if not query_id or not item["target_id"]:
            return
        source_records[source_path] += 1
        previous = next(
            (
                value for value in gold[query_id]
                if value["target_id"] == item["target_id"]
            ),
            None,
        )
        if previous is not None:
            duplicate_same += 1
            if not previous.get("semantic_type") and item.get("semantic_type"):
                previous["semantic_type"] = item["semantic_type"]
            elif (
                previous.get("semantic_type")
                and item.get("semantic_type")
                and previous["semantic_type"] != item["semantic_type"]
            ):
                semantic_type_conflicts.append({
                    "query_id": query_id,
                    "target_id": item["target_id"],
                    "first_semantic_type": previous["semantic_type"],
                    "second_semantic_type": item["semantic_type"],
                    "first_source": previous["source_path"],
                    "second_source": source_path,
                })
            return
        gold[query_id].append(item)

    for path in cache_paths:
        if not path:
            continue
        if not Path(path).is_file():
            raise FileNotFoundError("Gold cache does not exist: %s" % path)
        if torch is None:
            raise RuntimeError(
                "PyTorch is required to read --gold-cache; use --gold-jsonl "
                "in a symbolic-only environment"
            )
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # Older PyTorch.
            payload = torch.load(path, map_location="cpu")
        for row in extract_records(payload):
            add(row, path)

    for path in jsonl_paths:
        if not Path(path).is_file():
            raise FileNotFoundError("Gold JSONL does not exist: %s" % path)
        for row in read_jsonl(path):
            add(row, path)

    gold_result = dict(gold)
    return gold_result, {
        "cache_paths": list(cache_paths),
        "jsonl_paths": list(jsonl_paths),
        "source_record_counts": dict(source_records),
        "unique_queries": len(gold_result),
        "unique_query_target_labels": sum(
            len(values) for values in gold_result.values()
        ),
        "multi_target_queries": sum(
            1 for values in gold_result.values() if len(values) > 1
        ),
        "duplicate_same_labels": duplicate_same,
        "semantic_type_conflict_count": len(semantic_type_conflicts),
        "semantic_type_conflict_examples": semantic_type_conflicts[:20],
    }


def group_candidates(row: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in row.get("candidates", []) or []:
        semantic_type = normalize_semantic_type(
            candidate.get("target_semantic_type")
        )
        if semantic_type in SEMANTIC_TYPES:
            grouped[semantic_type].append(dict(candidate))
    return grouped


def resolve_gold_semantic_type(
    row: Mapping[str, Any], item: Mapping[str, Any]
) -> Tuple[str, str]:
    explicit = semantic_type_hint(item.get("semantic_type"))
    if explicit:
        return explicit, "gold_field_or_node_id"
    target_id = str(item.get("target_id") or "")
    matches = []
    for semantic_type, candidates in group_candidates(row).items():
        if any(
            str(candidate.get("target_id") or "") == target_id
            for candidate in candidates
        ):
            matches.append(semantic_type)
    if len(matches) == 1:
        return matches[0], "candidate_membership"
    return "", "ambiguous" if len(matches) > 1 else "unresolved"


def empty_rank_accumulator() -> Dict[str, float]:
    return {
        "n": 0, "candidate_coverage": 0, "hit1": 0,
        "hit5": 0, "reciprocal_rank": 0.0,
    }


def update_rank(
    accumulator: MutableMapping[str, float],
    candidates: Sequence[Mapping[str, Any]],
    target_ids: Sequence[str],
    score_key: str,
) -> None:
    accumulator["n"] += 1
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.get(score_key, candidate.get("confidence", 0.0)) or 0.0),
            str(candidate.get("target_id") or ""),
        ),
    )
    ids = [str(candidate.get("target_id") or "") for candidate in ordered]
    ranks = [ids.index(target_id) + 1 for target_id in target_ids if target_id in ids]
    if not ranks:
        return
    rank = min(ranks)
    accumulator["candidate_coverage"] += 1
    accumulator["hit1"] += int(rank == 1)
    accumulator["hit5"] += int(rank <= 5)
    accumulator["reciprocal_rank"] += 1.0 / rank


def finish_rank(accumulator: Mapping[str, float]) -> Dict[str, Any]:
    n = int(accumulator.get("n", 0) or 0)
    if n == 0:
        return {
            "available": False, "n": 0, "candidate_coverage": None,
            "recall@1": None, "recall@5": None, "mrr": None,
        }
    return {
        "available": True,
        "n": n,
        "candidate_coverage": accumulator["candidate_coverage"] / n,
        "recall@1": accumulator["hit1"] / n,
        "recall@5": accumulator["hit5"] / n,
        "mrr": accumulator["reciprocal_rank"] / n,
    }


def metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    result = {}
    for key in ("candidate_coverage", "recall@1", "recall@5", "mrr"):
        left, right = before.get(key), after.get(key)
        result[key] = right - left if left is not None and right is not None else None
    return result


def finish_ranking_stages(
    stages: Mapping[str, Mapping[str, float]]
) -> Dict[str, Any]:
    result = {stage: finish_rank(stages[stage]) for stage in RANKING_STAGES}
    result["delta_before_to_final"] = metric_delta(
        result["before_223"], result["after_collaboration"]
    )
    return result


def macro_ranking_stages(
    ranking_by_module: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Unweighted macro mean across the four available semantic modules."""
    result: Dict[str, Any] = {}
    for stage in RANKING_STAGES:
        available = [
            ranking_by_module[semantic_type][stage]
            for semantic_type in SEMANTIC_TYPES
            if ranking_by_module[semantic_type][stage].get("available") is True
        ]
        if not available:
            result[stage] = {
                "available": False,
                "modules": 0,
                "module_names": [],
                "candidate_coverage": None,
                "recall@1": None,
                "recall@5": None,
                "mrr": None,
            }
            continue
        result[stage] = {
            "available": True,
            "modules": len(available),
            "module_names": [
                semantic_type for semantic_type in SEMANTIC_TYPES
                if ranking_by_module[semantic_type][stage].get("available")
                is True
            ],
            **{
                key: sum(float(value[key]) for value in available) / len(available)
                for key in ("candidate_coverage", "recall@1", "recall@5", "mrr")
            },
        }
    result["delta_before_to_final"] = metric_delta(
        result["before_223"], result["after_collaboration"]
    )
    return result


def new_confirmation_accumulator() -> Dict[str, Any]:
    return {
        "total_objects": 0,
        "gold_objects": 0,
        "statuses": {
            status: {"confirmed": 0, "evaluable": 0, "correct": 0}
            for status in CONFIRMATION_STATUSES
        },
    }


def finish_confirmation_status(
    values: Mapping[str, int], total_objects: int, gold_objects: int
) -> Dict[str, Any]:
    confirmed = int(values.get("confirmed", 0) or 0)
    evaluable = int(values.get("evaluable", 0) or 0)
    correct = int(values.get("correct", 0) or 0)
    return {
        "confirmed": confirmed,
        "evaluable_confirmed": evaluable,
        "correct_confirmed": correct,
        "unlabeled_confirmed": confirmed - evaluable,
        "precision": correct / evaluable if evaluable else None,
        "coverage_all_objects": confirmed / total_objects if total_objects else None,
        "coverage_gold_objects": evaluable / gold_objects if gold_objects else None,
    }


def finish_confirmation(values: Mapping[str, Any]) -> Dict[str, Any]:
    total_objects = int(values.get("total_objects", 0) or 0)
    gold_objects = int(values.get("gold_objects", 0) or 0)
    statuses = {
        status: finish_confirmation_status(
            values["statuses"][status], total_objects, gold_objects
        )
        for status in CONFIRMATION_STATUSES
    }
    combined_raw = {
        key: sum(
            int(values["statuses"][status].get(key, 0) or 0)
            for status in CONFIRMATION_STATUSES
        )
        for key in ("confirmed", "evaluable", "correct")
    }
    return {
        "total_semantic_objects": total_objects,
        "gold_labeled_objects": gold_objects,
        "confirmed_hard": statuses["confirmed_hard"],
        "confirmed_soft": statuses["confirmed_soft"],
        "confirmed_combined": finish_confirmation_status(
            combined_raw, total_objects, gold_objects
        ),
        "coverage_note": (
            "coverage_all_objects needs no gold; coverage_gold_objects and "
            "precision use only gold-labeled semantic objects"
        ),
    }


def api_provenance_category(
    decision: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
    lineage_target_ids: Sequence[str],
) -> str:
    """Separate direct import/symbol evidence from reconstructed lineage."""
    target_id = str(decision.get("best_target_id") or "")
    candidate = next(
        (
            value for value in candidates
            if str(value.get("target_id") or "") == target_id
        ),
        {},
    )
    if target_id in lineage_target_ids:
        return "lineage_sidecar"
    origin = str(candidate.get("api_source_origin") or "unknown").lower()
    signal = str(candidate.get("api_source_signal_kind") or "").lower()
    if "lineage" in origin or "lineage" in signal:
        return "lineage_sidecar"
    if origin in ("direct_edge", "usg_node"):
        return "real_usg_import_or_symbol"
    return "unknown_or_unrecorded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--collaboration-summary", required=True)
    parser.add_argument("--gold-cache", action="append", default=[])
    parser.add_argument("--gold-jsonl", action="append", default=[])
    parser.add_argument("--api-provenance-sidecar", default="")
    parser.add_argument(
        "--ablation",
        choices=(
            "full", "no_api_hard_seed", "no_type_layout",
            "no_interface_feedback", "no_api_to_interface",
            "no_type_to_interface", "no_interface_incoming",
            "no_collaboration",
        ),
        default="full",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline_rows = list(read_jsonl(args.baseline))
    baseline = {str(row.get("query_id") or ""): row for row in baseline_rows}
    rows = list(read_jsonl(args.input))
    summary = json.loads(
        Path(args.collaboration_summary).read_text(encoding="utf-8")
    )
    gold, gold_audit = load_gold(args.gold_cache, args.gold_jsonl)
    lineage_target_ids = set()
    if args.api_provenance_sidecar:
        if not Path(args.api_provenance_sidecar).is_file():
            raise FileNotFoundError(
                "API provenance sidecar does not exist: %s"
                % args.api_provenance_sidecar
            )
        for provenance_row in read_jsonl(args.api_provenance_sidecar):
            target_id = str(
                provenance_row.get("node_id")
                or provenance_row.get("target_id")
                or ""
            )
            if target_id:
                lineage_target_ids.add(target_id)

    invalid = Counter()
    status_counts: Dict[str, Counter] = defaultdict(Counter)
    source_counts: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    multi_candidate = 0
    locked_violations = int(summary.get("locked_decision_violations", 0) or 0)
    query_ids = [str(row.get("query_id") or "") for row in rows]
    baseline_ids = [str(row.get("query_id") or "") for row in baseline_rows]

    rank_accumulators = {
        # ``overall`` is the micro accumulator over query-module objects.
        "overall": {stage: empty_rank_accumulator() for stage in RANKING_STAGES},
        **{
            semantic_type: {
                stage: empty_rank_accumulator() for stage in RANKING_STAGES
            }
            for semantic_type in SEMANTIC_TYPES
        },
    }
    legacy_cross_module_accumulators = {
        stage: empty_rank_accumulator() for stage in RANKING_STAGES
    }
    confirmation = {
        "overall": new_confirmation_accumulator(),
        **{
            semantic_type: new_confirmation_accumulator()
            for semantic_type in SEMANTIC_TYPES
        },
    }
    api_provenance_report = {
        category: Counter()
        for category in (
            "real_usg_import_or_symbol",
            "lineage_sidecar",
            "unknown_or_unrecorded",
        )
    }
    matched_gold_queries = 0
    matched_gold_labels = 0
    unresolved_gold_modules = Counter()
    resolved_gold: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        query_id = str(row.get("query_id") or "")
        items = gold.get(query_id)
        if not items:
            continue
        matched_gold_queries += 1
        matched_gold_labels += len(items)
        resolved_items = []
        for item in items:
            semantic_type, resolution = resolve_gold_semantic_type(row, item)
            resolved_items.append({
                **item,
                "semantic_type": semantic_type or None,
                "module_resolution": resolution,
            })
            if not semantic_type:
                unresolved_gold_modules[resolution] += 1
        resolved_gold[query_id] = resolved_items
        labeled_modules = {
            str(item.get("semantic_type") or "")
            for item in resolved_items
            if item.get("semantic_type") in SEMANTIC_TYPES
        }
        for semantic_type in labeled_modules:
            confirmation[semantic_type]["gold_objects"] += 1
            confirmation["overall"]["gold_objects"] += 1

    for row in rows:
        query_id = str(row.get("query_id") or "")
        decisions = (
            row.get("semantic_decisions")
            if isinstance(row.get("semantic_decisions"), Mapping)
            else {}
        )
        final_groups = group_candidates(row)
        baseline_groups = group_candidates(baseline.get(query_id, {}))

        semantic_types_present = set(final_groups)
        semantic_types_present.update(
            normalize_semantic_type(value) for value in decisions
        )
        for semantic_type in SEMANTIC_TYPES:
            if semantic_type not in semantic_types_present:
                continue
            decision = decisions.get(semantic_type)
            if not isinstance(decision, Mapping):
                invalid["missing_or_invalid_semantic_decision"] += 1
                continue
            status = str(decision.get("status") or "")
            status_counts[semantic_type][status] += 1
            confirmation[semantic_type]["total_objects"] += 1
            confirmation["overall"]["total_objects"] += 1
            if status not in ALL_STATUSES:
                invalid["invalid_status"] += 1
            if status == "confirmed_hard" and not bool(decision.get("locked")):
                invalid["unlocked_hard_confirmation"] += 1
            if status == "hypothesis" and float(
                decision.get("propagation_weight", 0.0) or 0.0
            ) > 1e-12:
                invalid["hypothesis_propagates"] += 1
            if len(final_groups.get(semantic_type, [])) > 1:
                multi_candidate += 1

            if status in CONFIRMATION_STATUSES:
                source = str(decision.get("confirmation_source") or "unknown")
                source_counts[semantic_type][status][source] += 1
                if semantic_type == "APISem" and status == "confirmed_hard":
                    category = api_provenance_category(
                        decision,
                        final_groups.get(semantic_type, []),
                        lineage_target_ids,
                    )
                    api_provenance_report[category][source] += 1
                confirmation[semantic_type]["statuses"][status]["confirmed"] += 1
                confirmation["overall"]["statuses"][status]["confirmed"] += 1
                module_items = [
                    item for item in resolved_gold.get(query_id, [])
                    if item.get("semantic_type") == semantic_type
                ]
                if module_items:
                    accepted_targets = {
                        str(item.get("target_id") or "")
                        for item in module_items
                    }
                    correct = int(
                        str(decision.get("best_target_id") or "")
                        in accepted_targets
                    )
                    for key in (semantic_type, "overall"):
                        confirmation[key]["statuses"][status]["evaluable"] += 1
                        confirmation[key]["statuses"][status]["correct"] += correct

        items = resolved_gold.get(query_id)
        if not items:
            continue
        # Historical diagnostic only: module scores are not calibrated onto a
        # common scale, so this pool must never be used as formal overall R@k.
        baseline_all = [
            candidate for candidates in baseline_groups.values()
            for candidate in candidates
        ]
        final_all = [
            candidate for candidates in final_groups.values()
            for candidate in candidates
        ]
        target_ids = [str(item.get("target_id") or "") for item in items]
        update_rank(
            legacy_cross_module_accumulators["before_223"], baseline_all,
            target_ids, "confidence",
        )
        update_rank(
            legacy_cross_module_accumulators["after_module_heads"], final_all,
            target_ids,
            "score_after_module_fusion_before_collaboration_223_v5_3",
        )
        update_rank(
            legacy_cross_module_accumulators["after_collaboration"], final_all,
            target_ids, "score_223_v5_3",
        )
        module_targets = defaultdict(list)
        for item in items:
            semantic_type = str(item.get("semantic_type") or "")
            if semantic_type in SEMANTIC_TYPES:
                module_targets[semantic_type].append(
                    str(item.get("target_id") or "")
                )
        for semantic_type, targets in module_targets.items():
            stage_inputs = (
                (
                    "before_223",
                    baseline_groups.get(semantic_type, []),
                    "confidence",
                ),
                (
                    "after_module_heads",
                    final_groups.get(semantic_type, []),
                    "score_after_module_fusion_before_collaboration_223_v5_3",
                ),
                (
                    "after_collaboration",
                    final_groups.get(semantic_type, []),
                    "score_223_v5_3",
                ),
            )
            for stage, candidates, score_key in stage_inputs:
                update_rank(
                    rank_accumulators[semantic_type][stage],
                    candidates,
                    targets,
                    score_key,
                )
                update_rank(
                    rank_accumulators["overall"][stage],
                    candidates,
                    targets,
                    score_key,
                )

    ranking_overall = finish_ranking_stages(rank_accumulators["overall"])
    ranking_by_module = {
        semantic_type: finish_ranking_stages(rank_accumulators[semantic_type])
        for semantic_type in SEMANTIC_TYPES
    }
    ranking_macro = macro_ranking_stages(ranking_by_module)
    ranking_cross_module_diagnostic = finish_ranking_stages(
        legacy_cross_module_accumulators
    )
    confirmation_quality = {
        "overall": finish_confirmation(confirmation["overall"]),
        "by_module": {
            semantic_type: finish_confirmation(confirmation[semantic_type])
            for semantic_type in SEMANTIC_TYPES
        },
        "confirmation_source_counts": {
            semantic_type: {
                status: dict(counts) for status, counts in statuses.items()
            }
            for semantic_type, statuses in source_counts.items()
        },
    }

    consistency_trace = summary.get("consistency_trace") or []
    summary_ablation = str(summary.get("ablation") or "full")
    consistency = {
        "metric_role": "raw cross-module constraint-consistency proxy",
        "definition": summary.get("consistency_definition"),
        "initial_raw_s_global": summary.get("initial_raw_s_global"),
        "final_raw_s_global": summary.get("final_raw_s_global"),
        "raw_s_global_delta": summary.get("raw_s_global_delta"),
        "initial_constraint_coverage": summary.get("initial_constraint_coverage"),
        "final_constraint_coverage": summary.get("final_constraint_coverage"),
        "rounds": consistency_trace,
    }

    collaboration_edges = int(summary.get("collaboration_edges", 0) or 0)
    directions = summary.get("directions")
    directions = directions if isinstance(directions, Mapping) else {}
    expected_type_layout = args.ablation not in (
        "no_type_layout", "no_collaboration"
    )
    has_type_layout = (
        "TypeSem->LayoutSem" in directions
        and "LayoutSem->TypeSem" in directions
    )
    checks = {
        "records_preserved": len(rows) == len(baseline_rows),
        "query_ids_preserved": Counter(query_ids) == Counter(baseline_ids),
        "statuses_valid": not invalid,
        "locked_222_decisions_preserved": locked_violations == 0,
        "multi_candidate_uncertainty_preserved": multi_candidate > 0,
        "collaboration_edges_configuration_valid": (
            collaboration_edges == 0
            if args.ablation == "no_collaboration"
            else collaboration_edges > 0
        ),
        "iterative_execution_present": int(
            summary.get("iterations_run", 0) or 0
        ) > 0,
        "type_layout_ablation_configuration_valid": (
            has_type_layout == expected_type_layout
        ),
        "interface_feedback_ablation_configuration_valid": (
            ("InterfaceSem->TypeSem" not in directions)
            if args.ablation in ("no_interface_feedback", "no_collaboration")
            else ("InterfaceSem->TypeSem" in directions)
        ),
        "api_to_interface_ablation_configuration_valid": (
            "APISem->InterfaceSem" not in directions
            if args.ablation in (
                "no_api_to_interface", "no_interface_incoming",
                "no_collaboration",
            )
            else True
        ),
        "type_to_interface_ablation_configuration_valid": (
            "TypeSem->InterfaceSem" not in directions
            if args.ablation in (
                "no_type_to_interface", "no_interface_incoming",
                "no_collaboration",
            )
            else True
        ),
        "ablation_label_matches_summary": summary_ablation == args.ablation,
        "signed_constraint_feedback_enabled": (
            summary.get("signed_constraint_feedback") is True
        ),
        "api_hard_seed_ablation_effective": (
            int(summary.get("suppressed_confirmed_source_count", 0) or 0) > 0
            if args.ablation == "no_api_hard_seed"
            else int(summary.get("suppressed_confirmed_source_count", 0) or 0)
            == 0
        ),
        "per_round_consistency_recorded": (
            len(consistency_trace)
            == int(summary.get("iterations_run", 0) or 0) + 1
            and all("raw_s_global" in item for item in consistency_trace)
        ),
    }

    warnings = []
    if not args.gold_cache and not args.gold_jsonl:
        warnings.append(
            "No gold supplied: Recall/MRR and confirmation precision are "
            "unavailable (null); label-free confirmation coverage is valid."
        )
    elif not gold:
        warnings.append("Gold sources produced no usable unique query labels.")
    elif matched_gold_queries == 0:
        warnings.append(
            "Gold labels were loaded, but none matched the evaluated query IDs."
        )
    if unresolved_gold_modules:
        warnings.append(
            "%d matched gold queries could not be assigned to one of the four "
            "modules; they are excluded from formal micro/macro metrics and "
            "retained only in the legacy cross-module diagnostic."
            % sum(unresolved_gold_modules.values())
        )
    if gold_audit["semantic_type_conflict_count"]:
        warnings.append(
            "%d duplicate query-target labels had conflicting semantic-type "
            "annotations; the first non-empty type was retained."
            % gold_audit["semantic_type_conflict_count"]
        )

    result = {
        "schema": "task2_223_evaluation_v6_7_validation_guarded",
        "implementation_revision": "v6_7_validation_guarded",
        "ablation": args.ablation,
        "records": len(rows),
        "gold_queries_loaded": len(gold),
        "gold_queries_matched": matched_gold_queries,
        "gold_labels_matched": matched_gold_labels,
        "gold_audit": gold_audit,
        "gold_module_resolution_failures": dict(unresolved_gold_modules),
        "semantic_status_counts": {
            key: dict(value) for key, value in status_counts.items()
        },
        "invalid": dict(invalid),
        "multiple_candidate_groups": multi_candidate,
        # Backward-compatible final view, now using the correct micro unit.
        "ranking": ranking_overall["after_collaboration"],
        "ranking_overall": ranking_overall,
        "ranking_micro": ranking_overall,
        "ranking_macro": ranking_macro,
        "ranking_by_module": ranking_by_module,
        "ranking_cross_module_flat_pool_diagnostic": (
            ranking_cross_module_diagnostic
        ),
        "formal_ranking_unit": "(query_id, semantic_type)",
        "cross_module_scores_compared_in_formal_metrics": False,
        "confirmation_quality": confirmation_quality,
        "api_hard_seed_provenance_report": {
            "real_usg_import_or_symbol": dict(
                api_provenance_report["real_usg_import_or_symbol"]
            ),
            "lineage_sidecar": dict(
                api_provenance_report["lineage_sidecar"]
            ),
            "unknown_or_unrecorded": dict(
                api_provenance_report["unknown_or_unrecorded"]
            ),
            "note": (
                "When --api-provenance-sidecar is supplied, membership in "
                "that file takes precedence; otherwise candidate "
                "api_source_origin / api_source_signal_kind are used. "
                "Lineage is never relabeled as raw import/symbol provenance."
            ),
            "sidecar_path": args.api_provenance_sidecar or None,
            "sidecar_target_ids_loaded": len(lineage_target_ids),
        },
        "consistency": consistency,
        "checks": checks,
        "pass_structural_checks": all(checks.values()),
        "warnings": warnings,
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
