#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose low confirmation coverage without treating coverage as accuracy."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Set, Tuple

from module_common_v5_3 import (
    CONFIRMED_STATUSES, CONFIRM_POLICY, SEMANTIC_TYPES, clamp01,
    load_confirm_policy, normalize_api_source, normalize_semantic_type,
    read_jsonl,
)


def load_gold(paths: Sequence[str]) -> Dict[Tuple[str, str], Set[str]]:
    result = defaultdict(set)
    for path in paths:
        for row in read_jsonl(path):
            query_id = str(row.get("query_id") or "")
            semantic_type = normalize_semantic_type(
                row.get("target_semantic_type") or row.get("semantic_type")
            )
            target_id = str(row.get("target_id") or "")
            if query_id and target_id and semantic_type in SEMANTIC_TYPES:
                result[(query_id, semantic_type)].add(target_id)
    return dict(result)


def score(candidate: Mapping[str, Any]) -> float:
    for key in ("confidence", "validated_score", "representation_score"):
        if key in candidate:
            return clamp01(candidate.get(key))
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--gold-jsonl", action="append", default=[])
    parser.add_argument("--confirm-policy")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: %s" % out)
    policy = load_confirm_policy(args.confirm_policy)
    gold = load_gold(args.gold_jsonl)

    module_groups = Counter()
    status_counts = defaultdict(Counter)
    blockers = defaultdict(Counter)
    gold_stats = defaultdict(Counter)
    candidate_counts = defaultdict(Counter)
    source_counts = Counter()
    rows = 0
    for row in read_jsonl(args.input):
        rows += 1
        query_id = str(row.get("query_id") or "")
        grouped = defaultdict(list)
        for candidate in row.get("candidates", []) or []:
            semantic_type = normalize_semantic_type(candidate.get("target_semantic_type"))
            if semantic_type in SEMANTIC_TYPES:
                grouped[semantic_type].append(candidate)
        decisions = row.get("semantic_decisions") if isinstance(row.get("semantic_decisions"), Mapping) else {}
        for semantic_type in SEMANTIC_TYPES:
            candidates = grouped.get(semantic_type, [])
            decision = decisions.get(semantic_type) if isinstance(decisions.get(semantic_type), Mapping) else None
            if not candidates and decision is None:
                continue
            module_groups[semantic_type] += 1
            candidate_counts[semantic_type][str(len(candidates))] += 1
            status = str((decision or {}).get("status") or "hypothesis")
            status_counts[semantic_type][status] += 1
            active = [value for value in candidates if value.get("candidate_status") != "rejected"]
            active.sort(key=lambda value: (-score(value), str(value.get("target_id") or "")))
            if not active:
                blockers[semantic_type]["no_active_candidate"] += 1
                continue
            best = active[0]
            best_score = score(best)
            second_score = score(active[1]) if len(active) > 1 else 0.0
            gap = max(0.0, best_score - second_score)
            coverage = clamp01(best.get("evidence_coverage", 0.0))
            config = policy[semantic_type]
            if status not in CONFIRMED_STATUSES:
                if not bool(config.get("soft_confirmation_enabled", True)):
                    blockers[semantic_type][
                        "soft_confirmation_disabled_fail_closed"
                    ] += 1
                if best_score < float(config["min_evidence"]):
                    blockers[semantic_type]["score_below_gate"] += 1
                if gap < float(config["gap_threshold"]):
                    blockers[semantic_type]["gap_below_gate"] += 1
                if coverage < float(config["min_evidence_coverage"]):
                    blockers[semantic_type]["coverage_below_gate"] += 1
                observed = best.get("component_observed") or {}
                if not observed.get("direct_support"):
                    blockers[semantic_type]["direct_support_missing"] += 1
                if not observed.get("constraint_consistency"):
                    blockers[semantic_type]["constraint_consistency_missing"] += 1
                if not observed.get("cross_object_collaboration"):
                    blockers[semantic_type]["collaboration_missing"] += 1
                if semantic_type == "APISem":
                    source = normalize_api_source(best.get("api_source"))
                    source_counts[source] += 1
                    if source not in set(config.get("hard_sources", ())) | set(config.get("soft_sources", ())):
                        blockers[semantic_type]["untrusted_or_unknown_api_source"] += 1

            targets = gold.get((query_id, semantic_type))
            if targets:
                gold_stats[semantic_type]["matched_groups"] += 1
                present = any(str(value.get("target_id") or "") in targets for value in active)
                gold_stats[semantic_type]["gold_in_candidates"] += int(present)
                if status in CONFIRMED_STATUSES:
                    gold_stats[semantic_type]["confirmed"] += 1
                    selected = str((decision or {}).get("best_target_id") or best.get("target_id") or "")
                    gold_stats[semantic_type]["confirmed_correct"] += int(selected in targets)

    module_report = {}
    total_groups = total_confirmed = 0
    for semantic_type in SEMANTIC_TYPES:
        groups = module_groups[semantic_type]
        confirmed = sum(status_counts[semantic_type][value] for value in CONFIRMED_STATUSES)
        total_groups += groups
        total_confirmed += confirmed
        gold_value = gold_stats[semantic_type]
        matched = gold_value["matched_groups"]
        gold_confirmed = gold_value["confirmed"]
        module_report[semantic_type] = {
            "semantic_object_groups": groups,
            "status_counts": dict(status_counts[semantic_type]),
            "confirmation_coverage": confirmed / groups if groups else 0.0,
            "candidate_count_histogram": dict(candidate_counts[semantic_type]),
            "hypothesis_blockers_nonexclusive": dict(blockers[semantic_type]),
            "gold": {
                "available": matched > 0,
                "matched_groups": matched,
                "candidate_coverage": gold_value["gold_in_candidates"] / matched if matched else None,
                "confirmed": gold_confirmed,
                "confirmed_precision": gold_value["confirmed_correct"] / gold_confirmed if gold_confirmed else None,
                "confirmed_correct_coverage": gold_value["confirmed_correct"] / matched if matched else None,
            },
        }

    report = {
        "schema": "task2_222_confirmation_coverage_audit_v6_6",
        "input_rows": rows,
        "semantic_object_groups": total_groups,
        "confirmed_groups": total_confirmed,
        "overall_confirmation_coverage": total_confirmed / total_groups if total_groups else 0.0,
        "denominator_note": "semantic query-module objects, not raw candidate rows",
        "by_module": module_report,
        "api_source_counts_for_unconfirmed_best": dict(source_counts),
        "active_policy": policy,
        "gold_paths": list(args.gold_jsonl),
        "interpretation": (
            "Coverage alone is not an accuracy metric. Promote a calibrated "
            "policy only when Validation precision floors and candidate "
            "coverage are both acceptable."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
