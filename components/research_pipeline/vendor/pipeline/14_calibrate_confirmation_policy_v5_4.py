#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrate v5.3 soft-confirmation gates on an independent Validation gold.

This script never changes deterministic APISem rules.  It grid-searches only
``min_evidence``, ``gap_threshold`` and ``min_evidence_coverage`` and chooses
the largest correctly constrained coverage subject to a per-module precision
floor.  If no grid point is safe, soft confirmation for that module is
disabled (fail-closed). Test-labelled gold is rejected by default to prevent
leakage.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from module_common_v5_3 import (
    CONFIRM_POLICY,
    SEMANTIC_TYPES,
    SOFT_API_SOURCES,
    clamp01,
    normalize_api_source,
    normalize_semantic_type,
    read_jsonl,
    write_json,
)


def ensure_output_available(path: str, overwrite: bool) -> None:
    if Path(path).exists() and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output: %s" % path
        )


def parse_grid(text: str) -> List[float]:
    values = sorted({round(float(item), 8) for item in text.split(",") if item})
    if not values or any(value < 0 or value > 1 for value in values):
        raise ValueError("Grid values must be a non-empty comma list in [0,1]")
    return values


def parse_precision_floors(text: str) -> Dict[str, float]:
    result = {
        "APISem": 0.99,
        "TypeSem": 0.95,
        "LayoutSem": 0.95,
        "InterfaceSem": 0.95,
    }
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        key, value = item.split("=", 1)
        semantic_type = normalize_semantic_type(key)
        if semantic_type not in SEMANTIC_TYPES:
            raise ValueError("Unknown module in precision floor: %s" % key)
        result[semantic_type] = float(value)
    if any(value < 0 or value > 1 for value in result.values()):
        raise ValueError("Precision floors must be in [0,1]")
    return result


def load_gold(
    paths: Sequence[str], allow_non_validation: bool
) -> Tuple[Dict[Tuple[str, str], Set[str]], Dict[str, Any]]:
    gold: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    splits: Counter = Counter()
    ignored = Counter()
    for path in paths:
        for row in read_jsonl(path):
            split = str(row.get("split") or "").strip().lower()
            splits[split or "missing"] += 1
            if not allow_non_validation and split not in (
                "validation", "valid", "val"
            ):
                ignored[split or "missing"] += 1
                continue
            query_id = str(
                row.get("query_id") or row.get("source_node") or ""
            )
            target_id = str(
                row.get("target_id") or row.get("target_node") or ""
            )
            semantic_type = normalize_semantic_type(
                row.get("target_semantic_type")
                or row.get("semantic_type")
                or ""
            )
            if query_id and target_id and semantic_type in SEMANTIC_TYPES:
                gold[(query_id, semantic_type)].add(target_id)
    if not allow_non_validation and ignored and not gold:
        raise RuntimeError(
            "No Validation gold remained. Pass gold_split=validation; never "
            "tune on Test. Observed splits: %s" % dict(splits)
        )
    return dict(gold), {
        "input_splits": dict(splits),
        "ignored_non_validation": dict(ignored),
        "usable_query_module_groups": len(gold),
    }


def candidate_score(candidate: Mapping[str, Any]) -> float:
    for key in ("validated_score", "confidence", "representation_score"):
        if key in candidate:
            return clamp01(candidate.get(key, 0.0))
    return 0.0


def examples_from_rows(
    path: str,
    gold: Mapping[Tuple[str, str], Set[str]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    by_module: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    matched_groups = 0
    gold_target_in_candidates = 0
    for row in read_jsonl(path):
        query_id = str(row.get("query_id") or "")
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for candidate in row.get("candidates", []) or []:
            semantic_type = normalize_semantic_type(
                candidate.get("target_semantic_type") or ""
            )
            if semantic_type in SEMANTIC_TYPES:
                grouped[semantic_type].append(candidate)
        for semantic_type, candidates in grouped.items():
            targets = gold.get((query_id, semantic_type))
            if not targets:
                continue
            matched_groups += 1
            active = [
                candidate for candidate in candidates
                if str(candidate.get("candidate_status") or "") != "rejected"
            ]
            if not active:
                continue
            active.sort(
                key=lambda value: (
                    -candidate_score(value), str(value.get("target_id") or "")
                )
            )
            best = active[0]
            second_score = candidate_score(active[1]) if len(active) > 1 else 0.0
            best_target = str(best.get("target_id") or "")
            target_present = any(
                str(candidate.get("target_id") or "") in targets
                for candidate in active
            )
            gold_target_in_candidates += int(target_present)
            hard = bool(best.get("hard_api_eligible"))
            api_source = normalize_api_source(best.get("api_source"))
            by_module[semantic_type].append({
                "query_id": query_id,
                "best_target_id": best_target,
                "correct": best_target in targets,
                "target_present": target_present,
                "score": candidate_score(best),
                "gap": max(0.0, candidate_score(best) - second_score),
                "coverage": clamp01(best.get("evidence_coverage", 0.0)),
                "hard": hard,
                "api_source": api_source,
            })
    return dict(by_module), {
        "matched_query_module_groups": matched_groups,
        "gold_target_in_candidate_groups": gold_target_in_candidates,
        "candidate_coverage": gold_target_in_candidates / max(matched_groups, 1),
    }


def evaluate_gate(
    examples: Iterable[Mapping[str, Any]],
    semantic_type: str,
    score_threshold: float,
    gap_threshold: float,
    coverage_threshold: float,
) -> Dict[str, Any]:
    values = list(examples)
    confirmed = correct = 0
    for row in values:
        if row.get("hard"):
            continue
        if (
            semantic_type == "APISem"
            and row.get("api_source") not in SOFT_API_SOURCES
        ):
            continue
        passes = (
            float(row["score"]) + 1e-12 >= score_threshold
            and float(row["gap"]) + 1e-12 >= gap_threshold
            and float(row["coverage"]) + 1e-12 >= coverage_threshold
        )
        if passes:
            confirmed += 1
            correct += int(bool(row["correct"]))
    n = sum(1 for row in values if not row.get("hard"))
    return {
        "n": n,
        "confirmed": confirmed,
        "correct": correct,
        "precision": correct / confirmed if confirmed else None,
        "coverage": confirmed / n if n else 0.0,
        "correct_coverage": correct / n if n else 0.0,
    }


def choose_policy(
    semantic_type: str,
    examples: Sequence[Mapping[str, Any]],
    score_grid: Sequence[float],
    gap_grid: Sequence[float],
    coverage_grid: Sequence[float],
    min_precision: float,
    min_confirmed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    default = CONFIRM_POLICY[semantic_type]
    baseline = evaluate_gate(
        examples,
        semantic_type,
        float(default["min_evidence"]),
        float(default["gap_threshold"]),
        float(default["min_evidence_coverage"]),
    )
    choices = []
    for score in score_grid:
        for gap in gap_grid:
            for coverage in coverage_grid:
                metrics = evaluate_gate(
                    examples, semantic_type, score, gap, coverage
                )
                precision = metrics["precision"]
                if (
                    metrics["confirmed"] >= min_confirmed
                    and precision is not None
                    and precision >= min_precision
                ):
                    # Maximize correct coverage, then confirmations and precision;
                    # prefer stricter thresholds on a complete tie.
                    choices.append((
                        metrics["correct_coverage"],
                        metrics["confirmed"],
                        precision,
                        score + gap + coverage,
                        score,
                        gap,
                        coverage,
                        metrics,
                    ))
    if not choices:
        policy = {
            "soft_confirmation_enabled": False,
            "min_evidence": float(default["min_evidence"]),
            "gap_threshold": float(default["gap_threshold"]),
            "min_evidence_coverage": float(default["min_evidence_coverage"]),
        }
        return policy, {
            "selection": "fail_closed_no_grid_point_met_precision_floor",
            "precision_floor": min_precision,
            "baseline": baseline,
            "selected": {
                "n": baseline["n"],
                "confirmed": 0,
                "correct": 0,
                "precision": None,
                "coverage": 0.0,
                "correct_coverage": 0.0,
            },
            "reason": (
                "No Validation grid point met both the precision floor and "
                "minimum confirmed count; unsafe fallback confirmation was "
                "disabled."
            ),
        }
    choices.sort(reverse=True)
    _, _, _, _, score, gap, coverage, selected = choices[0]
    return {
        "soft_confirmation_enabled": True,
        "min_evidence": score,
        "gap_threshold": gap,
        "min_evidence_coverage": coverage,
    }, {
        "selection": "validation_precision_constrained_grid_search",
        "precision_floor": min_precision,
        "baseline": baseline,
        "selected": selected,
        "eligible_grid_points": len(choices),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated", required=True)
    parser.add_argument("--gold-jsonl", action="append", required=True)
    parser.add_argument("--out-policy", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument(
        "--score-grid",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
    )
    parser.add_argument(
        "--gap-grid", default="0.05,0.10,0.15,0.20,0.25,0.30"
    )
    parser.add_argument(
        "--coverage-grid", default="0.20,0.35,0.45,0.55,0.80,1.00"
    )
    parser.add_argument(
        "--precision-floors",
        default="APISem=0.99,TypeSem=0.95,LayoutSem=0.95,InterfaceSem=0.95",
    )
    parser.add_argument("--min-confirmed", type=int, default=20)
    parser.add_argument("--allow-non-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Fail before an expensive scan so policy/report are committed together.
    ensure_output_available(args.out_policy, args.overwrite)
    ensure_output_available(args.out_report, args.overwrite)

    gold, gold_audit = load_gold(
        args.gold_jsonl, args.allow_non_validation
    )
    examples, example_audit = examples_from_rows(args.validated, gold)
    floors = parse_precision_floors(args.precision_floors)
    score_grid = parse_grid(args.score_grid)
    gap_grid = parse_grid(args.gap_grid)
    coverage_grid = parse_grid(args.coverage_grid)

    policies: Dict[str, Dict[str, Any]] = {}
    reports: Dict[str, Dict[str, Any]] = {}
    for semantic_type in SEMANTIC_TYPES:
        values = examples.get(semantic_type, [])
        policy, report = choose_policy(
            semantic_type,
            values,
            score_grid,
            gap_grid,
            coverage_grid,
            floors[semantic_type],
            args.min_confirmed,
        )
        policies[semantic_type] = policy
        reports[semantic_type] = {
            "validation_examples": len(values),
            "correct_top1": sum(int(value["correct"]) for value in values),
            "gold_target_in_candidates": sum(
                int(value["target_present"]) for value in values
            ),
            **report,
        }

    policy_payload = {
        "schema": "task2_222_confirm_policy_v6_7_fail_closed",
        "implementation_revision": "v6_7_validation_guarded",
        "calibration_split": "validation",
        "hard_api_policy_immutable": True,
        "unsafe_default_fallback_forbidden": True,
        "policies": policies,
    }
    report_payload = {
        "schema": "task2_222_confirm_policy_calibration_report_v6_7",
        "implementation_revision": "v6_7_validation_guarded",
        "validated": args.validated,
        "gold": list(args.gold_jsonl),
        "gold_audit": gold_audit,
        "example_audit": example_audit,
        "precision_floors": floors,
        "grids": {
            "score": score_grid,
            "gap": gap_grid,
            "coverage": coverage_grid,
        },
        "by_module": reports,
        "policy": policy_payload,
    }
    write_json(args.out_report, report_payload, overwrite=args.overwrite)
    write_json(args.out_policy, policy_payload, overwrite=args.overwrite)
    print(json.dumps(report_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
