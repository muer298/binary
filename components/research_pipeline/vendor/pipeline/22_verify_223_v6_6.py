#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performance-aware promotion gate for Task 2.2.3 v6.7 runs."""
import argparse
import json
import sys
from pathlib import Path


MODULES = ("APISem", "TypeSem", "LayoutSem", "InterfaceSem")


def canonical_module(value):
    token = "".join(character for character in str(value).lower() if character.isalpha())
    aliases = {
        "api": "APISem", "apisem": "APISem",
        "type": "TypeSem", "typesem": "TypeSem",
        "layout": "LayoutSem", "layoutsem": "LayoutSem",
        "interface": "InterfaceSem", "interfacesem": "InterfaceSem",
    }
    return aliases.get(token, str(value))


def parse_module_thresholds(text, label):
    result = {}
    for item in (part.strip() for part in str(text or "").split(",")):
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Invalid %s item: %s" % (label, item))
        name, raw_value = item.split("=", 1)
        module = canonical_module(name)
        if module not in MODULES:
            raise ValueError("Unknown module in %s: %s" % (label, name))
        value = float(raw_value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("%s values must be in [0,1]" % label)
        result[module] = value
    return result


def gate_metric(errors, label, before, after, tolerance):
    if before is None or after is None:
        errors.append("%s is unavailable" % label)
        return
    if float(after) + tolerance + 1e-12 < float(before):
        errors.append(
            "%s degraded: before=%.8f final=%.8f tolerance=%.8f"
            % (label, float(before), float(after), tolerance)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--expected-records", type=int, default=0)
    parser.add_argument("--min-gold-matches", type=int, default=0)
    parser.add_argument("--require-gold-metrics", action="store_true")
    parser.add_argument("--require-all-module-gold", action="store_true")
    parser.add_argument(
        "--require-ranking-nondegrading", action="store_true",
        help="Require formal micro R@1 and MRR not to degrade.",
    )
    parser.add_argument("--ranking-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--require-module-ranking-nondegrading", action="store_true",
        help="Require each module's R@1 and MRR not to degrade.",
    )
    parser.add_argument("--module-ranking-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--confirmation-precision-floors",
        default="",
        help="Comma list such as APISem=0.99,TypeSem=0.95.",
    )
    parser.add_argument(
        "--confirmation-coverage-floors",
        default="",
        help="Minimum evaluable Gold coverage; omitted modules default to 0.",
    )
    parser.add_argument("--min-evaluable-confirmations", type=int, default=20)
    args = parser.parse_args()
    if args.ranking_tolerance < 0 or args.module_ranking_tolerance < 0:
        raise ValueError("Ranking tolerances must be non-negative")
    if args.min_evaluable_confirmations < 0:
        raise ValueError("--min-evaluable-confirmations must be non-negative")
    precision_floors = parse_module_thresholds(
        args.confirmation_precision_floors, "confirmation precision floor"
    )
    coverage_floors = parse_module_thresholds(
        args.confirmation_coverage_floors, "confirmation coverage floor"
    )
    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    errors = []
    records = int(evaluation.get("records", 0) or 0)
    if args.expected_records > 0 and records != args.expected_records:
        errors.append("expected %d records, found %d" % (args.expected_records, records))
    if evaluation.get("pass_structural_checks") is not True:
        errors.append("pass_structural_checks is not true")
    for key, value in (evaluation.get("checks") or {}).items():
        if value is not True:
            errors.append("failed check: %s=%r" % (key, value))
    if int(summary.get("locked_decision_violations", 0) or 0) != 0:
        errors.append("2.2.2 locked decisions were modified")
    if summary.get("implementation_revision") != "v6_7_validation_guarded":
        errors.append("collaboration summary is not the v6.7 guarded mainline")
    if evaluation.get("implementation_revision") != "v6_7_validation_guarded":
        errors.append("evaluation is not the v6.7 guarded evaluator")
    if summary.get("specialised_module_scores_consumed") is not True:
        errors.append("specialised module scores were not consumed")
    if summary.get("confirmed_input_decisions_irreversible") is not True:
        errors.append("confirmed 2.2.2 decisions are not declared irreversible")
    if str(evaluation.get("ablation") or "full") != str(
        summary.get("ablation") or "full"
    ):
        errors.append("evaluation/summary ablation labels differ")
    matched_gold = int(evaluation.get("gold_queries_matched", 0) or 0)
    ranking = evaluation.get("ranking") or {}
    if args.min_gold_matches > 0 and matched_gold < args.min_gold_matches:
        errors.append(
            "expected at least %d matched gold queries, found %d"
            % (args.min_gold_matches, matched_gold)
        )
    if args.require_gold_metrics and ranking.get("available") is not True:
        errors.append("gold ranking metrics are unavailable")
    ranking_by_module = evaluation.get("ranking_by_module") or {}
    if args.require_all_module_gold:
        for semantic_type in MODULES:
            final_metrics = (
                ranking_by_module.get(semantic_type, {}).get(
                    "after_collaboration", {}
                )
            )
            if final_metrics.get("available") is not True:
                errors.append(
                    "no evaluable gold ranking records for %s"
                    % semantic_type
                )
    ranking_overall = evaluation.get("ranking_overall") or {}
    if args.require_ranking_nondegrading:
        before = ranking_overall.get("before_223") or {}
        final = ranking_overall.get("after_collaboration") or {}
        gate_metric(
            errors, "formal micro recall@1", before.get("recall@1"),
            final.get("recall@1"), args.ranking_tolerance,
        )
        gate_metric(
            errors, "formal micro MRR", before.get("mrr"),
            final.get("mrr"), args.ranking_tolerance,
        )
    if args.require_module_ranking_nondegrading:
        for semantic_type in MODULES:
            stages = ranking_by_module.get(semantic_type) or {}
            before = stages.get("before_223") or {}
            final = stages.get("after_collaboration") or {}
            gate_metric(
                errors, "%s recall@1" % semantic_type,
                before.get("recall@1"), final.get("recall@1"),
                args.module_ranking_tolerance,
            )
            gate_metric(
                errors, "%s MRR" % semantic_type,
                before.get("mrr"), final.get("mrr"),
                args.module_ranking_tolerance,
            )

    confirmation_by_module = (
        (evaluation.get("confirmation_quality") or {}).get("by_module") or {}
    )
    confirmation_gate = {}
    for semantic_type, floor in precision_floors.items():
        combined = (
            confirmation_by_module.get(semantic_type, {}).get(
                "confirmed_combined", {}
            )
        )
        confirmed = int(combined.get("confirmed", 0) or 0)
        evaluable = int(combined.get("evaluable_confirmed", 0) or 0)
        precision = combined.get("precision")
        coverage = combined.get("coverage_gold_objects")
        coverage_value = float(coverage) if coverage is not None else 0.0
        coverage_floor = coverage_floors.get(semantic_type, 0.0)
        confirmation_gate[semantic_type] = {
            "confirmed": confirmed,
            "evaluable": evaluable,
            "precision": precision,
            "precision_floor": floor,
            "gold_coverage": coverage,
            "coverage_floor": coverage_floor,
            "safe_abstention": confirmed == 0 and coverage_floor <= 0.0,
        }
        if coverage_value + 1e-12 < coverage_floor:
            errors.append(
                "%s confirmation Gold coverage %.8f is below %.8f"
                % (semantic_type, coverage_value, coverage_floor)
            )
        if confirmed == 0 and coverage_floor <= 0.0:
            # Fail-closed abstention is safe when no minimum coverage was set.
            continue
        if evaluable < args.min_evaluable_confirmations:
            errors.append(
                "%s has only %d evaluable confirmations; need at least %d"
                % (
                    semantic_type, evaluable,
                    args.min_evaluable_confirmations,
                )
            )
            continue
        if precision is None or float(precision) + 1e-12 < floor:
            errors.append(
                "%s confirmation precision %r is below %.8f"
                % (semantic_type, precision, floor)
            )
    print("records =", records)
    print("ablation =", summary.get("ablation"))
    print("iterations_run =", summary.get("iterations_run"))
    print("stop_reason =", summary.get("stop_reason"))
    print("collaboration_edges =", summary.get("collaboration_edges"))
    print("gold_queries_matched =", matched_gold)
    print("ranking_final_overall =", ranking)
    for semantic_type in MODULES:
        print(
            "ranking_final_%s =" % semantic_type,
            ranking_by_module.get(semantic_type, {}).get(
                "after_collaboration"
            ),
        )
    print(
        "confirmation_quality_overall =",
        (evaluation.get("confirmation_quality") or {}).get("overall"),
    )
    print("confirmation_per_module_gate =", confirmation_gate)
    print("ranking_macro =", evaluation.get("ranking_macro"))
    print(
        "legacy_cross_module_flat_pool_diagnostic =",
        evaluation.get("ranking_cross_module_flat_pool_diagnostic"),
    )
    print(
        "raw_s_global =",
        summary.get("initial_raw_s_global"),
        "->",
        summary.get("final_raw_s_global"),
        "delta =",
        summary.get("raw_s_global_delta"),
    )
    print("checks =", evaluation.get("checks"))
    if errors:
        print("VERIFY 2.2.3 RESULT: FAILED")
        for error in errors:
            print(" -", error)
        return 1
    print("VERIFY 2.2.3 RESULT: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
