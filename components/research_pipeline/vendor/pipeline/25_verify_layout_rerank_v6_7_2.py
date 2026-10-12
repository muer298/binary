#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Promotion gate for a LayoutSem v6.7.3 Validation full run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from layout_reranker_v6_7_2 import (
    IMPLEMENTATION_REVISION,
    load_layout_config,
)


MODULES = ("APISem", "TypeSem", "LayoutSem", "InterfaceSem")


def load(path: str) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: %s" % path)
    return value


def final_ranking(evaluation: Mapping[str, Any], module: str) -> Mapping[str, Any]:
    return evaluation["ranking_by_module"][module]["after_collaboration"]


def number(value: Any) -> float:
    return float(value or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-evaluation", required=True)
    parser.add_argument("--candidate-evaluation", required=True)
    parser.add_argument("--candidate-collaboration-summary", required=True)
    parser.add_argument("--candidate-four-module-summary", required=True)
    parser.add_argument("--layout-config", required=True)
    parser.add_argument("--min-layout-r1-gain", type=float, default=0.001)
    parser.add_argument("--metric-tolerance", type=float, default=1e-12)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: %s" % out)
    if args.min_layout_r1_gain < 0.0 or args.metric_tolerance < 0.0:
        raise ValueError("gain and tolerance must be non-negative")

    baseline = load(args.baseline_evaluation)
    candidate = load(args.candidate_evaluation)
    collaboration = load(args.candidate_collaboration_summary)
    four_module = load(args.candidate_four_module_summary)
    config, config_path, config_sha256 = load_layout_config(args.layout_config)

    checks: Dict[str, bool] = {}
    checks["validation_selected_config"] = (
        str(config.get("selected_on_split") or "").lower() == "validation"
        and config.get("test_used_for_selection") is False
    )
    checks["layout_revision_loaded"] = (
        four_module.get("layout_reranker_revision") == IMPLEMENTATION_REVISION
    )
    checks["layout_config_digest_matches"] = (
        four_module.get("layout_reranker_config_sha256") == config_sha256
    )
    checks["no_gold_during_module_inference"] = (
        four_module.get("layout_gold_used_during_inference") is False
        and four_module.get("gold_used_during_inference") is False
    )
    checks["records_preserved"] = (
        int(candidate.get("records") or 0) == int(baseline.get("records") or 0)
    )
    checks["gold_match_count_preserved"] = (
        int(candidate.get("gold_queries_matched") or 0)
        == int(baseline.get("gold_queries_matched") or 0)
        and int(candidate.get("gold_labels_matched") or 0)
        == int(baseline.get("gold_labels_matched") or 0)
    )
    checks["candidate_structural_checks_pass"] = (
        candidate.get("pass_structural_checks") is True
    )
    checks["layout_fail_closed_preserved"] = (
        "LayoutSem" in (
            collaboration.get("confirmation_policy_fail_closed_modules") or []
        )
    )
    direction = collaboration.get("configured_direction_weights") or {}
    checks["type_to_interface_remains_disabled"] = (
        number(direction.get("TypeSem->InterfaceSem")) == 0.0
    )
    module_weights = collaboration.get("module_fusion_weights") or {}
    checks["selected_layout_module_weight_consumed"] = (
        abs(
            number(module_weights.get("LayoutSem"))
            - number(config.get("recommended_module_weight"))
        ) <= args.metric_tolerance
    )

    baseline_layout = final_ranking(baseline, "LayoutSem")
    candidate_layout = final_ranking(candidate, "LayoutSem")
    layout_gain = number(candidate_layout.get("recall@1")) - number(
        baseline_layout.get("recall@1")
    )
    checks["layout_r1_improved"] = (
        layout_gain + args.metric_tolerance >= args.min_layout_r1_gain
    )
    checks["layout_r5_nondegrading"] = (
        number(candidate_layout.get("recall@5")) + args.metric_tolerance
        >= number(baseline_layout.get("recall@5"))
    )
    checks["layout_mrr_nondegrading"] = (
        number(candidate_layout.get("mrr")) + args.metric_tolerance
        >= number(baseline_layout.get("mrr"))
    )

    non_layout = {}
    for module in ("APISem", "TypeSem", "InterfaceSem"):
        before = final_ranking(baseline, module)
        after = final_ranking(candidate, module)
        module_check = {
            metric: number(after.get(metric)) + args.metric_tolerance
            >= number(before.get(metric))
            for metric in ("recall@1", "recall@5", "mrr")
        }
        non_layout[module] = module_check
        checks["%s_nondegrading" % module] = all(module_check.values())

    before_overall = baseline["ranking_overall"]["after_collaboration"]
    after_overall = candidate["ranking_overall"]["after_collaboration"]
    checks["overall_r1_nondegrading"] = (
        number(after_overall.get("recall@1")) + args.metric_tolerance
        >= number(before_overall.get("recall@1"))
    )
    checks["overall_mrr_nondegrading"] = (
        number(after_overall.get("mrr")) + args.metric_tolerance
        >= number(before_overall.get("mrr"))
    )

    report = {
        "schema": "task2_layoutsem_promotion_gate_v6_7_3",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "baseline_evaluation": str(Path(args.baseline_evaluation).resolve()),
        "candidate_evaluation": str(Path(args.candidate_evaluation).resolve()),
        "layout_config": config_path,
        "layout_config_sha256": config_sha256,
        "layout_before": dict(baseline_layout),
        "layout_after": dict(candidate_layout),
        "layout_recall1_gain": layout_gain,
        "non_layout_checks": non_layout,
        "checks": checks,
        "passed": all(checks.values()),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("LayoutSem baseline R@1 = %.2f%%" % (100.0 * number(baseline_layout.get("recall@1"))))
    print("LayoutSem candidate R@1 = %.2f%%" % (100.0 * number(candidate_layout.get("recall@1"))))
    print("LayoutSem R@1 gain = %.2f percentage points" % (100.0 * layout_gain))
    print("LayoutSem candidate R@5 = %.2f%%" % (100.0 * number(candidate_layout.get("recall@5"))))
    print("LayoutSem candidate MRR = %.4f" % number(candidate_layout.get("mrr")))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print("FAILED CHECKS:")
        for name in failed:
            print(" -", name)
        raise SystemExit("LAYOUTSEM v6.7.3 PROMOTION GATE: FAILED")
    print("report =", out)
    print("LAYOUTSEM v6.7.3 PROMOTION GATE: PASSED")


if __name__ == "__main__":
    main()
