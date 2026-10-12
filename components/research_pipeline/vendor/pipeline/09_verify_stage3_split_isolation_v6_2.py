#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify Stage3/Unified-Stage5 Train-only graph and single Test use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


APPROVED_OBJECTIVES = {
    "multi_positive_target_attention_v2_train_graph_firewall_v3": "train_samples_only_v3",
    "stage5_unified_three_loss_train_graph_v6_5": "train_samples_only_v6_5",
}


def get_path(value, *keys, default=None):
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--require-frozen-split", action="store_true")
    parser.add_argument("--require-complete-edge-scan", action="store_true")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    split = get_path(payload, "dataset", "split_info", default={}) or {}
    graph = get_path(payload, "dataset", "graph_split_audit", default={}) or {}
    evaluation = get_path(payload, "evaluation_audit", default={}) or {}
    test_overall = get_path(payload, "metrics", "test", "overall", default={}) or {}
    objective_revision = get_path(payload, "model", "objective_revision")
    expected_graph_policy = APPROVED_OBJECTIVES.get(objective_revision)

    checks = {
        "objective_revision_valid": objective_revision in APPROVED_OBJECTIVES,
        "sample_split_used": split.get("split_mode") == "sample",
        "sample_splits_disjoint": split.get("sample_overlap_count") == 0,
        "target_groups_disjoint": split.get("target_group_overlap_count") == 0,
        "train_only_graph_policy": graph.get("policy") == expected_graph_policy,
        "positive_edge_leakage_zero": graph.get("positive_edge_leakage_count") == 0,
        "negative_node_leakage_zero": graph.get("negative_node_leakage_count") == 0,
        "graph_firewall_passed": graph.get("passed") is True,
        "train_graph_nonempty": int(graph.get("retained_train_edges") or 0) > 0,
        "test_evaluated_once": evaluation.get("evaluation_count") == 1,
        "test_not_evaluated_during_training": (
            evaluation.get("test_evaluations_during_training") == 0
        ),
        "validation_selected_checkpoint": (
            evaluation.get("checkpoint_selection_split") == "validation"
        ),
        "model_frozen_before_test": (
            evaluation.get("model_parameters_frozen_before_test") is True
        ),
        "test_not_used_for_selection": (
            evaluation.get("test_labels_used_for_selection") is False
        ),
        "test_records_nonempty": int(test_overall.get("n") or 0) > 0,
    }
    if args.require_frozen_split:
        checks["frozen_split_manifest_used"] = (
            split.get("split_source") == "frozen_manifest"
            and isinstance(split.get("manifest"), dict)
            and int(split["manifest"].get("unique_samples") or 0) > 0
        )
    if args.require_complete_edge_scan:
        checks["edge_scan_complete"] = not bool(
            graph.get("scan_truncated_at_retained_limit")
        )

    report = {
        "schema": "task2_stage3_stage5_train_split_verification_v6_5",
        "metrics": str(metrics_path.resolve()),
        "checks": checks,
        "passed": all(checks.values()),
        "summary": {
            "split_source": split.get("split_source"),
            "objective_revision": objective_revision,
            "expected_graph_policy": expected_graph_policy,
            "train_pairs": get_path(payload, "dataset", "train_pairs"),
            "validation_pairs": get_path(payload, "dataset", "val_pairs"),
            "test_pairs": get_path(payload, "dataset", "test_pairs"),
            "retained_train_edges": graph.get("retained_train_edges"),
            "excluded_edge_counts": graph.get("excluded_edge_counts"),
            "train_negative_pool_nodes": graph.get("train_node_pool_count"),
            "test_evaluation_count": evaluation.get("evaluation_count"),
            "test_mrr": test_overall.get("mrr"),
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit("VERIFY FAILED: %s" % ", ".join(failed))
    print("VERIFY STAGE3 SPLIT ISOLATION: PASSED")


if __name__ == "__main__":
    main()
