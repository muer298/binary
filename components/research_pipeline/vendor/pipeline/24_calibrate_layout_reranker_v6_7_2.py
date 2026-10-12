#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select LayoutSem reranker blend on Validation only.

This program consumes the v6.7.3 evidence-bridge LayoutInfer probe output.  It does not train
on Validation labels; it selects two small fusion hyperparameters from an
explicit grid and records the entire audit.  Test input is rejected.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from layout_reranker_v6_7_2 import CONFIG_SCHEMA, DEFAULT_CONFIG


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Non-object JSONL row at %s:%d" % (path, line_number))
            yield value


def parse_grid(text: str) -> List[float]:
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("grid must contain values in [0,1]")
    return values


def normalise_semantic_type(value: Any) -> str:
    text = "".join(character for character in str(value or "").lower() if character.isalpha())
    return {
        "layout": "LayoutSem",
        "layoutsem": "LayoutSem",
        "layoutinfer": "LayoutSem",
    }.get(text, str(value or ""))


def load_gold(paths: Sequence[str], expected_split: str) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    if expected_split != "validation":
        raise ValueError("LayoutSem calibration accepts Validation only")
    labels: Dict[str, Set[str]] = defaultdict(set)
    audit = Counter()
    for path in paths:
        for row in read_jsonl(path):
            split = str(row.get("split") or expected_split).lower()
            if split != expected_split:
                raise ValueError("%s contains split=%r; expected validation" % (path, split))
            if normalise_semantic_type(
                row.get("target_semantic_type") or row.get("semantic_type")
            ) != "LayoutSem":
                continue
            query_id = str(row.get("query_id") or "")
            target_id = str(row.get("target_id") or "")
            if query_id and target_id:
                labels[query_id].add(target_id)
                audit["labels"] += 1
    audit["queries"] = len(labels)
    return dict(labels), dict(audit)


def is_direct(candidate: Mapping[str, Any]) -> bool:
    return (
        candidate.get("candidate_source") == "usg_direct_evidence"
        or int(candidate.get("direct_edge_count", 0) or 0) > 0
    )


def load_groups(
    module_result: str,
    gold: Mapping[str, Set[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    audit = Counter()
    for row in read_jsonl(module_result):
        if normalise_semantic_type(row.get("semantic_type")) != "LayoutSem":
            continue
        query_id = str(row.get("query_id") or "")
        targets = set(gold.get(query_id) or ())
        if not targets:
            continue
        candidates = list(row.get("candidate_scores") or [])
        ids = [str(candidate.get("target_id") or "") for candidate in candidates]
        if not targets.intersection(ids):
            audit["gold_missing_from_candidates"] += 1
            continue
        values = []
        for candidate in candidates:
            feature_trace = candidate.get("module_feature_trace_223_v6_6") or {}
            pair_trace = feature_trace.get("layout_pairwise_trace") or {}
            legacy = float(pair_trace.get("legacy_score", feature_trace.get("rule_score", 0.0)) or 0.0)
            pairwise = float(pair_trace.get("pairwise_score", legacy) or 0.0)
            values.append({
                "target_id": str(candidate.get("target_id") or ""),
                "baseline": float(candidate.get("confidence_before_223", candidate.get("confidence", 0.0)) or 0.0),
                "legacy": legacy,
                "pairwise": pairwise,
                "usable": bool(pair_trace.get("usable")),
                "direct": is_direct(candidate),
            })
        gold_direct = any(item["target_id"] in targets and item["direct"] for item in values)
        groups.append({
            "query_id": query_id,
            "targets": targets,
            "candidates": values,
            "usable": any(item["usable"] for item in values),
            "gold_direct": gold_direct,
        })
        audit["groups"] += 1
        audit["usable_groups"] += int(groups[-1]["usable"])
        audit["gold_direct_groups"] += int(gold_direct)
    return groups, dict(audit)


def evaluate(groups: Sequence[Mapping[str, Any]], pairwise_blend: float, module_weight: float) -> Dict[str, Any]:
    buckets = {
        "all": Counter(),
        "usable": Counter(),
        "gold_direct": Counter(),
        "gold_not_direct": Counter(),
    }
    ranks = Counter()
    for group in groups:
        scored = []
        for candidate in group["candidates"]:
            module_score = (
                (1.0 - pairwise_blend) * candidate["legacy"]
                + pairwise_blend * candidate["pairwise"]
                if candidate["usable"] else candidate["legacy"]
            )
            fused = (1.0 - module_weight) * candidate["baseline"] + module_weight * module_score
            scored.append((fused, candidate["target_id"]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ordered_ids = [target_id for _, target_id in scored]
        rank = min(
            ordered_ids.index(target) + 1
            for target in group["targets"]
            if target in ordered_ids
        )
        ranks[str(rank if rank <= 6 else "7+")] += 1
        bucket_names = ["all", "gold_direct" if group["gold_direct"] else "gold_not_direct"]
        if group["usable"]:
            bucket_names.append("usable")
        for name in bucket_names:
            bucket = buckets[name]
            bucket["n"] += 1
            bucket["hit1"] += int(rank == 1)
            bucket["hit5"] += int(rank <= 5)
            bucket["reciprocal_rank_scaled"] += int(round(1_000_000_000 / rank))

    result: Dict[str, Any] = {}
    for name, bucket in buckets.items():
        n = bucket["n"]
        result[name] = {
            "n": n,
            "recall@1": bucket["hit1"] / n if n else None,
            "recall@5": bucket["hit5"] / n if n else None,
            "mrr": bucket["reciprocal_rank_scaled"] / 1_000_000_000 / n if n else None,
        }
    result["rank_histogram"] = dict(ranks)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-module-result", required=True)
    parser.add_argument("--gold-validation", action="append", required=True)
    parser.add_argument("--expected-split", default="validation", choices=("validation",))
    parser.add_argument("--pairwise-blend-grid", default="0,0.25,0.5,0.65,0.75,0.85,1")
    parser.add_argument("--module-weight-grid", default="0.25,0.35,0.45,0.55,0.65")
    parser.add_argument("--min-usable-groups", type=int, default=100)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: %s" % out)
    if args.min_usable_groups <= 0:
        raise ValueError("--min-usable-groups must be positive")

    gold, gold_audit = load_gold(args.gold_validation, args.expected_split)
    groups, group_audit = load_groups(args.layout_module_result, gold)
    if not groups:
        raise SystemExit("ERROR: no evaluable LayoutSem Validation groups")
    if int(group_audit.get("usable_groups") or 0) < args.min_usable_groups:
        raise SystemExit(
            "ERROR: only %d usable LayoutSem groups; require at least %d. "
            "Inspect query/target offset export before calibration."
            % (int(group_audit.get("usable_groups") or 0), args.min_usable_groups)
        )

    baseline = evaluate(groups, pairwise_blend=0.0, module_weight=0.0)
    grid = []
    for pairwise_blend in parse_grid(args.pairwise_blend_grid):
        for module_weight in parse_grid(args.module_weight_grid):
            metrics = evaluate(groups, pairwise_blend, module_weight)
            all_metrics = metrics["all"]
            usable_metrics = metrics["usable"]
            if (all_metrics["recall@5"] or 0.0) + 1e-12 < (baseline["all"]["recall@5"] or 0.0):
                continue
            grid.append({
                "pairwise_blend": pairwise_blend,
                "module_weight": module_weight,
                "metrics": metrics,
            })
    if not grid:
        raise SystemExit("ERROR: every LayoutSem configuration degraded R@5")

    def selection_key(row: Mapping[str, Any]) -> Tuple[float, float, float, float, float]:
        overall = row["metrics"]["all"]
        usable = row["metrics"]["usable"]
        return (
            float(overall["recall@1"] or 0.0),
            float(overall["mrr"] or 0.0),
            float(usable["recall@1"] or 0.0),
            -float(row["module_weight"]),
            -float(row["pairwise_blend"]),
        )

    selected = max(grid, key=selection_key)
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config.update({
        "schema": CONFIG_SCHEMA,
        "implementation_revision": "v6_7_3_layout_evidence_bridge_reranker",
        "selected_on_split": "validation",
        "test_used_for_selection": False,
        "pairwise_blend": selected["pairwise_blend"],
        "recommended_module_weight": selected["module_weight"],
        "selection_method": "finite_grid_max_recall1_then_mrr_conservative_tiebreak",
        "gold_used_during_inference": False,
        "calibration_inputs": {
            "layout_module_result": str(Path(args.layout_module_result).resolve()),
            "gold_validation": [str(Path(path).resolve()) for path in args.gold_validation],
        },
        "gold_audit": gold_audit,
        "group_audit": group_audit,
        "baseline_metrics": baseline,
        "selected_metrics_before_collaboration": selected["metrics"],
        "grid": sorted(grid, key=selection_key, reverse=True),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("LayoutSem Validation groups =", len(groups))
    print("LayoutSem usable groups =", group_audit.get("usable_groups"))
    print("baseline R@1 = %.4f" % (baseline["all"]["recall@1"] or 0.0))
    print("selected pairwise_blend =", selected["pairwise_blend"])
    print("selected module_weight =", selected["module_weight"])
    print("selected R@1 = %.4f" % (selected["metrics"]["all"]["recall@1"] or 0.0))
    print("selected MRR = %.4f" % (selected["metrics"]["all"]["mrr"] or 0.0))
    print("config =", out)
    print("LAYOUTSEM v6.7.3 VALIDATION CALIBRATION: PASSED")


if __name__ == "__main__":
    main()
