#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 SymGen 四模块正式指标与候选难度，不参与模型选择或调参。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


MODULES = ("APISem", "TypeSem", "LayoutSem", "InterfaceSem")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def normalize_module(value: Any) -> str:
    text = str(value or "").lower()
    for token, module in (
        ("interface", "InterfaceSem"),
        ("layout", "LayoutSem"),
        ("api", "APISem"),
        ("type", "TypeSem"),
    ):
        if token in text:
            return module
    return ""


def score(candidate: Mapping[str, Any]) -> float:
    for key in (
        "score_223_v5_3",
        "score_after_module_fusion_before_collaboration_223_v5_3",
        "probability",
        "validated_score",
        "confidence",
    ):
        try:
            value = float(candidate.get(key))
            if math.isfinite(value):
                return value
        except (TypeError, ValueError):
            continue
    return 0.0


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = [float(row["candidate_count"]) for row in records]
    ranks = [float(row["gold_rank"]) for row in records if row["gold_rank"] is not None]
    gaps = [float(row["top1_top2_gap"]) for row in records if row["top1_top2_gap"] is not None]
    total = len(records)
    covered = len(ranks)
    return {
        "gold_query_module_objects": total,
        "gold_candidate_coverage": covered / total if total else None,
        "recall@1": (
            sum(row["gold_rank"] == 1 for row in records) / total
            if total else None
        ),
        "recall@5": (
            sum(
                row["gold_rank"] is not None and row["gold_rank"] <= 5
                for row in records
            ) / total
            if total else None
        ),
        "mrr": (
            sum(
                1.0 / float(row["gold_rank"])
                if row["gold_rank"] is not None else 0.0
                for row in records
            ) / total
            if total else None
        ),
        "candidate_count_mean": statistics.fmean(counts) if counts else None,
        "candidate_count_median": statistics.median(counts) if counts else None,
        "candidate_count_p90": percentile(counts, 0.90),
        "candidate_count_max": max(counts) if counts else None,
        "gold_rank_mean_when_covered": statistics.fmean(ranks) if ranks else None,
        "gold_rank_median_when_covered": statistics.median(ranks) if ranks else None,
        "gold_rank_p90_when_covered": percentile(ranks, 0.90),
        "gold_outside_top5_rate": (
            sum(row["gold_rank"] is None or row["gold_rank"] > 5 for row in records) / total
            if total else None
        ),
        "top1_top2_gap_mean": statistics.fmean(gaps) if gaps else None,
        "small_margin_rate_gap_lt_0_05": (
            sum(gap < 0.05 for gap in gaps) / len(gaps) if gaps else None
        ),
        "hard_case_rate": (
            sum(
                row["gold_rank"] is None
                or row["gold_rank"] > 5
                or (
                    row["top1_top2_gap"] is not None
                    and row["top1_top2_gap"] < 0.05
                )
                for row in records
            ) / total
            if total else None
        ),
        "difficulty_definition": (
            "hard = Gold missing from candidates, Gold rank > 5, or final Top1-Top2 gap < 0.05"
        ),
    }


def metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return ("%%.%df" % digits) % float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--gold-mapping-audit", required=True)
    parser.add_argument("--result-223", required=True)
    parser.add_argument("--evaluation-222", required=True)
    parser.add_argument("--evaluation-223", required=True)
    parser.add_argument("--summary-223", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--optimization-csv-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    args = parser.parse_args()
    outputs = [
        Path(args.json_out), Path(args.csv_out),
        Path(args.optimization_csv_out), Path(args.markdown_out),
    ]
    for output in outputs:
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)

    gold: Dict[tuple[str, str], set[str]] = defaultdict(set)
    gold_metadata: Dict[tuple[str, str], Dict[str, str]] = {}
    for row in read_jsonl(args.gold):
        query_id = str(row.get("query_id") or "")
        module = normalize_module(
            row.get("target_semantic_type") or row.get("semantic_type")
        )
        target_id = str(row.get("target_id") or "")
        if query_id and module and target_id:
            gold[(query_id, module)].add(target_id)
            gold_metadata.setdefault((query_id, module), {
                "sample_id": str(row.get("sample_id") or ""),
                "project": str(row.get("project") or "unknown_project"),
                "optimization": str(row.get("optimization") or "unknown"),
            })
    results = {
        str(row.get("query_id") or ""): row for row in read_jsonl(args.result_223)
    }
    detail_by_module: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    detail_by_optimization: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    detail_by_project: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    all_details = []
    for (query_id, module), targets in sorted(gold.items()):
        row = results.get(query_id, {})
        candidates = [
            dict(candidate)
            for candidate in row.get("candidates", []) or []
            if normalize_module(candidate.get("target_semantic_type")) == module
        ]
        candidates.sort(key=lambda item: (-score(item), str(item.get("target_id") or "")))
        ids = [str(item.get("target_id") or "") for item in candidates]
        ranks = [ids.index(target) + 1 for target in targets if target in ids]
        gap = score(candidates[0]) - score(candidates[1]) if len(candidates) > 1 else None
        detail = {
            "query_id": query_id,
            "module": module,
            **gold_metadata.get((query_id, module), {}),
            "candidate_count": len(candidates),
            "gold_rank": min(ranks) if ranks else None,
            "top1_top2_gap": gap,
        }
        detail_by_module[module].append(detail)
        detail_by_optimization[str(detail.get("optimization") or "unknown")].append(detail)
        detail_by_project[str(detail.get("project") or "unknown_project")].append(detail)
        all_details.append(detail)

    evaluation_222 = json.loads(Path(args.evaluation_222).read_text(encoding="utf-8"))
    evaluation_223 = json.loads(Path(args.evaluation_223).read_text(encoding="utf-8"))
    summary_223 = json.loads(Path(args.summary_223).read_text(encoding="utf-8"))
    mapping_audit = json.loads(
        Path(args.gold_mapping_audit).read_text(encoding="utf-8")
    )
    mapping_coverage = mapping_audit.get("mapping_coverage_by_module") or {}
    mapping_coverage_by_optimization = (
        mapping_audit.get("mapping_coverage_by_optimization") or {}
    )
    difficulty = {
        "overall": summarize(all_details),
        "by_module": {module: summarize(detail_by_module[module]) for module in MODULES},
        "by_optimization": {
            key: summarize(value)
            for key, value in sorted(detail_by_optimization.items())
        },
        "by_project": {
            key: summarize(value)
            for key, value in sorted(detail_by_project.items())
        },
    }
    report = {
        "schema": "symgen_external_metrics_and_candidate_difficulty_v1",
        "evaluation_role": "final external Test description only; never tuning",
        "gold_query_module_objects": len(gold),
        "gold_mapping_audit": {
            "oracle_cache_coverage": mapping_audit.get("oracle_cache_coverage"),
            "mapping_coverage_by_module": mapping_coverage,
            "mapping_coverage_by_optimization": mapping_coverage_by_optimization,
            "strict_gold_ready": mapping_audit.get("strict_gold_ready"),
        },
        "official_222": evaluation_222,
        "official_223": evaluation_223,
        "candidate_difficulty": difficulty,
        "s_global": {
            "initial": summary_223.get("initial_raw_s_global"),
            "final": summary_223.get("final_raw_s_global"),
            "delta": summary_223.get("raw_s_global_delta"),
            "iterations": summary_223.get("iterations_run"),
            "stop_reason": summary_223.get("stop_reason"),
        },
    }
    outputs[0].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fields = [
        "module", "gold_query_objects_before_mapping", "gold_mapping_rate",
        "gold_n", "candidate_coverage", "recall@1", "recall@5", "mrr",
        "candidate_count_mean", "candidate_count_p90", "gold_rank_mean_when_covered",
        "top1_top2_gap_mean", "hard_case_rate",
    ]
    csv_rows = []
    rankings = evaluation_223.get("ranking_by_module") or {}
    for module in MODULES:
        rank = (rankings.get(module) or {}).get("after_collaboration") or {}
        diff = difficulty["by_module"][module]
        mapped = mapping_coverage.get(module) or {}
        csv_rows.append({
            "module": module,
            "gold_query_objects_before_mapping": mapped.get("query_module_objects"),
            "gold_mapping_rate": mapped.get("mapping_rate"),
            "gold_n": rank.get("n"),
            "candidate_coverage": rank.get("candidate_coverage"),
            "recall@1": rank.get("recall@1"),
            "recall@5": rank.get("recall@5"),
            "mrr": rank.get("mrr"),
            "candidate_count_mean": diff.get("candidate_count_mean"),
            "candidate_count_p90": diff.get("candidate_count_p90"),
            "gold_rank_mean_when_covered": diff.get("gold_rank_mean_when_covered"),
            "top1_top2_gap_mean": diff.get("top1_top2_gap_mean"),
            "hard_case_rate": diff.get("hard_case_rate"),
        })
    with outputs[1].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    optimization_fields = [
        "optimization", "gold_query_objects_before_mapping", "gold_mapping_rate",
        "gold_n", "candidate_coverage", "recall@1",
        "recall@5", "mrr", "candidate_count_mean", "candidate_count_p90",
        "gold_rank_mean_when_covered", "top1_top2_gap_mean", "hard_case_rate",
    ]
    optimization_rows = []
    optimization_keys = sorted(
        set(difficulty["by_optimization"]) | set(mapping_coverage_by_optimization)
    )
    for optimization in optimization_keys:
        values = difficulty["by_optimization"].get(optimization) or summarize([])
        mapped = mapping_coverage_by_optimization.get(optimization) or {}
        optimization_rows.append({
            "optimization": optimization,
            "gold_query_objects_before_mapping": mapped.get("query_module_objects"),
            "gold_mapping_rate": mapped.get("mapping_rate"),
            "gold_n": values.get("gold_query_module_objects"),
            "candidate_coverage": values.get("gold_candidate_coverage"),
            "recall@1": values.get("recall@1"),
            "recall@5": values.get("recall@5"),
            "mrr": values.get("mrr"),
            "candidate_count_mean": values.get("candidate_count_mean"),
            "candidate_count_p90": values.get("candidate_count_p90"),
            "gold_rank_mean_when_covered": values.get("gold_rank_mean_when_covered"),
            "top1_top2_gap_mean": values.get("top1_top2_gap_mean"),
            "hard_case_rate": values.get("hard_case_rate"),
        })
    with outputs[2].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=optimization_fields)
        writer.writeheader()
        writer.writerows(optimization_rows)

    lines = [
        "# SymGen 外部 Test 最终结果",
        "",
        "> 本表只描述冻结模型的外部泛化结果，不用于重新训练、调参或选择传播方向。",
        "",
        "| 模块 | 映射前查询数 | Gold映射率 | Gold n | 候选覆盖率 | R@1 | R@5 | MRR | 平均候选数 | P90候选数 | 平均Gold名次 | Top1-Top2间隔 | 困难样本率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in csv_rows:
        lines.append(
            "| {module} | {pre_map_n} | {mapping_rate} | {gold_n} | {coverage} | {r1} | {r5} | {mrr} | {mean_c} | {p90_c} | {mean_rank} | {gap} | {hard} |".format(
                module=row["module"], gold_n=row["gold_n"] or 0,
                pre_map_n=row["gold_query_objects_before_mapping"] or 0,
                mapping_rate=metric(row["gold_mapping_rate"]),
                coverage=metric(row["candidate_coverage"]), r1=metric(row["recall@1"]),
                r5=metric(row["recall@5"]), mrr=metric(row["mrr"]),
                mean_c=metric(row["candidate_count_mean"], 2),
                p90_c=metric(row["candidate_count_p90"], 2),
                mean_rank=metric(row["gold_rank_mean_when_covered"], 2),
                gap=metric(row["top1_top2_gap_mean"]), hard=metric(row["hard_case_rate"]),
            )
        )
    lines.extend([
        "",
        "## 按优化级别",
        "",
        "| 优化级别 | 映射前查询数 | Gold映射率 | Gold n | 候选覆盖率 | R@1 | R@5 | MRR | 平均候选数 | 困难样本率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in optimization_rows:
        lines.append(
            "| {optimization} | {pre_map_n} | {mapping_rate} | {gold_n} | {coverage} | {r1} | {r5} | {mrr} | {mean_c} | {hard} |".format(
                optimization=row["optimization"], gold_n=row["gold_n"] or 0,
                pre_map_n=row["gold_query_objects_before_mapping"] or 0,
                mapping_rate=metric(row["gold_mapping_rate"]),
                coverage=metric(row["candidate_coverage"]),
                r1=metric(row["recall@1"]), r5=metric(row["recall@5"]),
                mrr=metric(row["mrr"]),
                mean_c=metric(row["candidate_count_mean"], 2),
                hard=metric(row["hard_case_rate"]),
            )
        )
    lines.extend([
        "",
        "## S_global",
        "",
        "- initial: `%s`" % metric(summary_223.get("initial_raw_s_global")),
        "- final: `%s`" % metric(summary_223.get("final_raw_s_global")),
        "- delta: `%s`" % metric(summary_223.get("raw_s_global_delta")),
        "- iterations: `%s`" % summary_223.get("iterations_run"),
        "- stop reason: `%s`" % summary_223.get("stop_reason"),
        "",
        "困难样本定义：Gold 不在候选集、Gold 名次大于 5，或最终 Top1-Top2 分差小于 0.05。",
        "",
    ])
    outputs[3].write_text("\n".join(lines), encoding="utf-8")
    print("SYMGEN EXTERNAL METRICS REPORT: WRITTEN")


if __name__ == "__main__":
    main()
