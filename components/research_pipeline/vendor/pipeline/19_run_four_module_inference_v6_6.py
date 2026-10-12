#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run four specialised 2.2.3 modules over frozen 2.2.1 representations."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from module_common_223_v5_3 import (
    MODULES, MODULE_TO_SEMANTIC, base_candidate_score, normalize_semantic_type,
    output_path, read_jsonl, write_json,
)
from module_specialized_v6_6 import (
    GraphContext, collect_wanted_ids, load_heads, score_candidate,
)
from layout_reranker_v6_7_2 import (
    IMPLEMENTATION_REVISION as LAYOUT_IMPLEMENTATION_REVISION,
    load_layout_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning", required=True)
    parser.add_argument("--nodes")
    parser.add_argument("--edges")
    parser.add_argument("--head-checkpoint")
    parser.add_argument(
        "--layout-config",
        help=(
            "Validation-selected LayoutSem v6.7.3 evidence-bridge reranker "
            "config. "
            "When omitted, the Gold-free conservative default is used."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    wanted = collect_wanted_ids(args.reasoning)
    context = GraphContext.load(args.nodes, args.edges, wanted)
    heads = load_heads(args.head_checkpoint)
    layout_config, layout_config_path, layout_config_sha256 = (
        load_layout_config(args.layout_config)
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        module: output_path(str(out_dir / (module.lower() + "_v6_6.jsonl")), args.overwrite)
        for module in MODULES
    }
    writers = {module: path.open("w", encoding="utf-8") for module, path in paths.items()}
    counts = defaultdict(Counter)
    records = 0
    try:
        for row in read_jsonl(args.reasoning):
            records += 1
            grouped = defaultdict(list)
            for raw_candidate in row.get("candidates", []) or []:
                candidate = dict(raw_candidate)
                semantic_type = normalize_semantic_type(candidate.get("target_semantic_type"))
                if semantic_type in MODULE_TO_SEMANTIC.values():
                    grouped[semantic_type].append(candidate)
            decisions = row.get("semantic_decisions") if isinstance(row.get("semantic_decisions"), dict) else {}
            for module in MODULES:
                semantic_type = MODULE_TO_SEMANTIC[module]
                candidates = grouped.get(semantic_type, [])
                if not candidates:
                    continue
                for candidate in candidates:
                    score, trace = score_candidate(
                        semantic_type,
                        row,
                        candidate,
                        context,
                        heads,
                        layout_config=layout_config,
                    )
                    candidate["confidence_before_223"] = base_candidate_score(candidate)
                    candidate["module_score_223_v5_3"] = score
                    candidate["module_score_223_v6_6"] = score
                    candidate["module_feature_trace_223_v6_6"] = trace
                candidates.sort(key=lambda item: (-float(item["module_score_223_v6_6"]), str(item.get("target_id") or "")))
                previous = decisions.get(semantic_type) if isinstance(decisions.get(semantic_type), dict) else {}
                result = {
                    "schema": "task2_223_specialised_module_result_v6_6",
                    "query_id": row.get("query_id"),
                    "sample_id": row.get("sample_id"),
                    "module": module,
                    "semantic_type": semantic_type,
                    "input_status": previous.get("status", "hypothesis"),
                    "input_locked": bool(previous.get("locked")),
                    "input_best_target_id": previous.get("best_target_id"),
                    "candidate_scores": candidates,
                    "preferred_target_id": candidates[0].get("target_id"),
                    "module_confidence": candidates[0]["module_score_223_v6_6"],
                    "module_gap": candidates[0]["module_score_223_v6_6"] - (candidates[1]["module_score_223_v6_6"] if len(candidates) > 1 else 0.0),
                    "frozen_221_representation_reused": True,
                    "specialised_algorithm": True,
                }
                writers[module].write(json.dumps(result, ensure_ascii=False) + "\n")
                counts[module]["records"] += 1
                counts[module]["candidates"] += len(candidates)
                counts[module]["locked_inputs"] += int(result["input_locked"])
                counts[module]["node_metadata_hits"] += sum(
                    int(item["module_feature_trace_223_v6_6"]["target_node_metadata_available"])
                    for item in candidates
                )
                if semantic_type == "LayoutSem":
                    layout_traces = [
                        item["module_feature_trace_223_v6_6"].get(
                            "layout_pairwise_trace"
                        ) or {}
                        for item in candidates
                    ]
                    counts[module]["layout_pairwise_candidates"] += len(
                        layout_traces
                    )
                    counts[module]["layout_pairwise_usable_candidates"] += sum(
                        int(bool(trace.get("usable")))
                        for trace in layout_traces
                    )
                    counts[module]["layout_query_groups_with_usable_pair"] += int(
                        any(bool(trace.get("usable")) for trace in layout_traces)
                    )
    finally:
        for writer in writers.values():
            writer.close()

    summary = {
        "schema": "task2_223_four_specialised_modules_summary_v6_6",
        "reasoning_records": records,
        "wanted_graph_ids": len(wanted),
        "loaded_node_metadata": len(context.nodes),
        "loaded_incident_edge_ids": len(context.incident),
        "outputs": {module: str(path) for module, path in paths.items()},
        "module_counts": {module: dict(value) for module, value in counts.items()},
        "learned_heads_loaded": sorted(heads),
        "gold_used_during_inference": False,
        "test_used_during_inference": False,
        "uncertainty_policy": "all candidates preserved; specialised module scores are recommendations",
        "representation_contract": "reuse frozen 2.2.1 node representation; no duplicate deep encoder",
        "implementation_revision": "v6_7_3_layoutsem_evidence_bridge",
        "layout_reranker_revision": LAYOUT_IMPLEMENTATION_REVISION,
        "layout_reranker_config_file": layout_config_path,
        "layout_reranker_config_sha256": layout_config_sha256,
        "layout_reranker_config": layout_config,
        "layout_gold_used_during_inference": False,
    }
    write_json(args.summary_out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
