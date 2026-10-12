#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_build_anchors.py

Task 2 Day-1 Step 3:
Build cross-layer anchors from USG evidence edges.

The research design treats evidence edges as natural cross-layer pairing signals.
This script converts edge_kind=evidence records into anchors grouped by evidence type:
  fixed_offset, deref_chain, def_use, api_call, unknown

Inputs:
  usg_nodes.jsonl
  usg_edges.jsonl

Outputs:
  anchors.jsonl
  anchor_stats.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def classify_anchor(edge, node_map):
    et = str(edge.get("edge_type") or edge.get("evidence_type") or "").lower()
    tgt = node_map.get(edge.get("target_node"), {})
    tgt_type = tgt.get("node_type", "")

    if any(x in et for x in ["fixed_offset", "offset", "layout", "field"]):
        return "fixed_offset"
    if any(x in et for x in ["deref", "pointer", "ptr", "load_chain"]):
        return "deref_chain"
    if any(x in et for x in ["def_use", "def-use", "entry", "param", "interface", "data_flow", "dfg"]):
        return "def_use"
    if any(x in et for x in ["api", "call", "libcall"]):
        return "api_call"

    # Fallback by high-level target type.
    if tgt_type == "LayoutSem":
        return "fixed_offset"
    if tgt_type == "TypeSem":
        return "deref_chain"
    if tgt_type == "InterfaceSem":
        return "def_use"
    if tgt_type == "APISem":
        return "api_call"

    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--min-strength", type=float, default=None)
    ap.add_argument("--resolved-only", action="store_true", help="Only keep anchors with resolved source and target node ids.")
    args = ap.parse_args()

    nodes_path = Path(args.nodes).resolve()
    edges_path = Path(args.edges).resolve()
    out = Path(args.out).resolve()
    stats_path = Path(args.stats).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    node_map = {}
    for n in load_jsonl(nodes_path):
        node_map[n["node_id"]] = n

    anchors = []
    cnt_type = Counter()
    cnt_sample = Counter()
    cnt_source_type = Counter()
    cnt_target_type = Counter()
    resolved_cnt = Counter()

    for e in load_jsonl(edges_path):
        if e.get("edge_kind") != "evidence":
            continue

        strength = e.get("strength")
        if args.min_strength is not None and isinstance(strength, (int, float)) and strength < args.min_strength:
            continue

        src_id = e.get("source_node", "")
        tgt_id = e.get("target_node", "")
        src = node_map.get(src_id, {})
        tgt = node_map.get(tgt_id, {})

        src_ok = bool(e.get("source_resolved")) and bool(src)
        tgt_ok = bool(e.get("target_resolved")) and bool(tgt)

        if args.resolved_only and not (src_ok and tgt_ok):
            continue

        atype = classify_anchor(e, node_map)
        sample_id = e.get("sample_id", "")

        # Choose function entry from target first because target is often high-level.
        function_entry = ""
        if tgt:
            function_entry = tgt.get("function_entry") or tgt.get("attrs", {}).get("function_entry") or tgt.get("attrs", {}).get("func_entry") or ""
        if not function_entry and src:
            function_entry = src.get("function_entry") or src.get("attrs", {}).get("function_entry") or src.get("attrs", {}).get("func_entry") or ""

        anchor_id = f"{sample_id}::anchor::{len(anchors)}"
        rec = {
            "anchor_id": anchor_id,
            "sample_id": sample_id,
            "evidence_type": atype,
            "edge_id": e.get("edge_id"),
            "edge_type_raw": e.get("edge_type"),
            "source_node": src_id,
            "target_node": tgt_id,
            "source_type": src.get("node_type", "EvidenceStub" if "EvidenceStub" in str(src_id) else "unresolved"),
            "target_type": tgt.get("node_type", "unresolved"),
            "source_resolved": src_ok,
            "target_resolved": tgt_ok,
            "source_raw": e.get("source_raw"),
            "target_raw": e.get("target_raw"),
            "strength": strength,
            "function_entry": str(function_entry),
            "source_attrs": src.get("attrs", {}),
            "target_attrs": tgt.get("attrs", {}),
            "trace": {
                "edge_source_path": e.get("source_path", ""),
                "source_json": src.get("json_path", tgt.get("json_path", "")),
            },
        }
        anchors.append(rec)

        cnt_type[atype] += 1
        cnt_sample[sample_id] += 1
        cnt_source_type[rec["source_type"]] += 1
        cnt_target_type[rec["target_type"]] += 1
        resolved_cnt[f"source_{src_ok}"] += 1
        resolved_cnt[f"target_{tgt_ok}"] += 1
        resolved_cnt[f"both_{src_ok and tgt_ok}"] += 1

    with out.open("w", encoding="utf-8") as f:
        for a in anchors:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    per_sample = list(cnt_sample.values())
    stats = {
        "nodes": str(nodes_path),
        "edges": str(edges_path),
        "anchor_count": len(anchors),
        "anchor_type_counts": dict(cnt_type),
        "source_type_counts": dict(cnt_source_type),
        "target_type_counts": dict(cnt_target_type),
        "resolution_counts": dict(resolved_cnt),
        "sample_count_with_anchors": len(cnt_sample),
        "avg_anchors_per_sample": (sum(per_sample) / len(per_sample)) if per_sample else 0,
        "max_anchors_per_sample": max(per_sample) if per_sample else 0,
        "min_anchors_per_sample": min(per_sample) if per_sample else 0,
        "out": str(out),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("wrote", out)
    print("wrote", stats_path)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
