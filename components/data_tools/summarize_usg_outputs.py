#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize USG JSON outputs produced by overall_usg_semantic_fixed_v5.py.

Example:
python summarize_usg_outputs.py --json-root ./usg_outputs --out-csv ./usg_outputs/summary.csv
"""
import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json-root", required=True)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def get_nested(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def summarize_one(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    low = data.get("low_level_sem", {})
    mid = data.get("mid_level_sem", {})
    high = data.get("high_level_sem", {})
    usg_sum = get_nested(data, ["usg", "summary"], {})

    cr = data.get("consistency_report", {})
    cr_sum = cr.get("summary", {})

    return {
        "file": str(path),
        "program": get_nested(data, ["program_info", "name"], ""),
        "schema_version": get_nested(data, ["program_info", "schema_version"], ""),
        "generator": get_nested(data, ["program_info", "generator"], ""),
        "instr_count": len(low.get("instr_sem", []) or []),
        "block_count": len(low.get("block_sem", []) or []),
        "cfg_func_count": len(mid.get("cfg_sem", []) or []),
        "dfg_func_count": len(mid.get("dataflow_sem", []) or []),
        "type_func_count": len(high.get("type_sem", []) or []),
        "interface_count": len(high.get("interface_sem", []) or []),
        "api_count": len(high.get("api_sem", []) or []),
        "layout_count": len(high.get("mem_layout_sem", []) or []),
        "evidence_edge_count": len(data.get("cross_layer_mapping", []) or []),
        "constraint_edge_count": len(data.get("constraint_edges", []) or []),
        "node_state_count": len(data.get("node_state", []) or []),
        "usg_node_count": usg_sum.get("node_count", ""),
        "usg_evidence_edge_count": usg_sum.get("evidence_edge_count", ""),
        "usg_constraint_edge_count": usg_sum.get("constraint_edge_count", ""),
        "consistency_status": cr_sum.get("status", ""),
        "consistency_errors": cr_sum.get("error_count", ""),
        "consistency_warnings": cr_sum.get("warning_count", ""),
        "consistency_infos": cr_sum.get("info_count", ""),
    }


def main():
    args = parse_args()
    root = Path(args.json_root)
    paths = sorted(root.glob("*_semantic_dict.json"))
    rows = []
    for p in paths:
        try:
            rows.append(summarize_one(p))
        except Exception as e:
            rows.append({"file": str(p), "program": "", "schema_version": "", "generator": "", "consistency_status": "parse_error", "consistency_errors": str(e)})

    fields = [
        "file", "program", "schema_version", "generator",
        "instr_count", "block_count", "cfg_func_count", "dfg_func_count",
        "type_func_count", "interface_count", "api_count", "layout_count",
        "evidence_edge_count", "constraint_edge_count", "node_state_count",
        "usg_node_count", "usg_evidence_edge_count", "usg_constraint_edge_count",
        "consistency_status", "consistency_errors", "consistency_warnings", "consistency_infos",
    ]
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
