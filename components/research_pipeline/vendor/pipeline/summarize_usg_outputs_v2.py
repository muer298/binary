#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_usg_outputs_v2.py

Purpose:
  Summarize USG semantic_dict JSON outputs and, when available, combine them
  with batch_report.json status / runtime information.

Outputs:
  1. summary CSV: one row per JSON output.
  2. status JSON: aggregate status counts, elapsed time, and output counts.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-status-json", default=None)
    return ap.parse_args()


def get_nested(d, path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def load_batch(root: Path):
    path = root / "batch_report.json"
    if not path.exists():
        return {}, []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, []
    by_final = {}
    for row in rows:
        final = row.get("final_json")
        if final:
            by_final[str(Path(final).resolve())] = row
    return by_final, rows


def summarize_json(path: Path, batch_info=None):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    low = data.get("low_level_sem", {})
    mid = data.get("mid_level_sem", {})
    high = data.get("high_level_sem", {})
    usg_sum = get_nested(data, ["usg", "summary"], {}) or {}
    cr_sum = get_nested(data, ["consistency_report", "summary"], {}) or {}

    row = {
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
        "run_status": "",
        "elapsed_sec": "",
    }
    if batch_info:
        row["run_status"] = batch_info.get("status", "")
        row["elapsed_sec"] = batch_info.get("elapsed_sec", "")
    return row


def main():
    args = parse_args()
    root = Path(args.json_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_status_json = Path(args.out_status_json).resolve() if args.out_status_json else out_csv.with_suffix(".status.json")

    batch_by_final, batch_rows = load_batch(root)
    json_paths = sorted(root.glob("*_semantic_dict.json"))

    rows = []
    parse_errors = []
    for path in json_paths:
        try:
            batch_info = batch_by_final.get(str(path.resolve()))
            rows.append(summarize_json(path, batch_info=batch_info))
        except Exception as exc:
            parse_errors.append({"file": str(path), "error": repr(exc)})

    fields = [
        "file", "program", "schema_version", "generator",
        "instr_count", "block_count", "cfg_func_count", "dfg_func_count",
        "type_func_count", "interface_count", "api_count", "layout_count",
        "evidence_edge_count", "constraint_edge_count", "node_state_count",
        "usg_node_count", "usg_evidence_edge_count", "usg_constraint_edge_count",
        "consistency_status", "consistency_errors", "consistency_warnings", "consistency_infos",
        "run_status", "elapsed_sec",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    status_counter = Counter(row.get("status", "unknown") for row in batch_rows)
    elapsed_sum = 0.0
    elapsed_count = 0
    for row in batch_rows:
        value = row.get("elapsed_sec")
        if isinstance(value, (int, float)):
            elapsed_sum += float(value)
            elapsed_count += 1

    status = {
        "json_root": str(root),
        "batch_report_rows": len(batch_rows),
        "batch_status_counts": dict(status_counter),
        "semantic_json_count": len(json_paths),
        "summary_rows": len(rows),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:50],
        "elapsed_sum_sec": round(elapsed_sum, 3),
        "elapsed_sum_min": round(elapsed_sum / 60, 3),
        "elapsed_count": elapsed_count,
    }
    out_status_json.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_csv} ({len(rows)} rows)")
    print(f"Wrote {out_status_json}")
    print("status_counts =", dict(status_counter))
    print("semantic_json_count =", len(json_paths))


if __name__ == "__main__":
    main()
