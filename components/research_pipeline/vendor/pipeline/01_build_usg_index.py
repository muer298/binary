#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_usg_index.py

Task 2 Day-1 Step 1:
Build an index over semantic_dict JSON files produced by Task 1.

Inputs:
  --json-root  /path/to/semantic_json
Outputs:
  --out        /path/to/work/graphs/usg_index.jsonl
  --stats      /path/to/work/reports/usg_index_stats.json

Each JSONL row records one sample and its semantic_dict path. If .meta.json
or batch_report.json exists, it also records run status and original binary path.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def infer_sample_id(json_path: Path) -> str:
    name = json_path.name
    if name.endswith("_semantic_dict.json"):
        return name[:-len("_semantic_dict.json")]
    return json_path.stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", required=True, help="Directory containing *_semantic_dict.json.")
    ap.add_argument("--out", required=True, help="Output JSONL index path.")
    ap.add_argument("--stats", required=True, help="Output stats JSON path.")
    args = ap.parse_args()

    json_root = Path(args.json_root).resolve()
    out = Path(args.out).resolve()
    stats_path = Path(args.stats).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    semantic_jsons = sorted(json_root.glob("*_semantic_dict.json"))

    batch = {}
    batch_report = json_root / "batch_report.json"
    if batch_report.exists():
        br = load_json(batch_report)
        if isinstance(br, list):
            for row in br:
                sid = row.get("safe_id") or row.get("sample_id")
                if sid:
                    batch[str(sid)] = row

    rows = []
    cnt = Counter()
    for p in semantic_jsons:
        sid = infer_sample_id(p)
        meta_path = json_root / f"{sid}.meta.json"
        meta = load_json(meta_path) if meta_path.exists() else {}
        if not isinstance(meta, dict):
            meta = {}

        # Prefer .meta.json, fallback to batch_report.
        b = batch.get(sid, {})
        manifest_row = meta.get("manifest_row") or b.get("manifest_row") or {}

        run_status = meta.get("status") or b.get("status") or "unknown"
        stripped_path = manifest_row.get("stripped_path") or meta.get("binary") or b.get("binary") or ""
        full_path = manifest_row.get("full_path") or ""

        program_name = ""
        schema_version = ""
        generator = ""
        try:
            d = load_json(p) or {}
            pi = d.get("program_info", {}) if isinstance(d, dict) else {}
            program_name = pi.get("name", "") or pi.get("program", "") or ""
            schema_version = pi.get("schema_version", "") or ""
            generator = pi.get("generator", "") or ""
        except Exception:
            pass

        row = {
            "sample_id": sid,
            "json_path": str(p),
            "json_root": str(json_root),
            "meta_path": str(meta_path) if meta_path.exists() else "",
            "run_status": run_status,
            "stripped_path": stripped_path,
            "full_path": full_path,
            "program_name": program_name,
            "schema_version": schema_version,
            "generator": generator,
        }
        rows.append(row)
        cnt[run_status] += 1

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "json_root": str(json_root),
        "semantic_json_count": len(semantic_jsons),
        "indexed_count": len(rows),
        "run_status_counts": dict(cnt),
        "output": str(out),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("wrote", out)
    print("wrote", stats_path)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
