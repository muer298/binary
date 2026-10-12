#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect semantic_dict JSON structure produced by over.py / over_v6_light.py.

Usage:
python3 inspect_semantic_json.py --json-root /path/to/semantic_json --limit 3
"""
import argparse
import json
from pathlib import Path


def summarize_obj(obj, depth=0, max_depth=3):
    indent = "  " * depth
    if depth > max_depth:
        print(indent + "...")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                print(f"{indent}{k}: dict({len(v)})")
                summarize_obj(v, depth + 1, max_depth)
            elif isinstance(v, list):
                print(f"{indent}{k}: list({len(v)})")
                if v and depth + 1 <= max_depth:
                    first = v[0]
                    if isinstance(first, dict):
                        print(f"{indent}  first item keys: {list(first.keys())[:30]}")
                    else:
                        print(f"{indent}  first item type: {type(first).__name__}")
            else:
                print(f"{indent}{k}: {type(v).__name__} = {repr(v)[:120]}")
    elif isinstance(obj, list):
        print(f"{indent}list({len(obj)})")
        if obj:
            summarize_obj(obj[0], depth + 1, max_depth)
    else:
        print(f"{indent}{type(obj).__name__}: {repr(obj)[:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", required=True)
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.json_root)
    paths = sorted(root.glob("*_semantic_dict.json"))
    print("json files =", len(paths))

    for p in paths[:args.limit]:
        print("\n" + "=" * 100)
        print("FILE:", p)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        print("top-level keys:", list(data.keys()))
        print("\nSTRUCTURE:")
        summarize_obj(data, depth=0, max_depth=3)

        hs = data.get("high_level_sem", {})
        print("\nHIGH_LEVEL_SEM keys:", list(hs.keys()) if isinstance(hs, dict) else type(hs).__name__)
        if isinstance(hs, dict):
            for key in ["type_sem", "interface_sem", "api_sem", "mem_layout_sem", "layout_sem"]:
                v = hs.get(key)
                if isinstance(v, list):
                    print(f"{key}: {len(v)}")
                    if v:
                        print("  first:", json.dumps(v[0], ensure_ascii=False)[:1000])
                elif v is not None:
                    print(f"{key}: {type(v).__name__}")


if __name__ == "__main__":
    main()
