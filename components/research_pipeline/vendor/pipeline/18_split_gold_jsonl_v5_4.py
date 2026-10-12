#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split an audited gold JSONL into frozen Train/Validation/Test files."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        split: out_dir / ("gold_%s_v5_4.jsonl" % split)
        for split in ("train", "validation", "test")
    }
    for path in paths.values():
        if path.exists() and not args.overwrite:
            raise FileExistsError("Refusing to overwrite: %s" % path)
    streams = {
        split: path.open("w", encoding="utf-8")
        for split, path in paths.items()
    }
    counts = Counter()
    try:
        with Path(args.gold).open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                split = str(row.get("split") or "")
                if split not in streams:
                    raise ValueError("Invalid split in gold: %s" % split)
                streams[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[split] += 1
    finally:
        for stream in streams.values():
            stream.close()
    result = {
        "schema": "research2_usg_gold_split_summary_v1",
        "source": args.gold,
        "outputs": {key: str(value) for key, value in paths.items()},
        "counts": dict(counts),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
