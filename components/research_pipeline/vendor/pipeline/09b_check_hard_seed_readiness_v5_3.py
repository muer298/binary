#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail early when a v5.3 run still has no auditable hard API seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--min-hard-seeds", type=int, default=1)
    parser.add_argument(
        "--allow-zero",
        action="store_true",
        help="Print a warning instead of returning a failure code",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    value = json.loads(summary_path.read_text(encoding="utf-8"))
    eligible = int(value.get("hard_api_eligible_candidate_count", 0) or 0)
    seeds = int(value.get("hard_api_seed_count", 0) or 0)
    conflicts = int(value.get("hard_seed_conflicts", 0) or 0)
    recovered = value.get("api_sources_recovered_from_nodes") or {}
    source_counts = value.get("hard_api_seed_sources") or {}

    report = {
        "schema": "task2_222_hard_seed_readiness_v5_3_1",
        "ready": seeds >= max(0, args.min_hard_seeds),
        "min_hard_seeds": max(0, args.min_hard_seeds),
        "hard_api_eligible_candidate_count": eligible,
        "hard_api_seed_count": seeds,
        "hard_seed_conflicts": conflicts,
        "api_sources_recovered_from_nodes": recovered,
        "hard_api_seed_sources": source_counts,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    if report["ready"]:
        return

    message = (
        "No usable deterministic APISem seed was recovered. The run is "
        "structurally executable but the v5.3 hard-seed propagation path is "
        "inactive. Inspect complete USG node provenance before a full run."
    )
    if args.allow_zero:
        print("WARNING: " + message, flush=True)
        return
    raise SystemExit("ERROR: " + message)


if __name__ == "__main__":
    main()
