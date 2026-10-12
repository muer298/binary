#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task2-src", required=True)
    args = parser.parse_args()

    source = Path(args.task2_src).resolve()
    sys.path.insert(0, str(source))
    common = importlib.import_module("module_common_223_v5_3")

    payload = {
        "schema": "task2_222_confirm_policy_v6_7_fail_closed",
        "policies": {
            "LayoutSem": {
                "soft_confirmation_enabled": False,
                "min_evidence": 0.50,
                "gap_threshold": 0.10,
                "min_evidence_coverage": 0.20,
            }
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        policy_path = Path(directory) / "policy.json"
        policy_path.write_text(
            json.dumps(payload), encoding="utf-8"
        )
        policy, resolved = common.load_confirmation_policy_223(
            str(policy_path)
        )

    candidates = [
        {
            "target_id": "best",
            "score_223_v5_3": 0.99,
            "evidence_coverage": 1.0,
        },
        {
            "target_id": "second",
            "score_223_v5_3": 0.10,
            "evidence_coverage": 1.0,
        },
    ]
    result = common.decision_for_group(
        "LayoutSem",
        candidates,
        {},
        confirmation_policy=policy,
    )
    assert resolved, resolved
    assert result["status"] == "hypothesis", result
    assert result["propagation_weight"] == 0.0, result
    assert result["locked"] is False, result
    assert (
        result["confirmation_source"]
        == "validation_policy_fail_closed"
    ), result
    print("2.2.3 POLICY HANDOFF RUNTIME TEST: PASSED")


if __name__ == "__main__":
    main()
