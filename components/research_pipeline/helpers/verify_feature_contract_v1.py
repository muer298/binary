#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐字段比较 ReSym 训练特征合同与 SymGen 推理特征合同。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONTRACT_KEYS = (
    "feature_schema",
    "feature_dim",
    "node_type_to_idx",
    "layer_to_idx",
    "numeric_feature_names",
    "hash_dims",
    "feature_layout",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-vocab", required=True)
    parser.add_argument("--external-vocab", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = Path(args.report)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(Path(args.frozen_vocab).read_text(encoding="utf-8"))
    external = json.loads(Path(args.external_vocab).read_text(encoding="utf-8"))
    mismatches = {}
    for key in CONTRACT_KEYS:
        if frozen.get(key) != external.get(key):
            mismatches[key] = {
                "frozen": frozen.get(key),
                "external": external.get(key),
            }
    report = {
        "schema": "frozen_feature_contract_verification_v1",
        "compared_keys": list(CONTRACT_KEYS),
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)
    print("FROZEN RESYM FEATURE CONTRACT ON SYMGEN: PASSED")


if __name__ == "__main__":
    main()
