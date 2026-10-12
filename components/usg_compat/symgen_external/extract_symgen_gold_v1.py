#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在任何 stripped 推理之前，从 SymGen full/DWARF 冻结四模块 Oracle。

输出是与 USG 候选无关的原始 Gold：APISem 来自 ELF 动态符号与独立
反汇编 callsite，TypeSem/InterfaceSem/LayoutSem 来自 DWARF。它不包含
USG target_id，后续只在推理完成后由 map_symgen_gold_to_usg_v1.py 映射。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from symgen_dwarf_oracle_v1 import ORACLE_SCHEMA, extract_oracle


MODULES = ("APISem", "TypeSem", "LayoutSem", "InterfaceSem")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_gold_row(pair: Dict[str, str], oracle: Dict[str, Any]) -> Dict[str, Any]:
    functions = list(oracle.get("functions") or [])
    return {
        "schema": "symgen_raw_four_module_gold_v1",
        "sample_id": pair["sample_id"],
        "project": pair.get("project") or "",
        "optimization": pair.get("optimization") or "",
        "split": "test",
        "full_sha256": pair.get("full_sha256") or "",
        "modules": {
            "APISem": {
                "undefined_dynamic_symbols": oracle.get("undefined_dynamic_symbols") or [],
                "callsite_to_symbol": oracle.get("callsite_to_symbol") or {},
                "callsite_extraction": oracle.get("callsite_extraction") or {},
            },
            "TypeSem": [
                {
                    "low_pc": item.get("low_pc"),
                    "param_categories": item.get("param_categories") or [],
                    "return_category": item.get("return_category"),
                }
                for item in functions
            ],
            "InterfaceSem": [
                {
                    "name": item.get("name"),
                    "low_pc": item.get("low_pc"),
                    "high_pc": item.get("high_pc"),
                    "param_count": item.get("param_count"),
                    "param_categories": item.get("param_categories") or [],
                    "return_category": item.get("return_category"),
                }
                for item in functions
            ],
            "LayoutSem": {
                "global_struct_offsets": oracle.get("global_struct_offsets") or [],
                "function_struct_offsets": [
                    {
                        "low_pc": item.get("low_pc"),
                        "struct_offsets": item.get("struct_offsets") or [],
                    }
                    for item in functions
                    if item.get("struct_offsets")
                ],
            },
        },
        "gold_role": "evaluation_only_after_inference",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("--raw-gold", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--objdump")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-all-modules", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.pair_manifest).resolve()
    oracle_dir = Path(args.oracle_dir).resolve()
    raw_gold = Path(args.raw_gold).resolve()
    report_path = Path(args.report).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if oracle_dir.exists() and any(oracle_dir.iterdir()):
        raise FileExistsError("refusing non-empty Oracle directory: %s" % oracle_dir)
    for output in (raw_gold, report_path):
        if output.exists():
            raise FileExistsError("refusing to overwrite: %s" % output)
        output.parent.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("status") == "ok"]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit("ERROR: no paired samples")

    outputs = []
    errors = []
    module_usable_samples = Counter()
    for index, pair in enumerate(rows, 1):
        sample_id = str(pair.get("sample_id") or "")
        full_path = Path(str(pair.get("full_path") or "")).resolve()
        expected_hash = str(pair.get("full_sha256") or sample_id)
        try:
            if not full_path.is_file():
                raise FileNotFoundError(full_path)
            if sha256_file(full_path) != expected_hash:
                raise ValueError("full SHA256 mismatch")
            oracle = extract_oracle(str(full_path), objdump_path=args.objdump)
            if oracle.get("schema") != ORACLE_SCHEMA or not oracle.get("has_dwarf"):
                raise ValueError("invalid or DWARF-free Oracle")
            cache = oracle_dir / (sample_id + ".oracle_symgen_v1.json")
            cache.write_text(
                json.dumps(oracle, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            output = raw_gold_row(pair, oracle)
            outputs.append(output)
            modules = output["modules"]
            module_usable_samples["APISem"] += int(bool(modules["APISem"]["callsite_to_symbol"]))
            module_usable_samples["TypeSem"] += int(bool(modules["TypeSem"]))
            module_usable_samples["InterfaceSem"] += int(bool(modules["InterfaceSem"]))
            module_usable_samples["LayoutSem"] += int(
                bool(modules["LayoutSem"]["global_struct_offsets"])
                or bool(modules["LayoutSem"]["function_struct_offsets"])
            )
        except Exception as error:
            errors.append({"sample_id": sample_id, "error": str(error)})
        if index % 25 == 0 or index == len(rows):
            print("[%d/%d] frozen_oracles=%d" % (index, len(rows), len(outputs)), flush=True)

    with raw_gold.open("x", encoding="utf-8") as stream:
        for row in outputs:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    missing_modules = [
        module for module in MODULES if int(module_usable_samples[module]) <= 0
    ]
    report = {
        "schema": "symgen_raw_gold_extraction_report_v1",
        "pair_manifest": str(manifest),
        "requested_samples": len(rows),
        "frozen_oracles": len(outputs),
        "errors": len(errors),
        "error_examples": errors[:100],
        "module_usable_sample_counts": {
            module: int(module_usable_samples[module]) for module in MODULES
        },
        "require_all_modules": args.require_all_modules,
        "missing_modules": missing_modules,
        "gold_extracted_before_inference": True,
        "usg_or_candidate_files_read": False,
        "passed": (
            bool(outputs)
            and (args.allow_partial or not errors)
            and (not args.require_all_modules or not missing_modules)
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("ERROR: Gold extraction incomplete")
    print("SYMGEN RAW FOUR-MODULE GOLD: FROZEN")


if __name__ == "__main__":
    main()
