#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由冻结 full 清单生成严格配对的 stripped ELF。

源 full ELF 不被修改；stripped 副本用于 Ghidra 和全部推理。脚本同时
验证 DWARF/SYMTAB 已从副本移除、架构保持一致、full 哈希没有变化。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import struct
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def machine_id(path: Path) -> int | None:
    with path.open("rb") as stream:
        header = stream.read(20)
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    byte_order = {1: "<", 2: ">"}.get(header[5])
    return struct.unpack(byte_order + "H", header[18:20])[0] if byte_order else None


def sections(path: Path, readelf: str) -> set[str]:
    completed = subprocess.run(
        [readelf, "-W", "-S", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=90,
    )
    result = set()
    for line in completed.stdout.splitlines():
        for token in line.replace("[", " ").replace("]", " ").split():
            if token.startswith("."):
                result.add(token)
                break
    return result


def select_tool(explicit: str) -> str:
    choices = [explicit] if explicit != "auto" else ["objcopy", "llvm-objcopy", "strip"]
    for choice in choices:
        resolved = shutil.which(choice)
        if resolved:
            return resolved
    raise SystemExit("ERROR: objcopy/llvm-objcopy/strip not found")


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)[:100]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument(
        "--inference-manifest",
        required=True,
        help=(
            "仅含 stripped_path 与非 Gold 元数据的 CSV；Ghidra 和全部推理只读取它"
        ),
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--strip-tool", default="auto")
    parser.add_argument("--readelf", default="readelf")
    args = parser.parse_args()

    source_manifest = Path(args.full_manifest).resolve()
    out_root = Path(args.out_root).resolve()
    pair_manifest = Path(args.pair_manifest).resolve()
    inference_manifest = Path(args.inference_manifest).resolve()
    report_path = Path(args.report).resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError("refusing non-empty out-root: %s" % out_root)
    for output in (pair_manifest, inference_manifest, report_path):
        if output.exists():
            raise FileExistsError("refusing to overwrite: %s" % output)
        output.parent.mkdir(parents=True, exist_ok=True)
    strip_tool = select_tool(args.strip_tool)
    readelf = shutil.which(args.readelf)
    if readelf is None:
        raise SystemExit("ERROR: readelf not found")

    with source_manifest.open("r", encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    if not source_rows:
        raise SystemExit("ERROR: full manifest is empty")
    stripped_root = out_root / "stripped"
    stripped_root.mkdir(parents=True, exist_ok=False)

    rows: list[Dict[str, Any]] = []
    failures = []
    for index, row in enumerate(source_rows, 1):
        full_path = Path(str(row.get("full_path") or "")).resolve()
        sample_id = str(row.get("sample_id") or "")
        expected_hash = str(row.get("full_sha256") or sample_id)
        if not full_path.is_file() or not sample_id:
            failures.append({"sample_id": sample_id, "error": "missing_full_or_id"})
            continue
        actual_hash = sha256_file(full_path)
        if actual_hash != expected_hash:
            failures.append({"sample_id": sample_id, "error": "full_hash_mismatch"})
            continue
        suffix = safe_name(full_path.name) or "binary"
        stripped_path = stripped_root / (sample_id + "__" + suffix)
        temporary = stripped_root / (".%s.%d.tmp" % (sample_id, os.getpid()))
        shutil.copy2(full_path, temporary)
        completed = subprocess.run(
            [strip_tool, "--strip-all", str(temporary)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            temporary.unlink(missing_ok=True)
            failures.append({
                "sample_id": sample_id,
                "error": "strip_failed",
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-1000:],
            })
            continue
        os.replace(temporary, stripped_path)
        stripped_sections = sections(stripped_path, readelf)
        pair_errors = []
        if ".debug_info" in stripped_sections:
            pair_errors.append("stripped_still_has_debug_info")
        if ".symtab" in stripped_sections:
            pair_errors.append("stripped_still_has_symtab")
        if machine_id(full_path) != machine_id(stripped_path):
            pair_errors.append("machine_mismatch")
        if sha256_file(full_path) != expected_hash:
            pair_errors.append("full_changed_after_strip")
        if pair_errors:
            failures.append({"sample_id": sample_id, "errors": pair_errors})
            continue
        rows.append({
            **row,
            "status": "ok",
            "full_path_role": "gold_only",
            "stripped_path": str(stripped_path),
            "stripped_sha256": sha256_file(stripped_path),
            "stripped_path_role": "inference_only",
            "pair_order": len(rows) + 1,
        })
        print("[%d/%d] %s" % (index, len(source_rows), sample_id[:16]), flush=True)

    fieldnames = list(rows[0]) if rows else ["sample_id", "full_path", "stripped_path", "status"]
    with pair_manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    inference_fields = (
        "sample_id", "project", "optimization", "architecture", "elf_class",
        "stripped_path", "stripped_sha256", "status", "stripped_path_role",
        "pair_order",
    )
    inference_rows = [
        {key: row.get(key, "") for key in inference_fields}
        for row in rows
    ]
    with inference_manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=inference_fields)
        writer.writeheader()
        writer.writerows(inference_rows)
    report = {
        "schema": "symgen_strict_pair_manifest_v1",
        "full_manifest": str(source_manifest),
        "strip_tool": strip_tool,
        "input_rows": len(source_rows),
        "paired_rows": len(rows),
        "inference_manifest": str(inference_manifest),
        "inference_manifest_contains_full_path": False,
        "inference_manifest_rows": len(inference_rows),
        "failures": len(failures),
        "failure_examples": failures[:100],
        "project_counts": dict(Counter(row.get("project") or "unknown" for row in rows)),
        "optimization_counts": dict(Counter(row.get("optimization") or "unknown" for row in rows)),
        "gold_isolation": "full=Gold only; stripped=inference only",
        "passed": (
            bool(rows)
            and not failures
            and len(rows) == len(source_rows)
            and len(inference_rows) == len(rows)
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("ERROR: strict pairing failed; inspect report")
    print("SYMGEN FULL/STRIPPED PAIR CONTRACT: PASSED")


if __name__ == "__main__":
    main()
