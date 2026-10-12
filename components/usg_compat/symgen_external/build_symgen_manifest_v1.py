#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""建立冻结的 SymGen full-ELF 清单。

首轮外部验证只选择 x86_64 Linux ELF、O0/O1/O2/O3，并要求 full ELF
含 DWARF。选择顺序只依赖相对路径，不读取任何模型分数或 Test 指标。
full 文件只登记为 Gold 来源，绝不作为后续推理输入。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import struct
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


MACHINES = {3: "x86_32", 8: "mips", 40: "arm", 62: "x86_64", 183: "aarch64"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def elf_header(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("rb") as stream:
            header = stream.read(64)
    except OSError:
        return None
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    byte_order = {1: "<", 2: ">"}.get(header[5])
    if byte_order is None:
        return None
    machine_id = struct.unpack(byte_order + "H", header[18:20])[0]
    return {
        "elf_class": {1: 32, 2: 64}.get(header[4], 0),
        "architecture": MACHINES.get(machine_id, "machine_%d" % machine_id),
        "machine_id": machine_id,
    }


def section_names(path: Path, readelf: str) -> set[str]:
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
    if completed.returncode != 0:
        return set()
    result = set()
    for line in completed.stdout.splitlines():
        for token in line.replace("[", " ").replace("]", " ").split():
            if token.startswith("."):
                result.add(token)
                break
    return result


def path_metadata(relative: Path) -> tuple[str, str]:
    parents = list(relative.parts[:-1])
    lowers = [value.lower() for value in parents]
    optimization = ""
    project = ""
    for index, value in enumerate(lowers):
        if value in {"o0", "o1", "o2", "o3"}:
            optimization = value.upper()
            if index + 1 < len(parents):
                project = parents[index + 1]
            elif index > 0:
                project = parents[index - 1]
            break
    if not project and parents:
        project = parents[-1]
    return optimization, project


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="SymGen 解压根目录")
    parser.add_argument("--manifest", required=True, help="冻结 full 清单 CSV")
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--arch", default="x86_64")
    parser.add_argument("--optimizations", default="O0,O1,O2,O3")
    parser.add_argument("--projects", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--readelf", default="readelf")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    readelf = shutil.which(args.readelf)
    if readelf is None:
        raise SystemExit("ERROR: readelf not found")
    outputs = [Path(args.manifest), Path(args.manifest_jsonl), Path(args.report)]
    for output in outputs:
        if output.exists():
            raise FileExistsError("refusing to overwrite: %s" % output)
        output.parent.mkdir(parents=True, exist_ok=True)

    wanted_opts = {
        item.strip().upper() for item in args.optimizations.split(",") if item.strip()
    }
    wanted_projects = {
        item.strip() for item in args.projects.split(",") if item.strip()
    }
    candidates = []
    scanned_files = 0
    elf_files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        scanned_files += 1
        header = elf_header(path)
        if header is None:
            continue
        elf_files += 1
        if header["architecture"] != args.arch:
            continue
        relative = path.relative_to(root)
        optimization, project = path_metadata(relative)
        if wanted_opts and optimization not in wanted_opts:
            continue
        if wanted_projects and project not in wanted_projects:
            continue
        sections = section_names(path, readelf)
        if ".debug_info" not in sections:
            continue
        candidates.append((relative.as_posix(), path, header, sections, optimization, project))

    candidates.sort(key=lambda item: item[0])
    if args.max_samples > 0:
        candidates = candidates[: args.max_samples]
    if not candidates:
        raise SystemExit("ERROR: no matching x86_64 DWARF ELF files")

    rows = []
    duplicate_content = []
    seen_hashes = set()
    for relative, path, header, sections, optimization, project in candidates:
        digest = sha256_file(path)
        if digest in seen_hashes:
            duplicate_content.append(relative)
            continue
        seen_hashes.add(digest)
        rows.append({
            "sample_id": digest,
            "project": project,
            "optimization": optimization,
            "architecture": header["architecture"],
            "elf_class": header["elf_class"],
            "full_path": str(path),
            "source_relative_path": relative,
            "full_sha256": digest,
            "size_bytes": path.stat().st_size,
            "has_debug_info": True,
            "has_symtab": ".symtab" in sections,
            "status": "selected",
            "full_path_role": "gold_only",
            "selection_order": len(rows) + 1,
        })

    fieldnames = list(rows[0])
    with outputs[0].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_jsonl(outputs[1], rows)
    report = {
        "schema": "symgen_full_manifest_v1",
        "root": str(root),
        "selection": {
            "architecture": args.arch,
            "optimizations": sorted(wanted_opts),
            "projects": sorted(wanted_projects),
            "order": "lexicographic_source_relative_path",
            "max_samples": args.max_samples,
            "model_or_test_metric_used": False,
        },
        "filesystem_files_scanned": scanned_files,
        "elf_files": elf_files,
        "selected_unique_full_elf": len(rows),
        "duplicate_content_skipped": len(duplicate_content),
        "project_counts": dict(Counter(row["project"] for row in rows)),
        "optimization_counts": dict(Counter(row["optimization"] for row in rows)),
        "full_path_role": "Gold extraction only; never inference",
        "passed": bool(rows),
    }
    outputs[2].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("SYMGEN FULL MANIFEST: FROZEN")


if __name__ == "__main__":
    main()
