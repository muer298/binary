"""Prepare provenance-fixed stripped inputs and isolated DWARF oracles.

No Stage-1/Stage-2 model is imported or loaded. Gold oracles never enter the
input tree. Every selected raw binary is verified against its SHA256 name.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def sections(path: Path) -> str:
    return subprocess.run(["readelf", "-SW", str(path)], check=True,
                          capture_output=True, text=True).stdout


def git(path: Path, rev: str) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", rev],
                          check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg["output_root"])
    preflight = json.loads((out / "FINAL_TEST_V2_PROVENANCE_PREFLIGHT.json").read_text())
    if preflight["status"] != "IDENTITY_PREFLIGHT_PASSED":
        raise RuntimeError("identity preflight failed")
    if len(preflight["candidates"]) != len(cfg["candidates"]):
        raise RuntimeError("preflight candidate count mismatch")
    if any(cfg[key] for key in (
        "final_stage2_checkpoint_load_allowed", "stage1_training_allowed",
        "hsym_training_allowed", "stage2_training_allowed", "ssq_allowed",
        "calibration_allowed", "test_prediction_allowed", "performance_evaluation_allowed",
    )):
        raise RuntimeError("forbidden operation enabled")

    inputs = out / "final_test_v2_inputs"
    gold = out / "final_test_v2_gold_sealed"
    if inputs.exists() or gold.exists():
        raise FileExistsError("refusing to overwrite input or Gold tree")
    (inputs / "pairs/stripped").mkdir(parents=True)
    (gold / "oracles").mkdir(parents=True)
    sys.path.insert(0, str(Path(cfg["research2_package"]) / "symgen_external"))
    from symgen_dwarf_oracle_v1 import ORACLE_SCHEMA, extract_oracle

    full_manifest: list[dict] = []
    inference_manifest: list[dict] = []
    pair_manifest: list[dict] = []
    split_manifest: list[dict] = []
    provenance: list[dict] = []
    for order, item in enumerate(cfg["candidates"], start=1):
        repo = item["repo"]
        sample_id = item["binary_id"]
        if preflight["candidates"][order-1]["repo"] != repo or preflight["candidates"][order-1]["identity_status"] != "PASSED":
            raise RuntimeError("candidate/preflight mismatch")
        raw = Path(cfg["raw_root"]) / "bin" / sample_id
        if not raw.is_file() or sha(raw) != sample_id:
            raise RuntimeError("raw binary identity mismatch: " + sample_id)
        sec = sections(raw)
        if ".debug_info" not in sec or ".debug_line" not in sec or ".symtab" not in sec:
            raise RuntimeError("raw binary lacks required DWARF/symtab")
        safe = repo.replace("/", "__")
        snapshot = out / "source_snapshots" / safe
        if git(snapshot, "HEAD") != item["source_commit"] or git(snapshot, "HEAD^{tree}") != item["source_tree"]:
            raise RuntimeError("source snapshot commit/tree mismatch: " + repo)
        source_files = [p for p in snapshot.rglob("*") if p.is_file() and p.suffix.lower() in {".c", ".cc", ".cpp", ".cxx"}]
        if not source_files:
            raise RuntimeError("source snapshot has no C/C++: " + repo)
        stripped = inputs / "pairs/stripped" / (sample_id + "__" + safe)
        shutil.copy2(raw, stripped)
        subprocess.run(["objcopy", "--strip-all", str(stripped)], check=True)
        stripped_sections = sections(stripped)
        if ".debug_info" in stripped_sections or ".symtab" in stripped_sections:
            raise RuntimeError("stripping failed")
        oracle = extract_oracle(str(raw), objdump_path="objdump")
        if oracle.get("schema") != ORACLE_SCHEMA or not oracle.get("has_dwarf"):
            raise RuntimeError("DWARF oracle extraction failed")
        oracle_path = gold / "oracles" / (sample_id + ".oracle.json")
        dump(oracle_path, oracle)
        base = {"sample_id": sample_id, "project": repo,
                "architecture": "x86_64", "optimization": "unknown_frozen_original"}
        full_manifest.append({**base, "full_path": str(raw), "full_sha256": sample_id,
                              "has_debug_info": True, "has_symtab": True})
        inference_manifest.append({**base, "elf_class": 64, "stripped_path": str(stripped),
                                   "stripped_sha256": sha(stripped), "status": "ok",
                                   "stripped_path_role": "inference_only", "pair_order": order})
        pair_manifest.append({"sample_id": sample_id, "project": repo,
                              "full_path": str(raw), "full_sha256": sample_id,
                              "stripped_path": str(stripped), "stripped_sha256": sha(stripped),
                              "status": "ok"})
        split_manifest.append({**base, "group_id": repo + "::" + sample_id,
                               "split": "final_test_v2_gold_build_only"})
        provenance.append({**base, "binary_id": sample_id, "full_sha256": sample_id,
                           "stripped_sha256": sha(stripped), "source_url": item["source_url"],
                           "source_commit": item["source_commit"], "source_tree": item["source_tree"],
                           "source_snapshot_time_utc": cfg["source_snapshot_time_utc"],
                           "source_file_count": len(source_files),
                           "archive_sha256": sha(out / "source_snapshots" / (safe + ".tar.gz")),
                           "oracle_sha256": sha(oracle_path),
                           "source_commit_matches_binary_build": "UNVERIFIED_RESYM_METADATA_DOES_NOT_PIN_COMMIT"})
        print(json.dumps({"prepared": repo, "binary_id": sample_id,
                          "oracle_functions": len(oracle.get("functions", [])),
                          "source_files": len(source_files)}), flush=True)
    write_csv(gold / "full_manifest.csv", full_manifest)
    write_csv(inputs / "inference_manifest.csv", inference_manifest)
    write_csv(gold / "pair_manifest.csv", pair_manifest)
    with (gold / "split_manifest.jsonl").open("x", encoding="utf-8") as fh:
        for item in split_manifest:
            fh.write(json.dumps(item, sort_keys=True) + "\n")
    dump(out / "SOURCE_AND_BINARY_PROVENANCE.json", {
        "schema": "seraph_final_test_v2_source_binary_provenance_v1",
        "records": provenance, "projects": len(provenance), "binaries": len(provenance),
        "source_commit_binary_build_link": "UNVERIFIED_FOR_ALL",
        "test_prediction_count": 0, "performance_evaluation_count": 0,
    })


if __name__ == "__main__":
    main()
