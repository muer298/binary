"""Export and verify stripped-binary instruction-to-pseudo-C mappings.

This is a data-build stage only. It never reads sealed Gold or loads Stage-2.
UNKNOWN is retained as UNKNOWN; no nearest-address or root-node fallback.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PYTHON = "/home/cn/miniconda3/envs/seraph_llm/bin/python"
MAPPING_SOURCE = Path("/data/chaoni/mjy/resym_exp/packages/seraph_local_joint_v0_2_2_mapping/src")
EXPORTER = Path("/data/chaoni/mjy/resym_exp/packages/seraph_local_joint_v0_2_2_mapping/ghidra_scripts/ExportSeraphMappingV022.py")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def export(cfg: dict) -> None:
    root = Path(cfg["output_root"])
    inputs = root / "final_test_v2_inputs"
    mapping = inputs / "instruction_mapping"
    if mapping.exists():
        raise FileExistsError(mapping)
    draft = inputs / "candidate_draft_v2.jsonl"
    if not draft.is_file():
        raise RuntimeError("candidate draft is not complete")
    by_sample = collections.defaultdict(set)
    for item in rows(draft):
        by_sample[item["sample_id"]].add(item["function_identity"]["function_entry"])
    if not by_sample:
        raise RuntimeError("candidate draft has no function entries")
    with (inputs / "inference_manifest.csv").open(newline="", encoding="utf-8") as handle:
        binaries = {row["sample_id"]: row for row in csv.DictReader(handle)}
    mapping.mkdir()
    # /data has little free space; keep Ghidra's transient project database on
    # the separate /tmp filesystem. Do not delete it without user approval.
    work_root = Path(tempfile.mkdtemp(prefix="seraph-final-test-v2-mapping-", dir="/tmp"))
    environment = dict(os.environ)
    environment["JAVA_HOME"] = cfg["java_home"]
    environment["PATH"] = str(Path(cfg["java_home"]) / "bin") + os.pathsep + environment.get("PATH", "")
    environment["LC_ALL"] = "C"
    analyze = Path(cfg["ghidra_home"]) / "support/analyzeHeadless"
    reports = []
    for sample_id in sorted(by_sample):
        if sample_id not in binaries:
            raise RuntimeError("candidate sample missing from inference manifest")
        sample_root = mapping / sample_id
        sample_root.mkdir()
        entries = sorted(by_sample[sample_id], key=lambda x: int(x, 16))
        entry_file = sample_root / "entries.txt"
        with entry_file.open("x", encoding="ascii") as handle:
            handle.write("\n".join(entries) + "\n")
        output = sample_root / "code_mapping_v0_2_2.json"
        log = sample_root / "ghidra.log"
        project = work_root / sample_id / "project"
        project.parent.mkdir()
        project.mkdir()
        binary = Path(binaries[sample_id]["stripped_path"])
        command = [str(analyze), str(project), "seraph_final_test_v2_map_" + sample_id[:12],
                   "-import", str(binary), "-scriptPath", str(EXPORTER.parent),
                   "-postScript", EXPORTER.name, str(output), sample_id, str(entry_file), "60", "16",
                   "-analysisTimeoutPerFile", "1200", "-max-cpu", "2", "-deleteProject"]
        print("MAPPING_EXPORT", sample_id, len(entries), flush=True)
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=handle,
                                       stderr=subprocess.STDOUT, env=environment, check=False)
        tail = log.read_text(encoding="utf-8", errors="replace")[-200000:]
        if completed.returncode or not output.is_file() or "SERAPH_MAPPING_EXPORT_V022_DONE %s" % sample_id not in tail:
            raise RuntimeError("Ghidra mapping export failed: " + str(log))
        payload = load(output)
        if payload.get("completed") is not True or payload.get("requested_entries") != entries:
            raise RuntimeError("Ghidra mapping export incomplete: " + sample_id)
        functions = payload.get("functions", [])
        if len(functions) != len(entries) or any(row.get("status") != "OK" for row in functions):
            raise RuntimeError("Ghidra mapping function export failed: " + sample_id)
        reports.append({"sample_id": sample_id, "stripped_binary": str(binary),
                        "stripped_sha256": sha(binary), "requested_functions": len(entries),
                        "output": str(output), "output_sha256": sha(output), "log": str(log),
                        "transient_project_root": str(project.parent)})
    dump(root / "INSTRUCTION_MAPPING_EXPORT_STATUS.json", {
        "status": "PASSED", "samples": reports, "exporter": str(EXPORTER),
        "transient_work_root": str(work_root),
        "exporter_sha256": sha(EXPORTER), "full_elf_opened": 0,
        "sealed_gold_opened": 0, "stage2_checkpoint_loaded": False,
        "test_prediction_count": 0, "performance_evaluation_count": 0})


def finalize(cfg: dict) -> None:
    root = Path(cfg["output_root"])
    inputs = root / "final_test_v2_inputs"
    draft = inputs / "candidate_draft_v2.jsonl"
    export_report = load(root / "INSTRUCTION_MAPPING_EXPORT_STATUS.json")
    if export_report.get("status") != "PASSED":
        raise RuntimeError("mapping export not passed")
    target = root / "FINAL_TEST_V2_MANIFEST.jsonl"
    report_path = root / "INSTRUCTION_MAPPING_BUILD_STATUS.json"
    if target.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite final manifest or mapping status")
    sys.path.insert(0, str(MAPPING_SOURCE))
    from audit_mapping import char_token_indices, classify
    from common import address_int, centered_window, norm_entry
    from transformers import AutoTokenizer

    functions = {}
    for sample in export_report["samples"]:
        payload = load(Path(sample["output"]))
        for function in payload["functions"]:
            key = (sample["sample_id"], norm_entry(function["function_entry"]))
            if key in functions:
                raise RuntimeError("duplicate mapping function")
            functions[key] = function
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"], local_files_only=True, use_fast=True)
    tokenized = {}
    output = []
    levels = collections.defaultdict(collections.Counter)
    reasons = collections.Counter()
    for item in rows(draft):
        module = item["module"]
        key = (item["sample_id"], norm_entry(item["function_identity"]["function_entry"]))
        function = functions.get(key)
        if function is None or function.get("status") != "OK":
            raise RuntimeError("missing exact function export: " + str(key))
        if key not in tokenized:
            encoded = tokenizer(function["function_text"], add_special_tokens=False,
                                return_offsets_mapping=True)
            tokenized[key] = {"ids": encoded["input_ids"], "offsets": encoded["offset_mapping"]}
        tok = tokenized[key]
        pc = address_int(item.get("instruction_address_for_mapping_only"))
        if pc is None:
            level, spans, evidence, reason = "L3_UNKNOWN", [], {"method": None}, "INSTRUCTION_PC_MISSING"
        else:
            level, spans, evidence, reason = classify(function, pc)
        indices = char_token_indices(tok["offsets"], spans) if spans else []
        window = centered_window(len(tok["ids"]), indices, 512) if indices else None
        if level != "L3_UNKNOWN" and window is None:
            level, spans, evidence, reason = "L3_UNKNOWN", [], {"method": None}, "VERIFIED_SPAN_TOKEN_WINDOW_UNUSABLE"
        if level == "L0_EXACT_TOKEN":
            bucket = "L0"
        elif level in ("L1_SYNTAX_EXACT_PC_SET", "L2_VERIFIED_NODE_RANGE"):
            bucket = "L2"
        else:
            bucket = "UNKNOWN"
            reasons[reason or "UNKNOWN"] += 1
        row = dict(item)
        row.pop("instruction_address_for_mapping_only", None)
        row["pseudo_c"] = function["function_text"]
        row["mapping_level"] = bucket
        row["instruction_mapping"] = {
            "raw_level": level, "level": bucket,
            "instruction_address": ("0x%x" % pc) if pc is not None else None,
            "mapping_status": "VERIFIED" if bucket != "UNKNOWN" else "UNKNOWN",
            "char_spans": spans, "token_indices": indices if bucket != "UNKNOWN" else [],
            "token_count": len(tok["ids"]), "window": window if bucket != "UNKNOWN" else None,
            "evidence": evidence, "unknown_reason": reason if bucket == "UNKNOWN" else None,
            "fallback_rule": "symbolic_only_if_unknown", "forbidden_fallbacks_used": [],
        }
        output.append(row)
        levels[module][bucket] += 1
    if not output:
        raise RuntimeError("final manifest would be empty")
    with target.open("x", encoding="utf-8") as handle:
        for row in sorted(output, key=lambda x: x["record_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    dump(report_path, {
        "status": "PASSED", "records": len(output),
        "level_counts_by_module": {k: dict(v) for k, v in sorted(levels.items())},
        "unknown_reasons": dict(reasons), "manifest_sha256": sha(target),
        "exact_token_policy": "L0_EXACT_TOKEN",
        "verified_syntax_policy": "L1_SYNTAX_EXACT_PC_SET and L2_VERIFIED_NODE_RANGE are reported in L2 bucket",
        "unknown_policy": "retained_as_UNKNOWN_no_nearest_root_or_string_fallback",
        "sealed_gold_opened": 0, "stage2_checkpoint_loaded": False,
        "test_prediction_count": 0, "performance_evaluation_count": 0})
    print(json.dumps({"records": len(output), "levels": {k: dict(v) for k, v in levels.items()},
                      "manifest_sha256": sha(target)}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("stage", choices=("export", "finalize"))
    args = parser.parse_args()
    cfg = load(Path(args.config))
    if args.stage == "export":
        export(cfg)
    else:
        finalize(cfg)


if __name__ == "__main__":
    main()
