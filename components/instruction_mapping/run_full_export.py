from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from common import dump_json, norm_entry, rows, sha256_file


def selected_binaries(path, count):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records or not {"sample_id", "stripped_path"}.issubset(records[0]):
        raise ValueError("selection CSV schema mismatch")
    if count != len(records):
        raise ValueError("v0.2.2 mapping requires the complete frozen selection: expected %d, found %d" % (count, len(records)))
    return records


def function_entries(task1_dir, sample_id):
    query_path = Path(task1_dir) / "samples" / sample_id / "queries.jsonl"
    entries = sorted(
        {norm_entry(query.get("function_entry")) for query in rows(query_path)},
        key=lambda value: int(value, 16),
    )
    return ["0x" + entry for entry in entries]


def validate_export(path, sample_id, entries):
    export = json.loads(Path(path).read_text(encoding="utf-8"))
    if export.get("schema") != "seraph_code_mapping_export_v0_2_2":
        raise ValueError("unexpected exporter schema")
    if export.get("sample_id") != sample_id or export.get("requested_entries") != entries:
        raise ValueError("export scope mismatch")
    if export.get("completed") is not True or len(export.get("functions", [])) != len(entries):
        raise ValueError("incomplete export")
    counters = Counter()
    node_classes = Counter()
    for function in export["functions"]:
        counters["functions"] += 1
        counters["functions_ok"] += int(function.get("status") == "OK")
        if function.get("status") != "OK":
            continue
        rebuilt = "\n".join(line["text"] for line in function["lines"])
        if rebuilt != function["function_text"]:
            raise ValueError("function text mismatch")
        known_nodes = {node["node_id"] for node in function["syntax_nodes"]}
        listing_addresses = {row["address"] for row in function["address_catalog"]}
        counters["listing_instructions"] += len(listing_addresses)
        for node in function["syntax_nodes"]:
            node_classes[node["class"]] += 1
            counters["syntax_nodes"] += 1
            if node["min_address"] is not None and node["max_address"] is not None:
                counters["syntax_nodes_with_address_range"] += 1
            if node["direct_pcode_addresses"]:
                counters["syntax_nodes_with_exact_pc_set"] += 1
        for line in function["lines"]:
            if rebuilt[line["char_start"]:line["char_end"]] != line["text"]:
                raise ValueError("line character span mismatch")
            for token in line["tokens"]:
                counters["tokens"] += 1
                if rebuilt[token["char_start"]:token["char_end"]] != token["text"]:
                    raise ValueError("token character span mismatch")
                if any(node_id not in known_nodes for node_id in token["syntax_node_ids"]):
                    raise ValueError("token references unknown syntax node")
                counters["tokens_with_direct_anchor"] += int(bool(token["anchors"]))
                counters["tokens_with_address_range"] += int(
                    token["token_min_address"] is not None and token["token_max_address"] is not None
                )
    return {"counts": dict(counters), "syntax_node_classes": dict(node_classes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    package_root = Path(__file__).resolve().parents[1]
    output_root = Path(cfg["output_root"])
    export_root = output_root / cfg.get("export_run_dir", "ghidra_mapping_exports")
    report_path = output_root / "ghidra_syntax_ancestry_export_report.json"
    if report_path.exists() or export_root.exists():
        raise SystemExit("Refusing to overwrite v0.2.2 mapping exports")
    export_root.mkdir(parents=True)
    script = package_root / "ghidra_scripts" / "ExportSeraphMappingV022.py"
    headless = Path(cfg["analyze_headless"])
    java_home = Path(cfg["java_home"])
    if not (java_home / "bin" / "java").is_file():
        raise FileNotFoundError(java_home / "bin" / "java")
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    environment["JAVA_HOME"] = str(java_home)
    environment["PATH"] = str(java_home / "bin") + os.pathsep + environment.get("PATH", "")

    samples = []
    aggregate = Counter()
    classes = Counter()
    for index, record in enumerate(selected_binaries(cfg["selection_pairs_csv"], int(cfg["sample_count"])), 1):
        sample_id = record["sample_id"]
        binary = Path(record["stripped_path"])
        if not binary.is_file():
            raise FileNotFoundError(binary)
        sample_root = export_root / sample_id
        sample_root.mkdir()
        entries = function_entries(cfg["task1_dir"], sample_id)
        entry_file = sample_root / "entries.txt"
        entry_file.write_text("\n".join(entries) + "\n", encoding="ascii")
        output = sample_root / "code_mapping_v0_2_2.json"
        log = sample_root / "ghidra.log"
        project = sample_root / "project"
        project.mkdir()
        command = [
            str(headless), str(project), "seraph_v022_" + sample_id[:12],
            "-import", str(binary), "-scriptPath", str(script.parent),
            "-postScript", script.name, str(output), sample_id, str(entry_file),
            str(cfg["decompile_timeout_seconds"]), str(cfg["max_syntax_ancestor_depth"]),
            "-analysisTimeoutPerFile", "600", "-max-cpu", "2", "-deleteProject",
        ]
        with log.open("w") as handle:
            completed = subprocess.run(
                command, stdin=subprocess.DEVNULL, stdout=handle,
                stderr=subprocess.STDOUT, env=environment, check=False,
            )
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError("Ghidra export failed for sample %s: %s" % (sample_id, log))
        log_tail = log.read_text(errors="replace")[-200000:]
        if "SERAPH_MAPPING_EXPORT_V022_DONE %s" % sample_id not in log_tail:
            raise RuntimeError("export completion marker missing: " + str(log))
        audit = validate_export(output, sample_id, entries)
        aggregate.update(audit["counts"])
        classes.update(audit["syntax_node_classes"])
        samples.append({
            "index": index, "sample_id": sample_id,
            "binary": str(binary), "binary_sha256": sha256_file(binary),
            "function_entries": len(entries), "output": str(output),
            "output_sha256": sha256_file(output), "log": str(log),
            "audit": audit,
        })
        print(json.dumps({"EXPORT_PROGRESS": "%d/%d" % (index, len(selected_binaries(cfg["selection_pairs_csv"], int(cfg["sample_count"])))), "sample_id": sample_id, "functions": len(entries), "audit": audit["counts"]}), flush=True)

    report = {
        "schema": "seraph_ghidra_syntax_ancestry_export_report_v0_2_2",
        "status": "PASSED",
        "sample_count": len(samples),
        "script": str(script), "script_sha256": sha256_file(script),
        "java_home": str(java_home),
        "mapping_evidence_only": [
            "ClangToken.getPcodeOp().getSeqnum().getTarget()",
            "ClangToken/ClangNode.getMinAddress()/getMaxAddress()",
            "ClangNode.Parent() ancestry",
            "Varnode.getAddress()/getPCAddress() when exposed",
            "program listing exact instruction address",
        ],
        "forbidden_fallbacks": ["nearest address", "nearest line", "string similarity", "token order guess"],
        "aggregate": dict(aggregate),
        "syntax_node_classes": dict(classes),
        "samples": samples,
    }
    dump_json(report_path, report)
    print(json.dumps({"EXPORT_STATUS": "PASSED", "aggregate": dict(aggregate), "syntax_node_classes": dict(classes)}, indent=2))


if __name__ == "__main__":
    main()
