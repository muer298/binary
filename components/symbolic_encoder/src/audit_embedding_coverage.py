#!/usr/bin/env python3
import argparse
import collections
import glob
import json
from pathlib import Path

from io_utils import dump, load_config, rows


ROLES = ("TRAIN", "DEV", "CALIBRATION", "TEST")
MODULES = ("TypeSem", "InterfaceSem", "LayoutSem", "APISem")


def new_bucket():
    return {key: 0 for key in (
        "records", "query_total", "query_present", "candidate_total", "candidate_present",
        "gold_total", "gold_present")}


def add_record(bucket, query, candidates, gold, index, missing, role, module):
    bucket["records"] += 1
    bucket["query_total"] += 1
    bucket["query_present"] += int(query in index)
    if query not in index:
        missing.append({"role": role, "module": module, "kind": "query", "node_id": query,
                        "node_type": query.split("::")[-2] if "::" in query else "unknown",
                        "reason": "not_in_clean_featurizer_eligibility"})
    for candidate in candidates:
        bucket["candidate_total"] += 1
        bucket["candidate_present"] += int(candidate in index)
        if candidate not in index:
            missing.append({"role": role, "module": module, "kind": "candidate", "node_id": candidate,
                            "node_type": candidate.split("::")[-2] if "::" in candidate else "unknown",
                            "reason": "not_in_clean_featurizer_eligibility"})
    if gold:
        bucket["gold_total"] += 1
        bucket["gold_present"] += int(gold in index)
        if gold not in index:
            missing.append({"role": role, "module": module, "kind": "gold", "node_id": gold,
                            "node_type": gold.split("::")[-2] if "::" in gold else "unknown",
                            "reason": "not_in_clean_featurizer_eligibility"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_root"])
    index = json.loads(Path(args.index).read_text())
    stats = {(role, module): new_bucket() for role in ROLES for module in MODULES}
    missing = []

    # TypeSem/InterfaceSem/LayoutSem are fully represented by the frozen global manifest.
    for record in rows(cfg["global_manifest"]):
        role, module = record["global_split"], record["module"]
        add_record(stats[(role, module)], str(record["query_id"]),
                   [str(value) for value in record["candidate_instance_ids"]],
                   str(record["gold_candidate_instance_id"]), index, missing, role, module)

    # APISem strict Gold remains diagnostic-only.  Join task queries to strict Gold,
    # then check identities only; no model scores or performance metrics are computed.
    gold_by_key = {}
    for gold in rows(cfg["strict_gold"]):
        module = gold.get("target_semantic_type") or gold.get("semantic_type")
        if module != "APISem":
            continue
        key = (str(gold.get("sample_id")), str(gold.get("query_id")))
        target = gold.get("target_id") or gold.get("gold_target_id") or gold.get("gold_id")
        if target:
            gold_by_key[key] = str(target)
    split_report = json.loads(Path(cfg["apisem_split_report"]).read_text())
    role_by_query = {(str(row["sample_id"]), str(row["query_id"])): str(row["global_split"])
                     for row in split_report["records"]}
    api_records = 0
    for path in sorted(glob.glob(str(Path(cfg["task1_samples"]) / "*" / "queries.jsonl"))):
        for query in rows(path):
            if query.get("semantic_type") != "APISem":
                continue
            key = (str(query.get("sample_id")), str(query.get("query_id")))
            gold = gold_by_key.get(key)
            if not gold or key not in role_by_query:
                continue
            candidates = [str(value.get("target_id")) for value in query.get("candidates", [])
                          if isinstance(value, dict) and value.get("target_id")]
            role = role_by_query[key]
            add_record(stats[(role, "APISem")], key[1], candidates, gold, index, missing, role, "APISem")
            api_records += 1
    if api_records != 80:
        raise RuntimeError("expected 80 strict APISem rows, got %d" % api_records)

    by_role_module = {}
    gate_failures = []
    for role in ROLES:
        by_role_module[role] = {}
        for module in MODULES:
            bucket = stats[(role, module)]
            for prefix in ("query", "candidate", "gold"):
                total = bucket[prefix + "_total"]
                present = bucket[prefix + "_present"]
                bucket[prefix + "_coverage"] = present / total if total else None
                if total and present != total:
                    gate_failures.append("%s/%s/%s" % (role, module, prefix))
            by_role_module[role][module] = bucket

    missing_summary = collections.Counter(
        (row["role"], row["module"], row["kind"], row["node_type"], row["reason"])
        for row in missing
    )
    report = {
        "schema": "seraph_clean_embedding_coverage_v0_2_8",
        "identity_index_existence_only": True,
        "performance_metrics_computed": False,
        "by_role_module": by_role_module,
        "missing_count": len(missing),
        "missing_summary": [{
            "role": key[0], "module": key[1], "kind": key[2], "node_type": key[3],
            "reason": key[4], "count": value,
        } for key, value in sorted(missing_summary.items())],
        "missing_examples": missing[:200],
        "coverage_gate_failures": gate_failures,
        "coverage_gate_policy": "100_percent_query_candidate_gold_for_all_nonempty_role_module_cells",
        "passed": not gate_failures,
    }
    dump(out / "clean_embedding_coverage.json", report)
    lines = ["# Clean h_sym coverage report", "",
             "This is an identity/index existence audit only. No DEV/CALIBRATION/TEST score, prediction, loss, or performance metric was computed.", "",
             "| Role | Module | Records | Query | Candidates | Gold |", "|---|---|---:|---:|---:|---:|"]
    for role in ROLES:
        for module in MODULES:
            b = by_role_module[role][module]
            def fmt(prefix):
                return "%d/%d" % (b[prefix + "_present"], b[prefix + "_total"])
            lines.append("| %s | %s | %d | %s | %s | %s |" %
                         (role, module, b["records"], fmt("query"), fmt("candidate"), fmt("gold")))
    lines.extend(["", "Coverage gate: **%s**" % ("PASSED" if report["passed"] else "FAILED"),
                  "", "Missing identities: %d" % len(missing)])
    (out / "CLEAN_HSYM_COVERAGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
