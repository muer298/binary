"""Data-only candidate construction with Gold physically separate from inputs.

Candidate membership is fixed by canonical semantic nodes and stable hashes
before the strict-Gold membership gate. No learned score is consulted.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

MODULES = ("TypeSem", "InterfaceSem", "LayoutSem", "APISem")
MEMBERSHIP_SEED = "SERAPH_FINAL_TEST_V2_MEMBERSHIP_20261003"
POSITION_SEED = "SERAPH_FINAL_TEST_V2_POSITION_20261003"
MAX_CANDIDATES = 4
MAX_QUERIES_PER_PROJECT_MODULE = 256


def rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip(): yield json.loads(line)


def dump(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as fh:
        json.dump(value, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")


def write_rows(path: Path, values: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as fh:
        for value in values:
            fh.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def stable(*values: object) -> str:
    raw = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def norm_entry(value: object) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("0x"): text = text[2:]
    return text.lstrip("0") or "0"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["output_root"])
    inputs = root / "final_test_v2_inputs"
    gold = root / "final_test_v2_gold_sealed"
    draft = inputs / "candidate_draft.jsonl"
    sealed = gold / "candidate_gold_draft.jsonl"
    audit_path = root / "CANDIDATE_CONSTRUCTION_AUDIT.json"
    for path in (draft, sealed, audit_path):
        if path.exists(): raise FileExistsError(path)
    if not (root / "CLEAN_HSYM_STAGE_STATUS.json").exists():
        raise RuntimeError("clean h_sym forward not completed")
    preflight = json.loads((root / "FINAL_TEST_V2_PROVENANCE_PREFLIGHT.json").read_text())
    if preflight["status"] != "IDENTITY_PREFLIGHT_PASSED":
        raise RuntimeError("provenance preflight not passed")
    old_src = Path("/data/chaoni/mjy/resym_exp/packages/seraph_local_joint_v0_2_7_global_split/src")
    sys.path.insert(0, str(old_src))
    from build_global_split import canonical_candidate

    project_by_sample = {item["binary_id"]: item["repo"] for item in cfg["candidates"]}
    embedding_index = json.loads((inputs / "clean/node_embedding_index_clean.json").read_text())
    repair = json.loads((root / "ORACLE_CACHE_FILENAME_REPAIR_V2.json").read_text())
    if repair.get("status") != "PASSED" or repair.get("original_gold_content_modified") is not False:
        raise RuntimeError("frozen oracle cache repair not verified")
    strict_gold = list(rows(gold / "mapped_strict_gold_v2.jsonl"))
    query_ids = {str(x["query_id"]) for x in strict_gold if x.get("target_semantic_type") in MODULES}
    query_nodes = {}
    targets = collections.defaultdict(list)
    for node in rows(inputs / "graphs/usg_nodes_raw.jsonl"):
        node_id = str(node.get("node_id") or "")
        if node_id in query_ids:
            if node_id in query_nodes: raise RuntimeError("duplicate query node")
            query_nodes[node_id] = node
        module = node.get("node_type")
        if module in MODULES:
            key = (str(node.get("sample_id")), norm_entry(node.get("function_entry_norm") or node.get("function_entry")), module)
            targets[key].append(node)
    if query_ids - set(query_nodes):
        raise RuntimeError(f"missing strict-Gold query nodes: {len(query_ids - set(query_nodes))}")

    canonical_by_group = {}
    semantic_duplicates = 0
    for key, values in targets.items():
        by_semantic = collections.defaultdict(list)
        for node in values:
            identity = canonical_candidate(node)
            by_semantic[identity["canonical_semantic_id"]].append((node, identity))
        semantic_duplicates += sum(max(0, len(v) - 1) for v in by_semantic.values())
        canonical_by_group[key] = [min(group, key=lambda x: str(x[0]["node_id"])) for group in by_semantic.values()]

    stats = collections.defaultdict(collections.Counter)
    per_project_module = collections.defaultdict(list)
    all_raw = collections.Counter()
    for row in strict_gold:
        module = row.get("target_semantic_type")
        if module not in MODULES: continue
        sid, qid, target = str(row.get("sample_id")), str(row.get("query_id")), str(row.get("target_id"))
        if sid not in project_by_sample or row.get("project") != project_by_sample[sid]:
            raise RuntimeError("strict Gold project/sample identity mismatch")
        project = project_by_sample[sid]
        stat_key = (project, module)
        all_raw[stat_key] += 1
        query = query_nodes[qid]
        function = norm_entry(query.get("function_entry_norm") or query.get("function_entry"))
        key = (sid, function, module)
        pool = canonical_by_group.get(key, ())
        if len(pool) < 2:
            stats[stat_key]["fewer_than_two_canonical_candidates"] += 1
            continue
        # Membership is fixed without Gold, clean-h_sym scores, Stage-2 scores,
        # or model confidence. Gold only gates validity after this selection.
        ranked = sorted(pool, key=lambda pair: stable(MEMBERSHIP_SEED, qid,
                      pair[1]["canonical_semantic_id"], str(pair[0]["node_id"])))
        selected = ranked[:MAX_CANDIDATES]
        ids = [str(pair[0]["node_id"]) for pair in selected]
        if target not in ids:
            stats[stat_key]["gold_not_in_frozen_candidate_membership"] += 1
            continue
        if qid not in embedding_index or any(candidate not in embedding_index for candidate in ids):
            stats[stat_key]["clean_hsym_coverage_missing"] += 1
            continue
        record_id = stable("final-test-v2", project, sid, module, qid, sorted(ids))
        candidate_map = {str(pair[0]["node_id"]): pair[1]["canonical_semantic_id"] for pair in selected}
        per_project_module[stat_key].append({
            "record_id": record_id, "project_id": project, "binary_id": sid,
            "sample_id": sid, "function_entry": function, "module": module,
            "query_id": qid, "query_node_type": query.get("node_type"),
            "query_attrs": query.get("attrs") or {}, "candidate_ids": ids,
            "candidate_canonical_by_instance": candidate_map,
            "gold_instance_id": target, "gold_source": row.get("gold_source"),
            "gold_strength": row.get("gold_strength"),
        })

    selected_records = []
    for key, records in sorted(per_project_module.items()):
        records.sort(key=lambda r: stable(MEMBERSHIP_SEED, "query_cap", r["record_id"]))
        selected_records.extend(records[:MAX_QUERIES_PER_PROJECT_MODULE])
        stats[key]["eligible_before_query_cap"] = len(records)
        stats[key]["fixed_cap_excluded"] = max(0, len(records) - MAX_QUERIES_PER_PROJECT_MODULE)

    strata = collections.defaultdict(list)
    for item in selected_records:
        strata[(item["module"], len(item["candidate_ids"]))].append(item)
    for key, records in strata.items():
        records.sort(key=lambda r: stable(POSITION_SEED, "position", key, r["record_id"]))
        for ordinal, item in enumerate(records):
            ids = item["candidate_ids"]
            gold_id = item["gold_instance_id"]
            others = sorted((x for x in ids if x != gold_id),
                            key=lambda x: stable(POSITION_SEED, "nongold", item["record_id"], x))
            desired = ordinal % len(ids)
            others.insert(desired, gold_id)
            item["candidate_order"] = others
            item["gold_position"] = desired

    input_rows, gold_rows = [], []
    for item in sorted(selected_records, key=lambda r: r["record_id"]):
        ordered = item["candidate_order"]
        input_rows.append({
            "schema": "seraph_final_test_v2_candidate_draft_v1",
            "record_id": item["record_id"], "project_id": item["project_id"],
            "binary_id": item["binary_id"], "sample_id": item["sample_id"],
            "function_identity": {"project_id": item["project_id"],
                                  "binary_id": item["binary_id"],
                                  "function_entry": item["function_entry"]},
            "module": item["module"], "query_identity": item["query_id"],
            "query_node_type": item["query_node_type"],
            "instruction_address_for_mapping_only": str(item["query_attrs"].get("address") or item["query_attrs"].get("block_entry") or ""),
            "candidate_semantic_identities": [item["candidate_canonical_by_instance"][x] for x in ordered],
            "candidate_instance_identities": ordered, "candidate_order": ordered,
            "candidate_count": len(ordered),
            "candidate_membership_sha256": stable(sorted(item["candidate_ids"])),
            "clean_hsym_rows": {"query": int(embedding_index[item["query_id"]]),
                                 "candidates": [int(embedding_index[x]) for x in ordered]},
            "mapping_level": "UNKNOWN_PENDING_VERIFIED_MAPPING",
        })
        gold_rows.append({
            "schema": "seraph_final_test_v2_gold_draft_v1",
            "record_id": item["record_id"], "project_id": item["project_id"],
            "binary_id": item["binary_id"], "sample_id": item["sample_id"],
            "module": item["module"], "query_identity": item["query_id"],
            "gold_candidate_instance_id": item["gold_instance_id"],
            "gold_semantic_id": item["candidate_canonical_by_instance"][item["gold_instance_id"]],
            "gold_position": item["gold_position"],
            "gold_source": item["gold_source"], "gold_strength": item["gold_strength"],
        })
    if not input_rows:
        raise RuntimeError("no valid Gold-in-candidate records")
    write_rows(draft, input_rows)
    write_rows(sealed, gold_rows)
    counts = {"%s/%s" % key: {**dict(stats[key]), "raw_strict_gold": all_raw[key],
                               "final_draft_records": sum(x["project_id"] == key[0] and x["module"] == key[1] for x in input_rows)}
              for key in sorted(all_raw)}
    dump(audit_path, {
        "schema": "seraph_final_test_v2_candidate_construction_audit_v1",
        "status": "PASSED", "candidate_membership_gold_dependent": False,
        "candidate_membership_model_score_dependent": False,
        "gold_used_only_for_membership_legality_and_balanced_position": True,
        "canonical_candidate_contract": str(old_src / "build_global_split.py"),
        "max_candidates": MAX_CANDIDATES,
        "max_queries_per_project_module": MAX_QUERIES_PER_PROJECT_MODULE,
        "semantic_duplicate_instance_count": semantic_duplicates,
        "by_project_module": counts,
        "total_draft_records": len(input_rows),
        "candidate_count_distribution": dict(collections.Counter(x["candidate_count"] for x in input_rows)),
        "test_prediction_count": 0, "performance_evaluation_count": 0,
    })
    print(json.dumps({"draft_records": len(input_rows), "by_project_module": counts}, sort_keys=True))


if __name__ == "__main__": main()
