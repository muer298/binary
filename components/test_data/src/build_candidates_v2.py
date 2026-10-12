"""Gold-blind, data-only candidate construction with cross-function templates.

Repair of v1's single-canonical-candidate contract failure for TypeSem and
InterfaceSem. For a query with one local semantic candidate, two distinct
canonical negatives are selected from other functions of the same binary by
a predeclared stable hash. Gold is used only after frozen membership to check
legality and balance answer positions; model scores are never consulted.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from build_candidates import dump, norm_entry, rows, stable, write_rows

MODULES = ("TypeSem", "InterfaceSem", "LayoutSem", "APISem")
MEMBERSHIP_SEED = "SERAPH_FINAL_TEST_V2_MEMBERSHIP_20261003"
POSITION_SEED = "SERAPH_FINAL_TEST_V2_POSITION_20261003"
MAX_CANDIDATES = 4
MIN_CANDIDATES = 2
OTHER_FUNCTION_TEMPLATES_IF_SINGLE_LOCAL = 2
MAX_QUERIES_PER_PROJECT_MODULE = 256


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = json.loads(Path(ap.parse_args().config).read_text())
    root = Path(cfg["output_root"])
    inputs = root / "final_test_v2_inputs"
    gold = root / "final_test_v2_gold_sealed"
    draft = inputs / "candidate_draft_v2.jsonl"
    sealed = gold / "candidate_gold_draft_v2.jsonl"
    audit_path = root / "CANDIDATE_CONSTRUCTION_AUDIT_V2.json"
    for path in (draft, sealed, audit_path):
        if path.exists(): raise FileExistsError(path)
    if json.loads((root / "CLEAN_HSYM_STAGE_STATUS.json").read_text()).get("status") != "PASSED":
        raise RuntimeError("clean h_sym forward not passed")
    if json.loads((root / "ORACLE_CACHE_FILENAME_REPAIR_V2.json").read_text()).get("status") != "PASSED":
        raise RuntimeError("frozen oracle cache repair not passed")
    v1 = json.loads((root / "CANDIDATE_CONSTRUCTION_AUDIT.json").read_text())
    if v1.get("status") != "PASSED" or v1.get("by_project_module", {}).get("Tarsnap/spiped/TypeSem", {}).get("final_draft_records") != 0:
        raise RuntimeError("v1 single-candidate contract defect not established")

    old_src = Path("/data/chaoni/mjy/resym_exp/packages/seraph_local_joint_v0_2_7_global_split/src")
    sys.path.insert(0, str(old_src))
    from build_global_split import canonical_candidate
    project_by_sample = {item["binary_id"]: item["repo"] for item in cfg["candidates"]}
    index = json.loads((inputs / "clean/node_embedding_index_clean.json").read_text())
    gold_rows = list(rows(gold / "mapped_strict_gold_v2.jsonl"))
    query_ids = {str(item["query_id"]) for item in gold_rows if item.get("target_semantic_type") in MODULES}
    queries = {}
    by_function = collections.defaultdict(lambda: collections.defaultdict(list))
    global_pool = collections.defaultdict(lambda: collections.defaultdict(list))
    for node in rows(inputs / "graphs/usg_nodes_raw.jsonl"):
        node_id = str(node.get("node_id") or "")
        if node_id in query_ids:
            if node_id in queries: raise RuntimeError("duplicate query node")
            queries[node_id] = node
        module = node.get("node_type")
        if module not in MODULES: continue
        sid = str(node.get("sample_id"))
        entry = norm_entry(node.get("function_entry_norm") or node.get("function_entry"))
        canon = canonical_candidate(node)["canonical_semantic_id"]
        by_function[(sid, entry, module)][canon].append(node)
        global_pool[(sid, module)][canon].append(node)
    if query_ids - set(queries):
        raise RuntimeError("strict Gold query nodes missing")

    # Canonical representative choice is Gold-independent and deterministic.
    local = {key: [(min(nodes, key=lambda x: str(x["node_id"])), canon)
                   for canon, nodes in sem.items()] for key, sem in by_function.items()}
    global_reps = {key: [(min(nodes, key=lambda x: str(x["node_id"])), canon)
                         for canon, nodes in sem.items()] for key, sem in global_pool.items()}
    stats = collections.defaultdict(collections.Counter)
    eligible = collections.defaultdict(list)
    all_raw = collections.Counter()
    for gold_item in gold_rows:
        module = gold_item.get("target_semantic_type")
        if module not in MODULES: continue
        sid, qid, target = str(gold_item["sample_id"]), str(gold_item["query_id"]), str(gold_item["target_id"])
        project = project_by_sample.get(sid)
        if not project or gold_item.get("project") != project:
            raise RuntimeError("Gold project/sample identity mismatch")
        key = (project, module)
        all_raw[key] += 1
        query = queries[qid]
        entry = norm_entry(query.get("function_entry_norm") or query.get("function_entry"))
        local_pool = local.get((sid, entry, module), [])
        ranked_local = sorted(local_pool, key=lambda pair: stable(MEMBERSHIP_SEED, qid, pair[1], pair[0]["node_id"]))
        selected = ranked_local[:MAX_CANDIDATES]
        scope = {str(node["node_id"]): "same_function" for node, _ in selected}
        if len(selected) < MIN_CANDIDATES:
            local_sem = {canon for _, canon in local_pool}
            negatives = [(node, canon) for node, canon in global_reps.get((sid, module), [])
                         if canon not in local_sem and norm_entry(node.get("function_entry_norm") or node.get("function_entry")) != entry]
            ranked_negative = sorted(negatives, key=lambda pair: stable(MEMBERSHIP_SEED, "other_function_template", qid, pair[1], pair[0]["node_id"]))
            selected.extend(ranked_negative[:OTHER_FUNCTION_TEMPLATES_IF_SINGLE_LOCAL])
            scope.update({str(node["node_id"]): "other_function_template" for node, _ in selected if str(node["node_id"]) not in scope})
        ids = [str(node["node_id"]) for node, _ in selected]
        if len(ids) < MIN_CANDIDATES:
            stats[key]["insufficient_distinct_semantic_candidates"] += 1
            continue
        if target not in ids:
            stats[key]["gold_not_in_frozen_candidate_membership"] += 1
            continue
        if qid not in index or any(candidate not in index for candidate in ids):
            stats[key]["clean_hsym_coverage_missing"] += 1
            continue
        if len(ids) != len(set(ids)) or len({canon for _, canon in selected}) != len(ids):
            raise RuntimeError("candidate identity/semantic uniqueness failure")
        record_id = stable("final-test-v2-candidate-contract-v2", project, sid, module, qid, sorted(ids))
        eligible[key].append({
            "record_id": record_id, "project_id": project, "binary_id": sid, "sample_id": sid,
            "function_entry": entry, "module": module, "query_id": qid,
            "query_node_type": query.get("node_type"), "query_attrs": query.get("attrs") or {},
            "candidate_ids": ids, "candidate_canonical": {str(node["node_id"]): canon for node, canon in selected},
            "candidate_scope": scope, "gold_instance_id": target,
            "gold_source": gold_item.get("gold_source"), "gold_strength": gold_item.get("gold_strength")})

    selected_records = []
    for key, values in sorted(eligible.items()):
        values.sort(key=lambda row: stable(MEMBERSHIP_SEED, "query_cap", row["record_id"]))
        selected_records.extend(values[:MAX_QUERIES_PER_PROJECT_MODULE])
        stats[key]["eligible_before_query_cap"] = len(values)
        stats[key]["fixed_cap_excluded"] = max(0, len(values) - MAX_QUERIES_PER_PROJECT_MODULE)

    strata = collections.defaultdict(list)
    for item in selected_records:
        strata[(item["module"], len(item["candidate_ids"]))].append(item)
    for key, values in strata.items():
        values.sort(key=lambda row: stable(POSITION_SEED, "position", key, row["record_id"]))
        for ordinal, item in enumerate(values):
            ids, gold_id = item["candidate_ids"], item["gold_instance_id"]
            others = sorted((cid for cid in ids if cid != gold_id),
                            key=lambda cid: stable(POSITION_SEED, "nongold", item["record_id"], cid))
            target_pos = ordinal % len(ids)
            others.insert(target_pos, gold_id)
            item["candidate_order"] = others
            item["gold_position"] = target_pos

    input_rows, sealed_rows = [], []
    for item in sorted(selected_records, key=lambda row: row["record_id"]):
        order = item["candidate_order"]
        input_rows.append({
            "schema": "seraph_final_test_v2_candidate_draft_v2",
            "record_id": item["record_id"], "project_id": item["project_id"],
            "binary_id": item["binary_id"], "sample_id": item["sample_id"],
            "function_identity": {"project_id": item["project_id"],
                                  "binary_id": item["binary_id"],
                                  "function_entry": item["function_entry"]},
            "module": item["module"], "query_identity": item["query_id"],
            "query_node_type": item["query_node_type"],
            "instruction_address_for_mapping_only": str(item["query_attrs"].get("address") or item["query_attrs"].get("block_entry") or ""),
            "candidate_semantic_identities": [item["candidate_canonical"][cid] for cid in order],
            "candidate_instance_identities": order, "candidate_order": order,
            "candidate_scope": [item["candidate_scope"][cid] for cid in order],
            "candidate_count": len(order),
            "candidate_membership_sha256": stable(sorted(item["candidate_ids"])),
            "clean_hsym_rows": {"query": int(index[item["query_id"]]),
                                 "candidates": [int(index[cid]) for cid in order]},
            "mapping_level": "UNKNOWN_PENDING_VERIFIED_MAPPING"})
        sealed_rows.append({
            "schema": "seraph_final_test_v2_gold_draft_v2",
            "record_id": item["record_id"], "project_id": item["project_id"],
            "binary_id": item["binary_id"], "sample_id": item["sample_id"],
            "module": item["module"], "query_identity": item["query_id"],
            "gold_candidate_instance_id": item["gold_instance_id"],
            "gold_semantic_id": item["candidate_canonical"][item["gold_instance_id"]],
            "gold_position": item["gold_position"],
            "gold_source": item["gold_source"], "gold_strength": item["gold_strength"]})
    if not input_rows:
        raise RuntimeError("no valid Gold-in-candidate records")
    write_rows(draft, input_rows)
    write_rows(sealed, sealed_rows)
    counts = {"%s/%s" % key: {**dict(stats[key]), "raw_strict_gold": all_raw[key],
                               "final_draft_records": sum(row["project_id"] == key[0] and row["module"] == key[1] for row in input_rows)}
              for key in sorted(all_raw)}
    dump(audit_path, {
        "schema": "seraph_final_test_v2_candidate_construction_audit_v2",
        "status": "PASSED", "repair_reason": "v1 single-canonical-candidate groups had no valid negatives",
        "v1_draft_and_audit_preserved": True,
        "membership_policy": "same-function canonical candidates first; if fewer than two, two stable-hash other-function templates from same binary",
        "membership_uses_gold_identity": False, "membership_uses_model_score": False,
        "gold_used_only_for_legality_and_balanced_position": True,
        "other_function_templates_are_not_function_facts": True,
        "canonical_candidate_contract": str(old_src / "build_global_split.py"),
        "max_candidates": MAX_CANDIDATES,
        "max_queries_per_project_module": MAX_QUERIES_PER_PROJECT_MODULE,
        "by_project_module": counts, "total_draft_records": len(input_rows),
        "candidate_count_distribution": dict(collections.Counter(row["candidate_count"] for row in input_rows)),
        "test_prediction_count": 0, "performance_evaluation_count": 0})
    print(json.dumps({"draft_records": len(input_rows), "by_project_module": counts}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
