from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
from pathlib import Path

from common import dump_json, jsonl_rows, load_json, sha256_file, sha256_json, write_jsonl
from model import CleanJointRanker
from preflight import run_preflight
from protocol import (
    SplitFirewall, adapt_core_record, candidate_order_hash,
    load_llm_contract, load_train_dev_without_deserializing_sealed,
    norm_entry, position_sanity_baselines,
)


QUERY_ID_PATTERN = re.compile(r'"query_id"\s*:\s*"([^"\\]+)"')


def _balanced_api_order(rows, seed, embedding_index):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["global_split"], len(row["candidate_ids"]))].append(row)
    for key, values in grouped.items():
        values.sort(key=lambda row: sha256_json([seed, key, row["record_id"]]))
        count = key[1]
        for offset, row in enumerate(values):
            gold = row["gold_candidate_id"]
            others = sorted((value for value in row["candidate_ids"] if value != gold),
                            key=lambda value: sha256_json([seed, row["record_id"], value]))
            target = offset % count
            others.insert(target, gold)
            row["candidate_ids"] = others
            row["gold_position"] = target
            row["symbolic_candidate_indices"] = [int(embedding_index[value]) for value in others]
            row["candidate_order_protocol"] = {
                "name": "v0_2_9_APISem_balanced_stratum_cycle",
                "seed": int(seed),
                "stratum": list(key),
            }
    return rows


def build_apisem(cfg, embedding_index, llm_source):
    report = load_json(cfg["apisem_split_report"])
    role_by_key = {(str(row["sample_id"]), str(row["query_id"])): str(row["global_split"])
                   for row in report["records"]}
    role_by_query = {}
    for (sample_id, query_id), role in role_by_key.items():
        previous = role_by_query.setdefault(query_id, (sample_id, role))
        if previous != (sample_id, role):
            raise RuntimeError("APISem query_id is not globally unique: %s" % query_id)
    gold_by_key = {}
    sealed_gold_deserialized = 0
    with Path(cfg["strict_gold"]).open("r", encoding="utf-8") as handle:
        for line in handle:
            match = QUERY_ID_PATTERN.search(line)
            if not match or match.group(1) not in role_by_query:
                continue
            expected_sample, role = role_by_query[match.group(1)]
            if role not in {"TRAIN", "DEV"}:
                continue
            row = json.loads(line)
            if str(row.get("sample_id")) != expected_sample:
                raise RuntimeError("APISem Gold sample/query identity mismatch")
            if (row.get("target_semantic_type") or row.get("semantic_type")) != "APISem":
                continue
            target = row.get("target_id") or row.get("gold_target_id") or row.get("gold_id")
            if target:
                gold_by_key[(str(row["sample_id"]), str(row["query_id"]))] = str(target)
    source_by_function = {}
    for value in llm_source.values():
        source_by_function.setdefault((str(value["sample_id"]), norm_entry(value["function_entry"])), value)
    task_code_cache = {}

    def task_code_function(sample_id, function_entry):
        cache_key = (sample_id, norm_entry(function_entry))
        if cache_key in task_code_cache:
            return task_code_cache[cache_key]
        reference_path = Path(cfg["task1_samples"]) / sample_id / "code_reference.json"
        reference = load_json(reference_path)
        code_path = Path(reference["path"])
        if sha256_file(code_path) != reference["sha256"]:
            raise RuntimeError("APISem code_reference SHA mismatch: %s" % sample_id)
        code = load_json(code_path)
        if str(code.get("sample_id")) != sample_id:
            raise RuntimeError("APISem code_reference sample mismatch")
        for function in code.get("functions", []):
            if norm_entry(function.get("function_entry")) != cache_key[1]:
                continue
            if function.get("status") != "OK":
                raise RuntimeError("APISem code function is not OK: %r" % (cache_key,))
            value = {
                "sample_id": sample_id,
                "function_entry": function["function_entry"],
                "pseudo_c": "\n".join(str(line.get("text", "")) for line in function["lines"]),
                "source_contract": "seraph_task1_audited_code_reference",
                "code_reference_sha256": reference["sha256"],
            }
            task_code_cache[cache_key] = value
            return value
        raise RuntimeError("APISem function missing from audited code_reference: %r" % (cache_key,))
    records = []
    sealed_query_deserialized = 0
    for path in sorted(glob.glob(str(Path(cfg["task1_samples"]) / "*" / "queries.jsonl"))):
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                match = QUERY_ID_PATTERN.search(line)
                if not match or match.group(1) not in role_by_query:
                    continue
                expected_sample, role = role_by_query[match.group(1)]
                if role not in {"TRAIN", "DEV"}:
                    continue
                query = json.loads(line)
                if str(query.get("sample_id")) != expected_sample:
                    raise RuntimeError("APISem query sample/query identity mismatch")
                if query.get("semantic_type") != "APISem":
                    continue
                key = (str(query["sample_id"]), str(query["query_id"]))
                if role_by_key.get(key) != role:
                    raise RuntimeError("APISem split identity mismatch")
                gold = gold_by_key.get(key)
                candidates = [str(item["target_id"]) for item in query.get("candidates", []) if item.get("target_id")]
                function_key = (key[0], norm_entry(query.get("function_entry")))
                source = source_by_function.get(function_key)
                if source is None:
                    source = task_code_function(key[0], query.get("function_entry"))
                if not gold or gold not in candidates:
                    raise RuntimeError("APISem strict record lacks Gold/candidate/function context: %r" % (key,))
                missing = [value for value in [key[1]] + candidates if value not in embedding_index]
                if missing:
                    raise RuntimeError("APISem clean embedding coverage mismatch: %s" % missing[:3])
                records.append({
                    "record_id": sha256_json(["APISem", key[0], key[1]]),
                    "source_manifest_key": None,
                    "global_split": role,
                    "module": "APISem",
                    "sample_id": key[0],
                    "query_id": key[1],
                    "candidate_ids": candidates,
                    "gold_candidate_id": gold,
                    "gold_position": candidates.index(gold),
                    "symbolic_query_index": int(embedding_index[key[1]]),
                    "symbolic_candidate_indices": [int(embedding_index[value]) for value in candidates],
                    "pseudo_c": source["pseudo_c"],
                    "function_entry": source["function_entry"],
                    "instruction_mapping": None,
                    "llm_source_contract": source.get("source_contract", "v0_2_4_audited_source_manifest"),
                    "code_reference_sha256": source.get("code_reference_sha256"),
                })
    records = _balanced_api_order(records, cfg["fixed_seeds"][0], embedding_index)
    counts = collections.Counter(row["global_split"] for row in records)
    expected = cfg["apisem_learned"]
    if counts["TRAIN"] != expected["train_expected"] or counts["DEV"] != expected["dev_expected"]:
        raise RuntimeError("APISem TRAIN/DEV count mismatch: %s" % dict(counts))
    return records, {
        "schema": "seraph_v0_2_9_APISem_sealed_read_audit",
        "sealed_gold_rows_deserialized": sealed_gold_deserialized,
        "sealed_query_rows_deserialized": sealed_query_deserialized,
        "calibration_evaluation_count": 0,
        "test_evaluation_count": 0,
        "task_code_reference_function_count": len(task_code_cache),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    if not cfg["execution"]["authorized_after_v028_handoff"]:
        raise SystemExit("STOP: explicit post-v0_2_8 user confirmation is required")
    preflight = run_preflight(cfg)
    if preflight["status"] != "PASS":
        raise SystemExit("STOP: v0_2_9 preflight failed")

    output = Path(cfg["output_root"])
    if output.exists():
        raise SystemExit("refusing to overwrite v0_2_9 output")
    pending = output.with_name(output.name + ".pending.%d" % os.getpid())
    if pending.exists():
        raise SystemExit("pending output already exists")
    index = load_json(cfg["clean_embedding_index"])
    core, sealing = load_train_dev_without_deserializing_sealed(cfg["global_manifest"])
    llm_source, mapping = load_llm_contract(
        cfg["llm"]["source_manifest"], cfg["llm"]["instruction_mapping"],
        {row["source_manifest_key"] for row in core}
    )
    records = [adapt_core_record(row, index, llm_source, mapping) for row in core]
    apisem_sealing = None
    if cfg["apisem_learned"]["enabled"]:
        apisem_records, apisem_sealing = build_apisem(cfg, index, llm_source)
        records.extend(apisem_records)
    firewall = SplitFirewall()
    train = firewall.training_rows([row for row in records if row["global_split"] == "TRAIN"])
    dev = firewall.development_rows([row for row in records if row["global_split"] == "DEV"])
    sealing.update(firewall.assert_sealed())

    order_hash = sha256_json([{
        "record_id": row["record_id"], "global_split": row["global_split"],
        "module": row["module"], "candidate_order": row["candidate_ids"],
        "gold": row["gold_candidate_id"],
    } for row in sorted(records, key=lambda value: value["record_id"])])
    if order_hash != cfg["expected_candidate_order_sha256"]:
        raise RuntimeError("candidate membership/order hash drift from v0_2_9 clean protocol")
    position = position_sanity_baselines(core)
    modules = list(cfg["enabled_modules"])
    if cfg["apisem_learned"]["enabled"]:
        modules.append("APISem")
    initialization = {}
    for arm in cfg["arms"]:
        for seed in cfg["fixed_seeds"]:
            model = CleanJointRanker(
                arm=arm, seed=seed, symbolic_dim=64, llm_dim=cfg["llm"]["hidden_dim"],
                bridge_dim=cfg["training_budget"]["bridge_dim"], modules=cfg["enabled_modules"],
                apisem_enabled=cfg["apisem_learned"]["enabled"],
                temperature=cfg["training_budget"]["candidate_temperature"],
            )
            initialization["%s/seed_%d" % (arm, seed)] = model.initialization_hashes()

    pending.mkdir(parents=True)
    write_jsonl(pending / "train_manifest.jsonl", train)
    write_jsonl(pending / "dev_manifest.jsonl", dev)
    dump_json(pending / "preflight_report.json", preflight)
    dump_json(pending / "candidate_position_sanity.json", position)
    dump_json(pending / "initialization_hashes.json", initialization)
    contract = {
        "schema": "seraph_clean_joint_contract_v0_2_10",
        "status": "READY_FOR_EXPLICITLY_AUTHORIZED_TRAINING",
        "manifest_sha256": sha256_file(cfg["global_manifest"]),
        "candidate_order_sha256": order_hash,
        "clean_embedding_sha256": preflight["clean_embedding_sha256"],
        "clean_stage1_adapter_sha256": cfg["llm"]["expected_stage1_adapter_sha256"],
        "clean_stage1_adapter_path": cfg["llm"]["stage1_adapter"],
        "clean_index_sha256": preflight["clean_index_sha256"],
        "clean_artifact_sha256": preflight["clean_artifact_sha256"],
        "training_config_sha256": sha256_file(args.config),
        "module_selection_rule": cfg["module_selection_rule"],
        "module_selection_code_sha256": sha256_file(Path(__file__).with_name("select_final.py")),
        "train_records": len(train),
        "dev_records": len(dev),
        "train_modules": dict(collections.Counter(row["module"] for row in train)),
        "dev_modules": dict(collections.Counter(row["module"] for row in dev)),
        "arms": list(cfg["arms"]),
        "seeds": cfg["fixed_seeds"],
        "training_budget": cfg["training_budget"],
        "initialization_hashes": initialization,
        "position_sanity": position,
        "split_sealing": sealing,
        "apisem_split_sealing": apisem_sealing,
        "constraint_loss": cfg["constraint_loss"],
        "apisem_learned": cfg["apisem_learned"],
        "compromised_test_v1_evaluation_count": 0,
        "final_test_v2_prediction_count": 0,
        "final_test_v2_evaluation_count": 0,
    }
    dump_json(pending / "v0_2_10_contract.json", contract)
    os.replace(pending, output)
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
