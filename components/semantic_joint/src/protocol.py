from __future__ import annotations

import collections
import glob
import json
import re
from pathlib import Path

from common import jsonl_rows, sha256_json


ROLE_PATTERN = re.compile(r'"global_split"\s*:\s*"(TRAIN|DEV|CALIBRATION|TEST)"')
SEALED_ROLES = {"CALIBRATION", "TEST"}


def norm_entry(value):
    text = str(value or "").strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    text = text.lstrip("0") or "0"
    return text


def load_train_dev_without_deserializing_sealed(path):
    """Deserialize TRAIN/DEV only; sealed lines are counted by role and skipped."""
    selected = []
    role_counts = collections.Counter()
    sealed_deserialized = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            match = ROLE_PATTERN.search(line)
            if not match:
                raise RuntimeError("manifest row has no global_split marker")
            role = match.group(1)
            role_counts[role] += 1
            if role in SEALED_ROLES:
                continue
            record = json.loads(line)
            if record["global_split"] != role:
                raise RuntimeError("role parser disagreement")
            selected.append(record)
    return selected, {
        "role_counts_from_raw_marker": dict(role_counts),
        "sealed_rows_deserialized": sealed_deserialized,
        "test_evaluation_count": 0,
        "calibration_evaluation_count": 0,
    }


def candidate_order_hash(records):
    payload = [{
        "record_id": row["record_id"],
        "global_split": row["global_split"],
        "module": row["module"],
        "candidate_order": row["candidate_instance_ids"],
        "gold": row["gold_candidate_instance_id"],
    } for row in sorted(records, key=lambda value: value["record_id"])]
    return sha256_json(payload)


def position_sanity_baselines(records):
    train = [row for row in records if row["global_split"] == "TRAIN"]
    dev = [row for row in records if row["global_split"] == "DEV"]
    if not train or not dev:
        raise RuntimeError("position baselines require non-empty TRAIN and DEV")
    majority_by_count = {}
    position_by_stratum = {}
    for count in sorted({len(row["candidate_instance_ids"]) for row in train}):
        values = [row["gold_position"] for row in train if len(row["candidate_instance_ids"]) == count]
        majority_by_count[count] = collections.Counter(values).most_common(1)[0][0]
    for module in sorted({row["module"] for row in train}):
        for count in sorted({len(row["candidate_instance_ids"]) for row in train if row["module"] == module}):
            values = [row["gold_position"] for row in train
                      if row["module"] == module and len(row["candidate_instance_ids"]) == count]
            position_by_stratum[(module, count)] = collections.Counter(values).most_common(1)[0][0]
    always_first = sum(row["gold_position"] == 0 for row in dev) / len(dev)
    majority = sum(row["gold_position"] == majority_by_count[len(row["candidate_instance_ids"])] for row in dev) / len(dev)
    position_only = sum(row["gold_position"] == position_by_stratum[(row["module"], len(row["candidate_instance_ids"]))]
                        for row in dev) / len(dev)
    return {
        "n": len(dev),
        "always_first_R1": always_first,
        "majority_position_R1": majority,
        "position_only_R1": position_only,
        "majority_position_learned_from": "TRAIN_only_by_candidate_count",
        "position_only_learned_from": "TRAIN_only_by_module_and_candidate_count",
    }


def adapt_core_record(row, embedding_index, llm_source_by_key, mapping_by_key):
    source_key = row["source_manifest_key"]
    if source_key not in llm_source_by_key or source_key not in mapping_by_key:
        raise RuntimeError("missing audited LLM contract for source key %s" % source_key)
    source = llm_source_by_key[source_key]
    candidates = [str(value) for value in row["candidate_instance_ids"]]
    gold = str(row["gold_candidate_instance_id"])
    query = str(row["query_id"])
    missing = [value for value in [query] + candidates if value not in embedding_index]
    if missing:
        raise RuntimeError("clean embedding coverage mismatch: %s" % missing[:3])
    return {
        "record_id": str(row["record_id"]),
        "source_manifest_key": source_key,
        "global_split": row["global_split"],
        "module": row["module"],
        "sample_id": row["sample_id"],
        "query_id": query,
        "candidate_ids": candidates,
        "gold_candidate_id": gold,
        "gold_position": candidates.index(gold),
        "symbolic_query_index": int(embedding_index[query]),
        "symbolic_candidate_indices": [int(embedding_index[value]) for value in candidates],
        "pseudo_c": source["pseudo_c"],
        "function_entry": source["function_entry"],
        "instruction_mapping": mapping_by_key[source_key],
        "candidate_order_protocol": row["candidate_order_protocol"],
    }


def load_llm_contract(source_manifest, instruction_mapping, selected_keys):
    """Deserialize only keys belonging to the already-filtered global TRAIN/DEV."""
    selected_keys = {str(key) for key in selected_keys}
    if not selected_keys:
        raise RuntimeError("No TRAIN/DEV LLM source keys")
    def selected_rows(path):
        result = {}
        decoder = json.JSONDecoder()
        marker = '"manifest_key":'
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                offset = line.find(marker)
                if offset < 0:
                    raise RuntimeError("LLM source/mapping row lacks manifest_key")
                key, _ = decoder.raw_decode(line[offset + len(marker):].lstrip())
                key = str(key)
                if key not in selected_keys:
                    continue  # No outer payload is deserialized.
                if key in result:
                    raise RuntimeError("Duplicate selected LLM manifest key")
                row = json.loads(line)
                if str(row["manifest_key"]) != key:
                    raise RuntimeError("LLM manifest key parser disagreement")
                result[key] = row
        return result
    source = selected_rows(source_manifest)
    mapping = selected_rows(instruction_mapping)
    if set(source) != selected_keys or set(mapping) != selected_keys:
        raise RuntimeError("Selected TRAIN/DEV LLM source/mapping key mismatch")
    return source, mapping


class SplitFirewall:
    def __init__(self):
        self.test_evaluation_count = 0
        self.calibration_evaluation_count = 0

    def training_rows(self, records):
        forbidden = [row for row in records if row.get("global_split") != "TRAIN"]
        if forbidden:
            raise RuntimeError("non-TRAIN row passed to training")
        return records

    def development_rows(self, records):
        forbidden = [row for row in records if row.get("global_split") != "DEV"]
        if forbidden:
            raise RuntimeError("non-DEV row passed to development evaluation")
        return records

    def claim_test_evaluation(self):
        self.test_evaluation_count += 1
        raise RuntimeError("TEST is sealed in v0_2_9")

    def claim_calibration_evaluation(self):
        self.calibration_evaluation_count += 1
        raise RuntimeError("CALIBRATION is sealed in v0_2_9")

    def assert_sealed(self):
        if self.test_evaluation_count or self.calibration_evaluation_count:
            raise RuntimeError("sealed split evaluation was attempted")
        return {"test_evaluation_count": 0, "calibration_evaluation_count": 0, "passed": True}
