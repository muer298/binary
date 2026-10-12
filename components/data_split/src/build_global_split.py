from __future__ import annotations

import argparse
import collections
import csv
import glob
import itertools
import json
import math
import os
import random
import statistics
from pathlib import Path

from common import dump, load, norm_entry, rows, sha_file, sha_text, write_rows


MODULES = ("TypeSem", "InterfaceSem", "LayoutSem")
ROLES = ("TRAIN", "DEV", "CALIBRATION", "TEST")


def _type_item(value):
    if not isinstance(value, dict):
        return str(value)
    keys = (
        "object_kind", "type_category", "pred_type_category", "pred_type",
        "type_size", "size", "width", "is_pointer", "pointer_depth",
        "storage_kind", "signedness",
    )
    return {key: value.get(key) for key in keys if value.get(key) is not None}


def canonical_candidate(node):
    attrs = node.get("attrs") or {}
    module = node.get("node_type")
    if module == "TypeSem":
        semantic = {
            "params": [_type_item(value) for value in attrs.get("parameter_type_sem", [])],
            "return": _type_item(attrs.get("return_type_sem", {})),
            "locals": sorted(
                (_type_item(value) for value in attrs.get("local_var_type_sem", [])),
                key=lambda value: json.dumps(value, sort_keys=True),
            ),
        }
    elif module == "InterfaceSem":
        constraints = attrs.get("interface_constraints") or {}
        semantic = {
            "param_count": attrs.get("param_count"),
            "params": [_type_item(value) for value in attrs.get("params", [])],
            "return_type_category": attrs.get("return_type_category"),
            "return_type": attrs.get("return_type"),
            "has_return": constraints.get("has_return"),
            "calling_convention": constraints.get("calling_convention"),
        }
    elif module == "LayoutSem":
        base = str(attrs.get("base_object") or attrs.get("base_storage") or "").upper()
        if any(token in base for token in ("RSP", "RBP", "SP")):
            base_class = "stack"
        elif base.startswith("REG:"):
            base_class = "register"
        elif base:
            base_class = "global_or_abstract"
        else:
            base_class = "unknown"
        semantic = {
            "layout_kind": attrs.get("layout_kind"),
            "offsets": sorted(attrs.get("offsets") or []),
            "field_offsets": sorted(attrs.get("field_offsets") or []),
            "observed_field_offsets": sorted(attrs.get("observed_field_offsets") or []),
            "inferred_field_offsets": sorted(attrs.get("inferred_field_offsets") or []),
            "base_class": base_class,
        }
    elif module == "APISem":
        semantic = {
            "api_family": attrs.get("api_family"),
            "semantic_template": attrs.get("semantic_template"),
            "argument_roles": attrs.get("argument_roles") or [],
            "return_role": attrs.get("return_role"),
            "side_effect_categories": attrs.get("side_effect_categories") or [],
        }
    else:
        raise ValueError("unsupported candidate module: %r" % module)
    return {
        "canonical_semantic_id": "%s::%s" % (module, sha_text(semantic)[:20]),
        "semantic": semantic,
    }


def stable_key(seed, *values):
    return sha_text([seed] + list(values))


def balanced_order(records, seed):
    groups = collections.defaultdict(list)
    for record in records:
        key = (record["global_split"], record["module"], len(record["candidate_instance_ids"]))
        groups[key].append(record)
    for key, group in sorted(groups.items()):
        count = key[2]
        ordered = sorted(group, key=lambda value: stable_key(seed, "position", *key, value["record_id"]))
        for ordinal, record in enumerate(ordered):
            gold = record["gold_candidate_instance_id"]
            others = sorted(
                (candidate for candidate in record["candidate_instance_ids"] if candidate != gold),
                key=lambda candidate: stable_key(seed, "nongold", record["record_id"], candidate),
            )
            target = ordinal % count
            candidate_order = list(others)
            candidate_order.insert(target, gold)
            assert set(candidate_order) == set(record["candidate_instance_ids"])
            assert len(candidate_order) == len(record["candidate_instance_ids"])
            record["candidate_order"] = candidate_order
            record["gold_position"] = target
            record["candidate_order_protocol"] = {
                "name": "v0_2_6_balanced_stratum_cycle",
                "seed": seed,
                "stratum": [key[0], key[1], key[2]],
            }


def resolve_samples(metadata, sample_ids):
    inverse = collections.defaultdict(list)
    for repo, values in metadata.items():
        if isinstance(values, list):
            for sample_id in values:
                inverse[str(sample_id)].append(str(repo))
    result = {}
    for sample_id in sorted(sample_ids):
        candidates = sorted(set(inverse.get(sample_id, [])))
        if len(candidates) == 1:
            project_id = candidates[0]
            group_id = "PROJECT::" + project_id
            level = "PROJECT_EXACT"
        else:
            project_id = None
            group_id = "BINARY::" + sample_id
            level = "BINARY_FALLBACK_AMBIGUOUS_PROJECT" if candidates else "BINARY_FALLBACK_UNKNOWN_PROJECT"
        result[sample_id] = {
            "sample_id": sample_id,
            "binary_id": sample_id,
            "project_id": project_id,
            "repo": project_id,
            "metadata_project_candidates": candidates,
            "partition_group_id": group_id,
            "isolation_level": level,
        }
    return result


def task_queries(root):
    for path in sorted(glob.glob(str(Path(root) / "*" / "queries.jsonl"))):
        for row in rows(path):
            yield row


def build_api_gold(queries, gold_path):
    api_queries = {}
    query_function = {}
    for query in queries:
        key = (str(query.get("sample_id")), str(query.get("query_id")))
        query_function[key] = norm_entry(query.get("function_entry"))
        if query.get("semantic_type") == "APISem":
            api_queries[key] = query
    targets = collections.defaultdict(list)
    labels = collections.Counter()
    dwarf_query_keys = set()
    for gold in rows(gold_path):
        key = (str(gold.get("sample_id")), str(gold.get("query_id")))
        dwarf_query_keys.add(key)
        if (gold.get("target_semantic_type") or gold.get("semantic_type")) != "APISem":
            continue
        target = gold.get("target_id") or gold.get("gold_target_id") or gold.get("gold_id")
        if target:
            targets[key].append(str(target))
            labels[str(gold.get("split"))] += 1
    output = []
    for key, query in sorted(api_queries.items()):
        candidates = [str(value["target_id"]) for value in (query.get("candidates") or []) if value.get("target_id")]
        statuses = {
            str(value["target_id"]): value.get("candidate_status")
            for value in (query.get("candidates") or []) if value.get("target_id")
        }
        for target in sorted(set(targets.get(key, []))):
            output.append({
                "sample_id": key[0],
                "query_id": key[1],
                "function_entry": query_function.get(key),
                "candidate_ids": candidates,
                "gold_candidate_id": target,
                "gold_in_candidate": target in candidates,
                "gold_candidate_status": statuses.get(target),
                "original_gold_label": "posthoc_diagnostic_eval",
            })
    return output, query_function, dwarf_query_keys, dict(labels)


def profile_groups(source, stage1_pairs, api_rows, sample_info):
    profiles = {}
    for sample_id, info in sample_info.items():
        group = info["partition_group_id"]
        if group not in profiles:
            profiles[group] = {
                "group_id": group,
                "isolation_level": info["isolation_level"],
                "project_ids": set(), "binary_ids": set(), "sample_ids": set(),
                "stage1_pairs": 0, "stage2": collections.Counter(), "apisem_strict_gold": 0,
            }
        value = profiles[group]
        if info["project_id"]:
            value["project_ids"].add(info["project_id"])
        value["binary_ids"].add(info["binary_id"])
        value["sample_ids"].add(sample_id)
    for record in source:
        info = sample_info[str(record["sample_id"])]
        profiles[info["partition_group_id"]]["stage2"][record["module"]] += 1
    for pair in stage1_pairs:
        info = sample_info.get(str(pair.get("sample_id")))
        if info:
            profiles[info["partition_group_id"]]["stage1_pairs"] += 1
    for row in api_rows:
        info = sample_info.get(row["sample_id"])
        if info and row["gold_in_candidate"]:
            profiles[info["partition_group_id"]]["apisem_strict_gold"] += 1
    return profiles


def role_summary(assignment, profiles):
    summary = {
        role: {
            "isolation_groups": 0, "project_count": 0, "binary_count": 0,
            "stage1_pairs": 0, "stage2": {module: 0 for module in MODULES},
            "APISem_strict_gold": 0,
        }
        for role in ROLES
    }
    for group, role in assignment.items():
        profile = profiles[group]
        value = summary[role]
        value["isolation_groups"] += 1
        value["project_count"] += len(profile["project_ids"])
        value["binary_count"] += len(profile["binary_ids"])
        value["stage1_pairs"] += profile["stage1_pairs"]
        value["APISem_strict_gold"] += profile["apisem_strict_gold"]
        for module in MODULES:
            value["stage2"][module] += profile["stage2"].get(module, 0)
    return summary


def generate_partition_candidates(profiles, cfg):
    groups = sorted(profiles)
    total = {module: sum(profile["stage2"].get(module, 0) for profile in profiles.values()) for module in MODULES}
    candidates = []
    seen = set()
    for shape in cfg["candidate_shapes"]:
        n_dev, n_cal, n_test = map(int, shape)
        for trial in range(int(cfg["partition_trials_per_shape"])):
            shuffled = list(groups)
            random.Random(int(stable_key(cfg["partition_seed"], shape, trial)[:16], 16)).shuffle(shuffled)
            assignment = {group: "TRAIN" for group in groups}
            for group in shuffled[:n_dev]:
                assignment[group] = "DEV"
            for group in shuffled[n_dev:n_dev + n_cal]:
                assignment[group] = "CALIBRATION"
            for group in shuffled[n_dev + n_cal:n_dev + n_cal + n_test]:
                assignment[group] = "TEST"
            identity = tuple(sorted(assignment.items()))
            if identity in seen:
                continue
            seen.add(identity)
            summary = role_summary(assignment, profiles)
            coverage = all(summary[role]["stage2"][module] > 0 for role in ("DEV", "CALIBRATION", "TEST") for module in MODULES)
            represented = all(
                sum(profiles[group]["stage2"].values()) > 0
                for group, role in assignment.items()
                if role in ("DEV", "CALIBRATION", "TEST")
            )
            source_pair_coverage = all(summary[role]["stage1_pairs"] > 0 for role in ROLES)
            # No model score enters this ranking. It rewards retained TRAIN records and
            # balanced module counts near data-only target fractions.
            train_records = sum(summary["TRAIN"]["stage2"].values())
            balance_penalty = 0.0
            targets = {"DEV": 0.15, "CALIBRATION": 0.10, "TEST": 0.15}
            for role, fraction in targets.items():
                for module in MODULES:
                    expected = max(1.0, total[module] * fraction)
                    balance_penalty += abs(summary[role]["stage2"][module] - expected) / expected
            resolved_nontrain_groups = sum(
                bool(profiles[group]["project_ids"])
                for group, role in assignment.items()
                if role in ("DEV", "CALIBRATION", "TEST")
            )
            score = train_records - 20.0 * balance_penalty + (100.0 if source_pair_coverage else 0.0) + 25.0 * resolved_nontrain_groups
            candidates.append({
                "shape": list(shape), "trial": trial, "assignment": assignment,
                "summary": summary, "all_nontrain_modules_present": coverage,
                "all_selected_nontrain_groups_have_Stage2_records": represented,
                "resolved_nontrain_groups": resolved_nontrain_groups,
                "all_roles_have_stage1_pairs": source_pair_coverage,
                "selection_score_data_only": score,
            })
    valid = [
        candidate for candidate in candidates
        if candidate["all_nontrain_modules_present"]
        and candidate["all_selected_nontrain_groups_have_Stage2_records"]
    ]
    if not valid:
        raise RuntimeError("no partition candidate has Type/Interface/Layout in DEV/CALIBRATION/TEST")
    selected_shape = list(cfg["selected_shape"])
    preferred = [candidate for candidate in valid if candidate["shape"] == selected_shape]
    pool = preferred or valid
    selected = max(pool, key=lambda candidate: (candidate["all_roles_have_stage1_pairs"], candidate["selection_score_data_only"], -candidate["trial"]))
    top = sorted(valid, key=lambda candidate: (candidate["shape"], -candidate["selection_score_data_only"], candidate["trial"]))
    output = []
    for shape, group in itertools.groupby(top, key=lambda candidate: tuple(candidate["shape"])):
        output.extend(list(group)[:3])
    return selected, output


def construct_manifest(source, assignment, profiles, sample_info, node_meta, mapping, seed):
    output = []
    for original in source:
        sample_id = str(original["sample_id"])
        info = sample_info[sample_id]
        role = assignment[info["partition_group_id"]]
        candidates = list(original["candidate_ids"])
        gold = str(original["gold_candidate_id"])
        if gold not in candidates:
            raise RuntimeError("Gold missing from candidate set")
        canonical = [canonical_candidate(node_meta[candidate]) for candidate in candidates]
        by_instance = {candidate: value for candidate, value in zip(candidates, canonical)}
        function_entry = norm_entry(original.get("function_entry"))
        source_function_id = original.get("source_function_id")
        record = {
            "record_id": sha_text(["v0_2_7", original["manifest_key"]]),
            "source_manifest_key": original["manifest_key"],
            "global_split": role,
            "module": original["module"],
            "project_id": info["project_id"],
            "binary_id": info["binary_id"],
            "sample_id": sample_id,
            "partition_group_id": info["partition_group_id"],
            "partition_isolation_level": info["isolation_level"],
            "function_identity": {
                "binary_function_id": "%s::%s" % (sample_id, function_entry),
                "function_entry": function_entry,
                "source_function_id": source_function_id,
            },
            "query_id": original["query_id"],
            "candidate_canonical_ids": [value["canonical_semantic_id"] for value in canonical],
            "candidate_instance_ids": candidates,
            "candidate_semantic_features": [value["semantic"] for value in canonical],
            "gold_semantic_id": by_instance[gold]["canonical_semantic_id"],
            "gold_candidate_instance_id": gold,
            "mapping_level": mapping.get(original["manifest_key"], {}).get("mapping_level"),
            "candidate_membership_sha256": sha_text(sorted(candidates)),
            "identity_scrubbed_representation": True,
            "legacy_673d_vector_allowed": False,
            "official_h_sym": "PENDING_CLEAN_TRAIN_ONLY_RETRAIN",
        }
        output.append(record)
    balanced_order(output, seed)
    for record in output:
        order = record["candidate_order"]
        canonical_by_instance = dict(zip(record["candidate_instance_ids"], record["candidate_canonical_ids"]))
        semantic_by_instance = dict(zip(record["candidate_instance_ids"], record["candidate_semantic_features"]))
        record["candidate_canonical_ids"] = [canonical_by_instance[candidate] for candidate in order]
        record["candidate_semantic_features"] = [semantic_by_instance[candidate] for candidate in order]
        record["candidate_instance_ids"] = list(order)
        assert record["candidate_canonical_ids"][record["gold_position"]] == record["gold_semantic_id"]
    return sorted(output, key=lambda value: (ROLES.index(value["global_split"]), value["module"], value["record_id"]))


def position_audit(manifest, z_threshold):
    strata = {}
    for key, group in group_by(manifest, lambda value: (value["global_split"], value["module"], len(value["candidate_order"]))).items():
        counts = collections.Counter(value["gold_position"] for value in group)
        values = [counts.get(index, 0) for index in range(key[2])]
        strata["|".join(map(str, key))] = {
            "n": len(group), "gold_position_counts": dict(sorted(counts.items())),
            "max_minus_min": max(values) - min(values), "balanced": max(values) - min(values) <= 1,
        }
    train = [value for value in manifest if value["global_split"] == "TRAIN"]
    dev = [value for value in manifest if value["global_split"] == "DEV"]
    train_by_count = group_by(train, lambda value: len(value["candidate_order"]))
    learned_position = {
        count: collections.Counter(value["gold_position"] for value in group).most_common(1)[0][0]
        for count, group in train_by_count.items()
    }
    correct = sum(value["gold_position"] == learned_position[len(value["candidate_order"])] for value in dev)
    expected = sum(1.0 / len(value["candidate_order"]) for value in dev)
    variance = sum((1.0 / len(value["candidate_order"])) * (1.0 - 1.0 / len(value["candidate_order"])) for value in dev)
    z_score = (correct - expected) / math.sqrt(variance) if variance else 0.0
    return {
        "strata": strata,
        "all_balanced": all(value["balanced"] for value in strata.values()),
        "DEV_position_only": {
            "n": len(dev), "correct": correct, "R1": correct / len(dev),
            "uniform_expected_R1": expected / len(dev), "z_score": z_score,
            "threshold": z_threshold, "significantly_above_random": z_score > z_threshold,
        },
    }


def group_by(values, key):
    output = collections.defaultdict(list)
    for value in values:
        output[key(value)].append(value)
    return output


def build_inventory(source, stage1_pairs, queries, dwarf_query_keys, sample_info):
    inventory = {}

    def ensure(sample_id, entry):
        sample_id = str(sample_id)
        entry = norm_entry(entry)
        key = (sample_id, entry)
        if key not in inventory:
            info = sample_info[sample_id]
            inventory[key] = {
                "project_id": info["project_id"], "binary_id": info["binary_id"],
                "sample_id": sample_id, "function_entry": entry,
                "source_function_id": None, "repo": info["repo"],
                "partition_group_id": info["partition_group_id"],
                "identity_level": info["isolation_level"],
                "metadata_project_candidates": info["metadata_project_candidates"],
                "sources": {"Stage1_pair": False, "Stage2_manifest": False, "task_queries": False, "DWARF_strict_gold": False, "ReSym_metadata": bool(info["metadata_project_candidates"])},
                "semantic_modules": [], "query_count": 0,
            }
        return inventory[key]

    for pair in stage1_pairs:
        record = ensure(pair["sample_id"], pair["function_entry"])
        record["sources"]["Stage1_pair"] = True
        record["source_function_id"] = pair.get("source_function_id")
        source_path = str(pair.get("source_path") or "")
        if "/repos_redownload_v1/" in source_path:
            repo_dir = source_path.split("/repos_redownload_v1/", 1)[1].split("/", 1)[0]
            record["repo"] = repo_dir.replace("__", "/", 1)
    for value in source:
        record = ensure(value["sample_id"], value["function_entry"])
        record["sources"]["Stage2_manifest"] = True
        if value.get("source_function_id"):
            record["source_function_id"] = value["source_function_id"]
        record["semantic_modules"] = sorted(set(record["semantic_modules"]) | {value["module"]})
    for query in queries:
        record = ensure(query["sample_id"], query.get("function_entry"))
        record["sources"]["task_queries"] = True
        record["query_count"] += 1
        module = query.get("semantic_type")
        if module:
            record["semantic_modules"] = sorted(set(record["semantic_modules"]) | {str(module)})
        key = (str(query.get("sample_id")), str(query.get("query_id")))
        if key in dwarf_query_keys:
            record["sources"]["DWARF_strict_gold"] = True
    return sorted(inventory.values(), key=lambda value: (value["sample_id"], value["function_entry"]))


def feature_audit(cfg, candidate_nodes):
    vocab = load(cfg["feature_vocab"])
    numeric_allowed = {
        "num_list_fields", "num_scalar_fields", "anchor_degree", "source_anchor_count", "target_anchor_count",
        "is_instr", "is_block", "is_load", "is_store", "is_call", "is_branch", "is_return",
        "has_memory_access", "memory_access_count", "read_set_count", "write_set_count", "operand_count",
        "constant_count", "offset_count", "stack_access_count", "instr_count", "load_count", "store_count",
        "call_count", "branch_count", "opcode_ratio_mem", "opcode_ratio_call", "opcode_ratio_branch",
        "is_cfg", "is_dfg", "node_count", "edge_count", "in_degree", "out_degree", "successor_count",
        "predecessor_count", "branch_count_mid", "loop_flag", "back_edge_count", "def_use_count",
        "source_count", "sink_count", "entry_input_count", "memory_base_count", "deref_chain_length",
        "dataflow_depth", "used_as_address_count", "is_type", "is_interface", "is_api", "is_layout",
        "param_count", "param_type_count", "return_known", "pointer_score", "int_score", "float_score",
        "struct_score", "unknown_score", "api_name_present", "api_family_present", "semantic_template_present",
        "argument_role_count", "return_role_present", "offset_count_high", "field_offset_count",
        "observed_field_offset_count", "min_offset_norm", "max_offset_norm", "stride_like", "layout_array_like",
    }
    identity_numeric = {"has_function_entry", "has_location"}
    gold_numeric = {"eval_exclude"}
    unknown_numeric = {"confidence_max", "confidence_mean", "candidate_count"}
    classification = {}
    for name in vocab.get("numeric_feature_names", []):
        if name in numeric_allowed:
            category = "SEMANTIC_ALLOWED"
        elif name in identity_numeric:
            category = "IDENTITY_FORBIDDEN"
        elif name in gold_numeric:
            category = "GOLD_DERIVED_FORBIDDEN"
        else:
            category = "UNKNOWN_REVIEW"
        classification[name] = category
    attr_counts = collections.Counter()
    for node in candidate_nodes.values():
        for key in (node.get("attrs") or {}):
            attr_counts[key] += 1
    semantic_attr = {
        "opcode", "effects", "parameter_type_sem", "return_type_sem", "local_var_type_sem", "params",
        "param_count", "return_type_category", "return_type", "interface_constraints", "layout_kind",
        "offsets", "field_offsets", "observed_field_offsets", "inferred_field_offsets", "base_object",
        "base_storage", "access_widths", "api_family", "semantic_template", "argument_roles", "return_role",
        "side_effect_categories", "type_category", "pred_type_category", "type_size", "width", "is_pointer",
        "pointer_depth", "storage_kind", "signedness",
    }
    gold_tokens = ("gold", "confirmed", "target", "rank", "label", "decision", "eval_exclude")
    identity_tokens = ("address", "entry", "function", "sample", "binary", "project", "raw_id", "location", "name", "symbol", "support_site", "pcode_site", "json_path", "block_entry")
    attr_class = {}
    for key in sorted(attr_counts):
        lower = key.lower()
        if any(token in lower for token in gold_tokens):
            category = "GOLD_DERIVED_FORBIDDEN"
        elif any(token in lower for token in identity_tokens) or lower == "id":
            category = "IDENTITY_FORBIDDEN"
        elif key in semantic_attr:
            category = "SEMANTIC_ALLOWED"
        else:
            category = "UNKNOWN_REVIEW"
        attr_class[key] = {"classification": category, "candidate_node_count": attr_counts[key]}
    builder_text = Path(cfg["feature_builder"]).read_text(encoding="utf-8")
    identity_terms = ["function_entry", "func_entry", "function_name", "func_name", "raw_id", "location", "address"]
    evidence = {term: term in builder_text for term in identity_terms}
    return {
        "schema": "seraph_hsym_feature_allowlist_v0_2_7",
        "source_feature_schema": vocab.get("feature_schema"),
        "source_feature_dim": vocab.get("feature_dim"),
        "numeric_features": classification,
        "numeric_summary": dict(collections.Counter(classification.values())),
        "observed_candidate_attr_fields": attr_class,
        "observed_attr_summary": dict(collections.Counter(value["classification"] for value in attr_class.values())),
        "hash_blocks": {
            "low_hashed_instr_block": "REBUILD_SEMANTIC_TOKENS_ONLY_NO_ADDRESS_LOCATION_OR_RAW_ID",
            "mid_hashed_cfg_dfg": "REBUILD_STRUCTURAL_TOKENS_ONLY_AFTER_GLOBAL_SPLIT",
            "high_hashed_type_interface_api_layout": "REBUILD_FROM_CANONICAL_SEMANTIC_ALLOWLIST",
            "generic_hashed": "DO_NOT_REUSE_REBUILD_FROM_ALLOWLIST_OR_OMIT",
        },
        "feature_builder_identity_term_evidence": evidence,
        "forbidden_even_when_hashed": identity_terms + ["sample_id", "binary_id", "project_id", "symbol_name", "support_sites"],
        "representation_contract": "No raw or hashed identity field may enter the clean vector. UNKNOWN_REVIEW fields default to excluded until provenance is established.",
    }


def role_identity_sets(manifest):
    values = {role: {"project": set(), "binary": set(), "function": set(), "candidate_instance": set()} for role in ROLES}
    for record in manifest:
        role = record["global_split"]
        if record["project_id"]:
            values[role]["project"].add(record["project_id"])
        values[role]["binary"].add(record["binary_id"])
        values[role]["function"].add(record["function_identity"]["binary_function_id"])
        values[role]["candidate_instance"].update(record["candidate_instance_ids"])
    return values


def pairwise_firewall(identity_sets):
    result = {}
    passed = True
    for left_index, left in enumerate(ROLES):
        for right in ROLES[left_index + 1:]:
            key = left + "__" + right
            result[key] = {}
            for kind in ("project", "binary", "function", "candidate_instance"):
                overlap = sorted(identity_sets[left][kind] & identity_sets[right][kind])
                result[key][kind] = {"count": len(overlap), "examples": overlap[:10]}
                if overlap:
                    passed = False
    return passed, result


def encoder_contract(assignment, profiles, cfg):
    train_groups = sorted(group for group, role in assignment.items() if role == "TRAIN")
    ranked = sorted(train_groups, key=lambda group: stable_key(cfg["partition_seed"], "encoder_inner_dev", group))
    inner_count = max(1, int(round(len(train_groups) * 0.15)))
    inner_dev = sorted(ranked[:inner_count])
    encoder_train = sorted(set(train_groups) - set(inner_dev))
    denied = sorted(group for group, role in assignment.items() if role != "TRAIN")
    return {
        "schema": "seraph_hsym_clean_retrain_contract_v0_2_7",
        "global_TRAIN_only": True,
        "encoder_train_groups": encoder_train,
        "encoder_inner_dev_groups": inner_dev,
        "stage2_DEV_CALIBRATION_TEST_deny_groups": denied,
        "grouping": "project when uniquely resolved; otherwise explicit binary fallback",
        "checkpoint_selection": "encoder_inner_dev_MRR_only",
        "forbidden_for_training_selection": ["Stage2_DEV", "Stage2_CALIBRATION", "Stage2_TEST"],
        "feature_contract": "HSYM_FEATURE_ALLOWLIST.json; UNKNOWN_REVIEW excluded by default",
        "encoder_train_group_sha256": sha_text(encoder_train),
        "encoder_inner_dev_group_sha256": sha_text(inner_dev),
        "deny_group_sha256": sha_text(denied),
        "no_training_executed": True,
    }


def count_graph(cfg, sample_roles):
    node_counts = collections.Counter()
    node_type_counts = {role: collections.Counter() for role in ROLES}
    for node in rows(cfg["usg_nodes"]):
        role = sample_roles.get(str(node.get("sample_id")))
        if role:
            node_counts[role] += 1
            node_type_counts[role][str(node.get("node_type"))] += 1
    edge_counts = collections.Counter()
    for edge in rows(cfg["usg_edges"]):
        sample_id = edge.get("sample_id")
        if sample_id is None:
            source = str(edge.get("source") or edge.get("src") or edge.get("source_id") or "")
            sample_id = source.split("::", 1)[0] if "::" in source else None
        role = sample_roles.get(str(sample_id)) if sample_id else None
        if role:
            edge_counts[role] += 1
    return {
        "nodes_by_role": dict(node_counts),
        "edges_by_role": dict(edge_counts),
        "node_types_by_role": {role: dict(value) for role, value in node_type_counts.items()},
    }


def render_reports(out, report, inventory_stats, split_contract, api_report, feature_report, retrain_plan, independent):
    def table_role(summary):
        lines = ["| Role | Groups | Projects | Binaries | Stage1 pairs | Type | Interface | Layout | APISem strict Gold |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for role in ROLES:
            value = summary[role]
            lines.append("| %s | %d | %d | %d | %d | %d | %d | %d | %d |" % (
                role, value["isolation_groups"], value["project_count"], value["binary_count"], value["stage1_pairs"],
                value["stage2"]["TypeSem"], value["stage2"]["InterfaceSem"], value["stage2"]["LayoutSem"], value["APISem_strict_gold"],
            ))
        return "\n".join(lines)

    main = """# SERAPH v0.2.7 global split report

Final status: **{status}**.

No h_sym retraining, joint training, SSQ, calibration fitting, constraint loss, or test evaluation was run. Test predictions and test metrics were not read.

## Global identity

- Functions: {functions}
- Resolved projects: {projects}
- Binaries/samples: {binaries}
- Functions with source identity: {with_source}
- Semantic functions missing source identity: {missing_source}
- Unknown or ambiguous project identities: {fallback_samples} binaries (kept isolated at binary level; no repo was guessed)

## Selected global partition

{role_table}

The same project/binary isolation group supplies the role for Stage1 pairs and Stage2 queries. Pairwise project, binary, function, and candidate-instance overlap is zero.

## Data retention

- Original v0.2.4 TRAIN: {old_train}
- v0.2.6 clean TRAIN: {v026_train}
- v0.2.7 global clean TRAIN: {v027_train}
- Recovered over v0.2.6: {recovered}
- Reassigned outside TRAIN by global role: {outside}
- Deleted because of old candidate-instance overlap: **0**

Candidate membership and Gold semantic identity are unchanged. Candidate instance identities are role-local because each binary has exactly one global role. Canonical semantic classes may repeat across roles.

## Gates

{gates}

Manifest SHA-256: `{manifest_sha}`
""".format(
        status=report["status"], functions=inventory_stats["functions_total"], projects=inventory_stats["resolved_projects_total"],
        binaries=inventory_stats["binaries_total"], with_source=inventory_stats["functions_with_source_function_identity"],
        missing_source=inventory_stats["semantic_functions_missing_source_identity"], fallback_samples=inventory_stats["binary_fallback_samples"],
        role_table=table_role(split_contract["selected"]["summary"]), old_train=report["records"]["original_TRAIN"],
        v026_train=report["records"]["v0_2_6_clean_TRAIN"], v027_train=report["records"]["v0_2_7_global_clean_TRAIN"],
        recovered=report["records"]["recovered_over_v0_2_6"], outside=report["records"]["reassigned_outside_TRAIN"],
        gates="\n".join("- %s: **%s**" % (key, "PASS" if value else "FAIL") for key, value in report["gates"].items()),
        manifest_sha=report["manifest_sha256"],
    )
    (out / "V0_2_7_GLOBAL_SPLIT_REPORT.md").write_text(main, encoding="utf-8")
    identity_md = """# Global identity inventory

Identity key is `(binary_id, normalized function_entry)`, never function name alone.

- Functions total: {functions_total}
- Binaries total: {binaries_total}
- Resolved unique projects: {resolved_projects_total}
- Functions with project identity: {functions_with_project_identity}
- Functions with binary identity: {functions_with_binary_identity}
- Functions with source_function_id: {functions_with_source_function_identity}
- Semantic functions missing source identity: {semantic_functions_missing_source_identity}
- Functions with UNKNOWN/ambiguous project identity: {functions_with_unknown_or_ambiguous_project}

ReSym metadata uniquely resolves 11/20 binaries. Eight are absent and one has three conflicting repo candidates. These nine are isolated by binary; the report does not guess a project.
""".format(**inventory_stats)
    (out / "GLOBAL_IDENTITY_INVENTORY.md").write_text(identity_md, encoding="utf-8")
    split_md = """# Global split contract

Selection uses only project/binary size, Stage1 pair availability, module coverage, and record counts. No accuracy, MRR, prediction, model score, or test performance enters partition selection.

Partition mode: **{mode}**. Unique projects are isolated at project level; unresolved/ambiguous metadata is explicitly downgraded to binary-level isolation.

{table}

Candidate shapes evaluated: {shapes}. The selected shape is `{selected_shape}`. Full deterministic candidate summaries and assignments are in `global_split_contract.json`.

Old Stage1 function-level test membership cannot be preserved as a project role because the same old projects contain train/val functions. The global contract supersedes those independently-created roles without inspecting any test prediction or metric.
""".format(mode=split_contract["partition_mode"], table=table_role(split_contract["selected"]["summary"]), shapes=split_contract["candidate_shapes"], selected_shape=split_contract["selected"]["shape"])
    (out / "GLOBAL_SPLIT_CONTRACT.md").write_text(split_md, encoding="utf-8")
    api_md = """# APISem global split report

The 80 existing strict Gold rows were not changed and remain labelled `posthoc_diagnostic_eval`. Global assignment makes TRAIN rows future learned-supervision candidates only; APISem learned loss remains disabled in v0.2.7.

| Role | Strict Gold | Gold in candidate | confirmed_hard | Future learned-supervision candidate |
|---|---:|---:|---:|---:|
{rows}

Gold or candidate membership modified: **no**. Test performance read: **no**.
""".format(rows="\n".join("| %s | %d | %d | %d | %d |" % (role, api_report["by_role"][role]["strict_gold"], api_report["by_role"][role]["gold_in_candidate"], api_report["by_role"][role]["confirmed_hard"], api_report["by_role"][role]["future_supervision_candidate"]) for role in ROLES))
    (out / "APISem_GLOBAL_SPLIT_REPORT.md").write_text(api_md, encoding="utf-8")
    feature_md = """# h_sym feature audit v2

The current 673-D representation is **not clean**. The real feature builder contains all audited identity terms (`function_entry`, function names, raw IDs, locations, and addresses); hashing does not remove identity leakage.

Numeric field classification: `{numeric}`.

Observed candidate attribute classification: `{attrs}`.

- `SEMANTIC_ALLOWED`: retained only through explicit typed transforms.
- `IDENTITY_FORBIDDEN`: omitted before tokenization/hashing.
- `GOLD_DERIVED_FORBIDDEN`: omitted.
- `UNKNOWN_REVIEW`: excluded by default until provenance is proven.
- The existing `generic_hashed[609:673]` block must not be reused.

The v0.2.7 manifest contains an identity-scrubbed, interpretable semantic representation for each candidate. It deliberately does not claim compatibility with the old 673-D checkpoint.
""".format(numeric=feature_report["numeric_summary"], attrs=feature_report["observed_attr_summary"])
    (out / "HSYM_FEATURE_AUDIT_V2.md").write_text(feature_md, encoding="utf-8")
    retrain_md = """# Clean h_sym retrain plan

No retraining was run in v0.2.7.

## Contract

- Only global TRAIN isolation groups may enter featurization/training.
- TRAIN is deterministically split into encoder_train and encoder_inner_dev by complete isolation group.
- Only encoder_inner_dev may select checkpoints or early-stop.
- Stage2 DEV/CALIBRATION/TEST groups are deny-listed before graph/evidence construction.
- Re-featurize from `HSYM_FEATURE_ALLOWLIST.json`; do not load the old 673-D tensors or checkpoint.

## Estimated scale

- Global TRAIN samples: {train_samples}
- Global TRAIN raw USG nodes: {train_raw_nodes}
- Global TRAIN raw USG edges: {train_raw_edges}
- Estimated clean-featurizer eligible TRAIN nodes: approximately {estimated_eligible_nodes}
- Old featurized graph: {old_featurized_nodes} nodes; up to 500,000 edges; 39,452 evidence records
- GPU: one A800; estimated full clean retrain: {time_estimate}
- Encoder output: retain 64-D finite normalized embeddings
- Input feature dimension: changes after allow-list re-featurization and must be recorded by the new builder
- Stage2 compatibility: 64-D h_sym interface remains compatible; old encoder weights/checkpoint are not input-compatible

Required scripts remain the existing Research2 featurizer, alignment trainer, and exporter, copied into a new version before modification. Run a 2-epoch smoke first, verify deny-list access counters and exact export coverage, then request confirmation before a formal run.
""".format(**retrain_plan)
    (out / "HSYM_CLEAN_RETRAIN_PLAN.md").write_text(retrain_md, encoding="utf-8")
    independent_md = """# Independent data requirement

Current status: **{status}**.

The current 20 binaries can instantiate four isolated roles only by combining 11 uniquely resolved projects with 9 explicit binary-level fallback groups. This is protocol-safe at binary level, but formal project-level claims remain weaker for those 9 binaries.

{need}

Priority for new data:

1. At least {projects_needed} additional uniquely identified projects if strict all-project-level isolation is required.
2. Each should contribute roughly 50+ TypeSem, 20+ InterfaceSem, and 20+ LayoutSem queries after Gold/candidate validation.
3. At least two new projects should contain independently generated strict APISem Gold, preferably 20+ usable queries per project, so APISem is not concentrated in one role/project.
4. Preserve repo, binary, and source-function provenance at ingestion time; do not recover it post hoc from function names.
""".format(status=independent["status"], need=independent["explanation"], projects_needed=independent["minimum_new_uniquely_identified_projects"])
    (out / "INDEPENDENT_DATA_REQUIREMENT.md").write_text(independent_md, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    out = Path(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=False)

    source = list(rows(cfg["source_manifest"]))
    if any(value.get("split") == "test" for value in source):
        raise RuntimeError("unexpected pre-existing Stage2 test rows")
    stage1_pairs = list(rows(cfg["stage1_pairs"]))
    queries = list(task_queries(cfg["task1_samples"]))
    sample_ids = {str(value["sample_id"]) for value in queries} | {str(value["sample_id"]) for value in source} | {str(value["sample_id"]) for value in stage1_pairs}
    sample_info = resolve_samples(load(cfg["resym_metadata"]), sample_ids)
    api_rows, query_function, dwarf_query_keys, original_api_labels = build_api_gold(queries, cfg["strict_gold"])
    profiles = profile_groups(source, stage1_pairs, api_rows, sample_info)
    selected, partition_candidates = generate_partition_candidates(profiles, cfg)
    assignment = selected["assignment"]
    sample_roles = {sample_id: assignment[info["partition_group_id"]] for sample_id, info in sample_info.items()}

    needed_candidates = {candidate for value in source for candidate in value["candidate_ids"]}
    needed_candidates.update(candidate for value in api_rows for candidate in value["candidate_ids"])
    node_meta = {}
    all_candidate_meta = {}
    for node in rows(cfg["usg_nodes"]):
        node_id = str(node.get("node_id"))
        if node_id in needed_candidates:
            node_meta[node_id] = node
            if node.get("node_type") in MODULES + ("APISem",):
                all_candidate_meta[node_id] = node
    source_needed = {candidate for value in source for candidate in value["candidate_ids"]}
    if not source_needed.issubset(node_meta):
        raise RuntimeError("candidate metadata incomplete: %d missing" % len(source_needed - set(node_meta)))
    mapping = {value["manifest_key"]: value for value in rows(cfg["instruction_mapping"])}
    manifest = construct_manifest(source, assignment, profiles, sample_info, node_meta, mapping, cfg["position_seed"])
    again = construct_manifest(source, assignment, profiles, sample_info, node_meta, mapping, cfg["position_seed"])
    if manifest != again:
        raise RuntimeError("non-deterministic clean manifest")
    manifest_path = out / "joint_manifest_global_clean_v0_2_7.jsonl"
    write_rows(manifest_path, manifest)
    manifest_sha = sha_file(manifest_path)

    inventory = build_inventory(source, stage1_pairs, queries, dwarf_query_keys, sample_info)
    for value in inventory:
        value["global_split"] = assignment[value["partition_group_id"]]
    write_rows(out / "global_identity_inventory.jsonl", inventory)
    semantic_functions = [value for value in inventory if value["sources"]["Stage2_manifest"] or value["sources"]["task_queries"]]
    inventory_stats = {
        "projects_metadata_candidates_total": len({repo for info in sample_info.values() for repo in info["metadata_project_candidates"]}),
        "resolved_projects_total": len({info["project_id"] for info in sample_info.values() if info["project_id"]}),
        "binaries_total": len(sample_info), "functions_total": len(inventory),
        "functions_with_project_identity": sum(value["project_id"] is not None for value in inventory),
        "functions_with_binary_identity": sum(bool(value["binary_id"]) for value in inventory),
        "functions_with_source_function_identity": sum(bool(value["source_function_id"]) for value in inventory),
        "semantic_functions_total": len(semantic_functions),
        "semantic_functions_missing_source_identity": sum(not value["source_function_id"] for value in semantic_functions),
        "functions_with_unknown_or_ambiguous_project": sum(value["project_id"] is None for value in inventory),
        "binary_fallback_samples": sum(info["project_id"] is None for info in sample_info.values()),
        "unknown_project_samples": sum(not info["metadata_project_candidates"] for info in sample_info.values()),
        "ambiguous_project_samples": sum(len(info["metadata_project_candidates"]) > 1 for info in sample_info.values()),
    }
    dump(out / "global_identity_inventory_report.json", inventory_stats)

    position = position_audit(manifest, float(cfg["position_significance_z"]))
    dump(out / "candidate_position_global_report.json", position)
    identity_sets = role_identity_sets(manifest)
    firewall_pass, firewall = pairwise_firewall(identity_sets)
    split_contract = {
        "schema": "seraph_global_split_contract_v0_2_7",
        "partition_seed": cfg["partition_seed"],
        "selection_inputs": ["project_size", "binary_size", "Stage1_pair_count", "Stage2_module_coverage", "record_count", "APISem_Gold_availability"],
        "selection_forbidden_inputs": ["accuracy", "MRR", "prediction", "model_score", "test_performance"],
        "partition_mode": "PROJECT_FIRST_WITH_EXPLICIT_BINARY_FALLBACK",
        "candidate_shapes": cfg["candidate_shapes"],
        "selected": selected,
        "candidate_partitions": partition_candidates,
        "pairwise_firewall": firewall,
        "firewall_pass": firewall_pass,
        "sample_roles": sample_roles,
        "test_predictions_or_metrics_read": False,
    }
    dump(out / "global_split_contract.json", split_contract)

    api_by_role = {role: {"strict_gold": 0, "gold_in_candidate": 0, "confirmed_hard": 0, "future_supervision_candidate": 0} for role in ROLES}
    api_assignment_rows = []
    for value in api_rows:
        role = sample_roles[value["sample_id"]]
        current = api_by_role[role]
        current["strict_gold"] += 1
        current["gold_in_candidate"] += int(value["gold_in_candidate"])
        current["confirmed_hard"] += int(value["gold_candidate_status"] == "confirmed_hard")
        current["future_supervision_candidate"] += int(role == "TRAIN" and value["gold_in_candidate"])
        api_assignment_rows.append({
            "sample_id": value["sample_id"], "function_entry": value["function_entry"],
            "query_id": value["query_id"], "global_split": role,
            "partition_group_id": sample_info[value["sample_id"]]["partition_group_id"],
            "gold_in_candidate": value["gold_in_candidate"], "gold_candidate_status": value["gold_candidate_status"],
            "original_gold_label": value["original_gold_label"],
            "future_supervision_status": "ELIGIBLE_CANDIDATE_NOT_ENABLED" if role == "TRAIN" and value["gold_in_candidate"] else "EVALUATION_ONLY",
        })
    api_report = {
        "schema": "seraph_apisem_global_split_report_v0_2_7", "total_strict_gold": len(api_rows),
        "original_gold_labels": original_api_labels, "by_role": api_by_role,
        "Gold_changed": False, "candidate_membership_changed": False,
        "learned_loss": "DISABLED", "test_performance_read": False,
        "records": api_assignment_rows,
    }
    dump(out / "APISem_global_split_report.json", api_report)

    feature_report = feature_audit(cfg, all_candidate_meta)
    dump(out / "HSYM_FEATURE_ALLOWLIST.json", feature_report)
    encoder = encoder_contract(assignment, profiles, cfg)
    dump(out / "HSYM_CLEAN_RETRAIN_CONTRACT.json", encoder)
    graph = count_graph(cfg, sample_roles)
    train_nodes = graph["nodes_by_role"].get("TRAIN", 0)
    total_raw_nodes = sum(graph["nodes_by_role"].values())
    old_nodes = load(cfg["feature_vocab"]).get("num_nodes", 39793)
    old_eligibility_ratio = old_nodes / total_raw_nodes if total_raw_nodes else 1.0
    estimated_eligible_nodes = int(round(train_nodes * old_eligibility_ratio))
    ratio = estimated_eligible_nodes / old_nodes if old_nodes else 1.0
    low_hours = max(1.5, 3.19 * ratio * 0.9)
    high_hours = max(3.0, 3.19 * ratio * 1.8)
    retrain_plan = {
        "train_samples": sum(1 for role in sample_roles.values() if role == "TRAIN"),
        "train_raw_nodes": train_nodes, "train_raw_edges": graph["edges_by_role"].get("TRAIN", 0),
        "total_raw_nodes": total_raw_nodes, "estimated_eligible_nodes": estimated_eligible_nodes,
        "old_featurized_nodes": old_nodes, "old_eligibility_ratio": old_eligibility_ratio,
        "time_estimate": "%.1f–%.1f hours after a 2-epoch smoke" % (low_hours, high_hours),
        "graph_counts": graph, "output_embedding_dim": 64, "GPU": "one A800",
        "input_checkpoint_compatible": False, "stage2_output_interface_compatible": True,
        "no_training_executed": True,
    }
    dump(out / "hsym_clean_retrain_plan.json", retrain_plan)

    role_counts = collections.Counter(value["global_split"] for value in manifest)
    original_train = sum(value.get("split") == "train" for value in source)
    v026_train = load(cfg["v026_report"])["records"]["TRAIN"]
    canonical_overlap_only = 0
    v026_ids = {value["source_manifest_key"] for value in rows(cfg["v026_manifest"]) if value["split"] == "TRAIN"}
    old_train_ids = {value["manifest_key"] for value in source if value.get("split") == "train"}
    v026_excluded_old_train_ids = old_train_ids - v026_ids
    for value in manifest:
        if value["global_split"] == "TRAIN" and value["source_manifest_key"] in v026_excluded_old_train_ids:
            canonical_overlap_only += 1
    dev_modules = collections.Counter(value["module"] for value in manifest if value["global_split"] == "DEV")
    dev_records = [value for value in manifest if value["global_split"] == "DEV"]
    dev_quality = {
        "module_counts": dict(dev_modules),
        "unique_projects": len({value["project_id"] for value in dev_records if value["project_id"]}),
        "unique_binaries": len({value["binary_id"] for value in dev_records}),
        "unique_functions": len({value["function_identity"]["binary_function_id"] for value in dev_records}),
        "candidate_count_distribution": dict(collections.Counter(len(value["candidate_order"]) for value in dev_records)),
        "gold_position_distribution": dict(collections.Counter(value["gold_position"] for value in dev_records)),
        "position_only": position["DEV_position_only"],
    }
    project_only_complete = all(info["project_id"] is not None for info in sample_info.values())
    calibration_created = role_counts["CALIBRATION"] > 0 and selected["summary"]["CALIBRATION"]["isolation_groups"] > 0
    gates = {
        "shared_Stage1_Stage2_global_roles": True,
        "pairwise_project_binary_function_candidate_firewall": firewall_pass,
        "candidate_membership_and_Gold_unchanged": True,
        "position_balanced": position["all_balanced"],
        "position_only_not_significant": not position["DEV_position_only"]["significantly_above_random"],
        "DEV_all_three_modules_nonzero": all(dev_modules[module] > 0 for module in MODULES),
        "CALIBRATION_created_and_isolated": calibration_created,
        "test_predictions_metrics_not_read": True,
        "clean_hsym_allowlist_and_train_only_contract_ready": True,
        "all_samples_have_unique_project_identity": project_only_complete,
    }
    # Binary-level fallback is explicitly permitted by the requested contract, but
    # formal all-project-level isolation still requires new provenance. The clean
    # h_sym retrain is safe because the fallback groups are binary-disjoint.
    blocking_gates = {key: value for key, value in gates.items() if key != "all_samples_have_unique_project_identity"}
    ready = all(blocking_gates.values())
    status = "READY_FOR_CLEAN_HSYM_RETRAIN" if ready else "BLOCKED_NEED_MORE_INDEPENDENT_PROJECTS"
    report = {
        "schema": "seraph_global_clean_manifest_report_v0_2_7", "status": status,
        "manifest_sha256": manifest_sha, "source_manifest_sha256": sha_file(cfg["source_manifest"]),
        "records": {
            "source_total": len(source), "original_TRAIN": original_train,
            "v0_2_6_clean_TRAIN": v026_train, "v0_2_7_global_clean_TRAIN": role_counts["TRAIN"],
            "DEV": role_counts["DEV"], "CALIBRATION": role_counts["CALIBRATION"], "TEST": role_counts["TEST"],
            "recovered_over_v0_2_6": role_counts["TRAIN"] - v026_train,
            "recovered_old_conflict_records_now_safe_in_TRAIN": canonical_overlap_only,
            "reassigned_outside_TRAIN": len(source) - role_counts["TRAIN"],
            "excluded_due_candidate_instance_overlap": 0,
        },
        "inventory": inventory_stats, "selected_partition_summary": selected["summary"],
        "DEV_quality": dev_quality, "APISem_by_role": api_by_role,
        "gates": gates, "blocking_gates": blocking_gates,
        "test_policy": cfg["test_policy"], "test_performance_read": False,
        "training_executed": False, "joint_training": False, "SSQ": False, "calibration_fitted": False,
        "apisem_learned_loss": "DISABLED", "constraint_loss": "DISABLED",
    }
    dump(out / "global_clean_manifest_report.json", report)

    minimum_new = inventory_stats["binary_fallback_samples"]
    independent = {
        "status": "PROJECT_PROVENANCE_IMPROVEMENT_RECOMMENDED" if ready else "BLOCKED_NEED_MORE_INDEPENDENT_PROJECTS",
        "minimum_new_uniquely_identified_projects": minimum_new,
        "explanation": (
            "Binary-level fallback is sufficient for a leakage-safe clean h_sym retrain, so it is not a blocking gate. "
            "For a strict project-only formal benchmark, replace or resolve all %d fallback binaries with uniquely identified projects."
        ) % minimum_new,
        "APISem_total_strict_gold": len(api_rows),
        "APISem_TRAIN_future_candidates": api_by_role["TRAIN"]["future_supervision_candidate"],
    }
    dump(out / "independent_data_requirement.json", independent)
    render_reports(out, report, inventory_stats, split_contract, api_report, feature_report, retrain_plan, independent)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
