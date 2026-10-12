#!/usr/bin/env python3
import argparse
import collections
import copy
import json
import math
from pathlib import Path

import torch

from io_utils import dump, load_config, rows, sha256


TYPE_FLAGS = {
    "is_instr": "InstrSem", "is_block": "BlockSem", "is_cfg": "CFGSem",
    "is_dfg": "DFGSem", "is_type": "TypeSem", "is_interface": "InterfaceSem",
    "is_api": "APISem", "is_layout": "LayoutSem",
}
TYPE_CATEGORIES = ("ptr", "int", "float", "struct", "unknown")
FORBIDDEN_IDENTITY_KEYS = {
    "sample_id", "binary_id", "project_id", "raw_id", "id", "function_entry",
    "function_entry_norm", "func_entry", "function_name", "func_name", "function_id",
    "symbol_name", "address", "block_entry", "pcode_index", "pcode_site_id", "location",
    "support_sites", "json_path", "source_path", "candidate_position", "gold_position",
    "split", "global_split", "partition_group_id",
}


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def as_number(value, default=0.0):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float(default)


def nested(attrs, *path, default=None):
    value = attrs
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def type_categories(attrs):
    values = []
    values.extend(as_list(attrs.get("param_type_categories")))
    values.append(attrs.get("return_type_category"))
    for key in ("parameter_type_sem", "local_var_type_sem"):
        for item in as_list(attrs.get(key)):
            if isinstance(item, dict):
                values.append(item.get("type_category") or item.get("pred_type_category"))
    ret = attrs.get("return_type_sem")
    if isinstance(ret, dict):
        values.append(ret.get("type_category") or ret.get("pred_type_category"))
    out = []
    for value in values:
        text = str(value or "").lower()
        if "ptr" in text or "pointer" in text:
            out.append("ptr")
        elif "float" in text or "double" in text:
            out.append("float")
        elif "struct" in text or "array" in text or "aggregate" in text:
            out.append("struct")
        elif any(token in text for token in ("int", "char", "bool", "enum", "uint", "long", "short")):
            out.append("int")
        else:
            out.append("unknown")
    return out


def offset_values(attrs):
    values = []
    for key in ("field_offsets", "observed_field_offsets", "inferred_field_offsets", "offsets"):
        for item in as_list(attrs.get(key)):
            if isinstance(item, (int, float)):
                values.append(float(item))
            elif isinstance(item, dict):
                for field in ("offset", "field_offset", "value"):
                    if isinstance(item.get(field), (int, float)):
                        values.append(float(item[field])); break
    for access in as_list(nested(attrs, "effects", "memory_accesses", default=[])):
        if isinstance(access, dict) and isinstance(access.get("offset"), (int, float)):
            values.append(float(access["offset"]))
    return values


def build_graph_stats(edge_path, eligible):
    result = collections.defaultdict(collections.Counter)
    retained = 0
    for edge in rows(edge_path):
        source, target = edge.get("source_node"), edge.get("target_node")
        if source not in eligible or target not in eligible:
            continue
        retained += 1
        kind = str(edge.get("edge_type") or edge.get("edge_kind") or "").lower()
        result[source]["out_degree"] += 1
        result[target]["in_degree"] += 1
        if any(token in kind for token in ("def_use", "data_flow", "dfg", "param_def_use", "return_use")):
            result[source]["def_use_count"] += 1
            result[target]["def_use_count"] += 1
        if any(token in kind for token in ("branch", "control", "cfg")):
            result[source]["branch_count_mid"] += 1
        if "back" in kind or "loop" in kind:
            result[source]["back_edge_count"] += 1
        if "anchor" in kind or edge.get("edge_kind") == "evidence":
            result[source]["source_anchor_count"] += 1
            result[target]["target_anchor_count"] += 1
    return result, retained


def semantic_values(node, graph):
    attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
    node_type = str(node.get("node_type") or "unknown")
    opcode = str(attrs.get("opcode") or "").upper()
    effects = attrs.get("effects") if isinstance(attrs.get("effects"), dict) else {}
    reads, writes = as_list(effects.get("reads")), as_list(effects.get("writes"))
    accesses = [x for x in as_list(effects.get("memory_accesses")) if isinstance(x, dict)]
    offsets = offset_values(attrs)
    categories = type_categories(attrs)
    category_counts = collections.Counter(categories)
    category_total = max(1, len(categories))
    params = as_list(attrs.get("params") or attrs.get("parameter_type_sem"))
    constraints = attrs.get("interface_constraints") if isinstance(attrs.get("interface_constraints"), dict) else {}
    evidence_summary = nested(attrs, "type_constraints", "evidence_summary", default={})
    evidence_summary = evidence_summary if isinstance(evidence_summary, dict) else {}
    semantic_template = attrs.get("semantic_template") if isinstance(attrs.get("semantic_template"), dict) else {}
    param_roles = as_list(semantic_template.get("param_roles"))
    control = str(effects.get("control_effect") or "").lower()
    memory_kinds = [str(x.get("access") or "").lower() for x in accesses]
    fixed_accesses = [x for x in accesses if x.get("is_fixed")]
    g = graph

    values = collections.defaultdict(float)
    for name, expected in TYPE_FLAGS.items():
        values[name] = float(node_type == expected)
    values["in_degree"] = g["in_degree"]
    values["out_degree"] = g["out_degree"]
    values["edge_count"] = g["in_degree"] + g["out_degree"]
    values["anchor_degree"] = g["source_anchor_count"] + g["target_anchor_count"]
    for key in ("def_use_count", "branch_count_mid", "back_edge_count", "source_anchor_count", "target_anchor_count"):
        values[key] = g[key]
    values["instr_count"] = as_number(attrs.get("instruction_count"), float(node_type == "InstrSem"))
    values["node_count"] = max(1.0, as_number(attrs.get("member_count"), 1.0))
    values["operand_count"] = len(as_list(attrs.get("input_value_ids"))) + len(as_list(attrs.get("output_value_ids")))
    values["read_set_count"] = len(reads)
    values["write_set_count"] = len(writes)
    values["memory_access_count"] = len(accesses)
    values["has_memory_access"] = float(bool(accesses))
    values["load_count"] = sum(k in {"load", "read"} for k in memory_kinds) + float("LOAD" in opcode)
    values["store_count"] = sum(k in {"store", "write"} for k in memory_kinds) + float("STORE" in opcode)
    values["stack_access_count"] = sum("stack" in str(x.get("base") or "").lower() or "rsp" in str(x.get("base") or "").lower() for x in accesses)
    values["memory_base_count"] = len({str(x.get("base") or "") for x in accesses if x.get("base")})
    values["used_as_address_count"] = len(accesses)
    values["deref_chain_length"] = len(accesses)
    values["constant_count"] = float("CONST" in opcode)
    values["is_load"] = float("LOAD" in opcode or "read" in memory_kinds)
    values["is_store"] = float("STORE" in opcode or "write" in memory_kinds)
    values["is_call"] = float("CALL" in opcode or control == "call")
    values["is_branch"] = float("BRANCH" in opcode or control in {"branch", "conditional_branch"})
    values["is_return"] = float("RETURN" in opcode or control == "return")
    values["call_count"] = values["is_call"]
    values["branch_count"] = values["is_branch"]
    values["loop_flag"] = float(g["back_edge_count"] > 0)
    values["opcode_ratio_branch"] = values["is_branch"]
    values["opcode_ratio_call"] = values["is_call"]
    values["opcode_ratio_mem"] = float(values["is_load"] or values["is_store"])
    values["predecessor_count"] = g["in_degree"]
    values["successor_count"] = g["out_degree"]
    values["source_count"] = len(reads)
    values["sink_count"] = len(writes)
    values["dataflow_depth"] = values["operand_count"]
    values["param_count"] = as_number(attrs.get("param_count"), len(params))
    values["param_type_count"] = len(categories)
    values["argument_role_count"] = len(param_roles)
    values["entry_input_count"] = len(as_list(constraints.get("cross_layer_constraints", {}).get("entry_input_locs") if isinstance(constraints.get("cross_layer_constraints"), dict) else []))
    values["return_known"] = float(bool(attrs.get("return_type_category") or attrs.get("return_type_sem")))
    values["return_role_present"] = float(bool(attrs.get("return_type_sem") or attrs.get("ret")))
    values["pointer_score"] = category_counts["ptr"] / category_total
    values["int_score"] = category_counts["int"] / category_total
    values["float_score"] = category_counts["float"] / category_total
    values["struct_score"] = category_counts["struct"] / category_total
    values["unknown_score"] = category_counts["unknown"] / category_total
    # Count only explicitly semantic containers/scalars. Counting all raw
    # attrs would indirectly encode identity-bearing pcode_index values.
    semantic_list_keys = (
        "parameter_type_sem", "local_var_type_sem", "param_type_categories",
        "params", "field_offsets", "observed_field_offsets",
        "inferred_field_offsets", "offsets", "access_widths",
        "memory_access_kinds",
    )
    semantic_scalar_keys = (
        "param_count", "instruction_count", "member_count",
        "member_unique_machine_instruction_count",
        "layout_evidence_cluster_member_count",
    )
    values["num_list_fields"] = sum(isinstance(attrs.get(key), list) for key in semantic_list_keys)
    values["num_scalar_fields"] = sum(isinstance(attrs.get(key), (bool, int, float)) for key in semantic_scalar_keys)
    values["field_offset_count"] = len(as_list(attrs.get("field_offsets")))
    values["observed_field_offset_count"] = len(as_list(attrs.get("observed_field_offsets")))
    values["offset_count"] = len(offsets)
    values["offset_count_high"] = len({value for value in offsets})
    values["min_offset_norm"] = max(-1.0, min(1.0, min(offsets) / 4096.0)) if offsets else 0.0
    values["max_offset_norm"] = max(-1.0, min(1.0, max(offsets) / 4096.0)) if offsets else 0.0
    values["stride_like"] = float(len(set(offsets)) >= 2)
    values["layout_array_like"] = float("array" in str(attrs.get("layout_kind") or "").lower() or values["stride_like"])
    values["api_name_present"] = float(bool(attrs.get("api_name")))
    values["api_family_present"] = float(bool(semantic_template.get("api_intent") or semantic_template.get("side_effects")))
    values["semantic_template_present"] = float(bool(semantic_template))
    # Fixed offset accesses are semantic; their sites/addresses are not used.
    values["offset_count"] = max(values["offset_count"], len(fixed_accesses))
    values["memory_access_count"] = max(values["memory_access_count"], as_number(evidence_summary.get("memory_access_count")))
    return values


def vectorize(node, columns, graph_stats):
    values = semantic_values(node, graph_stats)
    return [float(values[name]) for name in columns]


def mutate_identity_fields(value):
    """Recursively perturb every identity carrier used by the hard audit."""
    if isinstance(value, dict):
        return {
            key: ("IDENTITY_MUTATED" if key in FORBIDDEN_IDENTITY_KEYS else mutate_identity_fields(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [mutate_identity_fields(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", choices=["train", "coverage_only"], required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_root"])
    allowlist = json.loads(Path(cfg["feature_allowlist"]).read_text())
    numeric = allowlist["numeric_features"]
    columns = sorted(name for name, status in numeric.items() if status == "SEMANTIC_ALLOWED")
    if any(numeric[name] != "SEMANTIC_ALLOWED" for name in columns):
        raise AssertionError("non-allowed feature selected")
    if len(columns) != allowlist["numeric_summary"]["SEMANTIC_ALLOWED"]:
        raise RuntimeError("allow-list cardinality mismatch")

    nodes = list(rows(args.nodes))
    node_ids = [str(node["node_id"]) for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise RuntimeError("duplicate node IDs in feature input")
    eligible = set(node_ids)
    graph_stats, retained_edges = build_graph_stats(args.edges, eligible)
    vectors = [vectorize(node, columns, graph_stats[node_id]) for node, node_id in zip(nodes, node_ids)]
    features = torch.tensor(vectors, dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise RuntimeError("NaN/Inf in clean feature tensor")

    schema = {
        "schema": "seraph_clean_feature_schema_v0_2_8",
        "scope": args.scope,
        "representation": "explicit_allowlist_numeric_no_hash_no_embedding_lookup",
        "feature_dim": len(columns),
        "columns": [{
            "index": index,
            "name": name,
            "classification": numeric[name],
            "identity_bearing": False,
        } for index, name in enumerate(columns)],
        "node_ids_are_index_metadata_not_model_features": True,
        "forbidden_identity_feature_count": 0,
        "hash_feature_count": 0,
        "allowlist_sha256": sha256(cfg["feature_allowlist"]),
    }
    payload = {
        "features": features,
        "node_ids": node_ids,
        "node_index": {node_id: index for index, node_id in enumerate(node_ids)},
        "feature_dim": len(columns),
        "feature_schema": schema,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Mutation test: changing every known identity carrier must not change the vector.
    mutation_failures = []
    for node in nodes[: min(512, len(nodes))]:
        mutated = mutate_identity_fields(copy.deepcopy(node))
        original = vectorize(node, columns, graph_stats[str(node["node_id"])])
        changed = vectorize(mutated, columns, graph_stats[str(node["node_id"])])
        if original != changed:
            mutation_failures.append(str(node["node_id"]))

    leakage = {
        "schema": "seraph_clean_feature_leakage_audit_v0_2_8",
        "scope": args.scope,
        "tensor_path": str(output),
        "tensor_shape": list(features.shape),
        "tensor_dtype": str(features.dtype),
        "schema_column_count": len(columns),
        "forbidden_identity_feature_count": 0,
        "gold_derived_feature_count": 0,
        "unknown_review_feature_count": 0,
        "hash_feature_count": 0,
        "identity_mutation_cases": min(512, len(nodes)),
        "identity_mutation_failures": mutation_failures,
        "node_ids_role": "row/index metadata only; never included in tensor columns",
        "finite": bool(torch.isfinite(features).all()),
        "passed": not mutation_failures and features.shape[1] == len(columns),
    }
    if not leakage["passed"]:
        raise RuntimeError("identity leakage mutation audit failed")
    # A failed audit must never publish a tensor.
    torch.save(payload, output)

    report = {
        "schema": "seraph_clean_featurization_report_v0_2_8",
        "scope": args.scope,
        "input_nodes": str(Path(args.nodes).resolve()),
        "input_edges": str(Path(args.edges).resolve()),
        "eligible_node_count": len(nodes),
        "retained_edge_count": retained_edges,
        "feature_dim": len(columns),
        "feature_nonzero_count": int(torch.count_nonzero(features)),
        "feature_density": float(torch.count_nonzero(features) / max(1, features.numel())),
        "excluded_feature_classes": {
            "IDENTITY_FORBIDDEN": sum(v == "IDENTITY_FORBIDDEN" for v in numeric.values()),
            "GOLD_DERIVED_FORBIDDEN": sum(v == "GOLD_DERIVED_FORBIDDEN" for v in numeric.values()),
            "UNKNOWN_REVIEW": sum(v == "UNKNOWN_REVIEW" for v in numeric.values()),
            "legacy_hashed_blocks": len(allowlist.get("hash_blocks", {})),
        },
        "forbidden_identity_feature_count": 0,
        "passed": leakage["passed"],
    }
    preparation_path = out / "data_preparation_report.json"
    if preparation_path.exists():
        preparation = json.loads(preparation_path.read_text())
        if args.scope == "train":
            report.update({
                "raw_node_count": preparation["raw_node_count_global_train"],
                "excluded_node_count": preparation["excluded_node_count_global_train"],
                "exclusion_reasons": {
                    "not_in_research2_final_model_input_eligibility": preparation["excluded_node_count_global_train"],
                    "outer_global_role": 0,
                    "forbidden_feature_required": 0,
                },
            })
        else:
            report.update({
                "raw_node_count": preparation["raw_node_count_all"],
                "excluded_node_count": preparation["raw_node_count_all"] - len(nodes),
                "exclusion_reasons": {
                    "not_in_research2_final_model_input_eligibility": preparation["raw_node_count_all"] - len(nodes),
                    "forbidden_feature_required": 0,
                },
            })

    suffix = "" if args.scope == "train" else "_coverage_only"
    dump(out / ("clean_feature_schema%s.json" % suffix), schema)
    dump(out / ("clean_feature_vocab%s.json" % suffix), {
        "schema": "seraph_clean_feature_vocab_v0_2_8",
        "features": columns,
        "hash_buckets": 0,
        "categorical_identity_vocab": {},
    })
    dump(out / ("clean_featurization_report%s.json" % suffix), report)
    dump(out / ("clean_feature_leakage_audit%s.json" % suffix), leakage)
    if args.scope == "train":
        md = [
            "# Clean feature leakage audit", "",
            "Status: **PASSED**", "",
            "The serialized tensor has %d explicit numeric columns. Every column is marked `SEMANTIC_ALLOWED` by v0.2.7." % len(columns),
            "No generic/high/low/mid hash block or identity embedding lookup is present.", "",
            "- Forbidden identity features: 0",
            "- Gold-derived features: 0",
            "- UNKNOWN_REVIEW features: 0",
            "- Hash features: 0",
            "- Identity mutation checks: %d; failures: 0" % leakage["identity_mutation_cases"], "",
            "`node_ids` are serialized only as row/index metadata required to join graph nodes to feature rows; they are not tensor columns and never enter a learned embedding lookup.",
        ]
        (out / "CLEAN_FEATURE_LEAKAGE_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
