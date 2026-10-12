"""Evidence-preserving, Gold-blind Task2 v0.1.0 smoke core.

Inputs are newly exported Juliet USG files. Task1 model artifacts are never
modified. Raw Ghidra semantics are explicitly marked as raw hypotheses, not
misrepresented as frozen clean Stage2 predictions.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import time


MODES = ("STRUCTURAL_ONLY", "+TYPE", "+TYPE_INTERFACE", "+TYPE_INTERFACE_LAYOUT", "FULL_SERAPH")
SEMANTIC_TYPES = ("TypeSem", "InterfaceSem", "LayoutSem", "APISem")
FLOW_EDGES = {"param_def_use", "return_use"}
SOURCE_APIS = {"read", "recv", "recvfrom", "fgets", "scanf", "fscanf", "gets", "getenv"}
DANGEROUS_APIS = {
    "system": ("CWE-78", "command"), "popen": ("CWE-78", "command"),
    "execl": ("CWE-78", "command"), "execve": ("CWE-78", "command"),
    "printf": ("CWE-134", "format"), "fprintf": ("CWE-134", "format"),
    "sprintf": ("CWE-134", "format"),
    "strcpy": ("CWE-120", "unbounded_copy"), "strcat": ("CWE-120", "unbounded_copy"),
}
ALLOC_APIS = {"malloc", "calloc", "realloc", "new", "operator new"}
FREE_APIS = {"free", "delete", "operator delete"}


def read_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_rows_new(path: Path, rows) -> int:
    count = 0
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write("\n")


def norm_pc(value: object) -> str | None:
    if value is None or value == "":
        return None
    s = str(value).strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    try:
        return format(int(s, 16), "x")
    except ValueError:
        return None


def stable(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def blind_sample_id(sample: str) -> str:
    return "JULIET_" + hashlib.sha256(sample.encode("utf-8")).hexdigest()[:20]


def blind_ref(sample: str, reference: str | None) -> str | None:
    if reference is None:
        return None
    prefix = sample + "::"
    if not reference.startswith(prefix):
        raise RuntimeError(f"USG reference lacks exact sample prefix: {reference}")
    return blind_sample_id(sample) + reference[len(sample):]


def canonical_api_name(value: str) -> str:
    name = value.lower().lstrip("_")
    if name.startswith("isoc99_"):
        return name[len("isoc99_"):]
    return name


def compact_semantic(node: dict) -> dict:
    attrs = node.get("attrs") or {}
    module = node["node_type"]
    allowed = {
        "TypeSem": ("param_types", "param_type_categories", "return_type", "return_type_category", "parameter_type_sem", "return_type_sem", "type_constraints"),
        "InterfaceSem": ("params", "ret", "param_types", "param_type_categories", "interface_constraints"),
        "LayoutSem": ("layout_kind", "offsets", "field_offsets", "observed_field_offsets", "base_object", "support_sites", "location", "confidence"),
        "APISem": ("api_name", "callsite", "api_source", "deterministic_api_source", "api_source_signal_kind", "api_source_path"),
    }[module]
    return {
        "node_id": node["node_id"],
        "value": {key: attrs[key] for key in allowed if key in attrs},
        "semantic_source": "RAW_GHIDRA_USG_NOT_FROZEN_STAGE2" if module != "APISem" else "RAW_APISem_SYMBOL_PROVENANCE",
        "semantic_confidence": attrs.get("confidence") if isinstance(attrs.get("confidence"), (float, int)) else None,
        "confidence_kind": "RAW_EXPORT_HEURISTIC" if isinstance(attrs.get("confidence"), (float, int)) else "UNKNOWN",
        "evidence_refs": [],
    }


def build_input(cfg: dict, registry_path: Path, nodes_path: Path, edges_path: Path, output: Path) -> dict:
    registry = {r["sample_id"]: r for r in read_rows(registry_path)}
    hard_enabled = cfg.get("apisem_hard_evidence_enabled") is True
    if not registry:
        raise RuntimeError("empty Juliet sample registry")
    groups = collections.defaultdict(lambda: {
        "node_ids": [], "nodes_by_type": collections.defaultdict(list),
        "edge_refs": [], "evidence_refs": [], "constraint_refs": [],
        "callers": [], "callees": [], "api_hard": [], "unmapped_api": [],
    })
    node_func = {}
    raw_instr_node = {}
    instr_address_func = collections.defaultdict(set)
    instr_node_by_address = collections.defaultdict(list)
    apis_pending = []
    hard_api_symbols = collections.defaultdict(list)
    semantic_json_by_sample = {}
    observed_samples = set()
    node_types = collections.Counter()
    for node in read_rows(nodes_path):
        sample = node["sample_id"]
        if sample not in registry:
            raise RuntimeError("USG sample absent from Juliet registry")
        observed_samples.add(sample)
        semantic_json_by_sample[sample] = node["json_path"]
        module = node["node_type"]
        node_types[module] += 1
        if module == "APISem":
            api_attrs = node.get("attrs") or {}
            if hard_enabled and api_attrs.get("deterministic_api_source") is True and api_attrs.get("api_name"):
                hard_api_symbols[(sample, str(api_attrs["api_name"]))].append(node)
        entry = norm_pc(node.get("function_entry_norm") or node.get("function_entry"))
        if module == "APISem" and not entry:
            apis_pending.append(node)
            continue
        if not entry:
            continue
        key = (sample, entry)
        node_func[node["node_id"]] = key
        group = groups[key]
        group["node_ids"].append(node["node_id"])
        if module in SEMANTIC_TYPES:
            group["nodes_by_type"][module].append(compact_semantic(node))
        if module == "InstrSem":
            raw_instr_node[(sample, node.get("raw_id"))] = node["node_id"]
            address = norm_pc((node.get("attrs") or {}).get("address"))
            if address:
                instr_address_func[(sample, address)].add(entry)
                instr_node_by_address[(sample, address)].append(node["node_id"])
    api_mapping = collections.Counter()
    for node in apis_pending:
        sample = node["sample_id"]
        attrs = node.get("attrs") or {}
        address = norm_pc(attrs.get("callsite"))
        matches = instr_address_func.get((sample, address), set())
        if len(matches) != 1:
            api_mapping["unmapped_or_ambiguous"] += 1
            continue
        entry = next(iter(matches))
        key = (sample, entry)
        node_func[node["node_id"]] = key
        obj = compact_semantic(node)
        groups[key]["node_ids"].append(node["node_id"])
        groups[key]["nodes_by_type"]["APISem"].append(obj)
        if hard_enabled and attrs.get("deterministic_api_source") is True and attrs.get("api_name"):
            groups[key]["api_hard"].append({
                "node_id": node["node_id"], "api_name": str(attrs["api_name"]),
                "callsite_pc": address, "call_instruction_nodes": instr_node_by_address[(sample, address)],
                "provenance": attrs.get("api_source_path") or attrs.get("api_source") or "UNKNOWN",
                "semantic_source": "DETERMINISTIC_HARD_SYMBOL", "semantic_confidence": None,
                "evidence_refs": [],
            })
            api_mapping["hard_exact_callsite"] += 1
        else:
            api_mapping["unconfirmed_exact_callsite"] += 1
    exact_callsite_count = 0
    exact_symbol_callsite_count = 0
    call_dedup = set()
    for sample, json_path in sorted(semantic_json_by_sample.items()):
        semantic_json = json.loads(Path(json_path).read_text(encoding="utf-8"))
        for raw in (semantic_json.get("low_level_sem") or {}).get("instr_sem", []):
            if raw.get("opcode") not in {"CALL", "CALLIND"}:
                continue
            caller = norm_pc(raw.get("function_entry_norm") or raw.get("function_entry"))
            pc = norm_pc(raw.get("address"))
            node_id = raw_instr_node.get((sample, raw.get("id")))
            if not caller or not pc or not node_id or (sample, caller) not in groups:
                continue
            target = None
            operands = raw.get("operands") or []
            if len(operands) == 1:
                target = norm_pc(operands[0])
            callee = str(raw.get("callee") or "")
            dedup = (sample, caller, pc, target, callee)
            if dedup in call_dedup:
                continue
            call_dedup.add(dedup)
            if target and (sample, target) in groups and target != caller:
                groups[(sample, caller)]["callees"].append({
                    "function_entry": target, "callsite_pc": pc,
                    "edge_ref": None, "instruction_node_id": node_id,
                    "provenance": "RAW_INSTRSEM_CALL_EXACT_TARGET_ENTRY",
                })
                groups[(sample, target)]["callers"].append({
                    "function_entry": caller, "callsite_pc": pc,
                    "edge_ref": None, "instruction_node_id": node_id,
                    "provenance": "RAW_INSTRSEM_CALL_EXACT_TARGET_ENTRY",
                })
                exact_callsite_count += 1
            hard_matches = hard_api_symbols.get((sample, callee), [])
            if raw.get("call_target") == callee and len(hard_matches) == 1 and callee:
                symbol = hard_matches[0]
                symbol_attrs = symbol.get("attrs") or {}
                groups[(sample, caller)]["api_hard"].append({
                    "node_id": symbol["node_id"], "api_name": callee,
                    "callsite_pc": pc, "call_instruction_nodes": [node_id],
                    "provenance": "UNIQUE_DETERMINISTIC_HARD_SYMBOL_NAME_AND_RAW_CALL",
                    "semantic_source": "DETERMINISTIC_HARD_SYMBOL_PLUS_EXACT_CALL_INSTRUCTION",
                    "semantic_confidence": None, "evidence_refs": [],
                    "symbol_source": symbol_attrs.get("api_source"),
                    "call_target_operand": target,
                    "api_symbol_callsite_is_not_user_callsite": True,
                })
                exact_symbol_callsite_count += 1
    cross_calls = collections.Counter()
    edge_types = collections.Counter()
    for edge in read_rows(edges_path):
        sample = edge["sample_id"]
        edge_type = edge.get("edge_type")
        edge_types[edge_type] += 1
        src, dst = node_func.get(edge.get("source_node")), node_func.get(edge.get("target_node"))
        keys = {k for k in (src, dst) if k is not None}
        for key in keys:
            group = groups[key]
            group["edge_refs"].append(edge["edge_id"])
            if edge.get("edge_kind") == "evidence":
                group["evidence_refs"].append(edge["edge_id"])
            if edge.get("edge_kind") == "constraint":
                group["constraint_refs"].append(edge["edge_id"])
        if edge_type == "call_instruction" and src and dst and src == dst:
            for api in groups[src]["api_hard"]:
                if api["node_id"] == edge.get("target_node") and edge["edge_id"] not in api["evidence_refs"]:
                    api["evidence_refs"].append(edge["edge_id"])
        if edge_type == "call" and src and dst and src != dst and src[0] == dst[0]:
            callsite = norm_pc((edge.get("attrs") or {}).get("from"))
            if callsite and src[1] not in instr_address_func.get((sample, callsite), set()):
                # The callsite must lie in the caller by exact PC equality.
                continue
            groups[src]["callees"].append({"function_entry": dst[1], "callsite_pc": callsite, "edge_ref": edge["edge_id"]})
            groups[dst]["callers"].append({"function_entry": src[1], "callsite_pc": callsite, "edge_ref": edge["edge_id"]})
            cross_calls["exact"] += 1
    rows = []
    for (sample, entry), group in sorted(groups.items()):
        reg = registry[sample]
        semantics = group["nodes_by_type"]
        blind = blind_sample_id(sample)
        def blind_semantics(module: str) -> list[dict]:
            return [{**item, "node_id": blind_ref(sample, item["node_id"]),
                     "evidence_refs": [blind_ref(sample, ref) for ref in item["evidence_refs"]]}
                    for item in semantics.get(module, [])]
        def blind_calls(items: list[dict]) -> list[dict]:
            return [{**item, "edge_ref": blind_ref(sample, item.get("edge_ref")),
                     "instruction_node_id": blind_ref(sample, item.get("instruction_node_id"))}
                    for item in items]
        def blind_api(items: list[dict]) -> list[dict]:
            return [{**item, "node_id": blind_ref(sample, item["node_id"]),
                     "call_instruction_nodes": [blind_ref(sample, node) for node in item["call_instruction_nodes"]],
                     "evidence_refs": [blind_ref(sample, ref) for ref in item["evidence_refs"]]}
                    for item in items]
        row = {
            "schema": "seraph_task2_security_input_v0_1_0", "sample_id": blind,
            "source_case_id": None, "project_id": reg.get("project_id"),
            "binary_id": blind, "function_entry": entry,
            "build_variant": None, "optimization": None,
            "USG_node_ids": sorted({blind_ref(sample, node) for node in group["node_ids"]}),
            "USG_edge_refs": sorted({blind_ref(sample, ref) for ref in group["edge_refs"]}),
            "callers": sorted(blind_calls(group["callers"]), key=lambda r: (r["function_entry"], str(r["callsite_pc"]))),
            "callees": sorted(blind_calls(group["callees"]), key=lambda r: (r["function_entry"], str(r["callsite_pc"]))),
            "TypeSem_outputs": blind_semantics("TypeSem"),
            "TypeSem_confidence": None,
            "InterfaceSem_outputs": blind_semantics("InterfaceSem"),
            "InterfaceSem_confidence": None,
            "LayoutSem_outputs": blind_semantics("LayoutSem"),
            "LayoutSem_confidence": None,
            "APISem_outputs": blind_semantics("APISem"),
            "APISem_confidence": None,
            "APISem_confirmed_hard": blind_api(group["api_hard"]),
            "frozen_Stage2_outputs": None,
            "frozen_Stage2_status": "NOT_RUN_ON_JULIET",
            "evidence_refs": sorted({blind_ref(sample, ref) for ref in group["evidence_refs"]}),
            "constraint_refs": sorted({blind_ref(sample, ref) for ref in group["constraint_refs"]}),
            "constraint_refs_are_truth": False,
            "field_taint_granularity": "COARSE_FALLBACK",
            "semantic_provenance_warning": "Type/Interface/Layout values here are raw USG hypotheses, not frozen Stage2 predictions",
        }
        serialized = json.dumps(row, sort_keys=True)
        if sample in serialized or "__bad__" in serialized or "__good__" in serialized:
            raise RuntimeError("Juliet label/name leaked into candidate-facing security input")
        rows.append(row)
    count = write_rows_new(output, rows)
    return {"schema": "seraph_task2_security_input_build_report_v0_1_0",
            "status": "RAW_UPSTREAM_ONLY", "samples_in_registry": len(registry),
            "samples_observed": len(observed_samples), "functions": count,
            "node_types": dict(node_types), "edge_types": dict(edge_types),
            "APISem_exact_mapping": dict(api_mapping), "exact_cross_function_calls": cross_calls["exact"],
            "raw_instrsem_exact_target_calls": exact_callsite_count,
            "raw_instrsem_unique_hard_symbol_calls": exact_symbol_callsite_count,
            "APISem_hard_evidence_enabled": hard_enabled,
            "model_semantics_consumed": False, "Gold_read": False}


def inventories_and_summaries(inputs: Path, edges_path: Path, summary_out: Path,
                             source_out: Path, sink_out: Path) -> dict:
    rows = list(read_rows(inputs))
    by_key = {(r["sample_id"], r["function_entry"]): r for r in rows}
    node_owner = {node: key for key, row in by_key.items() for node in row["USG_node_ids"]}
    flow = collections.defaultdict(list)
    for edge in read_rows(edges_path):
        if edge.get("edge_type") not in FLOW_EDGES or not edge.get("source_resolved") or not edge.get("target_resolved"):
            continue
        sample = edge["sample_id"]
        src = blind_ref(sample, edge.get("source_node"))
        dst = blind_ref(sample, edge.get("target_node"))
        if node_owner.get(src) and node_owner.get(src) == node_owner.get(dst) and "::InstrSem::" in str(src) and "::InstrSem::" in str(dst):
            flow[node_owner[src]].append({"source_node": src, "target_node": dst,
                                          "edge_ref": blind_ref(sample, edge["edge_id"]), "edge_type": edge["edge_type"]})
    source_rows, sink_rows, summaries = [], [], []
    for key, row in sorted(by_key.items()):
        sources, sinks, resource = [], [], []
        for api in row["APISem_confirmed_hard"]:
            name = canonical_api_name(api["api_name"])
            anchor = {"sample_id": row["sample_id"], "function_entry": row["function_entry"],
                      "api_name": name, "api_symbol_name": api["api_name"],
                      "callsite_pc": api["callsite_pc"],
                      "semantic_object": api["node_id"], "evidence_refs": api["evidence_refs"],
                      "semantic_source": api["semantic_source"], "semantic_confidence": api["semantic_confidence"]}
            if name in SOURCE_APIS:
                sources.append({**anchor, "source_kind": "POTENTIAL_INPUT_API_CALL",
                                "taint_status": "UNVERIFIED_SOURCE_ORIGIN_AND_ARGUMENT_FLOW"})
            if name in DANGEROUS_APIS:
                cwe, kind = DANGEROUS_APIS[name]
                sinks.append({**anchor, "sink_kind": kind, "CWE": cwe})
            if name in ALLOC_APIS:
                resource.append({**anchor, "effect": "ALLOCATE", "pointer_identity": None})
            if name in FREE_APIS:
                resource.append({**anchor, "effect": "FREE", "pointer_identity": None})
        source_rows.extend(sources)
        sink_rows.extend(sinks)
        summaries.append({
            "schema": "seraph_task2_function_security_summary_v0_1_0",
            "sample_id": row["sample_id"], "project_id": row["project_id"],
            "binary_id": row["binary_id"], "function_entry": row["function_entry"],
            "TaintPropagationSummary": {
                "sources": sources, "sinks": sinks,
                "instruction_flow_edges": flow.get(key, []),
                "verified_source_to_sink_argument_paths": [],
                "sanitized": "UNKNOWN_NO_VERIFIED_CFG_DFG_DOMINANCE",
                "field_granularity": "COARSE_FALLBACK",
                "return_flow": "UNKNOWN", "output_pointer_flow": "UNKNOWN", "field_flow": "UNKNOWN",
            },
            "ResourceEffectSummary": {"effects": resource, "ownership_transfer": "UNKNOWN", "resource_use": "UNKNOWN"},
            "SafetyPreconditionSummary": {"nonnull": "UNKNOWN", "size_bound": "UNKNOWN", "range_bound": "UNKNOWN", "required_state": "UNKNOWN"},
            "callees": row["callees"], "callers": row["callers"],
            "semantic_source": "RAW_USG_AND_DETERMINISTIC_HARD_API",
            "model_semantics_used": False,
        })
    write_rows_new(summary_out, summaries)
    write_json_new(source_out, {"schema": "seraph_task2_taint_source_inventory_v0_1_0", "count": len(source_rows), "sources": source_rows,
                                "rule": "Only observed deterministic hard API names; no documentation-only source assumed"})
    write_json_new(sink_out, {"schema": "seraph_task2_taint_sink_inventory_v0_1_0", "count": len(sink_rows), "sinks": sink_rows,
                              "rule": "Only observed deterministic hard API names; memory sinks require additional verified instruction semantics"})
    return {"functions": len(summaries), "source_apis": len(source_rows), "sink_apis": len(sink_rows),
            "flow_edges": sum(map(len, flow.values())), "model_semantics_used": False}


def tarjan(graph: dict[str, set[str]]) -> list[list[str]]:
    indices, low, stack, onstack, components = {}, {}, [], set(), []
    counter = 0

    def visit(node: str) -> None:
        nonlocal counter
        indices[node] = low[node] = counter
        counter += 1
        stack.append(node)
        onstack.add(node)
        for nxt in sorted(graph[node]):
            if nxt not in indices:
                visit(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in onstack:
                low[node] = min(low[node], indices[nxt])
        if low[node] == indices[node]:
            component = []
            while True:
                item = stack.pop()
                onstack.remove(item)
                component.append(item)
                if item == node:
                    break
            components.append(sorted(component))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return components


def propagate(summaries_path: Path, output: Path) -> dict:
    rows = list(read_rows(summaries_path))
    by_sample = collections.defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][row["function_entry"]] = row
    report = {"schema": "seraph_task2_interprocedural_summary_report_v0_1_0",
              "propagation_kind": "OBSERVED_API_INVENTORY_ONLY_NOT_A_PROVEN_TAINT_PATH",
              "samples": {}, "terminated": True, "checkpoint_resume": "output exists => no rerun; per-sample snapshot below"}
    for sample, funcs in sorted(by_sample.items()):
        graph = {entry: {c["function_entry"] for c in row["callees"] if c["function_entry"] in funcs}
                 for entry, row in funcs.items()}
        comps = tarjan(graph)
        propagated = {entry: {
            "source_names": {x["api_name"] for x in row["TaintPropagationSummary"]["sources"]},
            "sink_names": {x["api_name"] for x in row["TaintPropagationSummary"]["sinks"]},
            "resource_effects": {x["effect"] for x in row["ResourceEffectSummary"]["effects"]},
        } for entry, row in funcs.items()}
        max_passes = len(funcs) + 1
        passes = 0
        while passes < max_passes:
            passes += 1
            changed = False
            for caller in sorted(graph):
                for callee in sorted(graph[caller]):
                    for field in ("source_names", "sink_names", "resource_effects"):
                        before = len(propagated[caller][field])
                        propagated[caller][field].update(propagated[callee][field])
                        changed |= len(propagated[caller][field]) > before
            if not changed:
                break
        else:
            raise RuntimeError(f"interprocedural fixed point did not terminate: {sample}")
        report["samples"][sample] = {
            "functions": len(funcs), "exact_call_edges": sum(map(len, graph.values())),
            "SCC_count": len(comps), "recursive_SCC_count": sum(len(c) > 1 or (len(c) == 1 and c[0] in graph[c[0]]) for c in comps),
            "max_passes": max_passes, "passes_to_fixed_point": passes,
            "leaf_functions": sum(not edges for edges in graph.values()),
            "functions_with_reachable_source_inventory": sum(bool(x["source_names"]) for x in propagated.values()),
            "functions_with_reachable_sink_inventory": sum(bool(x["sink_names"]) for x in propagated.values()),
            "recursive_SCC_rule": "Monotone finite set union to fixed point, bounded by function_count+1; no unbounded recursion",
        }
    write_json_new(output, report)
    return report


def generate_candidates(inputs_path: Path, summaries_path: Path, templates_path: Path,
                        output: Path, metrics_path: Path) -> dict:
    templates = json.loads(templates_path.read_text(encoding="utf-8"))["templates"]
    if len(templates) != 5:
        raise RuntimeError("template family count changed")
    inputs = {(r["sample_id"], r["function_entry"]): r for r in read_rows(inputs_path)}
    summaries = {(r["sample_id"], r["function_entry"]): r for r in read_rows(summaries_path)}
    if inputs.keys() != summaries.keys():
        raise RuntimeError("input/summary function identity mismatch")
    # A source and a dangerous sink in the same function are NOT sufficient:
    # argument identity and a verified def-use path are mandatory.
    candidates = []
    reasons = collections.Counter()
    for key, row in sorted(inputs.items()):
        summary = summaries[key]
        for template in templates:
            tid = template["template_id"]
            reasons[tid + ":functions_considered"] += 1
            if tid == "BOF_BOUNDARY_WRITE_V1":
                if row["frozen_Stage2_outputs"] is None:
                    reasons[tid + ":missing_frozen_layout_interface_type"] += 1
                    continue
                # A violation requires an exact object boundary, write offset,
                # write length, and path/sanitizer result. Raw offset hints do
                # not satisfy this contract.
                reasons[tid + ":missing_verified_boundary_length_and_path"] += 1
                continue
            if tid == "UAF_LIFECYCLE_V1":
                effects = summary["ResourceEffectSummary"]["effects"]
                if not effects:
                    reasons[tid + ":no_observed_resource_API_effect"] += 1
                    continue
                if any(effect.get("pointer_identity") is None for effect in effects):
                    reasons[tid + ":missing_pointer_alias_identity"] += 1
                    continue
                reasons[tid + ":missing_verified_free_to_deref_reachability"] += 1
                continue
            if tid == "INTEGER_RANGE_TO_SINK_V1":
                if row["frozen_Stage2_outputs"] is None:
                    reasons[tid + ":missing_frozen_signedness_width"] += 1
                    continue
                reasons[tid + ":missing_verified_arithmetic_range_and_sink_path"] += 1
                continue
            if tid == "NULL_DEREF_DOMINANCE_V1":
                if row["frozen_Stage2_outputs"] is None:
                    reasons[tid + ":missing_frozen_pointer_semantics"] += 1
                    continue
                reasons[tid + ":missing_verified_maybe_null_and_cfg_dominance"] += 1
                continue
            if tid == "DANGEROUS_API_ARGUMENT_V1":
                if not summary["TaintPropagationSummary"]["sources"] or not summary["TaintPropagationSummary"]["sinks"]:
                    reasons[tid + ":missing_observed_source_or_sink"] += 1
                    continue
                if not summary["TaintPropagationSummary"]["verified_source_to_sink_argument_paths"]:
                    reasons[tid + ":missing_argument_identity_path"] += 1
                    continue
            else:
                raise RuntimeError(f"unhandled template ID {tid}")
            # Reserved for a future verified path provider. Never emit a
            # candidate based on co-occurrence, source labels, or Gold.
            raise RuntimeError("verified path provider not implemented")
    write_rows_new(output, candidates)
    result = {"schema": "seraph_task2_candidate_generation_pre_gold_v0_1_0",
              "candidate_count": len(candidates), "functions_considered": len(inputs),
              "reason_counts": dict(reasons), "Gold_read": False,
              "status": "BLOCKED_MISSING_VERIFIED_CONSTRAINT_AND_TAINT_PREMISES"}
    write_json_new(metrics_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build-input", "summaries", "propagate", "candidates"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--registry")
    parser.add_argument("--nodes")
    parser.add_argument("--edges")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.output_root or cfg["output_root"])
    if cfg["semantics_mode"] not in MODES:
        raise RuntimeError("invalid SEMANTICS_MODE")
    if cfg["semantics_mode"] != "STRUCTURAL_ONLY":
        raise RuntimeError("higher semantics mode unavailable until frozen Stage2 Juliet inference is audited")
    start = time.monotonic()
    if args.action == "build-input":
        if not (args.registry and args.nodes and args.edges):
            parser.error("build-input requires --registry --nodes --edges")
        result = build_input(cfg, Path(args.registry), Path(args.nodes), Path(args.edges), out / "security_input_manifest.jsonl")
        write_json_new(out / "SECURITY_INPUT_BUILD_REPORT.json", result)
    elif args.action == "summaries":
        if not args.edges:
            parser.error("summaries requires --edges")
        result = inventories_and_summaries(out / "security_input_manifest.jsonl", Path(args.edges),
                                          out / "function_security_summaries.jsonl",
                                          out / "taint_source_inventory.json", out / "taint_sink_inventory.json")
        write_json_new(out / "FUNCTION_SUMMARY_BUILD_REPORT.json", result)
    elif args.action == "propagate":
        result = propagate(out / "function_security_summaries.jsonl", out / "interprocedural_summary_report.json")
    else:
        result = generate_candidates(out / "security_input_manifest.jsonl",
                                     out / "function_security_summaries.jsonl",
                                     out / "security_constraint_templates.json",
                                     out / "vulnerability_candidates.jsonl",
                                     out / "CANDIDATE_GENERATION_PREFLIGHT.json")
    print(json.dumps({"action": args.action, "elapsed_seconds": round(time.monotonic() - start, 3),
                      "result": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
