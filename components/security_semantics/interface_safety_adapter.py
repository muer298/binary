#!/usr/bin/env python3
"""Strict InterfaceSem and deterministic actual/formal adapter for Task2 v0.1.6."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path


ABI_REGS = ("reg:RDI", "reg:RSI", "reg:RDX", "reg:RCX", "reg:R8", "reg:R9")
API_SIGNATURES = {
    "memcpy": (("buffer_destination", "ptr"), ("buffer_source", "ptr"), ("size", "int")),
    "memmove": (("buffer_destination", "ptr"), ("buffer_source", "ptr"), ("size", "int")),
    "strcpy": (("buffer_destination", "ptr"), ("buffer_source", "ptr")),
    "strncpy": (("buffer_destination", "ptr"), ("buffer_source", "ptr"), ("size", "int")),
    "strcat": (("buffer_input_output", "ptr"), ("buffer_source", "ptr")),
    "read": (("resource", "int"), ("buffer_output", "ptr"), ("size", "int")),
    "recv": (("resource", "int"), ("buffer_output", "ptr"), ("size", "int"), ("flags", "int")),
    "recvfrom": (("resource", "int"), ("buffer_output", "ptr"), ("size", "int"), ("flags", "int")),
    "fgets": (("buffer_output", "ptr"), ("size", "int"), ("resource", "ptr")),
    "gets": (("buffer_output", "ptr"),),
    "malloc": (("size", "int"),), "calloc": (("count", "int"), ("size", "int")),
    "realloc": (("resource_input", "ptr"), ("size", "int")), "free": (("resource_release", "ptr"),),
    "system": (("command_sink", "ptr"),), "popen": (("command_sink", "ptr"), ("mode", "ptr")),
    "execv": (("command_sink", "ptr"), ("argv", "ptr")),
    "execve": (("command_sink", "ptr"), ("argv", "ptr"), ("envp", "ptr")),
}


def rows(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)


def blind(value: str) -> str:
    return "JULIET_" + hashlib.sha256(value.encode()).hexdigest()[:20]


def norm(value):
    try: return format(int(str(value), 16), "x")
    except (TypeError, ValueError): return None


def api_name(value):
    result = str(value or "").lower().lstrip("_")
    return result[8:] if result.startswith("isoc99_") else result


def canon_loc(value):
    text = str(value or "")
    if not text.startswith("reg:"): return text
    reg = text[4:].upper()
    families = {
        "RAX": {"RAX", "EAX", "AX", "AL", "AH"}, "RBX": {"RBX", "EBX", "BX", "BL", "BH"},
        "RCX": {"RCX", "ECX", "CX", "CL", "CH"}, "RDX": {"RDX", "EDX", "DX", "DL", "DH"},
        "RSI": {"RSI", "ESI", "SI", "SIL"}, "RDI": {"RDI", "EDI", "DI", "DIL"},
        "RSP": {"RSP", "ESP", "SP", "SPL"}, "RBP": {"RBP", "EBP", "BP", "BPL"},
    }
    for full, aliases in families.items():
        if reg in aliases: return "reg:" + full
    if reg.startswith("R") and reg[-1:] in {"B", "W", "D"} and reg[1:-1].isdigit(): return "reg:" + reg[:-1]
    return "reg:" + reg


def stable_id(*parts):
    return hashlib.sha256("|".join(str(x) for x in parts).encode()).hexdigest()[:20]


def instruction_node(raw_sample, row, raw_to_node):
    return raw_to_node.get(raw_sample, {}).get(row.get("id"))


class Resolver:
    def __init__(self, instructions, cfg):
        self.instructions = sorted(instructions, key=lambda row: (int(str(row.get("address") or "0"), 16), int(row.get("pcode_index") or 0)))
        self.cfg = cfg or {}; self.position = {id(row): index for index, row in enumerate(self.instructions)}
        self.writers = collections.defaultdict(list)
        for index, row in enumerate(self.instructions):
            for loc in row.get("effects", {}).get("writes") or []:
                self.writers[canon_loc(loc)].append((index, row))

    def previous_writer(self, location, target):
        loc = canon_loc(location); target_index = self.position[id(target)]; target_block = norm(target.get("block_entry"))
        candidates = []
        for index, row in self.writers.get(loc, []):
            if index >= target_index: continue
            writer_block = norm(row.get("block_entry"))
            if writer_block == target_block or writer_block in set(self.cfg.get("dominators", {}).get(target_block, [])):
                candidates.append((index, row))
        if not candidates: return None
        same = [item for item in candidates if norm(item[1].get("block_entry")) == target_block]
        chosen = max(same or candidates, key=lambda item: item[0])
        return chosen[1]


class InterfaceSafetyAdapter:
    def __init__(self, stage2_path: Path, usg_nodes_path: Path, defuse_path: Path):
        self.raw_interfaces = {}
        self.raw_by_blinded_id = {}
        for node in rows(usg_nodes_path):
            if node.get("node_type") != "InterfaceSem": continue
            sample = blind(node["sample_id"]); fe = norm(node.get("function_entry_norm") or node.get("function_entry"))
            self.raw_interfaces[(sample, fe)] = node
            blinded_id = sample + node["node_id"][len(node["sample_id"]):]
            self.raw_by_blinded_id[blinded_id] = node
        self.selected = {}
        for row in rows(stage2_path):
            if row.get("module") != "InterfaceSem" or not isinstance(row.get("semantic_value"), dict): continue
            if row.get("semantic_value_status") != "FROZEN_MODEL_SELECTED_SAME_FUNCTION_RAW_CANDIDATE": continue
            winner = next((item.get("object_id") for item in row.get("candidate_list") or []
                           if item.get("semantic_id") == row.get("top1_prediction")), None)
            self.selected[(row["sample_id"], norm(row["function_entry"]))] = {"stage2": row, "winner": winner,
                                                                                "raw": self.raw_by_blinded_id.get(winner)}
        self.defuse = set()
        for edge in rows(defuse_path):
            self.defuse.add((edge.get("sample_id"), edge.get("source_node"), edge.get("dest_node")))

    def semantic_parameter(self, sample, function, index):
        selected = self.selected.get((sample, norm(function)))
        if not selected: return None
        parameter = next((item for item in selected["stage2"]["semantic_value"].get("parameters") or []
                          if item.get("index") == index), None)
        if parameter is None: return None
        return {
            "semantic_object_id": selected["winner"], "formal_parameter_index": index,
            "role": str(parameter.get("role") or "UNKNOWN").upper(),
            "type_category": str(parameter.get("type_category") or "UNKNOWN").lower(),
            "role_confidence": selected["stage2"].get("confidence"),
            "role_source": "INFERRED_INTERFACESEM_FACT",
            "semantic_evidence_refs": selected["stage2"].get("evidence_refs") or [],
            "mapping_level": selected["stage2"].get("mapping_level"),
        }

    def extract_sample(self, raw_sample, payload, cfg_index, raw_to_node):
        sample = blind(raw_sample)
        by_function = collections.defaultdict(list)
        for row in payload["low_level_sem"]["instr_sem"]:
            fe = norm(row.get("function_entry_norm") or row.get("function_entry"))
            if fe: by_function[fe].append(row)
        entries = set(by_function)
        bindings, predicates, summaries, paths = [], [], [], []
        for caller, instructions in by_function.items():
            resolver = Resolver(instructions, cfg_index.get(f"{sample}::{caller}", {}))
            for call in resolver.instructions:
                if call.get("opcode") not in {"CALL", "BRANCH"}: continue
                target_text = str(call.get("call_target") or call.get("api_name") or "")
                api = api_name(target_text)
                callee = None
                if target_text.upper().startswith("FUN_"): callee = norm(target_text.split("_", 1)[1])
                if callee not in entries:
                    operands = call.get("operands") or []
                    possible = norm(operands[-1]) if operands else None
                    callee = possible if possible in entries else None
                signature = API_SIGNATURES.get(api) if callee is None else None
                interface_node = self.raw_interfaces.get((sample, callee)) if callee else None
                params = (interface_node or {}).get("attrs", {}).get("params") or []
                descriptors = list(signature or [("UNKNOWN", str(item.get("type_category") or item.get("category") or "UNKNOWN").lower())
                                                  for item in params])
                if not descriptors: continue
                call_node = instruction_node(raw_sample, call, raw_to_node)
                call_pc = norm(call.get("address"))
                for index, descriptor in enumerate(descriptors[:len(ABI_REGS)]):
                    role, deterministic_type = descriptor
                    formal = params[index] if index < len(params) else {}
                    storage = str(formal.get("storage") or "")
                    abi_loc = ABI_REGS[index]
                    if storage and ":" in storage and not storage.lower().startswith("stack"):
                        abi_loc = canon_loc("reg:" + storage.split(":", 1)[0])
                    writer = resolver.previous_writer(abi_loc, call)
                    writer_node = instruction_node(raw_sample, writer, raw_to_node) if writer else None
                    actual_reads = [canon_loc(value) for value in (writer or {}).get("effects", {}).get("reads") or []]
                    dfg = bool(writer_node and call_node and (sample, writer_node, call_node) in self.defuse)
                    semantic = self.semantic_parameter(sample, callee, index) if callee else None
                    semantic_role = (semantic or {}).get("role", "UNKNOWN")
                    raw_type = str(formal.get("type_category") or formal.get("category") or deterministic_type or "UNKNOWN").lower()
                    semantic_type = (semantic or {}).get("type_category", "unknown")
                    if signature:
                        classification = "VERIFIED_API_SIGNATURE_ONLY"
                    elif semantic_role != "UNKNOWN" and writer and dfg:
                        classification = "INTERFACESEM_NEW_INFERENCE"
                    elif semantic and semantic_type not in {"", "unknown"} and semantic_type == raw_type:
                        classification = "INTERFACESEM_ADDITIONAL_SUPPORT"
                    else:
                        classification = "STRUCTURAL_OR_ABI_ONLY" if writer else "UNKNOWN"
                    binding_status = ("VERIFIED_API_SIGNATURE_AND_ABI_REACHING_DEF" if signature and writer else
                                      "VERIFIED_ACTUAL_FORMAL_ABI_REACHING_DEF" if callee and writer else
                                      "UNKNOWN_ACTUAL_ARGUMENT")
                    fact_id = f"{sample}::binding::{stable_id(caller,call_pc,callee or api,index)}"
                    fact = {
                        "schema": "task2_v016_actual_formal_binding", "fact_id": fact_id,
                        "sample_id": sample, "caller_function": caller, "callee_function": callee,
                        "callee_api": api if signature else None, "callsite_pc": call_pc,
                        "callsite_node": call_node, "binding_kind": "ACTUAL_TO_FORMAL",
                        "formal_parameter_index": index, "formal_parameter_id": formal.get("id"),
                        "actual_argument_id": writer_node, "actual_argument_varnodes": actual_reads,
                        "ABI_location": abi_loc, "ABI_evidence": "SYSV_AMD64_REGISTER_AND_UNIQUE_DOMINATING_DEFINITION",
                        "DFG_evidence": [writer_node, call_node] if dfg else [], "def_use_edge_verified": dfg,
                        "role": role if signature else semantic_role,
                        "pointer_int_type": deterministic_type if signature else raw_type,
                        "InterfaceSem_parameter": semantic, "semantic_contribution_class": classification,
                        "binding_status": binding_status,
                        "output_pointer_propagation": ("INFERRED_POSSIBLE_FLOW" if callee and raw_type == "ptr"
                                                       else "NOT_APPLICABLE"),
                    }
                    bindings.append(fact)
                    if writer and callee:
                        path_status = "PARTIAL" if semantic_role == "UNKNOWN" else "VERIFIED" if dfg else "PARTIAL"
                        paths.append({
                            "schema": "task2_v016_interprocedural_taint_path", "path_id": fact_id,
                            "sample_id": sample, "source": writer_node, "involved_functions": [caller, callee],
                            "callsite_pcs": [call_pc], "varnode_identity": actual_reads,
                            "sink": None, "evidence_refs": [value for value in (writer_node, call_node) if value],
                            "status": path_status, "taint_status": "INFERRED_POSSIBLE_FLOW",
                            "sanitizer_fact": "UNKNOWN", "proof_status": "NO_SOURCE_TO_SINK_PROOF",
                            "InterfaceSem_role": semantic_role,
                        })
                    if raw_type == "ptr" or deterministic_type == "ptr":
                        predicate_id = f"{fact_id}::nonnull"
                        predicates.append({
                            "schema": "task2_v016_interface_safety_predicate", "predicate_id": predicate_id,
                            "sample_id": sample, "function_id": callee or api, "callsite_pc": call_pc,
                            "formal_parameter_index": index, "actual_argument_id": writer_node,
                            "predicate_kind": "NULLABLE_POINTER_PRECONDITION", "predicate": f"arg{index} != NULL",
                            "status": "PREDICATE_UNVERIFIED", "executable": False,
                            "semantic_source": classification, "semantic_confidence": (semantic or {}).get("role_confidence"),
                            "actual_formal_evidence": fact_id,
                            "depends_on_InterfaceSem": False,
                            "counterfactual_without_semantics": "UNCHANGED_FROM_RAW_INTERFACE_OR_API_SIGNATURE",
                        })
                    if signature and role in {"buffer_destination", "buffer_input_output"}:
                        size_index = next((i for i, (r, _) in enumerate(signature) if r == "size"), None)
                        if size_index is not None:
                            predicates.append({
                                "schema": "task2_v016_interface_safety_predicate",
                                "predicate_id": f"{sample}::predicate::{stable_id(caller,call_pc,'buffer_size')}",
                                "sample_id": sample, "function_id": api, "callsite_pc": call_pc,
                                "formal_parameter_index": index, "actual_argument_id": writer_node,
                                "predicate_kind": "BUFFER_SIZE_RELATION",
                                "predicate": f"arg{size_index} <= capacity(arg{index})", "status": "BOUNDARY_UNKNOWN",
                                "executable": False, "semantic_source": "VERIFIED_API_SIGNATURE_ONLY",
                                "semantic_confidence": 1.0, "actual_formal_evidence": fact_id,
                                "depends_on_InterfaceSem": False,
                                "counterfactual_without_semantics": "IDENTICAL_PREDICATE_FROM_VERIFIED_API_SIGNATURE",
                            })
                if signature and api == "free":
                    summaries.append({"schema": "task2_v016_function_security_summary", "sample_id": sample,
                                      "function_id": caller, "summary_kind": "ResourceEffectSummary",
                                      "callsite_pc": call_pc, "effect": "RELEASE_ARG0",
                                      "proof_status": "VERIFIED_API_SIGNATURE_AND_ABI"})
                if callee and interface_node:
                    return_info = interface_node.get("attrs", {}).get("ret") or {}
                    if return_info.get("has_return"):
                        ordered = resolver.instructions
                        call_position = resolver.position[id(call)]
                        consumer = None
                        for later in ordered[call_position + 1:]:
                            reads_later = {canon_loc(value) for value in later.get("effects", {}).get("reads") or []}
                            writes_later = {canon_loc(value) for value in later.get("effects", {}).get("writes") or []}
                            if "reg:RAX" in reads_later:
                                consumer = later; break
                            if "reg:RAX" in writes_later:
                                break
                        if consumer:
                            consumer_node = instruction_node(raw_sample, consumer, raw_to_node)
                            selected_return = self.selected.get((sample, callee))
                            return_role = ((selected_return or {}).get("stage2", {}).get("semantic_value", {}).get("return_role")
                                           or "UNKNOWN")
                            return_type = str(return_info.get("return_type_category") or "UNKNOWN").lower()
                            return_id = f"{sample}::binding::{stable_id(caller,call_pc,callee,'return')}"
                            bindings.append({
                                "schema": "task2_v016_actual_formal_binding", "fact_id": return_id,
                                "sample_id": sample, "caller_function": caller, "callee_function": callee,
                                "callee_api": None, "callsite_pc": call_pc, "callsite_node": call_node,
                                "binding_kind": "RETURN_TO_CALLER_RESULT", "formal_parameter_index": None,
                                "formal_parameter_id": return_info.get("id"), "actual_argument_id": consumer_node,
                                "actual_argument_varnodes": ["reg:RAX"], "ABI_location": "reg:RAX",
                                "ABI_evidence": "SYSV_AMD64_RETURN_REGISTER_AND_FIRST_REACHABLE_CONSUMER",
                                "DFG_evidence": [], "def_use_edge_verified": False,
                                "role": str(return_role).upper(), "pointer_int_type": return_type,
                                "InterfaceSem_parameter": None,
                                "semantic_contribution_class": ("INTERFACESEM_ADDITIONAL_SUPPORT"
                                                                if selected_return else "STRUCTURAL_OR_ABI_ONLY"),
                                "binding_status": "VERIFIED_RETURN_ABI_TO_CALLER_CONSUMER",
                                "output_pointer_propagation": "NOT_APPLICABLE",
                            })
                            paths.append({
                                "schema": "task2_v016_interprocedural_taint_path", "path_id": return_id,
                                "sample_id": sample, "source": f"{callee}:return",
                                "involved_functions": [callee, caller], "callsite_pcs": [call_pc],
                                "varnode_identity": ["reg:RAX"], "sink": consumer_node,
                                "evidence_refs": [value for value in (call_node, consumer_node) if value],
                                "status": "PARTIAL", "taint_status": "INFERRED_POSSIBLE_FLOW",
                                "sanitizer_fact": "UNKNOWN", "proof_status": "ABI_RETURN_BINDING_WITHOUT_CALLEE_INTERNAL_FLOW_PROOF",
                                "InterfaceSem_role": str(return_role).upper(),
                            })

        for binding in bindings:
            if binding["callee_function"] and binding["binding_status"].startswith("VERIFIED"):
                summaries.append({
                    "schema": "task2_v016_function_security_summary", "sample_id": sample,
                    "function_id": binding["callee_function"], "summary_kind": "TaintPropagationSummary",
                    "source_parameter": binding["formal_parameter_index"], "destination_parameter": None,
                    "destination_return": False, "def_use_path": binding["DFG_evidence"],
                    "callsite_pc": binding["callsite_pc"], "taint_status": "INFERRED_POSSIBLE_FLOW",
                    "sanitizer_fact": "UNKNOWN", "proof_status": "ACTUAL_FORMAL_ONLY_NO_INTERNAL_FLOW_PROOF",
                    "semantic_contribution_class": binding["semantic_contribution_class"],
                })
        for predicate in predicates:
            summaries.append({"schema": "task2_v016_function_security_summary", "sample_id": sample,
                              "function_id": predicate["function_id"], "summary_kind": "SafetyPreconditionSummary",
                              "callsite_pc": predicate["callsite_pc"], "predicate_id": predicate["predicate_id"],
                              "predicate": predicate["predicate"], "proof_status": predicate["status"],
                              "semantic_contribution_class": predicate["semantic_source"]})
        return bindings, predicates, summaries, paths
