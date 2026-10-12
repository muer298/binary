#!/usr/bin/env python3
"""Gold-blind actual/formal call-boundary binding prototype for SERAPH Task2 v0.1.6.

This sidecar deliberately separates deterministic Ghidra/P-code/ABI evidence from
InterfaceSem.  It does not construct or claim verified interprocedural taint paths.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable


ROOT = Path("/data/chaoni/mjy/resym_exp")
V010 = Path(os.environ.get("SERAPH_SAMPLE_ROOT", str(ROOT / "training_runs/seraph_task2/v0_1_0")))
V011 = Path(os.environ.get("SERAPH_SEMANTIC_ROOT", str(ROOT / "training_runs/seraph_task2/v0_1_1")))
DEFAULT_OUTPUT = ROOT / "training_runs/seraph_task2/v0_1_6_binding_prototype/actual_formal_binding_smoke.jsonl"
DEFAULT_REPORT = ROOT / "training_runs/seraph_task2/v0_1_6_binding_prototype/SMALL_BINDING_PROTOTYPE_REPORT.md"

FIXED_SMOKE_SAMPLES = (
    "JULIET_1cbb51fd8db3aac0cac1",
    "JULIET_31c5cd2d98c4063b3284",
    "JULIET_5d3f7da89ba3dedea0dd",
    "JULIET_93674a72d78d5311bb84",
    "JULIET_e9db8b465d9319fee488",
)

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
    "malloc": (("size", "int"),),
    "calloc": (("count", "int"), ("size", "int")),
    "realloc": (("resource_input", "ptr"), ("size", "int")),
    "free": (("resource_release", "ptr"),),
    "system": (("command_sink", "ptr"),),
    "popen": (("command_sink", "ptr"), ("mode", "ptr")),
    "execv": (("command_sink", "ptr"), ("argv", "ptr")),
    "execve": (("command_sink", "ptr"), ("argv", "ptr"), ("envp", "ptr")),
}


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def blind(value: str) -> str:
    return "JULIET_" + hashlib.sha256(value.encode()).hexdigest()[:20]


def norm(value: Any) -> str | None:
    try:
        return format(int(str(value), 16), "x")
    except (TypeError, ValueError):
        return None


def stable_id(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def canon_loc(value: Any) -> str:
    text = str(value or "")
    if not text.startswith("reg:"):
        return text
    register = text[4:].upper()
    families = {
        "RAX": {"RAX", "EAX", "AX", "AL", "AH"},
        "RBX": {"RBX", "EBX", "BX", "BL", "BH"},
        "RCX": {"RCX", "ECX", "CX", "CL", "CH"},
        "RDX": {"RDX", "EDX", "DX", "DL", "DH"},
        "RSI": {"RSI", "ESI", "SI", "SIL"},
        "RDI": {"RDI", "EDI", "DI", "DIL"},
        "RSP": {"RSP", "ESP", "SP", "SPL"},
        "RBP": {"RBP", "EBP", "BP", "BPL"},
    }
    for full, aliases in families.items():
        if register in aliases:
            return "reg:" + full
    if register.startswith("R") and register[-1:] in {"B", "W", "D"} and register[1:-1].isdigit():
        return "reg:" + register[:-1]
    return "reg:" + register


def instruction_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(str(row.get("address") or "0"), 16),
        int(row.get("pcode_index") or 0),
        int(row.get("seq") or 0),
    )


def direct_internal_target(call: dict[str, Any], entries: set[str]) -> str | None:
    """Resolve only exact Ghidra direct targets; never use fuzzy names/addresses."""
    target = str(call.get("call_target") or call.get("callee") or "")
    if target.upper().startswith("FUN_"):
        result = norm(target.split("_", 1)[1])
        return result if result in entries else None
    return None


def api_name(value: Any) -> str:
    result = str(value or "").lower().lstrip("_")
    return result[8:] if result.startswith("isoc99_") else result


def formal_abi_location(formal: dict[str, Any], index: int) -> tuple[str | None, str]:
    storage = str(formal.get("storage") or formal.get("storage_id") or "")
    if storage and not storage.lower().startswith("stack"):
        if storage.lower().startswith("reg:"):
            location = canon_loc(storage)
        else:
            token = storage.split(":", 1)[0]
            location = canon_loc("reg:" + token)
        if location.startswith("reg:"):
            return location, "RAW_FORMAL_STORAGE"
    if index < len(ABI_REGS):
        return ABI_REGS[index], "SYSV_AMD64_POSITION"
    return None, "UNSUPPORTED_STACK_FORMAL"


class ReachingDefinitionResolver:
    """Resolve an exact boundary definition without pretending all prior writes are unique."""

    def __init__(self, instructions: list[dict[str, Any]], cfg: dict[str, Any]):
        self.instructions = sorted(instructions, key=instruction_key)
        self.position = {id(row): index for index, row in enumerate(self.instructions)}
        self.cfg = cfg or {}
        self.dominators = self.cfg.get("dominators") or {}
        self.writers: dict[str, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
        for index, row in enumerate(self.instructions):
            for location in row.get("effects", {}).get("writes") or []:
                self.writers[canon_loc(location)].append((index, row))

    def resolve(self, location: str, target: dict[str, Any]) -> dict[str, Any]:
        location = canon_loc(location)
        target_index = self.position[id(target)]
        target_block = norm(target.get("block_entry"))
        candidates = []
        target_dominators = set(self.dominators.get(target_block, []))
        for index, row in self.writers.get(location, []):
            if index >= target_index:
                continue
            block = norm(row.get("block_entry"))
            if block == target_block or block in target_dominators:
                candidates.append((index, row, block))
        if not candidates:
            return {"status": "NO_REACHING_DEFINITION", "writer": None, "candidate_count": 0}

        same_block = [item for item in candidates if item[2] == target_block]
        if same_block:
            chosen = max(same_block, key=lambda item: item[0])
            return {
                "status": "UNIQUE_LATEST_SAME_BLOCK_DEFINITION",
                "writer": chosen[1],
                "candidate_count": len(candidates),
                "selected_block": chosen[2],
            }

        by_block: dict[str, list[tuple[int, dict[str, Any], str]]] = collections.defaultdict(list)
        for item in candidates:
            if item[2] is not None:
                by_block[item[2]].append(item)
        if not by_block:
            return {"status": "AMBIGUOUS_REACHING_DEFINITION", "writer": None,
                    "candidate_count": len(candidates)}
        depths = {block: len(set(self.dominators.get(block, []))) for block in by_block}
        best_depth = max(depths.values())
        best_blocks = [block for block, depth in depths.items() if depth == best_depth]
        if len(best_blocks) != 1:
            return {"status": "AMBIGUOUS_REACHING_DEFINITION", "writer": None,
                    "candidate_count": len(candidates), "candidate_blocks": sorted(best_blocks)}
        block = best_blocks[0]
        chosen = max(by_block[block], key=lambda item: item[0])
        return {
            "status": "UNIQUE_NEAREST_DOMINATOR_DEFINITION",
            "writer": chosen[1],
            "candidate_count": len(candidates),
            "selected_block": block,
        }

    def first_read_before_overwrite(self, location: str, call: dict[str, Any]) -> dict[str, Any] | None:
        location = canon_loc(location)
        start = self.position[id(call)]
        for later in self.instructions[start + 1:]:
            reads = {canon_loc(value) for value in later.get("effects", {}).get("reads") or []}
            writes = {canon_loc(value) for value in later.get("effects", {}).get("writes") or []}
            if location in reads:
                return later
            if location in writes:
                return None
        return None


def load_interface_roles(stage2_path: Path, selected_samples: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in rows(stage2_path):
        if row.get("sample_id") not in selected_samples or row.get("module") != "InterfaceSem":
            continue
        if row.get("semantic_value_status") != "FROZEN_MODEL_SELECTED_SAME_FUNCTION_RAW_CANDIDATE":
            continue
        if not isinstance(row.get("semantic_value"), dict):
            continue
        result[(row["sample_id"], norm(row.get("function_entry")))] = row["semantic_value"]
    return result


def parameter_role(roles: dict[tuple[str, str], dict[str, Any]], sample: str,
                   function: str, index: int) -> tuple[str, str]:
    value = roles.get((sample, function))
    if not value:
        return "UNKNOWN", "NO_SELECTED_INTERFACESEM_ROLE"
    parameter = next((item for item in value.get("parameters") or [] if item.get("index") == index), None)
    if not parameter:
        return "UNKNOWN", "NO_EXPORTED_INTERFACESEM_PARAMETER_ROLE"
    role = str(parameter.get("role") or "UNKNOWN").upper()
    return role, "FROZEN_INTERFACESEM_ROLE" if role != "UNKNOWN" else "FROZEN_INTERFACESEM_ROLE_UNKNOWN"


def return_role(roles: dict[tuple[str, str], dict[str, Any]], sample: str,
                function: str) -> tuple[str, str]:
    value = roles.get((sample, function))
    if not value:
        return "UNKNOWN", "NO_SELECTED_INTERFACESEM_RETURN_ROLE"
    role = str(value.get("return_role") or "UNKNOWN").upper()
    return role, "FROZEN_INTERFACESEM_RETURN_ROLE" if role != "UNKNOWN" else "FROZEN_INTERFACESEM_RETURN_ROLE_UNKNOWN"


def raw_node_id(raw_sample: str, row: dict[str, Any] | None,
                raw_to_node: dict[str, dict[str, str]]) -> str | None:
    if row is None:
        return None
    return raw_to_node.get(raw_sample, {}).get(row.get("id"))


def build_sample_bindings(
    raw_sample: str,
    payload: dict[str, Any],
    interfaces: dict[tuple[str, str], dict[str, Any]],
    roles: dict[tuple[str, str], dict[str, Any]],
    cfg_index: dict[str, Any],
    raw_to_node: dict[str, dict[str, str]],
    defuse: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    sample = blind(raw_sample)
    by_function: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in payload["low_level_sem"]["instr_sem"]:
        function = norm(row.get("function_entry_norm") or row.get("function_entry"))
        if function:
            by_function[function].append(row)
    entries = set(by_function)
    bindings = []

    for caller, instructions in sorted(by_function.items()):
        resolver = ReachingDefinitionResolver(instructions, cfg_index.get(f"{sample}::{caller}", {}))
        for call in resolver.instructions:
            if call.get("opcode") != "CALL":
                continue
            callee = direct_internal_target(call, entries)
            signature_name = None
            signature = None
            if callee is None:
                signature_name = api_name(call.get("call_target") or call.get("callee") or call.get("api_name"))
                signature = API_SIGNATURES.get(signature_name)
                if signature is None:
                    continue
                attrs = {"params": [
                    {"id": f"api::{signature_name}::arg{index}", "param_index": index,
                     "type_category": type_category, "storage": ABI_REGS[index],
                     "deterministic_role": role}
                    for index, (role, type_category) in enumerate(signature)
                ], "ret": {"has_return": False}}
                callee_identity = f"API::{signature_name}"
            else:
                interface = interfaces.get((sample, callee))
                if interface is None:
                    continue
                attrs = interface.get("attrs") or {}
                callee_identity = callee
            call_pc = norm(call.get("address"))
            call_node = raw_node_id(raw_sample, call, raw_to_node)
            argument_locs = {canon_loc(value) for value in call.get("argument_locs") or []}
            read_set = {canon_loc(value) for value in call.get("read_set") or []}

            for ordinal, formal in enumerate(attrs.get("params") or []):
                index = formal.get("param_index", formal.get("param_id", ordinal))
                try:
                    index = int(index)
                except (TypeError, ValueError):
                    index = ordinal
                abi_location, abi_source = formal_abi_location(formal, index)
                role, role_source = (("UNKNOWN", "NOT_INTERFACESEM_API_SIGNATURE") if signature else
                                     parameter_role(roles, sample, callee, index))
                deterministic_role = formal.get("deterministic_role") if signature else None
                common = {
                    "schema": "task2_v016_small_binding_sidecar_v1",
                    "sample_id": sample,
                    "binding_kind": "CALLER_ACTUAL_TO_CALLEE_FORMAL",
                    "caller": caller,
                    "callee": callee_identity,
                    "callsite_pc": call_pc,
                    "callsite_node": call_node,
                    "actual_varnode": abi_location,
                    "actual_index": index,
                    "formal_index": index,
                    "formal_parameter_id": formal.get("id"),
                    "formal_type_category": str(formal.get("type_category") or formal.get("category") or "UNKNOWN").lower(),
                    "interface_role": role,
                    "interface_role_source": role_source,
                    "deterministic_parameter_role": deterministic_role,
                    "InterfaceSem_unique_contribution": False,
                    "semantic_source": ("DETERMINISTIC_API_SIGNATURE" if signature else "STRUCTURAL_ABI_PCODE"),
                    "taint_path_status": "NOT_VERIFIED_NO_INTERPROCEDURAL_DFG_PROOF",
                }
                if abi_location is None:
                    common.update({
                        "binding_id": f"{sample}::sidecar::{stable_id(caller, callee_identity, call_pc, index, 'stack')}",
                        "evidence_type": ["GHIDRA_DIRECT_CALL_TARGET", "RAW_FORMAL_PARAMETER_INDEX"],
                        "evidence_refs": [value for value in (call_node, formal.get("id")) if value],
                        "binding_status": "UNRESOLVED_UNSUPPORTED_STACK_FORMAL",
                        "reaching_definition_status": "NOT_ATTEMPTED",
                        "reaching_definition_node": None,
                        "reaching_definition_input_varnodes": [],
                        "explicit_DFG_edge": False,
                    })
                    bindings.append(common)
                    continue

                resolution = resolver.resolve(abi_location, call)
                writer = resolution.get("writer")
                writer_node = raw_node_id(raw_sample, writer, raw_to_node)
                writer_inputs = sorted({canon_loc(value) for value in (writer or {}).get("effects", {}).get("reads") or []})
                explicit_dfg = bool(writer_node and call_node and (sample, writer_node, call_node) in defuse)
                boundary_visible = abi_location in argument_locs or abi_location in read_set
                verified = bool(writer_node and common["formal_parameter_id"] and boundary_visible and
                                str(resolution["status"]).startswith("UNIQUE_"))
                evidence = ["GHIDRA_DIRECT_CALL_TARGET", "RAW_FORMAL_PARAMETER_INDEX", abi_source]
                if signature:
                    evidence[0] = "GHIDRA_VERIFIED_API_CALL_TARGET"
                    evidence.append("DETERMINISTIC_API_SIGNATURE")
                if boundary_visible:
                    evidence.append("PCODE_CALL_ARGUMENT_LOCATION")
                if writer_node:
                    evidence.append("CFG_DOMINATING_PCODE_DEFINITION")
                if explicit_dfg:
                    evidence.append("EXPLICIT_INTRAFUNCTION_DFG_WRITER_TO_CALL")
                common.update({
                    "binding_id": f"{sample}::sidecar::{stable_id(caller, callee_identity, call_pc, index, abi_location)}",
                    "evidence_type": evidence,
                    "evidence_refs": [value for value in (writer_node, call_node, formal.get("id")) if value],
                    "binding_status": "VERIFIED_CALL_BOUNDARY_BINDING" if verified else "UNRESOLVED_CALL_BOUNDARY_BINDING",
                    "reaching_definition_status": resolution["status"],
                    "reaching_definition_candidate_count": resolution.get("candidate_count", 0),
                    "reaching_definition_node": writer_node,
                    "reaching_definition_opcode": (writer or {}).get("opcode"),
                    "reaching_definition_input_varnodes": writer_inputs,
                    "explicit_DFG_edge": explicit_dfg,
                })
                bindings.append(common)

            ret = attrs.get("ret") or {}
            if not signature and ret.get("has_return"):
                consumer = resolver.first_read_before_overwrite("reg:RAX", call)
                consumer_node = raw_node_id(raw_sample, consumer, raw_to_node)
                role, role_source = return_role(roles, sample, callee)
                call_returns = {canon_loc(value) for value in call.get("return_locs") or []}
                explicit_dfg = bool(call_node and consumer_node and (sample, call_node, consumer_node) in defuse)
                verified = bool("reg:RAX" in call_returns and call_node and consumer_node)
                evidence = ["GHIDRA_DIRECT_CALL_TARGET", "RAW_CALLEE_RETURN_OBJECT", "SYSV_AMD64_RETURN_REGISTER"]
                if consumer_node:
                    evidence.append("FIRST_RAX_READ_BEFORE_OVERWRITE")
                if explicit_dfg:
                    evidence.append("EXPLICIT_INTRAFUNCTION_DFG_CALL_TO_CONSUMER")
                bindings.append({
                    "schema": "task2_v016_small_binding_sidecar_v1",
                    "binding_id": f"{sample}::sidecar::{stable_id(caller, callee, call_pc, 'return')}",
                    "sample_id": sample,
                    "binding_kind": "CALLEE_RETURN_TO_CALLER_RESULT",
                    "caller": caller,
                    "callee": callee,
                    "callsite_pc": call_pc,
                    "callsite_node": call_node,
                    "actual_varnode": "reg:RAX",
                    "actual_index": None,
                    "formal_index": None,
                    "formal_parameter_id": ret.get("id"),
                    "formal_type_category": str(ret.get("return_type_category") or "UNKNOWN").lower(),
                    "caller_result_consumer": consumer_node,
                    "interface_role": role,
                    "interface_role_source": role_source,
                    "InterfaceSem_unique_contribution": False,
                    "semantic_source": "STRUCTURAL_ABI_PCODE",
                    "evidence_type": evidence,
                    "evidence_refs": [value for value in (ret.get("id"), call_node, consumer_node) if value],
                    "binding_status": "VERIFIED_RETURN_BOUNDARY_BINDING" if verified else "UNRESOLVED_RETURN_BOUNDARY_BINDING",
                    "reaching_definition_status": "ABI_RETURN_REGISTER_TO_FIRST_CONSUMER" if verified else "NO_EXACT_CALLER_RESULT_CONSUMER",
                    "explicit_DFG_edge": explicit_dfg,
                    "taint_path_status": "NOT_VERIFIED_NO_CALLEE_INTERNAL_DFG_PROOF",
                })
    return bindings


def identity_audit(bindings: list[dict[str, Any]], known_nodes: set[str]) -> dict[str, Any]:
    ids = [row["binding_id"] for row in bindings]
    verified = [row for row in bindings if row["binding_status"].startswith("VERIFIED_")]
    required_ok = all(
        row.get("caller") and row.get("callee") and row.get("callsite_pc") and
        row.get("actual_varnode") and isinstance(row.get("evidence_type"), list) and
        isinstance(row.get("evidence_refs"), list)
        for row in bindings
    )
    refs_ok = all(
        not row.get("callsite_node") or row["callsite_node"] in known_nodes
        for row in bindings
    ) and all(
        not row.get("reaching_definition_node") or row["reaching_definition_node"] in known_nodes
        for row in bindings
    )
    actual_formal = [row for row in bindings if row["binding_kind"] == "CALLER_ACTUAL_TO_CALLEE_FORMAL"]
    returns = [row for row in bindings if row["binding_kind"] == "CALLEE_RETURN_TO_CALLER_RESULT"]
    formal_ok = all(isinstance(row.get("formal_index"), int) and row["formal_index"] >= 0 and
                    bool(row.get("formal_parameter_id")) for row in actual_formal)
    actual_ok = all(row.get("actual_varnode") and
                    (not row["binding_status"].startswith("VERIFIED_") or row.get("reaching_definition_node"))
                    for row in actual_formal)
    return_ok = all(row.get("formal_index") is None and row.get("actual_varnode") == "reg:RAX" and
                    (not row["binding_status"].startswith("VERIFIED_") or row.get("caller_result_consumer"))
                    for row in returns)
    return {
        "binding_ids_unique": len(ids) == len(set(ids)),
        "required_identity_fields_complete": required_ok,
        "instruction_node_refs_resolve": refs_ok,
        "formal_parameter_identity_consistent": formal_ok,
        "actual_argument_identity_consistent": actual_ok,
        "return_identity_consistent": return_ok,
        "verified_binding_count": len(verified),
        "verified_taint_path_count": sum(row.get("taint_path_status") == "VERIFIED_TAINT_PATH" for row in bindings),
        "nonunknown_InterfaceSem_roles": sum(row.get("interface_role") != "UNKNOWN" for row in bindings),
        "InterfaceSem_unique_contribution_count": sum(bool(row.get("InterfaceSem_unique_contribution")) for row in bindings),
    }


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def write_report(path: Path, samples: list[str], bindings: list[dict[str, Any]], audit: dict[str, Any],
                 elapsed: float) -> None:
    by_kind = collections.Counter(row["binding_kind"] for row in bindings)
    by_status = collections.Counter(row["binding_status"] for row in bindings)
    reaching = collections.Counter(row.get("reaching_definition_status") for row in bindings)
    per_sample = collections.Counter(row["sample_id"] for row in bindings)
    explicit = sum(bool(row.get("explicit_DFG_edge")) for row in bindings)
    lines = [
        "# Small Binding Prototype Report", "",
        "This smoke run is Gold-blind. It uses only exact Ghidra direct-call identity, raw formal objects, SysV AMD64 ABI locations, P-code varnodes, CFG dominance, and frozen InterfaceSem roles.", "",
        "## Scope", "",
        f"- Fixed samples: {len(samples)}",
        f"- Total elapsed seconds: {elapsed:.6f}",
        "- Candidate generation: not run",
        "- Full 100-sample experiment: not run",
        "- Task1 frozen artifacts: read-only",
        "- Gold: not read", "",
        "## Results", "",
        f"- Binding rows: {len(bindings)}",
        f"- Actual-to-formal rows: {by_kind.get('CALLER_ACTUAL_TO_CALLEE_FORMAL', 0)}",
        f"- Return-to-caller rows: {by_kind.get('CALLEE_RETURN_TO_CALLER_RESULT', 0)}",
        f"- Verified call-boundary rows: {by_status.get('VERIFIED_CALL_BOUNDARY_BINDING', 0)}",
        f"- Verified return-boundary rows: {by_status.get('VERIFIED_RETURN_BOUNDARY_BINDING', 0)}",
        f"- Unresolved rows: {sum(count for status, count in by_status.items() if status.startswith('UNRESOLVED_'))}",
        f"- Explicit intrafunction DFG edges attached: {explicit}",
        f"- VERIFIED_TAINT_PATH rows: {audit['verified_taint_path_count']}",
        f"- Non-UNKNOWN InterfaceSem roles: {audit['nonunknown_InterfaceSem_roles']}",
        f"- InterfaceSem unique contributions: {audit['InterfaceSem_unique_contribution_count']}", "",
        "## Identity checks", "",
        f"- Binding IDs unique: {audit['binding_ids_unique']}",
        f"- Required identity fields complete: {audit['required_identity_fields_complete']}",
        f"- Instruction node references resolve: {audit['instruction_node_refs_resolve']}", "",
        f"- Formal parameter identity consistent: {audit['formal_parameter_identity_consistent']}",
        f"- Actual argument identity consistent: {audit['actual_argument_identity_consistent']}",
        f"- Return identity consistent: {audit['return_identity_consistent']}", "",
        "## Reaching-definition statuses", "",
    ]
    lines.extend(f"- `{status}`: {count}" for status, count in sorted(reaching.items(), key=lambda item: str(item[0])))
    lines.extend(["", "## Per-sample rows", ""])
    lines.extend(f"- `{sample}`: {per_sample.get(sample, 0)}" for sample in samples)
    lines.extend([
        "", "## Interpretation", "",
        "`VERIFIED_*_BOUNDARY_BINDING` means only that the call boundary identity is supported by the listed structural evidence. It does not mean that a taint path through the callee has been proven.", "",
        "ABI position is always recorded as deterministic structural evidence and is never counted as an InterfaceSem contribution. Frozen UNKNOWN roles remain UNKNOWN.", "",
        "Final status: `READY_FOR_SMALL_COUNTERFACTUAL_TEST`" if all((
            audit["binding_ids_unique"], audit["required_identity_fields_complete"],
            audit["instruction_node_refs_resolve"], audit["formal_parameter_identity_consistent"],
            audit["actual_argument_identity_consistent"], audit["return_identity_consistent"],
            audit["verified_taint_path_count"] == 0,
            audit["InterfaceSem_unique_contribution_count"] == 0, len(bindings) > 0,
        )) else "Final status: `BLOCKED_BINDING_PROTOTYPE`",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(samples: list[str], output: Path, report: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.time()
    selected = set(samples)
    index = {}
    for item in rows(V010 / "juliet_usg_final/usg_index.jsonl"):
        sample = blind(item["sample_id"])
        if sample in selected:
            index[sample] = item
    missing = selected - set(index)
    if missing:
        raise RuntimeError(f"fixed smoke sample(s) absent: {sorted(missing)}")

    interfaces = {}
    raw_to_node: dict[str, dict[str, str]] = collections.defaultdict(dict)
    known_nodes = set()
    for node in rows(V010 / "juliet_usg_final/usg_nodes.jsonl"):
        sample = blind(node["sample_id"])
        if sample not in selected:
            continue
        blinded_id = sample + node["node_id"][len(node["sample_id"]):]
        known_nodes.add(blinded_id)
        if node.get("node_type") == "InterfaceSem":
            function = norm(node.get("function_entry_norm") or node.get("function_entry"))
            interfaces[(sample, function)] = node
        elif node.get("node_type") == "InstrSem":
            raw_to_node[node["sample_id"]][node["raw_id"]] = blinded_id

    roles = load_interface_roles(V011 / "juliet_stage2_semantics_v2.jsonl", selected)
    cfg_index = json.loads((V011 / "cfg_dominance_index.json").read_text())["functions"]
    defuse = {
        (row["sample_id"], row["source_node"], row["dest_node"])
        for row in rows(V011 / "def_use_edges.jsonl") if row.get("sample_id") in selected
    }

    bindings = []
    for sample in samples:
        item = index[sample]
        payload = json.loads(Path(item["json_path"]).read_text())
        bindings.extend(build_sample_bindings(item["sample_id"], payload, interfaces, roles,
                                              cfg_index, raw_to_node, defuse))
    bindings.sort(key=lambda row: (row["sample_id"], row["caller"], row["callsite_pc"],
                                   row["binding_kind"], -1 if row["formal_index"] is None else row["formal_index"]))
    audit = identity_audit(bindings, known_nodes)
    write_jsonl(output, bindings)
    write_report(report, samples, bindings, audit, time.time() - started)
    return bindings, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", default=list(FIXED_SMOKE_SAMPLES))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generated, checks = run(arguments.samples, arguments.output, arguments.report)
    print(json.dumps({"bindings": len(generated), "identity_audit": checks}, sort_keys=True))
