#!/usr/bin/env python3
"""Gold-blind interprocedural evidence utilities for SERAPH Task2 v0.1.8.

The module deliberately keeps four evidence classes separate:
  A. deterministic N-pcode/CFG/DFG,
  B. deterministic ABI/API contracts,
  C. frozen predicted semantics, and
  D. verified semantic increments.

No function in this module opens the Juliet Gold file.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/data/chaoni/mjy/resym_exp")
V010 = Path(os.environ.get("SERAPH_SAMPLE_ROOT", str(ROOT / "training_runs/seraph_task2/v0_1_0")))
V011 = Path(os.environ.get("SERAPH_SEMANTIC_ROOT", str(ROOT / "training_runs/seraph_task2/v0_1_1")))
V012 = Path(os.environ.get("SERAPH_EVIDENCE_ROOT", str(ROOT / "training_runs/seraph_task2/v0_1_2")))
V017 = Path(os.environ.get("SERAPH_BINDING_ROOT", str(ROOT / "training_runs/seraph_task2/v0_1_7")))
SCC_ITERATION_LIMIT = 8
ABI_REGS = ("reg:RDI", "reg:RSI", "reg:RDX", "reg:RCX", "reg:R8", "reg:R9")


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
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()[:20]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


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
        "R8": {"R8", "R8D", "R8W", "R8B"},
        "R9": {"R9", "R9D", "R9W", "R9B"},
    }
    for full, aliases in families.items():
        if register in aliases:
            return "reg:" + full
    if register.startswith("R") and register[-1:] in {"B", "W", "D"} and register[1:-1].isdigit():
        return "reg:" + register[:-1]
    return "reg:" + register


def instruction_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(str(row.get("address") or "0"), 16),
            int(row.get("pcode_index") or 0), int(row.get("seq") or 0))


def direct_target(call: dict[str, Any], entries: set[str]) -> str | None:
    target = str(call.get("call_target") or call.get("callee") or "")
    if not target.upper().startswith("FUN_"):
        return None
    result = norm(target.split("_", 1)[1])
    return result if result in entries else None


def memory_operands(row: dict[str, Any]) -> list[str]:
    effects = row.get("effects") or {}
    return sorted({str(value) for value in ((effects.get("reads") or []) + (effects.get("writes") or []))
                   if str(value).startswith("mem[")})


def is_nonstack_memory(row: dict[str, Any]) -> bool:
    return any("RSP" not in value and "RBP" not in value for value in memory_operands(row))


class Corpus:
    """Selected frozen samples plus exact instruction/DFG/CFG identities."""

    def __init__(self, selected_samples: set[str]):
        self.selected_samples = selected_samples
        self.index: dict[str, dict[str, Any]] = {}
        self.raw_to_node: dict[tuple[str, str], str] = {}
        self.instructions: dict[tuple[str, str], dict[str, Any]] = {}
        self.by_function: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = collections.defaultdict(list)
        self.out_edges: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
        self.in_edges: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
        self.cfg = json.loads((V011 / "cfg_dominance_index.json").read_text())["functions"]

        for item in rows(V010 / "juliet_usg_final/usg_index.jsonl"):
            sample = blind(item["sample_id"])
            if sample in selected_samples:
                self.index[sample] = item
        missing = selected_samples - set(self.index)
        if missing:
            raise RuntimeError(f"selected sample(s) absent: {sorted(missing)}")

        for node in rows(V010 / "juliet_usg_final/usg_nodes.jsonl"):
            sample = blind(node["sample_id"])
            if sample not in selected_samples or node.get("node_type") != "InstrSem":
                continue
            blinded_node = sample + node["node_id"][len(node["sample_id"]):]
            self.raw_to_node[(node["sample_id"], node["raw_id"])] = blinded_node

        for sample, item in self.index.items():
            payload = json.loads(Path(item["json_path"]).read_text())
            for instruction in payload["low_level_sem"]["instr_sem"]:
                node = self.raw_to_node[(item["sample_id"], instruction["id"])]
                function = norm(instruction.get("function_entry_norm") or instruction.get("function_entry"))
                self.instructions[(sample, node)] = instruction
                if function:
                    self.by_function[(sample, function)].append((node, instruction))
        for values in self.by_function.values():
            values.sort(key=lambda pair: instruction_key(pair[1]))

        for edge in rows(V011 / "def_use_edges.jsonl"):
            sample = edge.get("sample_id")
            if sample not in selected_samples:
                continue
            self.out_edges[(sample, edge["source_node"])].append(edge)
            self.in_edges[(sample, edge["dest_node"])].append(edge)

        self.bindings = [row for row in rows(V017 / "VERIFIED_BINDING_FACTS.jsonl")
                         if row.get("sample_id") in selected_samples]
        self.internal_bindings = [row for row in self.bindings
                                  if row.get("semantic_source") == "STRUCTURAL_ABI_PCODE"]
        self.api_bindings = [row for row in self.bindings
                             if row.get("semantic_source") == "DETERMINISTIC_API_SIGNATURE"]

    def node(self, raw_sample: str, instruction: dict[str, Any] | None) -> str | None:
        if instruction is None:
            return None
        return self.raw_to_node.get((raw_sample, instruction.get("id")))

    def formal_seed(self, binding: dict[str, Any]) -> str | None:
        location = canon_loc(binding.get("actual_varnode"))
        for node, instruction in self.by_function.get((binding["sample_id"], binding["callee"]), []):
            reads_set = {canon_loc(value) for value in (instruction.get("effects") or {}).get("reads") or []}
            writes_set = {canon_loc(value) for value in (instruction.get("effects") or {}).get("writes") or []}
            if location in reads_set:
                return node
            if location in writes_set:
                return None
        return None

    def shortest_paths(self, sample: str, start: str, limit: int = 4096) -> tuple[dict[str, str | None], dict[str, dict[str, Any]]]:
        parent: dict[str, str | None] = {start: None}
        parent_edge: dict[str, dict[str, Any]] = {}
        queue = collections.deque([start])
        while queue and len(parent) < limit:
            source = queue.popleft()
            for edge in self.out_edges.get((sample, source), []):
                destination = edge["dest_node"]
                if destination in parent:
                    continue
                parent[destination] = source
                parent_edge[destination] = edge
                queue.append(destination)
        return parent, parent_edge

    @staticmethod
    def reconstruct(target: str, parent: dict[str, str | None],
                    parent_edge: dict[str, dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
        nodes = []
        edges = []
        current: str | None = target
        while current is not None:
            nodes.append(current)
            if current in parent_edge:
                edges.append(parent_edge[current])
            current = parent.get(current)
        nodes.reverse()
        edges.reverse()
        return nodes, edges

    def ancestors(self, sample: str, target: str, limit: int = 256) -> tuple[dict[str, str | None], dict[str, dict[str, Any]]]:
        successor: dict[str, str | None] = {target: None}
        successor_edge: dict[str, dict[str, Any]] = {}
        queue = collections.deque([target])
        while queue and len(successor) < limit:
            destination = queue.popleft()
            for edge in self.in_edges.get((sample, destination), []):
                source = edge["source_node"]
                if source in successor:
                    continue
                successor[source] = destination
                successor_edge[source] = edge
                queue.append(source)
        return successor, successor_edge

    def exact_common_memory_origin(self, sample: str, left: str, right: str) -> dict[str, Any] | None:
        left_succ, left_edges = self.ancestors(sample, left)
        right_succ, right_edges = self.ancestors(sample, right)
        left_memory = {(edge.get("object_identity"), edge.get("source_node")): edge
                       for edge in left_edges.values()
                       if str(edge.get("object_identity") or "").startswith("mem[")}
        right_memory = {(edge.get("object_identity"), edge.get("source_node")): edge
                        for edge in right_edges.values()
                        if str(edge.get("object_identity") or "").startswith("mem[")}
        common = sorted(set(left_memory) & set(right_memory), key=lambda item: (item[0], item[1]))
        if not common:
            return None
        object_identity, origin = common[0]

        def chain(start: str, successor: dict[str, str | None], edge_map: dict[str, dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
            nodes = [start]
            edges = []
            current = start
            while successor.get(current) is not None:
                edges.append(edge_map[current])
                current = successor[current]  # type: ignore[assignment]
                nodes.append(current)
            return nodes, edges

        left_nodes, left_path_edges = chain(origin, left_succ, left_edges)
        right_nodes, right_path_edges = chain(origin, right_succ, right_edges)
        return {
            "object_identity": object_identity,
            "origin_node": origin,
            "left_nodes": left_nodes,
            "left_edges": left_path_edges,
            "right_nodes": right_nodes,
            "right_edges": right_path_edges,
        }

    def cfg_reaches(self, sample: str, function: str, source_node: str, destination_node: str) -> bool:
        source = self.instructions.get((sample, source_node))
        destination = self.instructions.get((sample, destination_node))
        if not source or not destination:
            return False
        source_block = norm(source.get("block_entry"))
        destination_block = norm(destination.get("block_entry"))
        if source_block == destination_block:
            return instruction_key(source) < instruction_key(destination)
        cfg = self.cfg.get(f"{sample}::{function}") or {}
        return destination_block in set((cfg.get("reachable_from") or {}).get(source_block, []))

    def evidence_node(self, sample: str, node: str) -> dict[str, Any]:
        instruction = self.instructions[(sample, node)]
        return {
            "node": node,
            "function_entry": norm(instruction.get("function_entry_norm") or instruction.get("function_entry")),
            "pc": norm(instruction.get("address")),
            "pcode_index": int(instruction.get("pcode_index") or 0),
            "opcode": instruction.get("opcode"),
        }


def strongly_connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    result: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in sorted(graph.get(node, set())):
            if target not in indices:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], indices[target])
        if low[node] == indices[node]:
            component = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            result.append(sorted(component))

    nodes = set(graph) | {target for targets in graph.values() for target in targets}
    for node in sorted(nodes):
        if node not in indices:
            visit(node)
    return result


def build_return_bindings(corpus: Corpus) -> list[dict[str, Any]]:
    """Audit return boundaries; verification requires more than shared RAX."""
    results = []
    for sample, item in sorted(corpus.index.items()):
        raw_sample = item["sample_id"]
        entries = {function for (member_sample, function) in corpus.by_function if member_sample == sample}
        for (member_sample, caller), instructions in sorted(corpus.by_function.items()):
            if member_sample != sample:
                continue
            for position, (call_node, call) in enumerate(instructions):
                if call.get("opcode") != "CALL":
                    continue
                callee = direct_target(call, entries)
                if callee is None:
                    continue
                consumer_node = None
                for later_node, later in instructions[position + 1:]:
                    reads_set = {canon_loc(value) for value in (later.get("effects") or {}).get("reads") or []}
                    writes_set = {canon_loc(value) for value in (later.get("effects") or {}).get("writes") or []}
                    if "reg:RAX" in reads_set:
                        consumer_node = later_node
                        break
                    if "reg:RAX" in writes_set:
                        break

                exit_definitions = []
                for return_node, return_instruction in corpus.by_function.get((sample, callee), []):
                    if return_instruction.get("opcode") != "RETURN":
                        continue
                    return_block = norm(return_instruction.get("block_entry"))
                    candidates = []
                    cfg = corpus.cfg.get(f"{sample}::{callee}") or {}
                    dominators = set((cfg.get("dominators") or {}).get(return_block, []))
                    for node, instruction in corpus.by_function.get((sample, callee), []):
                        if instruction_key(instruction) >= instruction_key(return_instruction):
                            continue
                        writes_set = {canon_loc(value) for value in (instruction.get("effects") or {}).get("writes") or []}
                        block = norm(instruction.get("block_entry"))
                        if "reg:RAX" in writes_set and (block == return_block or block in dominators):
                            candidates.append((instruction_key(instruction), node, block))
                    if candidates:
                        candidates.sort()
                        exit_definitions.append({"return_node": return_node, "definition_node": candidates[-1][1],
                                                 "definition_block": candidates[-1][2]})

                explicit_caller_dfg = bool(consumer_node and any(
                    edge["dest_node"] == consumer_node
                    for edge in corpus.out_edges.get((sample, call_node), [])
                ))
                all_exits_defined = bool(exit_definitions) and len(exit_definitions) == sum(
                    instruction.get("opcode") == "RETURN"
                    for _, instruction in corpus.by_function.get((sample, callee), [])
                )
                if explicit_caller_dfg and all_exits_defined:
                    status = "VERIFIED_RETURN_FLOW"
                elif consumer_node and exit_definitions:
                    status = "PARTIAL_RETURN_FLOW"
                else:
                    status = "UNKNOWN"
                results.append({
                    "schema": "task2_v018_interprocedural_return_binding",
                    "binding_id": f"{sample}::return::{stable_id(caller, callee, norm(call.get('address')))}",
                    "sample_id": sample,
                    "caller": caller,
                    "callee": callee,
                    "callsite_pc": norm(call.get("address")),
                    "callsite_node": call_node,
                    "callee_exit_definitions": exit_definitions,
                    "caller_result_consumer": consumer_node,
                    "return_register": "reg:RAX",
                    "explicit_call_to_consumer_DFG": explicit_caller_dfg,
                    "all_callee_exits_have_dominating_RAX_definition": all_exits_defined,
                    "binding_status": status,
                    "interface_role": "UNKNOWN",
                    "semantic_source": ["A_STRUCTURAL_N_PCODE_CFG_DFG", "B_DETERMINISTIC_SYSV_ABI"],
                    "evidence_refs": [value for value in [call_node, consumer_node] if value] +
                                     [item["definition_node"] for item in exit_definitions],
                    "nonverification_reason": (None if status == "VERIFIED_RETURN_FLOW" else
                        "NO_EXPLICIT_CALL_RESULT_DFG_EDGE" if consumer_node and exit_definitions else
                        "MISSING_CALLEE_EXIT_DEFINITION_OR_CALLER_CONSUMER"),
                })
    return sorted(results, key=lambda row: (row["sample_id"], row["caller"], row["callsite_pc"]))


def build_output_pointer_flows(corpus: Corpus) -> list[dict[str, Any]]:
    legacy = list(rows(V012 / "verified_taint_paths.jsonl"))
    legacy_map = {(row["sample_id"], row["function_entry"], row["source_call_pc"], row["source_api"]): row
                  for row in legacy if row.get("verification_status") == "VERIFIED_EXTERNAL_VALUE_TO_SINK_ARGUMENT"}
    results = []
    for binding in corpus.bindings:
        if binding.get("formal_type_category") != "ptr":
            continue
        sample = binding["sample_id"]
        status = "UNKNOWN"
        pointee = None
        callee_write = None
        caller_use = None
        internal_store_paths = []
        if binding.get("semantic_source") == "DETERMINISTIC_API_SIGNATURE":
            api = str(binding["callee"]).split("::", 1)[-1]
            match = legacy_map.get((sample, binding["caller"], binding["callsite_pc"], api))
            if binding.get("deterministic_parameter_role") in {"buffer_output", "buffer_input_output"} and match:
                status = "VERIFIED_OUTPUT_FLOW"
                pointee = (match.get("shared_object_identities") or [None])[0]
                callee_write = "DETERMINISTIC_API_OUTPUT_SIGNATURE"
                source_pc = int(match["source_call_pc"], 16)
                later_loads = [node for node in match.get("evidence_chain") or []
                               if node.get("opcode") == "LOAD" and int(str(node.get("pc") or "0"), 16) > source_pc]
                caller_use = later_loads[0]["node"] if later_loads else None
            elif binding.get("deterministic_parameter_role") in {"buffer_output", "buffer_input_output"}:
                status = "PARTIAL_OUTPUT_FLOW"
                callee_write = "DETERMINISTIC_API_OUTPUT_SIGNATURE_WITHOUT_EXACT_POSTCALL_OBJECT_USE"
        else:
            seed = corpus.formal_seed(binding)
            if seed:
                parent, parent_edge = corpus.shortest_paths(sample, seed)
                for node, instruction in corpus.by_function.get((sample, binding["callee"]), []):
                    if (node in parent and instruction.get("opcode") == "STORE" and
                            is_nonstack_memory(instruction)):
                        nodes, edges = corpus.reconstruct(node, parent, parent_edge)
                        internal_store_paths.append({"store_node": node, "store_pc": norm(instruction.get("address")),
                                                     "nodes": nodes, "edges": edges})
                status = "PARTIAL_OUTPUT_FLOW"
                callee_write = "EXACT_FORMAL_DFG_WITHOUT_COMPLETE_STORE_AND_CALLER_RELOAD_PROOF"

        results.append({
            "schema": "task2_v018_output_pointer_propagation",
            "flow_id": f"{sample}::output::{stable_id(binding['binding_id'])}",
            "source_binding_id": binding["binding_id"],
            "sample_id": sample,
            "caller": binding["caller"],
            "callee": binding["callee"],
            "callsite_pc": binding["callsite_pc"],
            "actual_varnode": binding.get("actual_varnode"),
            "formal_index": binding.get("formal_index"),
            "formal_parameter_id": binding.get("formal_parameter_id"),
            "pointee_memory_object": pointee,
            "callee_write_evidence": callee_write,
            "caller_post_call_use_node": caller_use,
            "internal_store_paths": internal_store_paths,
            "binding_status": status,
            "interface_role": binding.get("interface_role", "UNKNOWN"),
            "semantic_source": (["A_STRUCTURAL_N_PCODE_CFG_DFG", "B_DETERMINISTIC_API_SIGNATURE"]
                                if binding.get("semantic_source") == "DETERMINISTIC_API_SIGNATURE"
                                else ["A_STRUCTURAL_N_PCODE_CFG_DFG", "B_DETERMINISTIC_SYSV_ABI"]),
            "evidence_refs": list(binding.get("evidence_refs") or []) +
                             ([caller_use] if caller_use else []) +
                             [path["store_node"] for path in internal_store_paths],
        })
    return sorted(results, key=lambda row: (row["sample_id"], row["caller"], row["callsite_pc"],
                                             -1 if row["formal_index"] is None else row["formal_index"]))


def build_verified_security_paths(corpus: Corpus) -> list[dict[str, Any]]:
    """Build exact UAF paths: free(alias) -> internal boundary -> callee dereference."""
    free_by_caller: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for binding in corpus.api_bindings:
        if binding.get("callee") == "API::free" and binding.get("deterministic_parameter_role") == "resource_release":
            free_by_caller[(binding["sample_id"], binding["caller"])].append(binding)

    verified = []
    for binding in corpus.internal_bindings:
        if binding.get("formal_type_category") != "ptr":
            continue
        sample = binding["sample_id"]
        actual_writer = binding.get("reaching_definition_node")
        call_node = binding.get("callsite_node")
        seed = corpus.formal_seed(binding)
        if not actual_writer or not call_node or not seed:
            continue
        parent, parent_edge = corpus.shortest_paths(sample, seed)
        sinks = []
        for node, instruction in corpus.by_function.get((sample, binding["callee"]), []):
            if (node in parent and instruction.get("opcode") == "LOAD" and
                    is_nonstack_memory(instruction)):
                sink_nodes, sink_edges = corpus.reconstruct(node, parent, parent_edge)
                sinks.append((node, instruction, sink_nodes, sink_edges))
        if not sinks:
            continue

        for release in free_by_caller.get((sample, binding["caller"]), []):
            release_writer = release.get("reaching_definition_node")
            release_call = release.get("callsite_node")
            if not release_writer or not release_call:
                continue
            if not corpus.cfg_reaches(sample, binding["caller"], release_call, call_node):
                continue
            alias = corpus.exact_common_memory_origin(sample, release_writer, actual_writer)
            if not alias:
                continue
            for sink_node, sink_instruction, sink_nodes, sink_edges in sinks:
                evidence_nodes = []
                for node in alias["left_nodes"] + [release_call] + alias["right_nodes"] + [call_node] + sink_nodes:
                    if node not in evidence_nodes and (sample, node) in corpus.instructions:
                        evidence_nodes.append(node)
                verified.append({
                    "schema": "task2_v018_verified_interprocedural_security_path",
                    "path_id": f"{sample}::path::{stable_id(release['binding_id'], binding['binding_id'], sink_node)}",
                    "sample_id": sample,
                    "verification_status": "VERIFIED_INTERPROCEDURAL_SECURITY_PATH",
                    "path_kind": "UAF_FREE_TO_CALLEE_DEREFERENCE_EXACT_ALIAS",
                    "source_api": "free",
                    "source_function": binding["caller"],
                    "source_call_pc": release["callsite_pc"],
                    "source_call_node": release_call,
                    "released_object_identity": alias["object_identity"],
                    "alias_origin_node": alias["origin_node"],
                    "internal_callsite_pc": binding["callsite_pc"],
                    "internal_callsite_node": call_node,
                    "actual_varnode": binding.get("actual_varnode"),
                    "formal_index": binding.get("formal_index"),
                    "formal_parameter_id": binding.get("formal_parameter_id"),
                    "callee": binding["callee"],
                    "formal_seed_node": seed,
                    "sink_kind": "NONSTACK_POINTER_DEREFERENCE",
                    "sink_opcode": sink_instruction.get("opcode"),
                    "sink_pc": norm(sink_instruction.get("address")),
                    "sink_node": sink_node,
                    "sink_memory_objects": [value for value in memory_operands(sink_instruction)
                                            if "RSP" not in value and "RBP" not in value],
                    "caller_alias_paths": {
                        "origin_to_free_argument": alias["left_nodes"],
                        "origin_to_internal_actual": alias["right_nodes"],
                    },
                    "callee_def_use_nodes": sink_nodes,
                    "callee_def_use_edges": sink_edges,
                    "cfg_status": "FREE_CALL_REACHES_INTERNAL_CALL",
                    "boundary_status": "EXACT_DIRECT_CALL_SYSV_ACTUAL_FORMAL",
                    "interface_role": binding.get("interface_role", "UNKNOWN"),
                    "semantic_sources": ["A_STRUCTURAL_N_PCODE_CFG_DFG", "B_DETERMINISTIC_ABI_API"],
                    "verified_semantic_increment": "D_EXACT_INTERPROCEDURAL_UAF_PATH",
                    "evidence_chain": [corpus.evidence_node(sample, node) for node in evidence_nodes],
                    "evidence_refs": evidence_nodes + [binding["formal_parameter_id"]],
                })
    return sorted(verified, key=lambda row: (row["sample_id"], row["source_call_pc"], row["sink_pc"]))


def build_security_summaries(corpus: Corpus, paths: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = corpus.selected_samples
    summaries = []
    for row in rows(V017 / "FUNCTION_SECURITY_SUMMARIES_V017.jsonl"):
        if row.get("sample_id") not in selected:
            continue
        value = dict(row)
        value["schema"] = "task2_v018_function_security_summary"
        value["summary_status"] = "INFERRED" if str(row.get("proof_status", "")).startswith("ACTUAL_FORMAL_ONLY") else "PARTIAL"
        value["v017_provenance"] = row.get("schema")
        summaries.append(value)

    by_pair: dict[tuple[str, str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for path in paths:
        key = (path["sample_id"], path["source_function"], path["callee"], path["internal_callsite_pc"])
        by_pair[key].append(path)
    for (sample, caller, callee, callsite), values in sorted(by_pair.items()):
        refs = sorted({ref for path in values for ref in path["evidence_refs"]})
        summaries.extend([
            {
                "schema": "task2_v018_function_security_summary", "sample_id": sample,
                "function_id": caller, "callee": callee, "callsite_pc": callsite,
                "summary_kind": "TaintPropagationSummary", "summary_status": "VERIFIED",
                "source_parameter": 0, "destination_parameter": 0,
                "propagation_kind": "EXACT_ALIAS_TO_CALLEE_DEREFERENCE",
                "verified_path_ids": [path["path_id"] for path in values], "evidence_refs": refs,
                "semantic_source": "D_VERIFIED_SEMANTIC_INCREMENT",
            },
            {
                "schema": "task2_v018_function_security_summary", "sample_id": sample,
                "function_id": caller, "callee": callee, "callsite_pc": callsite,
                "summary_kind": "ResourceEffectSummary", "summary_status": "VERIFIED",
                "resource_effect": "FREE_THEN_FORWARD_SAME_OBJECT",
                "object_identities": sorted({path["released_object_identity"] for path in values}),
                "verified_path_ids": [path["path_id"] for path in values], "evidence_refs": refs,
                "semantic_source": "D_VERIFIED_SEMANTIC_INCREMENT",
            },
            {
                "schema": "task2_v018_function_security_summary", "sample_id": sample,
                "function_id": caller, "callee": callee, "callsite_pc": callsite,
                "summary_kind": "SafetyPreconditionSummary", "summary_status": "VERIFIED",
                "predicate": "object_lifetime_valid_at_callee_dereference",
                "predicate_result": "VIOLATED_BY_FREE_REACHABLE_BEFORE_CALL",
                "verified_path_ids": [path["path_id"] for path in values], "evidence_refs": refs,
                "semantic_source": "D_VERIFIED_SEMANTIC_INCREMENT",
            },
        ])

    graph: dict[str, set[str]] = collections.defaultdict(set)
    local = collections.Counter()
    for binding in corpus.internal_bindings:
        caller = f"{binding['sample_id']}::{binding['caller']}"
        callee = f"{binding['sample_id']}::{binding['callee']}"
        graph[caller].add(callee)
    for path in paths:
        local[f"{path['sample_id']}::{path['source_function']}"] += 1
        local[f"{path['sample_id']}::{path['callee']}"] += 1
    propagated = dict(local)
    converged = False
    iterations = 0
    for iteration in range(1, SCC_ITERATION_LIMIT + 1):
        iterations = iteration
        changed = False
        updated = dict(propagated)
        for caller, callees in graph.items():
            value = max([local.get(caller, 0)] + [propagated.get(callee, 0) for callee in callees])
            if value != propagated.get(caller, 0):
                updated[caller] = value
                changed = True
        propagated = updated
        if not changed:
            converged = True
            break
    components = strongly_connected_components(graph)
    recursive = [component for component in components if len(component) > 1 or
                 any(node in graph.get(node, set()) for node in component)]
    audit = {
        "iteration_limit": SCC_ITERATION_LIMIT,
        "iterations_used": iterations,
        "converged": converged,
        "non_converged": not converged,
        "scc_count": len(components),
        "recursive_scc_count": len(recursive),
        "propagated_verified_path_function_count": sum(value > 0 for value in propagated.values()),
    }
    return summaries, audit


def derived_uaf_candidates(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for path in paths:
        groups[(path["sample_id"], path["source_function"], path["source_call_pc"],
                path["internal_callsite_pc"])].append(path)
    candidates = []
    for (sample, function, free_pc, call_pc), values in sorted(groups.items()):
        call_node = values[0]["internal_callsite_node"]
        refs = sorted({ref for value in values for ref in value["evidence_refs"]})
        candidate_id = hashlib.sha256(
            f"v018|{sample}|{function}|UAF|{free_pc}|{call_pc}".encode()
        ).hexdigest()
        candidates.append({
            "schema": "task2_v018_vulnerability_candidate",
            "candidate_id": candidate_id,
            "origin_candidate_id": None,
            "sample_id": sample,
            "function_entry": function,
            "family": "UAF",
            "instruction_anchor_pc": call_pc,
            "instruction_anchor_node": call_node,
            "minimum_trigger": "FREE_TO_CALLEE_DEREFERENCE_SAME_OBJECT_EXACT_INTERPROCEDURAL_DFG",
            "candidate_not_confirmed_vulnerability": True,
            "confirmation_evidence": "VERIFIED_DATAFLOW_AND_LIFETIME_PATH_NOT_DYNAMIC_EXPLOIT_CONFIRMATION",
            "pointer_object_identity": values[0]["released_object_identity"],
            "interprocedural_path_ids": [value["path_id"] for value in values],
            "interprocedural_sink_pcs": [value["sink_pc"] for value in values],
            "evidence_refs": refs,
            "supporting_evidence": [
                "GHIDRA_VERIFIED_API_TARGET_FREE", "EXACT_STACK_MEMORY_OBJECT_ALIAS",
                "CFG_FREE_REACHES_INTERNAL_CALL", "EXACT_SYSV_ACTUAL_FORMAL_BINDING",
                "EXACT_CALLEE_FORMAL_TO_NONSTACK_LOAD_DFG",
            ],
            "deterministic_structural_inputs_enabled": True,
            "deterministic_api_targets_enabled": True,
            "used_TypeSem": False,
            "used_InterfaceSem": False,
            "used_LayoutSem": False,
            "used_APISem": False,
            "InterfaceSem_role": "UNKNOWN",
            "semantic_membership_role": {
                "TypeSem": "DISABLED", "InterfaceSem": "DISABLED",
                "LayoutSem": "DISABLED", "APISem": "DISABLED",
            },
            "trigger_score": 0.9,
            "semantic_support_score": 0.0,
            "taint_support_score": 1.0,
            "sanitizer_penalty": 0.0,
            "upstream_confidence": 0.9,
        })
    return candidates
