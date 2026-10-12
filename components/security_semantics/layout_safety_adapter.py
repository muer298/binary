#!/usr/bin/env python3
"""Strict LayoutSem-to-memory adapter for SERAPH Task2 v0.1.5.

Only an exact frozen raw LayoutSem object plus one of its exact support_sites
may bind to a P-code memory access.  Function proximity, names, and nearest
addresses are deliberately unsupported.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def blind(value: str) -> str:
    return "JULIET_" + hashlib.sha256(value.encode()).hexdigest()[:20]


def norm(value):
    try:
        return format(int(str(value), 16), "x")
    except (TypeError, ValueError):
        return None


def canon_loc(value):
    text = str(value or "")
    if not text.startswith("reg:"):
        return text
    reg = text[4:].upper()
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
        if reg in aliases:
            return "reg:" + full
    if reg.startswith("R") and reg[-1:] in {"B", "W", "D"} and reg[1:-1].isdigit():
        return "reg:" + reg[:-1]
    return "reg:" + reg


class LayoutSafetyAdapter:
    def __init__(self, stage2_path: Path, usg_nodes_path: Path, memory_path: Path):
        self.raw = {}
        for node in rows(usg_nodes_path):
            if node.get("node_type") == "LayoutSem":
                key = blind(node["sample_id"]) + node["node_id"][len(node["sample_id"]):]
                self.raw[key] = node
        self.memory_by_site = collections.defaultdict(list)
        for fact in rows(memory_path):
            self.memory_by_site[(fact["sample_id"], norm(fact["function_entry"]), fact["pcode_site"])].append(fact)
        self.selected = collections.defaultdict(list)
        for row in rows(stage2_path):
            if row.get("module") != "LayoutSem":
                continue
            if row.get("semantic_value_status") != "FROZEN_MODEL_SELECTED_SAME_FUNCTION_RAW_CANDIDATE":
                continue
            if not isinstance(row.get("semantic_value"), dict):
                continue
            winner = next((item.get("object_id") for item in row.get("candidate_list") or []
                           if item.get("semantic_id") == row.get("top1_prediction")), None)
            raw = self.raw.get(winner)
            self.selected[(row["sample_id"], norm(row["function_entry"]))].append((row, winner, raw))

    def object_facts(self, sample_id, function_entry):
        facts = []
        for row, winner, raw in self.selected.get((sample_id, norm(function_entry)), []):
            value = row["semantic_value"]
            attrs = (raw or {}).get("attrs") or {}
            sites = attrs.get("support_sites") or []
            if not sites:
                sites = [None]
            for site in sites:
                mapped = self.memory_by_site.get((sample_id, norm(function_entry), site), []) if site else []
                facts.append({
                    "object_id": winner,
                    "allocation_site": attrs.get("allocation_site"),
                    "base_pointer": canon_loc(value.get("base_object") or attrs.get("base_object")),
                    "allocation_size": value.get("object_size", "UNKNOWN"),
                    "field_offset": value.get("offsets") or attrs.get("offsets") or [],
                    "field_size": value.get("field_size", "UNKNOWN"),
                    "memory_access_pc": norm(site.split(":", 1)[0]) if site else None,
                    "memory_access_site": site,
                    "access_width": mapped[0].get("width_bytes") if len(mapped) == 1 else None,
                    "semantic_source": "INFERRED_LAYOUTSEM_FACT",
                    "semantic_confidence": row.get("confidence"),
                    "layout_confidence": value.get("layout_confidence"),
                    "evidence_refs": list(row.get("evidence_refs") or []) + ([winner] if winner else []),
                    "proof_status": "EXACT_SUPPORT_SITE_BINDING" if len(mapped) == 1 else "UNKNOWN_NO_UNIQUE_MEMORY_BINDING",
                    "layout_kind": attrs.get("layout_kind"),
                    "raw_support_site_count": len(attrs.get("support_sites") or []),
                })
        return facts

    def bind_write(self, write_fact, enabled: bool):
        if not enabled:
            return {"state": "UNKNOWN", "impact": "DISABLED", "facts": []}
        candidates = [fact for fact in self.object_facts(write_fact["sample_id"], write_fact["function_entry"])
                      if fact.get("memory_access_site") == write_fact.get("pcode_site") and
                      fact.get("proof_status") == "EXACT_SUPPORT_SITE_BINDING"]
        base = canon_loc(write_fact.get("base_pointer"))
        exact = [fact for fact in candidates if canon_loc(fact.get("base_pointer")) == base]
        if len(exact) != 1:
            return {"state": "UNKNOWN", "impact": "LOADED_ONLY", "facts": [],
                    "reason": "NO_UNIQUE_EXACT_SITE_AND_BASE_LAYOUT_BINDING"}
        fact = exact[0]
        has_boundary = fact.get("field_size") not in (None, "UNKNOWN") or fact.get("allocation_size") not in (None, "UNKNOWN")
        return {
            "state": "INFERRED_LAYOUTSEM_FACT", "facts": [fact],
            "impact": "PREDICATE_AFFECTING" if has_boundary else "EVIDENCE_ONLY",
            "reason": "EXACT_PCODE_SUPPORT_SITE_AND_BASE_MATCH",
        }


def make_boundary_constraint(heap_object, write_fact, layout_binding):
    capacity = heap_object.get("allocation_size_value")
    offset = write_fact.get("offset_value")
    length = write_fact.get("write_length_value")
    executable = all(isinstance(value, int) and value >= 0 for value in (capacity, offset, length))
    end = offset + length if executable else None
    layout_field_sizes = [fact.get("field_size") for fact in layout_binding.get("facts") or []
                          if isinstance(fact.get("field_size"), int)]
    field_size = layout_field_sizes[0] if len(set(layout_field_sizes)) == 1 else None
    return {
        "schema": "task2_v015_heap_boundary_constraint",
        "sample_id": write_fact["sample_id"], "function_entry": write_fact["function_entry"],
        "heap_object_id": heap_object["object_id"], "allocation_site": heap_object["allocation_pc"],
        "write_sink_pc": write_fact["sink_pc"], "write_kind": write_fact["write_kind"],
        "predicate": "offset + write_length <= allocation_capacity",
        "offset_expression": write_fact.get("offset_expression"),
        "write_length_expression": write_fact.get("write_length_expression"),
        "allocation_capacity_expression": heap_object.get("allocation_size_expression"),
        "offset_value": offset, "write_length_value": length,
        "allocation_capacity_value": capacity, "field_size_value": field_size,
        "integer_width_bits": 64, "signedness": "UNSIGNED_SIZE_DOMAIN",
        "expression_wrap_status": "NO_WRAP_FOR_CONSTANTS" if executable else "UNKNOWN",
        "range_check_status": "CHECK_UNKNOWN", "CFG_reachable": bool(write_fact.get("CFG_reachable")),
        "same_object_proof": write_fact.get("alias_status") == "VERIFIED_UNIQUE_ALIAS",
        "executable": executable,
        "allocation_boundary_violation": (end > capacity) if executable else None,
        "field_boundary_violation": (end > field_size) if executable and field_size is not None else None,
        "status": ("BOUNDARY_VIOLATED" if executable and end > capacity else
                   "BOUNDARY_SATISFIED" if executable else "BOUNDARY_UNKNOWN"),
        "confirmation_evidence": ("EXECUTABLE_ALLOCATION_BOUNDARY_VIOLATION" if executable and end > capacity
                                  else "UNVERIFIED_CANDIDATE"),
        "layout_binding": layout_binding,
    }
