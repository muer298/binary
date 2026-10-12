#!/usr/bin/env python3
"""Adapt frozen TypeSem and deterministic P-code into integer safety facts."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


ARITHMETIC_OPS = {"INT_ADD", "INT_SUB", "INT_MULT"}
CONVERSION_OPS = {"INT_ZEXT", "INT_SEXT", "SUBPIECE", "CAST"}
INTEGER_CATEGORIES = {"int", "integer", "char", "short", "long", "enum", "bool"}


def location_width_bits(location):
    text = str(location or "")
    if text.startswith("tmp:"):
        try:
            return int(text.rsplit(":", 1)[1]) * 8
        except ValueError:
            return None
    if not text.startswith("reg:"):
        return None
    reg = text[4:].upper()
    if reg in {"AL", "AH", "BL", "BH", "CL", "CH", "DL", "DH"} or reg.endswith("B"):
        return 8
    if reg in {"AX", "BX", "CX", "DX", "SI", "DI", "SP", "BP"} or reg.endswith("W"):
        return 16
    if reg.startswith("E") or reg.endswith("D"):
        return 32
    if reg.startswith("R"):
        return 64
    return None


def instruction_width_bits(row):
    writes = row.get("effects", {}).get("writes") or row.get("write_set") or []
    widths = [location_width_bits(value) for value in writes]
    widths = [value for value in widths if value]
    return min(widths) if widths else None


def integer_bounds(width, signedness):
    if not isinstance(width, int) or width <= 0:
        return None
    if signedness == "SIGNED":
        return {"minimum": -(2 ** (width - 1)), "maximum": 2 ** (width - 1) - 1}
    if signedness == "UNSIGNED":
        return {"minimum": 0, "maximum": 2 ** width - 1}
    return None


def conversion_semantics(opcode, source_width, destination_width):
    if opcode == "INT_ZEXT":
        return {"kind": "ZERO_EXTENSION", "source_signedness_interpretation": "UNSIGNED",
                "source_width": source_width, "destination_width": destination_width,
                "state": "DETERMINISTIC_PCODE_FACT"}
    if opcode == "INT_SEXT":
        return {"kind": "SIGN_EXTENSION", "source_signedness_interpretation": "SIGNED",
                "source_width": source_width, "destination_width": destination_width,
                "state": "DETERMINISTIC_PCODE_FACT"}
    if opcode == "SUBPIECE":
        return {"kind": "NARROWING_OR_EXTRACTION", "source_signedness_interpretation": "UNKNOWN",
                "source_width": source_width, "destination_width": destination_width,
                "state": "DETERMINISTIC_PCODE_FACT"}
    if opcode == "CAST":
        return {"kind": "CAST", "source_signedness_interpretation": "UNKNOWN",
                "source_width": source_width, "destination_width": destination_width,
                "state": "DETERMINISTIC_PCODE_FACT"}
    return {"kind": "UNKNOWN", "source_width": source_width,
            "destination_width": destination_width, "state": "UNKNOWN"}


def predicate_templates(opcode, width, signedness):
    if opcode not in ARITHMETIC_OPS or not width:
        return {"signed_overflow": "UNKNOWN", "unsigned_wraparound": "UNKNOWN",
                "narrowing": "UNKNOWN", "executable": False}
    operator = {"INT_ADD": "+", "INT_SUB": "-", "INT_MULT": "*"}[opcode]
    signed_bounds = integer_bounds(width, "SIGNED")
    unsigned_bounds = integer_bounds(width, "UNSIGNED")
    return {
        "operation": f"mathematical_result = lhs {operator} rhs",
        "signed_overflow": (f"mathematical_result < {signed_bounds['minimum']} or "
                            f"mathematical_result > {signed_bounds['maximum']}"),
        "unsigned_wraparound": f"mathematical_result > {unsigned_bounds['maximum']}",
        "active_signedness": signedness,
        "narrowing": "evaluated only when a deterministic narrowing conversion is on the evidence path",
        "executable": False,
        "non_executable_reason": "operand value ranges are unavailable",
    }


class TypeSafetyAdapter:
    def __init__(self, stage2_path: Path):
        self.by_function = collections.defaultdict(list)
        with stage2_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("module") != "TypeSem":
                    continue
                if row.get("semantic_value_status") != "FROZEN_MODEL_SELECTED_SAME_FUNCTION_RAW_CANDIDATE":
                    continue
                if not isinstance(row.get("semantic_value"), dict):
                    continue
                self.by_function[(row["sample_id"], row["function_entry"])].append(row)

    def facts(self, sample_id, function_entry):
        facts = []
        for row in self.by_function.get((sample_id, function_entry), []):
            value = row["semantic_value"]
            for scope in ("locals", "parameters"):
                for index, item in enumerate(value.get(scope) or []):
                    category = str(item.get("type_category") or "UNKNOWN").lower()
                    signedness = str(item.get("signedness") or "UNKNOWN").upper()
                    facts.append({
                        "state": "INFERRED_TYPESEM_FACT", "module": "TypeSem",
                        "object_scope": scope[:-1].upper(), "object_index": item.get("index", index),
                        "integer_width": item.get("bit_width"), "signedness": signedness,
                        "is_pointer": item.get("is_pointer"), "type_category": category,
                        "is_integer_category": category in INTEGER_CATEGORIES,
                        "confidence": row.get("confidence"), "semantic_provenance": row.get("model_arm"),
                        "semantic_value_status": row.get("semantic_value_status"),
                        "object_id": row.get("object_id"), "USG_evidence_refs": row.get("evidence_refs") or [],
                        "mapping_scope": "FUNCTION_LEVEL_ONLY_NOT_EXACT_VARNODE",
                    })
        return facts

    def instantiate(self, candidate, instruction_by_node, enabled):
        path_nodes = [value for value in candidate.get("evidence_refs") or [] if value in instruction_by_node]
        path = [instruction_by_node[value] for value in path_nodes]
        arithmetic = [row for row in path if row.get("opcode") in ARITHMETIC_OPS]
        source = arithmetic[0] if arithmetic else None
        deterministic_width = instruction_width_bits(source) if source else None
        type_facts = self.facts(candidate["sample_id"], candidate["function_entry"]) if enabled else []
        matching = [fact for fact in type_facts if fact["is_integer_category"] and
                    not fact["is_pointer"] and fact["integer_width"] == deterministic_width]
        known_signedness = sorted({fact["signedness"] for fact in matching
                                   if fact["signedness"] in {"SIGNED", "UNSIGNED"}})
        signedness = known_signedness[0] if len(known_signedness) == 1 else "UNKNOWN"
        conversions = []
        for row in path:
            if row.get("opcode") in CONVERSION_OPS:
                destination_width = instruction_width_bits(row)
                read_widths = [location_width_bits(value) for value in
                               (row.get("effects", {}).get("reads") or row.get("read_set") or [])]
                source_width = min([value for value in read_widths if value], default=None)
                fact = conversion_semantics(row["opcode"], source_width, destination_width)
                fact.update({"node": row.get("node_id"), "pc": row.get("pc")})
                conversions.append(fact)
        context = candidate.get("minimum_trigger") in {
            "ARITHMETIC_TO_VARIABLE_INDEX_OR_OFFSET_DEF_USE",
            "ARITHMETIC_TO_ALLOCATION_ARGUMENT_DEF_USE",
        }
        impact = "PREDICATE_AFFECTING" if enabled and matching else (
            "UNKNOWN" if enabled else "TYPESEM_DISABLED")
        return {
            "schema": "task2_v014_integer_constraint",
            "candidate_id": candidate["candidate_id"], "sample_id": candidate["sample_id"],
            "function_entry": candidate["function_entry"],
            "arithmetic_node": source.get("node_id") if source else None,
            "arithmetic_pc": source.get("pc") if source else None,
            "opcode": source.get("opcode") if source else "UNKNOWN",
            "deterministic_pcode_fact": {
                "state": "DETERMINISTIC_PCODE_FACT" if source else "UNKNOWN",
                "result_width": deterministic_width,
                "evidence_node": source.get("node_id") if source else None,
            },
            "type_semantic_facts": matching,
            "type_semantic_fact_state": ("INFERRED_TYPESEM_FACT" if matching else "UNKNOWN"),
            "type_semantic_impact": impact,
            "signedness_interpretation": signedness,
            "conversion_chain": conversions,
            "predicate": predicate_templates(source.get("opcode") if source else None,
                                             deterministic_width, signedness),
            "risk_context": "SECURITY_SENSITIVE_SINK_DEF_USE" if context else "UNKNOWN",
            "range_check_status": "CHECK_UNKNOWN",
            "constraint_status": "POTENTIAL_OVERFLOW_UNVERIFIED" if source and context else "UNKNOWN",
            "candidate_membership_changed_by_typesem": False,
            "used_TypeSem_fields": (["integer_width", "is_pointer", "type_category", "signedness",
                                      "confidence", "semantic_provenance", "USG_evidence_refs"]
                                     if matching else []),
        }


def self_test():
    assert location_width_bits("reg:AL") == 8
    assert location_width_bits("reg:EAX") == 32
    assert location_width_bits("reg:RAX") == 64
    assert location_width_bits("tmp:0x123:2") == 16
    assert integer_bounds(8, "SIGNED") == {"minimum": -128, "maximum": 127}
    assert integer_bounds(8, "UNSIGNED") == {"minimum": 0, "maximum": 255}
    assert integer_bounds(None, "SIGNED") is None
    assert conversion_semantics("INT_ZEXT", 8, 32)["kind"] == "ZERO_EXTENSION"
    assert conversion_semantics("INT_SEXT", 8, 32)["kind"] == "SIGN_EXTENSION"
    assert conversion_semantics("SUBPIECE", 64, 32)["kind"] == "NARROWING_OR_EXTRACTION"
    assert conversion_semantics("UNKNOWN", None, None)["state"] == "UNKNOWN"
    assert predicate_templates("INT_ADD", 8, "UNKNOWN")["executable"] is False
    assert predicate_templates("INT_ADD", None, "UNKNOWN")["signed_overflow"] == "UNKNOWN"
    return {"status": "PASSED", "tests": 12}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
