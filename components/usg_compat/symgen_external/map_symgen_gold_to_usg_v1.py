#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把已冻结的 SymGen DWARF/ELF Oracle 映射成四模块 USG Gold。

The converter never copies an evidence edge's target as the answer.  Edges are
used only to identify which low-level queries exercise each module; the gold
target is independently matched from the frozen Oracle to USG high-level
node attributes.  Ambiguous or weak matches are written to an audit report and
are excluded from the strict gold JSONL. 本脚本不再读取 full ELF，只能在
2.2.2 与 2.2.3 推理完成后的评估阶段调用。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from symgen_dwarf_oracle_v1 import (
    ORACLE_SCHEMA,
    find_function,
    normalize_symbol,
    parse_int,
    type_category,
)


SEMANTIC_TYPES = ("APISem", "TypeSem", "LayoutSem", "InterfaceSem")
IMPLEMENTATION_REVISION = "symgen_external_v2_address_space_normalization"
LOW_LEVEL_QUERY_TYPES = ("InstrSem", "BlockSem", "DFGSem", "CFGSem")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def first(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def nested_mappings(value: Any, depth: int = 0) -> Iterable[Mapping[str, Any]]:
    if depth > 5:
        return
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from nested_mappings(child, depth + 1)
    elif isinstance(value, list):
        for child in value[:100]:
            yield from nested_mappings(child, depth + 1)


def flattened(node: Mapping[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for mapping in nested_mappings(node):
        for key, value in mapping.items():
            values.setdefault(str(key), value)
    return values


def numeric_offsets(value: Any) -> tuple[int, ...]:
    values = []
    if isinstance(value, Mapping):
        value = list(value.values())
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    for item in value:
        parsed = parse_int(item)
        if parsed is not None and 0 <= parsed < 65536:
            values.append(parsed)
    return tuple(sorted(set(values)))


def node_signature(node: Mapping[str, Any]) -> Dict[str, Any]:
    value = flattened(node)
    entry = parse_int(first(value, (
        "function_entry", "func_entry", "entry", "function_address"
    )))
    address = parse_int(first(value, (
        "callsite", "call_site", "instruction_address", "address", "location"
    )))
    api_name = normalize_symbol(first(value, (
        "api_name", "api_target", "symbol_name", "external_name", "name"
    )))
    type_value = first(value, (
        "pred_type_category", "type_category", "pred_type", "type_name",
        "type_candidates", "param_type_categories", "category",
    ))
    param_categories = first(value, (
        "param_type_categories", "parameter_type_categories", "param_types"
    ), [])
    if not isinstance(param_categories, list):
        param_categories = []
    parameter_truth = []
    nested_parameters = first(value, ("parameter_type_sem", "params"), [])
    if isinstance(nested_parameters, list):
        for position, item in enumerate(nested_parameters):
            if not isinstance(item, Mapping):
                continue
            parameter_truth.append({
                "param_index": parse_int(first(item, (
                    "param_index", "parameter_index", "param_id", "index"
                ), position)),
                "type_category": type_category(first(item, (
                    "pred_type_category", "type_category", "pred_type",
                    "type_candidates", "category"
                ))),
            })
    ret = first(value, ("ret", "return_type_sem"), {})
    ret = ret if isinstance(ret, Mapping) else {}
    interface_constraints = first(value, ("interface_constraints",), {})
    interface_constraints = (
        interface_constraints
        if isinstance(interface_constraints, Mapping) else {}
    )
    if not param_categories and parameter_truth:
        param_categories = [item["type_category"] for item in parameter_truth]
    param_count = parse_int(first(value, (
        "param_count", "parameter_count", "argc"
    )))
    if param_count is None:
        param_count = parse_int(interface_constraints.get("param_count"))
    if param_count is None and parameter_truth:
        param_count = len(parameter_truth)
    return_category = type_category(first(value, (
        "return_type_category", "return_category", "return_type"
    )))
    if return_category == "unknown":
        return_category = type_category(first(ret, (
            "return_type_category", "return_category", "return_type",
            "pred_type_category", "type_candidates", "category"
        )))
    return {
        "entry": entry,
        "address": address,
        "api_name": api_name,
        "type_category": type_category(type_value),
        "param_index": parse_int(first(value, ("param_index", "parameter_index", "index"))),
        "param_count": param_count,
        "param_categories": [type_category(item) for item in param_categories],
        "parameter_truth": parameter_truth,
        "return_category": return_category,
        "offsets": numeric_offsets(first(value, (
            "eval_field_offsets", "observed_field_offsets", "field_offsets",
            "offsets", "offset_set"
        ), [])),
        "opcode": str(first(value, ("opcode", "mnemonic"), "")).lower(),
    }


def signature_in_elf_space(
    signature: Mapping[str, Any], address_delta: int
) -> Dict[str, Any]:
    """Translate Ghidra addresses into the frozen ELF/DWARF address space.

    ``address_delta`` is selected only from the two protocol-declared address
    hypotheses: identity, or subtract the recorded Ghidra image base.  No
    semantic label, candidate score, edge target, or evaluation result is used.
    """
    result = dict(signature)
    for key in ("entry", "address"):
        value = result.get(key)
        if value is None:
            continue
        numeric = int(value)
        # Mixed exporters occasionally retain an already-relative address.
        # Never turn such an address negative by subtracting the image base.
        result[key] = (
            numeric - int(address_delta)
            if address_delta > 0 and numeric >= int(address_delta)
            else numeric
        )
    return result


def _read_program_info_prefix(path: Path) -> Mapping[str, Any]:
    """Read only the small top-level ``program_info`` object when possible.

    Semantic JSON files can be very large.  Ghidra writes ``program_info`` near
    the beginning, so use ``JSONDecoder.raw_decode`` on a bounded prefix instead
    of loading an entire inference artifact during Gold evaluation.
    """
    decoder = json.JSONDecoder()
    text = ""
    try:
        with path.open("r", encoding="utf-8") as stream:
            while len(text) < 16 * 1024 * 1024:
                block = stream.read(256 * 1024)
                if not block:
                    break
                text += block
                match = re.search(r'"program_info"\s*:', text)
                if match is None:
                    continue
                start = match.end()
                while start < len(text) and text[start].isspace():
                    start += 1
                try:
                    value, _ = decoder.raw_decode(text, start)
                except json.JSONDecodeError:
                    continue
                return value if isinstance(value, Mapping) else {}
    except OSError:
        return {}
    return {}


def ghidra_image_base(
    index_row: Mapping[str, Any],
) -> Tuple[Optional[int], str]:
    """Resolve the non-Gold Ghidra image base with an auditable source."""
    for key in (
        "ghidra_image_base", "program_image_base", "image_base",
    ):
        parsed = parse_int(index_row.get(key))
        if parsed is not None:
            return parsed, "usg_index.%s" % key
    json_path = Path(str(index_row.get("json_path") or ""))
    if json_path.is_file():
        program_info = _read_program_info_prefix(json_path)
        for key in ("base", "image_base", "imageBase"):
            parsed = parse_int(program_info.get(key))
            if parsed is not None:
                return parsed, "semantic_json.program_info.%s" % key
    return None, "missing"


def select_address_alignment(
    oracle: Mapping[str, Any],
    sample_nodes: Sequence[Mapping[str, Any]],
    signatures: Mapping[str, Mapping[str, Any]],
    node_types: Mapping[str, str],
    index_row: Mapping[str, Any],
) -> Dict[str, Any]:
    """Select identity or the recorded Ghidra-base translation.

    Selection uses only low-level function-entry addresses and frozen DWARF
    subprogram entry addresses.  Requiring unique structural support prevents
    an arbitrary offset from being fitted to final Gold labels.
    """
    ghidra_base, base_source = ghidra_image_base(index_row)
    usg_entries = set()
    for node in sample_nodes:
        node_id = str(node.get("node_id") or "")
        if node_types.get(node_id) not in LOW_LEVEL_QUERY_TYPES:
            continue
        entry = signatures.get(node_id, {}).get("entry")
        if entry is not None:
            usg_entries.add(int(entry))
    dwarf_entries = {
        int(value)
        for value in (
            parse_int(function.get("low_pc"))
            for function in oracle.get("functions", []) or []
            if isinstance(function, Mapping)
        )
        if value is not None
    }

    candidates = [("identity", 0)]
    if ghidra_base not in (None, 0):
        candidates.append(("subtract_ghidra_image_base", int(ghidra_base)))
    scored = []
    for name, delta in candidates:
        exact = sum(
            1 for entry in usg_entries
            if entry >= delta and entry - delta in dwarf_entries
        )
        scored.append({"mode": name, "delta": delta, "exact_support": exact})
    ranked = sorted(scored, key=lambda item: (-item["exact_support"], item["delta"]))
    best = ranked[0]
    second_support = ranked[1]["exact_support"] if len(ranked) > 1 else -1
    required_support = min(3, len(usg_entries), len(dwarf_entries))
    resolved = (
        required_support > 0
        and best["exact_support"] >= required_support
        and best["exact_support"] > second_support
    )
    selected = best if resolved else {"mode": "unresolved", "delta": 0,
                                      "exact_support": best["exact_support"]}
    return {
        "resolved": bool(resolved),
        "selected_mode": selected["mode"],
        "selected_delta": int(selected["delta"]),
        "ghidra_image_base": ghidra_base,
        "ghidra_image_base_source": base_source,
        "low_level_function_entries": len(usg_entries),
        "dwarf_function_entries": len(dwarf_entries),
        "required_exact_support": required_support,
        "candidate_scores": scored,
        "policy": (
            "identity_or_subtract_recorded_ghidra_image_base; "
            "selected_by_low_level_function_entry_overlap_only"
        ),
    }


def edge_query_callsites(edge: Mapping[str, Any]) -> Set[int]:
    """Return whitelisted query-side callsites without reading edge targets."""
    values: Set[int] = set()
    containers: List[Tuple[Mapping[str, Any], Sequence[str]]] = [
        (edge, ("query_callsite", "source_callsite", "instruction_address")),
    ]
    attrs = edge.get("attrs")
    if isinstance(attrs, Mapping):
        containers.append((attrs, (
            "query_callsite", "source_callsite", "instruction_address",
            "callsite", "call_site",
        )))
        for key in ("query_evidence", "trigger"):
            nested = attrs.get(key)
            if isinstance(nested, Mapping):
                containers.append((nested, (
                    "query_callsite", "source_callsite",
                    "instruction_address", "callsite", "call_site",
                )))
    for key in ("query_evidence", "trigger"):
        nested = edge.get(key)
        if isinstance(nested, Mapping):
            containers.append((nested, (
                "query_callsite", "source_callsite",
                "instruction_address", "callsite", "call_site",
            )))
    for container, keys in containers:
        for key in keys:
            parsed = parse_int(container.get(key))
            if parsed is not None:
                values.add(int(parsed))
    return values


def edge_endpoints(edge: Mapping[str, Any]) -> tuple[str, str]:
    source = str(first(edge, ("source_node", "source", "src", "from"), ""))
    target = str(first(edge, ("target_node", "target", "dst", "to"), ""))
    return source, target


def edge_semantic_type(edge: Mapping[str, Any], node_type: Mapping[str, str]) -> str:
    source, target = edge_endpoints(edge)
    target_type = node_type.get(target, "")
    if target_type in SEMANTIC_TYPES:
        return target_type
    text = " ".join(str(first(edge, (key,), "")) for key in (
        "evidence_type", "edge_type", "relation", "kind", "constraint_type"
    )).lower()
    for token, semantic_type in (
        ("api", "APISem"), ("deref", "TypeSem"),
        ("type", "TypeSem"), ("offset", "LayoutSem"),
        ("layout", "LayoutSem"), ("def_use", "InterfaceSem"),
        ("interface", "InterfaceSem"), ("param", "InterfaceSem"),
    ):
        if token in text:
            return semantic_type
    return ""


def function_truth(oracle: Mapping[str, Any], signature: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    return find_function(oracle, signature.get("entry") or signature.get("address"))


def target_belongs_to_query_function(
    oracle: Mapping[str, Any],
    target_signature: Mapping[str, Any],
    query_signature: Mapping[str, Any],
) -> bool:
    target_function = function_truth(oracle, target_signature)
    query_function = function_truth(oracle, query_signature)
    if target_function is None or query_function is None:
        target_entry = target_signature.get("entry")
        query_entry = query_signature.get("entry")
        return (
            target_entry is not None
            and query_entry is not None
            and int(target_entry) == int(query_entry)
        )
    return int(target_function["low_pc"]) == int(query_function["low_pc"])


def oracle_symbol_for_callsite(
    oracle: Mapping[str, Any],
    address: Optional[int],
) -> Optional[str]:
    """Return a unique ELF-derived API symbol for a query instruction."""
    if address is None:
        return None
    mapping = oracle.get("callsite_to_symbol", {})
    if not isinstance(mapping, Mapping):
        return None
    numeric = int(address)
    for key in (hex(numeric), str(numeric), format(numeric, "x")):
        value = mapping.get(key)
        if value:
            symbol = normalize_symbol(value)
            if symbol:
                return symbol
    value = mapping.get(numeric)
    if value:
        symbol = normalize_symbol(value)
        if symbol:
            return symbol
    return None


def oracle_cache_is_current(oracle: Mapping[str, Any]) -> bool:
    return (
        oracle.get("schema") == ORACLE_SCHEMA
        and isinstance(oracle.get("callsite_to_symbol"), Mapping)
        and isinstance(oracle.get("callsite_extraction"), Mapping)
    )


def semantic_match(
    semantic_type: str,
    signature: Mapping[str, Any],
    oracle: Mapping[str, Any],
    query_signature: Mapping[str, Any],
) -> tuple[bool, Dict[str, Any]]:
    if semantic_type == "APISem":
        target_symbol = normalize_symbol(signature.get("api_name"))
        imported_symbols = {
            normalize_symbol(symbol)
            for symbol in oracle.get("undefined_dynamic_symbols", []) or []
            if normalize_symbol(symbol)
        }
        query_callsite = query_signature.get("address")
        oracle_symbol = oracle_symbol_for_callsite(oracle, query_callsite)
        if query_callsite is None:
            reason = "query_callsite_missing"
        elif oracle_symbol is None:
            reason = "oracle_callsite_symbol_missing"
        elif not target_symbol:
            reason = "target_api_name_missing"
        elif target_symbol not in imported_symbols:
            reason = "target_symbol_not_imported"
        elif target_symbol != oracle_symbol:
            reason = "target_symbol_differs_from_callsite_oracle"
        else:
            reason = "matched"
        matched = reason == "matched"
        return matched, {
            "oracle_kind": "elf_disassembly_plt_relocation_callsite_symbol",
            "oracle_value": oracle_symbol,
            "target_api_name": target_symbol,
            "query_callsite": query_callsite,
            "target_node_callsite_required": False,
            "reason": reason,
            "strict": matched,
        }
    function = function_truth(oracle, signature) or function_truth(oracle, query_signature)
    if semantic_type == "InterfaceSem":
        if not target_belongs_to_query_function(
            oracle, signature, query_signature
        ):
            return False, {
                "oracle_kind": "dwarf_subprogram_signature",
                "reason": "target_not_in_query_function",
            }
        if not function:
            return False, {"oracle_kind": "dwarf_subprogram", "reason": "no_function_match"}
        observed = 0
        matches = 0
        if signature.get("param_count") is not None:
            observed += 1
            matches += int(signature["param_count"] == function.get("param_count"))
        if signature.get("return_category") not in (None, "unknown"):
            observed += 1
            matches += int(signature["return_category"] == function.get("return_category"))
        param_categories = signature.get("param_categories") or []
        if param_categories:
            observed += 1
            matches += int(param_categories == function.get("param_categories"))
        strict = observed >= 2 and matches == observed
        return strict, {
            "oracle_kind": "dwarf_subprogram_signature",
            "oracle_value": {
                "low_pc": function.get("low_pc"),
                "param_count": function.get("param_count"),
                "param_categories": function.get("param_categories"),
                "return_category": function.get("return_category"),
            },
            "observed_components": observed,
            "matched_components": matches,
            "strict": strict,
        }
    if semantic_type == "TypeSem":
        if not target_belongs_to_query_function(
            oracle, signature, query_signature
        ):
            return False, {
                "oracle_kind": "dwarf_parameter_type",
                "reason": "target_not_in_query_function",
            }
        if not function:
            return False, {"oracle_kind": "dwarf_type", "reason": "no_function_match"}
        categories = function.get("param_categories", []) or []
        components = []
        param_index = signature.get("param_index")
        if param_index is not None and 0 <= param_index < len(categories):
            components.append((param_index, signature.get("type_category")))
        for item in signature.get("parameter_truth", []) or []:
            item_index = item.get("param_index")
            if item_index is not None:
                components.append((item_index, item.get("type_category")))
        if not components and len(categories) == 1:
            components.append((0, signature.get("type_category")))
        unique_components = {
            (int(index), str(observed or "unknown"))
            for index, observed in components
            if index is not None and 0 <= int(index) < len(categories)
        }
        known_components = [
            (index, observed) for index, observed in unique_components
            if observed != "unknown" and categories[index] != "unknown"
        ]
        strict = bool(known_components) and all(
            observed == categories[index]
            for index, observed in known_components
        )
        return strict, {
            "oracle_kind": "dwarf_parameter_type",
            "oracle_value": categories,
            "observed_components": [list(item) for item in sorted(known_components)],
            "param_index": param_index,
            "strict": strict,
        }
    if semantic_type == "LayoutSem":
        if not target_belongs_to_query_function(
            oracle, signature, query_signature
        ):
            return False, {
                "oracle_kind": "dwarf_struct_layout",
                "reason": "target_not_in_query_function",
            }
        if not function:
            return False, {"oracle_kind": "dwarf_struct_layout", "reason": "no_function_match"}
        expected_sets = {
            tuple(sorted(int(item) for item in offsets))
            for offsets in function.get("struct_offsets", []) or []
        }
        observed = tuple(signature.get("offsets") or ())
        strict = bool(observed and observed in expected_sets)
        return strict, {
            "oracle_kind": "dwarf_function_local_struct_offsets",
            "oracle_value": [list(value) for value in sorted(expected_sets)],
            "observed": list(observed),
            "strict": strict,
        }
    return False, {"oracle_kind": "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usg-index", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-out", required=True)
    parser.add_argument("--oracle-cache-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--require-complete-oracle-coverage",
        action="store_true",
        help="fail strict_gold_ready when any considered sample lacks a frozen Oracle",
    )
    parser.add_argument(
        "--min-mapping-rate-per-module",
        type=float,
        default=0.0,
        help=(
            "每模块严格 Gold 映射率下限；只作预先声明的完整性门禁，不参与调参"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.min_mapping_rate_per_module <= 1.0:
        raise ValueError("--min-mapping-rate-per-module must be in [0,1]")

    for path in (Path(args.out), Path(args.audit_out)):
        if path.exists() and not args.overwrite:
            raise FileExistsError("Refusing to overwrite: %s" % path)
        path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.oracle_cache_dir).resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError("Oracle cache directory missing: %s" % cache_dir)

    index = {str(row.get("sample_id") or ""): row for row in read_jsonl(args.usg_index)}
    splits = {str(row.get("sample_id") or ""): row for row in read_jsonl(args.split_manifest)}
    sample_ids = sorted(set(index) & set(splits))
    if args.max_samples > 0:
        sample_ids = sample_ids[:args.max_samples]
    wanted_samples = set(sample_ids)

    nodes_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    node_type: Dict[str, str] = {}
    node_signatures: Dict[str, Dict[str, Any]] = {}
    for node in read_jsonl(args.nodes):
        sample_id = str(node.get("sample_id") or "")
        if sample_id not in wanted_samples:
            continue
        node_id = str(node.get("node_id") or "")
        if not sample_id or not node_id:
            continue
        nodes_by_sample[sample_id].append(node)
        node_type[node_id] = str(node.get("node_type") or "")
        node_signatures[node_id] = node_signature(node)

    queries: Dict[str, Dict[str, Dict[str, Set[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    for edge in read_jsonl(args.edges):
        edge_sample_id = str(edge.get("sample_id") or "")
        if edge_sample_id and edge_sample_id not in wanted_samples:
            continue
        source, target = edge_endpoints(edge)
        semantic_type = edge_semantic_type(edge, node_type)
        if semantic_type not in SEMANTIC_TYPES or not source:
            continue
        sample_id = str(edge.get("sample_id") or source.split("::", 1)[0])
        if sample_id not in wanted_samples:
            continue
        if node_type.get(source) not in LOW_LEVEL_QUERY_TYPES:
            continue
        # The target is intentionally not retained as the answer.  Only
        # whitelisted query-side callsite metadata is retained.
        queries[sample_id][semantic_type][source].update(
            edge_query_callsites(edge)
        )

    outputs: List[Dict[str, Any]] = []
    audit_examples = []
    counts = Counter()
    by_module = Counter()
    callsite_statuses = Counter()
    api_rejection_counts = Counter()
    address_alignments: Dict[str, Dict[str, Any]] = {}
    sample_ids = [sample_id for sample_id in sample_ids if sample_id in nodes_by_sample]
    query_module_totals = Counter()
    query_totals_by_optimization = Counter()
    mapped_by_optimization = Counter()
    for sample_id in sample_ids:
        optimization = str(splits[sample_id].get("optimization") or "unknown")
        for semantic_type, query_records in queries.get(sample_id, {}).items():
            query_module_totals[semantic_type] += len(query_records)
            query_totals_by_optimization[optimization] += len(query_records)
    for number, sample_id in enumerate(sample_ids, 1):
        cache_path = cache_dir / (sample_id + ".oracle_symgen_v1.json")
        try:
            if not cache_path.is_file():
                counts["missing_oracle_cache"] += 1
                raise FileNotFoundError(cache_path)
            oracle = json.loads(cache_path.read_text(encoding="utf-8"))
            if not oracle_cache_is_current(oracle):
                counts["stale_oracle_cache"] += 1
                raise ValueError("unexpected Oracle schema")
            counts["oracle_cache_hit"] += 1
        except Exception as error:
            counts["oracle_error"] += 1
            if len(audit_examples) < 50:
                audit_examples.append({
                    "sample_id": sample_id,
                    "stage": "load_frozen_oracle",
                    "error": str(error),
                    "oracle_cache": str(cache_path),
                })
            continue
        counts["samples_with_oracle"] += 1
        if oracle.get("has_dwarf"):
            counts["samples_with_dwarf"] += 1
        callsite_map = oracle.get("callsite_to_symbol", {})
        if isinstance(callsite_map, Mapping) and callsite_map:
            counts["samples_with_api_callsite_map"] += 1
            counts["api_callsites_resolved"] += len(callsite_map)
        extraction = oracle.get("callsite_extraction", {})
        if isinstance(extraction, Mapping):
            callsite_statuses[str(extraction.get("status") or "missing")] += 1

        alignment = select_address_alignment(
            oracle,
            nodes_by_sample[sample_id],
            node_signatures,
            node_type,
            index[sample_id],
        )
        address_alignments[sample_id] = alignment
        alignment_mode = str(alignment.get("selected_mode") or "unresolved")
        counts["address_alignment_%s" % alignment_mode] += 1
        address_delta = int(alignment.get("selected_delta") or 0)
        sample_signatures = {
            str(node.get("node_id")): signature_in_elf_space(
                node_signatures.get(str(node.get("node_id")), {}),
                address_delta,
            )
            for node in nodes_by_sample[sample_id]
        }

        targets_by_module: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
        for node in nodes_by_sample[sample_id]:
            semantic_type = str(node.get("node_type") or "")
            if semantic_type in SEMANTIC_TYPES:
                targets_by_module[semantic_type].append(
                    (node, sample_signatures[str(node["node_id"])])
                )

        for semantic_type, query_records in queries.get(sample_id, {}).items():
            candidates = targets_by_module.get(semantic_type, [])
            for query_id in sorted(query_records):
                query_signature = dict(sample_signatures.get(query_id, {}))
                raw_hints = query_records[query_id]
                callsite_hints = {
                    signature_in_elf_space(
                        {"address": hint}, address_delta
                    ).get("address")
                    for hint in raw_hints
                }
                callsite_hints.discard(None)
                if semantic_type == "APISem":
                    source_type = node_type.get(query_id, "")
                    if len(callsite_hints) == 1 and (
                        source_type != "InstrSem"
                        or query_signature.get("address") is None
                    ):
                        query_signature["address"] = next(iter(callsite_hints))
                        counts["api_query_callsite_from_edge_context"] += 1
                    elif len(callsite_hints) > 1:
                        counts["api_query_ambiguous_edge_callsites"] += 1
                matched = []
                weak = []
                for target, signature in candidates:
                    ok, lineage = semantic_match(
                        semantic_type, signature, oracle, query_signature
                    )
                    item = (target, lineage)
                    if ok:
                        matched.append(item)
                    else:
                        if semantic_type == "APISem":
                            api_rejection_counts[
                                str(lineage.get("reason") or "unknown")
                            ] += 1
                        if lineage.get("oracle_value") not in (None, [], {}, "unknown"):
                            weak.append(item)
                if len(matched) != 1:
                    counts["ambiguous_or_unmatched_query_module"] += 1
                    counts[
                        "ambiguous_or_unmatched_%s" % semantic_type
                    ] += 1
                    if len(audit_examples) < 100:
                        audit_examples.append({
                            "sample_id": sample_id,
                            "query_id": query_id,
                            "semantic_type": semantic_type,
                            "strict_matches": len(matched),
                            "weak_matches": len(weak),
                            "candidate_targets": len(candidates),
                            "address_alignment_mode": alignment_mode,
                            "query_callsite_hint_count": len(callsite_hints),
                        })
                    continue
                target, lineage = matched[0]
                outputs.append({
                    "schema": "task2_symgen_usg_gold_v1",
                    "sample_id": sample_id,
                    "group_id": splits[sample_id].get("group_id"),
                    "project": splits[sample_id].get("project"),
                    "optimization": splits[sample_id].get("optimization"),
                    "architecture": splits[sample_id].get("architecture"),
                    "split": splits[sample_id].get("split"),
                    "query_id": query_id,
                    "target_id": target.get("node_id"),
                    "target_semantic_type": semantic_type,
                    "gold_source": lineage.get("oracle_kind"),
                    "gold_strength": "strict_independent",
                    "oracle_cache": str(cache_path),
                    "lineage": lineage,
                })
                by_module[semantic_type] += 1
                mapped_by_optimization[
                    str(splits[sample_id].get("optimization") or "unknown")
                ] += 1
                counts["strict_gold_labels"] += 1
        if number % 25 == 0:
            print(json.dumps({
                "phase": "convert_gold", "samples": number,
                "labels": counts["strict_gold_labels"],
            }), flush=True)

    seen = set()
    deduplicated = []
    for row in outputs:
        key = (row["query_id"], row["target_id"], row["target_semantic_type"])
        if key in seen:
            counts["duplicate_labels_removed"] += 1
            continue
        seen.add(key)
        deduplicated.append(row)
    with Path(args.out).open("w", encoding="utf-8") as stream:
        for row in deduplicated:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_counts = Counter(row["split"] for row in deduplicated)
    group_splits: Dict[str, Set[str]] = defaultdict(set)
    for row in deduplicated:
        group_splits[str(row.get("group_id") or "")].add(str(row.get("split") or ""))
    overlap = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    missing_oracles = int(counts.get("missing_oracle_cache", 0))
    oracle_coverage = (
        float(len(sample_ids) - missing_oracles) / float(len(sample_ids))
        if sample_ids else 0.0
    )
    mapping_coverage_by_module = {}
    for module in SEMANTIC_TYPES:
        total = int(query_module_totals[module])
        mapped = int(by_module[module])
        mapping_coverage_by_module[module] = {
            "query_module_objects": total,
            "strict_gold_labels": mapped,
            "mapping_rate": (float(mapped) / float(total) if total else None),
        }
    mapping_rates_pass = all(
        item["mapping_rate"] is not None
        and float(item["mapping_rate"]) + 1e-12
        >= args.min_mapping_rate_per_module
        for item in mapping_coverage_by_module.values()
    )
    mapping_coverage_by_optimization = {}
    for optimization in sorted(query_totals_by_optimization):
        total = int(query_totals_by_optimization[optimization])
        mapped = int(mapped_by_optimization[optimization])
        mapping_coverage_by_optimization[optimization] = {
            "query_module_objects": total,
            "strict_gold_labels": mapped,
            "mapping_rate": (float(mapped) / float(total) if total else None),
        }
    strict_gold_ready = (
        bool(deduplicated)
        and len(overlap) == 0
        and all(by_module[module] > 0 for module in SEMANTIC_TYPES)
        and mapping_rates_pass
        and (
            not args.require_complete_oracle_coverage
            or missing_oracles == 0
        )
    )
    report = {
        "schema": "task2_symgen_usg_gold_audit_v1",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "inputs": {
            "usg_index": args.usg_index,
            "nodes": args.nodes,
            "edges": args.edges,
            "split_manifest": args.split_manifest,
            "oracle_cache_dir": str(cache_dir),
            "require_complete_oracle_coverage": (
                args.require_complete_oracle_coverage
            ),
            "min_mapping_rate_per_module": args.min_mapping_rate_per_module,
        },
        "samples_considered": len(sample_ids),
        "oracle_cache_coverage": oracle_coverage,
        "counts": dict(counts),
        "labels_by_module": dict(by_module),
        "mapping_coverage_by_module": mapping_coverage_by_module,
        "mapping_coverage_by_optimization": mapping_coverage_by_optimization,
        "mapping_rates_pass": mapping_rates_pass,
        "labels_by_split": dict(split_counts),
        "api_callsite_oracle_status_counts": dict(callsite_statuses),
        "api_candidate_rejection_counts": dict(api_rejection_counts),
        "address_alignment_by_sample": address_alignments,
        "address_alignment_policy": (
            "Only identity or subtraction of the recorded Ghidra image base "
            "is allowed. Selection uses low-level function-entry overlap with "
            "frozen DWARF and never uses candidate scores, semantic labels, "
            "edge targets, or evaluation metrics."
        ),
        "group_overlap_count": len(overlap),
        "group_overlap_examples": dict(list(overlap.items())[:20]),
        "strict_gold_ready": strict_gold_ready,
        "audit_examples": audit_examples,
        "note": (
            "Only unique independent DWARF/ELF matches are emitted. APISem "
            "uses query callsite -> ELF disassembly/PLT symbol; target-node "
            "callsites and USG edge targets are never used as gold answers. "
            "Ghidra/ELF address spaces are aligned by a protocol-declared "
            "structural rule. All Oracle files were frozen before stripped-"
            "binary inference."
        ),
    }
    Path(args.audit_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
