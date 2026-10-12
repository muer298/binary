#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_extract_nodes_edges.py

v5.3.2 provenance-preserving revision with the v6.7.3 LayoutSem evidence
bridge.

Task 2 Day-1 Step 2:
Extract USG nodes and edges from semantic_dict JSON files.

Outputs:
  usg_nodes.jsonl
  usg_edges.jsonl
  graph_schema_stats.json

The extractor is schema-tolerant. It supports:
  - high_level_sem.type_sem / interface_sem / api_sem / mem_layout_sem / layout_sem
  - low/mid-level sections when present
  - explicit evidence/constraint edges when present
  - fallback derived edges from common cross-reference fields

This is intended as the first stable data interface for Task 2.

v5.3.2 fixes an important information-loss bug in the original extractor:
``semantic_template.symbol_resolution`` is a nested dictionary, but the old
``summarize_attrs`` retained only scalar/list children of a dictionary.  As a
result, deterministic APISem provenance such as
``resolution_kind=external_or_thunk`` disappeared from ``usg_nodes.jsonl``.

The revision:
  - preserves bounded nested dictionaries;
  - emits flat, auditable APISem provenance fields;
  - marks derived EvidenceStub edges as observation-only;
  - can rebuild only the nodes file, so an existing very large edges file does
    not need to be regenerated.

The v6.7.3 evidence bridge additionally normalises fixed offsets from
``InstrSem.effects.memory_accesses`` into compact query-side LayoutSem fields.
This prevents the node allow-list from discarding the offset evidence required
by the 2.2.3 pairwise LayoutSem reranker.
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path


KNOWN_NODE_PATHS = [
    ("InstrSem", "instruction", ["instruction_sem", "instr_sem", "low_level_sem.instruction_sem", "low_level_sem.instr_sem", "semantic_objects.instruction_sem"]),
    ("BlockSem", "block", ["block_sem", "basic_block_sem", "low_level_sem.block_sem", "semantic_objects.block_sem"]),
    ("CFGSem", "cfg", ["cfg_sem", "control_flow_sem", "mid_level_sem.cfg_sem", "semantic_objects.cfg_sem"]),
    ("DFGSem", "dfg", ["dfg_sem", "data_flow_sem", "mid_level_sem.dfg_sem", "semantic_objects.dfg_sem"]),
    ("TypeSem", "high", ["high_level_sem.type_sem", "type_sem", "semantic_objects.type_sem"]),
    ("InterfaceSem", "high", ["high_level_sem.interface_sem", "interface_sem", "semantic_objects.interface_sem"]),
    ("APISem", "high", ["high_level_sem.api_sem", "api_sem", "semantic_objects.api_sem"]),
    ("LayoutSem", "high", ["high_level_sem.mem_layout_sem", "high_level_sem.layout_sem", "mem_layout_sem", "layout_sem", "semantic_objects.layout_sem"]),
]

EDGE_KEYS_HINTS = ("evidence", "constraint", "edge", "edges", "relations")

HARD_API_SOURCES = {"import_table", "symbol_table", "exact_signature"}
SOFT_API_SOURCES = {"flirt", "behavior_match"}

API_SOURCE_ALIASES = {
    "import": "import_table",
    "imports": "import_table",
    "importtable": "import_table",
    "import_table": "import_table",
    "dynamic_symbol": "import_table",
    "dynamic_symbol_table": "import_table",
    "dynsym": "import_table",
    ".dynsym": "import_table",
    "plt": "import_table",
    ".plt": "import_table",
    "got": "import_table",
    ".got": "import_table",
    "symbol": "symbol_table",
    "symbols": "symbol_table",
    "symboltable": "symbol_table",
    "symbol_table": "symbol_table",
    "symtab": "symbol_table",
    ".symtab": "symbol_table",
    "external": "symbol_table",
    "external_function": "symbol_table",
    "ghidra_external_function": "symbol_table",
    "external_or_thunk": "symbol_table",
    "external_symbol_or_thunk": "symbol_table",
    "exact_signature": "exact_signature",
    "signature_exact": "exact_signature",
    "exact_signature_match": "exact_signature",
    "flirt": "flirt",
    "flirt_match": "flirt",
    "behavior": "behavior_match",
    "behaviour": "behavior_match",
    "behavior_match": "behavior_match",
    "behaviour_match": "behavior_match",
}

API_SOURCE_KEYS = (
    "api_source",
    "match_source",
    "recognition_source",
    "resolution_source",
    "symbol_source",
    "signature_source",
    "source_kind",
    "source_type",
    "resolution_kind",
    "resolution_method",
    "match_method",
    "provenance",
)

API_SOURCE_FLAGS = {
    "is_import": "import_table",
    "is_imported": "import_table",
    "from_import_table": "import_table",
    "resolved_from_import": "import_table",
    "is_dynamic_symbol": "import_table",
    "is_plt": "import_table",
    "from_symbol_table": "symbol_table",
    "resolved_from_symbol": "symbol_table",
    "is_symbol": "symbol_table",
    "is_exact_signature": "exact_signature",
    "exact_signature_match": "exact_signature",
}

LAYOUT_QUERY_EVIDENCE_EXPORT_REVISION = (
    "v6_7_3_instr_memory_access_bridge"
)

PLACEHOLDER_API_NAMES = {
    "",
    "unknown",
    "external",
    "external_api",
    "external_stub",
    "api",
    "none",
    "null",
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_path(obj, path):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        # If dict maps id -> object, keep values and preserve key.
        out = []
        for k, v in x.items():
            if isinstance(v, dict):
                vv = dict(v)
                vv.setdefault("_dict_key", k)
                out.append(vv)
            else:
                out.append({"id": k, "value": v})
        return out
    return []


def stable_hash(s, n=12):
    return hashlib.sha1(str(s).encode("utf-8", errors="replace")).hexdigest()[:n]


def normalize_api_source(value):
    """Normalize only explicit recognition-source values."""
    if value is None or isinstance(value, (dict, list, tuple)):
        return "unknown"
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    key = re.sub(r"[\s:/\\-]+", "_", text)
    key = re.sub(r"_+", "_", key).strip("_")
    if key in API_SOURCE_ALIASES:
        return API_SOURCE_ALIASES[key]
    if "external" in key and ("symbol" in key or "thunk" in key):
        return "symbol_table"
    if "import" in key or "dynsym" in key or key in ("plt", "got"):
        return "import_table"
    if "exact" in key and "signature" in key:
        return "exact_signature"
    if "symbol" in key or "symtab" in key:
        return "symbol_table"
    if "flirt" in key:
        return "flirt"
    if ("behavior" in key or "behaviour" in key) and "match" in key:
        return "behavior_match"
    return "unknown"


def truthy(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {
        "1", "true", "yes", "y", "on", "present", "matched", "resolved"
    }


def compact_value(value, depth=0, max_depth=4):
    """Keep useful nested metadata without copying unbounded raw structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if depth >= max_depth:
        if isinstance(value, dict):
            return {"_truncated": True, "_size": len(value)}
        if isinstance(value, (list, tuple)):
            return ["<truncated:%d>" % len(value)]
        return str(value)[:200]
    if isinstance(value, (list, tuple)):
        return [
            compact_value(item, depth + 1, max_depth)
            for item in list(value)[:50]
        ]
    if isinstance(value, dict):
        out = {}
        for key, item in list(value.items())[:40]:
            out[str(key)] = compact_value(item, depth + 1, max_depth)
        return out
    return str(value)[:200]


def iter_mappings(value, path="", depth=0, max_depth=5):
    if not isinstance(value, dict) or depth > max_depth:
        return
    yield path, value
    for key, item in value.items():
        child_path = ("%s.%s" % (path, key)).strip(".")
        if isinstance(item, dict):
            for found in iter_mappings(
                item, child_path, depth + 1, max_depth
            ):
                yield found
        elif isinstance(item, list):
            for index, child in enumerate(item[:40]):
                if isinstance(child, dict):
                    for found in iter_mappings(
                        child,
                        "%s[%d]" % (child_path, index),
                        depth + 1,
                        max_depth,
                    ):
                        yield found


def concrete_api_name(value):
    name = str(value or "").strip()
    if not name or name.lower() in PLACEHOLDER_API_NAMES:
        return False
    normalized = name.lower()
    normalized = normalized.replace("@plt", "").replace(".plt", "")
    normalized = re.sub(r"@@?glibc[^,\s]*", "", normalized)
    if normalized in PLACEHOLDER_API_NAMES:
        return False
    if re.match(r"^(fun|sub|lab|loc)_[0-9a-f]+$", normalized):
        return False
    if re.match(r"^(ordinal|ord)_[0-9]+$", normalized):
        return False
    return bool(re.search(r"[a-zA-Z_]", normalized))


def infer_api_provenance(obj, allow_legacy_external_symbol=False):
    """Recover APISem recognition provenance with an explicit audit trail.

    A known API name by itself is never enough.  The normal path consumes the
    nested ``symbol_resolution`` emitted by the supplied v11 generator.  The
    legacy opt-in exists only for old semantic JSON generated by the same
    external-function-only APISem lineage before that nested field was added.
    """
    unknown = {
        "api_source": "unknown",
        "deterministic_api_source": False,
        "api_source_origin": "none",
        "api_source_path": "",
        "api_source_raw_value": "",
        "api_source_signal_kind": "none",
    }
    if not isinstance(obj, dict):
        return unknown

    api_name = get_first(obj, ["api_name", "api_target"], "")
    callsite = get_first(
        obj, ["callsite", "call_site", "address", "location"], ""
    )

    for path, mapping in iter_mappings(obj):
        normalized_keys = {
            str(key).strip().lower(): (str(key), value)
            for key, value in mapping.items()
        }
        for key in API_SOURCE_KEYS:
            if key not in normalized_keys:
                continue
            actual_key, raw_value = normalized_keys[key]
            source = normalize_api_source(raw_value)
            if source == "unknown":
                continue
            # external_or_thunk is deterministic only when it resolves a
            # concrete symbol at a real call site.
            if (
                str(raw_value).strip().lower() == "external_or_thunk"
                and not (concrete_api_name(api_name) and callsite)
            ):
                continue
            return {
                "api_source": source,
                "deterministic_api_source": source in HARD_API_SOURCES,
                "api_source_origin": "semantic_object",
                "api_source_path": (
                    "%s.%s" % (path, actual_key)
                ).strip("."),
                "api_source_raw_value": str(raw_value),
                "api_source_signal_kind": "provenance_field",
            }

        for key, source in API_SOURCE_FLAGS.items():
            if key not in normalized_keys:
                continue
            actual_key, raw_value = normalized_keys[key]
            if not truthy(raw_value):
                continue
            return {
                "api_source": source,
                "deterministic_api_source": source in HARD_API_SOURCES,
                "api_source_origin": "semantic_object",
                "api_source_path": (
                    "%s.%s" % (path, actual_key)
                ).strip("."),
                "api_source_raw_value": str(raw_value),
                "api_source_signal_kind": "deterministic_flag",
            }

    # Older variants retained "api_symbol_resolution" only as constraint
    # evidence.  It is still stronger than an API name heuristic.
    propagated = obj.get("propagated_constraints")
    constraint_evidence = []
    if isinstance(propagated, dict):
        constraint_evidence = propagated.get("constraint_evidence") or []
    if (
        concrete_api_name(api_name)
        and callsite
        and "api_symbol_resolution" in {
            str(item).strip().lower() for item in constraint_evidence
        }
    ):
        return {
            "api_source": "symbol_table",
            "deterministic_api_source": True,
            "api_source_origin": "semantic_object",
            "api_source_path": (
                "propagated_constraints.constraint_evidence"
            ),
            "api_source_raw_value": "api_symbol_resolution",
            "api_source_signal_kind": "symbol_resolution_evidence",
        }

    if (
        allow_legacy_external_symbol
        and concrete_api_name(api_name)
        and callsite
    ):
        return {
            "api_source": "symbol_table",
            "deterministic_api_source": True,
            "api_source_origin": "legacy_generator_lineage",
            "api_source_path": "high_level_sem.api_sem",
            "api_source_raw_value": "external_function_or_thunk_only",
            "api_source_signal_kind": "explicit_legacy_lineage_assertion",
        }
    return unknown


def norm_addr(s):
    if s is None:
        return ""
    s = str(s)
    m = re.search(r"0x[0-9a-fA-F]+|[0-9a-fA-F]{6,16}", s)
    if not m:
        return ""
    x = m.group(0)
    if x.startswith("0x"):
        x = x[2:]
    return x.lower().lstrip("0") or "0"


def get_first(obj, keys, default=""):
    if not isinstance(obj, dict):
        return default
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            return obj[k]
    return default


def object_raw_id(obj, node_type, idx):
    if isinstance(obj, dict):
        for k in ["id", "node_id", "semantic_id", "func_id", "function_id", "block_id", "api_id", "layout_id", "_dict_key"]:
            if obj.get(k):
                return str(obj.get(k))
        # function level fallback
        fe = get_first(obj, ["func_entry", "function_entry", "entry", "address", "location"], "")
        fn = get_first(obj, ["function_name", "func_name", "name"], "")
        if fe or fn:
            return f"{node_type}_{fe}_{fn}".strip("_")
        loc = get_first(obj, ["location", "addr", "address"], "")
        if loc:
            return f"{node_type}_{loc}"
    return f"{node_type}_{idx}"


def node_id(sample_id, node_type, raw_id):
    return f"{sample_id}::{node_type}::{stable_hash(raw_id)}"


def _bounded_integer(value, minimum=0, maximum=65536):
    """Return a bounded integer without accepting booleans or fractions."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return None
            number = int(text, 0)
        else:
            number = int(value)
            if float(value) != float(number):
                return None
    except (TypeError, ValueError, OverflowError):
        return None
    if number < minimum or number > maximum:
        return None
    return number


def _explicit_false(value):
    if value is False or value == 0:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {
            "false", "no", "off", "0", "dynamic", "unknown",
        }
    return False


def _memory_access_records(obj):
    """Yield schema-tolerant InstrSem memory-access dictionaries."""
    if not isinstance(obj, dict):
        return []

    containers = []
    for key in ("memory_accesses", "memory_access_patterns"):
        if isinstance(obj.get(key), list):
            containers.append(obj.get(key))

    effects = obj.get("effects")
    if isinstance(effects, dict):
        for key in ("memory_accesses", "memory_access_patterns"):
            if isinstance(effects.get(key), list):
                containers.append(effects.get(key))

    records = []
    seen = set()
    for container in containers:
        for access in container:
            if not isinstance(access, dict):
                continue
            identity = id(access)
            if identity in seen:
                continue
            seen.add(identity)
            records.append(access)
    return records


def normalise_instr_layout_evidence(obj):
    """Extract Gold-free, fixed memory evidence from one raw InstrSem."""
    compact_accesses = []
    offsets = set()
    widths = set()
    bases = set()
    kinds = set()

    for access in _memory_access_records(obj):
        if _explicit_false(access.get("is_fixed")):
            continue

        offset = None
        for key in ("offset", "byte_offset", "displacement", "disp"):
            if key in access:
                offset = _bounded_integer(access.get(key))
                if offset is not None:
                    break
        if offset is None:
            continue

        width = None
        for key in (
            "byte_width", "width", "access_width", "size_bytes", "size",
        ):
            if key in access:
                width = _bounded_integer(
                    access.get(key), minimum=1, maximum=65536,
                )
                if width is not None:
                    break

        base = str(access.get("base") or "").strip()
        kind = str(
            access.get("access") or access.get("kind") or ""
        ).strip().lower()

        offsets.add(offset)
        if width is not None:
            widths.add(width)
        if base:
            bases.add(base)
        if kind:
            kinds.add(kind)

        compact_accesses.append({
            "site": access.get("site"),
            "base": base or None,
            "offset": offset,
            "access": kind or None,
            "width": width,
            "byte_width": width,
            "is_fixed": True,
            "source": "effects.memory_accesses",
        })

    return {
        "layout_query_evidence_export_revision": (
            LAYOUT_QUERY_EVIDENCE_EXPORT_REVISION
        ),
        "local_observed_field_offsets": sorted(offsets),
        "observed_field_offsets": sorted(offsets),
        "access_widths": sorted(widths),
        "memory_access_bases": sorted(bases),
        "memory_access_kinds": sorted(kinds),
        "memory_access_evidence": compact_accesses,
        "layout_evidence_scope": (
            "instruction_local" if offsets else "unavailable"
        ),
    }


def aggregate_instr_layout_evidence(nodes):
    """Union offsets for InstrSem nodes sharing a function-local base.

    A single LOAD/STORE often exposes only one field.  Base-aware aggregation
    gives the pairwise reranker the observed field set for the same recovered
    object while avoiding cross-function or cross-base merging.
    """
    clusters = defaultdict(lambda: {
        "offsets": set(),
        "widths": set(),
        "node_ids": set(),
    })

    for node in nodes:
        if node.get("node_type") != "InstrSem":
            continue
        attrs = node.get("attrs") or {}
        function_entry = (
            node.get("function_entry_norm")
            or node.get("function_entry")
            or ""
        )
        local_offsets = attrs.get("local_observed_field_offsets") or []
        widths = attrs.get("access_widths") or []
        for base in attrs.get("memory_access_bases") or []:
            key = (str(function_entry), str(base))
            clusters[key]["offsets"].update(local_offsets)
            clusters[key]["widths"].update(widths)
            clusters[key]["node_ids"].add(node.get("node_id"))

    for node in nodes:
        if node.get("node_type") != "InstrSem":
            continue
        attrs = node.get("attrs") or {}
        function_entry = (
            node.get("function_entry_norm")
            or node.get("function_entry")
            or ""
        )
        offsets = set(attrs.get("local_observed_field_offsets") or [])
        widths = set(attrs.get("access_widths") or [])
        member_count = 1 if offsets else 0
        cluster_keys = []
        for base in attrs.get("memory_access_bases") or []:
            key = (str(function_entry), str(base))
            cluster = clusters.get(key)
            if not cluster:
                continue
            offsets.update(cluster["offsets"])
            widths.update(cluster["widths"])
            member_count = max(member_count, len(cluster["node_ids"]))
            cluster_keys.append("%s|%s" % key)

        attrs["observed_field_offsets"] = sorted(offsets)
        attrs["access_widths"] = sorted(widths)
        attrs["layout_evidence_scope"] = (
            "base_cluster" if cluster_keys and offsets
            else "instruction_local" if offsets
            else "unavailable"
        )
        attrs["layout_evidence_cluster_keys"] = sorted(set(cluster_keys))
        attrs["layout_evidence_cluster_member_count"] = member_count
        node["attrs"] = attrs


def summarize_attrs(
    obj,
    node_type="",
    allow_legacy_external_symbol=False,
):
    if not isinstance(obj, dict):
        return {}
    keys = [
        "id", "function_entry", "func_entry", "function_name", "func_name", "function_id", "func_id",
        "address", "location", "callsite", "call_site", "block_id", "opcode", "category", "sem_type", "status", "confidence",
        "type_candidates", "type_confidence", "pred_type", "pred_type_category", "type_category",
        "param_index", "parameter_index", "parameter_type_sem",
        "return_type_sem", "local_var_type_sem", "type_constraints",
        "param_count", "param_count_source", "param_types", "param_type_categories",
        "params", "ret", "interface_constraints", "calling_convention",
        "return_type", "return_type_category", "api_name", "api_target", "semantic_template",
        "propagated_constraints", "api_source", "match_source", "recognition_source",
        "resolution_source", "resolution_method", "match_method", "provenance",
        "layout_kind", "offsets", "field_offsets", "observed_field_offsets",
        "inferred_field_offsets", "eval_field_offsets", "offset_set",
        "base_object", "base_storage", "base_obj", "support_sites",
        "access_widths", "memory_access_bases", "memory_access_kinds",
        "eval_exclude", "eval_exclude_reason",
    ]
    attrs = {}
    for k in keys:
        if k in obj:
            attrs[k] = compact_value(obj[k])

    if node_type == "APISem":
        provenance = infer_api_provenance(
            obj,
            allow_legacy_external_symbol=allow_legacy_external_symbol,
        )
        attrs.update(provenance)
        attrs["api_provenance_audit"] = dict(provenance)
    elif node_type == "InstrSem":
        attrs.update(normalise_instr_layout_evidence(obj))
    return attrs


def build_nodes(
    data,
    sample_id,
    json_path,
    include_raw=False,
    allow_legacy_external_symbol=False,
):
    nodes = []
    alias = {}

    for node_type, layer, paths in KNOWN_NODE_PATHS:
        seen_raw = set()
        idx = 0
        for path in paths:
            items = as_list(get_path(data, path))
            for obj in items:
                idx += 1
                rid = object_raw_id(obj, node_type, idx)
                if rid in seen_raw:
                    continue
                seen_raw.add(rid)

                nid = node_id(sample_id, node_type, rid)
                fe = ""
                if isinstance(obj, dict):
                    fe = get_first(obj, ["function_entry", "func_entry", "entry"], "")
                attrs = summarize_attrs(
                    obj,
                    node_type=node_type,
                    allow_legacy_external_symbol=(
                        allow_legacy_external_symbol
                    ),
                )

                rec = {
                    "sample_id": sample_id,
                    "node_id": nid,
                    "node_type": node_type,
                    "layer": layer,
                    "raw_id": rid,
                    "function_entry": str(fe),
                    "function_entry_norm": norm_addr(fe),
                    "json_path": str(json_path),
                    "source_path": path,
                    "attrs": attrs,
                }
                if include_raw:
                    rec["raw"] = obj
                nodes.append(rec)

                # Build alias map for resolving edge endpoints.
                for a in {rid, attrs.get("id", ""), attrs.get("location", ""), attrs.get("address", ""), attrs.get("func_entry", ""), attrs.get("function_entry", "")}:
                    if a not in (None, ""):
                        alias[str(a)] = nid
                        na = norm_addr(a)
                        if na:
                            alias[na] = nid

    aggregate_instr_layout_evidence(nodes)
    return nodes, alias


def recursive_edge_candidates(obj, path="root"):
    """Find edge-like dictionaries recursively."""
    out = []
    if isinstance(obj, dict):
        # Edge-like if it has source/target-ish fields.
        keys = {str(k).lower() for k in obj.keys()}
        has_src = any(k in keys for k in ["source", "src", "from", "source_node", "source_ref", "source_id", "low_node"])
        has_dst = any(k in keys for k in ["target", "dst", "to", "target_node", "target_ref", "target_id", "high_node"])
        has_edge_type = any("evidence" in k or "constraint" in k or k in ("edge_type", "relation", "type") for k in keys)
        if has_src and has_dst and has_edge_type:
            out.append((path, obj))
        for k, v in obj.items():
            lk = str(k).lower()
            # Search all, but edge-heavy keys are usually deeper.
            out.extend(recursive_edge_candidates(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(recursive_edge_candidates(v, f"{path}[{i}]"))
    return out


def get_edge_endpoint(e, side):
    if side == "source":
        keys = ["source_node", "source_ref", "source", "src", "from", "source_id", "low_node", "low_ref"]
    else:
        keys = ["target_node", "target_ref", "target", "dst", "to", "target_id", "high_node", "high_ref"]
    for k in keys:
        if isinstance(e, dict) and k in e and e[k] not in (None, ""):
            return e[k]
    return ""


def edge_kind_and_type(e, path):
    ek = "evidence" if "evidence" in path.lower() else "constraint" if "constraint" in path.lower() else "relation"
    et = ""
    for k in ["evidence_type", "constraint_type", "edge_type", "relation", "type", "kind"]:
        if isinstance(e, dict) and e.get(k):
            et = str(e.get(k))
            break
    if not et:
        et = ek
    if "constraint" in str(et).lower():
        ek = "constraint"
    if "evidence" in str(et).lower():
        ek = "evidence"
    return ek, et


def resolve_endpoint(x, alias):
    if x is None:
        return "", False
    if isinstance(x, dict):
        for k in ["node_id", "id", "ref", "location", "address", "func_entry", "function_entry"]:
            if x.get(k):
                return resolve_endpoint(x.get(k), alias)
        raw = json.dumps(x, ensure_ascii=False, sort_keys=True)[:200]
    elif isinstance(x, list):
        raw = ",".join(map(str, x[:5]))
    else:
        raw = str(x)

    if raw in alias:
        return alias[raw], True
    na = norm_addr(raw)
    if na and na in alias:
        return alias[na], True
    return raw, False


def query_evidence_from_edge(edge):
    """Retain only query-side structural evidence needed by Gold mapping.

    In particular, a BlockSem -> APISem evidence edge may carry the concrete
    call instruction only inside ``trigger``.  Target identifiers, API names,
    predictions, and scores are deliberately excluded from this sidecar.
    """
    if not isinstance(edge, dict):
        return {}
    result = {}
    containers = [edge]
    for key in ("query_evidence", "trigger"):
        value = edge.get(key)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for key in (
            "query_callsite", "source_callsite", "instruction_address",
            "callsite", "call_site", "function_entry", "func_entry",
        ):
            value = container.get(key)
            if value not in (None, "", [], {}):
                result.setdefault(key, compact_value(value))
    return result


def extract_explicit_edges(data, sample_id, alias):
    edges = []
    candidates = recursive_edge_candidates(data)
    for i, (path, e) in enumerate(candidates):
        ek, et = edge_kind_and_type(e, path)
        src_raw = get_edge_endpoint(e, "source")
        dst_raw = get_edge_endpoint(e, "target")
        src, src_ok = resolve_endpoint(src_raw, alias)
        dst, dst_ok = resolve_endpoint(dst_raw, alias)
        strength = get_first(e, ["strength", "weight", "confidence", "score"], None)
        rec = {
            "sample_id": sample_id,
            "edge_id": f"{sample_id}::{ek}::{stable_hash(path + str(i) + str(src_raw) + str(dst_raw))}",
            "edge_kind": ek,
            "edge_type": et,
            "source_node": src,
            "target_node": dst,
            "source_resolved": src_ok,
            "target_resolved": dst_ok,
            "source_raw": src_raw,
            "target_raw": dst_raw,
            "strength": strength,
            "source_path": path,
            "attrs": {k: v for k, v in e.items() if isinstance(v, (str, int, float, bool, type(None)))},
        }
        query_evidence = query_evidence_from_edge(e)
        if query_evidence:
            rec["attrs"]["query_evidence"] = query_evidence
        edges.append(rec)
    return edges


def attach_api_provenance_to_direct_edges(edges, nodes):
    """Copy resolved APISem provenance onto real direct edges for auditing."""
    api_provenance = {}
    for node in nodes:
        if node.get("node_type") != "APISem":
            continue
        attrs = node.get("attrs")
        if not isinstance(attrs, dict):
            continue
        source = normalize_api_source(attrs.get("api_source"))
        if source == "unknown":
            continue
        api_provenance[node.get("node_id")] = {
            "api_source": source,
            "recognition_source": attrs.get(
                "api_source_raw_value", source
            ),
            "api_source_path": attrs.get("api_source_path", ""),
            "deterministic_api_source": (
                source in HARD_API_SOURCES
            ),
        }

    enriched = 0
    for edge in edges:
        if not edge.get("source_resolved"):
            continue
        provenance = api_provenance.get(edge.get("target_node"))
        if not provenance:
            continue
        attrs = edge.get("attrs")
        if not isinstance(attrs, dict):
            attrs = {}
            edge["attrs"] = attrs
        for key, value in provenance.items():
            attrs.setdefault(key, value)
        enriched += 1
    return enriched


def derived_edges_from_nodes(nodes, sample_id):
    """Add conservative derived edges when no explicit edges exist."""
    edges = []
    by_func = defaultdict(list)
    by_raw = {}
    for n in nodes:
        by_func[n.get("function_entry_norm", "")].append(n)
        by_raw[n["raw_id"]] = n["node_id"]
        # Also alias attrs id
        aid = n.get("attrs", {}).get("id")
        if aid:
            by_raw[str(aid)] = n["node_id"]

    # Function-scope compatibility edges among high-level objects in the same function.
    for fe, ns in by_func.items():
        if not fe:
            continue
        highs = [n for n in ns if n["layer"] == "high"]
        for a in highs:
            for b in highs:
                if a["node_id"] == b["node_id"]:
                    continue
                if a["node_type"] == "InterfaceSem" and b["node_type"] == "TypeSem":
                    et = "interface_type_ref"
                elif a["node_type"] == "LayoutSem" and b["node_type"] == "TypeSem":
                    et = "layout_type_compatible"
                else:
                    continue
                edges.append({
                    "sample_id": sample_id,
                    "edge_id": f"{sample_id}::constraint::{stable_hash(a['node_id'] + b['node_id'] + et)}",
                    "edge_kind": "constraint",
                    "edge_type": et,
                    "source_node": a["node_id"],
                    "target_node": b["node_id"],
                    "source_resolved": True,
                    "target_resolved": True,
                    "source_raw": a["raw_id"],
                    "target_raw": b["raw_id"],
                    "strength": None,
                    "source_path": "derived.function_scope",
                    "attrs": {},
                })

    # Evidence stubs from evidence-like attrs on high-level nodes.
    for n in nodes:
        if n["layer"] != "high":
            continue
        attrs = n.get("attrs", {})
        # Layout offsets are fixed_offset evidence.
        if n["node_type"] == "LayoutSem" and (attrs.get("field_offsets") or attrs.get("offsets")):
            src_stub = f"{sample_id}::EvidenceStub::{stable_hash(n['node_id'] + 'fixed_offset')}"
            edges.append({
                "sample_id": sample_id,
                "edge_id": f"{sample_id}::evidence::{stable_hash(src_stub + n['node_id'])}",
                "edge_kind": "evidence",
                "edge_type": "fixed_offset",
                "source_node": src_stub,
                "target_node": n["node_id"],
                "source_resolved": False,
                "target_resolved": True,
                "source_raw": "layout_offset_pattern",
                "target_raw": n["raw_id"],
                "strength": None,
                "source_path": "derived.layout_offsets",
                "attrs": {"offsets": attrs.get("field_offsets") or attrs.get("offsets")},
            })
        if n["node_type"] == "APISem":
            src_stub = f"{sample_id}::EvidenceStub::{stable_hash(n['node_id'] + 'api_call')}"
            edges.append({
                "sample_id": sample_id,
                "edge_id": f"{sample_id}::evidence::{stable_hash(src_stub + n['node_id'])}",
                "edge_kind": "evidence",
                "edge_type": "api_call",
                "source_node": src_stub,
                "target_node": n["node_id"],
                "source_resolved": False,
                "target_resolved": True,
                "source_raw": "api_call_observation",
                "target_raw": n["raw_id"],
                "strength": None,
                "source_path": "derived.api_sem",
                "attrs": {
                    "api_source": "unknown",
                    "deterministic_api_source": False,
                    "observation_only": True,
                },
            })
        if n["node_type"] == "InterfaceSem":
            src_stub = f"{sample_id}::EvidenceStub::{stable_hash(n['node_id'] + 'def_use')}"
            edges.append({
                "sample_id": sample_id,
                "edge_id": f"{sample_id}::evidence::{stable_hash(src_stub + n['node_id'])}",
                "edge_kind": "evidence",
                "edge_type": "def_use",
                "source_node": src_stub,
                "target_node": n["node_id"],
                "source_resolved": False,
                "target_resolved": True,
                "source_raw": "interface_entry_or_callsite_pattern",
                "target_raw": n["raw_id"],
                "strength": None,
                "source_path": "derived.interface_sem",
                "attrs": {},
            })
        if n["node_type"] == "TypeSem":
            cats = attrs.get("type_candidates") or attrs.get("pred_type_category") or attrs.get("type_category")
            if cats:
                src_stub = f"{sample_id}::EvidenceStub::{stable_hash(n['node_id'] + 'deref_chain')}"
                edges.append({
                    "sample_id": sample_id,
                    "edge_id": f"{sample_id}::evidence::{stable_hash(src_stub + n['node_id'])}",
                    "edge_kind": "evidence",
                    "edge_type": "deref_chain",
                    "source_node": src_stub,
                    "target_node": n["node_id"],
                    "source_resolved": False,
                    "target_resolved": True,
                    "source_raw": "type_inference_pattern",
                    "target_raw": n["raw_id"],
                    "strength": None,
                    "source_path": "derived.type_sem",
                    "attrs": {},
                })

    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="JSONL produced by 01_build_usg_index.py")
    ap.add_argument("--nodes-out", required=True)
    ap.add_argument(
        "--edges-out",
        help="Required unless --nodes-only is used.",
    )
    ap.add_argument("--stats", required=True)
    ap.add_argument("--include-raw", action="store_true", help="Store raw object in node JSONL; can be large.")
    ap.add_argument("--no-derived-edges", action="store_true")
    ap.add_argument(
        "--nodes-only",
        action="store_true",
        help=(
            "Rebuild only usg_nodes.jsonl. Useful when the existing edges "
            "file is very large and node IDs have not changed."
        ),
    )
    ap.add_argument(
        "--legacy-apisem-external-symbol",
        action="store_true",
        help=(
            "Opt-in for old semantic JSON produced by the supplied generator "
            "whose APISem lineage was external-function/thunk-only but did "
            "not yet store symbol_resolution. Never enable for mixed or "
            "heuristically generated APISem data."
        ),
    )
    ap.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Stop after the first unreadable semantic JSON and return 2.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print a progress record every N samples; 0 disables it.",
    )
    args = ap.parse_args()

    if not args.nodes_only and not args.edges_out:
        ap.error("--edges-out is required unless --nodes-only is used")

    index_path = Path(args.index).resolve()
    nodes_out = Path(args.nodes_out).resolve()
    edges_out = (
        Path(args.edges_out).resolve()
        if args.edges_out
        else None
    )
    stats_path = Path(args.stats).resolve()
    nodes_out.parent.mkdir(parents=True, exist_ok=True)
    if edges_out is not None:
        edges_out.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    total_nodes = 0
    total_edges = 0
    node_type_cnt = Counter()
    edge_kind_cnt = Counter()
    edge_type_cnt = Counter()
    api_source_cnt = Counter()
    api_source_signal_cnt = Counter()
    direct_edges_with_api_provenance = 0
    instrsem_with_local_layout_offsets = 0
    instrsem_with_clustered_layout_offsets = 0
    instrsem_with_access_widths = 0
    sample_cnt = 0
    successful_samples = 0
    errors = []
    stop_after_error = False

    with ExitStack() as stack:
        fin = stack.enter_context(index_path.open("r", encoding="utf-8"))
        fnodes = stack.enter_context(
            nodes_out.open("w", encoding="utf-8")
        )
        fedges = None
        if not args.nodes_only:
            fedges = stack.enter_context(
                edges_out.open("w", encoding="utf-8")
            )
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            json_path = Path(row["json_path"])
            sample_cnt += 1
            try:
                data = load_json(json_path)
                nodes, alias = build_nodes(
                    data,
                    sample_id,
                    json_path,
                    include_raw=args.include_raw,
                    allow_legacy_external_symbol=(
                        args.legacy_apisem_external_symbol
                    ),
                )
                edges = []
                if not args.nodes_only:
                    edges = extract_explicit_edges(
                        data, sample_id, alias
                    )
                    direct_edges_with_api_provenance += (
                        attach_api_provenance_to_direct_edges(
                            edges, nodes
                        )
                    )
                    if not args.no_derived_edges:
                        edges.extend(
                            derived_edges_from_nodes(nodes, sample_id)
                        )

                for n in nodes:
                    fnodes.write(json.dumps(n, ensure_ascii=False) + "\n")
                    node_type_cnt[n["node_type"]] += 1
                    if n["node_type"] == "APISem":
                        attrs = n.get("attrs") or {}
                        api_source_cnt[
                            normalize_api_source(
                                attrs.get("api_source")
                            )
                        ] += 1
                        api_source_signal_cnt[
                            str(
                                attrs.get(
                                    "api_source_signal_kind",
                                    "none",
                                )
                            )
                        ] += 1
                    elif n["node_type"] == "InstrSem":
                        attrs = n.get("attrs") or {}
                        instrsem_with_local_layout_offsets += int(bool(
                            attrs.get("local_observed_field_offsets")
                        ))
                        instrsem_with_clustered_layout_offsets += int(bool(
                            attrs.get("observed_field_offsets")
                        ))
                        instrsem_with_access_widths += int(bool(
                            attrs.get("access_widths")
                        ))
                for e in edges:
                    fedges.write(json.dumps(e, ensure_ascii=False) + "\n")
                    edge_kind_cnt[e["edge_kind"]] += 1
                    edge_type_cnt[e["edge_type"]] += 1

                total_nodes += len(nodes)
                total_edges += len(edges)
                successful_samples += 1
            except Exception as ex:
                errors.append({"sample_id": sample_id, "json_path": str(json_path), "error": str(ex)})
                if args.fail_on_error:
                    stop_after_error = True

            if (
                args.progress_every > 0
                and sample_cnt % args.progress_every == 0
            ):
                print(
                    json.dumps(
                        {
                            "phase": "extract_nodes_edges",
                            "samples_seen": sample_cnt,
                            "successful_samples": successful_samples,
                            "nodes_written": total_nodes,
                            "edges_written": total_edges,
                            "hard_api_source_nodes": sum(
                                count
                                for source, count in api_source_cnt.items()
                                if source in HARD_API_SOURCES
                            ),
                            "errors": len(errors),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if stop_after_error:
                break

    stats = {
        "index": str(index_path),
        "sample_count": sample_cnt,
        "successful_sample_count": successful_samples,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "node_type_counts": dict(node_type_cnt),
        "edge_kind_counts": dict(edge_kind_cnt),
        "edge_type_counts": dict(edge_type_cnt.most_common(50)),
        "api_source_counts": dict(api_source_cnt),
        "api_source_signal_counts": dict(api_source_signal_cnt),
        "hard_api_source_count": sum(
            count
            for source, count in api_source_cnt.items()
            if source in HARD_API_SOURCES
        ),
        "direct_edges_with_api_provenance": (
            direct_edges_with_api_provenance
        ),
        "layout_query_evidence_export_revision": (
            LAYOUT_QUERY_EVIDENCE_EXPORT_REVISION
        ),
        "instrsem_with_local_layout_offsets": (
            instrsem_with_local_layout_offsets
        ),
        "instrsem_with_clustered_layout_offsets": (
            instrsem_with_clustered_layout_offsets
        ),
        "instrsem_with_access_widths": instrsem_with_access_widths,
        "nodes_only": bool(args.nodes_only),
        "legacy_apisem_external_symbol": bool(
            args.legacy_apisem_external_symbol
        ),
        "error_count": len(errors),
        "errors": errors[:20],
        "nodes_out": str(nodes_out),
        "edges_out": str(edges_out) if edges_out else "",
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", nodes_out)
    if edges_out:
        print("wrote", edges_out)
    print("wrote", stats_path)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if errors and args.fail_on_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
