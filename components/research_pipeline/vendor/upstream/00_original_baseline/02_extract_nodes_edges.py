#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_extract_nodes_edges.py

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
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
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


def summarize_attrs(obj):
    if not isinstance(obj, dict):
        return {}
    keys = [
        "id", "function_entry", "func_entry", "function_name", "func_name", "function_id", "func_id",
        "address", "location", "block_id", "opcode", "category", "sem_type", "status", "confidence",
        "type_candidates", "type_confidence", "pred_type", "pred_type_category", "type_category",
        "param_count", "param_count_source", "param_types", "param_type_categories",
        "return_type", "return_type_category", "api_name", "api_target", "semantic_template",
        "layout_kind", "offsets", "field_offsets", "base_object", "base_storage",
        "eval_exclude", "eval_exclude_reason",
    ]
    attrs = {}
    for k in keys:
        if k in obj:
            v = obj[k]
            # Avoid storing very large raw substructures here.
            if isinstance(v, (str, int, float, bool)) or v is None:
                attrs[k] = v
            elif isinstance(v, list):
                attrs[k] = v[:50]
            elif isinstance(v, dict):
                attrs[k] = {kk: vv for kk, vv in list(v.items())[:30] if isinstance(vv, (str, int, float, bool, type(None), list))}
    return attrs


def build_nodes(data, sample_id, json_path, include_raw=False):
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
                attrs = summarize_attrs(obj)

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
        edges.append(rec)
    return edges


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
                "attrs": {},
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
    ap.add_argument("--edges-out", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--include-raw", action="store_true", help="Store raw object in node JSONL; can be large.")
    ap.add_argument("--no-derived-edges", action="store_true")
    args = ap.parse_args()

    index_path = Path(args.index).resolve()
    nodes_out = Path(args.nodes_out).resolve()
    edges_out = Path(args.edges_out).resolve()
    stats_path = Path(args.stats).resolve()
    nodes_out.parent.mkdir(parents=True, exist_ok=True)
    edges_out.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    total_nodes = 0
    total_edges = 0
    node_type_cnt = Counter()
    edge_kind_cnt = Counter()
    edge_type_cnt = Counter()
    sample_cnt = 0
    errors = []

    with index_path.open("r", encoding="utf-8") as fin, \
         nodes_out.open("w", encoding="utf-8") as fnodes, \
         edges_out.open("w", encoding="utf-8") as fedges:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            json_path = Path(row["json_path"])
            sample_cnt += 1
            try:
                data = load_json(json_path)
                nodes, alias = build_nodes(data, sample_id, json_path, include_raw=args.include_raw)
                edges = extract_explicit_edges(data, sample_id, alias)
                if not args.no_derived_edges:
                    edges.extend(derived_edges_from_nodes(nodes, sample_id))

                for n in nodes:
                    fnodes.write(json.dumps(n, ensure_ascii=False) + "\n")
                    node_type_cnt[n["node_type"]] += 1
                for e in edges:
                    fedges.write(json.dumps(e, ensure_ascii=False) + "\n")
                    edge_kind_cnt[e["edge_kind"]] += 1
                    edge_type_cnt[e["edge_type"]] += 1

                total_nodes += len(nodes)
                total_edges += len(edges)
            except Exception as ex:
                errors.append({"sample_id": sample_id, "json_path": str(json_path), "error": str(ex)})

    stats = {
        "index": str(index_path),
        "sample_count": sample_cnt,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "node_type_counts": dict(node_type_cnt),
        "edge_kind_counts": dict(edge_kind_cnt),
        "edge_type_counts": dict(edge_type_cnt.most_common(50)),
        "error_count": len(errors),
        "errors": errors[:20],
        "nodes_out": str(nodes_out),
        "edges_out": str(edges_out),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", nodes_out)
    print("wrote", edges_out)
    print("wrote", stats_path)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
