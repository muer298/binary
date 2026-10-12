#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05e_repair_api_direct_only.py

API EvidenceStub 修复的高精度版本：只使用 EvidenceStub/APISem 一跳或两跳能直接连到的 CALL InstrSem，
不使用“同 sample fallback CALL”。用于减少 05d repaired strict 中的噪声。
"""
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict

FLOW_KEYS = ["param_def_use", "return_use", "def_use", "data_flow", "dfg"]
REAL_LOW_TYPES = {"InstrSem", "DFGSem", "BlockSem", "CFGSem", "LayoutSem"}

def read_jsonl(p):
    with Path(p).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def dump_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def attrs(n):
    return n.get("attrs") if isinstance(n.get("attrs"), dict) else {}

def node_type_from_id(nid):
    parts = str(nid or "").split("::")
    return parts[-2] if len(parts) >= 3 else ""

def edge_type(e):
    a = e.get("attrs") if isinstance(e.get("attrs"), dict) else {}
    return str(e.get("edge_type") or a.get("relation_type") or a.get("evidence_type") or e.get("evidence_type") or e.get("edge_kind") or "unknown")

def is_call(n):
    if not n or n.get("node_type") != "InstrSem":
        return False
    a = attrs(n)
    op = str(a.get("opcode") or n.get("opcode") or "").upper()
    return "CALL" in op or "CALL" in json.dumps(n, ensure_ascii=False).upper()

def real_low_hit(rec, node_by_id):
    hits = []
    for n in rec.get("low_nodes", []) or []:
        nid = n.get("node_id")
        nt = n.get("node_type") or node_type_from_id(nid)
        if nt in REAL_LOW_TYPES:
            hits.append(nid)
    src = rec.get("source_node")
    if src:
        n = node_by_id.get(src)
        nt = n.get("node_type") if n else node_type_from_id(src)
        if nt in REAL_LOW_TYPES:
            hits.append(src)
    return [x for x in hits if x]

def load_nodes(path):
    return {n.get("node_id"): n for n in read_jsonl(path) if n.get("node_id")}

def load_adj(path):
    adj = defaultdict(list); cnt = Counter()
    for e in read_jsonl(path):
        s, t = e.get("source_node"), e.get("target_node")
        if not s or not t:
            continue
        et = edge_type(e); cnt[et] += 1
        adj[s].append((t, et)); adj[t].append((s, et))
    return adj, cnt

def direct_call_candidates(src, tgt, adj, node_by_id, max_calls=3):
    out, seen = [], set()
    for seed in [x for x in [src, tgt] if x]:
        for nb, et in adj.get(seed, []):
            if nb not in seen and is_call(node_by_id.get(nb)):
                out.append(nb); seen.add(nb)
                if len(out) >= max_calls:
                    return out
        for nb, et in adj.get(seed, []):
            for nb2, et2 in adj.get(nb, []):
                if nb2 not in seen and is_call(node_by_id.get(nb2)):
                    out.append(nb2); seen.add(nb2)
                    if len(out) >= max_calls:
                        return out
    return out

def flow_neighbors(call_ids, adj, node_by_id, max_flow=12):
    out, seen = [], set(call_ids)
    for cid in call_ids:
        for nb, et in adj.get(cid, []):
            etl = et.lower()
            if not any(k in etl for k in FLOW_KEYS):
                continue
            if nb in seen:
                continue
            n = node_by_id.get(nb)
            if n and n.get("node_type") in REAL_LOW_TYPES:
                out.append(nb); seen.add(nb)
                if len(out) >= max_flow:
                    return out
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--neighborhoods-in", required=True)
    ap.add_argument("--neighborhoods-out", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--max-call-candidates", type=int, default=3)
    ap.add_argument("--max-flow-nodes", type=int, default=12)
    ap.add_argument("--max-low-nodes", type=int, default=80)
    args = ap.parse_args()

    node_by_id = load_nodes(args.nodes)
    adj, edge_cnt = load_adj(args.edges)
    rows, stat, added_types = [], Counter(), Counter()
    examples = []

    for rec in read_jsonl(args.neighborhoods_in):
        stat["records"] += 1
        if rec.get("evidence_type") != "api_call":
            rows.append(rec); continue
        stat["api_call_records"] += 1
        if real_low_hit(rec, node_by_id):
            stat["api_call_already_real_low"] += 1
            rows.append(rec); continue
        stat["api_call_need_repair"] += 1
        calls = direct_call_candidates(rec.get("source_node"), rec.get("target_node"), adj, node_by_id, args.max_call_candidates)
        if not calls:
            stat["api_call_no_direct_call"] += 1
            rows.append(rec); continue
        flow = flow_neighbors(calls, adj, node_by_id, args.max_flow_nodes)
        add_ids = calls + flow
        existing = {n.get("node_id") for n in (rec.get("low_nodes", []) or []) if n.get("node_id")}
        new_low = list(rec.get("low_nodes", []) or [])
        added = 0
        for nid in add_ids:
            if nid in existing:
                continue
            n = node_by_id.get(nid)
            if not n:
                continue
            new_low.append(n); existing.add(nid); added += 1
            added_types[n.get("node_type", "unknown")] += 1
            if len(new_low) >= args.max_low_nodes:
                break
        if added > 0:
            out = dict(rec); out["low_nodes"] = new_low; out["api_direct_only_repaired"] = True
            out["api_direct_only_repair_info"] = {"call_candidates": len(calls), "flow_nodes": len(flow), "added": added}
            rows.append(out); stat["api_call_repaired"] += 1; stat["total_added_nodes"] += added
            if len(examples) < 20:
                examples.append({"anchor_id": rec.get("anchor_id"), "source_node": rec.get("source_node"), "target_node": rec.get("target_node"), "calls": len(calls), "flow": len(flow), "added": added})
        else:
            stat["api_call_no_new_nodes"] += 1; rows.append(rec)
    write_jsonl(args.neighborhoods_out, rows)
    dump_json(args.stats, {"stats": dict(stat), "added_type_counts": dict(added_types), "edge_type_top50": edge_cnt.most_common(50), "examples": examples, "params": vars(args)})

if __name__ == "__main__":
    main()
