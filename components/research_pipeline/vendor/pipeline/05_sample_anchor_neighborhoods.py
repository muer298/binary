#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_sample_anchor_neighborhoods.py

Task 2 Day-2 Step 1:
Build anchor-centered local neighborhoods from USG nodes/edges/anchors.

Input:
  task2_exp/data/graphs/usg_nodes.jsonl
  task2_exp/data/graphs/usg_edges.jsonl
  task2_exp/data/anchors/anchors.jsonl

Output:
  task2_exp/data/anchors/anchor_neighborhoods.jsonl
  task2_exp/reports/neighborhood_stats.json

Design:
  - Each anchor has a low-side endpoint and a high-side endpoint.
  - Neighborhoods are expanded on the USG topology.
  - k is adaptive by node type:
      InstrSem / BlockSem / high-level object: k=1
      CFGSem / DFGSem: k=2
  - Edges are treated as undirected for neighborhood collection by default,
    because evidence/constraint links are used to recover local semantic context.
"""

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path


HIGH_TYPES = {"TypeSem", "InterfaceSem", "APISem", "LayoutSem"}
STRUCT_TYPES = {"CFGSem", "DFGSem"}
LOW_TYPES = {"InstrSem", "BlockSem"}


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compact_node(n, include_attrs=True):
    out = {
        "node_id": n.get("node_id", ""),
        "node_type": n.get("node_type", "unknown"),
        "layer": n.get("layer", "unknown"),
        "raw_id": n.get("raw_id", ""),
        "function_entry": n.get("function_entry", ""),
        "function_entry_norm": n.get("function_entry_norm", ""),
    }
    if include_attrs:
        out["attrs"] = n.get("attrs", {})
    return out


def compact_edge(e):
    return {
        "edge_id": e.get("edge_id", ""),
        "edge_kind": e.get("edge_kind", ""),
        "edge_type": e.get("edge_type", ""),
        "source_node": e.get("source_node", ""),
        "target_node": e.get("target_node", ""),
        "strength": e.get("strength", None),
        "source_path": e.get("source_path", ""),
        "source_resolved": e.get("source_resolved", False),
        "target_resolved": e.get("target_resolved", False),
    }


def ensure_stub(node_id, node_map):
    if not node_id:
        return
    if node_id not in node_map:
        node_type = "EvidenceStub" if "EvidenceStub" in node_id else "Unresolved"
        node_map[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            "layer": "stub",
            "raw_id": node_id,
            "function_entry": "",
            "function_entry_norm": "",
            "attrs": {},
        }


def default_k(node, side, args):
    nt = node.get("node_type", "unknown")
    if nt in STRUCT_TYPES:
        return args.struct_k
    if nt in LOW_TYPES:
        return args.low_k
    if nt in HIGH_TYPES:
        return args.high_k
    if nt == "EvidenceStub":
        return args.stub_k
    return args.default_k


def bfs(start, k, adj, node_map, max_nodes):
    if not start:
        return set(), {}
    if start not in node_map:
        return {start}, {start: 0}

    seen = {start}
    depth = {start: 0}
    q = deque([start])

    while q:
        u = q.popleft()
        if depth[u] >= k:
            continue
        for v, _eid in adj.get(u, []):
            if v in seen:
                continue
            seen.add(v)
            depth[v] = depth[u] + 1
            q.append(v)
            if max_nodes and len(seen) >= max_nodes:
                return seen, depth
    return seen, depth


def internal_edges(node_set, edge_map, incident_edges):
    out = []
    used = set()
    for nid in node_set:
        for eid in incident_edges.get(nid, []):
            if eid in used:
                continue
            e = edge_map.get(eid)
            if not e:
                continue
            if e.get("source_node") in node_set and e.get("target_node") in node_set:
                out.append(e)
                used.add(eid)
    return out


def cross_edges(a_set, b_set, edge_map, incident_edges):
    out = []
    used = set()
    for nid in a_set:
        for eid in incident_edges.get(nid, []):
            if eid in used:
                continue
            e = edge_map.get(eid)
            if not e:
                continue
            s = e.get("source_node")
            t = e.get("target_node")
            if (s in a_set and t in b_set) or (s in b_set and t in a_set):
                out.append(e)
                used.add(eid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--low-k", type=int, default=1)
    ap.add_argument("--high-k", type=int, default=1)
    ap.add_argument("--struct-k", type=int, default=2)
    ap.add_argument("--stub-k", type=int, default=0)
    ap.add_argument("--default-k", type=int, default=1)
    ap.add_argument("--max-nodes-per-side", type=int, default=80)
    ap.add_argument("--no-node-attrs", action="store_true")
    args = ap.parse_args()

    nodes_path = Path(args.nodes).resolve()
    edges_path = Path(args.edges).resolve()
    anchors_path = Path(args.anchors).resolve()
    out_path = Path(args.out).resolve()
    stats_path = Path(args.stats).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    node_map = {}
    for n in load_jsonl(nodes_path):
        node_map[n["node_id"]] = n

    edge_map = {}
    adj = defaultdict(list)
    incident = defaultdict(list)

    for e in load_jsonl(edges_path):
        eid = e.get("edge_id") or f"edge_{len(edge_map)}"
        e["edge_id"] = eid
        s = e.get("source_node", "")
        t = e.get("target_node", "")
        ensure_stub(s, node_map)
        ensure_stub(t, node_map)

        edge_map[eid] = e
        if s and t:
            adj[s].append((t, eid))
            adj[t].append((s, eid))
            incident[s].append(eid)
            incident[t].append(eid)

    n_out = 0
    cnt_anchor_type = Counter()
    cnt_low_type = Counter()
    cnt_high_type = Counter()
    low_sizes = []
    high_sizes = []
    cross_sizes = []
    missing_endpoints = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for a in load_jsonl(anchors_path):
            src = a.get("source_node", "")
            tgt = a.get("target_node", "")
            ensure_stub(src, node_map)
            ensure_stub(tgt, node_map)

            src_node = node_map.get(src, {})
            tgt_node = node_map.get(tgt, {})

            if not src_node or not tgt_node:
                missing_endpoints += 1

            src_k = default_k(src_node, "low", args)
            tgt_k = default_k(tgt_node, "high", args)

            low_set, low_depth = bfs(src, src_k, adj, node_map, args.max_nodes_per_side)
            high_set, high_depth = bfs(tgt, tgt_k, adj, node_map, args.max_nodes_per_side)

            le = internal_edges(low_set, edge_map, incident)
            he = internal_edges(high_set, edge_map, incident)
            ce = cross_edges(low_set, high_set, edge_map, incident)

            rec = {
                "anchor_id": a.get("anchor_id"),
                "sample_id": a.get("sample_id"),
                "evidence_type": a.get("evidence_type", "unknown"),
                "strength": a.get("strength", None),
                "function_entry": a.get("function_entry", ""),
                "source_node": src,
                "target_node": tgt,
                "source_type": src_node.get("node_type", "unknown"),
                "target_type": tgt_node.get("node_type", "unknown"),
                "low_k": src_k,
                "high_k": tgt_k,
                "low_nodes": [compact_node(node_map[x], include_attrs=not args.no_node_attrs) for x in sorted(low_set)],
                "high_nodes": [compact_node(node_map[x], include_attrs=not args.no_node_attrs) for x in sorted(high_set)],
                "low_edges": [compact_edge(e) for e in le],
                "high_edges": [compact_edge(e) for e in he],
                "cross_edges": [compact_edge(e) for e in ce],
                "low_depth": low_depth,
                "high_depth": high_depth,
                "counts": {
                    "low_nodes": len(low_set),
                    "high_nodes": len(high_set),
                    "low_edges": len(le),
                    "high_edges": len(he),
                    "cross_edges": len(ce),
                },
                "trace": a.get("trace", {}),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

            cnt_anchor_type[rec["evidence_type"]] += 1
            cnt_low_type[rec["source_type"]] += 1
            cnt_high_type[rec["target_type"]] += 1
            low_sizes.append(len(low_set))
            high_sizes.append(len(high_set))
            cross_sizes.append(len(ce))

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    stats = {
        "nodes": str(nodes_path),
        "edges": str(edges_path),
        "anchors": str(anchors_path),
        "neighborhood_count": n_out,
        "anchor_type_counts": dict(cnt_anchor_type),
        "source_node_type_counts": dict(cnt_low_type),
        "target_node_type_counts": dict(cnt_high_type),
        "avg_low_nodes": avg(low_sizes),
        "avg_high_nodes": avg(high_sizes),
        "avg_cross_edges": avg(cross_sizes),
        "max_low_nodes": max(low_sizes) if low_sizes else 0,
        "max_high_nodes": max(high_sizes) if high_sizes else 0,
        "missing_endpoint_count": missing_endpoints,
        "params": {
            "low_k": args.low_k,
            "high_k": args.high_k,
            "struct_k": args.struct_k,
            "stub_k": args.stub_k,
            "default_k": args.default_k,
            "max_nodes_per_side": args.max_nodes_per_side,
        },
        "out": str(out_path),
    }

    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out_path)
    print("wrote", stats_path)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
