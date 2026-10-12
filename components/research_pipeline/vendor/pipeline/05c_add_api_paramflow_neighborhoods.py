#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05c_add_api_paramflow_neighborhoods.py

第 2 阶段：把 api_call 的 CALL 前后窗口 callctx 升级为参数 def-use 子图 paramflow。

核心目标：
  对每个 api_call anchor，找到 CALL 指令相关的：
    1) 参数寄存器来源：rdi/rsi/rdx/rcx/r8/r9 等在 CALL 前的定义点；
    2) 返回值使用：rax/eax 等在 CALL 后的使用点；
    3) 相关 DFGSem / BlockSem / InstrSem 节点；
  然后加入 low_nodes，生成 paramflow 版本 neighborhood。

输入：
  --nodes            usg_nodes.jsonl
  --edges            usg_edges.jsonl
  --neighborhoods-in anchor_neighborhoods_balanced_40k_api_filtered_callctx.jsonl
输出：
  --neighborhoods-out anchor_neighborhoods_balanced_40k_api_filtered_paramflow.jsonl
  --stats             neighborhood_api_paramflow_stats.json

特点：
  1. 不覆盖原文件；
  2. 非 api_call anchor 原样保留；
  3. 优先沿 def-use / dataflow 边找参数来源；
  4. 如果边信息不足，则退化到更宽的 CALL 前后寄存器过滤窗口；
  5. 控制 max_low_nodes，避免邻域爆炸。
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict, deque


X86_PARAM_REGS = [
    "rdi", "edi", "di",
    "rsi", "esi", "si",
    "rdx", "edx", "dx",
    "rcx", "ecx", "cx",
    "r8", "r8d", "r8w",
    "r9", "r9d", "r9w",
]
X86_RET_REGS = ["rax", "eax", "ax", "al"]

AARCH64_PARAM_REGS = [f"x{i}" for i in range(8)] + [f"w{i}" for i in range(8)]
AARCH64_RET_REGS = ["x0", "w0"]

RELEVANT_EDGE_KEYWORDS = [
    "def_use", "def-use", "data_flow", "dataflow", "dfg",
    "use", "definition", "entry_use", "entry-use",
    "call_instruction", "call_behavior_context",
    "api_type_propagation", "api_interface_propagation",
]


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_attrs(n):
    a = n.get("attrs", {})
    return a if isinstance(a, dict) else {}


def node_id(n):
    return n.get("node_id") or n.get("id") or ""


def node_type(n):
    return n.get("node_type") or get_attrs(n).get("node_type") or "unknown"


def get_sample(n):
    a = get_attrs(n)
    return n.get("sample_id") or a.get("sample_id") or ""


def get_function(n):
    a = get_attrs(n)
    return (
        n.get("function_entry")
        or n.get("function_entry_norm")
        or a.get("function_entry")
        or a.get("func_entry")
        or a.get("function")
        or ""
    )


def get_seq_or_addr(n):
    a = get_attrs(n)

    for k in ["seq", "sequence", "index", "order", "instr_index", "n_pcode_seq"]:
        v = a.get(k)
        try:
            return int(v)
        except Exception:
            pass

    for k in ["address", "addr", "ea", "location"]:
        v = a.get(k)
        if v is None:
            continue
        s = str(v)
        m = re.search(r"0x[0-9a-fA-F]+|[0-9a-fA-F]{6,}", s)
        if m:
            try:
                return int(m.group(0), 16)
            except Exception:
                pass

    raw = n.get("raw_id") or a.get("raw_id") or ""
    m = re.search(r"0x[0-9a-fA-F]+|[0-9a-fA-F]{6,}", str(raw))
    if m:
        try:
            return int(m.group(0), 16)
        except Exception:
            pass

    return 0


def lower_text(obj):
    try:
        return json.dumps(obj, ensure_ascii=False).lower()
    except Exception:
        return str(obj).lower()


def is_instr(n):
    return node_type(n) == "InstrSem"


def is_block_or_dfg(n):
    return node_type(n) in {"BlockSem", "DFGSem", "CFGSem"}


def is_call_like_node(n):
    a = get_attrs(n)
    txt = lower_text(a)
    op = str(a.get("opcode") or a.get("op") or a.get("mnemonic") or "").upper()
    return op == "CALL" or "call" in txt or "callee" in txt or "api_name" in txt or "api_target" in txt


def normalize_reg(r):
    s = str(r).lower().strip()
    s = s.replace("reg:", "").replace("%", "")
    return s


def collect_regs_from_value(v):
    regs = []
    if v is None:
        return regs
    if isinstance(v, str):
        # 只取常见寄存器名，避免把任意字符串当寄存器
        candidates = re.findall(r"\b(?:r(?:ax|bx|cx|dx|si|di|sp|bp)|e(?:ax|bx|cx|dx|si|di|sp|bp)|r[0-9]+d?|x[0-9]+|w[0-9]+)\b", v.lower())
        regs.extend(candidates)
    elif isinstance(v, list):
        for x in v:
            regs.extend(collect_regs_from_value(x))
    elif isinstance(v, dict):
        for x in v.values():
            regs.extend(collect_regs_from_value(x))
    return [normalize_reg(x) for x in regs]


def read_write_regs(n):
    a = get_attrs(n)
    reads, writes = [], []
    read_keys = ["read_set", "reads", "input_regs", "use_set", "uses", "src_regs", "read_regs"]
    write_keys = ["write_set", "writes", "output_regs", "def_set", "defs", "dst_regs", "write_regs"]

    for k in read_keys:
        if k in a:
            reads.extend(collect_regs_from_value(a.get(k)))
    for k in write_keys:
        if k in a:
            writes.extend(collect_regs_from_value(a.get(k)))

    # 如果没有明确字段，则从文本中粗略提取；同时放入 reads，作为 fallback。
    if not reads and not writes:
        regs = collect_regs_from_value(a)
        reads.extend(regs)

    return set(reads), set(writes)


def node_mentions_regs(n, regs):
    regs = {normalize_reg(x) for x in regs}
    reads, writes = read_write_regs(n)
    if reads & regs or writes & regs:
        return True
    txt = lower_text(get_attrs(n))
    return any(re.search(r"\b" + re.escape(r) + r"\b", txt) for r in regs)


def node_writes_regs(n, regs):
    regs = {normalize_reg(x) for x in regs}
    _, writes = read_write_regs(n)
    return bool(writes & regs)


def node_reads_regs(n, regs):
    regs = {normalize_reg(x) for x in regs}
    reads, _ = read_write_regs(n)
    return bool(reads & regs)


def edge_text(e):
    return lower_text({
        "edge_kind": e.get("edge_kind"),
        "edge_type": e.get("edge_type"),
        "type": e.get("type"),
        "kind": e.get("kind"),
        "evidence_type": e.get("evidence_type"),
        "relation_type": e.get("relation_type"),
        "rule_id": e.get("rule_id"),
    })


def is_relevant_edge(e):
    txt = edge_text(e)
    return any(k in txt for k in RELEVANT_EDGE_KEYWORDS)


def edge_source(e):
    return e.get("source_node") or e.get("from") or e.get("src") or ""


def edge_target(e):
    return e.get("target_node") or e.get("to") or e.get("dst") or ""


def compact_reason(kind, call_id, extra=None):
    r = {"kind": kind, "call_node": call_id}
    if extra:
        r.update(extra)
    return r


def collect_api_anchor_seeds(neighborhoods_in):
    """
    第一遍读 neighborhoods：
      - 收集 api_call anchor 的 source_node / low_nodes 中 CALL-like 节点；
      - 收集相关 function key，方便只索引相关函数的节点。
    """
    seed_ids = set()
    call_like_ids = set()
    func_keys = set()
    anchor_count = 0

    for d in load_jsonl(neighborhoods_in):
        if d.get("evidence_type") != "api_call":
            continue
        anchor_count += 1

        src = d.get("source_node")
        if src:
            seed_ids.add(src)

        sample = d.get("sample_id", "")
        func = d.get("function_entry", "") or d.get("func_entry", "")
        if sample or func:
            func_keys.add((sample, func))

        for n in d.get("low_nodes", []) or []:
            nid = node_id(n)
            if nid:
                seed_ids.add(nid)
            s = get_sample(n) or sample
            f = get_function(n) or func
            if s or f:
                func_keys.add((s, f))
            if is_instr(n) and is_call_like_node(n) and nid:
                call_like_ids.add(nid)

    return seed_ids, call_like_ids, func_keys, anchor_count


def load_seed_nodes(nodes_path, seed_ids):
    seed_nodes = {}
    seed_func_keys = set()
    for n in load_jsonl(nodes_path):
        nid = node_id(n)
        if nid in seed_ids:
            seed_nodes[nid] = n
            s, f = get_sample(n), get_function(n)
            if s or f:
                seed_func_keys.add((s, f))
    return seed_nodes, seed_func_keys


def load_relevant_nodes(nodes_path, func_keys, seed_ids, include_types=("InstrSem", "DFGSem", "BlockSem", "CFGSem")):
    """
    第二遍读 nodes：
      - 存储相关函数中的 InstrSem / DFGSem / BlockSem / CFGSem；
      - 存储 seed 节点。
    """
    nodes = {}
    instr_by_func = defaultdict(list)
    node_func = {}

    include_types = set(include_types)

    for n in load_jsonl(nodes_path):
        nid = node_id(n)
        nt = node_type(n)
        s, f = get_sample(n), get_function(n)
        key = (s, f)

        keep = False
        if nid in seed_ids:
            keep = True
        elif key in func_keys and nt in include_types:
            keep = True

        if not keep:
            continue

        nodes[nid] = n
        node_func[nid] = key
        if nt == "InstrSem":
            instr_by_func[key].append(n)

    pos = {}
    for key, arr in instr_by_func.items():
        arr.sort(key=get_seq_or_addr)
        for i, n in enumerate(arr):
            pos[node_id(n)] = (key, i)

    return nodes, instr_by_func, pos, node_func


def load_relevant_edges(edges_path, candidate_node_ids):
    """
    只保留相关节点之间，或一端为相关节点的 def-use/dataflow/call 相关边。
    """
    adj = defaultdict(list)
    edge_count = 0
    kept = 0
    candidate_node_ids = set(candidate_node_ids)

    for e in load_jsonl(edges_path):
        edge_count += 1
        if not is_relevant_edge(e):
            continue
        s, t = edge_source(e), edge_target(e)
        if not s or not t:
            continue
        if s in candidate_node_ids or t in candidate_node_ids:
            adj[s].append((t, e))
            adj[t].append((s, e))
            kept += 1

    return adj, {"edges_seen": edge_count, "edges_kept": kept}


def find_call_nodes_for_anchor(d, nodes):
    call_ids = []

    src = d.get("source_node")
    if src and src in nodes and is_call_like_node(nodes[src]):
        call_ids.append(src)

    for n in d.get("low_nodes", []) or []:
        nid = node_id(n)
        full = nodes.get(nid) or n
        if nid and is_instr(full) and is_call_like_node(full):
            call_ids.append(nid)

    # fallback：source_node 即使不是 call-like，也作为起点试一试
    if not call_ids and src:
        call_ids.append(src)

    # 去重
    out = []
    seen = set()
    for x in call_ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def add_node_if_available(nid, nodes, existing_ids, added_nodes, added_reasons, reason):
    if not nid or nid in existing_ids:
        return False
    n = nodes.get(nid)
    if not n:
        return False
    added_nodes.append(n)
    added_reasons[nid] = reason
    existing_ids.add(nid)
    return True


def paramflow_from_window(call_id, nodes, instr_by_func, pos, param_regs, ret_regs, pre_scan, post_scan, existing_ids):
    added = []
    reasons = {}
    if call_id not in pos:
        return added, reasons

    key, idx = pos[call_id]
    arr = instr_by_func.get(key, [])
    if not arr:
        return added, reasons

    # CALL 前向后扫描参数寄存器定义点
    start = max(0, idx - pre_scan)
    for j in range(idx - 1, start - 1, -1):
        n = arr[j]
        nid = node_id(n)
        if node_writes_regs(n, param_regs) or node_mentions_regs(n, param_regs):
            if nid not in existing_ids:
                added.append(n)
                reasons[nid] = compact_reason("param_reg_pre_call_window", call_id, {"distance": idx - j})
                existing_ids.add(nid)

    # CALL 后扫描返回值使用点
    end = min(len(arr), idx + post_scan + 1)
    for j in range(idx + 1, end):
        n = arr[j]
        nid = node_id(n)
        if node_reads_regs(n, ret_regs) or node_mentions_regs(n, ret_regs):
            if nid not in existing_ids:
                added.append(n)
                reasons[nid] = compact_reason("return_reg_post_call_window", call_id, {"distance": j - idx})
                existing_ids.add(nid)

    return added, reasons


def paramflow_from_edges(call_id, nodes, adj, param_regs, ret_regs, max_hops, max_nodes, existing_ids):
    added = []
    reasons = {}

    q = deque()
    q.append((call_id, 0))
    seen = {call_id}

    while q and len(added) < max_nodes:
        cur, depth = q.popleft()
        if depth >= max_hops:
            continue

        for nxt, e in adj.get(cur, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, depth + 1))

            n = nodes.get(nxt)
            if not n:
                continue

            nt = node_type(n)
            if nt not in {"InstrSem", "DFGSem", "BlockSem", "CFGSem"}:
                continue

            relevant = False
            kind = "edge_bfs_related"

            if node_mentions_regs(n, param_regs):
                relevant = True
                kind = "param_reg_defuse_edge"
            elif node_mentions_regs(n, ret_regs):
                relevant = True
                kind = "return_reg_defuse_edge"
            elif nt == "DFGSem":
                # DFGSem 本身就是参数流的重要结构节点，保守纳入少量。
                relevant = True
                kind = "dfg_context_edge"
            elif nt == "BlockSem" and depth == 1:
                relevant = True
                kind = "block_context_edge"

            if relevant and nxt not in existing_ids:
                added.append(n)
                reasons[nxt] = compact_reason(kind, call_id, {"hop": depth + 1, "edge_type": e.get("edge_type") or e.get("edge_kind")})
                existing_ids.add(nxt)

            if len(added) >= max_nodes:
                break

    return added, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="usg_nodes.jsonl")
    ap.add_argument("--edges", required=True, help="usg_edges.jsonl")
    ap.add_argument("--neighborhoods-in", required=True)
    ap.add_argument("--neighborhoods-out", required=True)
    ap.add_argument("--stats", required=True)

    ap.add_argument("--arch", default="x86_64", choices=["x86_64", "aarch64", "both"])
    ap.add_argument("--pre-scan", type=int, default=25, help="CALL 前向后扫描多少条 InstrSem")
    ap.add_argument("--post-scan", type=int, default=15, help="CALL 后扫描多少条 InstrSem")
    ap.add_argument("--max-hops", type=int, default=2, help="def-use/dataflow 边 BFS 最大跳数")
    ap.add_argument("--max-flow-nodes", type=int, default=40, help="每个 CALL 从边 BFS 最多加入多少节点")
    ap.add_argument("--max-low-nodes", type=int, default=120)
    args = ap.parse_args()

    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    neigh_in = Path(args.neighborhoods_in)
    neigh_out = Path(args.neighborhoods_out)
    stats_out = Path(args.stats)

    neigh_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.parent.mkdir(parents=True, exist_ok=True)

    if args.arch == "x86_64":
        param_regs = X86_PARAM_REGS
        ret_regs = X86_RET_REGS
    elif args.arch == "aarch64":
        param_regs = AARCH64_PARAM_REGS
        ret_regs = AARCH64_RET_REGS
    else:
        param_regs = X86_PARAM_REGS + AARCH64_PARAM_REGS
        ret_regs = X86_RET_REGS + AARCH64_RET_REGS

    stats = Counter()

    # Pass 1: 从 neighborhood 收集 api_call 种子
    seed_ids, call_like_ids, func_keys, api_anchor_count = collect_api_anchor_seeds(neigh_in)
    stats["api_anchor_count"] = api_anchor_count
    stats["initial_seed_ids"] = len(seed_ids)
    stats["initial_call_like_ids"] = len(call_like_ids)
    stats["initial_func_keys"] = len(func_keys)

    # Pass 2a: 读取 seed 节点，补充 function keys
    seed_nodes, seed_func_keys = load_seed_nodes(nodes_path, seed_ids)
    func_keys |= seed_func_keys
    stats["seed_nodes_found"] = len(seed_nodes)
    stats["func_keys_after_seed_scan"] = len(func_keys)

    # Pass 2b: 读取相关函数的低/中层节点
    nodes, instr_by_func, pos, node_func = load_relevant_nodes(nodes_path, func_keys, seed_ids)
    stats["relevant_nodes_loaded"] = len(nodes)
    stats["relevant_functions_loaded"] = len(instr_by_func)
    stats["relevant_instr_loaded"] = sum(len(v) for v in instr_by_func.values())

    # Pass 3: 读取相关 def-use / dataflow 边
    adj, edge_stats = load_relevant_edges(edges_path, nodes.keys())
    stats.update(edge_stats)
    stats["adj_nodes"] = len(adj)

    # Pass 4: 生成输出 neighborhoods
    with neigh_in.open("r", encoding="utf-8") as fin, neigh_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            d = json.loads(line)
            stats["total_records"] += 1

            if d.get("evidence_type") != "api_call":
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                continue

            stats["api_call_records"] += 1

            low_nodes = d.get("low_nodes", []) or []
            existing_ids = {node_id(n) for n in low_nodes if node_id(n)}

            call_ids = find_call_nodes_for_anchor(d, nodes)
            if call_ids:
                stats["api_call_with_call_node"] += 1
            else:
                stats["api_call_no_call_node"] += 1

            added_nodes = []
            added_reasons = {}

            for cid in call_ids:
                # 1) def-use/dataflow 边
                a1, r1 = paramflow_from_edges(
                    cid, nodes, adj, param_regs, ret_regs,
                    args.max_hops, args.max_flow_nodes, existing_ids
                )
                added_nodes.extend(a1)
                added_reasons.update(r1)

                # 2) CALL 前后寄存器过滤窗口作为补充
                a2, r2 = paramflow_from_window(
                    cid, nodes, instr_by_func, pos,
                    param_regs, ret_regs,
                    args.pre_scan, args.post_scan,
                    existing_ids
                )
                added_nodes.extend(a2)
                added_reasons.update(r2)

            # 加入 low_nodes，保留原始 low_nodes 优先
            if added_nodes:
                stats["api_call_added_paramflow"] += 1
                stats["added_nodes_total"] += len(added_nodes)
                low_nodes = low_nodes + added_nodes
            else:
                stats["api_call_no_paramflow_added"] += 1

            if len(low_nodes) > args.max_low_nodes:
                stats["low_nodes_truncated"] += 1
                low_nodes = low_nodes[:args.max_low_nodes]

            d["low_nodes"] = low_nodes
            d["api_call_paramflow_context"] = {
                "version": "paramflow_v1",
                "arch": args.arch,
                "param_regs": param_regs,
                "ret_regs": ret_regs,
                "call_node_ids": call_ids,
                "added_count": len(added_nodes),
                "added_reasons": added_reasons,
                "pre_scan": args.pre_scan,
                "post_scan": args.post_scan,
                "max_hops": args.max_hops,
            }

            fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    stats["avg_added_nodes_per_api_call"] = (
        stats["added_nodes_total"] / stats["api_call_records"]
        if stats["api_call_records"] else 0
    )

    stats_out.write_text(json.dumps(dict(stats), indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", neigh_out)
    print("wrote", stats_out)
    print(json.dumps(dict(stats), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
