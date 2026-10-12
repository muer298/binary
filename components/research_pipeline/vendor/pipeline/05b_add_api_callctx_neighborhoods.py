#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05b_add_api_callctx_neighborhoods.py

给 api_call anchor 的 low-side neighborhood 增加调用点上下文。

用途：
  05_sample_anchor_neighborhoods.py 已经围绕 source_node/target_node 生成 low_nodes/high_nodes。
  但 api_call 的语义常常依赖 CALL 前的参数准备指令和 CALL 后的返回值使用。
  这个脚本会对 evidence_type == api_call 的记录额外加入同函数内 CALL 前 PRE_N 条、后 POST_N 条 InstrSem。

输入：
  --nodes      usg_nodes.jsonl
  --in-neigh   anchor_neighborhoods_balanced_40k_api_filtered.jsonl
输出：
  --out-neigh  anchor_neighborhoods_balanced_40k_api_filtered_callctx.jsonl
  --stats      neighborhood_api_call_callctx_stats.json

说明：
  这是 callctx 版本主线使用的脚本。它是“顺序窗口增强”，不是完整 parameter-flow def-use 子图。
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict, Counter


def load_jsonl(path):
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def attrs(n):
    a = n.get('attrs', {})
    return a if isinstance(a, dict) else {}


def get_sample(n):
    return n.get('sample_id') or attrs(n).get('sample_id') or ''


def get_func(n):
    a = attrs(n)
    return n.get('function_entry') or n.get('function_entry_norm') or a.get('function_entry') or a.get('func_entry') or ''


def seq_key(n):
    a = attrs(n)
    for k in ['seq', 'sequence', 'index', 'order']:
        v = a.get(k)
        try:
            return int(v)
        except Exception:
            pass
    for k in ['address', 'addr', 'ea', 'location']:
        v = a.get(k)
        if v is None:
            continue
        m = re.search(r'0x[0-9a-fA-F]+|[0-9a-fA-F]{6,}', str(v))
        if m:
            try:
                return int(m.group(0), 16)
            except Exception:
                pass
    return 0


def is_instr(n):
    return n.get('node_type') == 'InstrSem'


def is_call_like(n):
    a = attrs(n)
    txt = json.dumps(a, ensure_ascii=False).lower()
    return str(a.get('opcode', '')).upper() == 'CALL' or 'call' in txt or 'callee' in txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', required=True)
    ap.add_argument('--in-neigh', required=True)
    ap.add_argument('--out-neigh', required=True)
    ap.add_argument('--stats', required=True)
    ap.add_argument('--pre-n', type=int, default=5)
    ap.add_argument('--post-n', type=int, default=2)
    ap.add_argument('--max-low-nodes', type=int, default=80)
    args = ap.parse_args()

    node_by_id = {}
    instr_by_func = defaultdict(list)

    for n in load_jsonl(args.nodes):
        nid = n.get('node_id')
        if not nid:
            continue
        node_by_id[nid] = n
        if is_instr(n):
            instr_by_func[(get_sample(n), get_func(n))].append(n)

    for key in list(instr_by_func.keys()):
        instr_by_func[key].sort(key=seq_key)

    pos = {}
    for key, arr in instr_by_func.items():
        for i, n in enumerate(arr):
            pos[n.get('node_id')] = (key, i)

    out_path = Path(args.out_neigh)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    added_counts = []

    with Path(args.in_neigh).open('r', encoding='utf-8') as fin, out_path.open('w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            d = json.loads(line)
            stats['total'] += 1

            if d.get('evidence_type') != 'api_call':
                fout.write(json.dumps(d, ensure_ascii=False) + '\n')
                continue

            stats['api_call'] += 1
            source = d.get('source_node')
            low_nodes = d.get('low_nodes', []) or []
            existing = {x.get('node_id') for x in low_nodes if x.get('node_id')}

            candidates = []
            if source in node_by_id:
                candidates.append(node_by_id[source])
            for x in low_nodes:
                nid = x.get('node_id')
                if nid in node_by_id and is_instr(node_by_id[nid]) and is_call_like(node_by_id[nid]):
                    candidates.append(node_by_id[nid])

            added = []
            for src in candidates:
                sid = src.get('node_id')
                if sid not in pos:
                    continue
                key, i = pos[sid]
                arr = instr_by_func.get(key, [])
                lo = max(0, i - args.pre_n)
                hi = min(len(arr), i + args.post_n + 1)
                for n in arr[lo:hi]:
                    nid = n.get('node_id')
                    if nid and nid not in existing:
                        low_nodes.append(n)
                        existing.add(nid)
                        added.append(nid)

            if len(low_nodes) > args.max_low_nodes:
                low_nodes = low_nodes[:args.max_low_nodes]

            d['low_nodes'] = low_nodes
            d['api_call_callsite_context'] = {'pre_n': args.pre_n, 'post_n': args.post_n, 'added_instr_count': len(added)}
            added_counts.append(len(added))
            if added:
                stats['api_call_added_context'] += 1
                stats['added_instr_total'] += len(added)
            else:
                stats['api_call_no_added_context'] += 1
            fout.write(json.dumps(d, ensure_ascii=False) + '\n')

    stats['avg_added_instr_per_api_call'] = sum(added_counts) / len(added_counts) if added_counts else 0
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(dict(stats), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(stats), indent=2, ensure_ascii=False))
    print('wrote', out_path)
    print('wrote', stats_path)


if __name__ == '__main__':
    main()
