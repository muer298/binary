#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_train_alignment_stage5_full.py

阶段5完整版工程脚本：
补充以下内容：
1. edge-type-aware GNN
   - 对 DFG/CFG/paramflow/constraint/relation/evidence 等边建立 relation embedding
   - 使用 edge-aware attention message passing

2. constraint attention
   - 高层 TypeSem / InterfaceSem / APISem / LayoutSem 子图使用 constraint edge attention
   - 对进入同一节点的 constraint messages 做 softmax attention

3. 真实 DFG/CFG message passing
   - 使用 usg_edges.jsonl 中真实的 data_flow / def_use / param_def_use / return_use / cfg / control_flow / ordered_membership 等边
   - 在 anchor 的 low_nodes induced subgraph 上做 message passing

4. 高层 constraint-edge attention
   - 使用 high_nodes induced subgraph 中的 constraint / api_to_type / layout_to_type / interface_to_type 等边
   - 生成 hhigh(a)

模型结构：
  InstrSem sequence Transformer -> hinstr(a)
  Low DFG/CFG edge-aware GNN -> hstruct(a)
  High constraint attention GNN -> hhigh(a)
  evidence_type gate -> hlow(a) = gate(hinstr,hstruct)
  multi-positive InfoNCE 对齐 hlow(a), hhigh(a)
  按 target_node 学习多证据注意力，得到目标级 h_evidence(v)

说明：
这是阶段5的“完整工程版”，比 skeleton 更接近论文方案。
由于不同 USG 的字段和 edge_type 命名可能不完全一致，脚本通过
evidence_objective_v2 先做保守 schema 适配，再做目标分组。缺失 target ID 时
使用行级唯一键，不会把多个未知目标错误合并为正样本。
"""

import argparse
import csv
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict, Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from evidence_objective_v2 import (
    MultiPositiveEvidenceObjective,
    TargetGroupedBatchSampler,
    make_target_group_stats,
    parse_evidence_weights,
    resolve_evidence_record,
)
from split_isolation_v3 import (
    partition_pairs_by_sample,
    resolve_node_sample_id,
)


EVIDENCE_TYPES = ["deref_chain", "fixed_offset", "def_use", "api_call", "unknown"]
ETYPE_TO_IDX = {x: i for i, x in enumerate(EVIDENCE_TYPES)}


LOW_TYPES = {"InstrSem", "BlockSem", "CFGSem", "DFGSem"}
HIGH_TYPES = {"TypeSem", "InterfaceSem", "APISem", "LayoutSem"}
OBJECTIVE_REVISION = "clean_hsym_train_only_evidence_structure_v0_2_8"
NODE_EMBEDDING_EXPORT_CONTRACT = {
    "encoder_branch": "node_base",
    "input": "full USG node feature row",
    "normalization": "l2",
    "trained_by": [
        "multi_positive_target_attention_evidence_loss",
        "train_only_nonconstraint_edge_reconstruction_loss",
    ],
    "evidence_gradient_path": (
        "node_base -> mean(low/high node embeddings) -> residual evidence "
        "representations -> multi-positive target-attention loss"
    ),
    "export_policy": "clean_hsym_allowlist_only_node_base_v0_2_8",
}


def sinusoidal_position_encoding(length, dimension, device, dtype):
    """Parameter-free position signal for the ordered InstrSem sequence.

    Earlier Stage5 code sorted instructions by ``seq`` but fed them to a
    Transformer without a position signal, making the encoder permutation
    equivariant.  The deterministic encoding keeps old checkpoint parameter
    shapes compatible while making LOAD->ADD distinct from ADD->LOAD.
    """
    if length <= 0 or dimension <= 0:
        return torch.zeros((max(0, length), max(0, dimension)), device=device, dtype=dtype)
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    even_count = (dimension + 1) // 2
    exponent = torch.arange(even_count, device=device, dtype=dtype)
    divisor = torch.exp(exponent * (-math.log(10000.0) / max(1, even_count)))
    angles = position * divisor.unsqueeze(0)
    encoding = torch.zeros((length, dimension), device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(angles[:, :encoding[:, 0::2].shape[1]])
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(angles[:, :encoding[:, 1::2].shape[1]])
    return encoding


def scheduled_constraint_weight(epoch, total_epochs, start_weight,
                                final_weight, warmup_epochs,
                                ramp_epochs):
    """Return the evidence-first constraint curriculum weight.

    Research Content 2 section 2.2.1.4 asks evidence to dominate early,
    constraint influence to increase gradually, and the structure term to
    remain stable.  This pure function is intentionally easy to audit.
    """
    epoch = max(1, int(epoch))
    total_epochs = max(1, int(total_epochs))
    warmup_epochs = max(0, int(warmup_epochs))
    if ramp_epochs <= 0:
        ramp_epochs = max(1, total_epochs - warmup_epochs)
    start_weight = max(0.0, float(start_weight))
    final_weight = max(0.0, float(final_weight))
    if epoch <= warmup_epochs:
        return start_weight
    progress = min(
        1.0,
        max(0.0, (epoch - warmup_epochs) / float(ramp_epochs)),
    )
    return start_weight + progress * (final_weight - start_weight)


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_feature_payload(path):
    p = Path(path)
    try:
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(p, map_location="cpu")
    except Exception:
        with p.open("rb") as f:
            obj = pickle.load(f)

    feats = obj["features"]
    if not torch.is_tensor(feats):
        feats = torch.tensor(feats, dtype=torch.float32)
    else:
        feats = feats.float()

    node_ids = obj["node_ids"]
    return {
        "features": feats,
        "node_ids": node_ids,
        "node_index": obj.get("node_index") or {nid: i for i, nid in enumerate(node_ids)},
        "feature_dim": int(obj.get("feature_dim", feats.shape[1])),
        "feature_schema": obj.get("feature_schema", ""),
    }


def infer_node_type_from_id(node_id):
    s = str(node_id or "")
    parts = s.split("::")
    if len(parts) >= 3:
        return parts[-2]
    for t in list(LOW_TYPES) + list(HIGH_TYPES):
        if t in s:
            return t
    return "unknown"


def infer_sample_id(node_id):
    s = str(node_id or "")
    return s.split("::", 1)[0] if "::" in s else ""


def load_node_info(nodes_path, feat):
    info = {}
    if nodes_path and Path(nodes_path).exists():
        for n in read_jsonl(nodes_path):
            nid = n.get("node_id")
            if not nid:
                continue
            attrs = n.get("attrs") if isinstance(n.get("attrs"), dict) else {}
            info[nid] = {
                "node_type": n.get("node_type") or infer_node_type_from_id(nid),
                "sample_id": n.get("sample_id") or infer_sample_id(nid),
                "raw_id": n.get("raw_id") or attrs.get("id") or attrs.get("raw_id") or "",
                "source_path": n.get("source_path") or "",
                "attrs": attrs,
            }

    for nid in feat["node_ids"]:
        if nid not in info:
            info[nid] = {
                "node_type": infer_node_type_from_id(nid),
                "sample_id": infer_sample_id(nid),
                "raw_id": "",
                "source_path": "",
                "attrs": {},
            }
    return info


def norm_etype(x):
    x = str(x or "unknown")
    return x if x in ETYPE_TO_IDX else "unknown"


def seq_key(n):
    attrs = n.get("attrs") if isinstance(n.get("attrs"), dict) else {}
    for k in ["seq", "address", "raw_id", "node_id"]:
        v = attrs.get(k) or n.get(k)
        if v is not None:
            return str(v)
    return ""


def edge_type_name(e):
    attrs = e.get("attrs") if isinstance(e.get("attrs"), dict) else {}
    return str(
        e.get("edge_type")
        or attrs.get("relation_type")
        or attrs.get("evidence_type")
        or e.get("evidence_type")
        or e.get("edge_kind")
        or attrs.get("edge_kind")
        or "unknown"
    )


def edge_group(e, et):
    attrs = e.get("attrs") if isinstance(e.get("attrs"), dict) else {}
    ek = str(e.get("edge_kind") or attrs.get("edge_kind") or "").lower()
    etl = str(et or "").lower()
    if ek == "constraint" or "constraint" in etl or "api_to" in etl or "layout_to" in etl or "interface_to" in etl:
        return "constraint"
    if ek == "evidence" or et in ETYPE_TO_IDX:
        return "evidence"
    if any(x in etl for x in ["def_use", "data_flow", "dfg", "param_def_use", "return_use"]):
        return "data_flow"
    if any(x in etl for x in ["cfg", "control", "branch", "ordered_membership", "structural"]):
        return "control_flow"
    return "relation"


def canonical_rel_key(e, et):
    grp = edge_group(e, et)
    etl = str(et).lower()

    if grp == "constraint":
        if "api" in etl and "type" in etl:
            return "constraint:api_to_type"
        if "api" in etl and "interface" in etl:
            return "constraint:api_to_interface"
        if "layout" in etl and "type" in etl:
            return "constraint:layout_to_type"
        if "interface" in etl and "type" in etl:
            return "constraint:interface_to_type"
        return "constraint:%s" % et

    if grp == "data_flow":
        if "param_def_use" in etl:
            return "dfg:param_def_use"
        if "return_use" in etl:
            return "dfg:return_use"
        if "def_use" in etl:
            return "dfg:def_use"
        return "dfg:%s" % et

    if grp == "control_flow":
        return "cfg:%s" % et

    if grp == "evidence":
        return "evidence:%s" % et

    return "relation:%s" % et


class GraphStore:
    def __init__(self):
        self.rel_vocab = {}
        self.edges = []
        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)
        self.edge_group_counts = Counter()
        self.rel_counts = Counter()

    def rel_id(self, key):
        if key not in self.rel_vocab:
            self.rel_vocab[key] = len(self.rel_vocab)
        return self.rel_vocab[key]

    def add(self, src_idx, dst_idx, rel_key, group):
        rid = self.rel_id(rel_key)
        row = (src_idx, rid, dst_idx, rel_key, group)
        self.edges.append(row)
        self.out_edges[src_idx].append(row)
        self.in_edges[dst_idx].append(row)
        self.edge_group_counts[group] += 1
        self.rel_counts[rel_key] += 1


def load_graph(edges_path, feat, node_info, max_edges=500000):
    g = GraphStore()
    idx = feat["node_index"]
    if not edges_path:
        return g

    for e in read_jsonl(edges_path):
        s, t = e.get("source_node"), e.get("target_node")
        if s not in idx or t not in idx:
            continue
        et = edge_type_name(e)
        grp = edge_group(e, et)
        rel_key = canonical_rel_key(e, et)
        g.add(idx[s], idx[t], rel_key, grp)
        if len(g.edges) >= max_edges:
            break
    return g


class TrainLossGraph:
    """Train-only graph view used exclusively by optimization losses."""

    def __init__(self):
        self.edges = []
        self.constraint_edges = []
        self.known = set()
        self.pools_by_type = defaultdict(list)
        self.pools_by_sample_type = defaultdict(list)
        self.train_node_indices = []
        self.train_node_samples = {}
        self.audit = {}


def build_train_loss_graph(graph, feat, node_info, split_assignments):
    """Filter positives and negative pools to frozen Train samples only."""

    train_samples = {
        sample_id for sample_id, split in split_assignments.items() if split == "train"
    }
    if not train_samples:
        raise RuntimeError("No Train samples are available for Stage5 graph losses")
    result = TrainLossGraph()
    node_ids = feat["node_ids"]
    excluded_nodes = Counter()
    for index, node_id in enumerate(node_ids):
        sample_id, sample_source = resolve_node_sample_id(node_id, node_info)
        if not sample_id:
            excluded_nodes["unresolved_sample"] += 1
            continue
        if sample_id not in train_samples:
            excluded_nodes[split_assignments.get(sample_id, "not_in_split")] += 1
            continue
        node_type = node_info.get(node_id, {}).get("node_type", infer_node_type_from_id(node_id))
        result.train_node_indices.append(index)
        result.train_node_samples[index] = sample_id
        result.pools_by_type[node_type].append(index)
        result.pools_by_sample_type[(sample_id, node_type)].append(index)
    if len(result.train_node_indices) < 2:
        raise RuntimeError("Train-only graph negative pool has fewer than two nodes")

    excluded_edges = Counter()
    retained_samples = set()
    for source, relation, target, relation_key, group in graph.edges:
        source_id = node_ids[source]
        target_id = node_ids[target]
        source_sample, _ = resolve_node_sample_id(source_id, node_info)
        target_sample, _ = resolve_node_sample_id(target_id, node_info)
        if not source_sample or not target_sample:
            excluded_edges["unresolved_sample"] += 1
            continue
        if source_sample != target_sample:
            excluded_edges["cross_sample"] += 1
            continue
        if source_sample not in train_samples:
            excluded_edges[split_assignments.get(source_sample, "not_in_split")] += 1
            continue
        source_type = node_info.get(source_id, {}).get("node_type", infer_node_type_from_id(source_id))
        target_type = node_info.get(target_id, {}).get("node_type", infer_node_type_from_id(target_id))
        row = {
            "src": source,
            "dst": target,
            "rel": relation,
            "rel_key": relation_key,
            "group": group,
            "sample_id": source_sample,
            "src_type": source_type,
            "dst_type": target_type,
            "weight": 2.0 if group == "constraint" else 1.5 if group == "evidence" else 1.0,
        }
        if group == "constraint":
            result.constraint_edges.append(row)
        else:
            # Constraint edges may be consumed as message-passing context in
            # the final Research2 architecture, but v0_2_8 forbids any
            # constraint supervision objective.  They are therefore excluded
            # from the edge-reconstruction loss pool.
            result.edges.append(row)
        result.known.add((source, relation, target))
        retained_samples.add(source_sample)

    positive_leakage = sum(
        1 for row in result.edges if row["sample_id"] not in train_samples
    )
    negative_leakage = sum(
        1 for index in result.train_node_indices
        if result.train_node_samples.get(index) not in train_samples
    )
    if positive_leakage or negative_leakage:
        raise AssertionError("Stage5 Train-only graph firewall retained non-Train data")
    result.audit = {
        "policy": "train_samples_only_v6_5",
        "allowed_train_samples": len(train_samples),
        "source_graph_edges": len(graph.edges),
        "retained_train_edges": len(result.edges),
        "retained_constraint_edges": len(result.constraint_edges),
        "retained_train_sample_count": len(retained_samples),
        "excluded_edge_counts": dict(excluded_edges),
        "train_node_pool_count": len(result.train_node_indices),
        "excluded_node_counts": dict(excluded_nodes),
        "positive_edge_leakage_count": positive_leakage,
        "negative_node_leakage_count": negative_leakage,
        "passed": positive_leakage == 0 and negative_leakage == 0,
    }
    return result


def filter_local_edges_by_split(records, feat, node_info, split_assignments):
    """Remove cross-sample/local edges before any split is encoded."""

    node_ids = feat["node_ids"]
    audit = Counter()
    for record in records:
        record_sample = str(record.get("sample_id") or "")
        split = split_assignments.get(record_sample, "")
        audit["records_" + (split or "unassigned")] += 1
        for field in ("low_edges", "high_edges"):
            retained = []
            for source, relation, target in record.get(field, []):
                source_sample, _ = resolve_node_sample_id(node_ids[source], node_info)
                target_sample, _ = resolve_node_sample_id(node_ids[target], node_info)
                if source_sample == record_sample and target_sample == record_sample:
                    retained.append((source, relation, target))
                else:
                    audit[field + "_cross_sample_removed"] += 1
            record[field] = retained
            audit[field + "_retained"] += len(retained)
    audit["passed"] = True
    return dict(audit)


def induced_edges(node_indices, graph, allowed_groups=None, max_edges=256, undirected=True):
    node_set = set(node_indices)
    rows = []
    seen = set()

    for u in node_indices:
        for row in graph.out_edges.get(u, []):
            s, r, t, key, grp = row
            if t not in node_set:
                continue
            if allowed_groups and grp not in allowed_groups:
                continue
            k = (s, r, t)
            if k not in seen:
                rows.append(row)
                seen.add(k)
            if undirected:
                rk = (t, r, s)
                if rk not in seen:
                    rows.append((t, r, s, key, grp))
                    seen.add(rk)
            if len(rows) >= max_edges:
                return rows

        for row in graph.in_edges.get(u, []):
            s, r, t, key, grp = row
            if s not in node_set:
                continue
            if allowed_groups and grp not in allowed_groups:
                continue
            k = (s, r, t)
            if k not in seen:
                rows.append(row)
                seen.add(k)
            if len(rows) >= max_edges:
                return rows

    return rows[:max_edges]


def split_node_groups(nodes):
    instr = []
    struct = []
    high = []
    for n in nodes or []:
        nt = n.get("node_type", "")
        if nt == "InstrSem":
            instr.append(n)
        elif nt in {"BlockSem", "CFGSem", "DFGSem"}:
            struct.append(n)
        elif nt in HIGH_TYPES:
            high.append(n)
    return instr, struct, high


def build_records(neighborhoods, feat, graph, node_info, max_anchors=None, max_local_edges=256):
    idx = feat["node_index"]
    records = []
    skipped = Counter()
    counts = Counter()
    local_edge_stats = Counter()
    target_group_sources = Counter()
    compatibility_paths = Counter()

    for row_index, rec in enumerate(read_jsonl(neighborhoods)):
        if max_anchors is not None and len(records) >= max_anchors:
            break

        resolved = resolve_evidence_record(rec, row_index=row_index)
        source_node = resolved["source_node"]
        target_node = resolved["target_node"]

        low_ids = [node_id for node_id in resolved["low_node_ids"] if node_id in idx]
        high_side_ids = [node_id for node_id in resolved["high_node_ids"] if node_id in idx]
        instr_ids = [
            node_id for node_id in low_ids
            if node_info.get(node_id, {}).get("node_type", infer_node_type_from_id(node_id)) == "InstrSem"
        ]
        instr_ids = sorted(instr_ids, key=lambda node_id: seq_key(node_info.get(node_id, {})))
        struct_ids = [
            node_id for node_id in low_ids
            if node_info.get(node_id, {}).get("node_type", infer_node_type_from_id(node_id))
            in {"BlockSem", "CFGSem", "DFGSem"}
        ]
        high_ids = [
            node_id for node_id in high_side_ids
            if node_info.get(node_id, {}).get("node_type", infer_node_type_from_id(node_id)) in HIGH_TYPES
            or node_id == target_node
        ]

        if source_node in idx and source_node not in instr_ids and source_node not in struct_ids:
            instr_ids.append(source_node)
        if target_node in idx and target_node not in high_ids:
            high_ids.append(target_node)

        instr_idx = [idx[x] for x in instr_ids]
        struct_idx = [idx[x] for x in struct_ids]
        high_idx = [idx[x] for x in high_ids]

        low_idx = []
        seen = set()
        for x in instr_idx + struct_idx:
            if x not in seen:
                low_idx.append(x)
                seen.add(x)

        if not low_idx:
            skipped["no_low"] += 1
            continue
        if not high_idx:
            skipped["no_high"] += 1
            continue

        low_edges = induced_edges(
            low_idx,
            graph,
            allowed_groups={"data_flow", "control_flow", "relation", "evidence"},
            max_edges=max_local_edges,
            undirected=True,
        )

        high_edges = induced_edges(
            high_idx,
            graph,
            allowed_groups={"constraint", "relation"},
            max_edges=max_local_edges,
            undirected=True,
        )

        local_edge_stats["low_edges"] += len(low_edges)
        local_edge_stats["high_edges"] += len(high_edges)

        et = norm_etype(resolved["evidence_type"])
        records.append({
            "anchor_id": resolved["anchor_id"],
            "sample_id": resolved["sample_id"] or infer_sample_id(source_node) or infer_sample_id(target_node),
            "target_node": target_node,
            "target_group_id": resolved["target_group_id"],
            "target_group_source": resolved["target_group_source"],
            "evidence_type": et,
            "etype_idx": ETYPE_TO_IDX[et],
            "strength": resolved["strength"],
            "instr_idx": instr_idx,
            "struct_idx": struct_idx,
            "low_idx": low_idx,
            "high_idx": high_idx,
            "low_edges": [(a, b, c) for a, b, c, _, _ in low_edges],
            "high_edges": [(a, b, c) for a, b, c, _, _ in high_edges],
        })
        counts[et] += 1
        target_group_sources[resolved["target_group_source"]] += 1
        for field, path in resolved["compatibility_paths"].items():
            compatibility_paths["%s:%s" % (field, path or "missing")] += 1

    avg_low = local_edge_stats["low_edges"] / max(1, len(records))
    avg_high = local_edge_stats["high_edges"] / max(1, len(records))
    group_stats = make_target_group_stats([r["target_group_id"] for r in records])
    return records, {
        "usable_records": len(records),
        "skipped": dict(skipped),
        "evidence_type_counts": dict(counts),
        "avg_low_edges_per_anchor": avg_low,
        "avg_high_edges_per_anchor": avg_high,
        "target_group_sources": dict(target_group_sources),
        "target_group_stats": group_stats,
        "schema_compatibility_paths": dict(compatibility_paths),
    }


def split_indices(records, mode, seed, train_ratio=0.8, val_ratio=0.1):
    rnd = random.Random(seed)
    if mode == "sample":
        by_sample = defaultdict(list)
        for i, r in enumerate(records):
            by_sample[r.get("sample_id") or "no_sample_%d" % i].append(i)
        samples = list(by_sample.keys())
        rnd.shuffle(samples)

        n = len(samples)
        a = int(n * train_ratio)
        b = int(n * (train_ratio + val_ratio))
        tr_s, va_s, te_s = set(samples[:a]), set(samples[a:b]), set(samples[b:])

        tr, va, te = [], [], []
        for s, ids in by_sample.items():
            if s in tr_s:
                tr.extend(ids)
            elif s in va_s:
                va.extend(ids)
            else:
                te.extend(ids)

        return tr, va, te, {
            "split_mode": "sample",
            "train_samples": len(tr_s),
            "val_samples": len(va_s),
            "test_samples": len(te_s),
        }

    ids = list(range(len(records)))
    rnd.shuffle(ids)
    n = len(ids)
    a = int(n * train_ratio)
    b = int(n * (train_ratio + val_ratio))
    return ids[:a], ids[a:b], ids[b:], {"split_mode": "anchor"}


class RecDS(Dataset):
    def __init__(self, records, indices):
        self.records = records
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.records[self.indices[i]]


def collate(batch):
    return batch


class EdgeAwareLayer(nn.Module):
    def __init__(self, hidden_dim, rel_dim, dropout=0.1):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(hidden_dim + rel_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.att = nn.Sequential(
            nn.Linear(hidden_dim * 2 + rel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edges, rel_emb, softmax_attention=True):
        if not edges:
            return h

        device = h.device
        src = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
        rel = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)
        dst = torch.tensor([e[2] for e in edges], dtype=torch.long, device=device)

        r = rel_emb(rel)
        hs = h.index_select(0, src)
        hd = h.index_select(0, dst)

        m = self.msg(torch.cat([hs, r], dim=-1))
        a = self.att(torch.cat([hs, hd, r], dim=-1)).squeeze(-1)

        if softmax_attention:
            # softmax over incoming edges per dst
            weights = torch.zeros_like(a)
            for d in torch.unique(dst):
                mask = dst == d
                weights[mask] = F.softmax(a[mask], dim=0)
        else:
            weights = torch.sigmoid(a)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m * weights.unsqueeze(-1))
        return self.norm(h + agg)


class Stage5FullModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embed_dim=64, etype_dim=16,
                 rel_count=128, rel_dim=64, nhead=4, transformer_layers=1, gnn_layers=2, dropout=0.1,
                 attention_hidden_dim=128, attention_dropout=0.1,
                 attention_strength_scale=1.0, attention_temperature=1.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.etype_emb = nn.Embedding(len(EVIDENCE_TYPES), etype_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.instr_encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        self.rel_emb = nn.Embedding(max(1, rel_count), rel_dim)
        self.low_gnn = nn.ModuleList([EdgeAwareLayer(hidden_dim, rel_dim, dropout) for _ in range(gnn_layers)])
        self.high_gnn = nn.ModuleList([EdgeAwareLayer(hidden_dim, rel_dim, dropout) for _ in range(gnn_layers)])

        self.struct_fallback = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.high_fallback = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        self.gate = nn.Sequential(
            nn.Linear(etype_dim, 2),
            nn.Softmax(dim=-1),
        )

        self.low_out = nn.Linear(hidden_dim, embed_dim)
        self.high_out = nn.Linear(hidden_dim, embed_dim)
        # Shared, exportable node encoder trained by the same optimizer as the
        # Stage5 Transformer/GNN evidence branch.  This closes the former
        # Stage3-loss/Stage5-encoder split while retaining an embedding defined
        # for every USG node.
        self.node_base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.graph_rel_emb = nn.Embedding(max(1, rel_count), embed_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # This is target-level attention across multiple evidence records.  It
        # is separate from high_gnn's incoming constraint-edge attention.
        self.evidence_objective = MultiPositiveEvidenceObjective(
            embed_dim=embed_dim,
            evidence_type_count=len(EVIDENCE_TYPES),
            evidence_type_dim=etype_dim,
            attention_hidden_dim=attention_hidden_dim,
            attention_dropout=attention_dropout,
            strength_log_scale=attention_strength_scale,
            attention_temperature=attention_temperature,
        )

    def encode_instr(self, features, instr_idx, device):
        hidden_dim = self.input_proj.out_features
        if not instr_idx:
            return torch.zeros(hidden_dim, device=device)
        idx = torch.tensor(instr_idx, dtype=torch.long, device=device)
        x = self.input_proj(features.index_select(0, idx))
        x = x + sinusoidal_position_encoding(
            x.size(0), x.size(1), x.device, x.dtype
        )
        x = x.unsqueeze(0)
        z = self.instr_encoder(x).mean(dim=1).squeeze(0)
        return z

    def encode_low_graph(self, features, low_idx, low_edges, device):
        hidden_dim = self.input_proj.out_features
        if not low_idx:
            return torch.zeros(hidden_dim, device=device)

        local_pos = {gidx: i for i, gidx in enumerate(low_idx)}
        idx = torch.tensor(low_idx, dtype=torch.long, device=device)
        h = self.input_proj(features.index_select(0, idx))

        local_edges = []
        for s, r, t in low_edges:
            if s in local_pos and t in local_pos:
                local_edges.append((local_pos[s], r, local_pos[t]))

        if local_edges:
            for layer in self.low_gnn:
                h = layer(h, local_edges, self.rel_emb, softmax_attention=False)
            return h.mean(dim=0)

        return self.struct_fallback(h).mean(dim=0)

    def encode_high_graph(self, features, high_idx, high_edges, device):
        hidden_dim = self.input_proj.out_features
        if not high_idx:
            return torch.zeros(hidden_dim, device=device)

        local_pos = {gidx: i for i, gidx in enumerate(high_idx)}
        idx = torch.tensor(high_idx, dtype=torch.long, device=device)
        h = self.input_proj(features.index_select(0, idx))

        local_edges = []
        for s, r, t in high_edges:
            if s in local_pos and t in local_pos:
                local_edges.append((local_pos[s], r, local_pos[t]))

        if local_edges:
            for layer in self.high_gnn:
                h = layer(h, local_edges, self.rel_emb, softmax_attention=True)
            return h.mean(dim=0)

        return self.high_fallback(h).mean(dim=0)

    def encode_one(self, features, rec, device):
        hinstr = self.encode_instr(features, rec["instr_idx"], device)
        hstruct = self.encode_low_graph(features, rec["low_idx"], rec["low_edges"], device)
        hhigh = self.encode_high_graph(features, rec["high_idx"], rec["high_edges"], device)

        et = torch.tensor([rec["etype_idx"]], dtype=torch.long, device=device)
        gate = self.gate(self.etype_emb(et)).squeeze(0)
        hlow = gate[0] * hinstr + gate[1] * hstruct

        # The exportable node encoder is intentionally part of the evidence
        # path.  Earlier versions optimized Stage5's local Transformer/GNN and
        # optimized ``node_base`` only with graph losses; exporting node_base
        # therefore discarded evidence-learning gradients.  The residual
        # means below make the same node_base parameters receive all three
        # objectives while preserving the richer local encoders.
        low_idx = torch.tensor(rec["low_idx"], dtype=torch.long, device=device)
        high_idx = torch.tensor(rec["high_idx"], dtype=torch.long, device=device)
        low_base = self.enc_node(features.index_select(0, low_idx)).mean(dim=0)
        high_base = self.enc_node(features.index_select(0, high_idx)).mean(dim=0)

        return (
            F.normalize(self.low_out(hlow) + low_base, p=2, dim=-1),
            F.normalize(self.high_out(hhigh) + high_base, p=2, dim=-1),
            gate.detach().cpu().tolist(),
        )

    def encode_batch(self, features, batch, device):
        lows, highs, gates = [], [], []
        for rec in batch:
            l, h, g = self.encode_one(features, rec, device)
            lows.append(l)
            highs.append(h)
            gates.append(g)
        return torch.stack(lows), torch.stack(highs), gates

    def enc_node(self, features):
        return F.normalize(self.node_base(features), p=2, dim=-1)

    def edge_score(self, source_embedding, relation, target_embedding):
        relation_embedding = self.graph_rel_emb(relation)
        joined = torch.cat([
            source_embedding,
            relation_embedding,
            target_embedding,
            source_embedding * target_embedding,
            torch.abs(source_embedding - target_embedding),
        ], dim=-1)
        return self.edge_mlp(joined).squeeze(-1)


def sample_rows(rows, count):
    if not rows:
        return []
    return random.sample(rows, min(int(count), len(rows)))


def sample_train_negative_targets(rows, graph, device):
    negatives = []
    for row in rows:
        pools = [
            graph.pools_by_sample_type.get((row["sample_id"], row["dst_type"]), []),
            graph.pools_by_type.get(row["dst_type"], []),
            graph.train_node_indices,
        ]
        chosen = None
        for pool in pools:
            if not pool:
                continue
            candidates = list(pool)
            random.shuffle(candidates)
            for candidate in candidates:
                if (
                    candidate != row["dst"]
                    and candidate in graph.train_node_samples
                    and (row["src"], row["rel"], candidate) not in graph.known
                ):
                    chosen = candidate
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(
                "No Train-only negative target for src=%s rel=%s dst=%s"
                % (row["src"], row["rel"], row["dst"])
            )
        negatives.append(chosen)
    return torch.tensor(negatives, dtype=torch.long, device=device)


def graph_batch_tensors(rows, graph, device):
    source = torch.tensor([row["src"] for row in rows], dtype=torch.long, device=device)
    target = torch.tensor([row["dst"] for row in rows], dtype=torch.long, device=device)
    relation = torch.tensor([row["rel"] for row in rows], dtype=torch.long, device=device)
    negative = sample_train_negative_targets(rows, graph, device)
    weight = torch.tensor([row["weight"] for row in rows], dtype=torch.float32, device=device)
    return source, relation, target, negative, weight


def constraint_margin_loss(model, features, graph, device, batch_size, margin):
    rows = sample_rows(graph.constraint_edges, batch_size)
    if not rows:
        return torch.tensor(0.0, device=device)
    source, relation, target, negative, weight = graph_batch_tensors(rows, graph, device)
    source_z = model.enc_node(features.index_select(0, source))
    target_z = model.enc_node(features.index_select(0, target))
    negative_z = model.enc_node(features.index_select(0, negative))
    positive_score = model.edge_score(source_z, relation, target_z)
    negative_score = model.edge_score(source_z, relation, negative_z)
    per_edge = F.relu(float(margin) - positive_score + negative_score)
    return (per_edge * weight).sum() / weight.sum().clamp_min(1.0)


def edge_reconstruction_loss(model, features, graph, device, batch_size):
    rows = sample_rows(graph.edges, batch_size)
    if not rows:
        return torch.tensor(0.0, device=device)
    source, relation, target, negative, weight = graph_batch_tensors(rows, graph, device)
    source_z = model.enc_node(features.index_select(0, source))
    target_z = model.enc_node(features.index_select(0, target))
    negative_z = model.enc_node(features.index_select(0, negative))
    positive_score = model.edge_score(source_z, relation, target_z)
    negative_score = model.edge_score(source_z, relation, negative_z)
    scores = torch.cat([positive_score, negative_score], dim=0)
    labels = torch.cat([torch.ones_like(positive_score), torch.zeros_like(negative_score)], dim=0)
    weights = torch.cat([weight, torch.ones_like(weight)], dim=0)
    per_edge = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
    return (per_edge * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def encode_split(model, features, records, ids, device, batch_size=128):
    model.eval()
    lows, highs, targets, etypes = [], [], [], []
    gate_acc = defaultdict(list)

    ds = RecDS(records, ids)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    for batch in dl:
        l, h, gates = model.encode_batch(features, batch, device)
        lows.append(l.cpu())
        highs.append(h.cpu())
        targets += [r["target_group_id"] for r in batch]
        etypes += [r["evidence_type"] for r in batch]
        for r, g in zip(batch, gates):
            gate_acc[r["evidence_type"]].append(g)

    return {
        "low": torch.cat(lows),
        "high": torch.cat(highs),
        "targets": targets,
        "etypes": etypes,
        "gate_acc": gate_acc,
    }


def rank_metrics(low, high, targets):
    scores = low @ high.t()
    n = scores.size(0)
    pos = defaultdict(list)
    for i, t in enumerate(targets):
        pos[t].append(i)

    r1 = r5 = 0
    rr = 0.0
    for i in range(n):
        order = torch.argsort(scores[i], descending=True).tolist()
        positives = set(pos[targets[i]])
        rank = n + 1
        for k, j in enumerate(order, start=1):
            if j in positives:
                rank = k
                break
        r1 += rank <= 1
        r5 += rank <= 5
        rr += 1.0 / rank

    return {"recall@1": r1 / n, "recall@5": r5 / n, "mrr": rr / n, "n": n}


def summarize_gates(gate_acc):
    out = {}
    for et, vals in gate_acc.items():
        if not vals:
            continue
        a = sum(v[0] for v in vals) / len(vals)
        b = sum(v[1] for v in vals) / len(vals)
        out[et] = {"instr_gate": a, "struct_gate": b, "n": len(vals)}
    return out


@torch.no_grad()
def eval_model(model, features, records, ids, device, batch_size=128):
    enc = encode_split(model, features, records, ids, device, batch_size)
    overall = rank_metrics(enc["low"], enc["high"], enc["targets"])

    by = defaultdict(list)
    for i, et in enumerate(enc["etypes"]):
        by[et].append(i)

    bym = {}
    for et, inds in by.items():
        if len(inds) >= 2:
            ix = torch.tensor(inds, dtype=torch.long)
            bym[et] = rank_metrics(
                enc["low"].index_select(0, ix),
                enc["high"].index_select(0, ix),
                [enc["targets"][i] for i in inds],
            )

    return {"overall": overall, "by_evidence_type": bym, "gate_summary": summarize_gates(enc["gate_acc"])}


def train(args):
    wall_start = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.lambda_constraint_margin_start < 0:
        raise ValueError("--lambda-constraint-margin-start must be non-negative")
    if args.lambda_constraint_margin < args.lambda_constraint_margin_start:
        raise ValueError(
            "--lambda-constraint-margin must be >= its start weight"
        )
    if args.constraint_warmup_epochs < 0:
        raise ValueError("--constraint-warmup-epochs must be non-negative")
    if args.lambda_constraint_margin != 0 or args.lambda_constraint_margin_start != 0:
        raise RuntimeError("v0_2_8 clean h_sym contract requires constraint loss weights to be zero")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    feat = load_feature_payload(args.node_features)
    node_info = load_node_info(args.nodes, feat)
    graph = load_graph(args.edges, feat, node_info, args.max_graph_edges)
    records, record_stats = build_records(
        args.neighborhoods,
        feat,
        graph,
        node_info,
        args.max_anchors,
        args.max_local_edges,
    )

    if args.require_split_manifest and not args.split_manifest:
        raise RuntimeError("--require-split-manifest was set but --split-manifest is empty")
    graph_losses_enabled = (
        args.lambda_constraint_margin > 0 or args.lambda_structure_recon > 0
    )
    if graph_losses_enabled and args.split_mode != "sample":
        raise RuntimeError(
            "Stage5 unified graph losses require --split-mode sample; anchor splitting leaks sample edges"
        )
    if args.split_mode == "sample":
        partitions, split_assignments, split_info = partition_pairs_by_sample(
            records,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.val_ratio,
            split_manifest=args.split_manifest,
            missing_manifest_policy=args.missing_split_policy,
            require_nonempty=False,
        )
        tr = partitions["train"]
        va = partitions["validation"]
        te = partitions["test"]
    else:
        tr, va, te, split_info = split_indices(
            records, args.split_mode, args.seed, args.train_ratio, args.val_ratio
        )
        split_assignments = {}
        for split_name, indexes in (("train", tr), ("validation", va), ("test", te)):
            for index in indexes:
                split_assignments[records[index]["sample_id"]] = split_name

    if not tr or not va:
        raise RuntimeError("encoder TRAIN and encoder inner-dev must both be non-empty")
    if te:
        raise RuntimeError("outer/test records are forbidden in the v0_2_8 training process")

    local_edge_audit = filter_local_edges_by_split(
        records, feat, node_info, split_assignments
    )
    train_graph = build_train_loss_graph(
        graph, feat, node_info, split_assignments
    )
    (out / "graph_split_audit.json").write_text(
        json.dumps({
            "global_train_loss_graph": train_graph.audit,
            "local_record_edges": local_edge_audit,
            "split": split_info,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if graph_losses_enabled and not train_graph.edges:
        raise RuntimeError("Train-only graph has no edges for enabled Stage5 graph losses")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    gpu_info = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "logical_device": str(device),
        "name": None,
        "free_vram_before_bytes": None,
        "total_vram_bytes": None,
    }
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_index)
        gpu_info.update({
            "name": torch.cuda.get_device_name(cuda_index),
            "logical_device_index": int(cuda_index),
            "free_vram_before_bytes": int(free_bytes),
            "total_vram_bytes": int(total_bytes),
        })
        torch.cuda.reset_peak_memory_stats(cuda_index)
    features = feat["features"].to(device)

    model = Stage5FullModel(
        input_dim=feat["feature_dim"],
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        etype_dim=args.etype_dim,
        rel_count=max(1, len(graph.rel_vocab)),
        rel_dim=args.rel_dim,
        nhead=args.nhead,
        transformer_layers=args.transformer_layers,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        attention_hidden_dim=args.attention_hidden_dim,
        attention_dropout=args.attention_dropout,
        attention_strength_scale=args.attention_strength_scale,
        attention_temperature=args.attention_temperature,
    ).to(device)

    trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameter_count = sum(p.numel() for p in model.parameters())
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    evidence_weight_tensor = torch.tensor(
        parse_evidence_weights(args.evidence_weights, EVIDENCE_TYPES),
        dtype=torch.float32,
        device=device,
    )

    train_ds = RecDS(records, tr)
    target_sampler = None
    if args.batching_mode == "target_grouped":
        target_sampler = TargetGroupedBatchSampler(
            [records[index]["target_group_id"] for index in tr],
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
            max_group_size=args.max_target_group_size,
        )
        dl = DataLoader(train_ds, batch_sampler=target_sampler, collate_fn=collate)
        batching_audit = target_sampler.audit()
    else:
        dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        batching_audit = {
            "batching_mode": "random",
            "warning": "same-target evidence can be split across batches",
        }

    logs = []
    best = -1.0
    best_path = out / "alignment_model_best.pt"
    optimizer_steps = 0
    gradient_steps = 0
    max_gradient_norm = 0.0

    for ep in range(1, args.epochs + 1):
        model.train()
        active_constraint_weight = scheduled_constraint_weight(
            ep,
            args.epochs,
            args.lambda_constraint_margin_start,
            args.lambda_constraint_margin,
            args.constraint_warmup_epochs,
            args.constraint_ramp_epochs,
        )
        if target_sampler is not None:
            target_sampler.set_epoch(ep)
        losses = []
        comps = defaultdict(list)

        for batch in dl:
            low, high, _ = model.encode_batch(features, batch, device)
            etype_ids = torch.tensor(
                [record["etype_idx"] for record in batch],
                dtype=torch.long,
                device=device,
            )
            strengths = torch.tensor(
                [record["strength"] for record in batch],
                dtype=torch.float32,
                device=device,
            )
            target_group_ids = [record["target_group_id"] for record in batch]
            l_ev, ev_diag, _ = model.evidence_objective(
                low,
                high,
                etype_ids,
                strengths,
                target_group_ids,
                evidence_weight_tensor,
                temperature=args.temperature,
                row_multi_positive_weight=args.row_multi_positive_weight,
                target_attention_weight=args.target_attention_weight,
            )
            l_constraint = torch.zeros((), dtype=low.dtype, device=device)
            l_structure = edge_reconstruction_loss(
                model,
                features,
                train_graph,
                device,
                args.structure_batch_size,
            )
            loss = (
                args.lambda_evidence * l_ev
                + active_constraint_weight * l_constraint
                + args.lambda_structure_recon * l_structure
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
            gradient_steps += 1
            opt.step()
            optimizer_steps += 1

            losses.append(loss.item())
            comps["l_evidence"].append(l_ev.item())
            comps["l_constraint_margin"].append(l_constraint.item())
            comps["l_structure_recon"].append(l_structure.item())
            comps["l_ev_row_multi_positive"].append(ev_diag["row_multi_positive_loss"])
            comps["l_ev_target_attention"].append(ev_diag["target_attention_loss"])
            comps["attention_entropy"].append(ev_diag["attention_entropy_mean"])
            comps["effective_evidence_count"].append(ev_diag["effective_evidence_count_mean"])
            comps["non_diagonal_positive_pairs"].append(ev_diag["non_diagonal_positive_pairs"])
            comps["batches_with_negative_targets"].append(float(ev_diag["has_negative_targets"]))

        if ep % args.eval_every == 0 or ep == args.epochs:
            val = eval_model(model, features, records, va, device, args.eval_batch_size)
            row = {
                "epoch": ep,
                "train_loss": sum(losses) / max(1, len(losses)),
                "l_evidence": sum(comps["l_evidence"]) / max(1, len(comps["l_evidence"])),
                "l_constraint_margin": sum(comps["l_constraint_margin"]) / max(1, len(comps["l_constraint_margin"])),
                "l_structure_recon": sum(comps["l_structure_recon"]) / max(1, len(comps["l_structure_recon"])),
                "lambda_evidence_active": args.lambda_evidence,
                "lambda_constraint_active": active_constraint_weight,
                "lambda_structure_active": args.lambda_structure_recon,
                "l_ev_row_multi_positive": sum(comps["l_ev_row_multi_positive"]) / max(1, len(comps["l_ev_row_multi_positive"])),
                "l_ev_target_attention": sum(comps["l_ev_target_attention"]) / max(1, len(comps["l_ev_target_attention"])),
                "attention_entropy": sum(comps["attention_entropy"]) / max(1, len(comps["attention_entropy"])),
                "effective_evidence_count": sum(comps["effective_evidence_count"]) / max(1, len(comps["effective_evidence_count"])),
                "non_diagonal_positive_pairs": sum(comps["non_diagonal_positive_pairs"]),
                "batches_with_negative_targets_rate": sum(comps["batches_with_negative_targets"]) / max(1, len(comps["batches_with_negative_targets"])),
                "val_recall@1": val["overall"]["recall@1"],
                "val_recall@5": val["overall"]["recall@5"],
                "val_mrr": val["overall"]["mrr"],
            }
            logs.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

            if val["overall"]["mrr"] > best:
                best = val["overall"]["mrr"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "objective_revision": OBJECTIVE_REVISION,
                    "selected_by": "encoder_inner_dev_mrr",
                    "best_epoch": ep,
                    "best_encoder_inner_dev_mrr": best,
                    "loss_weight_schedule": {
                        "policy": "clean_evidence_plus_train_only_structure_no_constraint_v0_2_8",
                        "lambda_evidence": args.lambda_evidence,
                        "lambda_constraint_start": args.lambda_constraint_margin_start,
                        "lambda_constraint_final": args.lambda_constraint_margin,
                        "constraint_warmup_epochs": args.constraint_warmup_epochs,
                        "constraint_ramp_epochs": args.constraint_ramp_epochs,
                        "lambda_structure_constant": args.lambda_structure_recon,
                    },
                    "node_embedding_export_contract": NODE_EMBEDDING_EXPORT_CONTRACT,
                    "graph_split_audit": train_graph.audit,
                }, best_path)

    if best_path.exists():
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    parameter_deltas = {}
    changed_parameter_tensors = 0
    max_parameter_abs_delta = 0.0
    for name, value in model.state_dict().items():
        if name not in initial_state or not value.is_floating_point():
            continue
        delta = float((value.detach().cpu() - initial_state[name]).abs().max())
        parameter_deltas[name] = delta
        if delta > 0:
            changed_parameter_tensors += 1
        max_parameter_abs_delta = max(max_parameter_abs_delta, delta)

    # The only post-selection metrics are TRAIN and encoder inner-dev.  Outer
    # DEV/CAL/TEST records are not present in any file opened by this process.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    train_m = eval_model(model, features, records, tr, device, args.eval_batch_size)
    val_m = eval_model(model, features, records, va, device, args.eval_batch_size)

    peak_allocated = 0
    peak_reserved = 0
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        peak_allocated = int(torch.cuda.max_memory_allocated(cuda_index))
        peak_reserved = int(torch.cuda.max_memory_reserved(cuda_index))
    wall_seconds = time.time() - wall_start

    evaluation_audit = {
        "passed": len(te) == 0,
        "checkpoint_selection_split": "encoder_inner_dev_from_global_train",
        "outer_dev_records_opened": 0,
        "outer_cal_records_opened": 0,
        "outer_test_records_opened": 0,
        "outer_labels_used_for_selection": False,
        "test_performance_evaluations": 0,
    }

    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "objective_revision": OBJECTIVE_REVISION,
        "selected_by": "encoder_inner_dev_mrr",
        "best_encoder_inner_dev_mrr": best,
        "evaluation_audit": evaluation_audit,
        "graph_split_audit": train_graph.audit,
        "node_embedding_export_contract": NODE_EMBEDDING_EXPORT_CONTRACT,
        "training_truth": {
            "optimizer_steps": optimizer_steps,
            "gradient_steps": gradient_steps,
            "max_gradient_norm": max_gradient_norm,
            "changed_parameter_tensors": changed_parameter_tensors,
            "max_parameter_abs_delta": max_parameter_abs_delta,
        },
    }, out / "alignment_model_final.pt")

    metrics = {
        "dataset": {
            "node_features": str(Path(args.node_features).resolve()),
            "neighborhoods": str(Path(args.neighborhoods).resolve()),
            "nodes": str(Path(args.nodes).resolve()),
            "edges": str(Path(args.edges).resolve()),
            "feature_dim": feat["feature_dim"],
            "records": len(records),
            "train_records": len(tr),
            "val_records": len(va),
            "test_records": len(te),
            "split_info": split_info,
            "record_stats": record_stats,
            "batching_audit": batching_audit,
            "graph_split_audit": train_graph.audit,
            "local_edge_split_audit": local_edge_audit,
            "graph_edge_count": len(graph.edges),
            "edge_group_counts": dict(graph.edge_group_counts),
            "relation_vocab": graph.rel_vocab,
            "relation_counts_top50": graph.rel_counts.most_common(50),
        },
        "model": {
            "type": "stage5_full_multi_positive_target_attention_edge_aware_gnn",
            "objective_revision": OBJECTIVE_REVISION,
            "components": [
                "InstrSem TransformerEncoder",
                "Low DFG/CFG edge-type-aware GNN",
                "High constraint-edge attention GNN",
                "Evidence-type gate fusion",
                "Row-level multi-positive evidence InfoNCE",
                "Target-level learned evidence attention",
                "Shared exportable node_base encoder",
                "Evidence-gradient residual path through node_base",
                "Train-only non-constraint edge-reconstruction loss",
            ],
            "split_mode": args.split_mode,
            "evidence_objective": {
                "row_multi_positive_weight": args.row_multi_positive_weight,
                "target_attention_weight": args.target_attention_weight,
                "attention_hidden_dim": args.attention_hidden_dim,
                "attention_dropout": args.attention_dropout,
                "attention_strength_scale": args.attention_strength_scale,
                "attention_temperature": args.attention_temperature,
                "group_key": "target_group_id resolved from target_node; missing target uses row-unique fallback",
            },
            "loss_weight_schedule": {
                "policy": "clean_evidence_plus_train_only_structure_no_constraint_v0_2_8",
                "lambda_evidence": args.lambda_evidence,
                "lambda_constraint_start": args.lambda_constraint_margin_start,
                "lambda_constraint_final": args.lambda_constraint_margin,
                "constraint_warmup_epochs": args.constraint_warmup_epochs,
                "constraint_ramp_epochs": args.constraint_ramp_epochs,
                "lambda_structure_constant": args.lambda_structure_recon,
            },
            "outer_evaluation_policy": "outer DEV/CAL/TEST absent and never evaluated",
        },
        "metrics": {"encoder_train": train_m, "encoder_inner_dev": val_m},
        "evaluation_audit": evaluation_audit,
        "training_truth": {
            "epochs_requested": args.epochs,
            "epochs_logged": len(logs),
            "optimizer_steps": optimizer_steps,
            "gradient_steps": gradient_steps,
            "max_gradient_norm": max_gradient_norm,
            "trainable_parameter_count": trainable_parameter_count,
            "total_parameter_count": total_parameter_count,
            "changed_parameter_tensors": changed_parameter_tensors,
            "max_parameter_abs_delta": max_parameter_abs_delta,
            "parameter_max_abs_deltas": parameter_deltas,
            "peak_allocated_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": peak_reserved,
            "wall_time_seconds": wall_seconds,
            "gpu": gpu_info,
            "nan_or_inf_loss": any(
                not math.isfinite(float(row["train_loss"])) for row in logs
            ),
        },
        "checkpoints": {"best": str(best_path), "final": str(out / "alignment_model_final.pt")},
    }

    (out / "alignment_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    if logs:
        with (out / "alignment_train_log.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
            w.writeheader()
            w.writerows(logs)

    print("wrote", out / "alignment_metrics.json")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-features", required=True)
    ap.add_argument("--neighborhoods", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--split-mode", choices=["anchor", "sample"], default="sample")
    ap.add_argument(
        "--split-manifest",
        default="",
        help="Frozen Train/Validation/Test manifest (JSONL/JSON/CSV)",
    )
    ap.add_argument("--require-split-manifest", action="store_true")
    ap.add_argument(
        "--missing-split-policy",
        choices=["error", "drop"],
        default="error",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument(
        "--batching-mode",
        choices=["target_grouped", "random"],
        default="target_grouped",
        help="target_grouped preserves all same-target positives inside a batch",
    )
    ap.add_argument(
        "--max-target-group-size",
        type=int,
        default=0,
        help="0 keeps complete groups; set a cap only when an oversized target causes OOM",
    )
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--etype-dim", type=int, default=16)
    ap.add_argument("--rel-dim", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--transformer-layers", type=int, default=1)
    ap.add_argument("--gnn-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--lambda-evidence", type=float, default=1.0)
    ap.add_argument("--lambda-constraint-margin", type=float, default=0.0)
    ap.add_argument(
        "--lambda-constraint-margin-start",
        type=float,
        default=0.0,
        help="Must remain zero under the v0_2_8 clean h_sym contract.",
    )
    ap.add_argument(
        "--constraint-warmup-epochs",
        type=int,
        default=2,
        help="Retained for checkpoint schema compatibility; constraint is disabled.",
    )
    ap.add_argument(
        "--constraint-ramp-epochs",
        type=int,
        default=10,
        help="Retained for checkpoint schema compatibility; constraint is disabled.",
    )
    ap.add_argument("--lambda-structure-recon", type=float, default=0.04)
    ap.add_argument("--constraint-margin", type=float, default=0.25)
    ap.add_argument("--constraint-batch-size", type=int, default=256)
    ap.add_argument("--structure-batch-size", type=int, default=256)
    ap.add_argument("--attention-hidden-dim", type=int, default=128)
    ap.add_argument("--attention-dropout", type=float, default=0.1)
    ap.add_argument("--attention-strength-scale", type=float, default=1.0)
    ap.add_argument("--attention-temperature", type=float, default=1.0)
    ap.add_argument("--row-multi-positive-weight", type=float, default=1.0)
    ap.add_argument("--target-attention-weight", type=float, default=1.0)
    ap.add_argument(
        "--evidence-weights",
        default="deref_chain=1.0,fixed_offset=1.0,def_use=1.5,api_call=3.0,unknown=1.0",
    )

    ap.add_argument("--max-anchors", type=int, default=None)
    ap.add_argument("--max-graph-edges", type=int, default=500000)
    ap.add_argument("--max-local-edges", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
