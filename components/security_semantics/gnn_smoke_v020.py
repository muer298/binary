#!/usr/bin/env python3
"""SERAPH Task2 v0.2.0 candidate-level GNN/MIL smoke.

The script consumes the frozen v0.1.12 DEVELOPMENT_SMOKE candidate set.  It
does not generate candidates and never reads any historical holdout.  Juliet
Gold is used only to form function-level MIL bags after the candidate set has
already been frozen by v0.1.12.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn


ROOT = Path("/data/chaoni/mjy/resym_exp")
V010 = ROOT / "training_runs/seraph_task2/v0_1_0"
V012 = ROOT / "training_runs/seraph_task2/v0_1_12"
PACKAGE = ROOT / "packages/seraph_task2_vuln_v0_2_0"
OUT = ROOT / "training_runs/seraph_task2/v0_2_0"
CANDIDATES = V012 / "arms/A2_FULL_AVAILABLE_SERAPH/vulnerability_candidates_v0_1_12.jsonl"
GOLD = V010 / "vulnerability_gold.jsonl"
MANIFEST = V010 / "security_input_manifest.jsonl"
USG_NODES = V010 / "juliet_usg_final/usg_nodes.jsonl"
USG_EDGES = V010 / "juliet_usg_final/usg_edges.jsonl"
HSYM = V010 / "juliet_clean_hsym_forward/node_embeddings_clean_juliet.pt"
HSYM_INDEX = V010 / "juliet_clean_hsym_forward/node_embedding_index_clean_juliet.json"
SEED = 20261010
MAX_NODES = 64
HOPS = 2
STEPS = 20
BATCH_BAGS = 8
FIXED_BUDGET_FRACTION = 0.25


def rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def blind(sample_id: str) -> str:
    return "JULIET_" + hashlib.sha256(sample_id.encode()).hexdigest()[:20]


def norm(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        return format(int(str(value), 16), "x")
    except ValueError:
        return None


def stable_bucket(value: str, n: int) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % n


def finite_number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def reachable_closure(graph: Dict[str, Set[str]], start: str) -> Set[str]:
    todo = [start]
    seen = {start}
    while todo:
        node = todo.pop()
        for nxt in graph.get(node, set()):
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return seen


NODE_TYPES = ["InstrSem", "TypeSem", "InterfaceSem", "LayoutSem", "APISem", "Evidence", "Other"]
LAYERS = ["instruction", "type", "interface", "layout", "api", "other"]
FAMILIES = ["BOF", "INTEGER_OVERFLOW", "UAF", "NULL_DEREF", "DANGEROUS_API", "OTHER"]


def structural_node_schema() -> List[str]:
    return (
        [f"node_type={x}" for x in NODE_TYPES]
        + [f"layer={x}" for x in LAYERS]
        + [f"opcode_hash_{i}" for i in range(16)]
        + [
            "is_anchor", "same_candidate_function", "log1p_degree", "has_memory_access",
            "log1p_memory_access_count", "has_layout_offsets", "has_access_widths",
            "constant_bias",
        ]
    )


STRUCT_NODE_SCHEMA = structural_node_schema()
STRUCT_CAND_SCHEMA = (
    [f"family={x}" for x in FAMILIES]
    + [
        "trigger_score", "upstream_confidence", "log1p_evidence_ref_count",
        "is_interprocedural", "is_complete_verified_path", "is_partial_path",
        "is_integer_event", "is_sink_related_integer_risk",
    ]
)
SEMANTIC_SCHEMA = [
    "used_TypeSem", "used_InterfaceSem", "used_LayoutSem", "used_APISem",
    "has_type_safety_predicate", "type_predicate_executable", "typesem_fact_count",
    "known_signedness", "has_layout_evidence", "known_layout_boundary",
    "interface_role_known", "apisem_support", "semantic_evidence_ref_count",
    "semantic_support_item_count", "semantic_confidence", "semantic_available_bias",
]


def one_hot(value: str, names: Sequence[str]) -> List[float]:
    return [1.0 if value == name else 0.0 for name in names]


def node_features(row: dict, anchor: str, candidate_function: str, degree: int) -> List[float]:
    attrs = row.get("attrs") or {}
    node_type = row.get("node_type") if row.get("node_type") in NODE_TYPES else "Other"
    layer = row.get("layer") if row.get("layer") in LAYERS else "other"
    opcode = str(attrs.get("opcode") or "UNKNOWN")
    opcode_vec = [0.0] * 16
    opcode_vec[stable_bucket(opcode, 16)] = 1.0
    memory = attrs.get("memory_access_evidence") or []
    offsets = attrs.get("observed_field_offsets") or attrs.get("local_observed_field_offsets") or []
    widths = attrs.get("access_widths") or []
    tail = [
        float(row["node_id"] == anchor),
        float(norm(row.get("function_entry_norm") or row.get("function_entry")) == candidate_function),
        math.log1p(degree) / 8.0,
        float(bool(memory)),
        math.log1p(len(memory)) / 8.0,
        float(bool(offsets)),
        float(bool(widths)),
        1.0,
    ]
    result = one_hot(node_type, NODE_TYPES) + one_hot(layer, LAYERS) + opcode_vec + tail
    assert len(result) == len(STRUCT_NODE_SCHEMA)
    return result


def candidate_struct_features(row: dict) -> List[float]:
    family = row.get("family") if row.get("family") in FAMILIES else "OTHER"
    status_text = " ".join(str(row.get(k) or "") for k in (
        "binding_status", "path_status", "evidence_chain_status", "confirmation_evidence"))
    refs = row.get("evidence_refs") or []
    result = one_hot(family, FAMILIES) + [
        finite_number(row.get("trigger_score")),
        finite_number(row.get("upstream_confidence")),
        math.log1p(len(refs)) / 8.0,
        float(bool(row.get("deterministic_interprocedural_DFG_enabled") or row.get("interprocedural"))),
        float("VERIFIED_INTERPROCEDURAL_PATH" in status_text or bool(row.get("verified_interprocedural_path"))),
        float("PARTIAL" in status_text or "UNKNOWN" in status_text),
        float(row.get("integer_candidate_scope") == "INTEGER_OVERFLOW_EVENT"),
        float(row.get("integer_candidate_scope") == "SINK_RELATED_INTEGER_RISK"),
    ]
    assert len(result) == len(STRUCT_CAND_SCHEMA)
    return result


def semantic_features(row: dict) -> List[float]:
    pred = row.get("type_safety_predicate") or {}
    pred_body = pred.get("predicate") or {}
    facts = pred.get("type_semantic_facts") or []
    support = row.get("supporting_evidence") or []
    used = [bool(row.get(f"used_{name}")) for name in ("TypeSem", "InterfaceSem", "LayoutSem", "APISem")]
    use_type, use_interface, use_layout, use_api = used
    semantic_refs = []
    if use_type:
        for item in facts:
            semantic_refs.extend(item.get("evidence_refs") or []) if isinstance(item, dict) else None
    semantic_conf = max([0.0] + (
        [finite_number(row.get("semantic_confidence"))]
        + [finite_number(item.get("confidence")) for item in facts if isinstance(item, dict)]
    ) if any(used) else [0.0])
    signedness = str(pred.get("signedness_interpretation") or pred_body.get("active_signedness") or "UNKNOWN")
    layout_text = json.dumps({k: v for k, v in row.items() if "layout" in k.lower()}, sort_keys=True)
    interface_text = json.dumps({k: v for k, v in row.items() if "interface" in k.lower()}, sort_keys=True)
    semantic_support_count = sum(
        any(module.upper() in str(item).upper() for module, enabled in zip(
            ("TypeSem", "InterfaceSem", "LayoutSem", "APISem"), used) if enabled)
        for item in support
    )
    result = [
        *[float(x) for x in used],
        float(use_type and bool(pred)),
        float(use_type and bool(pred_body.get("executable"))),
        min(1.0, len(facts) / 4.0) if use_type else 0.0,
        float(use_type and signedness not in {"", "UNKNOWN", "None"}),
        float(use_layout and layout_text not in {"{}", ""}),
        float(use_layout and any(token in layout_text.upper() for token in ("BOUNDARY_KNOWN", "FIELD_BOUNDARY", "WHOLE_ALLOCATION"))),
        float(use_interface and any(token in interface_text.upper() for token in ("SOURCE", "SINK", "SIZE", "BUFFER", "POINTER")) and "UNKNOWN" not in interface_text.upper()),
        float(use_api),
        min(1.0, len(set(semantic_refs)) / 8.0),
        min(1.0, semantic_support_count / 8.0),
        semantic_conf,
        float(any(used)),
    ]
    assert len(result) == len(SEMANTIC_SCHEMA)
    return result


def load_exact_hsym() -> Tuple[torch.Tensor, Dict[str, int], int]:
    payload = torch.load(HSYM, map_location="cpu")
    if not isinstance(payload, dict) or "embeddings" not in payload:
        raise RuntimeError("clean h_sym artifact has unexpected schema")
    embeddings = payload["embeddings"].detach().float().cpu()
    index_payload = json.loads(HSYM_INDEX.read_text(encoding="utf-8"))
    if isinstance(index_payload, dict) and "node_index" in index_payload:
        index_payload = index_payload["node_index"]
    if not isinstance(index_payload, dict):
        raise RuntimeError("clean h_sym index is not a node-id mapping")
    index = {str(k): int(v) for k, v in index_payload.items()}
    if max(index.values(), default=-1) >= embeddings.shape[0]:
        raise RuntimeError("clean h_sym index exceeds embedding rows")
    return embeddings, index, int(embeddings.shape[1])


def choose_dev_groups(gold: List[dict]) -> Tuple[Set[str], dict]:
    by_cwe: Dict[str, Set[str]] = collections.defaultdict(set)
    for row in gold:
        by_cwe[row["CWE"]].add(row["source_case_id"])
    selected = set()
    rationale = {}
    for cwe, groups in sorted(by_cwe.items()):
        ranked = sorted(groups, key=lambda x: hashlib.sha256(f"{SEED}:{x}".encode()).hexdigest())
        selected.add(ranked[0])
        rationale[cwe] = ranked[0]
    return selected, rationale


class GraphRiskModel(nn.Module):
    def __init__(self, node_dim: int, cand_dim: int, hidden: int = 32):
        super().__init__()
        self.node_dim = node_dim
        self.cand_dim = cand_dim
        self.lin1 = nn.Linear(node_dim, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2 + cand_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def candidate_logit(self, graph: dict, device: torch.device, use_semantics: bool, use_hsym: bool) -> torch.Tensor:
        x = graph["x_struct"].to(device)
        if use_hsym:
            x = torch.cat([x, graph["x_hsym"].to(device), graph["x_hsym_available"].to(device)], dim=1)
        a = graph["adj_norm"].to(device)
        h = torch.relu(a @ self.lin1(x))
        h = torch.relu(a @ self.lin2(h))
        anchor = h[graph["anchor_index"]]
        pooled = h.mean(dim=0)
        cand = graph["cand_struct"].to(device)
        if use_semantics:
            cand = torch.cat([cand, graph["cand_semantic"].to(device)], dim=0)
        return self.head(torch.cat([anchor, pooled, cand], dim=0)).squeeze(-1)


def mil_logit(logits: torch.Tensor, temperature: float = 4.0) -> torch.Tensor:
    # Smooth maximum.  Every candidate instance receives gradient.
    return torch.logsumexp(temperature * logits, dim=0) / temperature - math.log(logits.numel()) / temperature


def binary_metrics(labels: List[int], probs: List[float]) -> dict:
    pred = [int(x >= 0.5) for x in probs]
    tp = sum(p == 1 and y == 1 for p, y in zip(pred, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(pred, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(pred, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(pred, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    positives = [p for p, y in zip(probs, labels) if y]
    negatives = [p for p, y in zip(probs, labels) if not y]
    auc_terms = [1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives]
    auc = sum(auc_terms) / len(auc_terms) if auc_terms else None
    return {"n": len(labels), "positive": sum(labels), "negative": len(labels) - sum(labels),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1, "auroc": auc}


def create_subgraphs(candidates: List[dict], gold: List[dict]) -> Tuple[List[dict], dict, Dict[str, str]]:
    blind_to_raw = {blind(row["sample_id"]): row["sample_id"] for row in gold}
    needed_samples = set(blind_to_raw.values())
    nodes: Dict[str, dict] = {}
    sample_nodes: Dict[str, Set[str]] = collections.defaultdict(set)
    for row in rows(USG_NODES):
        if row.get("sample_id") in needed_samples:
            nodes[row["node_id"]] = row
            sample_nodes[row["sample_id"]].add(row["node_id"])
    adjacency: Dict[str, Set[str]] = collections.defaultdict(set)
    edge_types: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
    resolved_edge_count = 0
    for edge in rows(USG_EDGES):
        if edge.get("sample_id") not in needed_samples:
            continue
        src, dst = edge.get("source_node"), edge.get("target_node")
        if src in nodes and dst in nodes and edge.get("source_resolved") and edge.get("target_resolved"):
            adjacency[src].add(dst)
            adjacency[dst].add(src)
            edge_types[(src, dst)].add(str(edge.get("edge_type") or "UNKNOWN"))
            resolved_edge_count += 1
    embeddings, hsym_index, hdim = load_exact_hsym()
    graphs: List[dict] = []
    failures = collections.Counter()
    hsym_node_mapped = 0
    hsym_node_total = 0
    anchor_hsym_mapped = 0
    index_rows = []
    for cand in candidates:
        blind_sample = cand["sample_id"]
        raw_sample = blind_to_raw.get(blind_sample)
        if not raw_sample:
            failures["SAMPLE_ID_NOT_IN_DEVELOPMENT_SMOKE"] += 1
            continue
        blind_anchor = cand.get("instruction_anchor_node") or cand.get("evidence_refs", [None])[0]
        if not isinstance(blind_anchor, str) or not blind_anchor.startswith(blind_sample + "::"):
            failures["NO_EXACT_ANCHOR_ID"] += 1
            continue
        raw_anchor = raw_sample + blind_anchor[len(blind_sample):]
        if raw_anchor not in nodes:
            failures["ANCHOR_NOT_IN_USG"] += 1
            continue
        selected = [raw_anchor]
        seen = {raw_anchor}
        frontier = [raw_anchor]
        for _ in range(HOPS):
            nxt = []
            for node in frontier:
                for neighbor in sorted(adjacency.get(node, set())):
                    if neighbor not in seen and len(selected) < MAX_NODES:
                        seen.add(neighbor)
                        selected.append(neighbor)
                        nxt.append(neighbor)
            frontier = nxt
            if not frontier or len(selected) >= MAX_NODES:
                break
        local = {node: i for i, node in enumerate(selected)}
        adj = torch.zeros((len(selected), len(selected)), dtype=torch.float32)
        local_edges = []
        local_edge_types = collections.Counter()
        for src in selected:
            i = local[src]
            adj[i, i] = 1.0
            for dst in adjacency.get(src, set()):
                if dst in local:
                    j = local[dst]
                    adj[i, j] = 1.0
                    if i < j:
                        types = sorted(edge_types.get((src, dst), set()) | edge_types.get((dst, src), set()))
                        local_edges.append([i, j, types])
                        for typ in types:
                            local_edge_types[typ] += 1
        degree = adj.sum(dim=1).clamp(min=1.0)
        inv = degree.rsqrt()
        adj_norm = inv[:, None] * adj * inv[None, :]
        function = norm(cand.get("function_entry")) or ""
        x_struct = torch.tensor([node_features(nodes[n], raw_anchor, function, len(adjacency.get(n, set()))) for n in selected], dtype=torch.float32)
        hrows = []
        havail = []
        for node in selected:
            hsym_node_total += 1
            if node in hsym_index:
                hsym_node_mapped += 1
                hrows.append(embeddings[hsym_index[node]])
                havail.append([1.0])
            else:
                hrows.append(torch.zeros(hdim))
                havail.append([0.0])
        if raw_anchor in hsym_index:
            anchor_hsym_mapped += 1
        graph = {
            "candidate_id": cand["candidate_id"], "sample_id": blind_sample,
            "raw_sample_id": raw_sample, "function_entry": function,
            "family": cand.get("family"), "anchor_raw_node": raw_anchor,
            "node_ids": selected, "local_edges": local_edges,
            "x_struct": x_struct, "adj_norm": adj_norm,
            "x_hsym": torch.stack(hrows).float(),
            "x_hsym_available": torch.tensor(havail, dtype=torch.float32),
            "cand_struct": torch.tensor(candidate_struct_features(cand), dtype=torch.float32),
            "cand_semantic": torch.tensor(semantic_features(cand), dtype=torch.float32),
            "anchor_index": 0,
        }
        graphs.append(graph)
        index_rows.append({
            "candidate_id": cand["candidate_id"], "sample_id": blind_sample,
            "raw_sample_id": raw_sample, "function_entry": function,
            "family": cand.get("family"), "anchor_raw_node": raw_anchor,
            "hop_radius": HOPS, "node_count": len(selected), "edge_count": len(local_edges),
            "edge_type_counts": dict(sorted(local_edge_types.items())),
            "candidate_instance_label": "UNKNOWN_MIL_INSTANCE_LABEL",
            "exact_clean_hsym_anchor": raw_anchor in hsym_index,
        })
    with (OUT / "CANDIDATE_SUBGRAPH_INDEX.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    torch.save(graphs, OUT / "candidate_subgraphs_v020.pt")
    coverage = {
        "candidate_total": len(candidates), "candidate_subgraphs": len(graphs),
        "candidate_subgraph_rate": len(graphs) / max(1, len(candidates)),
        "failure_counts": dict(sorted(failures.items())), "resolved_usg_edges_loaded": resolved_edge_count,
        "hop_radius": HOPS, "max_nodes": MAX_NODES,
        "mean_nodes": sum(len(g["node_ids"]) for g in graphs) / max(1, len(graphs)),
        "exact_clean_hsym_anchor_count": anchor_hsym_mapped,
        "exact_clean_hsym_anchor_rate": anchor_hsym_mapped / max(1, len(graphs)),
        "exact_clean_hsym_node_count": hsym_node_mapped,
        "exact_clean_hsym_node_total": hsym_node_total,
        "exact_clean_hsym_node_rate": hsym_node_mapped / max(1, hsym_node_total),
        "hsym_dimension": hdim,
    }
    return graphs, coverage, blind_to_raw


def build_bags(graphs: List[dict], gold: List[dict]) -> Tuple[List[dict], List[dict], dict]:
    graph_by_sample_function: Dict[Tuple[str, str], List[int]] = collections.defaultdict(list)
    for i, graph in enumerate(graphs):
        graph_by_sample_function[(graph["sample_id"], graph["function_entry"])].append(i)
    callgraphs: Dict[str, Dict[str, Set[str]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    for row in rows(MANIFEST):
        sample, function = row.get("sample_id"), norm(row.get("function_entry"))
        if not sample or not function:
            continue
        for callee in row.get("callees") or []:
            target = norm(callee.get("function_entry"))
            if target:
                callgraphs[sample][function].add(target)
    dev_groups, rationale = choose_dev_groups(gold)
    bags, excluded = [], []
    for row in gold:
        sample = blind(row["sample_id"])
        entry = norm(row["function_entry_full_symbol_exact"])
        closure = reachable_closure(callgraphs.get(sample, {}), entry) if entry else set()
        graph_indices = []
        exact_count = 0
        for function in sorted(closure):
            ids = graph_by_sample_function.get((sample, function), [])
            graph_indices.extend(ids)
            if function == entry:
                exact_count += len(ids)
        if not graph_indices:
            excluded.append({
                "sample_id": row["sample_id"], "source_case_id": row["source_case_id"],
                "CWE": row["CWE"], "vulnerable": bool(row["vulnerable"]),
                "reason": "NO_EXACT_OR_TRANSITIVE_CALLGRAPH_CANDIDATE_SUBGRAPH",
                "split": "DEV" if row["source_case_id"] in dev_groups else "TRAIN",
            })
            continue
        bags.append({
            "bag_id": row["sample_id"], "sample_id": sample, "source_case_id": row["source_case_id"],
            "CWE": row["CWE"], "label": int(bool(row["vulnerable"])),
            "optimization": "O2" if row["sample_id"].endswith("__O2") else "O0",
            "gold_function_entry": entry, "candidate_graph_indices": sorted(set(graph_indices)),
            "exact_candidate_count": exact_count,
            "reachable_candidate_count": len(set(graph_indices)) - exact_count,
            "split": "DEV" if row["source_case_id"] in dev_groups else "TRAIN",
        })
    split_audit = {
        "method": "ONE_SOURCE_CASE_PER_CWE_SELECTED_BY_SHA256_OF_FIXED_SEED_AND_SOURCE_CASE_ID",
        "seed": SEED, "dev_group_by_CWE": rationale, "dev_groups": sorted(dev_groups),
        "train_groups": sorted({r["source_case_id"] for r in gold} - dev_groups),
        "group_overlap": sorted(dev_groups & ({r["source_case_id"] for r in gold} - dev_groups)),
        "usable_bags": len(bags), "excluded_records": len(excluded),
        "usable_by_split_label": dict(sorted(collections.Counter(f"{b['split']}_label_{b['label']}" for b in bags).items())),
        "excluded_by_reason": dict(sorted(collections.Counter(x["reason"] for x in excluded).items())),
    }
    return bags, excluded, split_audit


def arm_config(name: str, hdim: int) -> dict:
    if name == "G0_STRUCTURAL_GNN":
        return {"use_semantics": False, "use_hsym": False, "node_dim": len(STRUCT_NODE_SCHEMA), "cand_dim": len(STRUCT_CAND_SCHEMA)}
    if name == "G1_STRUCTURAL_PLUS_SERAPH_FEATURES":
        return {"use_semantics": True, "use_hsym": False, "node_dim": len(STRUCT_NODE_SCHEMA), "cand_dim": len(STRUCT_CAND_SCHEMA) + len(SEMANTIC_SCHEMA)}
    if name == "G2_G1_PLUS_FROZEN_SERAPH_EMBEDDINGS":
        return {"use_semantics": True, "use_hsym": True, "node_dim": len(STRUCT_NODE_SCHEMA) + hdim + 1, "cand_dim": len(STRUCT_CAND_SCHEMA) + len(SEMANTIC_SCHEMA)}
    raise ValueError(name)


def score_bag(model: GraphRiskModel, bag: dict, graphs: List[dict], device: torch.device, config: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = torch.stack([model.candidate_logit(graphs[i], device, config["use_semantics"], config["use_hsym"]) for i in bag["candidate_graph_indices"]])
    return mil_logit(logits), logits


def evaluate_model(model: GraphRiskModel, bags: List[dict], graphs: List[dict], device: torch.device, config: dict) -> dict:
    labels, probabilities, budget_probabilities = [], [], []
    good_full_burden = 0
    good_budget_burden = 0
    total_instances = 0
    retained_instances = 0
    per_bag = []
    model.eval()
    with torch.no_grad():
        for bag in bags:
            bag_logit, instance_logits = score_bag(model, bag, graphs, device, config)
            count = int(instance_logits.numel())
            keep = max(1, math.ceil(count * FIXED_BUDGET_FRACTION))
            selected = torch.topk(instance_logits, keep).values
            budget_logit = mil_logit(selected)
            prob = float(torch.sigmoid(bag_logit).cpu())
            budget_prob = float(torch.sigmoid(budget_logit).cpu())
            labels.append(bag["label"])
            probabilities.append(prob)
            budget_probabilities.append(budget_prob)
            total_instances += count
            retained_instances += keep
            if bag["label"] == 0:
                good_full_burden += count
                good_budget_burden += keep
            per_bag.append({"bag_id": bag["bag_id"], "source_case_id": bag["source_case_id"], "CWE": bag["CWE"],
                            "label": bag["label"], "candidate_count": count, "retained_at_budget": keep,
                            "probability": prob, "budget_probability": budget_prob})
    return {
        "all_candidate_function_metrics": binary_metrics(labels, probabilities),
        "fixed_budget_function_metrics": binary_metrics(labels, budget_probabilities),
        "fixed_budget_fraction": FIXED_BUDGET_FRACTION,
        "candidate_instances_total": total_instances, "candidate_instances_retained": retained_instances,
        "candidate_retention_rate": retained_instances / max(1, total_instances),
        "good_candidate_burden_full": good_full_burden,
        "good_candidate_burden_at_budget": good_budget_burden,
        "good_burden_reduction_rate": 1 - good_budget_burden / max(1, good_full_burden),
        "candidate_level_accuracy_reported": False,
        "candidate_level_accuracy_reason": "NO_RELIABLE_CANDIDATE_LEVEL_GOLD",
        "bags": per_bag,
    }


def train_arm(name: str, bags: List[dict], graphs: List[dict], hdim: int, device: torch.device) -> dict:
    config = arm_config(name, hdim)
    train_bags = [x for x in bags if x["split"] == "TRAIN"]
    dev_bags = [x for x in bags if x["split"] == "DEV"]
    if not train_bags or not dev_bags or len({x["label"] for x in train_bags}) < 2 or len({x["label"] for x in dev_bags}) < 2:
        raise RuntimeError(f"{name}: TRAIN/DEV MIL supervision lacks both classes")
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats(device)
    model = GraphRiskModel(config["node_dim"], config["cand_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.BCEWithLogitsLoss()
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rng = random.Random(SEED)
    schedule = []
    for _ in range(STEPS):
        schedule.append(rng.sample(range(len(train_bags)), min(BATCH_BAGS, len(train_bags))))
    losses, grad_norms = [], []
    started = time.time()
    model.train()
    for selection in schedule:
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        for index in selection:
            bag = train_bags[index]
            logit, _ = score_bag(model, bag, graphs, device, config)
            target = torch.tensor(float(bag["label"]), device=device)
            step_losses.append(criterion(logit, target))
        loss = torch.stack(step_losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: non-finite loss")
        loss.backward()
        grad_sq = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_sq += float(parameter.grad.detach().pow(2).sum().cpu())
        grad_norm = math.sqrt(grad_sq)
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError(f"{name}: zero or non-finite gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(grad_norm)
    elapsed = time.time() - started
    max_delta = max(float((value.detach().cpu() - initial[key]).abs().max()) for key, value in model.state_dict().items())
    if max_delta <= 0:
        raise RuntimeError(f"{name}: parameters did not update")
    dev_metrics = evaluate_model(model, dev_bags, graphs, device, config)
    checkpoint = OUT / "checkpoints" / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config, "seed": SEED, "steps": STEPS,
                "feature_schema_sha256": sha(OUT / "FEATURE_SCHEMA.json")}, checkpoint)
    before = [x["probability"] for x in dev_metrics["bags"]]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded = GraphRiskModel(config["node_dim"], config["cand_dim"]).to(device)
    payload = torch.load(checkpoint, map_location=device)
    reloaded.load_state_dict(payload["model_state"])
    reload_metrics = evaluate_model(reloaded, dev_bags, graphs, device, config)
    after = [x["probability"] for x in reload_metrics["bags"]]
    max_reload_difference = max((abs(a - b) for a, b in zip(before, after)), default=0.0)
    if max_reload_difference > 1e-7:
        raise RuntimeError(f"{name}: checkpoint reload changed predictions: {max_reload_difference}")
    return {
        "arm": name, "config": config, "optimizer_steps": STEPS,
        "train_bags": len(train_bags), "dev_bags": len(dev_bags),
        "trainable_parameters": sum(p.numel() for p in reloaded.parameters() if p.requires_grad),
        "losses": losses, "initial_loss": losses[0], "final_loss": losses[-1],
        "all_loss_finite": all(math.isfinite(x) for x in losses),
        "gradient_norms": grad_norms, "all_gradients_nonzero": all(x > 0 for x in grad_norms),
        "max_parameter_update": max_delta, "parameter_update_verified": max_delta > 0,
        "mil_backprop_verified": all(x > 0 for x in grad_norms),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "reload_passed": max_reload_difference <= 1e-7,
        "reload_max_probability_difference": max_reload_difference,
        "dev": dev_metrics,
    }


def make_reports(metrics: dict, excluded: List[dict]) -> None:
    cov = metrics["subgraph_coverage"]
    split = metrics["supervision"]["split_audit"]
    arms = metrics["arms"]
    supervision = [
        "# GNN Supervision Audit", "",
        "## Label contract", "",
        "- Reliable candidate-level vulnerability labels: **NO**.",
        "- The Juliet artifact identifies a vulnerable function but has no verified source/instruction-to-candidate mapping.",
        "- Therefore no candidate inside a bad function is assigned a positive label. Every instance remains `UNKNOWN_MIL_INSTANCE_LABEL`.",
        "- Supervision method: function-level Multiple Instance Learning (MIL): candidate subgraph scores are smooth-max aggregated to a function/bag score.", "",
        "## Bag identity and exclusions", "",
        f"- Gold records: **100** (50 bad / 50 good); usable exact-or-transitive-callgraph bags: **{split['usable_bags']}**.",
        f"- Excluded records: **{split['excluded_records']}**, only because no exact/reachable candidate subgraph could be mapped.",
        "- Function and callgraph mapping use exact ELF function entries and manifest call targets. No nearest-address, name-similarity, or Gold-address feature was used.", "",
        "## Leakage control", "",
        f"- DEV groups: **{len(split['dev_groups'])}**, TRAIN groups: **{len(split['train_groups'])}**, overlap: **{len(split['group_overlap'])}**.",
        "- One source-case group per CWE was selected by a fixed SHA256 rule before model training. All good/bad and O0/O2 variants of a group stay together.",
        "- `TASK2_VALIDATION_V1`, `TASK2_HOLDOUT_V2`, and `HOLDOUT_V3` were not opened or used.", "",
        "## Excluded records", "",
        "```json", json.dumps(excluded, indent=2, sort_keys=True), "```",
    ]
    write(OUT / "GNN_SUPERVISION_AUDIT.md", "\n".join(supervision))

    subgraph = [
        "# Candidate Subgraph Report", "",
        f"- Frozen candidate records: **{cov['candidate_total']}**.",
        f"- Exact 2-hop local USG subgraphs built: **{cov['candidate_subgraphs']}** ({cov['candidate_subgraph_rate']:.2%}).",
        f"- Mean nodes/subgraph: **{cov['mean_nodes']:.2f}**; cap: {cov['max_nodes']} nodes.",
        f"- Exact clean h_sym anchor coverage: **{cov['exact_clean_hsym_anchor_count']}/{cov['candidate_subgraphs']} ({cov['exact_clean_hsym_anchor_rate']:.2%})**.",
        f"- Exact clean h_sym node coverage: **{cov['exact_clean_hsym_node_count']}/{cov['exact_clean_hsym_node_total']} ({cov['exact_clean_hsym_node_rate']:.2%})**.",
        "- Missing embeddings are represented only by zeros plus an explicit availability bit; no old embedding, nearest address, or fuzzy identity fills a gap.",
        "- Subgraphs use resolved USG edges, exact anchor node identity and up to two undirected message-passing hops. Direction and edge-type counts remain in the index audit.", "",
        "## Artifacts", "",
        "- `candidate_subgraphs_v020.pt`: tensor artifact used for training.",
        "- `CANDIDATE_SUBGRAPH_INDEX.jsonl`: candidate-to-node/edge identity audit.",
        "- `FEATURE_SCHEMA.json`: exact structural, semantic and embedding feature contract.", "",
        "## Failures", "", "```json", json.dumps(cov["failure_counts"], indent=2, sort_keys=True), "```",
    ]
    write(OUT / "CANDIDATE_SUBGRAPH_REPORT.md", "\n".join(subgraph))

    table = ["| Arm | loss first→last | grad min | reload | DEV P/R/F1/AUROC | budget good burden |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, arm in arms.items():
        m = arm["dev"]["all_candidate_function_metrics"]
        table.append(f"| {name} | {arm['initial_loss']:.4f}→{arm['final_loss']:.4f} | {min(arm['gradient_norms']):.3g} | {arm['reload_passed']} | {m['precision']:.3f}/{m['recall']:.3f}/{m['f1']:.3f}/{m['auroc'] if m['auroc'] is not None else 'NA'} | {arm['dev']['good_candidate_burden_at_budget']}/{arm['dev']['good_candidate_burden_full']} |")
    conclusion = metrics["seraph_ranking_conclusion"]
    report = [
        "# SERAPH Task2 v0.2.0 — GNN Vulnerability Risk Ranking Smoke", "",
        f"Final status: **`{metrics['final_status']}`**.", "",
        "## Scope and integrity", "",
        "- Candidate generation rules are frozen at v0.1.12; this run only ranks the DEVELOPMENT_SMOKE candidate set.",
        f"- Frozen candidate input: **{metrics['candidate_count']}** records, SHA256 `{metrics['candidate_sha256']}`.",
        "- Task1 models, Gold, v0.1.12 outputs and all historical validation/holdout artifacts remained read-only.",
        "- No long training, symbolic execution, LLM security query, Task1 retraining or candidate rule change was performed.", "",
        "## Supervision", "",
        "Candidate-level labels are unavailable, so the smoke uses function-level MIL. Candidates in a bad function are not individually labeled positive.",
        f"Usable bags: **{split['usable_bags']}**; excluded: **{split['excluded_records']}**. Source-case group leakage: **{len(split['group_overlap'])}**.", "",
        "## Subgraph and semantic coverage", "",
        f"Candidate subgraphs: **{cov['candidate_subgraphs']}/{cov['candidate_total']} ({cov['candidate_subgraph_rate']:.2%})**.",
        f"Candidates with a nonzero SERAPH semantic feature vector: **{metrics['semantic_feature_coverage']['nonzero_candidates']}/{cov['candidate_subgraphs']} ({metrics['semantic_feature_coverage']['rate']:.2%})**.",
        f"Exact clean h_sym anchor mapping: **{cov['exact_clean_hsym_anchor_rate']:.2%}**.", "",
        "## 20-step smoke", "", *table, "",
        "All arms used the same mapped candidate set, split, optimizer, step count and evaluator. Candidate-level instruction accuracy is intentionally not reported.", "",
        "## Ranking attribution", "",
        f"- Conclusion: **`{conclusion}`**.",
        f"- G1−G0 DEV F1: {metrics['comparisons']['G1_minus_G0']['f1_delta']:+.4f}; AUROC: {metrics['comparisons']['G1_minus_G0']['auroc_delta']:+.4f}; budget good-burden delta: {metrics['comparisons']['G1_minus_G0']['good_budget_burden_delta']:+d}.",
        f"- G2−G0 DEV F1: {metrics['comparisons']['G2_minus_G0']['f1_delta']:+.4f}; AUROC: {metrics['comparisons']['G2_minus_G0']['auroc_delta']:+.4f}; budget good-burden delta: {metrics['comparisons']['G2_minus_G0']['good_budget_burden_delta']:+d}.",
        "- This is one deterministic small split and 20 optimizer steps. Any improvement is preliminary smoke evidence, not a formal statistical claim.", "",
        "## Gate", "",
        "The data path, MIL supervision, gradients, parameter updates, checkpoints and true reload are operational. Formal training should use repeated group splits/seeds and predeclared ranking metrics, while preserving all validation/holdout sets for later untouched evaluation.", "",
        f"Final status: `{metrics['final_status']}`",
    ]
    write(OUT / "TASK2_V0_2_0_GNN_SMOKE_REPORT.md", "\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    input_hashes = {str(path): sha(path) for path in (CANDIDATES, GOLD, MANIFEST, USG_NODES, USG_EDGES, HSYM, HSYM_INDEX)}
    candidates = list(rows(CANDIDATES))
    gold = list(rows(GOLD))
    if len(candidates) != 2158 or len(gold) != 100 or sum(bool(x["vulnerable"]) for x in gold) != 50:
        raise RuntimeError("DEVELOPMENT_SMOKE frozen input population changed")
    feature_schema = {
        "schema": "seraph_task2_v020_feature_schema", "candidate_set": "V012_FROZEN_DEVELOPMENT_SMOKE_A2",
        "candidate_level_label": "UNKNOWN_MIL_INSTANCE_LABEL", "structural_node": STRUCT_NODE_SCHEMA,
        "structural_candidate": STRUCT_CAND_SCHEMA, "semantic_candidate": SEMANTIC_SCHEMA,
        "frozen_clean_hsym": {"dimension": 64, "mapping": "EXACT_RAW_USG_NODE_ID_ONLY"},
        "prohibited_identity_fallbacks": ["nearest-address", "function-name-similarity", "Gold-identity-feature"],
    }
    dump(OUT / "FEATURE_SCHEMA.json", feature_schema)
    graphs, coverage, _ = create_subgraphs(candidates, gold)
    bags, excluded, split_audit = build_bags(graphs, gold)
    dump(OUT / "MIL_BAG_SPLIT.json", {"split_audit": split_audit, "bags": bags, "excluded": excluded})
    semantic_nonzero = sum(bool(float(g["cand_semantic"].abs().sum()) > 0) for g in graphs)
    semantic_module_use = {
        name: sum(bool(c.get(f"used_{name}")) for c in candidates)
        for name in ("TypeSem", "InterfaceSem", "LayoutSem", "APISem")
    }
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    arms = {}
    hdim = coverage["hsym_dimension"]
    for name in ("G0_STRUCTURAL_GNN", "G1_STRUCTURAL_PLUS_SERAPH_FEATURES", "G2_G1_PLUS_FROZEN_SERAPH_EMBEDDINGS"):
        arms[name] = train_arm(name, bags, graphs, hdim, device)
        dump(OUT / "reload_records" / f"{name}.json", {
            "checkpoint": arms[name]["checkpoint"], "checkpoint_sha256": arms[name]["checkpoint_sha256"],
            "reload_passed": arms[name]["reload_passed"],
            "reload_max_probability_difference": arms[name]["reload_max_probability_difference"],
        })
    def compact(arm: dict) -> Tuple[float, float, int]:
        m = arm["dev"]["all_candidate_function_metrics"]
        return m["f1"], (m["auroc"] if m["auroc"] is not None else 0.0), arm["dev"]["good_candidate_burden_at_budget"]
    g0 = compact(arms["G0_STRUCTURAL_GNN"])
    comparisons = {}
    for key, name in (("G1_minus_G0", "G1_STRUCTURAL_PLUS_SERAPH_FEATURES"), ("G2_minus_G0", "G2_G1_PLUS_FROZEN_SERAPH_EMBEDDINGS")):
        other = compact(arms[name])
        comparisons[key] = {"f1_delta": other[0] - g0[0], "auroc_delta": other[1] - g0[1],
                            "good_budget_burden_delta": other[2] - g0[2]}
    exceeds = any(value["f1_delta"] > 0 and value["auroc_delta"] > 0 and value["good_budget_burden_delta"] <= 0 for value in comparisons.values())
    conclusion = "PRELIMINARY_SERAPH_RANKING_SIGNAL_NOT_YET_VERIFIED" if exceeds else "NO_VERIFIED_SERAPH_RANKING_GAIN"
    all_pass = (
        coverage["candidate_subgraphs"] > 0
        and not split_audit["group_overlap"]
        and all(len({b["label"] for b in bags if b["split"] == s}) == 2 for s in ("TRAIN", "DEV"))
        and all(a["all_loss_finite"] and a["all_gradients_nonzero"] and a["parameter_update_verified"] and a["reload_passed"] for a in arms.values())
    )
    metrics = {
        "schema": "seraph_task2_v020_gnn_smoke_metrics", "final_status": "READY_FOR_GNN_FORMAL_TRAINING_DESIGN" if all_pass else "BLOCKED_GNN_DATA_OR_SUPERVISION",
        "dataset": "DEVELOPMENT_SMOKE_ONLY", "historical_holdouts_used": False,
        "candidate_count": len(candidates), "candidate_sha256": sha(CANDIDATES), "input_hashes": input_hashes,
        "candidate_level_gold_available": False, "supervision_method": "FUNCTION_LEVEL_MULTIPLE_INSTANCE_LEARNING",
        "subgraph_coverage": coverage,
        "semantic_feature_coverage": {"nonzero_candidates": semantic_nonzero, "rate": semantic_nonzero / max(1, len(graphs)), "module_used_candidate_counts": semantic_module_use},
        "supervision": {"usable_bags": len(bags), "excluded_records": len(excluded), "split_audit": split_audit},
        "device": {"requested": args.device, "actual": str(device), "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"},
        "training_budget": {"optimizer_steps": STEPS, "batch_bags": BATCH_BAGS, "seed": SEED, "formal_long_training": False},
        "arms": arms, "comparisons": comparisons, "seraph_ranking_conclusion": conclusion,
        "elapsed_seconds": time.time() - started,
    }
    dump(OUT / "GNN_SMOKE_METRICS.json", metrics)
    make_reports(metrics, excluded)
    dump(OUT / "INPUT_FREEZE_AND_INTEGRITY.json", {"input_hashes_before": input_hashes,
         "input_hashes_after": {path: sha(Path(path)) for path in input_hashes},
         "unchanged": all(sha(Path(path)) == digest for path, digest in input_hashes.items()),
         "historical_holdout_paths_opened": [], "task1_write_operations": 0, "gold_write_operations": 0})
    print(json.dumps({"status": metrics["final_status"], "subgraphs": len(graphs), "bags": len(bags),
                      "ranking_conclusion": conclusion,
                      "report": str(OUT / "TASK2_V0_2_0_GNN_SMOKE_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
