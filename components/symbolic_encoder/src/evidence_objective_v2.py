#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared multi-positive evidence objective for 2.2.1.4.

The guide describes a conceptual schema.  Real-world USG exports use several
slightly different field layouts, so this module deliberately separates:

* schema adaptation (``resolve_evidence_record``),
* target-preserving batching (``TargetGroupedBatchSampler``),
* row-level multi-positive InfoNCE, and
* target-level learned evidence attention.

No Gold label is read here.  Group identity comes only from the USG target
node (or a conservative row-unique fallback when that identity is absent).
The module is compatible with Python 3.8 and does not require torch-scatter.
"""

import random
import re
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


CANONICAL_EVIDENCE_TYPES = (
    "deref_chain",
    "fixed_offset",
    "def_use",
    "api_call",
    "unknown",
)

CANONICAL_TARGET_TYPES = (
    "TypeSem",
    "InterfaceSem",
    "APISem",
    "LayoutSem",
    "unknown",
)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _nonempty(value):
    return value is not None and value != "" and value != [] and value != {}


def _sections(record):
    """Yield known wrappers from broad to narrow without recursive surprises."""
    root = _mapping(record)
    yielded = set()
    queue = [root]
    wrapper_names = (
        "attrs",
        "attributes",
        "metadata",
        "payload",
        "anchor",
        "evidence",
        "edge",
        "relation",
        "neighborhood",
        "context",
    )
    while queue:
        section = queue.pop(0)
        marker = id(section)
        if marker in yielded:
            continue
        yielded.add(marker)
        yield section
        for name in wrapper_names:
            nested = section.get(name)
            if isinstance(nested, dict):
                queue.append(nested)


def first_value(record, aliases, default=None):
    # Alias priority is more important than wrapper depth.  For example, a
    # nested explicit ``evidence_type`` must beat a root-level generic
    # ``type`` field used for an unrelated record discriminator.
    sections = list(_sections(record))
    for name in aliases:
        for section in sections:
            value = section.get(name)
            if _nonempty(value):
                return value, name
    return default, ""


def _node_id(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        for key in (
            "node_id",
            "id",
            "ref",
            "node",
            "source_node",
            "target_node",
            "raw_id",
        ):
            if _nonempty(value.get(key)):
                return _node_id(value.get(key))
    return ""


def extract_node_ids(value):
    """Accept lists of node dictionaries, strings, or nested side objects."""
    if value is None:
        return []
    if isinstance(value, (str, int, float, dict)):
        if isinstance(value, dict):
            for key in ("nodes", "node_ids", "members", "items"):
                if isinstance(value.get(key), (list, tuple)):
                    return extract_node_ids(value.get(key))
        node = _node_id(value)
        return [node] if node else []
    if isinstance(value, (list, tuple, set)):
        out = []
        seen = set()
        for item in value:
            for node in extract_node_ids(item):
                if node and node not in seen:
                    out.append(node)
                    seen.add(node)
        return out
    return []


def normalize_evidence_type(value):
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "unknown").lower()).strip("_")
    aliases = {
        "fixedoffset": "fixed_offset",
        "offset": "fixed_offset",
        "field_offset": "fixed_offset",
        "memory_offset": "fixed_offset",
        "access_offset": "fixed_offset",
        "deref": "deref_chain",
        "dereference": "deref_chain",
        "dereference_chain": "deref_chain",
        "pointer_chain": "deref_chain",
        "defuse": "def_use",
        "data_flow": "def_use",
        "dataflow": "def_use",
        "param_def_use": "def_use",
        "return_use": "def_use",
        "api": "api_call",
        "call": "api_call",
        "call_instruction": "api_call",
        "external_call": "api_call",
    }
    text = aliases.get(text, text)
    return text if text in CANONICAL_EVIDENCE_TYPES else "unknown"


def normalize_target_type(value):
    text = re.sub(r"[^a-z0-9]+", "", str(value or "unknown").lower())
    aliases = {
        "typesem": "TypeSem",
        "type": "TypeSem",
        "interfacesem": "InterfaceSem",
        "interface": "InterfaceSem",
        "apisem": "APISem",
        "api": "APISem",
        "layoutsem": "LayoutSem",
        "layout": "LayoutSem",
    }
    return aliases.get(text, "unknown")


def infer_sample_id(*node_ids):
    for node_id in node_ids:
        text = str(node_id or "")
        if "::" in text:
            return text.split("::", 1)[0]
    return ""


def resolve_strength(record, default=1.0):
    aliases = (
        "strength",
        "evidence_strength",
        "edge_strength",
        "confidence",
        "score",
        "weight",
        "probability",
    )
    value, source = first_value(record, aliases, None)
    try:
        if value is not None:
            return max(0.0, min(1.0, float(value))), source
    except (TypeError, ValueError):
        pass

    values = []
    root = _mapping(record)
    for name in ("cross_edges", "low_edges", "high_edges", "edges"):
        for edge in root.get(name, []) or []:
            if not isinstance(edge, dict):
                continue
            edge_value, _ = first_value(edge, aliases, None)
            try:
                if edge_value is not None:
                    values.append(max(0.0, min(1.0, float(edge_value))))
            except (TypeError, ValueError):
                continue
    if values:
        return sum(values) / len(values), "nested_edges_mean"
    return float(default), "default"


def _global_target_key(sample_id, target_node):
    target_node = str(target_node or "")
    sample_id = str(sample_id or "")
    if not target_node:
        return ""
    if "::" in target_node or not sample_id:
        return target_node
    return "%s::%s" % (sample_id, target_node)


def resolve_evidence_record(record, row_index=0):
    """Adapt canonical and observed USG neighborhood records.

    The fallback target group is intentionally unique per input row.  It is
    unsafe to treat two missing target identifiers as the same semantic target.
    """
    source_value, source_path = first_value(
        record,
        ("source_node", "source", "src", "from_node", "low_node", "query_node"),
        "",
    )
    target_value, target_path = first_value(
        record,
        ("target_node", "target", "dst", "to_node", "high_node", "semantic_node"),
        "",
    )
    source_node = _node_id(source_value)
    target_node = _node_id(target_value)

    low_value, low_path = first_value(
        record,
        ("low_nodes", "low_node_ids", "source_nodes", "low_side", "evidence_nodes"),
        [],
    )
    high_value, high_path = first_value(
        record,
        ("high_nodes", "high_node_ids", "target_nodes", "high_side", "semantic_nodes"),
        [],
    )
    low_node_ids = extract_node_ids(low_value)
    high_node_ids = extract_node_ids(high_value)
    if source_node and source_node not in low_node_ids:
        low_node_ids.append(source_node)
    if target_node and target_node not in high_node_ids:
        high_node_ids.append(target_node)

    # A single high-side node is a safe target identity fallback.  Multiple
    # high nodes are not: choosing one would silently change supervision.
    target_group_source = "target_node"
    if not target_node and len(high_node_ids) == 1:
        target_node = high_node_ids[0]
        target_group_source = "single_high_node"

    sample_value, sample_path = first_value(
        record,
        ("sample_id", "program_id", "binary_id", "project_id", "sample"),
        "",
    )
    sample_id = str(sample_value or "") or infer_sample_id(source_node, target_node)

    anchor_value, anchor_path = first_value(
        record,
        ("anchor_id", "evidence_id", "pair_id", "record_id", "id"),
        "",
    )
    anchor_id = str(anchor_value or "") or "anchor_%d" % int(row_index)

    evidence_value, evidence_path = first_value(
        record,
        ("evidence_type", "anchor_type", "relation_type", "edge_type", "type"),
        "unknown",
    )
    target_type_value, target_type_path = first_value(
        record,
        ("target_type", "semantic_type", "target_node_type", "high_type", "node_type"),
        "unknown",
    )
    strength, strength_path = resolve_strength(record)

    target_group_id = _global_target_key(sample_id, target_node)
    if not target_group_id:
        target_group_source = "row_unique_fallback"
        target_group_id = "__unresolved_target__::%s::%s::%d" % (
            sample_id or "unknown_sample",
            anchor_id,
            int(row_index),
        )

    return {
        "sample_id": sample_id,
        "anchor_id": anchor_id,
        "source_node": source_node,
        "target_node": target_node,
        "target_group_id": target_group_id,
        "target_group_source": target_group_source,
        "evidence_type": normalize_evidence_type(evidence_value),
        "target_type": normalize_target_type(target_type_value),
        "strength": strength,
        "low_node_ids": low_node_ids,
        "high_node_ids": high_node_ids,
        "compatibility_paths": {
            "source": source_path,
            "target": target_path,
            "low_nodes": low_path,
            "high_nodes": high_path,
            "sample": sample_path,
            "anchor": anchor_path,
            "evidence_type": evidence_path,
            "target_type": target_type_path,
            "strength": strength_path,
        },
    }


def parse_evidence_weights(spec, evidence_types=None):
    evidence_types = tuple(evidence_types or CANONICAL_EVIDENCE_TYPES)
    weights = {
        "deref_chain": 1.0,
        "fixed_offset": 1.0,
        "def_use": 1.5,
        "api_call": 3.0,
        "unknown": 1.0,
    }
    if spec:
        for part in str(spec).split(","):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            try:
                weights[normalize_evidence_type(key)] = float(value)
            except ValueError:
                raise ValueError("invalid evidence weight: %s" % part)
    return [float(weights.get(name, 1.0)) for name in evidence_types]


def make_target_group_stats(group_keys):
    counts = Counter(str(key) for key in group_keys)
    sizes = list(counts.values())
    return {
        "records": len(group_keys),
        "unique_targets": len(counts),
        "multi_evidence_targets": sum(size > 1 for size in sizes),
        "multi_evidence_records": sum(size for size in sizes if size > 1),
        "max_evidence_per_target": max(sizes) if sizes else 0,
        "mean_evidence_per_target": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "singleton_target_rate": (sum(size == 1 for size in sizes) / len(sizes)) if sizes else 0.0,
    }


class TargetGroupedBatchSampler(Sampler):
    """Pack complete target groups into a batch whenever possible.

    ``max_group_size=0`` preserves every complete group.  A positive value is
    an explicit memory-safety escape hatch and is reported as split groups.
    """

    def __init__(
        self,
        target_group_ids,
        batch_size,
        shuffle=True,
        seed=1337,
        drop_last=False,
        max_group_size=0,
    ):
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.max_group_size = max(0, int(max_group_size or 0))
        self.epoch = 0
        grouped = defaultdict(list)
        for local_index, target in enumerate(target_group_ids):
            grouped[str(target)].append(local_index)
        self.groups = list(grouped.items())
        self.split_target_groups = sum(
            self.max_group_size > 0 and len(indices) > self.max_group_size
            for _, indices in self.groups
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _chunks(self):
        chunks = []
        for target, indices in self.groups:
            if self.max_group_size and len(indices) > self.max_group_size:
                for start in range(0, len(indices), self.max_group_size):
                    chunks.append((target, list(indices[start:start + self.max_group_size])))
            else:
                chunks.append((target, list(indices)))
        return chunks

    def _batches(self):
        rng = random.Random(self.seed + self.epoch)
        chunks = self._chunks()
        if self.shuffle:
            rng.shuffle(chunks)
        batches = []
        current = []
        for _, indices in chunks:
            if self.shuffle:
                rng.shuffle(indices)
            if current and len(current) + len(indices) > self.batch_size:
                batches.append(current)
                current = []
            if len(indices) > self.batch_size:
                # Exact grouping takes priority over the nominal batch size.
                if current:
                    batches.append(current)
                    current = []
                batches.append(indices)
            else:
                current.extend(indices)
        if current and (not self.drop_last or len(current) >= self.batch_size):
            batches.append(current)
        return batches

    def __iter__(self):
        for batch in self._batches():
            yield batch

    def __len__(self):
        return len(self._batches())

    def audit(self):
        sizes = [len(indices) for _, indices in self.groups]
        return {
            "batching_mode": "target_grouped",
            "target_groups": len(self.groups),
            "batch_size": self.batch_size,
            "batches": len(self),
            "oversized_target_groups": sum(size > self.batch_size for size in sizes),
            "split_target_groups": self.split_target_groups,
            "max_group_size_limit": self.max_group_size,
            "max_observed_group_size": max(sizes) if sizes else 0,
        }


def build_positive_mask(target_group_ids, device=None):
    size = len(target_group_ids)
    mask = torch.zeros((size, size), dtype=torch.bool, device=device)
    groups = defaultdict(list)
    for index, target in enumerate(target_group_ids):
        groups[str(target)].append(index)
    for indices in groups.values():
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        mask[idx.unsqueeze(1), idx.unsqueeze(0)] = True
    return mask


def _weighted_mean(values, weights):
    if weights is None:
        return values.mean()
    weights = weights.to(values.dtype).clamp_min(0.0)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def multi_positive_info_nce(
    support,
    target,
    target_group_ids,
    temperature=0.07,
    sample_weight=None,
    symmetric=True,
):
    """InfoNCE whose numerator contains every valid same-target positive."""
    if support.ndim != 2 or target.ndim != 2 or support.shape != target.shape:
        raise ValueError("support and target must be equal-shape 2D tensors")
    if support.size(0) != len(target_group_ids):
        raise ValueError("target_group_ids length does not match batch")
    if support.size(0) == 0:
        return support.sum() * 0.0, {
            "positive_pairs": 0,
            "non_diagonal_positive_pairs": 0,
            "unique_targets": 0,
            "has_negative_targets": False,
        }

    support = F.normalize(support, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)
    logits = support @ target.t() / max(float(temperature), 1e-6)
    positive_mask = build_positive_mask(target_group_ids, logits.device)

    all_lse = torch.logsumexp(logits, dim=1)
    pos_lse = torch.logsumexp(logits.masked_fill(~positive_mask, float("-inf")), dim=1)
    row_loss = all_lse - pos_lse
    loss = _weighted_mean(row_loss, sample_weight)

    if symmetric:
        all_lse_t = torch.logsumexp(logits.t(), dim=1)
        pos_lse_t = torch.logsumexp(
            logits.t().masked_fill(~positive_mask.t(), float("-inf")),
            dim=1,
        )
        col_loss = all_lse_t - pos_lse_t
        loss = 0.5 * (loss + _weighted_mean(col_loss, sample_weight))

    positive_pairs = int(positive_mask.sum().detach().cpu().item())
    unique_targets = len(set(str(x) for x in target_group_ids))
    return loss, {
        "positive_pairs": positive_pairs,
        "non_diagonal_positive_pairs": positive_pairs - support.size(0),
        "unique_targets": unique_targets,
        "has_negative_targets": unique_targets > 1,
    }


class TargetEvidenceAttention(nn.Module):
    """Learn evidence importance, then aggregate once per semantic target."""

    def __init__(
        self,
        embed_dim,
        evidence_type_count,
        evidence_type_dim=16,
        hidden_dim=128,
        dropout=0.1,
        strength_log_scale=1.0,
        attention_temperature=1.0,
    ):
        super().__init__()
        self.evidence_type_embedding = nn.Embedding(
            max(1, int(evidence_type_count)),
            int(evidence_type_dim),
        )
        input_dim = int(embed_dim) * 3 + int(evidence_type_dim)
        self.score = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.Tanh(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.strength_log_scale = max(0.0, float(strength_log_scale))
        self.attention_temperature = max(float(attention_temperature), 1e-6)

    def forward(self, support, target, evidence_type_ids, strengths, target_group_ids):
        if support.size(0) != len(target_group_ids):
            raise ValueError("target_group_ids length does not match attention batch")
        strengths = strengths.to(support.dtype).clamp(1e-6, 1.0)
        etype = self.evidence_type_embedding(evidence_type_ids)
        learned_input = torch.cat(
            [support, target, torch.abs(support - target), etype],
            dim=-1,
        )
        logits = self.score(learned_input).squeeze(-1)
        # This explicit, non-learned bias guarantees that equal evidence with
        # greater strength receives greater weight.  Strength also remains in
        # the loss weight, so later L2 normalization cannot erase it.
        logits = logits + self.strength_log_scale * torch.log(strengths)

        groups = defaultdict(list)
        order = []
        for index, target_group in enumerate(target_group_ids):
            key = str(target_group)
            if key not in groups:
                order.append(key)
            groups[key].append(index)

        row_weights = torch.zeros_like(logits)
        aggregate_support = []
        aggregate_target = []
        group_strength = []
        entropies = []
        effective_counts = []
        max_weights = []

        for key in order:
            indices = groups[key]
            idx = torch.tensor(indices, dtype=torch.long, device=support.device)
            local_logits = logits.index_select(0, idx) / self.attention_temperature
            weights = F.softmax(local_logits, dim=0)
            row_weights.index_copy_(0, idx, weights)

            support_group = support.index_select(0, idx)
            target_group = target.index_select(0, idx)
            aggregate_support.append(F.normalize((weights.unsqueeze(-1) * support_group).sum(0), p=2, dim=-1))
            aggregate_target.append(F.normalize((weights.unsqueeze(-1) * target_group).sum(0), p=2, dim=-1))
            group_strength.append((weights * strengths.index_select(0, idx)).sum())

            entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum()
            entropies.append(entropy)
            effective_counts.append(torch.exp(entropy))
            max_weights.append(weights.max())

        if aggregate_support:
            support_out = torch.stack(aggregate_support)
            target_out = torch.stack(aggregate_target)
            strength_out = torch.stack(group_strength)
            entropy_out = torch.stack(entropies)
            effective_out = torch.stack(effective_counts)
            max_weight_out = torch.stack(max_weights)
        else:
            dim = support.size(-1)
            support_out = support.new_zeros((0, dim))
            target_out = target.new_zeros((0, dim))
            strength_out = strengths.new_zeros((0,))
            entropy_out = strengths.new_zeros((0,))
            effective_out = strengths.new_zeros((0,))
            max_weight_out = strengths.new_zeros((0,))

        return {
            "support": support_out,
            "target": target_out,
            "group_strength": strength_out,
            "target_group_ids": order,
            "row_attention_weights": row_weights,
            "attention_entropy": entropy_out,
            "effective_evidence_count": effective_out,
            "max_attention_weight": max_weight_out,
        }


def target_level_info_nce(aggregate, temperature=0.07):
    support = aggregate["support"]
    target = aggregate["target"]
    count = support.size(0)
    if count <= 1:
        return support.sum() * 0.0
    logits = F.normalize(support, p=2, dim=-1) @ F.normalize(target, p=2, dim=-1).t()
    logits = logits / max(float(temperature), 1e-6)
    labels = torch.arange(count, dtype=torch.long, device=support.device)
    per_row = 0.5 * (
        F.cross_entropy(logits, labels, reduction="none")
        + F.cross_entropy(logits.t(), labels, reduction="none")
    )
    group_strength = aggregate.get("group_strength")
    if group_strength is not None:
        group_strength = group_strength.detach()
    return _weighted_mean(per_row, group_strength)


class MultiPositiveEvidenceObjective(nn.Module):
    """Combined row multi-positive and target-attention evidence loss."""

    def __init__(
        self,
        embed_dim,
        evidence_type_count,
        evidence_type_dim=16,
        attention_hidden_dim=128,
        attention_dropout=0.1,
        strength_log_scale=1.0,
        attention_temperature=1.0,
    ):
        super().__init__()
        self.attention = TargetEvidenceAttention(
            embed_dim=embed_dim,
            evidence_type_count=evidence_type_count,
            evidence_type_dim=evidence_type_dim,
            hidden_dim=attention_hidden_dim,
            dropout=attention_dropout,
            strength_log_scale=strength_log_scale,
            attention_temperature=attention_temperature,
        )

    def forward(
        self,
        support,
        target,
        evidence_type_ids,
        strengths,
        target_group_ids,
        evidence_type_weights,
        temperature=0.07,
        row_multi_positive_weight=1.0,
        target_attention_weight=1.0,
    ):
        evidence_type_weights = evidence_type_weights.to(support.device, support.dtype)
        sample_weight = evidence_type_weights.index_select(0, evidence_type_ids)
        sample_weight = sample_weight * (0.5 + strengths.to(support.dtype).clamp(0.0, 1.0))

        row_loss, row_stats = multi_positive_info_nce(
            support,
            target,
            target_group_ids,
            temperature=temperature,
            sample_weight=sample_weight,
            symmetric=True,
        )
        aggregate = self.attention(
            support,
            target,
            evidence_type_ids,
            strengths,
            target_group_ids,
        )
        target_loss = target_level_info_nce(aggregate, temperature=temperature)

        row_weight = max(0.0, float(row_multi_positive_weight))
        target_weight = max(0.0, float(target_attention_weight))
        denominator = max(row_weight + target_weight, 1e-8)
        total = (row_weight * row_loss + target_weight * target_loss) / denominator

        entropy = aggregate["attention_entropy"]
        effective = aggregate["effective_evidence_count"]
        max_weight = aggregate["max_attention_weight"]
        diagnostics = dict(row_stats)
        diagnostics.update({
            "row_multi_positive_loss": float(row_loss.detach().cpu().item()),
            "target_attention_loss": float(target_loss.detach().cpu().item()),
            "target_groups_in_batch": len(aggregate["target_group_ids"]),
            "attention_entropy_mean": float(entropy.mean().detach().cpu().item()) if entropy.numel() else 0.0,
            "effective_evidence_count_mean": float(effective.mean().detach().cpu().item()) if effective.numel() else 0.0,
            "max_attention_weight_mean": float(max_weight.mean().detach().cpu().item()) if max_weight.numel() else 0.0,
        })
        return total, diagnostics, aggregate


def attention_audit_rows(aggregate, evidence_ids=None):
    """Create JSON-serializable attention rows for optional diagnostics."""
    row_weights = aggregate["row_attention_weights"].detach().cpu().tolist()
    evidence_ids = list(evidence_ids or ["row_%d" % i for i in range(len(row_weights))])
    return [
        {"evidence_id": evidence_ids[i], "attention_weight": float(weight)}
        for i, weight in enumerate(row_weights)
    ]
