#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export an approved frozen node encoder as a production embedding table.

The exporter deliberately uses ``node_base`` from either the audited Stage3
checkpoint or the unified Stage5 v6.5 checkpoint.  This branch is defined for
every USG node.  In unified Stage5 it lies on the evidence residual path and
is therefore trained by evidence, typed constraint-margin and Train-only graph
reconstruction losses.  Stage3 remains supported for reproducible baselines.

Outputs (written atomically):

* node_embeddings.pt
* node_embedding_index.json
* node_embedding_export_manifest.json

The JSON index follows the exact row order of the tensor.  Each record carries
an ``embedding_ref`` suitable for the later BindingStore implementation.
Input schema adapters accept aliases observed in compatible USG
artifacts; the source files themselves are never renamed or rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised on the GPU server
    torch = None
    nn = None
    F = None


SCHEMA = "task2_node_embedding_export_v6_3"
OBJECTIVE_REVISION = "multi_positive_target_attention_v2_train_graph_firewall_v3"
ALLOWED_OBJECTIVE_REVISIONS = {
    OBJECTIVE_REVISION,
    "stage5_unified_three_loss_train_graph_v6_5",
}
ALLOWED_GRAPH_POLICIES = {
    "train_samples_only_v3",
    "train_samples_only_v6_5",
}
EXPORT_POLICIES = {
    OBJECTIVE_REVISION: "stage3_node_base_train_graph_encoder_v1",
    "stage5_unified_three_loss_train_graph_v6_5": (
        "stage5_unified_node_base_three_loss_encoder_v6_5"
    ),
}

NODE_ID_KEYS = (
    "node_id", "node_ref", "id", "uid", "global_id", "semantic_node_id",
)
SAMPLE_ID_KEYS = (
    "sample_id", "sample_hash", "binary_id", "binary_hash", "program_id",
    "sha256",
)
NODE_TYPE_KEYS = (
    "node_type", "semantic_type", "node_kind", "kind", "type",
)
LAYER_KEYS = ("layer", "semantic_layer", "level")
RAW_ID_KEYS = ("raw_id", "local_id", "object_id")
NESTED_KEYS = (
    "attrs", "attributes", "metadata", "meta", "context", "payload",
    "node", "record", "data",
)


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is required for node embedding export")


def _iter_dicts(value: Any) -> Iterable[Mapping[str, Any]]:
    """Yield a record and known nested schema containers without recursion loops."""
    if not isinstance(value, Mapping):
        return
    queue: List[Mapping[str, Any]] = [value]
    seen = set()
    while queue:
        current = queue.pop(0)
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        yield current
        for key in NESTED_KEYS:
            child = current.get(key)
            if isinstance(child, Mapping):
                queue.append(child)


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        for key in NODE_ID_KEYS + SAMPLE_ID_KEYS + ("name", "value"):
            nested = value.get(key)
            if nested is not None and not isinstance(nested, (Mapping, list, tuple)):
                return str(nested).strip()
        return ""
    if isinstance(value, (list, tuple, set)):
        return ""
    return str(value).strip()


def resolve_alias(record: Mapping[str, Any], keys: Sequence[str]) -> Tuple[str, str]:
    # Alias priority is global: a nested ``node_id`` must beat a root-level
    # generic ``id``.  This is common in wrapped USG records.
    containers = list(_iter_dicts(record))
    for key in keys:
        for container in containers:
            if key not in container:
                continue
            value = _scalar(container.get(key))
            if value:
                return value, key
    return "", ""


def infer_sample_id(node_id: str) -> str:
    value = str(node_id or "")
    return value.split("::", 1)[0] if "::" in value else ""


def infer_node_type(node_id: str) -> str:
    value = str(node_id or "")
    parts = value.split("::")
    if len(parts) >= 3 and parts[-2]:
        return parts[-2]
    for candidate in (
        "InstrSem", "BlockSem", "CFGSem", "DFGSem", "TypeSem",
        "InterfaceSem", "APISem", "LayoutSem",
    ):
        if candidate in value:
            return candidate
    return "unknown"


def resolve_node_metadata(record: Mapping[str, Any]) -> Dict[str, str]:
    node_id, node_id_source = resolve_alias(record, NODE_ID_KEYS)
    sample_id, sample_source = resolve_alias(record, SAMPLE_ID_KEYS)
    node_type, type_source = resolve_alias(record, NODE_TYPE_KEYS)
    layer, layer_source = resolve_alias(record, LAYER_KEYS)
    raw_id, raw_id_source = resolve_alias(record, RAW_ID_KEYS)
    if not sample_id:
        sample_id = infer_sample_id(node_id)
        sample_source = "node_id_prefix" if sample_id else ""
    if not node_type:
        node_type = infer_node_type(node_id)
        type_source = "node_id_token" if node_type != "unknown" else ""
    return {
        "node_id": node_id,
        "sample_id": sample_id,
        "node_type": node_type or "unknown",
        "layer": layer,
        "raw_id": raw_id,
        "node_id_source": node_id_source,
        "sample_id_source": sample_source,
        "node_type_source": type_source,
        "layer_source": layer_source,
        "raw_id_source": raw_id_source,
    }


def read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Invalid JSONL at %s:%d: %s" % (path, line_number, exc)
                ) from exc
            if not isinstance(row, Mapping):
                raise RuntimeError(
                    "Node JSONL row must be an object at %s:%d" % (path, line_number)
                )
            yield row


def load_serialized(path: Path) -> Any:
    _require_torch()
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception as torch_error:
        try:
            with path.open("rb") as stream:
                return pickle.load(stream)
        except Exception as pickle_error:
            raise RuntimeError(
                "Could not load %s as torch or pickle payload: torch=%s; pickle=%s"
                % (path, torch_error, pickle_error)
            ) from pickle_error


def _extract_tensor(payload: Any) -> "torch.Tensor":
    _require_torch()
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, Mapping):
        for key in (
            "features", "node_features", "x", "feature_tensor", "tensor", "data",
        ):
            value = payload.get(key)
            if torch.is_tensor(value):
                return value
            if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
                return torch.tensor(value, dtype=torch.float32)
    raise RuntimeError(
        "Feature payload must be a Tensor or contain features/node_features/x"
    )


def _node_id_from_item(item: Any) -> str:
    if isinstance(item, Mapping):
        return resolve_node_metadata(item)["node_id"]
    return str(item or "").strip()


def _extract_node_ids(payload: Any, row_count: int) -> List[str]:
    if not isinstance(payload, Mapping):
        return []
    for key in ("node_ids", "node_refs", "ids", "nodes"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)):
            result = [_node_id_from_item(item) for item in value]
            if len(result) != row_count:
                raise RuntimeError(
                    "%s length %d does not match feature rows %d"
                    % (key, len(result), row_count)
                )
            return result

    node_index = payload.get("node_index") or payload.get("index_by_node")
    if isinstance(node_index, Mapping):
        result = [""] * row_count
        for node_id, index in node_index.items():
            try:
                position = int(index)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Non-integer node index for %r" % node_id) from exc
            if position < 0 or position >= row_count:
                raise RuntimeError("Node index out of range for %r: %d" % (node_id, position))
            if result[position]:
                raise RuntimeError("Duplicate feature index %d" % position)
            result[position] = str(node_id)
        if any(not node_id for node_id in result):
            raise RuntimeError("node_index does not cover every feature row")
        return result
    return []


def load_feature_payload(path: Path) -> Tuple["torch.Tensor", List[str], Dict[str, Any]]:
    payload = load_serialized(path)
    features = _extract_tensor(payload)
    if features.ndim != 2:
        raise RuntimeError("Features must be rank-2, got shape %r" % (tuple(features.shape),))
    features = features.detach().cpu().float().contiguous()
    if features.shape[0] <= 0 or features.shape[1] <= 0:
        raise RuntimeError("Feature tensor is empty")
    if not torch.isfinite(features).all():
        raise RuntimeError("Feature tensor contains NaN or Inf")
    node_ids = _extract_node_ids(payload, int(features.shape[0]))
    meta = {
        "feature_schema": payload.get("feature_schema", "") if isinstance(payload, Mapping) else "",
        "feature_dim_declared": payload.get("feature_dim") if isinstance(payload, Mapping) else None,
        "backend": payload.get("backend", "") if isinstance(payload, Mapping) else "",
        "node_ids_source": "feature_payload" if node_ids else "nodes_jsonl_order",
    }
    return features, node_ids, meta


def scan_nodes(
    nodes_path: Path,
    expected_node_ids: Sequence[str],
    stop_when_complete: bool,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Align heterogeneous node JSONL metadata to the feature row order."""
    expected_count = len(expected_node_ids)
    index_by_id = {node_id: index for index, node_id in enumerate(expected_node_ids)}
    if len(index_by_id) != expected_count:
        counts = Counter(expected_node_ids)
        duplicates = [node_id for node_id, count in counts.items() if count > 1][:20]
        raise RuntimeError("Duplicate node IDs in feature payload: %r" % duplicates)

    metadata: List[Optional[Dict[str, str]]] = [None] * expected_count
    seen_rows = 0
    matched_rows = 0
    extra_rows = 0
    duplicate_rows = 0
    source_counts: Counter = Counter()

    for row in read_jsonl(nodes_path):
        seen_rows += 1
        resolved = resolve_node_metadata(row)
        node_id = resolved["node_id"]
        position = index_by_id.get(node_id)
        if position is None:
            extra_rows += 1
            continue
        previous = metadata[position]
        if previous is not None:
            duplicate_rows += 1
            if (
                previous.get("sample_id") != resolved.get("sample_id")
                or previous.get("node_type") != resolved.get("node_type")
            ):
                raise RuntimeError(
                    "Conflicting duplicate node metadata for %s" % node_id
                )
            continue
        metadata[position] = resolved
        matched_rows += 1
        source_counts["node_id:" + (resolved["node_id_source"] or "missing")] += 1
        source_counts["sample_id:" + (resolved["sample_id_source"] or "missing")] += 1
        source_counts["node_type:" + (resolved["node_type_source"] or "missing")] += 1
        if stop_when_complete and matched_rows == expected_count:
            break

    missing_examples = []
    final_metadata: List[Dict[str, str]] = []
    for index, node_id in enumerate(expected_node_ids):
        record = metadata[index]
        if record is None:
            if len(missing_examples) < 20:
                missing_examples.append(node_id)
            record = {
                "node_id": node_id,
                "sample_id": infer_sample_id(node_id),
                "node_type": infer_node_type(node_id),
                "layer": "",
                "raw_id": "",
                "node_id_source": "feature_payload",
                "sample_id_source": "node_id_prefix",
                "node_type_source": "node_id_token",
                "layer_source": "",
                "raw_id_source": "",
            }
        final_metadata.append(record)

    coverage = matched_rows / max(1, expected_count)
    return final_metadata, {
        "nodes_rows_scanned": seen_rows,
        "matched_node_rows": matched_rows,
        "missing_node_rows": expected_count - matched_rows,
        "extra_node_rows_seen": extra_rows,
        "duplicate_node_rows": duplicate_rows,
        "node_metadata_coverage": coverage,
        "missing_node_examples": missing_examples,
        "schema_source_counts": dict(source_counts),
        "scan_stopped_when_complete": bool(stop_when_complete and matched_rows == expected_count),
    }


def load_nodes_in_file_order(nodes_path: Path, expected_count: int) -> Tuple[List[str], List[Dict[str, str]]]:
    node_ids: List[str] = []
    metadata: List[Dict[str, str]] = []
    for row in read_jsonl(nodes_path):
        resolved = resolve_node_metadata(row)
        if not resolved["node_id"]:
            raise RuntimeError("Node row has no resolvable ID")
        node_ids.append(resolved["node_id"])
        metadata.append(resolved)
    if len(node_ids) != expected_count:
        raise RuntimeError(
            "Nodes JSONL rows %d do not match feature rows %d and feature payload has no IDs"
            % (len(node_ids), expected_count)
        )
    if len(set(node_ids)) != len(node_ids):
        raise RuntimeError("Nodes JSONL contains duplicate node IDs")
    return node_ids, metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(path: Path, policy: str) -> Dict[str, Any]:
    stat = path.stat()
    result = {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": None,
    }
    if policy == "sha256":
        result["sha256"] = sha256_file(path)
    return result


def normalize_state_dict(payload: Any) -> Tuple[Dict[str, "torch.Tensor"], Mapping[str, Any]]:
    _require_torch()
    checkpoint = payload if isinstance(payload, Mapping) else {}
    state: Any = payload
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                state = candidate
                break
    if not isinstance(state, Mapping):
        raise RuntimeError("Checkpoint has no model_state_dict/state_dict mapping")
    normalized = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        name = str(key)
        while name.startswith("module.") or name.startswith("model."):
            name = name.split(".", 1)[1]
        normalized[name] = value.detach().cpu()
    return normalized, checkpoint


def _state_value(state: Mapping[str, "torch.Tensor"], suffix: str) -> "torch.Tensor":
    if suffix in state:
        return state[suffix]
    matches = [value for key, value in state.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one checkpoint tensor ending in %s, found %d" % (suffix, len(matches))
        )
    return matches[0]


def _state_value_one_of(
    state: Mapping[str, "torch.Tensor"], suffixes: Sequence[str]
) -> Tuple["torch.Tensor", str]:
    for suffix in suffixes:
        if suffix in state:
            return state[suffix], suffix
        matches = [(key, value) for key, value in state.items() if key.endswith(suffix)]
        if len(matches) == 1:
            return matches[0][1], suffix
        if len(matches) > 1:
            raise RuntimeError(
                "Ambiguous checkpoint tensors ending in %s" % suffix
            )
    raise RuntimeError(
        "Checkpoint has none of the required tensors: %s" % ", ".join(suffixes)
    )


class FrozenNodeBaseEncoder(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        _require_torch()
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, features: "torch.Tensor") -> "torch.Tensor":
        return F.normalize(self.layers(features), p=2, dim=-1)


def build_frozen_encoder(
    state: Mapping[str, "torch.Tensor"], feature_dim: int
) -> Tuple[FrozenNodeBaseEncoder, Dict[str, int]]:
    first_weight = _state_value(state, "node_base.0.weight")
    output_weight, output_weight_suffix = _state_value_one_of(
        state, ("node_base.4.weight", "node_base.5.weight")
    )
    output_prefix = output_weight_suffix.rsplit(".", 1)[0]
    if first_weight.ndim != 2 or output_weight.ndim != 2:
        raise RuntimeError("node_base linear weights must be rank-2")
    input_dim = int(first_weight.shape[1])
    hidden_dim = int(first_weight.shape[0])
    embedding_dim = int(output_weight.shape[0])
    if input_dim != int(feature_dim):
        raise RuntimeError(
            "Checkpoint input dimension %d != feature dimension %d"
            % (input_dim, feature_dim)
        )
    if int(output_weight.shape[1]) != hidden_dim:
        raise RuntimeError("node_base output layer hidden dimension mismatch")

    encoder = FrozenNodeBaseEncoder(input_dim, hidden_dim, embedding_dim)
    encoder_state = {
        "0.weight": first_weight,
        "0.bias": _state_value(state, "node_base.0.bias"),
        "1.weight": _state_value(state, "node_base.1.weight"),
        "1.bias": _state_value(state, "node_base.1.bias"),
        "4.weight": output_weight,
        "4.bias": _state_value(state, output_prefix + ".bias"),
    }
    encoder.layers.load_state_dict(encoder_state, strict=True)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    return encoder, {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
    }


def resolve_device(requested: str) -> str:
    _require_torch()
    requested = str(requested or "auto").lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def encode_all_nodes(
    encoder: FrozenNodeBaseEncoder,
    features: "torch.Tensor",
    device: str,
    batch_size: int,
) -> Tuple["torch.Tensor", Dict[str, Any]]:
    _require_torch()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    encoder = encoder.to(device)
    output_dim = int(encoder.layers[4].out_features)
    embeddings = torch.empty((features.shape[0], output_dim), dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, int(features.shape[0]), batch_size):
            end = min(int(features.shape[0]), start + batch_size)
            batch = features[start:end].to(device, non_blocking=True)
            encoded = encoder(batch).detach().cpu().float()
            if not torch.isfinite(encoded).all():
                raise RuntimeError("Encoder produced NaN/Inf for rows %d:%d" % (start, end))
            embeddings[start:end].copy_(encoded)
            if start == 0 or end == int(features.shape[0]) or end % (batch_size * 50) == 0:
                print(json.dumps({"phase": "encode", "rows": end, "total": int(features.shape[0])}), flush=True)
    norms = embeddings.norm(dim=1)
    if not torch.isfinite(norms).all():
        raise RuntimeError("Embedding norms contain NaN/Inf")
    if torch.any(norms <= 0):
        raise RuntimeError("At least one exported embedding has zero norm")
    return embeddings.contiguous(), {
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "norm_mean": float(norms.mean().item()),
        "all_finite": True,
        "zero_norm_count": int((norms <= 0).sum().item()),
    }


def atomic_torch_save(payload: Any, destination: Path) -> None:
    _require_torch()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix="." + destination.name + ".", dir=str(destination.parent))
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_index(
    destination: Path,
    metadata: Sequence[Mapping[str, str]],
    embeddings_filename: str,
) -> Counter:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix="." + destination.name + ".", dir=str(destination.parent))
    os.close(handle)
    temp_path = Path(temp_name)
    type_counts: Counter = Counter()
    try:
        with temp_path.open("w", encoding="utf-8") as stream:
            stream.write("[")
            for index, record in enumerate(metadata):
                node_id = str(record.get("node_id") or "")
                sample_id = str(record.get("sample_id") or infer_sample_id(node_id))
                node_type = str(record.get("node_type") or infer_node_type(node_id) or "unknown")
                row = {
                    "index": index,
                    "node_id": node_id,
                    "node_type": node_type,
                    "sample_id": sample_id,
                    "embedding_ref": "%s#row=%d" % (embeddings_filename, index),
                }
                layer = str(record.get("layer") or "")
                raw_id = str(record.get("raw_id") or "")
                if layer:
                    row["layer"] = layer
                if raw_id:
                    row["raw_id"] = raw_id
                if index:
                    stream.write(",")
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                type_counts[node_type] += 1
            stream.write("]\n")
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return type_counts


def atomic_write_json(destination: Path, payload: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix="." + destination.name + ".", dir=str(destination.parent))
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def validate_checkpoint(
    checkpoint: Mapping[str, Any], allow_unverified: bool
) -> Dict[str, Any]:
    revision = str(checkpoint.get("objective_revision") or "")
    selected_by = str(checkpoint.get("selected_by") or "")
    evaluation_audit = checkpoint.get("evaluation_audit")
    if not isinstance(evaluation_audit, Mapping):
        evaluation_audit = {}
    validation_selected = (
        selected_by == "validation_mrr"
        or evaluation_audit.get("checkpoint_selection_split") == "validation"
    )
    objective_compatible = revision in ALLOWED_OBJECTIVE_REVISIONS
    if not allow_unverified:
        if not objective_compatible:
            raise RuntimeError(
                "Formal export requires an approved objective_revision=%s, got %r"
                % (sorted(ALLOWED_OBJECTIVE_REVISIONS), revision)
            )
        if not validation_selected:
            raise RuntimeError(
                "Formal export requires a Validation-selected checkpoint; use alignment_model_best.pt"
            )
    return {
        "objective_revision": revision,
        "objective_compatible": objective_compatible,
        "selected_by": selected_by,
        "validation_selected": validation_selected,
        "best_epoch": checkpoint.get("best_epoch"),
        "best_validation_mrr": checkpoint.get("best_validation_mrr"),
        "explicit_export_contract": checkpoint.get("node_embedding_export_contract"),
    }


def validate_stage3_metrics(
    metrics_path: Path, allow_unverified: bool
) -> Dict[str, Any]:
    if not metrics_path.is_file() or metrics_path.stat().st_size <= 0:
        if allow_unverified:
            return {
                "path": str(metrics_path),
                "available": False,
                "passed": False,
                "errors": ["missing_stage3_metrics"],
            }
        raise FileNotFoundError(
            "Formal export requires Stage3 alignment_metrics.json: %s" % metrics_path
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    graph = payload.get("dataset", {}).get("graph_split_audit", {})
    evaluation = payload.get("evaluation_audit", {})
    revision = payload.get("model", {}).get("objective_revision")
    errors = []
    if revision not in ALLOWED_OBJECTIVE_REVISIONS:
        errors.append("objective_revision")
    if graph.get("policy") not in ALLOWED_GRAPH_POLICIES:
        errors.append("train_graph_policy")
    if graph.get("passed") is not True:
        errors.append("graph_split_audit")
    if graph.get("positive_edge_leakage_count") != 0:
        errors.append("positive_edge_leakage")
    if graph.get("negative_node_leakage_count") != 0:
        errors.append("negative_node_leakage")
    if evaluation.get("passed") is not True:
        errors.append("evaluation_audit")
    if evaluation.get("evaluation_count") != 1:
        errors.append("test_evaluation_count")
    if evaluation.get("test_evaluations_during_training") != 0:
        errors.append("test_evaluated_during_training")
    if evaluation.get("checkpoint_selection_split") != "validation":
        errors.append("checkpoint_selection_split")
    if evaluation.get("test_labels_used_for_selection") is not False:
        errors.append("test_labels_used_for_selection")
    result = {
        "path": str(metrics_path.resolve()),
        "available": True,
        "passed": not errors,
        "errors": errors,
        "objective_revision": revision,
        "graph_policy": graph.get("policy"),
        "positive_edge_leakage_count": graph.get("positive_edge_leakage_count"),
        "negative_node_leakage_count": graph.get("negative_node_leakage_count"),
        "test_evaluation_count": evaluation.get("evaluation_count"),
        "test_evaluations_during_training": evaluation.get("test_evaluations_during_training"),
        "checkpoint_selection_split": evaluation.get("checkpoint_selection_split"),
        "test_labels_used_for_selection": evaluation.get("test_labels_used_for_selection"),
    }
    if errors and not allow_unverified:
        raise RuntimeError(
            "Stage3 metrics are not eligible for formal export: %s"
            % ", ".join(errors)
        )
    return result


def ensure_outputs_available(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: %s" % ", ".join(existing)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--stage3-metrics",
        default="",
        help="Stage3 or unified Stage5 metrics; defaults beside the checkpoint",
    )
    parser.add_argument("--node-features", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--embeddings-name", default="node_embeddings.pt")
    parser.add_argument("--index-name", default="node_embedding_index.json")
    parser.add_argument("--manifest-name", default="node_embedding_export_manifest.json")
    parser.add_argument(
        "--legacy-index-name",
        default="",
        help="Optional compatibility copy, e.g. node_index.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="0 exports all nodes; any positive value creates a non-formal smoke artifact",
    )
    parser.add_argument(
        "--input-hash-policy",
        choices=("sha256", "metadata"),
        default="sha256",
    )
    parser.add_argument("--require-node-metadata-coverage", action="store_true")
    parser.add_argument(
        "--require-exact-node-set",
        action="store_true",
        help="Fail if nodes JSONL has missing, extra, or duplicate node IDs",
    )
    parser.add_argument("--allow-unverified-checkpoint", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _require_torch()
    checkpoint_path = Path(args.checkpoint).resolve()
    metrics_path = (
        Path(args.stage3_metrics).resolve()
        if args.stage3_metrics
        else checkpoint_path.parent / "alignment_metrics.json"
    )
    features_path = Path(args.node_features).resolve()
    nodes_path = Path(args.nodes).resolve()
    for path in (checkpoint_path, features_path, nodes_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError("Missing or empty input: %s" % path)

    out_dir = Path(args.out_dir).resolve()
    embeddings_path = out_dir / args.embeddings_name
    index_path = out_dir / args.index_name
    manifest_path = out_dir / args.manifest_name
    legacy_index_path = out_dir / args.legacy_index_name if args.legacy_index_name else None
    output_paths = [embeddings_path, index_path, manifest_path]
    if legacy_index_path is not None:
        output_paths.append(legacy_index_path)
    ensure_outputs_available(output_paths, args.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, node_ids, feature_meta = load_feature_payload(features_path)
    full_feature_rows = int(features.shape[0])
    max_nodes = max(0, int(args.max_nodes))
    if max_nodes:
        features = features[:max_nodes].contiguous()
        if node_ids:
            node_ids = node_ids[:max_nodes]

    if node_ids:
        metadata, node_audit = scan_nodes(
            nodes_path,
            node_ids,
            stop_when_complete=bool(max_nodes),
        )
    else:
        node_ids, metadata = load_nodes_in_file_order(nodes_path, int(features.shape[0]))
        node_audit = {
            "nodes_rows_scanned": len(node_ids),
            "matched_node_rows": len(node_ids),
            "missing_node_rows": 0,
            "extra_node_rows_seen": 0,
            "duplicate_node_rows": 0,
            "node_metadata_coverage": 1.0,
            "missing_node_examples": [],
            "schema_source_counts": {},
            "scan_stopped_when_complete": False,
        }

    if len(node_ids) != int(features.shape[0]):
        raise RuntimeError("Node ID / feature row mismatch after slicing")
    if args.require_node_metadata_coverage and node_audit["node_metadata_coverage"] != 1.0:
        raise RuntimeError(
            "Formal export requires 100%% node metadata coverage; got %.6f"
            % node_audit["node_metadata_coverage"]
        )
    if args.require_exact_node_set and (
        node_audit["missing_node_rows"] != 0
        or node_audit["extra_node_rows_seen"] != 0
        or node_audit["duplicate_node_rows"] != 0
    ):
        raise RuntimeError(
            "Formal export requires the feature IDs and USG node set to be exact: "
            "missing=%d extra=%d duplicate=%d"
            % (
                node_audit["missing_node_rows"],
                node_audit["extra_node_rows_seen"],
                node_audit["duplicate_node_rows"],
            )
        )
    if any(not node_id for node_id in node_ids):
        raise RuntimeError("At least one feature row has an empty node ID")
    if len(set(node_ids)) != len(node_ids):
        raise RuntimeError("Feature rows contain duplicate node IDs")

    checkpoint_payload = load_serialized(checkpoint_path)
    state, checkpoint = normalize_state_dict(checkpoint_payload)
    checkpoint_audit = validate_checkpoint(
        checkpoint, args.allow_unverified_checkpoint
    )
    export_policy = EXPORT_POLICIES.get(
        checkpoint_audit["objective_revision"],
        "unverified_node_base_encoder",
    )
    stage3_metrics_audit = validate_stage3_metrics(
        metrics_path, args.allow_unverified_checkpoint
    )
    encoder, architecture = build_frozen_encoder(state, int(features.shape[1]))
    frozen = all(not parameter.requires_grad for parameter in encoder.parameters())
    if not frozen:
        raise RuntimeError("Node encoder was not frozen before export")

    device = resolve_device(args.device)
    embeddings, tensor_audit = encode_all_nodes(
        encoder, features, device, int(args.batch_size)
    )

    type_counts = atomic_write_index(index_path, metadata, embeddings_path.name)
    index_sha256 = sha256_file(index_path)

    input_fingerprints = {
        "checkpoint": input_fingerprint(checkpoint_path, args.input_hash_policy),
        "stage3_metrics": (
            input_fingerprint(metrics_path, args.input_hash_policy)
            if metrics_path.is_file() else None
        ),
        "node_features": input_fingerprint(features_path, args.input_hash_policy),
        "nodes": input_fingerprint(nodes_path, args.input_hash_policy),
    }
    formal_ready = (
        max_nodes == 0
        and args.input_hash_policy == "sha256"
        and checkpoint_audit["objective_compatible"]
        and checkpoint_audit["validation_selected"]
        and stage3_metrics_audit["passed"]
        and node_audit["node_metadata_coverage"] == 1.0
        and node_audit["extra_node_rows_seen"] == 0
        and node_audit["duplicate_node_rows"] == 0
        and frozen
        and tensor_audit["all_finite"]
        and tensor_audit["zero_norm_count"] == 0
        and len(node_ids) == full_feature_rows
    )

    tensor_payload = {
        "schema": SCHEMA,
        "embeddings": embeddings,
        # Compatibility alias. torch.save preserves the shared storage rather
        # than materialising a second tensor.
        "node_embeddings": embeddings,
        "num_nodes": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype).replace("torch.", ""),
        "normalized": True,
        "export_policy": export_policy,
        "index_filename": index_path.name,
        "index_sha256": index_sha256,
        "checkpoint_sha256": input_fingerprints["checkpoint"]["sha256"],
        "objective_revision": checkpoint_audit["objective_revision"],
        "formal_ready": formal_ready,
    }
    atomic_torch_save(tensor_payload, embeddings_path)
    embeddings_sha256 = sha256_file(embeddings_path)

    legacy_sha256 = None
    if legacy_index_path is not None:
        shutil.copyfile(index_path, legacy_index_path)
        legacy_sha256 = sha256_file(legacy_index_path)

    manifest = {
        "schema": SCHEMA,
        "formal_ready": formal_ready,
        "export_policy": export_policy,
        "explanation": (
            "Frozen node_base encoder exported for every feature row. "
            + (
                "This unified Stage5 node_base received evidence, constraint-margin, "
                "and Train-only structure-reconstruction gradients."
                if checkpoint_audit["objective_revision"]
                == "stage5_unified_three_loss_train_graph_v6_5"
                else "This is the audited Stage3 Train-graph baseline encoder."
            )
        ),
        "outputs": {
            "embeddings": {
                "path": str(embeddings_path),
                "sha256": embeddings_sha256,
                "size": embeddings_path.stat().st_size,
            },
            "index": {
                "path": str(index_path),
                "sha256": index_sha256,
                "size": index_path.stat().st_size,
            },
            "legacy_index": (
                {
                    "path": str(legacy_index_path),
                    "sha256": legacy_sha256,
                    "size": legacy_index_path.stat().st_size,
                }
                if legacy_index_path is not None else None
            ),
        },
        "inputs": input_fingerprints,
        "input_hash_policy": args.input_hash_policy,
        "checkpoint": checkpoint_audit,
        "stage3_metrics": stage3_metrics_audit,
        "architecture": architecture,
        "feature_payload": feature_meta,
        "nodes": node_audit,
        "tensor": {
            "rows": int(embeddings.shape[0]),
            "dimension": int(embeddings.shape[1]),
            "dtype": "float32",
            "normalized": True,
            **tensor_audit,
        },
        "index": {
            "rows": len(metadata),
            "unique_node_ids": len(set(node_ids)),
            "node_type_counts": dict(type_counts),
            "order_contract": "index[i].node_id is the source of embeddings[i]",
            "embedding_ref_contract": "%s#row=<zero-based-index>" % embeddings_path.name,
        },
        "export_run": {
            "device": device,
            "batch_size": int(args.batch_size),
            "max_nodes": max_nodes,
            "full_feature_rows": full_feature_rows,
            "model_parameters_frozen": frozen,
            "test_labels_read": False,
            "source_nodes_rewritten": False,
        },
        "checks": {
            "row_counts_match": len(metadata) == int(embeddings.shape[0]),
            "node_ids_unique": len(set(node_ids)) == len(node_ids),
            "all_embeddings_finite": tensor_audit["all_finite"],
            "zero_norm_count_is_zero": tensor_audit["zero_norm_count"] == 0,
            "validation_selected_checkpoint": checkpoint_audit["validation_selected"],
            "stage3_split_and_test_audit_passed": stage3_metrics_audit["passed"],
            "train_graph_firewall_objective": checkpoint_audit["objective_compatible"],
            "node_metadata_complete": node_audit["node_metadata_coverage"] == 1.0,
            "feature_and_usg_node_sets_exact": (
                node_audit["missing_node_rows"] == 0
                and node_audit["extra_node_rows_seen"] == 0
                and node_audit["duplicate_node_rows"] == 0
            ),
            "model_frozen": frozen,
            "no_test_labels_used": True,
        },
    }
    atomic_write_json(manifest_path, manifest)

    print(json.dumps({
        "schema": SCHEMA,
        "formal_ready": formal_ready,
        "embeddings": str(embeddings_path),
        "index": str(index_path),
        "manifest": str(manifest_path),
        "rows": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "node_metadata_coverage": node_audit["node_metadata_coverage"],
        "checkpoint_validation_selected": checkpoint_audit["validation_selected"],
    }, ensure_ascii=False, indent=2))
    if not formal_ready and not args.allow_unverified_checkpoint and max_nodes == 0:
        raise RuntimeError(
            "Export completed but is not formal-ready; inspect %s" % manifest_path
        )


if __name__ == "__main__":
    main()
