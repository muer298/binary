#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strictly verify a v6.3 node embedding export before 2.2.2 consumes it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

try:
    import torch
except ImportError:  # pragma: no cover - executed on the GPU server
    torch = None


EXPECTED_SCHEMA = "task2_node_embedding_export_v6_3"
APPROVED_OBJECTIVE_POLICIES = {
    "multi_positive_target_attention_v2_train_graph_firewall_v3": (
        "stage3_node_base_train_graph_encoder_v1"
    ),
    "stage5_unified_three_loss_train_graph_v6_5": (
        "stage5_unified_node_base_three_loss_encoder_v6_5"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_tensor_payload(path: Path) -> tuple[Mapping[str, Any], "torch.Tensor"]:
    if torch is None:
        raise RuntimeError("PyTorch is required for embedding verification")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        return {}, payload
    if not isinstance(payload, Mapping):
        raise RuntimeError("Embedding payload must be a Tensor or dictionary")
    for key in ("embeddings", "node_embeddings", "features", "vectors", "tensor"):
        value = payload.get(key)
        if torch.is_tensor(value):
            return payload, value
    raise RuntimeError("Embedding payload has no tensor field")


def check(condition: bool, name: str, checks: Dict[str, bool], errors: List[str]) -> None:
    checks[name] = bool(condition)
    if not condition:
        errors.append(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-formal", action="store_true")
    parser.add_argument("--norm-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    embeddings_path = Path(args.embeddings).resolve()
    index_path = Path(args.index).resolve()
    manifest_path = Path(args.manifest).resolve()
    report_path = Path(args.report).resolve()
    for path in (embeddings_path, index_path, manifest_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError("Missing or empty export artifact: %s" % path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, list):
        raise RuntimeError("node_embedding_index.json must be a JSON list")
    payload, embeddings = load_tensor_payload(embeddings_path)
    embeddings = embeddings.detach().cpu().float()

    checks: Dict[str, bool] = {}
    errors: List[str] = []
    checkpoint = manifest.get("checkpoint", {})
    objective_revision = checkpoint.get("objective_revision")
    expected_policy = APPROVED_OBJECTIVE_POLICIES.get(objective_revision)
    check(manifest.get("schema") == EXPECTED_SCHEMA, "manifest_schema", checks, errors)
    check(payload.get("schema") == EXPECTED_SCHEMA, "tensor_schema", checks, errors)
    check(expected_policy is not None, "objective_revision_approved", checks, errors)
    check(manifest.get("export_policy") == expected_policy, "manifest_export_policy", checks, errors)
    check(payload.get("export_policy") == expected_policy, "tensor_export_policy", checks, errors)
    check(embeddings.ndim == 2, "tensor_rank_2", checks, errors)

    row_count = int(embeddings.shape[0]) if embeddings.ndim == 2 else -1
    dimension = int(embeddings.shape[1]) if embeddings.ndim == 2 else -1
    check(row_count > 0 and dimension > 0, "nonempty_tensor", checks, errors)
    check(len(index) == row_count, "index_tensor_row_count", checks, errors)
    check(payload.get("num_nodes") == row_count, "payload_num_nodes", checks, errors)
    check(payload.get("embedding_dim") == dimension, "payload_embedding_dim", checks, errors)
    check(manifest.get("tensor", {}).get("rows") == row_count, "manifest_tensor_rows", checks, errors)
    check(manifest.get("index", {}).get("rows") == len(index), "manifest_index_rows", checks, errors)

    finite = bool(torch.isfinite(embeddings).all().item()) if embeddings.numel() else False
    check(finite, "all_embeddings_finite", checks, errors)
    norms = embeddings.norm(dim=1) if embeddings.ndim == 2 and embeddings.numel() else torch.empty(0)
    zero_norm_count = int((norms <= 0).sum().item()) if norms.numel() else row_count
    max_norm_error = float(torch.abs(norms - 1.0).max().item()) if norms.numel() else float("inf")
    check(zero_norm_count == 0, "zero_norm_count_is_zero", checks, errors)
    check(max_norm_error <= args.norm_tolerance, "l2_normalized", checks, errors)

    node_ids = set()
    index_valid = True
    missing_identity_examples = []
    for expected_index, row in enumerate(index):
        if not isinstance(row, Mapping):
            index_valid = False
            continue
        node_id = str(row.get("node_id") or row.get("id") or "")
        if (
            row.get("index") != expected_index
            or not node_id
            or not str(row.get("node_type") or "")
            or not str(row.get("sample_id") or "")
            or row.get("embedding_ref") != "%s#row=%d" % (embeddings_path.name, expected_index)
        ):
            index_valid = False
            if len(missing_identity_examples) < 20:
                missing_identity_examples.append(expected_index)
        if node_id in node_ids:
            index_valid = False
        node_ids.add(node_id)
    check(index_valid, "index_contract", checks, errors)
    check(len(node_ids) == len(index), "unique_node_ids", checks, errors)

    embeddings_sha256 = sha256_file(embeddings_path)
    index_sha256 = sha256_file(index_path)
    check(
        manifest.get("outputs", {}).get("embeddings", {}).get("sha256") == embeddings_sha256,
        "embeddings_sha256",
        checks,
        errors,
    )
    check(
        manifest.get("outputs", {}).get("index", {}).get("sha256") == index_sha256,
        "index_sha256",
        checks,
        errors,
    )
    check(payload.get("index_sha256") == index_sha256, "payload_index_sha256", checks, errors)
    check(payload.get("normalized") is True, "payload_normalized", checks, errors)
    check(manifest.get("export_run", {}).get("model_parameters_frozen") is True, "model_frozen", checks, errors)
    check(manifest.get("export_run", {}).get("test_labels_read") is False, "no_test_labels_read", checks, errors)
    check(manifest.get("export_run", {}).get("source_nodes_rewritten") is False, "source_nodes_untouched", checks, errors)

    check(objective_revision in APPROVED_OBJECTIVE_POLICIES, "objective_revision", checks, errors)
    check(checkpoint.get("validation_selected") is True, "validation_selected_checkpoint", checks, errors)
    stage3_metrics = manifest.get("stage3_metrics", {})
    check(stage3_metrics.get("passed") is True, "stage3_metrics_passed", checks, errors)
    check(stage3_metrics.get("positive_edge_leakage_count") == 0, "stage3_positive_edge_leakage_zero", checks, errors)
    check(stage3_metrics.get("negative_node_leakage_count") == 0, "stage3_negative_node_leakage_zero", checks, errors)
    check(stage3_metrics.get("test_evaluation_count") == 1, "stage3_test_evaluated_once", checks, errors)
    check(stage3_metrics.get("test_evaluations_during_training") == 0, "stage3_no_test_during_training", checks, errors)

    if args.require_formal:
        check(manifest.get("formal_ready") is True, "formal_ready_manifest", checks, errors)
        check(payload.get("formal_ready") is True, "formal_ready_tensor", checks, errors)
        check(manifest.get("input_hash_policy") == "sha256", "formal_sha256_inputs", checks, errors)
        check(manifest.get("export_run", {}).get("max_nodes") == 0, "formal_full_node_export", checks, errors)
        check(
            manifest.get("nodes", {}).get("node_metadata_coverage") == 1.0,
            "formal_node_metadata_coverage",
            checks,
            errors,
        )
        check(
            manifest.get("nodes", {}).get("missing_node_rows") == 0
            and manifest.get("nodes", {}).get("extra_node_rows_seen") == 0
            and manifest.get("nodes", {}).get("duplicate_node_rows") == 0,
            "formal_exact_node_set",
            checks,
            errors,
        )
        for input_name in ("checkpoint", "stage3_metrics", "node_features", "nodes"):
            fingerprint = manifest.get("inputs", {}).get(input_name, {})
            if not isinstance(fingerprint, Mapping):
                fingerprint = {}
            check(
                isinstance(fingerprint.get("sha256"), str)
                and len(fingerprint.get("sha256")) == 64,
                "formal_input_sha256_" + input_name,
                checks,
                errors,
            )

    report = {
        "schema": "task2_node_embedding_export_verification_v6_3",
        "passed": not errors,
        "require_formal": bool(args.require_formal),
        "artifacts": {
            "embeddings": str(embeddings_path),
            "index": str(index_path),
            "manifest": str(manifest_path),
        },
        "tensor": {
            "rows": row_count,
            "dimension": dimension,
            "finite": finite,
            "zero_norm_count": zero_norm_count,
            "max_norm_error": max_norm_error,
        },
        "index": {
            "rows": len(index),
            "unique_node_ids": len(node_ids),
            "contract_error_examples": missing_identity_examples,
        },
        "checks": checks,
        "errors": errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(
            "VERIFY NODE EMBEDDING EXPORT: FAILED (%s)" % ", ".join(errors)
        )
    print("VERIFY NODE EMBEDDING EXPORT: PASSED")


if __name__ == "__main__":
    main()
