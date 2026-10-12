#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify cryptographic lineage from formal export through 2.2.1.6 output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
except ImportError:
    torch = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tensor(path: Path):
    if torch is None:
        raise RuntimeError("PyTorch is required")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, Mapping):
        for key in ("embeddings", "node_embeddings", "tensor", "x", "data"):
            value = payload.get(key)
            if torch.is_tensor(value):
                return value
    raise RuntimeError("No embedding tensor in %s" % path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--resolution-manifest", required=True)
    parser.add_argument("--base-export-manifest", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    paths = {name: Path(value).resolve() for name, value in (
        ("embeddings", args.embeddings), ("index", args.index),
        ("resolution_manifest", args.resolution_manifest),
        ("base_export_manifest", args.base_export_manifest),
    )}
    for path in paths.values():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    resolution = json.loads(paths["resolution_manifest"].read_text(encoding="utf-8"))
    base = json.loads(paths["base_export_manifest"].read_text(encoding="utf-8"))
    index = json.loads(paths["index"].read_text(encoding="utf-8"))
    tensor = load_tensor(paths["embeddings"]).detach().cpu().float()
    checks = {}
    checks["base_export_formal"] = base.get("formal_ready") is True
    checks["base_export_no_test"] = base.get("export_run", {}).get("test_labels_read") is False
    checks["resolution_formal"] = resolution.get("formal_ready") is True
    checks["resolution_schema"] = resolution.get("schema") == "task2_mask_project_blend_batch_v6_5"
    source = resolution.get("source_fingerprints") or {}
    checks["base_embedding_lineage"] = source.get("embeddings_sha256") == base.get("outputs", {}).get("embeddings", {}).get("sha256")
    checks["base_index_lineage"] = source.get("index_sha256") == base.get("outputs", {}).get("index", {}).get("sha256")
    outputs = resolution.get("outputs") or {}
    checks["resolved_embedding_hash"] = outputs.get(paths["embeddings"].name, {}).get("sha256") == sha256(paths["embeddings"])
    checks["resolved_index_hash"] = outputs.get(paths["index"].name, {}).get("sha256") == sha256(paths["index"])
    checks["rank2_finite_tensor"] = tensor.ndim == 2 and tensor.numel() > 0 and bool(torch.isfinite(tensor).all())
    checks["index_row_count"] = isinstance(index, list) and len(index) == int(tensor.shape[0])
    checks["unique_node_ids"] = isinstance(index, list) and len({str(row.get("node_id") or "") for row in index}) == len(index)
    checks["resolved_sidecar_not_source_rewrite"] = all(
        str(row.get("embedding_ref") or "").startswith(paths["embeddings"].name + "#row=")
        for row in index
    )
    errors = [name for name, value in checks.items() if not value]
    report = {
        "schema": "task2_resolved_representation_chain_v6_6",
        "passed": not errors,
        "checks": checks,
        "errors": errors,
        "rows": int(tensor.shape[0]) if tensor.ndim == 2 else 0,
        "dimension": int(tensor.shape[1]) if tensor.ndim == 2 else 0,
        "lineage": {
            "base_export_manifest": str(paths["base_export_manifest"]),
            "resolution_manifest": str(paths["resolution_manifest"]),
            "resolved_embeddings": str(paths["embeddings"]),
            "resolved_index": str(paths["index"]),
        },
        "formal_222_input_ready": not errors,
    }
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
