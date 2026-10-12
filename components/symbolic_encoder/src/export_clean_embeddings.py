#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from io_utils import dump
from train_clean_alignment import Stage5FullModel


def load_torch(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = load_torch(args.features)
    checkpoint = load_torch(args.checkpoint)
    schema = json.loads(Path(args.schema).read_text())
    vocab = json.loads(Path(args.vocab).read_text())
    features = payload["features"].float()
    node_ids = list(payload["node_ids"])
    if features.shape[1] != schema["feature_dim"] or len(vocab["features"]) != features.shape[1]:
        raise RuntimeError("feature/schema/vocab dimension mismatch")
    state = checkpoint["model_state_dict"]
    config = checkpoint["args"]
    rel_count = int(state["rel_emb.weight"].shape[0])
    model = Stage5FullModel(
        input_dim=int(features.shape[1]),
        hidden_dim=int(config["hidden_dim"]),
        embed_dim=int(config["embed_dim"]),
        etype_dim=int(config["etype_dim"]),
        rel_count=rel_count,
        rel_dim=int(config["rel_dim"]),
        nhead=int(config["nhead"]),
        transformer_layers=int(config["transformer_layers"]),
        gnn_layers=int(config["gnn_layers"]),
        dropout=float(config["dropout"]),
        attention_hidden_dim=int(config["attention_hidden_dim"]),
        attention_dropout=float(config["attention_dropout"]),
        attention_strength_scale=float(config["attention_strength_scale"]),
        attention_temperature=float(config["attention_temperature"]),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Real reload-forward before bulk export.
    with torch.no_grad():
        smoke = model.enc_node(features[: min(32, len(features))].to(device))
    if smoke.shape[1] != 64 or not torch.isfinite(smoke).all():
        raise RuntimeError("checkpoint reload-forward failed")

    output_chunks = []
    with torch.no_grad():
        for start in range(0, len(features), args.batch_size):
            output_chunks.append(model.enc_node(features[start:start + args.batch_size].to(device)).cpu())
    embeddings = torch.cat(output_chunks, dim=0) if output_chunks else torch.empty((0, 64))
    if embeddings.shape != (len(node_ids), 64):
        raise RuntimeError("unexpected clean embedding shape")
    nan_count = int(torch.isnan(embeddings).sum())
    inf_count = int(torch.isinf(embeddings).sum())
    if nan_count or inf_count:
        raise RuntimeError("NaN/Inf in clean embeddings")
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    torch.save({
        "embeddings": embeddings,
        "node_embeddings": embeddings,
        "node_ids": node_ids,
        "node_index": index,
        "embedding_dim": 64,
        "feature_schema": schema,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "scope": "coverage_export_without_outer_performance_evaluation",
    }, args.output)
    dump(args.index_output, index)
    report = {
        "schema": "seraph_clean_hsym_embedding_export_v0_2_8",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "reload_passed": True,
        "real_forward_passed": True,
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "embedding_dim": 64,
        "node_count": len(node_ids),
        "index_count": len(index),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "duplicate_node_id_count": len(node_ids) - len(index),
        "outer_performance_metrics_computed": False,
        "passed": len(node_ids) == len(index) and not nan_count and not inf_count,
    }
    dump(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
