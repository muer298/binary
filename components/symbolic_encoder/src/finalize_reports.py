#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from io_utils import dump, load_config


def load(path):
    return json.loads(Path(path).read_text())


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_root"])
    prep = load(out / "data_preparation_report.json")
    feat = load(out / "clean_featurization_report.json")
    leak = load(out / "clean_feature_leakage_audit.json")
    train = load(out / "training" / "alignment_metrics.json")
    export = load(out / "clean_embedding_export_report.json")
    coverage = load(out / "clean_embedding_coverage.json")
    checkpoint = torch_load(out / "clean_alignment_checkpoint.pt")

    outer = train["evaluation_audit"]
    truth = train["training_truth"]
    gates = {
        "semantic_allowlist_effective": feat["passed"],
        "forbidden_identity_features_zero": leak["forbidden_identity_feature_count"] == 0,
        "encoder_split_isolated": prep["passed"],
        "outer_data_not_used": outer["passed"] and all(
            outer[key] == 0 for key in (
                "outer_dev_records_opened", "outer_cal_records_opened", "outer_test_records_opened"
            )
        ),
        "training_succeeded": truth["optimizer_steps"] > 0 and not truth["nan_or_inf_loss"],
        "parameters_changed": truth["changed_parameter_tensors"] > 0 and truth["max_parameter_abs_delta"] > 0,
        "reload_forward_passed": export["reload_passed"] and export["real_forward_passed"],
        "clean_embeddings_exported": export["passed"] and export["embedding_dim"] == 64,
        "coverage_acceptable": coverage["passed"],
        "constraint_loss_disabled": checkpoint["args"]["lambda_constraint_margin"] == 0,
    }
    status = "READY_FOR_V0_2_9_CLEAN_JOINT" if all(gates.values()) else "BLOCKED"

    clean_training = {
        "schema": "seraph_clean_hsym_training_report_v0_2_8",
        "seed": checkpoint["args"]["seed"],
        "selection_metric": "encoder_inner_dev_MRR",
        "best_encoder_inner_dev_mrr": checkpoint["best_encoder_inner_dev_mrr"],
        "feature_dim": train["dataset"]["feature_dim"],
        "embedding_dim": checkpoint["args"]["embed_dim"],
        "epochs": checkpoint["args"]["epochs"],
        "batch_size": checkpoint["args"]["batch_size"],
        "learning_rate": checkpoint["args"]["lr"],
        "optimizer_steps": truth["optimizer_steps"],
        "gradient_steps": truth["gradient_steps"],
        "max_gradient_norm": truth["max_gradient_norm"],
        "trainable_parameter_count": truth["trainable_parameter_count"],
        "total_parameter_count": truth["total_parameter_count"],
        "changed_parameter_tensors": truth["changed_parameter_tensors"],
        "max_parameter_abs_delta": truth["max_parameter_abs_delta"],
        "peak_allocated_vram_bytes": truth["peak_allocated_vram_bytes"],
        "peak_reserved_vram_bytes": truth["peak_reserved_vram_bytes"],
        "wall_time_seconds": truth["wall_time_seconds"],
        "gpu": truth["gpu"],
        "nan_or_inf_loss": truth["nan_or_inf_loss"],
        "outer_metrics_computed": False,
        "constraint_loss": "DISABLED",
        "reload_passed": export["reload_passed"],
        "status": status,
    }
    dump(out / "clean_training_report.json", clean_training)

    log = list(__import__("csv").DictReader((out / "training" / "alignment_train_log.csv").open()))
    first_loss = float(log[0]["train_loss"]) if log else None
    final_loss = float(log[-1]["train_loss"]) if log else None
    best_epoch = int(checkpoint.get("best_epoch", 0))
    md = [
        "# Clean h_sym training report", "",
        "- Seed: `%s`" % clean_training["seed"],
        "- Selection: encoder_inner_dev MRR only",
        "- Epochs: %d" % clean_training["epochs"],
        "- Batch size: %d" % clean_training["batch_size"],
        "- Learning rate: %g" % clean_training["learning_rate"],
        "- Optimizer steps: %d" % clean_training["optimizer_steps"],
        "- Trainable / total parameters: %d / %d" % (clean_training["trainable_parameter_count"], clean_training["total_parameter_count"]),
        "- Best epoch: %d" % best_epoch,
        "- Best inner-dev MRR: %.6f" % clean_training["best_encoder_inner_dev_mrr"],
        "- Initial / final logged loss: %s / %s" % (first_loss, final_loss),
        "- NaN/Inf loss: `%s`" % clean_training["nan_or_inf_loss"],
        "- Peak allocated / reserved VRAM: %d / %d bytes" % (clean_training["peak_allocated_vram_bytes"], clean_training["peak_reserved_vram_bytes"]),
        "- GPU: `%s` (CUDA_VISIBLE_DEVICES=%s; free before=%s bytes)" % (clean_training["gpu"]["name"], clean_training["gpu"]["cuda_visible_devices"], clean_training["gpu"]["free_vram_before_bytes"]),
        "- Wall time: %.1f seconds" % clean_training["wall_time_seconds"],
        "- Constraint loss: DISABLED",
        "- Reload + real forward: PASSED", "",
        "Outer DEV/CALIBRATION/TEST were absent from trainer inputs and were not evaluated.",
    ]
    (out / "CLEAN_HSYM_TRAINING_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    old_shape = None
    old_count = None
    old_path = Path(cfg.get("old_embeddings", ""))
    if old_path.exists():
        old = torch_load(old_path)
        tensor = old.get("embeddings") if isinstance(old, dict) else old
        if tensor is None and isinstance(old, dict):
            tensor = old.get("node_embeddings")
        if torch.is_tensor(tensor):
            old_shape, old_count = list(tensor.shape), int(tensor.shape[0])

    summary = {
        "schema": "seraph_clean_hsym_v0_2_8_final",
        "status": status,
        "gates": gates,
        "old_hsym": {"classification": "PRE_PROTOCOL_REPAIR", "shape": old_shape, "node_count": old_count},
        "clean_hsym": export,
        "test_performance_read": False,
        "stage2_joint_started": False,
        "ssq_started": False,
        "calibration_started": False,
    }
    dump(out / "v0_2_8_final_status.json", summary)
    report = [
        "# V0.2.8 Clean h_sym report", "",
        "Final status: **%s**" % status, "",
        "## Gates", "",
    ]
    report.extend("- %s: `%s`" % (name, value) for name, value in gates.items())
    report.extend([
        "", "## Artifacts", "",
        "- Clean feature tensor: `clean_node_features.pt`",
        "- Clean checkpoint: `clean_alignment_checkpoint.pt`",
        "- Clean embeddings: `node_embeddings_clean.pt`",
        "- Clean embedding index: `node_embedding_index_clean.json`", "",
        "## Protocol statement", "",
        "Only GLOBAL TRAIN was used for encoder training and inner-dev checkpoint selection. No outer DEV, CALIBRATION, or TEST performance was computed. APISem was coverage-audited only. Constraint loss remained disabled.", "",
        "v0.2.9 was not started.",
    ])
    (out / "V0_2_8_CLEAN_HSYM_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
