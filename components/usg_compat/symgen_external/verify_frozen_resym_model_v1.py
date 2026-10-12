#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 ReSym checkpoint 与推理策略是否可作为冻结外部验证合同。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping


REQUIRED = (
    "alignment_model_best.pt",
    "alignment_metrics.json",
    "feature_vocab.json",
    "confirmation_policy.json",
    "layout_reranker_config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()
    report_path = Path(args.report).resolve()
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit("ERROR: missing frozen model files: %s" % ", ".join(missing))
    try:
        import torch
    except ImportError as error:
        raise SystemExit("ERROR: PyTorch is required") from error

    checkpoint_path = model_dir / "alignment_model_best.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise SystemExit("ERROR: checkpoint is not a mapping")
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise SystemExit("ERROR: checkpoint has no model state")
    checkpoint_args = checkpoint.get("args") if isinstance(checkpoint.get("args"), Mapping) else {}

    metrics = json.loads((model_dir / "alignment_metrics.json").read_text(encoding="utf-8"))
    vocab = json.loads((model_dir / "feature_vocab.json").read_text(encoding="utf-8"))
    policy = json.loads((model_dir / "confirmation_policy.json").read_text(encoding="utf-8"))
    layout = json.loads((model_dir / "layout_reranker_config.json").read_text(encoding="utf-8"))
    if isinstance(metrics.get("evaluation_audit"), Mapping):
        evaluation = metrics["evaluation_audit"]
    elif isinstance(metrics.get("evaluation"), Mapping):
        evaluation = metrics["evaluation"]
    else:
        evaluation = {}
    raw_selection = (
        evaluation.get("checkpoint_selection_split")
        or metrics.get("checkpoint_selection_split")
        or checkpoint.get("selected_by")
    )
    selection = (
        "validation" if raw_selection in ("validation", "validation_mrr")
        else raw_selection
    )
    test_evaluations = evaluation.get(
        "test_evaluations_during_training",
        metrics.get("test_evaluations_during_training"),
    )
    checkpoint_dim = checkpoint_args.get("feature_dim") or checkpoint_args.get("input_dim")
    vocab_dim = vocab.get("feature_dim") or vocab.get("total_dim")
    errors = []
    if selection != "validation":
        errors.append("checkpoint_not_validation_selected")
    if test_evaluations != 0:
        errors.append("test_used_during_checkpoint_training_or_selection")
    if evaluation.get("passed") is not True:
        errors.append("evaluation_audit_not_passed")
    if evaluation.get("evaluation_count") != 1:
        errors.append("test_not_single_use")
    if evaluation.get("test_labels_used_for_selection") is not False:
        errors.append("test_labels_used_for_selection")
    if evaluation.get("checkpoint_frozen_before_test") is not True and evaluation.get(
        "model_parameters_frozen_before_test"
    ) is not True:
        errors.append("checkpoint_not_frozen_before_test")
    if policy.get("implementation_revision") != "v6_7_1_223_policy_handoff":
        errors.append("confirmation_policy_revision_mismatch")
    layout_policy = ((policy.get("policies") or {}).get("LayoutSem") or {})
    if layout_policy.get("soft_confirmation_enabled") is not False:
        errors.append("layout_confirmation_not_fail_closed")
    if layout.get("test_used_for_selection") is not False:
        errors.append("layout_config_used_test_for_selection")
    try:
        layout_module_weight = float(layout.get("recommended_module_weight", 0.25))
    except (TypeError, ValueError):
        layout_module_weight = -1.0
    if not 0.0 <= layout_module_weight <= 1.0:
        errors.append("layout_recommended_module_weight_invalid")
    if checkpoint_dim and vocab_dim and int(checkpoint_dim) != int(vocab_dim):
        errors.append("feature_dimension_mismatch")
    migration_path = model_dir / "migration_manifest.json"
    migration = (
        json.loads(migration_path.read_text(encoding="utf-8"))
        if migration_path.is_file() else {}
    )
    protocol = migration.get("protocol") if isinstance(migration.get("protocol"), Mapping) else {}
    if protocol.get("symgen_training_allowed") is True:
        errors.append("migration_contract_allows_symgen_training")

    files = {
        name: {
            "path": str(model_dir / name),
            "sha256": sha256_file(model_dir / name),
            "size_bytes": (model_dir / name).stat().st_size,
        }
        for name in REQUIRED
    }
    optional_head = model_dir / "module_head_checkpoint.pt"
    head_audit: dict[str, Any] = {"present": optional_head.is_file()}
    if optional_head.is_file():
        files[optional_head.name] = {
            "path": str(optional_head),
            "sha256": sha256_file(optional_head),
            "size_bytes": optional_head.stat().st_size,
        }
        try:
            head = json.loads(optional_head.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            head = {}
        head_modules = head.get("modules") if isinstance(head.get("modules"), Mapping) else {}
        head_audit.update({
            "schema": head.get("schema"),
            "modules": sorted(head_modules),
            "test_used_for_training_or_selection": head.get(
                "test_used_for_training_or_selection"
            ),
            "module_blend_selected_on_validation": head.get(
                "module_blend_selected_on_validation"
            ),
        })
        if head.get("schema") != "task2_223_specialised_heads_v6_7":
            errors.append("module_head_schema_mismatch")
        if head.get("test_used_for_training_or_selection") is not False:
            errors.append("module_head_used_test")
        if head.get("train_validation_query_overlap") != 0:
            errors.append("module_head_train_validation_overlap")
        if head.get("representation_encoder_frozen") is not True:
            errors.append("module_head_encoder_not_frozen")
        if head.get("module_blend_selected_on_validation") is not True:
            errors.append("module_head_blend_not_validation_selected")
        if set(head_modules) != {"APISem", "TypeSem", "LayoutSem", "InterfaceSem"}:
            errors.append("module_head_missing_module")
    report: dict[str, Any] = {
        "schema": "frozen_resym_external_contract_v1",
        "model_dir": str(model_dir),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "checkpoint_tensors": len(state),
        "checkpoint_selection_split": selection,
        "checkpoint_selection_raw": raw_selection,
        "test_evaluations_during_training": test_evaluations,
        "feature_dim_checkpoint": checkpoint_dim,
        "feature_dim_vocab": vocab_dim,
        "policy_revision": policy.get("implementation_revision"),
        "layout_test_used_for_selection": layout.get("test_used_for_selection"),
        "layout_recommended_module_weight": layout_module_weight,
        "frozen_direction_overrides": {"TypeSem->InterfaceSem": 0.0},
        "files": files,
        "optional_module_head": head_audit,
        "errors": errors,
        "passed": not errors,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)
    print("FROZEN RESYM MODEL CONTRACT: PASSED")


if __name__ == "__main__":
    main()
