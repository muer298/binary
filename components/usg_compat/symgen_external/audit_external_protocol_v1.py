#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终审计冻结模型、Gold 隔离、特征合同和推理结果的外部验证协议。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def load(path: str) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object: %s" % path)
    return value


def same_number(value: Any, expected: float, tolerance: float = 1e-12) -> bool:
    try:
        return abs(float(value) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def same_path(left: Any, right: Any) -> bool:
    if not left or not right:
        return False
    return Path(str(left)).resolve(strict=False) == Path(str(right)).resolve(strict=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-report", required=True)
    parser.add_argument("--raw-gold-report", required=True)
    parser.add_argument("--firewall-report", required=True)
    parser.add_argument("--model-report", required=True)
    parser.add_argument("--feature-report", required=True)
    parser.add_argument("--embedding-manifest", required=True)
    parser.add_argument("--reasoning-summary", required=True)
    parser.add_argument("--evaluation-222-pre-gold", required=True)
    parser.add_argument("--module-summary", required=True)
    parser.add_argument("--collaboration-summary", required=True)
    parser.add_argument("--evaluation-223-pre-gold", required=True)
    parser.add_argument("--mapped-gold-audit", required=True)
    parser.add_argument("--final-evaluation-223", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output = Path(args.report)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    reports = {
        name: load(getattr(args, name))
        for name in (
            "pair_report", "raw_gold_report", "firewall_report", "model_report",
            "feature_report", "embedding_manifest", "reasoning_summary",
            "evaluation_222_pre_gold", "module_summary", "collaboration_summary",
            "evaluation_223_pre_gold", "mapped_gold_audit", "final_evaluation_223",
        )
    }
    errors = []
    for name in ("pair_report", "raw_gold_report", "firewall_report", "model_report", "feature_report"):
        if reports[name].get("passed") is not True:
            errors.append(name + "_failed")
    embedding = reports["embedding_manifest"]
    if embedding.get("formal_ready") is not True:
        errors.append("embedding_not_formal_ready")
    checks = embedding.get("checks") if isinstance(embedding.get("checks"), Mapping) else {}
    if checks.get("model_frozen") is not True:
        errors.append("encoder_not_frozen")
    if checks.get("no_test_labels_used") is not True:
        errors.append("embedding_read_test_labels")
    reasoning = reports["reasoning_summary"]
    if isinstance(reasoning.get("neural_symbolic_closed_loop"), Mapping):
        closed_loop = reasoning["neural_symbolic_closed_loop"]
    elif isinstance(reasoning.get("closed_loop"), Mapping):
        closed_loop = reasoning["closed_loop"]
    else:
        closed_loop = {}
    if closed_loop.get("gold_used_during_inference") is not False:
        errors.append("gold_used_in_222_inference")
    pre222 = reports["evaluation_222_pre_gold"]
    if int(pre222.get("gold_queries") or 0) != 0:
        errors.append("gold_present_in_222_runner")
    frozen_222 = {
        "max_iterations": 10,
        "global_threshold": 0.85,
        "stable_rounds_required": 2,
        "min_global_delta": 0.000001,
        "max_state_change_rate_for_stable": 0.0001,
        "embedding_rescore_blend": 0.25,
        "score_feedback_blend": 0.35,
        "max_score_drop_per_round": 0.08,
        "max_score_rise_per_round": 0.08,
        "soft_switch_margin": 0.03,
        "soft_release_margin": 0.04,
        "soft_release_gap_margin": 0.03,
        "retrieval_interval": 1,
        "retrieval_top_k_per_type": 5,
        "retrieval_max_candidates_per_type": 6,
        "retrieval_min_probability": 0.55,
        "retrieval_replacement_margin": 0.02,
    }
    for key, expected in frozen_222.items():
        if not same_number(reasoning.get(key), expected):
            errors.append("frozen_222_parameter_changed:%s" % key)
    if reasoning.get("confirmed_states_irreversible") is not True:
        errors.append("222_confirmation_lock_changed")
    if reasoning.get("raw_monotonic_transaction_enabled") is not True:
        errors.append("222_transaction_policy_changed")
    if reasoning.get("iterative_retrieval_enabled") is not True:
        errors.append("222_retrieval_policy_changed")
    model = reports["model_report"]
    model_files = model.get("files") if isinstance(model.get("files"), Mapping) else {}
    for name, metadata in model_files.items():
        if not isinstance(metadata, Mapping):
            errors.append("frozen_artifact_metadata_invalid:%s" % name)
            continue
        artifact = Path(str(metadata.get("path") or ""))
        if not artifact.is_file():
            errors.append("frozen_artifact_missing_at_final_audit:%s" % name)
        elif sha256_file(artifact) != metadata.get("sha256"):
            errors.append("frozen_artifact_changed_after_precheck:%s" % name)
    embedding_inputs = (
        embedding.get("inputs") if isinstance(embedding.get("inputs"), Mapping)
        else {}
    )
    for input_name, model_name in (
        ("checkpoint", "alignment_model_best.pt"),
        ("stage3_metrics", "alignment_metrics.json"),
    ):
        used = embedding_inputs.get(input_name) or {}
        frozen = model_files.get(model_name) or {}
        if used.get("sha256") != frozen.get("sha256"):
            errors.append("embedding_did_not_use_frozen_%s" % input_name)
        if not same_path(used.get("path"), frozen.get("path")):
            errors.append("embedding_%s_path_changed" % input_name)
    policy_file = model_files.get("confirmation_policy.json") or {}
    if not same_path(reasoning.get("confirm_policy_input"), policy_file.get("path")):
        errors.append("222_confirmation_policy_not_frozen_resym_file")
    modules = reports["module_summary"]
    if modules.get("gold_used_during_inference") is not False:
        errors.append("gold_used_in_module_inference")
    if modules.get("test_used_during_inference") is not False:
        errors.append("test_used_in_module_inference")
    optional_head = model.get("optional_module_head") or {}
    if optional_head.get("present") is True and set(
        modules.get("learned_heads_loaded") or []
    ) != {"APISem", "TypeSem", "LayoutSem", "InterfaceSem"}:
        errors.append("frozen_module_head_not_fully_loaded")
    layout_file = model_files.get("layout_reranker_config.json") or {}

    # model_report.files[*].sha256 protects the original file bytes, while
    # four_module_summary records the effective hash after Layout config
    # normalisation.  Compare effective hash with effective hash here.
    frozen_layout_path = layout_file.get("path")
    frozen_layout_effective_sha256 = None
    try:
        package_root = Path(__file__).resolve().parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from layout_reranker_v6_7_2 import load_layout_config

        (
            _,
            frozen_layout_path,
            frozen_layout_effective_sha256,
        ) = load_layout_config(str(frozen_layout_path or ""))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        errors.append("frozen_layout_config_canonicalization_failed")

    if (
        modules.get("layout_reranker_config_sha256")
        != frozen_layout_effective_sha256
    ):
        errors.append("layout_config_not_frozen_resym_file")
    if not same_path(
        modules.get("layout_reranker_config_file"), frozen_layout_path
    ):
        errors.append("layout_config_path_changed")
    collaboration = reports["collaboration_summary"]
    direction = collaboration.get("configured_direction_weights") or {}
    if not same_number(direction.get("TypeSem->InterfaceSem"), 0.0):
        errors.append("frozen_direction_changed")
    module_weights = collaboration.get("module_fusion_weights") or {}
    for module in ("APISem", "TypeSem", "InterfaceSem"):
        if not same_number(module_weights.get(module), 0.25):
            errors.append("frozen_module_weight_changed:%s" % module)
    if not same_number(
        module_weights.get("LayoutSem"),
        model.get("layout_recommended_module_weight"),
    ):
        errors.append("frozen_layout_weight_changed")
    if not same_number(collaboration.get("global_threshold"), 0.85):
        errors.append("frozen_223_global_threshold_changed")
    if not same_number(collaboration.get("max_score_change_per_round"), 0.08):
        errors.append("frozen_223_score_change_changed")
    if not same_path(
        collaboration.get("confirmation_policy_file"), policy_file.get("path")
    ):
        errors.append("223_confirmation_policy_not_frozen_resym_file")
    pre223 = reports["evaluation_223_pre_gold"]
    gold_audit = pre223.get("gold_audit") if isinstance(pre223.get("gold_audit"), Mapping) else {}
    if int(gold_audit.get("unique_queries") or 0) != 0:
        errors.append("gold_present_in_223_runner")
    mapped = reports["mapped_gold_audit"]
    if mapped.get("strict_gold_ready") is not True:
        errors.append("mapped_gold_not_ready")
    final_eval = reports["final_evaluation_223"]
    final_gold = final_eval.get("gold_audit") if isinstance(final_eval.get("gold_audit"), Mapping) else {}
    if int(final_gold.get("unique_queries") or 0) <= 0:
        errors.append("final_evaluation_has_no_gold")

    report = {
        "schema": "symgen_external_validation_protocol_audit_v1",
        "protocol": {
            "training_dataset": "ReSym",
            "external_dataset": "SymGen",
            "symgen_training_or_tuning": False,
            "full_elf_role": "Gold only",
            "stripped_elf_role": "all inference",
            "gold_mapping_time": "after 2.2.2 and 2.2.3 inference",
            "frozen_direction_overrides": {"TypeSem->InterfaceSem": 0.0},
            "inference_hyperparameters_frozen": True,
        },
        "errors": errors,
        "passed": not errors,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)
    print("SYMGEN EXTERNAL VALIDATION PROTOCOL: PASSED")


if __name__ == "__main__":
    main()
