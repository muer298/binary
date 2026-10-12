from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import dump_json, load_json, sha256_file


PLACEHOLDER = "REQUIRED_PLACEHOLDER"
FORBIDDEN_ARTIFACT_TOKENS = ("alignment_model_best.pt", "/node_embeddings.pt", "673-D", "673d",
                             "PRE_PROTOCOL_REPAIR")


def _embedding_tensor(payload):
    if torch.is_tensor(payload):
        return payload.float()
    if isinstance(payload, dict) and torch.is_tensor(payload.get("embeddings")):
        return payload["embeddings"].float()
    raise RuntimeError("unsupported clean embedding payload")


def _load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run_preflight(cfg):
    checks = {}
    errors = []

    def check(name, value, detail=None):
        passed = bool(value)
        checks[name] = {"passed": passed, "detail": detail}
        if not passed:
            errors.append(name)

    config_text = json.dumps(cfg, ensure_ascii=False)
    check("no_required_placeholder_remaining", PLACEHOLDER not in config_text,
          "all formal artifact paths must be explicitly bound")
    check("no_pre_protocol_repair_artifact_reference",
          not any(token in config_text for token in FORBIDDEN_ARTIFACT_TOKENS),
          "old node_embeddings/alignment/673-D artifacts are forbidden")
    adapter_path = Path(cfg["llm"]["stage1_adapter"])
    expected_adapter_root = Path("/data/chaoni/mjy/resym_exp/training_runs/seraph_local/stage1_clean_v0_1_5/contrastive/adapter")
    check("old_contaminated_adapter_path_forbidden",
          adapter_path == expected_adapter_root and "first_batch_v0_1_3" not in config_text,
          str(adapter_path))
    clean_status = load_json(cfg["stage1_clean_status"])
    clean_training = load_json(cfg["stage1_clean_training_report"])
    contamination = load_json(cfg["contamination_registry"])
    retired_test = load_json(cfg["compromised_test_v1_registry"])
    check("clean_stage1_ready_and_reloaded",
          clean_status.get("status") == "READY_FOR_CLEAN_STAGE2_RETRAIN"
          and clean_training.get("status") == "DONE"
          and clean_training.get("reload_smoke", {}).get("passed") is True
          and clean_training.get("outer_DEV_opened") == 0
          and clean_training.get("CALIBRATION_opened") == 0
          and clean_training.get("COMPROMISED_TEST_V1_opened") == 0,
          clean_status.get("status"))
    actual_adapter_sha = sha256_file(adapter_path / "adapter_model.safetensors")
    old_adapter_sha = contamination["entries"][0]["sha256"]
    check("clean_adapter_sha256_only",
          actual_adapter_sha == cfg["llm"]["expected_stage1_adapter_sha256"]
          == clean_training.get("adapter_sha256")
          and actual_adapter_sha != old_adapter_sha,
          {"actual": actual_adapter_sha, "expected": cfg["llm"]["expected_stage1_adapter_sha256"]})
    check("old_test_compromised_and_unused",
          retired_test.get("status") == "COMPROMISED_TEST_V1"
          and "FINAL_TEST_REPORT" in retired_test.get("forbidden_use", []),
          retired_test.get("status"))
    check("constraint_loss_disabled",
          cfg["constraint_loss"]["enabled"] is False and float(cfg["constraint_loss"]["weight"]) == 0.0,
          cfg["constraint_loss"])
    check("three_fixed_seeds", len(cfg["fixed_seeds"]) == 3 and len(set(cfg["fixed_seeds"])) == 3,
          cfg["fixed_seeds"])
    check("three_arms", set(cfg["arms"]) == {
        "symbolic_only_clean", "unified_llm_gate_clean", "module_aware_clean"
    }, sorted(cfg["arms"]))

    embedding_path = cfg["clean_embedding_path"]
    index_path = cfg["clean_embedding_index"]
    placeholders_resolved = embedding_path != PLACEHOLDER and index_path != PLACEHOLDER
    check("clean_embedding_paths_resolved", placeholders_resolved,
          {"embedding": embedding_path, "index": index_path})
    if not placeholders_resolved:
        return {"schema": "seraph_v0_2_9_preflight", "status": "STOP", "checks": checks,
                "failed_checks": errors, "execution_authorized": False}

    required = {
        "v028_final": cfg["v028_final_report"],
        "v028_training": cfg["v028_training_report"],
        "feature_leakage": cfg["v028_feature_leakage_audit"],
        "coverage": cfg["v028_coverage_report"],
        "export_report": cfg["v028_export_report"],
        "v027_report": cfg["v027_report"],
        "global_manifest": cfg["global_manifest"],
        "clean_embedding": embedding_path,
        "clean_index": index_path,
        "clean_checkpoint": cfg["clean_alignment_checkpoint"],
        "clean_schema": cfg["clean_feature_schema"],
        "clean_vocab": cfg["clean_feature_vocab"],
    }
    missing = [name for name, path in required.items() if not Path(path).is_file()]
    check("required_files_exist", not missing, missing)
    if missing:
        return {"schema": "seraph_v0_2_9_preflight", "status": "STOP", "checks": checks,
                "failed_checks": errors, "execution_authorized": False}

    final = load_json(cfg["v028_final_report"])
    training = load_json(cfg["v028_training_report"])
    leakage = load_json(cfg["v028_feature_leakage_audit"])
    coverage = load_json(cfg["v028_coverage_report"])
    export = load_json(cfg["v028_export_report"])
    v027 = load_json(cfg["v027_report"])
    global_split = load_json(cfg["global_split_contract"])
    check("global_train_dev_and_sealed_role_firewall",
          global_split.get("firewall_pass") is True
          and global_split.get("test_predictions_or_metrics_read") is False
          and all(int(item["count"]) == 0
                  for kinds in global_split.get("pairwise_firewall", {}).values()
                  for item in kinds.values()),
          global_split.get("firewall_pass"))
    alignment = load_json(Path(cfg["v028_root"]) / "training" / "alignment_metrics.json")
    check("v028_ready", final.get("status") == "READY_FOR_V0_2_9_CLEAN_JOINT", final.get("status"))
    clean_final = final.get("clean_hsym", {})
    expected_shape = list(cfg["expected_clean_embedding_shape"])
    expected_index_entries = int(cfg["expected_clean_index_entries"])
    check("v028_final_embedding_contract",
          clean_final.get("shape") == expected_shape
          and clean_final.get("index_count") == expected_index_entries
          and clean_final.get("nan_count") == 0
          and clean_final.get("inf_count") == 0
          and clean_final.get("duplicate_node_id_count") == 0
          and clean_final.get("reload_passed") is True
          and clean_final.get("real_forward_passed") is True,
          clean_final)
    check("v028_export_report_passed",
          export.get("passed") is True
          and export.get("shape") == expected_shape
          and export.get("index_count") == expected_index_entries
          and export.get("nan_count") == 0 and export.get("inf_count") == 0
          and export.get("reload_passed") is True
          and export.get("real_forward_passed") is True
          and export.get("outer_performance_metrics_computed") is False,
          export)
    check("v028_checkpoint_binding",
          clean_final.get("checkpoint") == cfg["clean_alignment_checkpoint"]
          and export.get("checkpoint") == cfg["clean_alignment_checkpoint"],
          {"config": cfg["clean_alignment_checkpoint"],
           "final": clean_final.get("checkpoint"), "export": export.get("checkpoint")})
    check("manifest_sha256", sha256_file(cfg["global_manifest"]) == cfg["expected_global_manifest_sha256"],
          sha256_file(cfg["global_manifest"]))
    check("feature_forbidden_identity_zero",
          leakage.get("passed") is True and leakage.get("forbidden_identity_feature_count") == 0
          and leakage.get("hash_feature_count") == 0, leakage)
    check("v028_outer_metrics_not_computed", training.get("outer_metrics_computed") is False,
          training.get("outer_metrics_computed"))
    evaluation = alignment.get("evaluation_audit", {})
    truth = alignment.get("training_truth", {})
    split = alignment.get("dataset", {}).get("split_info", {})
    loss_schedule = alignment.get("model", {}).get("loss_weight_schedule", {})
    check("alignment_test_records_zero",
          alignment.get("dataset", {}).get("test_records") == 0
          and split.get("pair_counts", {}).get("test") == 0,
          {"dataset_test": alignment.get("dataset", {}).get("test_records"),
           "split_test": split.get("pair_counts", {}).get("test")})
    check("alignment_optimizer_gradient_steps_positive",
          int(truth.get("optimizer_steps", 0)) > 0
          and int(truth.get("gradient_steps", 0)) > 0,
          {"optimizer_steps": truth.get("optimizer_steps"),
           "gradient_steps": truth.get("gradient_steps")})
    check("alignment_parameters_changed",
          int(truth.get("changed_parameter_tensors", 0)) > 0
          and float(truth.get("max_parameter_abs_delta", 0.0)) > 0,
          {"changed_parameter_tensors": truth.get("changed_parameter_tensors"),
           "max_parameter_abs_delta": truth.get("max_parameter_abs_delta")})
    check("alignment_finite", truth.get("nan_or_inf_loss") is False,
          truth.get("nan_or_inf_loss"))
    check("alignment_constraint_disabled",
          float(loss_schedule.get("lambda_constraint_final", -1)) == 0.0
          and float(loss_schedule.get("lambda_constraint_start", -1)) == 0.0,
          loss_schedule)
    check("test_evaluation_count_zero", evaluation.get("test_performance_evaluations") == 0,
          evaluation.get("test_performance_evaluations"))
    check("outer_records_not_opened", all(evaluation.get(key) == 0 for key in (
        "outer_dev_records_opened", "outer_cal_records_opened", "outer_test_records_opened")), evaluation)
    check("v027_test_performance_unread", v027.get("test_performance_read") is False,
          v027.get("test_performance_read"))
    check("coverage_is_identity_only", coverage.get("performance_metrics_computed") is False,
          coverage.get("performance_metrics_computed"))

    payload = _load_torch(embedding_path)
    embeddings = _embedding_tensor(payload)
    index = load_json(index_path)
    nan_count = int(torch.isnan(embeddings).sum())
    inf_count = int(torch.isinf(embeddings).sum())
    check("embedding_shape_exact", list(embeddings.shape) == expected_shape,
          list(embeddings.shape))
    check("embedding_nan_zero", nan_count == 0, nan_count)
    check("embedding_inf_zero", inf_count == 0, inf_count)
    check("embedding_index_count_matches",
          len(index) == embeddings.shape[0] == expected_index_entries,
          {"index": len(index), "embedding": int(embeddings.shape[0])})
    index_values = [int(value) for value in index.values()]
    check("embedding_index_unique_in_range",
          len(set(index_values)) == len(index_values)
          and min(index_values, default=-1) == 0
          and max(index_values, default=-1) == embeddings.shape[0] - 1,
          {"unique": len(set(index_values)), "min": min(index_values, default=None),
           "max": max(index_values, default=None)})

    expected_hashes = cfg["expected_clean_artifact_sha256"]
    actual_hashes = {
        "checkpoint": sha256_file(cfg["clean_alignment_checkpoint"]),
        "embedding": sha256_file(embedding_path),
        "index": sha256_file(index_path),
        "schema": sha256_file(cfg["clean_feature_schema"]),
        "vocab": sha256_file(cfg["clean_feature_vocab"]),
    }
    check("clean_artifact_hash_binding", actual_hashes == expected_hashes,
          {"expected": expected_hashes, "actual": actual_hashes})

    coverage_errors = []
    gate = cfg["coverage_gate"]
    for role in gate["roles"]:
        for module in gate["modules"]:
            bucket = coverage["by_role_module"][role][module]
            for kind, threshold_key in (
                ("query", "minimum_query_coverage"),
                ("candidate", "minimum_candidate_coverage"),
                ("gold", "minimum_gold_coverage"),
            ):
                value = bucket[kind + "_coverage"]
                if value is None or value < float(gate[threshold_key]):
                    coverage_errors.append({"role": role, "module": module, "kind": kind,
                                            "coverage": value, "required": gate[threshold_key]})
    check("train_dev_coverage_gate", not coverage_errors, coverage_errors)
    api_enabled = bool(cfg["apisem_learned"]["enabled"])
    api_coverage_ok = not any(row["module"] == "APISem" for row in coverage_errors)
    check("apisem_enablement_contract", (not api_enabled) or api_coverage_ok,
          {"enabled": api_enabled, "coverage_ok": api_coverage_ok})

    status = "PASS" if not errors else "STOP"
    return {
        "schema": "seraph_v0_2_9_preflight",
        "status": status,
        "checks": checks,
        "failed_checks": errors,
        "clean_embedding_sha256": sha256_file(embedding_path),
        "clean_index_sha256": sha256_file(index_path),
        "manifest_sha256": sha256_file(cfg["global_manifest"]),
        "embedding_shape": list(embeddings.shape),
        "clean_artifact_sha256": actual_hashes,
        "execution_authorized": bool(cfg["execution"]["authorized_after_v028_handoff"]),
        "test_evaluation_count": 0,
        "calibration_evaluation_count": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    cfg = load_json(args.config)
    report = run_preflight(cfg)
    if args.report:
        dump_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
