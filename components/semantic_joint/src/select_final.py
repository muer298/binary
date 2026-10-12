"""Predeclared clean DEV-only module selection after all nine Stage-2 runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import dump_json, load_json, sha256_file


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def choose_arm(arms, module_metrics, rule):
    """Lexicographic rule fixed in config before any v0_2_10 DEV evaluation."""
    tol = float(rule["tie_tolerance"])
    order = list(rule["fixed_arm_order"])
    require(set(arms) == set(order), "arm selection set changed")

    def better(left, right):
        a, b = module_metrics[left], module_metrics[right]
        comparisons = (
            (a["R@1"]["mean"], b["R@1"]["mean"], True),
            (a["R@1"]["std_population"], b["R@1"]["std_population"], False),
            (a["MRR"]["mean"], b["MRR"]["mean"], True),
        )
        for x, y, higher in comparisons:
            require(math.isfinite(float(x)) and math.isfinite(float(y)), "non-finite DEV selection metric")
            if abs(x - y) > tol:
                return x > y if higher else x < y
        return order.index(left) < order.index(right)

    winner = order[0]
    for arm in order[1:]:
        if better(arm, winner):
            winner = arm
    return winner


def metric_table(aggregate, modules, arms, seeds, name):
    head = "| Module | Arm | " + " | ".join(str(seed) for seed in seeds) + " | Mean | Std |"
    lines = [head, "|---|---|" + "---:|" * (len(seeds) + 2)]
    for module in modules:
        for arm in arms:
            item = aggregate["metrics"][module][arm][name]
            cells = [module, arm] + ["%.6f" % x for x in item["values"]]
            cells += ["%.6f" % item["mean"], "%.6f" % item["std_population"]]
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_json(args.config)
    root = Path(cfg["output_root"])
    contract = load_json(root / "v0_2_10_contract.json")
    aggregate = load_json(root / "CLEAN_MULTI_SEED_METRICS_V2.json")
    checkpoints = load_json(root / "CHECKPOINT_MANIFEST_V2.json")
    status = load_json(root / "v0_2_10_aggregate_status.json")
    require(not (root / "STAGE2_FINAL_SELECTION_CONTRACT_CLEAN_V2.json").exists(),
            "refusing to overwrite clean V2 module selection")
    require(not (root / "V0_2_10_CLEAN_STAGE2_REPORT.md").exists(),
            "refusing to overwrite clean V2 report")
    checks = {}

    def gate(name, ok):
        checks[name] = bool(ok)

    gate("nine_runs_complete", status.get("status") == "PASS_AWAITING_CLEAN_MODULE_SELECTION"
         and aggregate.get("status") == "PASS_AWAITING_CLEAN_MODULE_SELECTION"
         and checkpoints.get("actual_count") == checkpoints.get("expected_count") == 9
         and checkpoints.get("all_reload_passed") is True)
    gate("no_numerical_or_reload_failures", not aggregate.get("failures")
         and all(report.get("nan_or_inf") is False and report.get("reload_smoke", {}).get("passed") is True
                 for arm in cfg["arms"] for report in aggregate["all_seed_reports"][arm].values()))
    gate("clean_adapter_bound", contract["clean_stage1_adapter_sha256"]
         == cfg["llm"]["expected_stage1_adapter_sha256"]
         == sha256_file(Path(cfg["llm"]["stage1_adapter"]) / "adapter_model.safetensors")
         and "first_batch_v0_1_3" not in cfg["llm"]["stage1_adapter"])
    gate("clean_hsym_bound", contract["clean_embedding_sha256"]
         == cfg["expected_clean_artifact_sha256"]["embedding"]
         == sha256_file(cfg["clean_embedding_path"]))
    gate("fixed_global_manifest_and_order", contract["manifest_sha256"] == cfg["expected_global_manifest_sha256"]
         and contract["candidate_order_sha256"] == cfg["expected_candidate_order_sha256"])
    gate("position_and_permutation", aggregate.get("position_sanity_passed") is True
         and aggregate.get("permutation_robustness_passed") is True)
    gate("selection_rule_predeclared", contract["module_selection_rule"] == cfg["module_selection_rule"]
         and contract["module_selection_code_sha256"] == sha256_file(__file__)
         and contract["training_config_sha256"] == sha256_file(args.config))
    gate("all_sealed_splits_unused", all(x == 0 for x in (
         aggregate.get("test_evaluation_count"), aggregate.get("calibration_evaluation_count"),
         checkpoints.get("test_evaluation_count"), checkpoints.get("calibration_evaluation_count"),
         contract.get("compromised_test_v1_evaluation_count"),
         contract.get("final_test_v2_prediction_count"), contract.get("final_test_v2_evaluation_count"))))
    gate("constraint_disabled", cfg["constraint_loss"]["enabled"] is False
         and float(cfg["constraint_loss"]["weight"]) == 0)
    require(all(checks.values()), "BLOCKED: " + ", ".join(k for k, v in checks.items() if not v))

    arms = list(cfg["arms"])
    seeds = list(cfg["fixed_seeds"])
    modules = list(cfg["enabled_modules"])
    rule = cfg["module_selection_rule"]
    selected = {}
    by_checkpoint = {(row["arm"], int(row["seed"])): row["checkpoint_sha256"]
                     for row in checkpoints["checkpoints"]}
    require(len(by_checkpoint) == 9, "checkpoint identity collision")
    for module in modules:
        arm = choose_arm(arms, aggregate["metrics"][module], rule)
        artifacts = []
        for seed in seeds:
            path = root / "arms" / arm / ("seed_%d" % seed) / "ranker.pt"
            digest = sha256_file(path)
            require(digest == by_checkpoint[(arm, seed)], "selected checkpoint SHA mismatch")
            artifacts.append({"seed": seed, "path": str(path), "sha256": digest})
        selected[module] = {
            "selected_arm": arm, "checkpoint_artifacts": artifacts,
            "DEV_single_seed": {metric: aggregate["metrics"][module][arm][metric]
                                for metric in ("R@1", "MRR", "R@5")},
            "selection_inputs": {candidate: {metric: aggregate["metrics"][module][candidate][metric]
                                            for metric in ("R@1", "MRR", "R@5")}
                                 for candidate in arms},
        }
    policy = {
        "schema": "seraph_clean_stage2_final_selection_contract_v0_2_10",
        "status": "FROZEN_CLEAN_V2", "selection_source": "v0_2_10_DEV_only",
        "selection_rule": rule, "selection_rule_code_sha256": sha256_file(__file__),
        "module_policy": selected,
        "seeds": seeds,
        "ensemble_rule": "per-seed softmax then arithmetic mean aligned by candidate semantic identity",
        "manifest_sha256": contract["manifest_sha256"],
        "candidate_order_sha256": contract["candidate_order_sha256"],
        "clean_hsym_sha256": contract["clean_embedding_sha256"],
        "clean_stage1_adapter_sha256": contract["clean_stage1_adapter_sha256"],
        "APISem_mode": "DETERMINISTIC_CONFIRMED_HARD",
        "APISem_learned_result": "LOW_SAMPLE_AUXILIARY_DIAGNOSTIC_NOT_FOR_SELECTION",
        "explicit_ssq_enabled": False,
        "calibration_mode": "NOT_APPLICABLE_NO_VALID_S_LLM",
        "constraint_mode": "NO_EXECUTABLE_CONSTRAINT_SUPERVISION",
        "COMPROMISED_TEST_V1_opened": 0,
        "FINAL_TEST_V2_prediction_count": 0,
        "FINAL_TEST_V2_evaluation_count": 0,
    }
    dump_json(root / "STAGE2_FINAL_SELECTION_CONTRACT_CLEAN_V2.json", policy)
    v2_status = "NEED_NEW_FINAL_TEST_DATA"
    final = {"schema": "seraph_clean_joint_v0_2_10_final_status",
             "status": "READY_FOR_FINAL_FREEZE_CLEAN_V2", "checks": checks,
             "nine_runs": 9, "all_reload_passed": True,
             "clean_adapter_sha256": contract["clean_stage1_adapter_sha256"],
             "clean_hsym_sha256": contract["clean_embedding_sha256"],
             "calibration_evaluation_count": 0,
             "compromised_test_v1_prediction_count": 0,
             "compromised_test_v1_evaluation_count": 0,
             "final_test_v2_prediction_count": 0,
             "final_test_v2_evaluation_count": 0,
             "final_test_v2_preparation_status": v2_status,
             "final_freeze_started": False}
    dump_json(root / "v0_2_10_final_status.json", final)
    report = ["# SERAPH clean Stage-2 v0_2_10", "",
              "Final status: **READY_FOR_FINAL_FREEZE_CLEAN_V2**", "",
              "All 3 arms × 3 fixed seeds were retrained from the unchanged v0_2_7 TRAIN/DEV and v0_2_8 clean h_sym, with the v0_1_5 clean adapter only. No old v0_2_9 metric/checkpoint was reused.", "",
              "The module choices below follow the predeclared clean-DEV mean/stability rule. APISem learned results (DEV n=8) are auxiliary only; deterministic confirmed-hard remains the final APISem path.", "",
              "## Module selection", "",
              "| Module | Selected arm | Mean R@1 | Std R@1 | Mean MRR |", "|---|---|---:|---:|---:|"]
    for module in modules:
        row = selected[module]
        report.append("| %s | %s | %.6f | %.6f | %.6f |" % (
            module, row["selected_arm"], row["DEV_single_seed"]["R@1"]["mean"],
            row["DEV_single_seed"]["R@1"]["std_population"], row["DEV_single_seed"]["MRR"]["mean"]))
    for metric in ("R@1", "MRR", "R@5"):
        report += ["", "## DEV " + metric, "", *metric_table(aggregate, modules, arms, seeds, metric)]
    report += ["", "R@5 is auxiliary because candidate sets typically contain only 3–4 items.", "",
               "Position sanity: %s; five-permutation robustness: %s; all nine reload real forwards: PASS." % (
                   aggregate["position_sanity_passed"], aggregate["permutation_robustness_passed"]),
               "CALIBRATION, COMPROMISED_TEST_V1, and FINAL_TEST_V2 performance were unused. Constraint and explicit SSQ remain disabled; calibration is N/A.",
               "FINAL_TEST_V2 has not yet been populated with new independent projects; no final test or final freeze was run.", ""]
    (root / "V0_2_10_CLEAN_STAGE2_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (root / "FINAL_TEST_V2_PREPARATION_STATUS.md").write_text(
        "# FINAL_TEST_V2 preparation status\n\nStatus: **NEED_NEW_FINAL_TEST_DATA**.\n\n"
        "The old global TEST is `COMPROMISED_TEST_V1` and remains diagnostic-only. "
        "The clean Stage-2 runs used only global TRAIN/DEV and did not predict, evaluate, select architecture on, or deserialize FINAL_TEST_V2. "
        "New independent project/binary identities (minimum two projects, preferably three) still need to be acquired and processed through the clean feature, instruction mapping, candidate, strict-Gold sealing, and identity firewall pipeline. "
        "Only identity/provenance/coverage/Gold-sealed artifacts may be prepared before separate final-test authorization.\n",
        encoding="utf-8")
    print(json.dumps({"status": final["status"], "selected_arms": {m: selected[m]["selected_arm"] for m in modules},
                      "final_test_v2": v2_status, "final_freeze_started": False}, indent=2))


if __name__ == "__main__":
    main()
