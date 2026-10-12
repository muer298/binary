from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from checkpointing import checkpoint_dir
from common import dump_json, load_json, sha256_file


def _mean_std(values):
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError("non-finite or empty metric series")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std_population": math.sqrt(variance), "values": values}


def _table(reports, arms, modules, seeds, metric):
    lines = ["| Module | Arm | " + " | ".join("Seed %s" % seed for seed in seeds) + " | Mean | Std |",
             "|---|---|" + "---:|" * (len(seeds) + 2)]
    for module in modules:
        for arm in arms:
            values = [reports[arm][str(seed)]["validation"][module][metric] for seed in seeds]
            summary = _mean_std(values)
            cells = [module, arm] + ["%.6f" % value for value in values]
            cells += ["%.6f" % summary["mean"], "%.6f" % summary["std_population"]]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _paired_analysis(reports, seeds, core_modules):
    comparisons = ("unified_llm_gate_clean", "module_aware_clean")
    result = {}
    for seed in seeds:
        symbolic = {row["record_id"]: row for row in
                    reports["symbolic_only_clean"][str(seed)]["validation_details"]}
        result[str(seed)] = {}
        for arm in comparisons:
            current = {row["record_id"]: row for row in reports[arm][str(seed)]["validation_details"]}
            if set(current) != set(symbolic):
                raise RuntimeError("paired DEV identity set mismatch: seed=%s arm=%s" % (seed, arm))
            result[str(seed)][arm] = {}
            for module in core_modules:
                counts = {"symbolic_wrong_llm_correct": 0, "symbolic_correct_llm_wrong": 0,
                          "both_correct": 0, "both_wrong": 0}
                for record_id, left in symbolic.items():
                    if left["module"] != module:
                        continue
                    right = current[record_id]
                    if right["module"] != module or right["gold_candidate_id"] != left["gold_candidate_id"]:
                        raise RuntimeError("paired DEV module/Gold mismatch")
                    left_ok = left["top1_candidate_id"] == left["gold_candidate_id"]
                    right_ok = right["top1_candidate_id"] == right["gold_candidate_id"]
                    if not left_ok and right_ok:
                        counts["symbolic_wrong_llm_correct"] += 1
                    elif left_ok and not right_ok:
                        counts["symbolic_correct_llm_wrong"] += 1
                    elif left_ok and right_ok:
                        counts["both_correct"] += 1
                    else:
                        counts["both_wrong"] += 1
                counts["n"] = sum(counts.values())
                result[str(seed)][arm][module] = counts
    stability = {}
    for arm in comparisons:
        stability[arm] = {}
        for module in core_modules:
            stability[arm][module] = {
                key: _mean_std([result[str(seed)][arm][module][key] for seed in seeds])
                for key in ("symbolic_wrong_llm_correct", "symbolic_correct_llm_wrong",
                            "both_correct", "both_wrong")
            }
    return {"per_seed": result, "cross_seed_stability": stability}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    root = Path(cfg["output_root"])
    contract = load_json(root / "v0_2_10_contract.json")
    arms = [name for name, value in cfg["arms"].items() if value["enabled"]]
    seeds = list(cfg["fixed_seeds"])
    core_modules = list(cfg["enabled_modules"])
    modules = core_modules + (["APISem"] if cfg["apisem_learned"]["enabled"] else [])
    expected_runs = len(arms) * len(seeds)
    reports = {}
    checkpoints = []
    trainer_code_hashes = set()
    failures = []
    for arm in arms:
        reports[arm] = {}
        for seed in seeds:
            directory = checkpoint_dir(root, arm, seed)
            report_path = directory / "training_report.json"
            manifest_path = directory / "checkpoint_manifest.json"
            if not report_path.is_file() or not manifest_path.is_file():
                raise SystemExit("STOP: missing arm/seed report: %s seed=%s" % (arm, seed))
            report = load_json(report_path)
            manifest = load_json(manifest_path)
            identity = "%s/seed_%s" % (arm, seed)
            if report.get("status") != "DONE" or report.get("seed") != seed or report.get("arm") != arm:
                failures.append(identity + ": report identity/status mismatch")
            if report.get("test_evaluation_count") != 0 or report.get("calibration_evaluation_count") != 0:
                failures.append(identity + ": sealed split counter nonzero")
            if any(report.get(key) != 0 for key in (
                    "compromised_test_v1_opened", "compromised_test_v1_evaluated",
                    "final_test_v2_opened", "final_test_v2_predicted", "final_test_v2_evaluated")):
                failures.append(identity + ": compromised TEST or FINAL_TEST_V2 was opened")
            if report.get("nan_or_inf") is not False:
                failures.append(identity + ": numerical stability marker failed")
            if not report.get("reload_smoke", {}).get("passed"):
                failures.append(identity + ": reload real forward failed")
            if arm != "symbolic_only_clean" and (
                    report.get("adapter_sha256_before") != contract["clean_stage1_adapter_sha256"]
                    or report.get("adapter_sha256_after") != contract["clean_stage1_adapter_sha256"]):
                failures.append(identity + ": clean adapter hash mismatch")
            if sha256_file(directory / "ranker.pt") != manifest.get("checkpoint_sha256"):
                failures.append(identity + ": checkpoint hash mismatch")
            expected_init = contract["initialization_hashes"][identity]
            if report.get("parameter_initialization_hashes") != expected_init:
                failures.append(identity + ": initialization hash mismatch")
            if manifest.get("parameter_initialization_hashes") != expected_init:
                failures.append(identity + ": checkpoint initialization hash mismatch")
            for key in ("manifest_sha256", "candidate_order_sha256", "clean_embedding_sha256",
                        "training_config_sha256"):
                if manifest.get(key) != contract.get(key):
                    failures.append(identity + ": %s mismatch" % key)
            if manifest.get("test_evaluation_count") != 0 or manifest.get("calibration_evaluation_count") != 0:
                failures.append(identity + ": checkpoint sealed counter nonzero")
            if not all(module in report.get("validation", {}) for module in modules):
                failures.append(identity + ": missing module validation")
            reports[arm][str(seed)] = report
            checkpoints.append(manifest)
            trainer_code_hashes.add(manifest.get("trainer_code_sha256"))

    metrics = {}
    for module in modules:
        metrics[module] = {}
        for arm in arms:
            metrics[module][arm] = {
                metric: _mean_std([reports[arm][str(seed)]["validation"][module][metric]
                                   for seed in seeds])
                for metric in ("R@1", "R@5", "MRR")
            }
    deltas = {}
    comparisons = (
        ("unified_llm_gate_clean", "symbolic_only_clean"),
        ("module_aware_clean", "symbolic_only_clean"),
        ("module_aware_clean", "unified_llm_gate_clean"),
    )
    for module in core_modules:
        deltas[module] = {}
        for left, right in comparisons:
            name = "%s_minus_%s" % (left, right)
            deltas[module][name] = {
                metric: metrics[module][left][metric]["mean"] - metrics[module][right][metric]["mean"]
                for metric in ("R@1", "R@5", "MRR")
            }

    paired = _paired_analysis(reports, seeds, core_modules)
    protocol = cfg["candidate_protocol"]
    position = contract["position_sanity"]
    position_max = max(position["always_first_R1"], position["majority_position_R1"],
                       position["position_only_R1"])
    position_passed = position_max < float(protocol["position_artifact_block_threshold_R1"])
    if not position_passed:
        failures.append("candidate position baseline exceeds predeclared threshold")

    permutation_passed = True
    for arm in arms:
        for seed in seeds:
            current = reports[arm][str(seed)]["permutation_robustness"]
            passed = (current["top1_identity_agreement"] >= protocol["permutation_min_top1_identity_agreement"]
                      and current["mean_rank_correlation"] >= protocol["permutation_min_rank_correlation"]
                      and current["mean_R1_variance"] <= protocol["permutation_max_R1_variance"])
            permutation_passed = permutation_passed and passed
            if not passed:
                failures.append("permutation robustness failed: %s seed=%s" % (arm, seed))

    gate_collapse = []
    for arm in ("unified_llm_gate_clean", "module_aware_clean"):
        for seed in seeds:
            for module, stats in reports[arm][str(seed)]["gate_statistics"].items():
                if stats is None:
                    continue
                if stats["p90"] < 0.01 or stats["p10"] > 0.99:
                    gate_collapse.append({"arm": arm, "seed": seed, "module": module,
                                          "classification": "near_boundary_gate_collapse", "stats": stats})

    if contract["split_sealing"].get("test_evaluation_count") != 0:
        failures.append("contract TEST evaluation counter nonzero")
    if contract["split_sealing"].get("calibration_evaluation_count") != 0:
        failures.append("contract CALIBRATION evaluation counter nonzero")
    if contract.get("apisem_split_sealing") and (
            contract["apisem_split_sealing"].get("sealed_gold_rows_deserialized") != 0
            or contract["apisem_split_sealing"].get("sealed_query_rows_deserialized") != 0):
        failures.append("APISem sealed row was deserialized")
    if len(checkpoints) != expected_runs:
        failures.append("checkpoint count is not nine")
    if len(trainer_code_hashes) != 1 or None in trainer_code_hashes:
        failures.append("trainer code hash differs across runs")

    final_status = "PASS_AWAITING_CLEAN_MODULE_SELECTION" if not failures else "BLOCKED"
    aggregate = {
        "schema": "seraph_clean_joint_multi_seed_v0_2_10", "status": final_status,
        "arms": arms, "seeds": seeds, "primary_modules": core_modules,
        "auxiliary_modules": ["APISem"] if cfg["apisem_learned"]["enabled"] else [],
        "manifest_sha256": contract["manifest_sha256"],
        "candidate_order_sha256": contract["candidate_order_sha256"],
        "clean_embedding_sha256": contract["clean_embedding_sha256"],
        "clean_stage1_adapter_sha256": contract["clean_stage1_adapter_sha256"],
        "training_config_sha256": contract["training_config_sha256"],
        "trainer_code_sha256": next(iter(trainer_code_hashes)) if len(trainer_code_hashes) == 1 else None,
        "metrics": metrics, "mean_deltas": deltas, "paired_analysis": paired,
        "position_sanity": position, "position_sanity_passed": position_passed,
        "permutation_robustness_passed": permutation_passed,
        "gate_collapse_observations": gate_collapse,
        "failures": failures, "all_seed_reports": reports,
        "test_evaluation_count": 0, "calibration_evaluation_count": 0,
    }
    dump_json(root / "CLEAN_MULTI_SEED_METRICS_V2.json", aggregate)
    dump_json(root / "CHECKPOINT_MANIFEST_V2.json", {
        "schema": "seraph_clean_joint_all_checkpoints_v0_2_10",
        "checkpoints": checkpoints, "expected_count": expected_runs,
        "actual_count": len(checkpoints), "all_reload_passed": not any(
            not reports[arm][str(seed)]["reload_smoke"]["passed"] for arm in arms for seed in seeds),
        "test_evaluation_count": 0, "calibration_evaluation_count": 0,
    })

    report = ["# V0.2.10 Clean Joint Aggregate", "", "Status: **%s**" % final_status, "",
              "All three frozen seeds are reported; no best-seed selection was performed.", "",
              "## Primary R@1", "", _table(reports, arms, core_modules, seeds, "R@1"), "",
              "## Primary MRR", "", _table(reports, arms, core_modules, seeds, "MRR"), "",
              "## Primary R@5", "", _table(reports, arms, core_modules, seeds, "R@5"), "",
              "R@5 has low discriminative power where candidate sets contain only 3–4 candidates.", "",
              "## Mean deltas", "", "```json", json.dumps(deltas, indent=2, sort_keys=True), "```", "",
              "## Candidate-position sanity", "", "```json", json.dumps(position, indent=2, sort_keys=True), "```", "",
              "## Paired analysis", "", "```json", json.dumps(paired, indent=2, sort_keys=True), "```", "",
              "TEST evaluation count: 0", "", "CALIBRATION evaluation count: 0", "",
              "Failures: `%s`" % json.dumps(failures), ""]
    (root / "V0_2_10_AGGREGATE_DRAFT.md").write_text("\n".join(report), encoding="utf-8")

    stability = ["# Clean Seed Stability", "",
                 "All predeclared seeds are included; no seed was selected or hidden.", ""]
    for module in core_modules:
        for arm in arms:
            item = metrics[module][arm]
            stability.append("- %s / %s: R@1 std %.6f; MRR std %.6f; R@5 std %.6f" % (
                module, arm, item["R@1"]["std_population"], item["MRR"]["std_population"],
                item["R@5"]["std_population"]))
    stability += ["", "## Paired cross-seed stability", "", "```json",
                  json.dumps(paired["cross_seed_stability"], indent=2, sort_keys=True), "```", ""]
    (root / "CLEAN_SEED_STABILITY_V2.md").write_text("\n".join(stability), encoding="utf-8")

    permutation = ["# Clean Permutation Robustness", ""]
    gate = ["# Clean Gate Analysis", ""]
    for arm in arms:
        for seed in seeds:
            current = reports[arm][str(seed)]
            perm = current["permutation_robustness"]
            permutation.append("- %s / %s: top1 identity agreement %.6f; mean rank correlation %.6f; mean R@1 variance %.6f" % (
                arm, seed, perm["top1_identity_agreement"], perm["mean_rank_correlation"], perm["mean_R1_variance"]))
            gate.append("- %s / %s: `%s`" % (arm, seed, json.dumps(current["gate_statistics"], sort_keys=True)))
    permutation += ["", "Passed predeclared thresholds: **%s**" % permutation_passed, ""]
    gate += ["", "Near-boundary collapse observations: `%s`" % json.dumps(gate_collapse), ""]
    (root / "CLEAN_PERMUTATION_ROBUSTNESS_V2.md").write_text("\n".join(permutation), encoding="utf-8")
    (root / "CLEAN_GATE_ANALYSIS.md").write_text("\n".join(gate), encoding="utf-8")

    api = ["# Clean APISem Report", "", "Classification: **LOW_SAMPLE_AUXILIARY_MODULE**", "",
           "Learned APISem enabled: `%s`" % str(cfg["apisem_learned"]["enabled"]).lower(), "",
           "Deterministic confirmed_hard seed: ENABLED_SEPARATE_FROM_LEARNED_HEAD", "",
           "The 8 DEV records were not used for architecture, seed, split, Gold, or h_sym selection.", "",
           "CALIBRATION records remain sealed: 4", "", "TEST records remain sealed: 4", ""]
    if cfg["apisem_learned"]["enabled"]:
        api += ["TRAIN records: 64", "", "DEV records: 8", "",
                "## R@1", "", _table(reports, arms, ["APISem"], seeds, "R@1"), "",
                "## MRR", "", _table(reports, arms, ["APISem"], seeds, "MRR"), "",
                "## R@5", "", _table(reports, arms, ["APISem"], seeds, "R@5"), ""]
    (root / "CLEAN_APISem_REPORT_V2.md").write_text("\n".join(api), encoding="utf-8")
    dump_json(root / "v0_2_10_aggregate_status.json", {
        "schema": "seraph_clean_joint_v0_2_10_aggregate", "status": final_status,
        "run_count": len(checkpoints), "expected_run_count": expected_runs,
        "all_reload_passed": all(reports[arm][str(seed)]["reload_smoke"]["passed"]
                                  for arm in arms for seed in seeds),
        "position_sanity_passed": position_passed,
        "permutation_robustness_passed": permutation_passed,
        "clean_embedding_sha256": contract["clean_embedding_sha256"],
        "calibration_evaluation_count": 0, "test_evaluation_count": 0,
        "failures": failures, "next_stage_started": False,
    })
    print(json.dumps({"status": final_status, "arms": arms, "seeds": seeds,
                      "runs": len(checkpoints), "failures": failures,
                      "test_evaluation_count": 0}, indent=2))


if __name__ == "__main__":
    main()
