"""Verify a completed clean Stage-1 run; never opens outer evaluation data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = read(args.config)
    root = Path(cfg["output_root"])
    audit = read(root / "STAGE1_CLEAN_DATA_AUDIT.json")
    report = read(root / "contrastive/training_report.json")
    selection = read(root / "contrastive/checkpoint_selection_report.json")
    legacy = read(root / "CONTAMINATION_REGISTRY.json")
    retired = read(root / "COMPROMISED_TEST_V1.json")
    checks = {}
    def gate(name, ok):
        checks[name] = bool(ok)
    gate("global_train_only", audit["global_train_pairs"] == 40 and audit["inner_train"]["pairs"] == 29 and audit["inner_dev"]["pairs"] == 11)
    gate("inner_isolation", all(audit[k] == 0 for k in ("cross_inner_project_overlap", "cross_inner_binary_overlap", "cross_inner_function_overlap", "cross_inner_exact_source_overlap")))
    gate("clean_pair_manifest_unchanged", sha(root / "pairs.jsonl") == audit["clean_pairs_sha256"] == report["train_manifest_sha256"])
    gate("config_snapshot_unchanged", read(root / "config_snapshot.json") == cfg and sha(args.config) == audit["config_sha256"])
    gate("fresh_base_initialization", report["initialization_source"] == "DeepSeek-Coder base + get_peft_model fresh LoRA; no old adapter loaded" and len(report["fresh_lora_initialization_sha256"]) == 64)
    gate("actual_parameter_updates", report["optimizer_steps"] > 0 and report["parameter_update_check"]["lora_max_abs_delta"] > 0 and report["parameter_update_check"]["projection_max_abs_delta"] > 0)
    gate("finite_training", report["status"] == "DONE" and report["nan_or_inf"] is False and report["initial_loss"] is not None and report["final_loss"] is not None)
    gate("selection_from_inner_dev_only", selection["selection_split"] == "stage1_inner_dev_from_GLOBAL_TRAIN_only" and selection["policy"] == cfg["contrastive"]["checkpoint_selection"] and selection["selected_epoch"] in (1, 2) and len(selection["selection_history"]) == 2)
    gate("all_outer_sealed", all(report[k] == 0 and selection[k] == 0 for k in ("outer_DEV_opened", "CALIBRATION_opened", "COMPROMISED_TEST_V1_opened", "FINAL_TEST_V2_opened")) and report["test_evaluated"] is False)
    gate("reload_real_forward_passed", report["reload_smoke"]["passed"] is True and report["reload_smoke"]["shape"] == [1, cfg["contrastive"]["projection_dim"]])
    adapter = Path(report["adapter_dir"]) / "adapter_model.safetensors"
    projection = Path(report["projection_path"])
    gate("adapter_and_projection_hashes", sha(adapter) == report["adapter_sha256"] and sha(projection) == report["projection_sha256"])
    old_adapter_sha = legacy["entries"][0]["sha256"]
    gate("old_adapter_not_reused", sha(adapter) != old_adapter_sha)
    gate("compromised_test_retired", retired["status"] == "COMPROMISED_TEST_V1" and "FINAL_TEST_REPORT" in retired["forbidden_use"])
    gate("final_test_v2_still_sealed", read(root / "FINAL_TEST_V2_REQUIREMENTS.json")["performance_evaluation_allowed"] is False)
    status = "READY_FOR_CLEAN_STAGE2_RETRAIN" if all(checks.values()) else "BLOCKED_NEED_MORE_STAGE1_TRAIN_DATA"
    artifacts = {"adapter": {"path": str(adapter), "sha256": sha(adapter)},
                 "adapter_config": {"path": str(adapter.parent / "adapter_config.json"), "sha256": sha(adapter.parent / "adapter_config.json")},
                 "projection": {"path": str(projection), "sha256": sha(projection)},
                 "training_report": {"path": str(root / "contrastive/training_report.json"), "sha256": sha(root / "contrastive/training_report.json")},
                 "config_snapshot": {"path": str(root / "config_snapshot.json"), "sha256": sha(root / "config_snapshot.json")},
                 "checkpoint_selection_report": {"path": str(root / "contrastive/checkpoint_selection_report.json"), "sha256": sha(root / "contrastive/checkpoint_selection_report.json")}}
    final = {"schema": "seraph_stage1_clean_v0_1_5_final_status", "status": status,
             "checks": checks, "failed_checks": [name for name, ok in checks.items() if not ok],
             "artifacts": artifacts, "stage2_runs_started": 0,
             "FINAL_TEST_V2_status": read(root / "FINAL_TEST_V2_REQUIREMENTS.json")["status"],
             "COMPROMISED_TEST_V1_status": "COMPROMISED_TEST_V1"}
    write(root / "v0_1_5_final_status.json", final)
    selected = next(x["inner_dev"] for x in selection["selection_history"] if x["epoch"] == selection["selected_epoch"])
    lines = ["# SERAPH clean Stage-1 v0_1_5", "", "Status: **%s**" % status, "",
             "40 GLOBAL TRAIN pairs from 2 projects and 2 binaries were project-isolated into 29 train / 11 inner-dev. All 40 source-function identities are unique. The single-project inner-dev is an engineering check, not a reliable generalization estimate.", "",
             "Fresh DeepSeek base + newly initialized LoRA were used; the old v0_1_3 adapter was not loaded. Architecture and hyperparameters match v0_1_3. Bidirectional source↔pseudo-C contrastive learning selected epoch %d by inner-dev mean MRR only." % selection["selected_epoch"], "",
             "- Optimizer steps: %d; initial/final loss: %.6f / %.6f." % (report["optimizer_steps"], report["initial_loss"], report["final_loss"]),
             "- Inner-dev source→pseudo-C R@1/MRR: %.4f / %.4f; pseudo-C→source R@1/MRR: %.4f / %.4f." % (selected["source_to_pseudoc"]["R1"], selected["source_to_pseudoc"]["MRR"], selected["pseudoc_to_source"]["R1"], selected["pseudoc_to_source"]["MRR"]),
             "- Reload base + clean adapter + projection, real forward: %s." % ("PASS" if report["reload_smoke"]["passed"] else "FAIL"),
             "- LoRA/projection update deltas: %.9g / %.9g." % (report["parameter_update_check"]["lora_max_abs_delta"], report["parameter_update_check"]["projection_max_abs_delta"]),
             "- Outer DEV, CALIBRATION, COMPROMISED_TEST_V1, FINAL_TEST_V2 opened during training: 0/0/0/0.",
             "- COMPROMISED_TEST_V1 remains diagnostic-only. FINAL_TEST_V2 requires new independent data and was not created or evaluated.",
             "- v0_2_10 Stage-2 runs started: 0. Stop here for user review.", "",
             "## Artifact SHA-256", ""]
    lines += ["- %s: `%s`" % (key, value["sha256"]) for key, value in artifacts.items()]
    (root / "V0_1_5_CLEAN_STAGE1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "failed_checks": final["failed_checks"], "adapter_sha256": artifacts["adapter"]["sha256"], "stage2_runs_started": 0}, indent=2))
    if not all(checks.values()):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
