"""Prepare GLOBAL-TRAIN-only Stage-1 pairs and immutable repair provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def field_from_json_line(line, field):
    """Decode only one top-level identity field before deciding to read payload."""
    marker = json.dumps(field) + ":"
    pos = line.find(marker)
    require(pos >= 0, "missing pair identity field: " + field)
    value, _ = json.JSONDecoder().raw_decode(line[pos + len(marker):].lstrip())
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    require(Path(args.config).resolve().parent == Path(cfg["package_root"]).resolve(), "package binding mismatch")
    out = Path(cfg["output_root"])
    require(not out.exists(), "refusing to overwrite clean Stage-1 output")
    split = load(cfg["global_split_contract"])
    require(split.get("firewall_pass") is True and split.get("test_predictions_or_metrics_read") is False,
            "global split contract failed")
    roles = {str(k): v for k, v in split["sample_roles"].items()}
    require(sha_file(cfg["global_manifest"]) == cfg["expected_global_manifest_sha256"], "global manifest SHA changed")
    old_final = load(cfg["v032_model_manifest"])
    clean = old_final["clean_hsym"]["artifacts"]
    require(clean["embeddings"]["sha256"] == cfg["expected_clean_hsym_sha256"], "clean h_sym binding changed")
    for name, item in clean.items():
        require(sha_file(item["path"]) == item["sha256"], "clean h_sym artifact changed: " + name)
    base = old_final["llm_base_model"]
    require(sha_file(base["config"]["path"]) == base["config"]["sha256"], "base config changed")
    for part in base["weight_parts"]:
        require(sha_file(part["path"]) == part["sha256"], "base model weight changed")

    meta = {}
    with Path(cfg["global_identity_inventory"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sid = str(row["sample_id"])
            identity = (row.get("project_id"), row.get("binary_id"))
            if sid in meta:
                require(meta[sid] == identity, "sample project/binary identity conflict")
            meta[sid] = identity
    inner = cfg["stage1_inner_split"]
    desired = {inner["train_project"]: "train", inner["inner_dev_project"]: "val"}
    require(len(desired) == 2, "inner train and dev project must differ")
    rows = []
    role_counts = Counter()
    with Path(cfg["old_stage1_pairs"]).open(encoding="utf-8") as stream:
        for line in stream:
            sid = str(field_from_json_line(line, "sample_id"))
            role = roles.get(sid, "UNKNOWN")
            role_counts[role] += 1
            if role != "TRAIN":
                continue  # Outer payload is not deserialized or copied.
            row = json.loads(line)
            require(sid in meta, "missing global identity for clean pair")
            project, binary = meta[sid]
            require(project in desired and binary, "unexpected GLOBAL TRAIN project/binary")
            clean_row = dict(row)
            clean_row["split"] = desired[project]
            clean_row["global_role"] = "TRAIN"
            clean_row["project_id"] = project
            clean_row["binary_id"] = binary
            rows.append(clean_row)
    require(len(rows) == cfg["pair_minimum"] == 40, "GLOBAL TRAIN pair count is not 40")
    counts = Counter(r["split"] for r in rows)
    require(dict(counts) == inner["expected_pairs"], "project-isolated inner split size changed")
    source_ids = [r["source_function_id"] for r in rows]
    require(len(set(source_ids)) == inner["expected_unique_source_functions"], "source function identity duplicate/missing")
    projects = {role: {r["project_id"] for r in rows if r["split"] == role} for role in ("train", "val")}
    binaries = {role: {r["binary_id"] for r in rows if r["split"] == role} for role in ("train", "val")}
    functions = {role: {r["source_function_id"] for r in rows if r["split"] == role} for role in ("train", "val")}
    source_hashes = {role: {hashlib.sha256(r["source_code"].encode()).hexdigest() for r in rows if r["split"] == role} for role in ("train", "val")}
    require(not projects["train"] & projects["val"], "project overlap in Stage-1 inner split")
    require(not binaries["train"] & binaries["val"], "binary overlap in Stage-1 inner split")
    require(not functions["train"] & functions["val"], "function overlap in Stage-1 inner split")
    require(not source_hashes["train"] & source_hashes["val"], "exact source-code overlap in Stage-1 inner split")
    require(all(r["global_role"] == "TRAIN" and roles[str(r["sample_id"])] == "TRAIN" for r in rows),
            "outer role was admitted to clean pairs")
    root = Path(cfg["global_split_contract"]).parent.parent
    checkpoint_manifest = load(cfg["v029_checkpoint_manifest"])
    legacy_checkpoints = []
    for item in checkpoint_manifest["checkpoints"]:
        if item["arm"] in ("unified_llm_gate_clean", "module_aware_clean"):
            path = root / "clean_joint_v0_2_9" / "arms" / item["arm"] / ("seed_%d" % int(item["seed"])) / "ranker.pt"
            require(sha_file(path) == item["checkpoint_sha256"], "legacy checkpoint hash changed")
            legacy_checkpoints.append({"arm": item["arm"], "seed": item["seed"],
                                       "path": str(path), "sha256": item["checkpoint_sha256"]})
    require(len(legacy_checkpoints) == 6, "expected six adapter-dependent old Stage-2 checkpoints")
    adapter_path = Path(cfg["old_stage1_adapter"]) / "adapter_model.safetensors"
    old_adapter_sha = sha_file(adapter_path)
    require(old_adapter_sha == old_final["stage1_adapter"]["weights"]["sha256"], "old adapter binding changed")
    contaminated = {
        "schema": "seraph_v0_1_5_contamination_registry", "status": "CONTAMINATED_FOR_FINAL_EVALUATION",
        "retention_role": "PRE_CLEAN_STAGE1_DEVELOPMENT_HISTORY", "delete_legacy_results": False,
        "entries": [
            {"kind": "Stage1_adapter", "path": str(adapter_path), "sha256": old_adapter_sha,
             "status": "CONTAMINATED_FOR_FINAL_EVALUATION"},
            {"kind": "Stage2_unified_and_module_aware_checkpoints", "artifacts": legacy_checkpoints,
             "status": "CONTAMINATED_FOR_FINAL_EVALUATION"},
            {"kind": "SSQ_v0_3_0_and_v0_3_1_results", "paths": [
                str(root / "local_ssq_v0_3_0_smoke"), str(root / "local_ssq_v0_3_1_scoring"),
                str(root / "ssq_layout_dev_supplement_v0_3_1a"), str(root / "local_ssq_v0_3_1_scoring_v2")],
             "status": "CONTAMINATED_FOR_FINAL_EVALUATION"},
            {"kind": "v0_3_2_final_model_selection", "path": cfg["v032_model_manifest"],
             "sha256": sha_file(cfg["v032_model_manifest"]),
             "status": "CONTAMINATED_FOR_FINAL_EVALUATION"},
        ],
        "cause": "v0_1_3 adapter trained/selected on 55 non-global-TRAIN pairs, including 13 global TEST pairs",
    }
    test_ids = sorted(sid for sid, role in roles.items() if role == "TEST")
    retired = {"schema": "seraph_compromised_test_v1_retirement", "status": "COMPROMISED_TEST_V1",
               "permitted_use": "diagnostic_only", "forbidden_use": ["FINAL_TEST_REPORT", "final_performance_claim", "model_selection"],
               "global_test_sample_identity_sha256": sha_json(test_ids),
               "global_test_sample_count": len(test_ids),
               "global_manifest_sha256": cfg["expected_global_manifest_sha256"],
               "test_performance_opened": False}
    v2 = {"schema": "seraph_final_test_v2_requirements", "status": "NEED_NEW_FINAL_TEST_DATA",
          "min_independent_projects": 2, "preferred_independent_projects": 3,
          "independence": "new project/binary identities absent from all Stage1 training/selection, h_sym training, Stage2 training, DEV, CALIBRATION and SSQ development",
          "required_modules": ["TypeSem", "InterfaceSem", "LayoutSem"],
          "optional_module": "APISem strict Gold", "allowed_creation_outputs": ["identity", "manifest", "Gold sealed file", "SHA256"],
          "performance_evaluation_allowed": False, "data_selected": False,
          "does_not_block_clean_stage1_stage2_retrain": True}
    audit = {"schema": "seraph_stage1_clean_data_audit_v0_1_5", "status": "PASS_ENGINEERING_SCALE",
             "global_train_pairs": len(rows), "projects": sorted(projects["train"] | projects["val"]),
             "unique_projects": 2, "unique_binaries": len(binaries["train"] | binaries["val"]),
             "unique_source_functions": len(set(source_ids)),
             "inner_train": {"pairs": counts["train"], "projects": sorted(projects["train"]),
                              "binaries": sorted(binaries["train"]), "source_groups": len(functions["train"])},
             "inner_dev": {"pairs": counts["val"], "projects": sorted(projects["val"]),
                            "binaries": sorted(binaries["val"]), "source_groups": len(functions["val"])},
             "cross_inner_project_overlap": 0, "cross_inner_binary_overlap": 0,
             "cross_inner_function_overlap": 0, "cross_inner_exact_source_overlap": 0,
             "outer_payload_rows_deserialized": 0,
             "source_file_roles_identity_scan": dict(role_counts),
             "validation_caveat": "only one inner-dev project; engineering check, not stable generalization estimate"}

    pending = out.with_name(out.name + ".pending.%d" % os.getpid())
    pending.mkdir(parents=True, exist_ok=False)
    pair_path = pending / "pairs.jsonl"
    with pair_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    audit["clean_pairs_sha256"] = sha_file(pair_path)
    audit["stage1_inner_dev_identity_sha256"] = sha_json(sorted((r["sample_id"], r["source_function_id"]) for r in rows if r["split"] == "val"))
    audit["source_v013_pairs_sha256"] = sha_file(cfg["old_stage1_pairs"])
    audit["global_split_contract_sha256"] = sha_file(cfg["global_split_contract"])
    audit["clean_hsym_sha256"] = clean["embeddings"]["sha256"]
    audit["base_model_config_sha256"] = base["config"]["sha256"]
    audit["base_model_weight_sha256"] = [x["sha256"] for x in base["weight_parts"]]
    audit["config_sha256"] = sha_file(args.config)
    save(pending / "STAGE1_CLEAN_DATA_AUDIT.json", audit)
    save(pending / "CONTAMINATION_REGISTRY.json", contaminated)
    save(pending / "COMPROMISED_TEST_V1.json", retired)
    save(pending / "FINAL_TEST_V2_REQUIREMENTS.json", v2)
    save(pending / "config_snapshot.json", cfg)
    (pending / "FINAL_TEST_V2_PLAN.md").write_text(
        "# FINAL_TEST_V2 plan\n\nStatus: **NEED_NEW_FINAL_TEST_DATA**.\n\n"
        "Retire current global TEST as `COMPROMISED_TEST_V1`; it may be used only for diagnostics, never final performance or model selection.\n\n"
        "Acquire at least two, preferably three, independent new projects/binaries never used by Stage-1 training or selection, h_sym training, Stage-2 training, DEV, CALIBRATION or SSQ development. "
        "Build TypeSem, InterfaceSem and LayoutSem with strict Gold and valid candidates; APISem strict Gold is desirable. "
        "Run clean feature, mapping, position-balanced candidate and identity-firewall checks. "
        "Produce a sealed identity/manifest/Gold file and SHA-256; do not predict or evaluate. "
        "No new project has yet been designated FINAL_TEST_V2. This does not block clean retraining.\n",
        encoding="utf-8")
    os.replace(pending, out)
    print(json.dumps({"status": audit["status"], "global_train_pairs": 40,
                      "inner_train": 29, "inner_dev": 11,
                      "clean_pairs_sha256": audit["clean_pairs_sha256"],
                      "final_test_v2": v2["status"], "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
