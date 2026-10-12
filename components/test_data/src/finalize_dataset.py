"""Audit and seal FINAL_TEST_V2 without opening a Stage-2 model.

Only identity, legality, position, feature, and mapping statistics are computed.
No model prediction, Gold-based architecture choice, or performance metric.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
from pathlib import Path


RUNS = Path("/data/chaoni/mjy/resym_exp/training_runs/seraph_local")
MODULES = ("TypeSem", "InterfaceSem", "LayoutSem", "APISem")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_md(path: Path, lines: list[str]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def identity_sets() -> dict[str, set[str]]:
    return {key: set() for key in ("project", "binary", "function", "candidate_instance")}


def add_identity(sets: dict[str, set[str]], record: dict) -> None:
    project = record.get("project_id") or record.get("project")
    binary = record.get("binary_id") or record.get("sample_id")
    if project:
        sets["project"].add(str(project))
    if binary:
        sets["binary"].add(str(binary))
    function = record.get("function_identity") or {}
    entry = function.get("function_entry") if isinstance(function, dict) else None
    entry = entry or record.get("function_entry_norm") or record.get("function_entry")
    if binary and entry:
        sets["function"].add(str(binary) + "::" + str(entry).removeprefix("0x").lower().lstrip("0"))
    for key in ("candidate_instance_identities", "candidate_instance_ids", "candidate_ids"):
        values = record.get(key)
        if isinstance(values, list):
            sets["candidate_instance"].update(str(value) for value in values)
            break
    if record.get("node_type") in MODULES and record.get("node_id"):
        sets["candidate_instance"].add(str(record["node_id"]))


def historical_roles(registry: dict) -> tuple[dict[str, dict[str, set[str]]], dict[str, list[str]]]:
    roles = {key: identity_sets() for key in (
        "Stage1 TRAIN", "Stage1 inner-dev", "old Stage1 val/test",
        "v0_2_8 h_sym train", "Stage2 TRAIN", "Stage2 DEV",
        "CALIBRATION", "COMPROMISED_TEST_V1", "SSQ DESIGN",
        "SSQ HOLDOUT", "janet supplement", "ALL_HISTORICAL")}
    sources = collections.defaultdict(list)
    for project in registry["projects"]:
        roles["ALL_HISTORICAL"]["project"].add(project)
    for binary in registry["binary_ids"]:
        roles["ALL_HISTORICAL"]["binary"].add(binary)
    sources["ALL_HISTORICAL"].append(str(registry["metadata_path"]))

    stage1 = RUNS / "stage1_clean_v0_1_5/pairs.jsonl"
    if stage1.is_file():
        sources["Stage1 TRAIN"].append(str(stage1))
        for row in rows(stage1):
            split = str(row.get("split") or "").lower()
            role = "Stage1 TRAIN" if split == "train" else "Stage1 inner-dev" if split in ("val", "validation", "inner_dev", "inner-dev") else "old Stage1 val/test"
            add_identity(roles[role], row)
    for stage1_old in sorted(RUNS.glob("first_batch_v0_1_*/pairs.jsonl")):
        sources["old Stage1 val/test"].append(str(stage1_old))
        for row in rows(stage1_old):
            if str(row.get("split") or "").lower() in ("val", "validation", "test"):
                add_identity(roles["old Stage1 val/test"], row)
    stage2 = RUNS / "joint_v0_2_7_global_split/joint_manifest_global_clean_v0_2_7.jsonl"
    if stage2.is_file():
        map_role = {"TRAIN": "Stage2 TRAIN", "DEV": "Stage2 DEV",
                    "CALIBRATION": "CALIBRATION", "TEST": "COMPROMISED_TEST_V1"}
        for value in map_role.values():
            sources[value].append(str(stage2))
        for row in rows(stage2):
            role = map_role.get(row.get("global_split"))
            if role:
                add_identity(roles[role], row)
    hsym = RUNS / "clean_hsym_v0_2_8/data/encoder_inner_split_manifest.jsonl"
    if hsym.is_file():
        sources["v0_2_8 h_sym train"].append(str(hsym))
        for row in rows(hsym):
            if str(row.get("split") or "").upper() in ("TRAIN", "VALIDATION", "INNER_TRAIN", "INNER_DEV"):
                add_identity(roles["v0_2_8 h_sym train"], row)
    hsym_nodes = RUNS / "clean_hsym_v0_2_8/data/train_nodes.jsonl"
    if hsym_nodes.is_file():
        sources["v0_2_8 h_sym train"].append(str(hsym_nodes))
        for row in rows(hsym_nodes):
            add_identity(roles["v0_2_8 h_sym train"], row)
    ssq = RUNS / "ssq_layout_dev_supplement_v0_3_1a/SSQ_SCORING_SPLIT_V2.json"
    if ssq.is_file():
        source = load(ssq)
        records = source.get("records") or []
        if isinstance(records, list):
            for role in ("SSQ DESIGN", "SSQ HOLDOUT"):
                sources[role].append(str(ssq))
            for row in records:
                if not isinstance(row, dict):
                    continue
                tag = str(row.get("ssq_split") or row.get("scoring_split") or row.get("split") or "").upper()
                role = "SSQ DESIGN" if "DESIGN" in tag else "SSQ HOLDOUT" if "HOLDOUT" in tag else None
                if role:
                    add_identity(roles[role], row)
    roles["janet supplement"]["project"].add("janet-lang/janet")
    sources["janet supplement"].append("explicit_development_exclusion")
    for role, identities in roles.items():
        if role == "ALL_HISTORICAL":
            continue
        for kind in ("function", "candidate_instance"):
            roles["ALL_HISTORICAL"][kind].update(identities[kind])
    return roles, sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(Path(args.config))
    root = Path(cfg["output_root"])
    inputs = root / "final_test_v2_inputs"
    gold_dir = root / "final_test_v2_gold_sealed"
    manifest = root / "FINAL_TEST_V2_MANIFEST.jsonl"
    gold_draft = gold_dir / "candidate_gold_draft_v2.jsonl"
    gold_final = gold_dir / "FINAL_TEST_V2_GOLD_SEALED.jsonl"
    if not manifest.is_file() or not gold_draft.is_file():
        raise RuntimeError("final manifest and Gold draft required")
    required_status = ("GHIDRA_STAGE_STATUS.json", "CLEAN_HSYM_STAGE_STATUS.json",
                       "CANDIDATE_CONSTRUCTION_AUDIT_V2.json", "INSTRUCTION_MAPPING_BUILD_STATUS.json")
    for name in required_status:
        if load(root / name).get("status") != "PASSED":
            raise RuntimeError("required stage not passed: " + name)
    outputs = [gold_final] + [root / name for name in (
        "FINAL_TEST_V2_LEAKAGE_MATRIX.json", "FINAL_TEST_V2_FEATURE_LEAKAGE_AUDIT.md",
        "FINAL_TEST_V2_MAPPING_REPORT.md", "FINAL_TEST_V2_COVERAGE_REPORT.md",
        "FINAL_TEST_V2_POSITION_SANITY.md", "FINAL_TEST_V2_BUILD_REPORT.md",
        "FINAL_TEST_V2_FINAL_STATUS.json", "GOLD_PHYSICAL_ISOLATION_TEST.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite final artifact")

    records = list(rows(manifest))
    gold_rows = list(rows(gold_draft))
    by_gold = {item["record_id"]: item for item in gold_rows}
    if len(by_gold) != len(gold_rows) or len(records) != len(gold_rows):
        raise RuntimeError("manifest/Gold cardinality or duplicate mismatch")
    forbidden_fields = {"gold_candidate_id", "gold_candidate_instance_id", "gold_position", "gold_semantic_id", "gold_strength"}
    for row in records:
        if set(row) & forbidden_fields:
            raise RuntimeError("Gold field appeared in predictor manifest")
        gold = by_gold.get(row["record_id"])
        if gold is None or gold["gold_candidate_instance_id"] not in row["candidate_instance_identities"]:
            raise RuntimeError("Gold absent from candidate set")
        pos = int(gold["gold_position"])
        if row["candidate_order"][pos] != gold["gold_candidate_instance_id"]:
            raise RuntimeError("Gold position/order mismatch")
        if len(set(row["candidate_instance_identities"])) != row["candidate_count"]:
            raise RuntimeError("duplicate candidate instance")
        if len(set(row["candidate_semantic_identities"])) != row["candidate_count"]:
            raise RuntimeError("semantic-equivalent candidates retained")
        if row["candidate_order"] != row["candidate_instance_identities"]:
            raise RuntimeError("candidate order/instance list disagreement")
        if row["sample_id"] != row["binary_id"]:
            raise RuntimeError("sample/binary identity mismatch")
        if any(not str(candidate).startswith(row["sample_id"] + "::") for candidate in row["candidate_instance_identities"]):
            raise RuntimeError("candidate identity not sample-qualified")
    if not records:
        raise RuntimeError("no final test records")

    registry = load(root / "ALL_USED_PROJECTS_AND_BINARIES.json")
    historical, historical_sources = historical_roles(registry)
    final_ids = identity_sets()
    for row in records:
        add_identity(final_ids, row)
    leakage = {}
    for role, prior in historical.items():
        overlaps = {kind: sorted(final_ids[kind] & prior[kind]) for kind in final_ids}
        leakage[role] = {
            "source_paths": historical_sources.get(role, []),
            "historical_counts": {kind: len(prior[kind]) for kind in prior},
            "overlap_counts": {kind: len(overlaps[kind]) for kind in overlaps},
            "overlaps": overlaps,
            "function_and_candidate_instance_identity_include_binary_id": True,
        }
    if any(any(count for count in value["overlap_counts"].values()) for value in leakage.values()):
        raise RuntimeError("historical leakage detected")
    dump(root / "FINAL_TEST_V2_LEAKAGE_MATRIX.json", {
        "schema": "seraph_final_test_v2_leakage_matrix_v1", "status": "PASSED",
        "final_counts": {key: len(value) for key, value in final_ids.items()},
        "matrix": leakage, "canonical_semantic_class_overlap_allowed": True,
        "old_test_performance_opened": 0, "test_prediction_count": 0,
        "performance_evaluation_count": 0})

    feature = load(root / "clean_feature_leakage_audit_coverage_only.json")
    feature_report = load(root / "clean_featurization_report_coverage_only.json")
    schema = load(root / "clean_feature_schema_coverage_only.json")
    export = load(inputs / "clean/clean_embedding_export_report.json")
    if not all((feature.get("passed"), feature.get("finite"), feature_report.get("passed"), export.get("passed"))):
        raise RuntimeError("clean feature/embedding audit failed")
    if any(feature.get(key) for key in ("forbidden_identity_feature_count", "gold_derived_feature_count",
                                       "unknown_review_feature_count", "hash_feature_count", "identity_mutation_failures")):
        raise RuntimeError("forbidden or Gold-derived feature detected")
    if (schema.get("feature_dim") != feature_report.get("feature_dim")
            or export.get("embedding_dim") != 64
            or export.get("shape", [None, None])[1] != 64):
        raise RuntimeError("feature or embedding dimension mismatch")
    write_md(root / "FINAL_TEST_V2_FEATURE_LEAKAGE_AUDIT.md", [
        "# FINAL_TEST_V2 feature leakage audit", "", "Status: PASSED", "",
        f"Feature dimension: {schema['feature_dim']}; embedding dimension: {export['embedding_dim']}.",
        f"Forbidden identity features: {feature['forbidden_identity_feature_count']}.",
        f"Gold-derived features: {feature['gold_derived_feature_count']}.",
        f"Hash features: {feature['hash_feature_count']}.",
        f"Identity mutation cases: {feature['identity_mutation_cases']}; failures: {len(feature['identity_mutation_failures'])}.",
        "Node IDs are row-index metadata, never tensor features. Frozen v0_2_8 encoder was forward-only on CPU.",
    ])

    strict_gold = collections.Counter()
    strict_gold_queries = collections.defaultdict(set)
    for row in rows(gold_dir / "mapped_strict_gold_v2.jsonl"):
        key = (row.get("project"), row.get("target_semantic_type"))
        strict_gold[key] += 1
        strict_gold_queries[key].add(row.get("query_id"))
    selected = collections.Counter((row["project_id"], row["module"]) for row in records)
    cap = load(root / "CANDIDATE_CONSTRUCTION_AUDIT.json")
    emb_index = load(inputs / "clean/node_embedding_index_clean.json")
    coverage = collections.defaultdict(lambda: {"query": 0, "candidate": 0, "gold": 0,
                                            "query_present": 0, "candidate_present": 0, "gold_present": 0})
    levels = collections.defaultdict(collections.Counter)
    windows = collections.Counter()
    position = collections.defaultdict(collections.Counter)
    api_hard = collections.Counter()
    for row in records:
        key = (row["project_id"], row["module"])
        cov = coverage[key]
        cov["query"] += 1
        cov["query_present"] += int(row["query_identity"] in emb_index)
        cov["candidate"] += len(row["candidate_instance_identities"])
        cov["candidate_present"] += sum(candidate in emb_index for candidate in row["candidate_instance_identities"])
        gold_id = by_gold[row["record_id"]]["gold_candidate_instance_id"]
        cov["gold"] += 1
        cov["gold_present"] += int(gold_id in emb_index)
        levels[row["module"]][row["mapping_level"]] += 1
        windows[row["module"]] += int(row["instruction_mapping"]["window"] is not None)
        position[(row["module"], row["candidate_count"])][int(by_gold[row["record_id"]]["gold_position"])] += 1
        if row["module"] == "APISem":
            strength = str(by_gold[row["record_id"]].get("gold_strength") or "").lower()
            api_hard["confirmed_hard_available" if strength == "confirmed_hard" else "no_confirmed_hard_abstain"] += 1
    if any(cov["query"] != cov["query_present"] or cov["candidate"] != cov["candidate_present"]
           or cov["gold"] != cov["gold_present"] for cov in coverage.values()):
        raise RuntimeError("clean h_sym coverage hole")
    by_project = {}
    for project in sorted(final_ids["project"]):
        by_project[project] = {}
        for module in MODULES:
            count = selected[(project, module)]
            by_project[project][module] = {
                "query_count": len(strict_gold_queries[(project, module)]),
                "strict_gold_count": strict_gold[(project, module)],
                "gold_in_candidate_final_count": count,
                "candidate_count_distribution": dict(collections.Counter(
                    row["candidate_count"] for row in records if row["project_id"] == project and row["module"] == module)),
                "clean_hsym_coverage": coverage[(project, module)],
            }
    position_rows = []
    balanced = True
    for (module, count), positions in sorted(position.items()):
        values = [positions.get(i, 0) for i in range(count)]
        n = sum(values)
        balanced &= max(values) - min(values) <= 1
        position_rows.append((module, count, n, values, values[0] / n, max(values) / n))

    write_md(root / "FINAL_TEST_V2_COVERAGE_REPORT.md", [
        "# FINAL_TEST_V2 coverage", "", "No Stage-2 predictions or performance metrics were computed.", "",
        "Query count below is the number of unique strict-Gold query IDs; unlabeled task queries are not enumerated here.", "",
        "| Project | Module | Strict-Gold query IDs | Strict Gold | Gold-in-candidate | h_sym Q/C/G |",
        "|---|---|---:|---:|---:|---|",
        *[f"| {project} | {module} | {v['query_count']} | {v['strict_gold_count']} | {v['gold_in_candidate_final_count']} | "
          f"{v['clean_hsym_coverage']['query_present']}/{v['clean_hsym_coverage']['query']}, "
          f"{v['clean_hsym_coverage']['candidate_present']}/{v['clean_hsym_coverage']['candidate']}, "
          f"{v['clean_hsym_coverage']['gold_present']}/{v['clean_hsym_coverage']['gold']} |"
          for project, modules in by_project.items() for module, v in modules.items()],
        "", f"APISem confirmed_hard available: {api_hard['confirmed_hard_available']}; "
        f"no confirmed_hard (future ABSTAIN): {api_hard['no_confirmed_hard_abstain']}.",
        "APISem learned head was not run. Counts are contract availability, not accuracy.",
    ])
    write_md(root / "FINAL_TEST_V2_MAPPING_REPORT.md", [
        "# FINAL_TEST_V2 instruction mapping", "",
        "L0 = exact token anchor. L2 = verified non-root syntax-node evidence (raw L1/L2 retained). "
        "UNKNOWN remains UNKNOWN with symbolic-only fallback.", "",
        "| Module | L0 | L2 | UNKNOWN | Span-centred window usable |",
        "|---|---:|---:|---:|---:|",
        *[f"| {module} | {levels[module]['L0']} | {levels[module]['L2']} | {levels[module]['UNKNOWN']} | "
          f"{windows[module]}/{sum(levels[module].values())} |" for module in MODULES],
        "", "No nearest-PC, nearest-line, string similarity, or function-root fallback was used.",
    ])
    write_md(root / "FINAL_TEST_V2_POSITION_SANITY.md", [
        "# FINAL_TEST_V2 candidate-position sanity", "",
        f"Fixed deterministic stratum cycling; all module × candidate-count strata balanced: {balanced}.",
        "Expected always-first and majority-position rates below are label-position priors, not model performance.", "",
        "| Module | Candidates | n | Gold position counts | Always-first expected | Majority-position expected |",
        "|---|---:|---:|---|---:|---:|",
        *[f"| {m} | {c} | {n} | {v} | {first:.4f} | {majority:.4f} |"
          for m, c, n, v, first, majority in position_rows],
    ])

    # Seal only after all joins and summaries are complete. Mode 000 on the
    # entire Gold directory makes every oracle/draft/final Gold unavailable to
    # an ordinary prediction subprocess under the same non-root account.
    with gold_final.open("x", encoding="utf-8") as handle:
        for row in sorted(gold_rows, key=lambda x: x["record_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    gold_hash = sha(gold_final)
    os.chmod(gold_final, 0o000)
    os.chmod(gold_dir, 0o000)
    isolation_probe = (
        "import pathlib,sys\n"
        "gold=pathlib.Path(sys.argv[1]); manifest=pathlib.Path(sys.argv[2])\n"
        "assert manifest.open('rb').read(1)\n"
        "for op in (lambda: gold.open('rb').read(1), lambda: list(gold.parent.iterdir())):\n"
        "    try: op()\n"
        "    except PermissionError: continue\n"
        "    else: sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    check = subprocess.run(["/home/cn/miniconda3/envs/seraph_llm/bin/python", "-c",
                            isolation_probe, str(gold_final), str(manifest)],
                           capture_output=True, text=True)
    isolation_passed = check.returncode == 0
    dump(root / "GOLD_PHYSICAL_ISOLATION_TEST.json", {
        "status": "PASSED" if isolation_passed else "FAILED",
        "gold_path": str(gold_final), "gold_file_mode_octal": "000",
        "gold_directory_mode_octal": "000",
        "prediction_subprocess_open_and_list_gold_denied": isolation_passed,
        "prediction_subprocess_manifest_readable": isolation_passed,
        "caveat": "The owner can deliberately chmod the file later; this guards ordinary prediction code, not a malicious same-UID process.",
        "subprocess_exit": check.returncode,
        "gold_sha256_before_chmod": gold_hash})

    independent = len(final_ids["project"])
    primary_modules = {module: sum(row["module"] == module for row in records) for module in MODULES[:3]}
    ready = (independent >= 2 and balanced and isolation_passed and
             all(primary_modules.values()) and
             all(not any(data["overlap_counts"].values()) for data in leakage.values()))
    status = "READY_FOR_FINAL_FREEZE_CLEAN_V2" if ready else "BLOCKED_NEED_MORE_FINAL_TEST_DATA"
    status_obj = {
        "status": status, "independent_project_count": independent,
        "binary_count": len(final_ids["binary"]), "final_record_count": len(records),
        "module_records": {module: sum(row["module"] == module for row in records) for module in MODULES},
        "manifest_path": str(manifest), "manifest_sha256": sha(manifest),
        "gold_path": str(gold_final), "gold_record_count": len(gold_rows),
        "gold_sha256": gold_hash, "gold_file_mode_octal": "000",
        "gold_directory_mode_octal": "000",
        "gold_isolation_test_passed": isolation_passed,
        "historical_identity_overlap_count": 0,
        "candidate_position_balanced": balanced,
        "clean_hsym_coverage_complete": True,
        "stage2_checkpoint_load_count": 0, "stage2_prediction_count": 0,
        "performance_evaluation_count": 0,
        "source_commit_matches_binary_build": "UNVERIFIED_RESYM_METADATA_DOES_NOT_PIN_COMMIT",
        "final_model_freeze_executed": False,
    }
    dump(root / "FINAL_TEST_V2_FINAL_STATUS.json", status_obj)
    write_md(root / "FINAL_TEST_V2_BUILD_REPORT.md", [
        "# FINAL_TEST_V2 independent dataset build", "",
        f"Status: **{status}**", "",
        f"Independent projects: {independent}; binaries: {len(final_ids['binary'])}; records: {len(records)}.",
        "Projects: " + ", ".join(sorted(final_ids["project"])) + ".", "",
        "| Module | Final records | Mapping L0 | Mapping L2 | UNKNOWN |",
        "|---|---:|---:|---:|---:|",
        *[f"| {module} | {sum(row['module'] == module for row in records)} | "
          f"{levels[module]['L0']} | {levels[module]['L2']} | {levels[module]['UNKNOWN']} |"
          for module in MODULES], "",
        "Historical project/binary/function/candidate-instance overlaps: 0 in the inspected identity sources.",
        f"Candidate-position strata balanced: {balanced}.",
        f"Gold sealed SHA256: `{gold_hash}`; mode 000 subprocess read-denial: {isolation_passed}.",
        f"Predictor manifest SHA256: `{sha(manifest)}`; no Gold candidate ID/position field.",
        "Source commits and tree hashes are recorded, but ReSym metadata does not prove those commits produced the stored binaries.",
        "No Stage-2 checkpoint loaded, final-test prediction, performance evaluation, SSQ, calibration, or training was run.",
        "No final freeze was executed. Human approval is required before any later stage.",
    ])
    if not ready:
        write_md(root / "FINAL_TEST_V2_ADDITIONAL_DATA_REQUIREMENT.md", [
            "# Additional FINAL_TEST_V2 data required", "",
            f"Independent projects currently usable: {independent}; minimum: 2.",
            "Primary modules with no valid Gold-in-candidate records: " +
            (", ".join(module for module, n in primary_modules.items() if not n) or "none") + ".",
            "Do not lower the project firewall, alter Gold, change candidate membership using model scores, or use historical projects.",
        ])
    print(json.dumps(status_obj, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
