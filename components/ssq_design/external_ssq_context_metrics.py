"""DESIGN-only context-stratified analysis for the frozen External SSQ branch.

Run only after all 628 requests and the original DESIGN evaluator complete.
Never reads HOLDOUT Gold, CALIBRATION or any FINAL TEST artifact.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import external_ssq_design_client as design
from external_ssq_design_eval import metrics_for_module


OUT = design.OUT
CONTRACT = OUT / "EXTERNAL_SSQ_CONTEXT_FALLBACK_CONTRACT.json"
CONTRACT_DIGEST = OUT / "EXTERNAL_SSQ_CONTEXT_FALLBACK_CONTRACT.sha256"
CASE_FILE = OUT / "EXTERNAL_SSQ_DESIGN_CASE_METRICS.jsonl"
ALL = "ALL"
LOCAL = "VERIFIED_LOCAL_ONLY"
FALLBACK = "FALLBACK_ONLY"
MODULES = ("TypeSem", "InterfaceSem", "LayoutSem")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def show(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def engineering(attempts: list[dict], original_pass: bool) -> dict:
    parseable = complete = legal = invented = 0
    for result in attempts:
        for attempt in result["attempts"]:
            if attempt.get("parse_valid"):
                parseable += 1
                complete += bool(attempt.get("candidate_completeness"))
                legal += bool(attempt.get("candidate_legality"))
            invented += bool(attempt.get("invented_candidate"))
    valid = sum(bool(r["valid"]) for r in attempts)
    n = len(attempts)
    candidate_ok = parseable > 0 and complete == parseable and legal == parseable and invented == 0
    return {
        "n": n,
        "valid": valid,
        "valid_rate": valid / n,
        "parseable_attempts": parseable,
        "complete_parseable_attempts": complete,
        "legal_parseable_attempts": legal,
        "invented_candidate_attempts": invented,
        "candidate_legality_100_percent": candidate_ok,
        "original_design_engineering_pass": bool(original_pass),
        "engineering_pass": valid / n >= 0.95 and candidate_ok and bool(original_pass),
    }


def get_strata(module: str, cases: list[dict], context_by_id: dict[str, dict]) -> dict[str, list[dict]]:
    own = [r for r in cases if r["module"] == module]
    result = {ALL: own}
    if module in ("TypeSem", "LayoutSem"):
        result[LOCAL] = [r for r in own if context_by_id[r["record_id"]]["context_mode"] == "VERIFIED_LOCAL"]
        result[FALLBACK] = [r for r in own if context_by_id[r["record_id"]]["context_mode"] == "FUNCTION_LEVEL_FALLBACK"]
    return result


def decide(module: str, strata: dict[str, dict], eng: dict) -> dict:
    overall = strata[ALL]
    reasons = []
    context_effect = "CONSISTENT_OR_NOT_APPLICABLE"
    if overall["net_correction"] <= 0:
        reasons.append("ALL_NET_CORRECTION_NOT_POSITIVE")
    if not eng["engineering_pass"]:
        reasons.append("API_ENGINEERING_OR_CANDIDATE_LEGALITY_FAILED")
    if module in ("TypeSem", "LayoutSem"):
        local = strata[LOCAL]
        local_harm = (local["net_correction"] < 0 or
                      local["api_raw"]["R@1"] < local["stage2"]["R@1"] or
                      local["api_raw"]["MRR"] < local["stage2"]["MRR"])
        if local_harm:
            reasons.append("VERIFIED_LOCAL_ONLY_NEGATIVE")
            if overall["net_correction"] > 0:
                context_effect = "CONTEXT_EFFECT_INCONSISTENT"
    return {
        "module": module,
        "enabled_for_future_HOLDOUT": not reasons,
        "context_effect": context_effect,
        "reasons": reasons,
        "selection_basis": "Pre-registered context contract and DESIGN metrics only; no HOLDOUT or TEST",
    }


def main() -> None:
    if CONTRACT_DIGEST.read_text(encoding="utf-8").strip() != f"{sha(CONTRACT)}  {CONTRACT.name}":
        raise RuntimeError("frozen fallback contract SHA256 mismatch")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["status"] != "FROZEN_BEFORE_API" or contract["design_records"] != 628:
        raise RuntimeError("fallback contract status/count drift")
    if sha(design.OLD / "EXTERNAL_SSQ_API_REQUESTS_V2.jsonl") != contract["request_file_sha256"]:
        raise RuntimeError("frozen request file changed")
    if sha(OUT / "DESIGN_PREFLIGHT.json") != contract["design_preflight_sha256"]:
        raise RuntimeError("DESIGN preflight changed")
    for path, digest in contract["source_manifest_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"DESIGN source manifest changed: {path}")

    contexts = contract["records"]
    by_id = {r["record_id"]: r for r in contexts}
    if len(by_id) != 628:
        raise RuntimeError("contract record IDs not unique")
    cases = read_jsonl(CASE_FILE)
    if len(cases) != 628 or {r["record_id"] for r in cases} != set(by_id):
        raise RuntimeError("DESIGN case coverage mismatch")
    if any(r["module"] != by_id[r["record_id"]]["module"] for r in cases):
        raise RuntimeError("DESIGN case module mismatch")
    originals = json.loads((OUT / "EXTERNAL_SSQ_DESIGN_METRICS.json").read_text(encoding="utf-8"))
    attempts_dir = OUT / "attempts"
    attempt_paths = list(attempts_dir.glob("*.json"))
    if len(attempt_paths) != 628 or {p.stem for p in attempt_paths} != set(by_id):
        raise RuntimeError("628/628 DESIGN attempts not complete")
    attempts = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in attempt_paths}
    for rid, item in attempts.items():
        if (item["module"] != by_id[rid]["module"] or
                item["request_sha256"] != by_id[rid]["request_sha256"]):
            raise RuntimeError(f"attempt/request contract drift: {rid}")

    strata_metrics = {}
    engineering_metrics = {}
    decisions = {}
    for module in MODULES:
        cohorts = get_strata(module, cases, by_id)
        if any(not rows for rows in cohorts.values()):
            raise RuntimeError(f"empty required {module} context stratum")
        measured = {name: metrics_for_module(rows) for name, rows in cohorts.items()}
        if measured[ALL] != originals["module_metrics"][module]:
            raise RuntimeError(f"overall metric reproduction failed: {module}")
        strata_metrics[module] = measured
        own_attempts = [attempts[r["record_id"]] for r in cohorts[ALL]]
        engineering_metrics[module] = engineering(own_attempts, originals["module_engineering_pass"][module])
        decisions[module] = decide(module, measured, engineering_metrics[module])

    global_candidate_legal = all(x["candidate_legality_100_percent"] for x in engineering_metrics.values())
    ready = global_candidate_legal and any(x["enabled_for_future_HOLDOUT"] for x in decisions.values())
    status = "READY_FOR_EXTERNAL_SSQ_HOLDOUT" if ready else "BLOCKED_EXTERNAL_SSQ_PERFORMANCE"
    report = {
        "schema": "seraph_external_ssq_design_context_stratified_metrics_v0_4_0c",
        "status": status,
        "contract_sha256": sha(CONTRACT),
        "design_records": 628,
        "type_fallback": 18,
        "layout_fallback": 29,
        "interface_function_context": 28,
        "abstain_invalid_policy": "API abstain/invalid contributes 0 to raw R@1/MRR; paired correction/harm excludes these and reports them separately",
        "metrics": strata_metrics,
        "engineering": engineering_metrics,
        "module_decisions": decisions,
        "HOLDOUT_api_requests": 0,
        "HOLDOUT_evaluations": 0,
        "CALIBRATION_opened": 0,
        "FINAL_TEST_opened": 0,
    }

    table = [
        "| Module | Cohort | n | Stage2 R@1 | API R@1 | Stage2 MRR | API MRR | W→C | C→W | Both correct | Both wrong | Net | Rescue precision | Rescue coverage | Harm rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for module in MODULES:
        for cohort, v in strata_metrics[module].items():
            table.append("| " + " | ".join(map(str, (
                module, cohort, v["n"], show(v["stage2"]["R@1"]), show(v["api_raw"]["R@1"]),
                show(v["stage2"]["MRR"]), show(v["api_raw"]["MRR"]), v["wrong_to_correct"],
                v["correct_to_wrong"], v["both_correct"], v["both_wrong"], v["net_correction"],
                show(v["rescue_precision"]), show(v["rescue_coverage"]), show(v["harm_rate"]),
            ))) + " |")

    analysis_lines = [
        "# External SSQ DESIGN fallback analysis", "",
        "The context contract was frozen before any API request and SHA256-checked on analysis.",
        "TypeSem fallback = 18; LayoutSem fallback = 29. These 47 records lack a *usable* verified local instruction mapping and retain the original function-level pseudo-C request. InterfaceSem's 28 function-level records are its primary contract, not part of the 47 exceptional fallbacks.",
        "A mapping status of VERIFIED alone is insufficient if instruction_usable is false; the original request scope and content are authoritative. No nearest-address, nearest-line, string similarity or unverified local span is introduced.", "",
        "Paired correction/harm categories exclude explicit abstain and invalid API outputs. Raw API retrieval metrics count them as zero, so this table also reports their counts in the JSON artifact.", "",
        *table, "",
        "## Per-module decisions", "",
    ]
    for module, decision in decisions.items():
        analysis_lines.append(f"- {module}: enabled={decision['enabled_for_future_HOLDOUT']}; context_effect={decision['context_effect']}; reasons={decision['reasons']}.")
    analysis_lines += ["", "Fallback performance did not alter the frozen mapping threshold or request policy. HOLDOUT has not been run.", ""]

    report_lines = [
        "# v0_4_0c External SSQ DESIGN report", "",
        f"Status: **{status}**. This is DESIGN only; HOLDOUT, CALIBRATION and FINAL TEST remain unopened in this branch.",
        f"Frozen context contract SHA256: `{sha(CONTRACT)}`. DESIGN requests: 628/628. TypeSem fallback: 18; LayoutSem fallback: 29; InterfaceSem primary function-level: 28.",
        "The 47 TypeSem/LayoutSem fallbacks have no usable verified local mapping and were sent with their original function-level pseudo-C; they are **not** instruction-local samples.", "",
        "## Context-stratified retrieval and correction/harm", "", *table, "",
        "## Engineering and policy", "",
    ]
    for module in MODULES:
        e = engineering_metrics[module]
        d = decisions[module]
        report_lines.append(f"- {module}: valid={e['valid']}/{e['n']}; parseable attempts={e['parseable_attempts']}; complete={e['complete_parseable_attempts']}; legal={e['legal_parseable_attempts']}; invented={e['invented_candidate_attempts']}; engineering_pass={e['engineering_pass']}; enabled={d['enabled_for_future_HOLDOUT']}; context_effect={d['context_effect']}; reasons={d['reasons']}.")
    report_lines += ["", "The pre-registered module gate requires positive ALL net correction, valid engineering and 100% candidate legality; TypeSem/LayoutSem additionally require conservative no-harm on VERIFIED_LOCAL_ONLY. A positive ALL result cannot override negative local-context evidence. No fallback threshold, prompt, candidate, Gold or request was changed after seeing DESIGN metrics.", "", "No HOLDOUT was executed. Future HOLDOUT may use only enabled modules and exactly the frozen context/API/prompt contract, subject to separate authorization.", ""]

    output_map = {
        "EXTERNAL_SSQ_CONTEXT_STRATIFIED_METRICS.json": json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        "EXTERNAL_SSQ_FALLBACK_ANALYSIS.md": "\n".join(analysis_lines),
        "V0_4_0C_EXTERNAL_SSQ_DESIGN_REPORT.md": "\n".join(report_lines),
        "EXTERNAL_SSQ_MODULE_POLICY.json": json.dumps({"schema": "seraph_external_ssq_design_module_policy_v0_4_0c", "status": status, "contract_sha256": sha(CONTRACT), "modules": decisions, "HOLDOUT_executed": False}, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        "V0_4_0C_FINAL_STATUS.json": json.dumps({"status": status, "design_records": 628, "contract_sha256": sha(CONTRACT), "HOLDOUT_executed": False, "CALIBRATION_opened": False, "FINAL_TEST_opened": False}, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    }
    for name in output_map:
        if (OUT / name).exists():
            raise FileExistsError(f"refusing to overwrite existing result: {name}")
    for name, body in output_map.items():
        write_exclusive(OUT / name, body)
    print(json.dumps({"status": status, "module_enabled": {m: d["enabled_for_future_HOLDOUT"] for m, d in decisions.items()}, "contract_sha256": sha(CONTRACT), "HOLDOUT_executed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
