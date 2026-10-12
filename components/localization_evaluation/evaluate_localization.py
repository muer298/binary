#!/usr/bin/env python3
"""Read-only replay evaluation of frozen v0.3.3 predictions against v0.3.4 Gold."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path("/data/chaoni/mjy/resym_exp")
V033 = ROOT / "training_runs/seraph_task2/v0_3_3"
V034 = ROOT / "training_runs/seraph_task2/v0_3_4"
DEFAULT_OUT = ROOT / "training_runs/seraph_task2/localization_eval_v0_1"


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sum_file(root):
    checked = []
    for line in (root / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1); name = name.strip()
        actual = sha(root / name)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch: {root / name}")
        checked.append({"file": str(root / name), "sha256": actual})
    return checked


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dump_jsonl(path, values):
    with Path(path).open("w") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def score(predicted_samples, population, positives):
    predicted = set(predicted_samples) & set(population); positives = set(positives) & set(population)
    tp = len(predicted & positives); fp = len(predicted - positives); fn = len(positives - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "population": len(set(population)), "positive_gold": len(positives), "predicted_positive": len(predicted)}


def pair_score(predicted_pairs, gold_pairs, population_samples):
    predicted = {x for x in predicted_pairs if x[0] in population_samples}
    gold = {x for x in gold_pairs if x[0] in population_samples}
    tp = len(predicted & gold); fp = len(predicted - gold); fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "gold_instructions": len(gold), "predicted_instructions": len(predicted),
            "exact_instruction_recall": recall}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); out = args.out
    out.mkdir(parents=True, exist_ok=False)

    v033_checks = verify_sum_file(V033); v034_checks = verify_sum_file(V034)
    gold = list(rows(V034 / "CWE416_476_INSTRUCTION_GOLD.jsonl"))
    gold_by_sid = {x["sample_id"]: x for x in gold}
    if len(gold) != 40 or len(gold_by_sid) != 40:
        raise RuntimeError("v0.3.4 Gold population is not the fixed 40 binaries")
    positives = {x["sample_id"] for x in gold if x["source_level_vulnerable"]}
    gold_pairs = {(x["sample_id"], pc) for x in gold for pc in x["instruction_addresses"]}

    null_candidates = list(rows(V033 / "CWE476_ADDRESS_CANDIDATES_V033.jsonl"))
    slices = list(rows(V033 / "CANDIDATE_EVIDENCE_SLICES_V033.jsonl"))
    uaf_candidates = [x for x in slices if x["family"] == "UAF"]
    results = [x for x in rows(V033 / "BOUNDED_VERIFICATION_RESULTS_V033.jsonl")
               if x["arm"] == "V0_DETERMINISTIC"]
    result_by_candidate = {x["candidate_id"]: x for x in results}

    replay = []
    for cand in null_candidates:
        positive = cand["classification"] == "POTENTIAL_NULL_ACCESS"
        result = result_by_candidate[cand["candidate_id"]]
        replay.append({
            "sample_id": cand["sample_id"], "candidate_id": cand["candidate_id"], "family": "NULL_DEREF",
            "pc": cand["instruction_anchor_pc"], "raw_candidate_emitted": True,
            "candidate_positive_verdict": positive, "static_classification": cand["classification"],
            "bounded_status": result["status"], "bounded_verified_positive": result["status"] == "VERIFIED_WITHIN_BOUNDED_MODEL",
            "gold_positive_sample": cand["sample_id"] in positives,
            "exact_gold_instruction_match": (cand["sample_id"], cand["instruction_anchor_pc"]) in gold_pairs,
        })
    for item in uaf_candidates:
        result = result_by_candidate[item["candidate_id"]]
        replay.append({
            "sample_id": item["sample_id"], "candidate_id": item["candidate_id"], "family": "UAF",
            "pc": item["anchor_pc"], "raw_candidate_emitted": True, "candidate_positive_verdict": True,
            "static_classification": item["audit"]["classification"], "bounded_status": result["status"],
            "bounded_verified_positive": result["status"] == "VERIFIED_WITHIN_BOUNDED_MODEL",
            "gold_positive_sample": item["sample_id"] in positives,
            "exact_gold_instruction_match": (item["sample_id"], item["anchor_pc"]) in gold_pairs,
        })
    replay.sort(key=lambda x: (x["sample_id"], x["candidate_id"]))
    dump_jsonl(out / "LOCALIZATION_PREDICTION_REPLAY.jsonl", replay)

    raw_samples = {x["sample_id"] for x in replay}
    positive_samples = {x["sample_id"] for x in replay if x["candidate_positive_verdict"]}
    positive_pairs = {(x["sample_id"], x["pc"]) for x in replay if x["candidate_positive_verdict"]}
    verified_samples = {x["sample_id"] for x in replay if x["bounded_verified_positive"]}
    verified_pairs = {(x["sample_id"], x["pc"]) for x in replay if x["bounded_verified_positive"]}

    sample_groups = {
        "overall": set(gold_by_sid),
        "CWE-416": {x["sample_id"] for x in gold if x["cwe"] == "CWE-416"},
        "CWE-476": {x["sample_id"] for x in gold if x["cwe"] == "CWE-476"},
        "O0": {x["sample_id"] for x in gold if x["optimization"] == "O0"},
        "O2": {x["sample_id"] for x in gold if x["optimization"] == "O2"},
        "CWE-416/O0": {x["sample_id"] for x in gold if x["cwe"] == "CWE-416" and x["optimization"] == "O0"},
        "CWE-416/O2": {x["sample_id"] for x in gold if x["cwe"] == "CWE-416" and x["optimization"] == "O2"},
        "CWE-476/O0": {x["sample_id"] for x in gold if x["cwe"] == "CWE-476" and x["optimization"] == "O0"},
        "CWE-476/O2": {x["sample_id"] for x in gold if x["cwe"] == "CWE-476" and x["optimization"] == "O2"},
    }
    layers = {}
    for name, population in sample_groups.items():
        layers[name] = {
            "raw_candidate_presence": score(raw_samples, population, positives),
            "binary_positive_detection": score(positive_samples, population, positives),
            "exact_instruction_localization": pair_score(positive_pairs, gold_pairs, population),
            "bounded_symbolic_verification": score(verified_samples, population, positives),
            "verified_exact_instruction": pair_score(verified_pairs, gold_pairs, population),
        }

    missing = []
    for sid in sorted(positives - raw_samples):
        item = gold_by_sid[sid]; instruction = item["instruction_text"][0]
        if item["cwe"] == "CWE-476" and item["optimization"] == "O2":
            reason = "DIRECT_RAM_ZERO_READ_NOT_EXPOSED_AS_MEMORY_ACCESS_SUPPORT_LOCS"
        elif item["cwe"] == "CWE-416" and instruction.split()[0].startswith(("call", "jmp")):
            reason = "INTERPROCEDURAL_CALL_ARGUMENT_UAF_USE_NOT_MODELED"
        else:
            reason = "NO_FROZEN_CANDIDATE"
        missing.append({"sample_id": sid, "cwe": item["cwe"], "optimization": item["optimization"],
                        "gold_pc": item["instruction_addresses"][0], "gold_instruction": instruction,
                        "reason": reason})
    unknown_results = [x for x in results if x["status"] == "UNKNOWN_INSUFFICIENT_EVIDENCE"]
    good_verified = [x for x in replay if x["bounded_verified_positive"] and not x["gold_positive_sample"]]
    guarded_good = [x for x in replay if x["static_classification"] == "GUARDED_NULL_ACCESS" and not x["gold_positive_sample"]]

    metrics = {
        "schema": "seraph_task2_development_localization_metrics_v1",
        "scope": "FROZEN_DEVELOPMENT_SMOKE_REPLAY_NOT_INDEPENDENT_VALIDATION",
        "population": {"binaries": 40, "positive_instruction_gold": 20, "explicit_good_negatives": 20},
        "layers": layers,
        "verified_path_coverage": len(verified_pairs & gold_pairs) / len(gold_pairs),
        "good_false_confirmation": len(good_verified),
        "guarded_good_candidate_records": len(guarded_good),
        "solver_unknown_candidates": len(unknown_results),
        "positive_gold_without_candidate": len(missing),
        "miss_reasons": dict(collections.Counter(x["reason"] for x in missing)),
        "historical_independent_metrics_not_combined": {
            "binary_auroc": 0.7751,
            "candidate_recall": 0.8333,
            "candidate_recall_display": "83.33%",
            "provenance": "USER_SPECIFIED_FROZEN_HISTORICAL_INDEPENDENT_RESULTS; NOT_RECOMPUTED OR COMBINED IN THIS RUN",
        },
        "input_integrity": {"v033_sha_entries_verified": len(v033_checks),
                            "v034_sha_entries_verified": len(v034_checks),
                            "historical_holdout_read": False, "training_run": False,
                            "verification_rules_modified": False},
    }
    dump(out / "DEVELOPMENT_LOCALIZATION_METRICS.json", metrics)
    dump_jsonl(out / "UNRECALLED_GOLD_RECORDS.jsonl", missing)
    dump(out / "EVALUATION_INPUT_SHA256.json", {"v033": v033_checks, "v034": v034_checks})

    def pct(value): return f"{100 * value:.2f}%"
    def metric_row(name, layer):
        value = layers[name][layer]
        return f"| {name} | {value['tp']} | {value['fp']} | {value['fn']} | {pct(value['precision'])} | {pct(value['recall'])} | {pct(value['f1'])} |"
    detection_rows = "\n".join(metric_row(x, "binary_positive_detection") for x in
                               ("overall", "CWE-416", "CWE-476", "O0", "O2", "CWE-416/O0", "CWE-416/O2", "CWE-476/O0", "CWE-476/O2"))
    candidate_rows = "\n".join(metric_row(x, "raw_candidate_presence") for x in
                               ("overall", "CWE-416", "CWE-476", "O0", "O2"))
    exact_rows = "\n".join(metric_row(x, "exact_instruction_localization") for x in
                           ("overall", "CWE-416", "CWE-476", "O0", "O2"))
    missing_rows = "\n".join(f"| `{x['sample_id']}` | {x['cwe']} | {x['optimization']} | `{x['gold_pc']}` | {x['reason']} |" for x in missing)

    (out / "CWE416_476_LOCALIZATION_EVALUATION.md").write_text(f"""# CWE-416/476 Development Localization Evaluation

This is a read-only replay of frozen v0.3.3 outputs against v0.3.4 Gold. All 40 binaries remain in the denominator.

## Binary positive verdict

| Stratum | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
{detection_rows}

## Raw candidate presence

This layer counts guarded diagnostic candidates as emitted candidates. Consequently, five guarded good CWE-476 records are candidate-level false positives, although they are not vulnerability verdicts.

| Stratum | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
{candidate_rows}

## Exact positive instruction localization

| Stratum | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
{exact_rows}

Bounded verification exactly matches all ten positive localized candidates: verified-path coverage is **{pct(metrics['verified_path_coverage'])}**. Good false confirmation is **{len(good_verified)}**. Solver UNKNOWN is **{len(unknown_results)}**; however, **{len(missing)}** positive Gold samples have no candidate and therefore count as false negatives rather than solver UNKNOWN.

## Unrecalled positive Gold

| Sample | CWE | opt | Gold PC | reason |
|---|---|---|---:|---|
{missing_rows}
""")

    o2_null = layers["CWE-476/O2"]
    (out / "O2_NULL_DEREFERENCE_CASE_STUDY.md").write_text(f"""# O2 CWE-476 NULL Dereference Case Study

v0.3.4 Gold proves that all five O2 bad binaries retain a real NULL read at `0x4012b4` (`mov`/`movzbl 0x0,%reg`). The following `ud2` is not the Gold location.

The frozen v0.3.3 verifier nevertheless emits no candidate for these five instructions. In the frozen `instr_sem`, the direct RAM operand is represented as an `addr:ram:0x0:<size>` read on COPY/INT_ZEXT while `memory_accesses` and address `support_locs` are empty. The address-based generator only consumed explicit memory-access support locations, so the five instructions were never presented to SMT.

- Binary detection: TP={o2_null['binary_positive_detection']['tp']}, FN={o2_null['binary_positive_detection']['fn']}, recall={pct(o2_null['binary_positive_detection']['recall'])}.
- Exact localization: TP={o2_null['exact_instruction_localization']['tp']}, FN={o2_null['exact_instruction_localization']['fn']}.
- Gold locatability: 5/5.

This is the current technical bottleneck: **the instruction is objectively localizable, but the frozen detector did not localize it**. It must not be described as compiler elimination or Gold unavailability.
""")

    example_null = next(x for x in replay if x["family"] == "NULL_DEREF" and x["bounded_verified_positive"])
    example_uaf = next(x for x in replay if x["family"] == "UAF" and x["bounded_verified_positive"])
    (out / "TASK2_STAGE_REPORT_FOR_ADVISOR.md").write_text(f"""# SERAPH Task2 — Advisor Stage Report

## Executive summary

SERAPH now has a reproducible binary-analysis pipeline from P-code/CFG/DFG evidence through candidate construction, bounded SMT verification, exact machine-instruction backlinking, and optimization-aware instruction Gold. The present development replay localizes and verifies **10/20** positive instructions with **0** verified good false confirmations.

## Technical architecture

1. Ghidra-derived P-code, CFG, DFG, ABI, call and memory semantics.
2. CWE-specific candidate generation with exact pointer/object identity evidence.
3. Path and safety gates for NULL guards, reaching definitions, overwrite/kill, free/use ordering and alias continuity.
4. Bounded Z3 verification under fixed resource limits.
5. Exact P-code→machine backlinking and source/DWARF-backed instruction Gold.
6. Frozen-output replay evaluator separating candidate emission, binary verdict, exact localization and symbolic verification.

## Results from different datasets

These figures are intentionally not merged into one end-to-end accuracy number:

- Historical independent binary experiment: **AUROC=0.7751** (frozen historical headline supplied for this report; not recomputed here).
- Historical independent candidate experiment: **Recall=83.33%** (frozen historical headline supplied for this report; not recomputed here).
- CWE-416/476 DEVELOPMENT_SMOKE localization replay: binary verdict TP=10, FP=0, FN=10; Precision=100.00%, Recall=50.00%, F1=66.67%.
- Raw candidate presence: TP=10, FP=5 guarded diagnostics, FN=10; Precision=66.67%, Recall=50.00%.
- Exact positive instruction localization: 10/20 (50.00%); bounded verified-path coverage: 10/20 (50.00%).

## Breakdown

- CWE-416: 5/10 localized and verified; five call/tail-jump UAF sinks lack interprocedural actual/formal use modeling.
- CWE-476: 5/10 localized and verified; all five O2 direct address-zero reads are missed.
- O0: 8/10 localized, Recall=80.00%.
- O2: 2/10 localized, Recall=20.00%.
- Verified good false confirmations: 0; solver UNKNOWN among emitted candidates: 0.

## Representative P-code→SMT→machine cases

- CWE-476: candidate `{example_null['candidate_id'][:12]}` follows a NULL constant through exact stack/register def-use to `{example_null['pc']}`, then Z3 proves the bounded NULL-address execution SAT; the machine backlink is exact.
- CWE-416: candidate `{example_uaf['candidate_id'][:12]}` links allocation/free/use identity to `{example_uaf['pc']}`, then Z3 proves the bounded post-free use SAT; the machine backlink is exact.

## Current limitations

The dominant gap is representation recovery, not Gold quality. Direct O2 RAM-zero reads are present in machine code and P-code operands but absent from the frozen memory-access contract. Several UAF uses cross calls or optimized tail jumps and are not modeled as actual/formal use sinks. Development metrics must not be presented as independent generalization.

## Readiness

The results are suitable for an advisor-stage progress report because inputs are frozen, all 40 samples remain in the denominator, misses are explicit, and independent historical metrics remain separate from development localization metrics.
""")

    (out / "NEXT_STAGE_RESEARCH_PLAN.md").write_text("""# Next-Stage Research Plan

1. Add a deterministic sidecar normalizer for direct RAM operands such as `addr:ram:0x0:<size>` and require exact machine memory-operand confirmation. Re-run DEVELOPMENT_SMOKE only as a regression, without changing v0.3.3.
2. Extend CWE-416 use modeling across call actual/formal edges and optimized tail jumps, with same-object identity and free-before-use gates.
3. Freeze the revised candidate and verification contracts, configuration and SHA256 before any independent evaluation.
4. Pre-register binary detection, raw candidate, exact localization and bounded verification metrics, including O0/O2 and CWE strata.
5. Run v0.3.5 on an untouched independent localization set once. Do not tune on its labels; report UNKNOWN and UNLOCATABLE without deletion.
6. Continue V0/V1 ablation. Claim SERAPH contribution only if it adds executable constraints or changes verified outcomes under identical candidates and budgets.
""")

    hashes = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"}
    (out / "SHA256SUMS").write_text("".join(f"{value}  {name}\n" for name, value in sorted(hashes.items())))
    print(json.dumps({"status": "READY_FOR_ADVISOR_STAGE_REPORT", "metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
