"""Gold-free candidate-identity permutation robustness evaluation.

Consumes only the frozen DESIGN-order manifest and its saved API responses.
It does not read HOLDOUT Gold, CALIBRATION, or FINAL TEST artifacts.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys


PKG_C = Path("/data/chaoni/mjy/resym_exp/packages/seraph_external_ssq_v0_4_0c_design")
sys.path.insert(0, str(PKG_C))
import external_ssq_design_eval as design_eval  # noqa: E402
import external_ssq_order_client as client  # noqa: E402

OUT = client.OUT
PRIMARY = "FIVE_DISTINCT_ORDERS"
TWO = "TWO_ORDER_EXHAUSTIVE_DIAGNOSTIC"


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def write_text_new(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def write_lines_new(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def spearman_by_identity(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or set(a) != set(b) or len(a) < 2:
        raise RuntimeError("ranked candidate identities differ")
    n = len(a)
    pos_a = {candidate: i + 1 for i, candidate in enumerate(a)}
    pos_b = {candidate: i + 1 for i, candidate in enumerate(b)}
    return 1 - 6 * sum((pos_a[c] - pos_b[c]) ** 2 for c in pos_a) / (n * (n * n - 1))


def measure_record(record: dict, attempts: dict[str, dict]) -> dict:
    order_results = []
    canonical_ids = record["original_candidate_ids"]
    for order in record["orders"]:
        item = attempts[order["variant_id"]]
        valid = bool(item["valid"])
        abstain = bool(item["parsed"]["abstain"]) if valid else False
        normalized = None
        ranked = None
        winner = None
        tie = None
        if valid:
            normalized = design_eval.normalized_scores(item["parsed"]["scores"], canonical_ids)
            if not abstain:
                maximum = max(normalized.values())
                tie = sum(abs(v - maximum) <= 1e-12 for v in normalized.values()) > 1
                # A tied top score has no unique winner. Never turn a tie into
                # apparent order stability via the canonical-ID tie-breaker.
                if not tie:
                    ranked = design_eval.rank(normalized, canonical_ids)
                    winner = ranked[0]
        order_results.append({
            "order_index": order["order_index"],
            "variant_id": order["variant_id"],
            "candidate_ids_in_presentation_order": order["candidate_ids"],
            "valid": valid,
            "abstain": abstain,
            "winner_candidate_id": winner,
            "ranked_candidate_ids": ranked,
            "normalized_scores_by_candidate_id": normalized,
            "top_score_tie": tie,
        })
    comparisons = list(itertools.combinations(order_results, 2))
    agreements = [int(a["winner_candidate_id"] is not None and
                      a["winner_candidate_id"] == b["winner_candidate_id"])
                  for a, b in comparisons]
    correlations = [spearman_by_identity(a["ranked_candidate_ids"], b["ranked_candidate_ids"])
                    if a["ranked_candidate_ids"] is not None and b["ranked_candidate_ids"] is not None
                    else 0.0 for a, b in comparisons]
    winners = [r["winner_candidate_id"] for r in order_results]
    perfect = all(w is not None and w == winners[0] for w in winners)
    score_std = []
    score_ranges = []
    for candidate in canonical_ids:
        values = [r["normalized_scores_by_candidate_id"][candidate] for r in order_results
                  if r["normalized_scores_by_candidate_id"] is not None and not r["abstain"]]
        if len(values) >= 2:
            score_std.append(statistics.pstdev(values))
            score_ranges.append(max(values) - min(values))
    return {
        "record_id": record["record_id"],
        "module": record["module"],
        "cohort": record["cohort"],
        "candidate_count": record["candidate_count"],
        "order_count": len(order_results),
        "valid_count": sum(r["valid"] for r in order_results),
        "abstain_count": sum(r["abstain"] for r in order_results),
        "top1_identity_agreement": statistics.mean(agreements),
        "winner_flip_rate": 1 - statistics.mean(agreements),
        "perfect_agreement": perfect,
        "mean_pairwise_spearman_rank_consistency": statistics.mean(correlations),
        "mean_candidate_normalized_score_std": statistics.mean(score_std) if score_std else None,
        "mean_candidate_normalized_score_range": statistics.mean(score_ranges) if score_ranges else None,
        "top_score_ties": sum(bool(r["top_score_tie"]) for r in order_results),
        "orders": order_results,
    }


def summarize(cases: list[dict], attempts: dict[str, dict]) -> dict:
    if not cases:
        raise RuntimeError("empty robustness cohort")
    variants = [o["variant_id"] for case in cases for o in case["orders"]]
    items = [attempts[v] for v in variants]
    all_attempts = [attempt for item in items for attempt in item["attempts"]]
    parseable = [attempt for attempt in all_attempts if attempt.get("parse_valid")]
    valid = sum(bool(item["valid"]) for item in items)
    complete = sum(bool(a.get("candidate_completeness")) for a in parseable)
    legal = sum(bool(a.get("candidate_legality")) for a in parseable)
    invented = sum(bool(a.get("invented_candidate")) for a in all_attempts)
    candidate_ok = bool(parseable) and complete == legal == len(parseable) and invented == 0
    values_std = [case["mean_candidate_normalized_score_std"] for case in cases
                  if case["mean_candidate_normalized_score_std"] is not None]
    values_range = [case["mean_candidate_normalized_score_range"] for case in cases
                    if case["mean_candidate_normalized_score_range"] is not None]
    mean_agreement = statistics.mean(case["top1_identity_agreement"] for case in cases)
    return {
        "records": len(cases),
        "request_variants": len(items),
        "api_attempts": len(all_attempts),
        "valid_final_responses": valid,
        "valid_structured_rate": valid / len(items),
        "parseable_final_responses": sum(bool(item["attempts"][-1].get("parse_valid")) for item in items),
        "parse_validity_rate": sum(bool(item["attempts"][-1].get("parse_valid")) for item in items) / len(items),
        "parseable_attempts": len(parseable),
        "candidate_complete_parseable_attempts": complete,
        "candidate_legal_parseable_attempts": legal,
        "invented_candidate_attempts": invented,
        "candidate_legality_and_completeness_100_percent": candidate_ok,
        "abstain_final_responses": sum(bool(item["valid"] and item["parsed"]["abstain"]) for item in items),
        "top_score_tie_final_responses": sum(case["top_score_ties"] for case in cases),
        "mean_top1_identity_agreement": mean_agreement,
        "perfect_agreement_record_rate": sum(case["perfect_agreement"] for case in cases) / len(cases),
        "winner_flip_rate": 1 - mean_agreement,
        "mean_rank_consistency": statistics.mean(case["mean_pairwise_spearman_rank_consistency"] for case in cases),
        "mean_candidate_normalized_score_std": statistics.mean(values_std) if values_std else None,
        "mean_candidate_normalized_score_range": statistics.mean(values_range) if values_range else None,
        "retry_count": sum(item["retry_count"] for item in items),
        "timeout_count": sum(bool(a.get("timeout")) for a in all_attempts),
    }


def pass_gate(metrics: dict) -> tuple[bool, list[str]]:
    reasons = []
    if not metrics["candidate_legality_and_completeness_100_percent"]:
        reasons.append("CANDIDATE_LEGALITY_OR_COMPLETENESS_FAILED")
    if metrics["valid_structured_rate"] < 0.95:
        reasons.append("VALID_STRUCTURED_RATE_BELOW_95_PERCENT")
    if metrics["mean_top1_identity_agreement"] < 0.90:
        reasons.append("TOP1_IDENTITY_AGREEMENT_BELOW_90_PERCENT")
    if metrics["winner_flip_rate"] > 0.10 + 1e-12:
        reasons.append("WINNER_FLIP_RATE_ABOVE_10_PERCENT")
    if metrics["mean_rank_consistency"] < 0.90:
        reasons.append("MEAN_RANK_CONSISTENCY_BELOW_90_PERCENT")
    return not reasons, reasons


def resources(attempts: dict[str, dict]) -> dict:
    usage = collections.Counter()
    all_items = list(attempts.values())
    for item in all_items:
        for a in item["attempts"]:
            u = a.get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                        "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
                usage[key] += int(u.get(key) or 0)
            usage["reasoning_tokens"] += int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    offpeak = (usage["prompt_cache_hit_tokens"] * 0.003 +
               usage["prompt_cache_miss_tokens"] * 0.15 +
               usage["completion_tokens"] * 0.6) / 1_000_000
    return {
        "request_variants": len(all_items),
        "api_attempt_count": sum(len(item["attempts"]) for item in all_items),
        "retry_count": sum(item["retry_count"] for item in all_items),
        "timeout_count": sum(bool(a.get("timeout")) for item in all_items for a in item["attempts"]),
        "summed_request_runtime_seconds": round(sum(float(item["elapsed_seconds"]) for item in all_items), 3),
        "runtime_definition": "Sum of per-request wall times, including retries/backoff; no model training",
        "token_usage_all_attempts": dict(usage),
        "estimated_cost_USD_offpeak": round(offpeak, 6),
        "estimated_cost_USD_peak": round(offpeak * 2, 6),
        "cost_source": "https://api-docs.deepseek.com/quick_start/pricing/",
        "cost_note": "deepseek-flash USD rates checked 2026-10-07; NOT BILLING RECORD",
    }


def main() -> None:
    manifest = client.frozen_manifest()
    expected = {order["variant_id"]: (record, order) for record in manifest for order in record["orders"]}
    paths = list((OUT / "attempts").glob("*.json"))
    if len(expected) != len(paths) or len(paths) != 520 or {p.stem for p in paths} != set(expected):
        raise RuntimeError("order audit API response coverage incomplete or has extra variants")
    attempts = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in paths}
    for variant, item in attempts.items():
        record, order = expected[variant]
        if (item["variant_id"] != variant or item["record_id"] != record["record_id"] or
                item["module"] != record["module"] or item["cohort"] != record["cohort"] or
                item["order_index"] != order["order_index"] or item["candidate_ids"] != order["candidate_ids"] or
                item["request_sha256"] != order["request_sha256"] or
                item["prompt_sha256"] != client.fixed.PROMPT_SHA):
            raise RuntimeError(f"saved response/request mismatch: {variant}")
    outputs = (
        "V0_4_0E_EXTERNAL_SSQ_ORDER_ROBUSTNESS_REPORT.md",
        "EXTERNAL_SSQ_ORDER_ROBUSTNESS_METRICS.json", "EXTERNAL_SSQ_ORDER_CASES.jsonl",
        "EXTERNAL_SSQ_FINAL_MODULE_CONTRACT.json", "v0_4_0e_final_status.json",
    )
    if any((OUT / name).exists() for name in outputs):
        raise RuntimeError("order audit result already exists; refusing to overwrite")

    cases = [measure_record(record, attempts) for record in manifest]
    interface = [case for case in cases if case["module"] == "InterfaceSem"]
    layout = [case for case in cases if case["module"] == "LayoutSem"]
    layout_five = [case for case in layout if case["cohort"] == PRIMARY]
    layout_two = [case for case in layout if case["cohort"] == TWO]
    if (len(interface), len(layout), len(layout_five), len(layout_two)) != (28, 100, 60, 40):
        raise RuntimeError("order audit cohort drift")
    summaries = {
        "InterfaceSem_ALL": summarize(interface, attempts),
        "LayoutSem_ALL": summarize(layout, attempts),
        "LayoutSem_FIVE_DISTINCT_ORDERS": summarize(layout_five, attempts),
        "LayoutSem_TWO_ORDER_EXHAUSTIVE_DIAGNOSTIC": summarize(layout_two, attempts),
    }
    gate_results = {name: {"pass": p, "reasons": reasons}
                    for name, metrics in summaries.items()
                    for p, reasons in [pass_gate(metrics)]}
    interface_pass = gate_results["InterfaceSem_ALL"]["pass"]
    layout_pass = all(gate_results[name]["pass"] for name in (
        "LayoutSem_ALL", "LayoutSem_FIVE_DISTINCT_ORDERS",
        "LayoutSem_TWO_ORDER_EXHAUSTIVE_DIAGNOSTIC"))
    final_modules = {
        "TypeSem": {"External_SSQ": "DISABLED", "reason": "FROZEN_DESIGN_POLICY_DISABLED; no API call in this audit"},
        "InterfaceSem": {"External_SSQ": "ENABLED" if interface_pass else "DISABLED_ORDER_UNSTABLE",
                         "reason": gate_results["InterfaceSem_ALL"]["reasons"]},
        "LayoutSem": {"External_SSQ": "ENABLED" if layout_pass else "DISABLED_ORDER_UNSTABLE",
                      "reason": {k: gate_results[k]["reasons"] for k in (
                          "LayoutSem_ALL", "LayoutSem_FIVE_DISTINCT_ORDERS",
                          "LayoutSem_TWO_ORDER_EXHAUSTIVE_DIAGNOSTIC")}},
    }
    status = ("READY_FOR_EXTERNAL_SSQ_CALIBRATION" if interface_pass or layout_pass
              else "BLOCKED_EXTERNAL_SSQ_ORDER_ROBUSTNESS")
    usage = resources(attempts)
    result = {
        "schema": "seraph_external_ssq_order_robustness_metrics_v0_4_0e",
        "status": status,
        "manifest_sha256": client.sha(OUT / "EXTERNAL_SSQ_ORDER_AUDIT_MANIFEST.jsonl"),
        "seed": client.SEED,
        "records": 128,
        "module_and_cohort_metrics": summaries,
        "pre_registered_gate_results": gate_results,
        "resources": usage,
        "HOLDOUT_Gold_read": False,
        "CALIBRATION_opened": False,
        "FINAL_TEST_V2_opened": False,
        "FINAL_TEST_V3_opened": False,
    }
    design_metrics = json.loads((client.DESIGN / "EXTERNAL_SSQ_CONTEXT_STRATIFIED_METRICS.json").read_text(encoding="utf-8"))
    holdout_decision = json.loads((client.HOLDOUT / "EXTERNAL_SSQ_HOLDOUT_DECISION.json").read_text(encoding="utf-8"))
    historical = {
        module: {
            "DESIGN_ALL_net_correction": design_metrics["metrics"][module]["ALL"]["net_correction"],
            "HOLDOUT_External_SSQ": holdout_decision["modules"][module]["External_SSQ"],
            "HOLDOUT_evidence": holdout_decision["modules"][module]["HOLDOUT_evidence"],
        }
        for module in ("TypeSem", "InterfaceSem", "LayoutSem")
    }
    contract = {
        "schema": "seraph_external_ssq_final_module_contract_v0_4_0e",
        "status": status,
        "design_module_policy_sha256": client.DESIGN_POLICY_SHA,
        "holdout_decision_sha256": client.HOLDOUT_DECISION_SHA,
        "design_context_fallback_contract_sha256": client.CONTEXT_SHA,
        "order_manifest_sha256": result["manifest_sha256"],
        "module_policy": {module: {**final_modules[module],
                                  "DESIGN_and_HOLDOUT_evidence": historical[module],
                                  "order_robustness_evidence": summaries["InterfaceSem_ALL"] if module == "InterfaceSem"
                                  else {k: summaries[k] for k in summaries if k.startswith("LayoutSem")}
                                  if module == "LayoutSem" else None}
                          for module in ("TypeSem", "InterfaceSem", "LayoutSem")},
        "LayoutSem_fallback_limit": "HOLDOUT fallback n=0; this audit uses verified-local LayoutSem only, so function-level fallback remains unvalidated",
        "CALIBRATION_opened": False,
        "FINAL_TEST_V2_opened": False,
        "FINAL_TEST_V3_opened": False,
    }
    final_status = {
        "status": status,
        "InterfaceSem_External_SSQ": final_modules["InterfaceSem"]["External_SSQ"],
        "LayoutSem_External_SSQ": final_modules["LayoutSem"]["External_SSQ"],
        "TypeSem_External_SSQ": "DISABLED",
        "manifest_sha256": result["manifest_sha256"],
        "CALIBRATION_opened": False,
        "FINAL_TEST_V2_opened": False,
        "FINAL_TEST_V3_opened": False,
    }
    table = [
        "| Cohort | n | Variants | Valid | Top1 agreement | Perfect record rate | Winner flip | Mean Spearman | Score std | Candidate legality |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in summaries.items():
        table.append(f"| {name} | {m['records']} | {m['request_variants']} | {m['valid_structured_rate']:.4f} | {m['mean_top1_identity_agreement']:.4f} | {m['perfect_agreement_record_rate']:.4f} | {m['winner_flip_rate']:.4f} | {m['mean_rank_consistency']:.4f} | {m['mean_candidate_normalized_score_std']} | {m['candidate_legality_and_completeness_100_percent']} |")
    report = [
        "# v0_4_0e External SSQ candidate-order robustness audit", "",
        f"Final status: **{status}**. No HOLDOUT Gold, CALIBRATION, or FINAL TEST data were opened.",
        f"Manifest SHA256: `{result['manifest_sha256']}`; fixed seed `{client.SEED}`; 128 DESIGN records and 520 pre-registered request variants.",
        "Sampling used module/record_id/fixed seed within structural eligibility cohorts, never Gold, API correctness or Stage2 correctness. InterfaceSem uses all 28 DESIGN records. LayoutSem uses 60 verified-local records with at least three candidates for five distinct orders, plus 40 two-candidate records with both mathematically possible orders as a separate exhaustive diagnostic.",
        "Only the candidate array order changed. Query, code, evidence, constraints, candidate IDs/text, prompt and API configuration remained frozen. Scores/ranks/winners were compared by candidate identity, not presentation position.",
        "", "## Module and cohort metrics", "", *table, "",
        "Top1 agreement is the mean, over records, of the fraction of all unordered order pairs with the same unique, non-abstaining winner identity. A tied top score has no unique winner and contributes disagreement; its rank pair contributes zero, as do invalid/abstain pairs. Winner flip is the complement. Mean Spearman is the mean of identity-mapped pairwise rank correlations. Score std is the mean within-candidate normalized-score dispersion across distinct orders.",
        "", "## Frozen gates and final module policy", "",
    ]
    for name, gate in gate_results.items():
        report.append(f"- {name}: pass={gate['pass']}; reasons={gate['reasons']}.")
    for module in ("TypeSem", "InterfaceSem", "LayoutSem"):
        report.append(f"- {module}: External SSQ={final_modules[module]['External_SSQ']}; reason={final_modules[module]['reason']}.")
    report += [
        "", "The two-candidate cohort cannot have five distinct orders; its two-order results are not misrepresented as five-order evidence. A successful order audit checks operational stability, not semantic correctness or calibration benefit. The frozen HOLDOUT performance result remains historical and was not recomputed.",
        "", "## Resource accounting", "",
        f"Requests={usage['request_variants']}; API attempts={usage['api_attempt_count']}; retries={usage['retry_count']}; timeouts={usage['timeout_count']}; summed request runtime={usage['summed_request_runtime_seconds']:.3f}s.",
        f"Prompt tokens={usage['token_usage_all_attempts'].get('prompt_tokens', 0)}; completion tokens={usage['token_usage_all_attempts'].get('completion_tokens', 0)}; reasoning tokens={usage['token_usage_all_attempts'].get('reasoning_tokens', 0)}.",
        f"Estimated cost USD off-peak={usage['estimated_cost_USD_offpeak']:.6f}; peak={usage['estimated_cost_USD_peak']:.6f}. NOT BILLING RECORD.",
        "", "No TypeSem API call, no CALIBRATION, no FINAL_TEST_V2/V3, no Stage1/Stage2 training, no prompt or threshold search. Stop after this audit.", "",
    ]
    write_lines_new(OUT / "EXTERNAL_SSQ_ORDER_CASES.jsonl", cases)
    write_new(OUT / "EXTERNAL_SSQ_ORDER_ROBUSTNESS_METRICS.json", result)
    write_new(OUT / "EXTERNAL_SSQ_FINAL_MODULE_CONTRACT.json", contract)
    write_new(OUT / "v0_4_0e_final_status.json", final_status)
    write_text_new(OUT / "V0_4_0E_EXTERNAL_SSQ_ORDER_ROBUSTNESS_REPORT.md", "\n".join(report))
    print(json.dumps({"status": status,
                      "InterfaceSem": final_modules["InterfaceSem"]["External_SSQ"],
                      "LayoutSem": final_modules["LayoutSem"]["External_SSQ"],
                      "manifest_sha256": result["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
