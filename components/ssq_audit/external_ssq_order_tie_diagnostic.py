"""Gold-free interpretation of pre-frozen order-audit tie handling.

This is an explanatory addendum only. It does not modify gates, module policy,
frozen requests, or any API result.
"""

from __future__ import annotations

import collections
import itertools
import json
from pathlib import Path

import external_ssq_order_client as client
import external_ssq_order_eval as evaluator


OUT = client.OUT


def cohort_summary(cases: list[dict]) -> dict:
    resolved_same = 0
    resolved_different = 0
    ambiguous = 0
    resolved_rank = []
    for case in cases:
        for left, right in itertools.combinations(case["orders"], 2):
            a, b = left["winner_candidate_id"], right["winner_candidate_id"]
            if a is None or b is None:
                ambiguous += 1
            elif a == b:
                resolved_same += 1
            else:
                resolved_different += 1
            ra, rb = left["ranked_candidate_ids"], right["ranked_candidate_ids"]
            if ra is not None and rb is not None:
                resolved_rank.append(evaluator.spearman_by_identity(ra, rb))
    resolved = resolved_same + resolved_different
    total = resolved + ambiguous
    return {
        "records": len(cases),
        "variants": sum(len(c["orders"]) for c in cases),
        "top_tie_variants": sum(c["top_score_ties"] for c in cases),
        "records_with_at_least_one_top_tie": sum(c["top_score_ties"] > 0 for c in cases),
        "all_order_pairs": total,
        "resolved_same_winner_pairs": resolved_same,
        "resolved_different_winner_pairs": resolved_different,
        "ambiguous_pairs_from_tie_or_invalid_or_abstain": ambiguous,
        "actual_winner_flip_rate_given_two_unique_winners": resolved_different / resolved if resolved else None,
        "conditional_top1_agreement_given_two_unique_winners": resolved_same / resolved if resolved else None,
        "conditional_mean_spearman_given_two_unique_rankings": sum(resolved_rank) / len(resolved_rank) if resolved_rank else None,
        "pooled_pair_effective_disagreement_rate": (resolved_different + ambiguous) / total,
        "pre_registered_effective_disagreement_rate": sum(1 - c["top1_identity_agreement"] for c in cases) / len(cases),
    }


def main() -> None:
    manifest = client.frozen_manifest()
    cases_path = OUT / "EXTERNAL_SSQ_ORDER_CASES.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    if len(cases) != len(manifest) or {r["record_id"] for r in cases} != {r["record_id"] for r in manifest}:
        raise RuntimeError("order cases do not match frozen manifest")
    main_metrics = json.loads((OUT / "EXTERNAL_SSQ_ORDER_ROBUSTNESS_METRICS.json").read_text(encoding="utf-8"))
    if main_metrics["status"] != "BLOCKED_EXTERNAL_SSQ_ORDER_ROBUSTNESS":
        raise RuntimeError("unexpected frozen gate status")
    groups = {
        "InterfaceSem_ALL": [r for r in cases if r["module"] == "InterfaceSem"],
        "LayoutSem_ALL": [r for r in cases if r["module"] == "LayoutSem"],
        "LayoutSem_FIVE_DISTINCT_ORDERS": [r for r in cases if r["module"] == "LayoutSem" and r["cohort"] == "FIVE_DISTINCT_ORDERS"],
        "LayoutSem_TWO_ORDER_EXHAUSTIVE_DIAGNOSTIC": [r for r in cases if r["module"] == "LayoutSem" and r["cohort"] == "TWO_ORDER_EXHAUSTIVE_DIAGNOSTIC"],
    }
    values = {name: cohort_summary(rows) for name, rows in groups.items()}
    for name, details in values.items():
        official = main_metrics["module_and_cohort_metrics"][name]["winner_flip_rate"]
        if abs(details["pre_registered_effective_disagreement_rate"] - official) > 1e-10:
            raise RuntimeError(f"effective disagreement mismatch: {name}")
    result = {
        "schema": "seraph_external_ssq_order_tie_interpretation_v0_4_0e",
        "status_unchanged": main_metrics["status"],
        "manifest_sha256": main_metrics["manifest_sha256"],
        "official_gate_unchanged": True,
        "note": "The pre-registered winner_flip_rate is a conservative effective disagreement rate: different unique winners plus any ambiguous top-tie/invalid/abstain pair. It must not be interpreted as only observed identity flips.",
        "cohorts": values,
        "Gold_read": False,
        "new_API_calls": 0,
    }
    lines = [
        "# Candidate-order audit: top-score tie interpretation", "",
        "This is a Gold-free interpretation addendum. The pre-registered gates and final module policy are unchanged. No new API request was made.", "",
        "The main report's `winner_flip_rate` is the complement of conservative top-1 agreement. It includes pairs where a top-score tie leaves no unique winner, so it is not a literal count of observed candidate-identity flips.", "",
        "| Cohort | Top-tie variants | Affected records | Unique-winner pairs | Actual different-winner pairs | Ambiguous pairs | Conditional flip rate | Pre-registered effective disagreement |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, d in values.items():
        n_resolved = d["resolved_same_winner_pairs"] + d["resolved_different_winner_pairs"]
        lines.append(f"| {name} | {d['top_tie_variants']} | {d['records_with_at_least_one_top_tie']} | {n_resolved} | {d['resolved_different_winner_pairs']} | {d['ambiguous_pairs_from_tie_or_invalid_or_abstain']} | {d['actual_winner_flip_rate_given_two_unique_winners']:.4f} | {d['pre_registered_effective_disagreement_rate']:.4f} |")
    lines.extend(["", "InterfaceSem has no observed identity flip among pairs with two unique winners; its failed conservative gate is driven by top-score ties and ambiguous comparisons. This does not justify enabling it: the frozen gate requires robust, uniquely decidable top-1 outputs across all orders. LayoutSem additionally exhibits genuine different-winner pairs after ambiguity is excluded.", "", "No HOLDOUT Gold, CALIBRATION, FINAL_TEST_V2, or FINAL_TEST_V3 was read. The final status remains BLOCKED_EXTERNAL_SSQ_ORDER_ROBUSTNESS.", ""])
    with (OUT / "EXTERNAL_SSQ_ORDER_TIE_DIAGNOSTIC.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (OUT / "EXTERNAL_SSQ_ORDER_TIE_INTERPRETATION.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({k: {"top_tie_variants": v["top_tie_variants"],
                          "actual_unique_winner_flip_rate": v["actual_winner_flip_rate_given_two_unique_winners"]}
                      for k, v in values.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
