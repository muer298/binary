#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize the five Task 2.2.3 full/ablation runs into JSON and CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


VARIANTS = (
    "full",
    "no_api_hard_seed",
    "no_type_layout",
    "no_interface_feedback",
    "no_collaboration",
)
MODULES = ("overall", "APISem", "TypeSem", "LayoutSem", "InterfaceSem")


def load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Expected JSON object: %s" % path)
    return value


def metric(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def delta(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    return value - baseline if value is not None and baseline is not None else None


def ranking(evaluation: Mapping[str, Any], module: str) -> Mapping[str, Any]:
    if module == "overall":
        return (
            evaluation.get("ranking_overall", {}).get(
                "after_collaboration", {}
            )
        )
    return (
        evaluation.get("ranking_by_module", {}).get(module, {}).get(
            "after_collaboration", {}
        )
    )


def confirmation(
    evaluation: Mapping[str, Any], module: str
) -> Mapping[str, Any]:
    quality = evaluation.get("confirmation_quality", {})
    if module == "overall":
        return quality.get("overall", {})
    return quality.get("by_module", {}).get(module, {})


def make_row(
    variant: str,
    module: str,
    evaluation: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    rank = ranking(evaluation, module)
    quality = confirmation(evaluation, module)
    hard = quality.get("confirmed_hard", {})
    soft = quality.get("confirmed_soft", {})
    combined = quality.get("confirmed_combined", {})
    return {
        "variant": variant,
        "module": module,
        "gold_n": rank.get("n"),
        "candidate_coverage": rank.get("candidate_coverage"),
        "recall@1": rank.get("recall@1"),
        "recall@5": rank.get("recall@5"),
        "mrr": rank.get("mrr"),
        "hard_precision": hard.get("precision"),
        "hard_coverage_all": hard.get("coverage_all_objects"),
        "hard_coverage_gold": hard.get("coverage_gold_objects"),
        "soft_precision": soft.get("precision"),
        "soft_coverage_all": soft.get("coverage_all_objects"),
        "soft_coverage_gold": soft.get("coverage_gold_objects"),
        "combined_precision": combined.get("precision"),
        "combined_coverage_all": combined.get("coverage_all_objects"),
        "initial_raw_s_global": summary.get("initial_raw_s_global"),
        "final_raw_s_global": summary.get("final_raw_s_global"),
        "raw_s_global_delta": summary.get("raw_s_global_delta"),
        "collaboration_edges": summary.get("collaboration_edges"),
        "suppressed_confirmed_sources": summary.get(
            "suppressed_confirmed_source_count"
        ),
        "pass_structural_checks": evaluation.get("pass_structural_checks"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated variants actually present in the suite.",
    )
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    suite = Path(args.suite_dir)
    variants = tuple(
        value.strip() for value in args.variants.split(",") if value.strip()
    )
    if not variants or "full" not in variants:
        raise ValueError("--variants must include full")
    rows: List[Dict[str, Any]] = []
    evaluations: Dict[str, Dict[str, Any]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        evaluation_path = suite / variant / "evaluation_v6_6.json"
        summary_path = suite / variant / "collaboration_summary_v6_6.json"
        if not evaluation_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                "Missing ablation output for %s under %s" % (variant, suite)
            )
        evaluations[variant] = load(evaluation_path)
        summaries[variant] = load(summary_path)
        for module in MODULES:
            rows.append(
                make_row(
                    variant,
                    module,
                    evaluations[variant],
                    summaries[variant],
                )
            )

    full_by_module = {
        row["module"]: row for row in rows if row["variant"] == "full"
    }
    for row in rows:
        reference = full_by_module[row["module"]]
        for name in ("candidate_coverage", "recall@1", "recall@5", "mrr"):
            row["delta_vs_full_%s" % name] = delta(
                metric(row.get(name)), metric(reference.get(name))
            )

    payload = {
        "schema": "task2_223_ablation_summary_v6_7",
        "implementation_revision": "v6_7_validation_guarded",
        "suite_dir": str(suite.resolve()),
        "variants": list(variants),
        "modules": list(MODULES),
        "rows": rows,
        "interpretation": (
            "Deltas are ablation minus full; negative ranking deltas indicate "
            "the removed collaboration component helped the full system."
        ),
    }

    json_out = Path(args.json_out)
    csv_out = Path(args.csv_out)
    for output in (json_out, csv_out):
        if output.exists() and not args.overwrite:
            raise FileExistsError("Refusing to overwrite: %s" % output)
        output.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(rows[0])
    with csv_out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
