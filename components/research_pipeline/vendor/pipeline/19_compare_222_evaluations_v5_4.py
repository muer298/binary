#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare default and calibrated 2.2.2 runs on the identical frozen Test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping


SEMANTIC_TYPES = ("APISem", "TypeSem", "LayoutSem", "InterfaceSem")


def load(path: str) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Evaluation must be a JSON object: %s" % path)
    return value


def number(mapping: Mapping[str, Any], key: str) -> float:
    try:
        return float(mapping.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def module_metrics(evaluation: Mapping[str, Any], module: str) -> Dict[str, Any]:
    ranking = evaluation.get("ranking")
    ranking = ranking if isinstance(ranking, Mapping) else {}
    by_module = ranking.get("by_module")
    by_module = by_module if isinstance(by_module, Mapping) else {}
    row = by_module.get(module)
    row = row if isinstance(row, Mapping) else {}
    n = int(row.get("n", 0) or 0)
    confirmed = int(row.get("confirmed", 0) or 0)
    correct = int(row.get("correct_confirmed", 0) or 0)
    status_rows = row.get("confirmation_by_status")
    status_rows = status_rows if isinstance(status_rows, Mapping) else {}
    return {
        "n": n,
        "candidate_coverage": number(row, "candidate_coverage"),
        "recall@1": number(row, "recall@1"),
        "recall@5": number(row, "recall@5"),
        "mrr": number(row, "mrr"),
        "confirmed": confirmed,
        "correct_confirmed": correct,
        "confirmation_coverage": confirmed / n if n else 0.0,
        "confirmed_precision": correct / confirmed if confirmed else None,
        "confirmation_by_status": dict(status_rows),
    }


def delta(new: Any, old: Any) -> Any:
    if new is None or old is None:
        return None
    return float(new) - float(old)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--calibrated", required=True)
    parser.add_argument("--out")
    parser.add_argument("--max-ranking-drop", type=float, default=0.0)
    parser.add_argument("--max-candidate-coverage-drop", type=float, default=0.0)
    parser.add_argument(
        "--precision-floors",
        default="APISem=0.99,TypeSem=0.95,LayoutSem=0.95,InterfaceSem=0.95",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    floors = {}
    for item in args.precision_floors.split(","):
        key, value = item.split("=", 1)
        floors[key.strip()] = float(value)

    baseline = load(args.baseline)
    calibrated = load(args.calibrated)
    comparisons = {}
    reasons = []
    aligned = True
    any_confirmation_gain = False
    for module in SEMANTIC_TYPES:
        before = module_metrics(baseline, module)
        after = module_metrics(calibrated, module)
        if before["n"] != after["n"]:
            aligned = False
            reasons.append(
                "%s gold group count differs: %d != %d"
                % (module, before["n"], after["n"])
            )
        if before["n"] <= 0 or after["n"] <= 0:
            reasons.append("%s has no matched frozen Test gold groups" % module)
        changes = {
            key: delta(after[key], before[key])
            for key in (
                "candidate_coverage", "recall@1", "recall@5", "mrr",
                "confirmation_coverage", "confirmed_precision",
            )
        }
        if changes["candidate_coverage"] is not None and (
            changes["candidate_coverage"]
            < -args.max_candidate_coverage_drop - 1e-12
        ):
            reasons.append("%s candidate coverage decreased" % module)
        for key in ("recall@1", "recall@5", "mrr"):
            if changes[key] is not None and (
                changes[key] < -args.max_ranking_drop - 1e-12
            ):
                reasons.append("%s %s decreased" % (module, key))
        for status, floor in (
            ("confirmed_hard", 0.99),
            ("confirmed_soft", floors[module]),
        ):
            status_row = after["confirmation_by_status"].get(status, {})
            status_row = status_row if isinstance(status_row, Mapping) else {}
            status_confirmed = int(status_row.get("confirmed", 0) or 0)
            status_precision = (
                float(status_row.get("precision", 0.0) or 0.0)
                if status_confirmed else None
            )
            if (
                status_confirmed > 0
                and status_precision is not None
                and status_precision + 1e-12 < floor
            ):
                reasons.append(
                    "%s %s precision %.6f is below %.6f"
                    % (module, status, status_precision, floor)
                )
        if after["confirmation_coverage"] > before["confirmation_coverage"] + 1e-12:
            any_confirmation_gain = True
        comparisons[module] = {
            "baseline": before,
            "calibrated": after,
            "delta": changes,
        }

    if baseline.get("pass_structural_checks") is not True:
        reasons.append("baseline run failed structural checks")
    if calibrated.get("pass_structural_checks") is not True:
        reasons.append("calibrated run failed structural checks")
    if not any_confirmation_gain:
        reasons.append("no module gained confirmation coverage")
    safe = aligned and not reasons
    result = {
        "schema": "task2_222_frozen_test_comparison_v5_4",
        "baseline": args.baseline,
        "calibrated": args.calibrated,
        "same_frozen_gold_scope": aligned,
        "precision_floors": floors,
        "by_module": comparisons,
        "safe_to_accept_calibrated_policy": safe,
        "reasons_not_accepted": reasons,
        "decision_rule": (
            "Accept only on the identical frozen Test scope when structural "
            "checks pass, candidate coverage/ranking do not regress beyond "
            "tolerance, confirmed precision meets the floor, and at least one "
            "module gains confirmation coverage."
        ),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        if out.exists() and not args.overwrite:
            raise FileExistsError("Refusing to overwrite: %s" % out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not safe:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
