#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify a completed Task 2.2.2 v5.5 closed-loop smoke or full run.

This checker deliberately reads three different summaries because the fields
are produced by different pipeline stages:

* validation_summary_v5_3.json: provenance and hard API seeds;
* reasoning_summary_v5_3.json: iterations and convergence;
* evaluation_v5_3.json: structural checks and warnings.

Exit status is 0 only when the run is safe to promote to the next scale.
Python 3.8+.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


JSONL_OUTPUTS = (
    "hypotheses_v5_3.jsonl",
    "validated_v5_3.jsonl",
    "reasoning_final_v5_3.jsonl",
    "reasoning_scope_for_223_v5_3.jsonl",
)

REQUIRED_OUTPUTS = JSONL_OUTPUTS + (
    "validation_summary_v5_3.json",
    "node_embeddings_reasoned_v5_3.pt",
    "reasoning_summary_v5_3.json",
    "evaluation_v5_3.json",
)

ACCEPTED_STOP_REASONS = {
    "global_consistency_reached",
    "state_stable",
    "max_iterations",
}


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("Cannot read JSON %s: %s" % (path, exc))
    if not isinstance(value, dict):
        raise RuntimeError("Expected a JSON object: %s" % path)
    return value


def count_nonempty_lines(path: Path) -> int:
    """Count JSONL records with bounded memory and no JSON decoding cost."""
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                count += 1
    return count


def nested_count(
    value: Mapping[str, Any],
    semantic_type: str,
    status: str,
) -> int:
    semantic = value.get("semantic_status_counts")
    if not isinstance(semantic, Mapping):
        return 0
    status_counts = semantic.get(semantic_type)
    if not isinstance(status_counts, Mapping):
        return 0
    try:
        return int(status_counts.get(status, 0) or 0)
    except (TypeError, ValueError):
        return 0


def add_if_false(
    errors: List[str], condition: bool, message: str
) -> None:
    if not condition:
        errors.append(message)


def check_mapping_is_empty(
    errors: List[str], value: Any, label: str
) -> None:
    if isinstance(value, Mapping) and value:
        errors.append("%s: %s" % (label, dict(value)))
    elif value not in (None, {}, []):
        errors.append("%s has an invalid value: %r" % (label, value))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Task 2.2.2 v5.5 output directory"
    )
    parser.add_argument("--out", required=True, help="Pipeline output directory")
    parser.add_argument(
        "--expected-records",
        type=int,
        default=0,
        help="Expected record count; 0 only requires all JSONL counts to agree",
    )
    parser.add_argument("--min-hard-seeds", type=int, default=1)
    parser.add_argument(
        "--require-sidecar",
        action="store_true",
        help="Require api_provenance_sidecar_used=true",
    )
    parser.add_argument(
        "--require-retrieved-candidate",
        action="store_true",
        help="Require at least one genuinely new per-round retrieved candidate.",
    )
    parser.add_argument(
        "--require-score-decrease",
        action="store_true",
        help="Require observed negative feedback in this particular run.",
    )
    parser.add_argument(
        "--allow-closed-loop-ablation",
        action="store_true",
        help=(
            "Allow intentionally disabled retrieval, negative feedback, or "
            "soft-decision revision while retaining audit/schema checks."
        ),
    )
    args = parser.parse_args()

    root = Path(args.out).expanduser().resolve()
    errors: List[str] = []
    notices: List[str] = []

    if not root.is_dir():
        print("VERIFY RESULT: FAILED")
        print(" - Output directory does not exist: %s" % root)
        return 1

    for name in REQUIRED_OUTPUTS:
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append("Missing or empty output: %s" % name)

    if errors:
        print("VERIFY RESULT: FAILED")
        for error in errors:
            print(" - " + error)
        return 1

    line_counts = {
        name: count_nonempty_lines(root / name)
        for name in JSONL_OUTPUTS
    }
    unique_counts = set(line_counts.values())
    for name, count in line_counts.items():
        print("%-44s records = %d" % (name, count))

    add_if_false(
        errors,
        all(count > 0 for count in line_counts.values()),
        "At least one JSONL output contains no records",
    )
    add_if_false(
        errors,
        len(unique_counts) == 1,
        "JSONL record counts do not agree: %s" % line_counts,
    )
    if args.expected_records > 0:
        for name, count in line_counts.items():
            if count != args.expected_records:
                errors.append(
                    "%s expected %d records but found %d"
                    % (name, args.expected_records, count)
                )

    validation = load_json(root / "validation_summary_v5_3.json")
    reasoning = load_json(root / "reasoning_summary_v5_3.json")
    evaluation = load_json(root / "evaluation_v5_3.json")

    expected = args.expected_records or next(iter(line_counts.values()))
    for label, value in (
        ("validation", validation),
        ("reasoning", reasoning),
        ("evaluation", evaluation),
    ):
        records = int(value.get("records", 0) or 0)
        print("%-44s records = %d" % (label + " summary", records))
        add_if_false(
            errors,
            records == expected,
            "%s summary has %d records; expected %d"
            % (label, records, expected),
        )

    sidecar_used = bool(validation.get("api_provenance_sidecar_used"))
    provenance_input = str(validation.get("api_provenance_input") or "")
    recovered_sources = validation.get("api_sources_recovered_from_nodes")
    hard_candidates = int(
        validation.get("hard_api_eligible_candidate_count", 0) or 0
    )
    hard_seeds = int(validation.get("hard_api_seed_count", 0) or 0)
    hard_sources = validation.get("hard_api_seed_sources") or {}
    validation_hard_conflicts = int(
        validation.get("hard_seed_conflicts", 0) or 0
    )
    api_hard = nested_count(validation, "APISem", "confirmed_hard")

    print("api_provenance_sidecar_used                 = %s" % sidecar_used)
    print("api_provenance_input                        = %s" % provenance_input)
    print("api_sources_recovered_from_nodes            = %s" % recovered_sources)
    print("hard_api_eligible_candidate_count           = %d" % hard_candidates)
    print("hard_api_seed_count                         = %d" % hard_seeds)
    print("hard_api_seed_sources                       = %s" % hard_sources)
    print("APISem.confirmed_hard                       = %d" % api_hard)
    print("validation hard_seed_conflicts              = %d" % validation_hard_conflicts)

    if args.require_sidecar:
        add_if_false(
            errors,
            sidecar_used,
            "The run did not report use of the API provenance sidecar",
        )
    add_if_false(
        errors,
        bool(provenance_input),
        "api_provenance_input is empty",
    )
    add_if_false(
        errors,
        hard_candidates >= args.min_hard_seeds,
        "Only %d hard-eligible API candidates; need at least %d"
        % (hard_candidates, args.min_hard_seeds),
    )
    add_if_false(
        errors,
        hard_seeds >= args.min_hard_seeds,
        "Only %d deterministic API seeds; need at least %d"
        % (hard_seeds, args.min_hard_seeds),
    )
    add_if_false(
        errors,
        api_hard >= args.min_hard_seeds,
        "APISem.confirmed_hard is %d; need at least %d"
        % (api_hard, args.min_hard_seeds),
    )
    add_if_false(
        errors,
        validation_hard_conflicts == 0,
        "Validation detected %d deterministic seed conflicts"
        % validation_hard_conflicts,
    )

    iterations = int(reasoning.get("iterations_run", 0) or 0)
    stop_reason = str(reasoning.get("stop_reason") or "")
    final_consistency = float(
        reasoning.get("final_global_consistency", 0.0) or 0.0
    )
    monotonic = bool(reasoning.get("global_consistency_monotonic"))
    raw_decreases = int(
        reasoning.get("raw_consistency_decrease_count", 0) or 0
    )
    reasoning_hard_conflicts = int(
        reasoning.get("hard_constraint_conflicts", 0) or 0
    )
    representation_mode = str(
        reasoning.get("representation_update_mode") or ""
    )
    revision = str(reasoning.get("implementation_revision") or "")
    closed_loop = reasoning.get("neural_symbolic_closed_loop")
    closed_loop = closed_loop if isinstance(closed_loop, Mapping) else {}
    total_retrieved = int(
        reasoning.get("total_retrieved_candidates", 0) or 0
    )
    total_pruned = int(
        reasoning.get("total_pruned_candidates", 0) or 0
    )
    score_increases = int(reasoning.get("total_score_increases", 0) or 0)
    score_decreases = int(reasoning.get("total_score_decreases", 0) or 0)
    iteration_trace = reasoning.get("iteration_trace")
    iteration_trace = iteration_trace if isinstance(iteration_trace, list) else []

    print("iterations_run                              = %d" % iterations)
    print("stop_reason                                 = %s" % stop_reason)
    print("final_global_consistency                    = %.12f" % final_consistency)
    print("global_consistency_monotonic                = %s" % monotonic)
    print("raw_consistency_decrease_count              = %d" % raw_decreases)
    print("reasoning hard_constraint_conflicts         = %d" % reasoning_hard_conflicts)
    print("representation_update_mode                 = %s" % representation_mode)
    print("implementation_revision                    = %s" % revision)
    print("total_retrieved_candidates                 = %d" % total_retrieved)
    print("total_pruned_candidates                    = %d" % total_pruned)
    print("total_score_increases                      = %d" % score_increases)
    print("total_score_decreases                      = %d" % score_decreases)

    add_if_false(errors, iterations > 0, "No reasoning iteration was recorded")
    add_if_false(
        errors,
        stop_reason in ACCEPTED_STOP_REASONS,
        "Unexpected reasoning stop_reason: %r" % stop_reason,
    )
    add_if_false(
        errors,
        0.0 <= final_consistency <= 1.0,
        "final_global_consistency is outside [0, 1]",
    )
    add_if_false(
        errors,
        monotonic,
        "Reported global consistency is not monotonic",
    )
    add_if_false(
        errors,
        reasoning_hard_conflicts == 0,
        "Reasoning detected %d deterministic constraint conflicts"
        % reasoning_hard_conflicts,
    )
    add_if_false(
        errors,
        representation_mode.startswith("signed_joint_space_message_passing"),
        "Signed neural representation update was not active: %r"
        % representation_mode,
    )
    add_if_false(
        errors,
        revision == "v5_6_irreversible_confirmation_transactional_raw_sglobal",
        "Unexpected or missing v5.6 implementation revision: %r" % revision,
    )
    required_capabilities = [
        "neural_state_updated",
        "symbolic_messages_signed",
    ]
    if not args.allow_closed_loop_ablation:
        required_capabilities.extend((
            "candidate_scores_bidirectional",
            "confirmed_states_irreversible",
            "actual_raw_sglobal_transactional",
            "candidate_set_retrieved_each_round",
        ))
        add_if_false(
            errors,
            closed_loop.get("soft_decisions_revisable") is False,
            "Document-compliant mode must not revise confirmed_soft states",
        )
        add_if_false(
            errors,
            raw_decreases == 0,
            "Actual raw S_global decreased in %d accepted round(s)"
            % raw_decreases,
        )
    for name in required_capabilities:
        add_if_false(
            errors,
            closed_loop.get(name) is True,
            "Closed-loop capability is inactive: %s=%r"
            % (name, closed_loop.get(name)),
        )
    add_if_false(
        errors,
        closed_loop.get("gold_used_during_inference") is False,
        "Inference must not read gold labels",
    )
    add_if_false(
        errors,
        bool(iteration_trace),
        "No v5.6 iteration trace was recorded",
    )
    for item in iteration_trace[1:]:
        if not isinstance(item, Mapping):
            errors.append("Malformed iteration trace item")
            continue
        for field in (
            "message_score_feedback",
            "embedding_rescore",
            "iterative_retrieval",
        ):
            if field not in item:
                errors.append(
                    "Iteration %r is missing %s"
                    % (item.get("iteration"), field)
                )
    if args.require_retrieved_candidate:
        add_if_false(
            errors,
            total_retrieved > 0,
            "No new candidate was recovered by iterative retrieval",
        )
    if args.require_score_decrease:
        add_if_false(
            errors,
            score_decreases > 0,
            "No candidate score decrease was observed",
        )
    if stop_reason == "state_stable" and final_consistency < float(
        reasoning.get("global_threshold", 0.85) or 0.85
    ):
        notices.append(
            "State stabilized before S_global reached the configured threshold; "
            "this is a valid convergence branch, not a failure."
        )
    if stop_reason == "max_iterations":
        notices.append(
            "The run stopped at max_iterations; inspect the last iteration "
            "deltas before committing substantial full-run resources."
        )

    checks = evaluation.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}
    pass_structural = evaluation.get("pass_structural_checks") is True
    warnings = evaluation.get("warnings")
    warnings = warnings if isinstance(warnings, list) else []

    print("pass_structural_checks                      = %s" % pass_structural)
    print("checks:")
    for name, passed in checks.items():
        print("  %-42s = %s" % (name, passed))
    print("warnings:")
    if warnings:
        for warning in warnings:
            print("  - %s" % warning)
    else:
        print("  []")

    add_if_false(
        errors,
        bool(checks),
        "evaluation_v5_3.json does not contain structural checks",
    )
    add_if_false(
        errors,
        pass_structural,
        "evaluation_v5_3.json reports pass_structural_checks=false",
    )
    for name, passed in checks.items():
        if passed is not True:
            errors.append("Structural check failed: %s=%r" % (name, passed))

    audit = evaluation.get("audit")
    audit = audit if isinstance(audit, Mapping) else {}
    check_mapping_is_empty(
        errors, audit.get("invalid_statuses"), "Invalid statuses"
    )
    check_mapping_is_empty(
        errors,
        audit.get("hard_confirmation_errors"),
        "Hard confirmation errors",
    )
    check_mapping_is_empty(
        errors,
        audit.get("soft_confirmation_errors"),
        "Soft confirmation errors",
    )
    check_mapping_is_empty(
        errors,
        audit.get("hypothesis_errors"),
        "Hypothesis errors",
    )

    if raw_decreases > 0:
        notices.append(
            "Raw S_global decreased in %d round(s); the evaluation policy "
            "decides whether this blocks promotion." % raw_decreases
        )
    notices.extend(str(item) for item in warnings)

    print()
    if notices:
        print("NOTICES:")
        for notice in notices:
            print(" - " + notice)
        print()

    if errors:
        print("VERIFY RESULT: FAILED")
        for error in errors:
            print(" - " + error)
        return 1

    print("VERIFY RESULT: PASSED")
    print("This output passed the v5.6 closed-loop structural promotion gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
