#!/usr/bin/env python3
"""Regression checks for the v6.7.4 controlled missing-source bridge."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / "26_enrich_layout_evidence_v6_7_4.py"


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def run_case(
    nodes: Path,
    out: Path,
    report: Path,
    extra: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--nodes-in",
            str(nodes),
            "--nodes-out",
            str(out),
            "--report-out",
            str(report),
            "--fail-on-source-error",
            *extra,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="layout_v674_test_") as raw:
        root = Path(raw)
        available = root / "available_semantic_dict.json"
        missing = root / "missing_semantic_dict.json"
        malformed = root / "malformed_semantic_dict.json"
        nodes = root / "nodes.jsonl"
        allowlist = root / "allowed_missing.txt"

        write_json(
            available,
            {
                "low_level_sem": {
                    "instr_sem": [
                        {
                            "id": "raw_query",
                            "function_entry": "0x1000",
                            "effects": {
                                "memory_accesses": [
                                    {
                                        "site": "0x1010",
                                        "base": "rdi",
                                        "offset": 0,
                                        "byte_width": 4,
                                        "access": "load",
                                        "is_fixed": True,
                                    },
                                    {
                                        "site": "0x1014",
                                        "base": "rdi",
                                        "offset": 8,
                                        "byte_width": 8,
                                        "access": "store",
                                        "is_fixed": True,
                                    },
                                ]
                            },
                        }
                    ]
                },
                "high_level_sem": {
                    "mem_layout_sem": [
                        {
                            "id": "raw_layout",
                            "fields": [
                                {"offset": 0, "byte_width": 4},
                                {"offset": 8, "byte_width": 8},
                            ],
                        }
                    ]
                },
            },
        )
        malformed.write_text("{not-json\n", encoding="utf-8")
        write_jsonl(
            nodes,
            [
                {
                    "sample_id": "available",
                    "node_id": "q_available",
                    "node_type": "InstrSem",
                    "layer": "instruction",
                    "raw_id": "raw_query",
                    "function_entry": "0x1000",
                    "function_entry_norm": "1000",
                    "json_path": str(available),
                    "source_path": "low_level_sem.instr_sem",
                    "attrs": {},
                },
                {
                    "sample_id": "available",
                    "node_id": "l_available",
                    "node_type": "LayoutSem",
                    "layer": "high",
                    "raw_id": "raw_layout",
                    "function_entry": "",
                    "function_entry_norm": "",
                    "json_path": str(available),
                    "source_path": "high_level_sem.mem_layout_sem",
                    "attrs": {},
                },
                {
                    "sample_id": "missing",
                    "node_id": "q_missing",
                    "node_type": "InstrSem",
                    "layer": "instruction",
                    "raw_id": "raw_missing",
                    "function_entry": "0x2000",
                    "function_entry_norm": "2000",
                    "json_path": str(missing),
                    "source_path": "low_level_sem.instr_sem",
                    "attrs": {"sentinel": "preserve"},
                },
            ],
        )
        allowlist.write_text(str(missing) + "\n", encoding="utf-8")

        failed = run_case(
            nodes,
            root / "strict_nodes.jsonl",
            root / "strict_report.json",
            [],
        )
        assert failed.returncode != 0, failed.stdout
        assert "semantic source read failed" in failed.stdout

        output = root / "allowed_nodes.jsonl"
        report_path = root / "allowed_report.json"
        passed = run_case(
            nodes,
            output,
            report_path,
            [
                "--allow-missing-source-file",
                str(allowlist),
                "--max-allowed-missing-sources",
                "1",
            ],
        )
        assert passed.returncode == 0, passed.stdout

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["approved_missing_source_count"] == 1
        assert report["unapproved_source_error_count"] == 0
        assert report["source_files_loaded"] == 1
        assert report["controlled_missing_source_mode"] is True
        assert report["approved_missing_requested_rows"] == 1
        assert report["approved_missing_requested_node_types"] == {
            "InstrSem": 1
        }
        assert report["semantic_source_load_coverage"] == 0.5

        rows = [
            json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {row["node_id"]: row for row in rows}
        assert by_id["q_available"]["attrs"][
            "observed_field_offsets"
        ] == [0, 8]
        assert by_id["l_available"]["attrs"]["field_offsets"] == [0, 8]
        assert by_id["q_missing"]["attrs"] == {"sentinel": "preserve"}

        malformed_nodes = root / "malformed_nodes.jsonl"
        write_jsonl(
            malformed_nodes,
            [
                {
                    "sample_id": "bad",
                    "node_id": "q_bad",
                    "node_type": "InstrSem",
                    "raw_id": "bad",
                    "json_path": str(malformed),
                    "source_path": "low_level_sem.instr_sem",
                    "attrs": {},
                }
            ],
        )
        malformed_allowlist = root / "malformed_allowlist.txt"
        malformed_allowlist.write_text(str(malformed) + "\n", encoding="utf-8")
        malformed_result = run_case(
            malformed_nodes,
            root / "malformed_out.jsonl",
            root / "malformed_report.json",
            [
                "--allow-missing-source-file",
                str(malformed_allowlist),
                "--max-allowed-missing-sources",
                "1",
            ],
        )
        assert malformed_result.returncode != 0, malformed_result.stdout
        assert "semantic source read failed" in malformed_result.stdout

        scope = root / "scope.jsonl"
        write_jsonl(
            scope,
            [
                {
                    "query_id": "q_available",
                    "candidates": [
                        {
                            "target_id": "l_available",
                            "target_semantic_type": "LayoutSem",
                        }
                    ],
                }
            ],
        )
        runner_output = root / "runner_output"
        runner_env = dict(os.environ)
        runner_env.update({
            "TASK2_ROOT": str(root),
            "TASK2_SRC": str(PACKAGE / "files"),
            "REPAIR_TAG": "fixture",
            "NODES": str(nodes),
            "SCOPE_222_V5_6": str(scope),
            "LAYOUT_ALLOWED_MISSING_SOURCE_FILE": str(allowlist),
            "LAYOUT_MAX_ALLOWED_MISSING_SOURCES": "1",
            "LAYOUT_EVIDENCE_DIR": str(runner_output),
            "MIN_LAYOUT_QUERY_EVIDENCE_COVERAGE": "1.0",
            "MIN_LAYOUT_PAIRWISE_GROUP_COVERAGE": "1.0",
        })
        runner = subprocess.run(
            [
                "bash",
                str(
                    PACKAGE
                    / "files"
                    / "run_layout_evidence_repair_v6_7_4.sh"
                ),
            ],
            env=runner_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert runner.returncode == 0, runner.stdout
        assert "CONTROLLED MISSING-SOURCE CONTRACT: PASSED" in runner.stdout
        assert "LAYOUT EVIDENCE SCOPE GATE: PASSED" in runner.stdout
        assert "LAYOUT EVIDENCE REPAIR v6.7.4: PASSED" in runner.stdout

        baseline_dir = root / "baseline"
        candidate_dir = root / "candidate"
        baseline_dir.mkdir()
        candidate_dir.mkdir()
        for directory, layout_r1, layout_mrr in (
            (baseline_dir, 0.13, 0.51),
            (candidate_dir, 0.25, 0.60),
        ):
            write_json(
                directory / "evaluation_v6_6.json",
                {
                    "ranking_overall": {
                        "after_collaboration": {
                            "n": 10,
                            "recall@1": 0.7,
                            "recall@5": 1.0,
                            "mrr": 0.8,
                        }
                    },
                    "ranking_by_module": {
                        "LayoutSem": {
                            "after_collaboration": {
                                "n": 4,
                                "recall@1": layout_r1,
                                "recall@5": 1.0,
                                "mrr": layout_mrr,
                            }
                        }
                    },
                },
            )
        paired_report = root / "paired_report.json"
        report_result = subprocess.run(
            [
                sys.executable,
                str(
                    PACKAGE
                    / "files"
                    / "28_report_layoutsem_v6_7_4.py"
                ),
                "--baseline-dir",
                str(baseline_dir),
                "--candidate-dir",
                str(candidate_dir),
                "--evidence-report",
                str(runner_output / "layout_evidence_enrichment_report_v6_7_4.json"),
                "--scope-audit",
                str(runner_output / "layout_evidence_scope_audit_v6_7_4.json"),
                "--out",
                str(paired_report),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert report_result.returncode == 0, report_result.stdout
        paired = json.loads(paired_report.read_text(encoding="utf-8"))
        assert abs(paired["metrics"]["layout_recall@1_delta"] - 0.12) < 1e-12

        print("CONTROLLED MISSING-SOURCE UNIT TESTS: PASSED")


if __name__ == "__main__":
    main()
