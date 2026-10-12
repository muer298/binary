#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_usg_headless_stream_v2.py

Purpose:
  Robust batch runner for Ghidra Headless + a Jython USG export script.

Key improvements over the earlier runner:
  1. Resolves all important paths to absolute paths to avoid cwd-related failures.
  2. Streams Ghidra output in real time while saving per-sample log files.
  3. Supports checkpoint/resume: existing *_semantic_dict.json outputs are skipped by default.
  4. Supports timeout per sample and writes structured *.meta.json and batch_report.json.
  5. Validates manifest format and also supports a simple path-list manifest as a fallback.

Accepted manifest formats:
  A) CSV with header:
       sample_id,stripped_path,status
       sample1,/abs/path/to/binary,ok
  B) Plain path list:
       /abs/path/to/binary1
       /abs/path/to/binary2

Example:
  python3 run_usg_headless_stream_v2.py \
    --manifest /path/to/manifest.csv \
    --ghidra-support /path/to/ghidra/support \
    --jython-script /path/to/over_v11_typesem_plus_paramflow.py \
    --out-root /path/to/work/semantic_json \
    --work-root /path/to/work/ghidra_projects \
    --max-samples 10 \
    --timeout-sec 1200 \
    --analysis-timeout-per-file 300
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def safe_id(value: str) -> str:
    text = str(value)
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)[:220]


def read_manifest(manifest_path: Path):
    """Read either a CSV manifest or a plain path list."""
    text = manifest_path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return [], "empty"

    first = lines[0]
    # CSV manifest: first row has named columns.
    if "," in first and ("stripped_path" in first or "binary_path" in first or "path" in first):
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return rows, "csv"

    # Fallback: plain path list.
    rows = []
    for idx, line in enumerate(lines, start=1):
        rows.append({
            "sample_id": Path(line).name or f"sample_{idx}",
            "stripped_path": line,
            "status": "ok",
        })
    return rows, "path_list"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ghidra-support", required=True)
    parser.add_argument("--jython-script", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--project-prefix", default="research2_usg")
    parser.add_argument("--status-field", default="status")
    parser.add_argument("--only-status", default="ok", help="Only run rows with this status; use 'any' to disable.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--analysis-timeout-per-file", type=int, default=900)
    parser.add_argument("--overwrite", action="store_true", help="Re-run even when final JSON already exists.")
    parser.add_argument("--keep-projects", action="store_true")
    parser.add_argument("--no-stream", action="store_true", help="Do not print Ghidra output to terminal in real time.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_process_streaming(cmd, cwd: Path, log_file: Path, timeout_sec: int, stream: bool):
    """Run one Ghidra process, stream stdout to terminal and log, enforce timeout."""
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    timed_out = False
    with log_file.open("w", encoding="utf-8", errors="replace") as lf:
        lf.write("[CMD] " + " ".join(cmd) + "\n\n")
        lf.flush()

        while True:
            line = ""
            if proc.stdout is not None:
                line = proc.stdout.readline()

            if line:
                lf.write(line)
                lf.flush()
                if stream:
                    print(line, end="")
                    sys.stdout.flush()

            rc = proc.poll()
            if rc is not None:
                # Drain any remaining output.
                rest = proc.stdout.read() if proc.stdout is not None else ""
                if rest:
                    lf.write(rest)
                    lf.flush()
                    if stream:
                        print(rest, end="")
                        sys.stdout.flush()
                return rc, time.time() - start, timed_out

            if time.time() - start > timeout_sec:
                timed_out = True
                lf.write(f"\n[TIMEOUT after {timeout_sec}s]\n")
                lf.flush()
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                return -999, time.time() - start, timed_out

            time.sleep(0.05)


def main():
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()
    ghidra_support = Path(args.ghidra_support).resolve()
    analyze_headless = ghidra_support / "analyzeHeadless"
    jython_script = Path(args.jython_script).resolve()
    out_root = Path(args.out_root).resolve()
    work_root = Path(args.work_root).resolve()

    if not manifest_path.exists():
        print(f"[ERROR] manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    if not analyze_headless.exists():
        print(f"[ERROR] analyzeHeadless not found: {analyze_headless}", file=sys.stderr)
        return 2
    if not jython_script.exists():
        print(f"[ERROR] Jython script not found: {jython_script}", file=sys.stderr)
        return 2

    out_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    raw_rows, manifest_mode = read_manifest(manifest_path)
    valid_rows = []
    invalid_rows = []
    for row in raw_rows:
        if args.only_status != "any" and row.get(args.status_field, "ok") != args.only_status:
            continue
        binary_path = row.get("stripped_path") or row.get("binary_path") or row.get("path")
        if not binary_path:
            row["_skip_reason"] = "missing_binary_field"
            invalid_rows.append(row)
            continue
        p = Path(binary_path)
        if not p.exists():
            row["_skip_reason"] = "binary_not_found"
            invalid_rows.append(row)
            continue
        valid_rows.append(row)

    valid_rows = valid_rows[args.start_index:]
    if args.max_samples is not None:
        valid_rows = valid_rows[:args.max_samples]

    print(f"[INFO] manifest: {manifest_path}")
    print(f"[INFO] manifest_mode: {manifest_mode}")
    print(f"[INFO] raw_rows: {len(raw_rows)}")
    print(f"[INFO] valid_rows_selected: {len(valid_rows)}")
    print(f"[INFO] invalid_rows: {len(invalid_rows)}")
    print(f"[INFO] ghidra: {analyze_headless}")
    print(f"[INFO] jython_script: {jython_script}")
    print(f"[INFO] out_root: {out_root}")
    print(f"[INFO] work_root: {work_root}")
    print()

    if not valid_rows:
        report_path = out_root / "batch_report.json"
        report_path.write_text("[]\n", encoding="utf-8")
        invalid_path = out_root / "invalid_rows.json"
        invalid_path.write_text(json.dumps(invalid_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[WARN] No valid rows selected. Nothing to run.")
        print(f"Batch report: {report_path}")
        print(f"Invalid rows: {invalid_path}")
        return 0

    batch_rows = []
    for idx, row in enumerate(valid_rows, start=1):
        sample_id = row.get("sample_id") or Path(row.get("stripped_path") or row.get("binary_path") or row.get("path")).name
        sid = safe_id(sample_id)
        binary = Path(row.get("stripped_path") or row.get("binary_path") or row.get("path")).resolve()

        sample_work = work_root / sid
        project_dir = sample_work / "project"
        run_cwd = sample_work / "run"
        project_dir.mkdir(parents=True, exist_ok=True)
        run_cwd.mkdir(parents=True, exist_ok=True)

        final_json = out_root / f"{sid}_semantic_dict.json"
        log_file = out_root / f"{sid}.log"
        meta_file = out_root / f"{sid}.meta.json"

        print("=" * 100)
        print(f"[{idx}/{len(valid_rows)}] {sid}")
        print(f"binary: {binary}")

        if final_json.exists() and not args.overwrite:
            print(f"[SKIP] exists: {final_json}")
            meta = {
                "sample_id": sample_id,
                "safe_id": sid,
                "binary": str(binary),
                "status": "skipped_exists",
                "final_json": str(final_json),
                "log_file": str(log_file),
                "manifest_row": row,
            }
            batch_rows.append(meta)
            continue

        cmd = [
            str(analyze_headless),
            str(project_dir),
            f"{args.project_prefix}_{sid}",
            "-import", str(binary),
            "-scriptPath", str(jython_script.parent),
            "-postScript", jython_script.name,
            "-analysisTimeoutPerFile", str(args.analysis_timeout_per_file),
        ]
        if not args.keep_projects:
            cmd.append("-deleteProject")

        print("cmd:", " ".join(cmd))
        print("log:", log_file)
        print("-" * 100)

        if args.dry_run:
            batch_rows.append({"sample_id": sample_id, "safe_id": sid, "status": "dry_run", "binary": str(binary)})
            continue

        rc, elapsed, timed_out = run_process_streaming(
            cmd=cmd,
            cwd=run_cwd,
            log_file=log_file,
            timeout_sec=args.timeout_sec,
            stream=not args.no_stream,
        )

        generated_dir = run_cwd / "ghidra_semantic_output"
        candidates = sorted(generated_dir.glob("*_semantic_dict.json"))
        if timed_out:
            status = "timeout"
            print(f"!! timeout, see {log_file}")
        elif rc != 0:
            status = f"failed_rc_{rc}"
            print(f"!! failed_rc_{rc}, see {log_file}")
        elif candidates:
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            shutil.copy2(newest, final_json)
            status = "ok"
            print(f"-> {final_json}")
        else:
            status = "failed_no_json"
            print(f"!! failed_no_json, see {log_file}")

        meta = {
            "sample_id": sample_id,
            "safe_id": sid,
            "binary": str(binary),
            "returncode": rc,
            "elapsed_sec": round(elapsed, 3),
            "status": status,
            "final_json": str(final_json) if final_json.exists() else None,
            "log_file": str(log_file),
            "manifest_row": row,
        }
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        batch_rows.append(meta)

    report_path = out_root / "batch_report.json"
    report_path.write_text(json.dumps(batch_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"Batch report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
