#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dwarf_progress_wrapper.py

Run eval_resym_dwarf_semantics_v6.py with visible progress.

It invokes the existing evaluator by chunks, appends per-sample CSV rows,
and prints progress after each chunk. Use --chunk-size 1 if you truly want
one log line per sample; use 10/20/50 for faster full evaluation.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


def semantic_paths(json_root: Path):
    return sorted(json_root.glob("*_semantic_dict.json"))


def sample_id_from_json(p: Path):
    name = p.name
    return name[:-len("_semantic_dict.json")] if name.endswith("_semantic_dict.json") else p.stem


def read_existing_done(out_csv: Path):
    done = set()
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return done
    try:
        with out_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                jp = row.get("json") or row.get("json_path") or ""
                if jp:
                    done.add(str(Path(jp).resolve()))
    except Exception:
        pass
    return done


def append_csv_rows(final_csv: Path, chunk_csv: Path):
    if not chunk_csv.exists() or chunk_csv.stat().st_size == 0:
        return 0, Counter()
    with chunk_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        return 0, Counter()
    final_exists = final_csv.exists() and final_csv.stat().st_size > 0
    with final_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not final_exists:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows), Counter(r.get("status", "unknown") for r in rows)


def parse_summary(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_chunk(args, chunk, tmp_base, chunk_idx):
    tmp_json = tmp_base / f"json_chunk_{chunk_idx}"
    tmp_json.mkdir(parents=True, exist_ok=True)
    for p in chunk:
        sid = sample_id_from_json(p)
        dst = tmp_json / p.name
        if not dst.exists():
            os.symlink(str(p.resolve()), str(dst))
        meta = p.parent / f"{sid}.meta.json"
        if meta.exists():
            mdst = tmp_json / meta.name
            if not mdst.exists():
                os.symlink(str(meta.resolve()), str(mdst))

    chunk_csv = tmp_base / f"chunk_{chunk_idx}.csv"
    chunk_summary = tmp_base / f"chunk_{chunk_idx}_summary.json"
    cmd = [
        sys.executable, str(Path(args.evaluator).resolve()),
        "--json-root", str(tmp_json),
        "--full-root", str(Path(args.full_root).resolve()),
        "--out", str(chunk_csv),
        "--summary", str(chunk_summary),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    return proc.returncode, elapsed, chunk_csv, chunk_summary, proc.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator", required=True)
    ap.add_argument("--json-root", required=True)
    ap.add_argument("--full-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--verbose-child", action="store_true")
    args = ap.parse_args()

    json_root = Path(args.json_root).resolve()
    out_csv = Path(args.out).resolve()
    summary_path = Path(args.summary).resolve()
    log_path = Path(args.log).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    paths = semantic_paths(json_root)
    if args.limit is not None:
        paths = paths[:args.limit]

    done = set() if args.no_resume else read_existing_done(out_csv)
    pending = [p for p in paths if str(p.resolve()) not in done]

    start = time.time()
    wrapper_counts = Counter()
    if done:
        wrapper_counts["skipped_resume"] = len(done)

    with log_path.open("a", encoding="utf-8") as log:
        def lp(msg):
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

        lp(f"START {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lp(f"json_root={json_root}")
        lp(f"full_root={Path(args.full_root).resolve()}")
        lp(f"total_json={len(paths)} already_done={len(done)} pending={len(pending)} chunk_size={args.chunk_size}")

        with tempfile.TemporaryDirectory(prefix="dwarf_eval_progress_") as td:
            tmp_base = Path(td)
            processed = 0
            for i in range(0, len(pending), args.chunk_size):
                chunk = pending[i:i + args.chunk_size]
                chunk_idx = i // args.chunk_size + 1
                ids = [sample_id_from_json(p) for p in chunk]

                rc, elapsed, chunk_csv, chunk_summary, child_out = run_chunk(args, chunk, tmp_base, chunk_idx)
                n_rows, cnt = append_csv_rows(out_csv, chunk_csv)
                wrapper_counts.update(cnt)
                if rc != 0:
                    wrapper_counts["evaluator_rc_nonzero"] += 1

                processed += len(chunk)
                total_done = len(done) + processed
                rate = processed / max(time.time() - start, 1e-6)
                remain = len(pending) - processed
                eta_min = (remain / rate / 60) if rate > 0 else 0
                summ = parse_summary(chunk_summary)
                chunk_status = summ.get("status_counts", dict(cnt))

                lp(f"[{total_done}/{len(paths)}] chunk={chunk_idx} size={len(chunk)} rows={n_rows} "
                   f"rc={rc} elapsed={elapsed:.2f}s status={chunk_status} "
                   f"first={ids[0]} last={ids[-1]} eta_min={eta_min:.2f}")

                if args.verbose_child or rc != 0:
                    lp(child_out[-4000:])

    final_status = Counter()
    rows = 0
    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows += 1
                final_status[row.get("status", "unknown")] += 1

    wall = time.time() - start
    final_summary = {
        "json_root": str(json_root),
        "full_root": str(Path(args.full_root).resolve()),
        "out": str(out_csv),
        "total_json": len(paths),
        "rows": rows,
        "status_counts": dict(final_status),
        "wrapper_counts": dict(wrapper_counts),
        "wall_seconds": wall,
        "wall_minutes": wall / 60,
        "wall_hours": wall / 3600,
        "note": "Progress wrapper around eval_resym_dwarf_semantics_v6.py; progress log is in --log."
    }
    summary_path.write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out_csv)
    print("wrote", summary_path)
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
