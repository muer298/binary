#!/usr/bin/env python3
"""Deterministic ~100-sample diagnostic controller for BlockSem/USG v7.1.0.

Design:
- sample selection uses only manifest status/path + SHA256(seed, sample_id)
- no Gold, model scores, completion order, or previous rankings influence selection
- reuses already verified semantic_dict outputs when available
- missing selected samples are regenerated with the v7 upstream exporter
- immutable snapshot + whole-stage checkpoints
- nohup-friendly; no tmux required
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter
from contextlib import contextmanager

HERE = Path(__file__).resolve().parent
P = Path(os.environ.get("P", "/data/chaoni/mjy/resym_exp"))
RUN = Path(os.environ.get("RUN_DIR", str(P / "validation_runs/blocksem_subset100_v7_1_0")))
MAN = Path(os.environ.get("FULL_MANIFEST", str(P / "samples_s80/manifest_s80.csv")))
UP = HERE / "vendor" / "upstream"
REUSE_ROOT = Path(os.environ.get(
    "REUSE_SEMANTIC_ROOT",
    str(P / "validation_runs/blocksem_v7_selection_20260910T062045Z/full/semantic_json"),
))
SEM_ROOT = RUN / "semantic_json"
SUBSET_SIZE = int(os.environ.get("SUBSET_SIZE", "100"))
SUBSET_SEED = os.environ.get("SUBSET_SEED", "blocksem-subset100-v7.1.0")
FROZEN_FILES = (
    "alignment_model_best.pt",
    "alignment_metrics.json",
    "feature_vocab.json",
    "confirmation_policy.json",
    "layout_reranker_config.json",
)
SUFFIX = "_semantic_dict.json"


def stamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def dump(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp." + uuid.uuid4().hex[:8])
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe_id(s):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))[:220]


def signature(p):
    st = Path(p).stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def csv_write(path, rows, keys):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp." + uuid.uuid4().hex[:8])
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(p)


@contextmanager
def lock(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = p.open("a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise RuntimeError("同一任务已有进程持锁：" + str(p))
    try:
        yield f
    finally:
        f.close()


def manifest_rows():
    if not MAN.is_file():
        raise RuntimeError("manifest 不存在：" + str(MAN))
    with MAN.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        need = {"sample_id", "stripped_path", "status"}
        if not need <= set(reader.fieldnames or []):
            raise RuntimeError("manifest 缺字段；需要 %r；实际=%r" % (sorted(need), reader.fieldnames))
        rows = [dict(r) for r in reader if str(r.get("status", "ok")).strip().lower() == "ok"]
    seen = set()
    good = []
    for r in rows:
        sid = safe_id(r.get("sample_id"))
        if not sid or sid in seen:
            raise RuntimeError("重复/空 sample_id safe_id：" + repr(sid))
        seen.add(sid)
        for k in ("stripped_path", "full_path"):
            if r.get(k):
                p = Path(os.path.expandvars(r[k])).expanduser()
                r[k] = str((MAN.parent / p).resolve() if not p.is_absolute() else p.resolve())
        r["safe_id"] = sid
        if Path(r["stripped_path"]).is_file():
            good.append(r)
    return good


def selection_key(sample_id):
    return hashlib.sha256((SUBSET_SEED + "\0" + str(sample_id)).encode("utf-8")).hexdigest()


def selected_rows():
    rows = manifest_rows()
    if SUBSET_SIZE <= 0:
        raise RuntimeError("SUBSET_SIZE 必须 > 0")
    if len(rows) < SUBSET_SIZE:
        raise RuntimeError("有效二进制不足：eligible=%d requested=%d" % (len(rows), SUBSET_SIZE))
    ranked = sorted(rows, key=lambda r: (selection_key(r["sample_id"]), str(r["sample_id"])))
    return ranked[:SUBSET_SIZE], len(rows)


def freeze_selection():
    RUN.mkdir(parents=True, exist_ok=True)
    chosen, eligible = selected_rows()
    report_path = RUN / "selection_report.json"
    manifest_path = RUN / "subset_manifest.csv"
    report = {
        "schema": "blocksem_subset_selection_v1",
        "version": "v7.1.0",
        "created_utc": stamp(),
        "source_manifest": str(MAN),
        "source_manifest_sha256": sha(MAN),
        "eligible_status_ok_existing_binary": eligible,
        "subset_size": SUBSET_SIZE,
        "subset_seed": SUBSET_SEED,
        "selection_rule": "sort by SHA256(seed + NUL + sample_id), take first N",
        "selection_uses_gold": False,
        "selection_uses_model_scores": False,
        "selection_uses_completion_order": False,
        "scope": "diagnostic subset; not independent Formal Test",
        "selected_sample_ids": [r["sample_id"] for r in chosen],
        "selected_safe_ids": [r["safe_id"] for r in chosen],
    }
    if report_path.exists() or manifest_path.exists():
        if not (report_path.exists() and manifest_path.exists()):
            raise RuntimeError("冻结清单只有一半存在；请不要手工修补：" + str(RUN))
        old = read_json(report_path)
        keys = ("source_manifest_sha256", "subset_size", "subset_seed", "selected_sample_ids")
        if any(old.get(k) != report.get(k) for k in keys):
            raise RuntimeError("已有冻结 selection 与当前配置不同；请使用新 RUN_DIR。")
        with manifest_path.open(encoding="utf-8-sig", newline="") as f:
            frozen = list(csv.DictReader(f))
        return frozen
    csv_write(manifest_path, chosen, ["sample_id", "full_path", "stripped_path", "status", "safe_id"])
    dump(report_path, report)
    return chosen


def contract(data):
    if not isinstance(data, dict):
        raise ValueError("semantic 根节点不是对象")
    info = data.get("program_info") or {}
    if info.get("blocksem_v7_error"):
        raise ValueError("blocksem_v7_error=" + str(info["blocksem_v7_error"]))
    if not isinstance(data.get("blocksem_v7"), dict):
        raise ValueError("缺少 blocksem_v7")
    low = data.get("low_level_sem") or {}
    ins = low.get("instr_sem") or []
    blocks = low.get("block_sem") or []
    if not isinstance(ins, list) or not isinstance(blocks, list) or not ins or not blocks:
        raise ValueError("InstrSem/BlockSem 空或 schema 不匹配")
    stats = data["blocksem_v7"].get("stats") or {}
    c = {
        "instructions": len(ins),
        "blocks": len(blocks),
        "blocks_with_ordered_members": sum(bool(b.get("ordered_instruction_ids")) for b in blocks),
        "blocks_member_complete": sum(b.get("member_list_complete") is True for b in blocks),
        "instructions_with_value_ids": sum(bool(x.get("input_value_ids") or x.get("output_value_ids")) for x in ins),
        "dfg_value_edges": int(stats.get("dfg_value_edges") or 0),
    }
    for k in ("blocks_with_ordered_members", "blocks_member_complete", "instructions_with_value_ids", "dfg_value_edges"):
        if not c[k]:
            raise ValueError("v7 semantic 合同未满足：" + k)
    return c


def check_semantic(root, row):
    sid = safe_id(row["sample_id"])
    p = Path(root) / (sid + SUFFIX)
    m = Path(root) / (sid + ".meta.json")
    out = {"sample_id": row["sample_id"], "safe_id": sid, "json": str(p), "meta": str(m)}
    try:
        if not p.is_file() or p.stat().st_size == 0:
            raise ValueError("missing_final_json")
        if not m.is_file() or m.stat().st_size == 0:
            raise ValueError("missing_success_meta")
        s0, m0 = signature(p), signature(m)
        meta = read_json(m)
        if meta.get("status") not in ("ok", "skipped_exists"):
            raise ValueError("meta_not_ok:" + str(meta.get("status")))
        if int(meta.get("returncode", 0) or 0) != 0:
            raise ValueError("meta_returncode_nonzero")
        msid = str(meta.get("safe_id") or safe_id(meta.get("sample_id", "")))
        if msid != sid:
            raise ValueError("meta_sample_id_mismatch")
        if meta.get("binary") and Path(meta["binary"]).resolve() != Path(row["stripped_path"]).resolve():
            raise ValueError("meta_binary_mismatch")
        c = contract(read_json(p))
        if s0 != signature(p) or m0 != signature(m):
            raise ValueError("files_changed_during_check")
        out.update(ready=True, counts=c)
    except Exception as e:
        out.update(ready=False, reason=str(e))
    return out


def find_ghidra():
    supplied = os.environ.get("GHIDRA_SUPPORT", "").strip()
    if supplied:
        d = Path(supplied).expanduser().resolve()
        if not (d / "analyzeHeadless").is_file():
            raise RuntimeError("GHIDRA_SUPPORT 不存在：" + str(d))
        return d
    candidates = set()
    oldrun = P / "validation_runs/blocksem_v7_selection_20260910T062045Z"
    for log in (oldrun / "logs/01_full_upstream.log", oldrun / "logs/00_smoke_upstream.log"):
        if log.is_file():
            with log.open(errors="replace") as f:
                for line in f:
                    if line.startswith("[INFO] ghidra:"):
                        q = Path(line.split(":", 1)[1].strip())
                        if q.is_file():
                            candidates.add(str(q.parent))
    for q in P.glob("ghidra*/support/analyzeHeadless"):
        if q.is_file():
            candidates.add(str(q.parent))
    if len(candidates) == 1:
        return Path(next(iter(candidates)))
    raise RuntimeError("无法唯一确定 Ghidra；请 export GHIDRA_SUPPORT=/.../support")


def prepare_semantics():
    with lock(RUN / ".prepare.lock"):
        chosen = freeze_selection()
        SEM_ROOT.mkdir(parents=True, exist_ok=True)
        (RUN / "gold_zone").mkdir(exist_ok=True)
        audit = []
        pending = []
        reused = 0
        for r in chosen:
            dst = check_semantic(SEM_ROOT, r)
            if dst["ready"]:
                dst["action"] = "already_ready"
                audit.append(dst)
                continue
            src = check_semantic(REUSE_ROOT, r) if REUSE_ROOT.is_dir() else {"ready": False, "reason": "reuse_root_missing"}
            if src.get("ready"):
                sid = safe_id(r["sample_id"])
                for suffix in (SUFFIX, ".meta.json"):
                    source = REUSE_ROOT / (sid + suffix)
                    target = SEM_ROOT / source.name
                    tmp = target.with_name(target.name + ".tmp." + uuid.uuid4().hex[:8])
                    shutil.copy2(source, tmp)
                    tmp.replace(target)
                chk = check_semantic(SEM_ROOT, r)
                if not chk["ready"]:
                    raise RuntimeError("复用后校验失败：" + json.dumps(chk, ensure_ascii=False))
                chk["action"] = "reused_verified"
                audit.append(chk)
                reused += 1
            else:
                pending.append(r)
                audit.append({
                    "sample_id": r["sample_id"], "safe_id": safe_id(r["sample_id"]),
                    "ready": False, "action": "needs_ghidra",
                    "dest_reason": dst.get("reason"), "reuse_reason": src.get("reason")
                })
        print("selection=%d reused=%d pending_ghidra=%d" % (len(chosen), reused, len(pending)), flush=True)
        if pending:
            # The upstream runner skips an existing final_json. Therefore an interrupted/invalid
            # output must be quarantined before retry, otherwise a missing/bad meta file would
            # make the retry silently skip forever.
            qdir = RUN / "semantic_quarantine" / (stamp() + "_" + str(os.getpid()))
            qdir.mkdir(parents=True, exist_ok=True)
            for r in pending:
                sid = safe_id(r["sample_id"])
                for suffix in (SUFFIX, ".meta.json", ".log"):
                    oldp = SEM_ROOT / (sid + suffix)
                    if oldp.exists():
                        shutil.move(str(oldp), str(qdir / oldp.name))
            g = find_ghidra()
            pending_manifest = RUN / "pending_semantic.csv"
            csv_write(pending_manifest, pending, ["sample_id", "stripped_path", "status"])
            work = RUN / ("ghidra_work_subset100_" + stamp() + "_" + str(os.getpid()))
            cmd = [
                sys.executable, "-u",
                str(UP / "01_modified_upstream/run_usg_headless_stream_v2.py"),
                "--manifest", str(pending_manifest),
                "--ghidra-support", str(g),
                "--jython-script", str(UP / "01_modified_upstream/over_v11_typesem_plus_paramflow_callfix_blocksem_v7.py"),
                "--out-root", str(SEM_ROOT),
                "--work-root", str(work),
                "--timeout-sec", "3600",
                "--analysis-timeout-per-file", "1200",
            ]
            print("COMMAND:", shlex.join(cmd), flush=True)
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print("Ghidra runner rc=%d; will still audit individual outputs" % rc, flush=True)
        final = [check_semantic(SEM_ROOT, r) for r in chosen]
        bad = [x for x in final if not x["ready"]]
        dump(RUN / "semantic_prepare_audit.json", {
            "selected": len(chosen),
            "ready": len(final) - len(bad),
            "bad": bad,
            "rows": final,
        })
        if bad:
            raise RuntimeError("100 样本 semantic 尚未全部完成：ready=%d bad=%d；再次 prepare/continue 只补失败项"
                               % (len(final)-len(bad), len(bad)))
        # Freeze inference inputs only after all N selected samples are valid.
        csv_write(RUN / "inference_manifest.csv", chosen, ["sample_id", "stripped_path", "status"])
        csv_write(RUN / "gold_zone/pairs.csv", chosen, ["sample_id", "full_path", "stripped_path", "status"])
        files = {
            "inference_manifest.csv": sha(RUN / "inference_manifest.csv"),
            "gold_zone/pairs.csv": sha(RUN / "gold_zone/pairs.csv"),
            "subset_manifest.csv": sha(RUN / "subset_manifest.csv"),
            "selection_report.json": sha(RUN / "selection_report.json"),
        }
        counts = Counter()
        for r in chosen:
            sid = safe_id(r["sample_id"])
            p = SEM_ROOT / (sid + SUFFIX)
            files[str(p.relative_to(RUN))] = sha(p)
            counts.update(contract(read_json(p)))
        snapshot = {
            "schema": "blocksem_subset100_snapshot_v1",
            "version": "v7.1.0",
            "created_utc": stamp(),
            "selected_samples": len(chosen),
            "subset_size": SUBSET_SIZE,
            "subset_seed": SUBSET_SEED,
            "source_manifest": str(MAN),
            "source_manifest_sha256": sha(MAN),
            "selection": "deterministic SHA256(sample_id) selection; no Gold/model/completion-order selection",
            "scope": "diagnostic subset; not Formal Test",
            "counts": dict(counts),
            "files_sha256": files,
        }
        dump(RUN / "snapshot.json", snapshot)
        print("PREPARE DONE:", RUN, flush=True)
        return RUN


def models():
    supplied = os.environ.get("FROZEN_MODEL_DIR", "").strip()
    if supplied:
        d = Path(supplied).expanduser().resolve()
        missing = [n for n in FROZEN_FILES if not (d / n).is_file()]
        if missing:
            raise RuntimeError("FROZEN_MODEL_DIR 不完整：%s 缺 %s" % (d, missing))
        return [d]
    found, seen = [], set()
    for root, dirs, names in os.walk(P):
        rel = Path(root).relative_to(P)
        dirs[:] = [d for d in dirs if d not in {
            "ghidra_work", "ghidra_projects", "semantic_json", "semantic_json_stripped",
            "node_modules", ".git", ".venv", "venv", "__pycache__", "full", "stripped",
            "background_jobs", "partial_runs",
        } and not d.startswith("ghidra_work")]
        if len(rel.parts) > 8:
            dirs[:] = []
        if all(n in names for n in FROZEN_FILES):
            d = Path(root).resolve()
            if d not in seen:
                found.append(d)
                seen.add(d)
    return sorted(found)


def choose_frozen():
    candidates = models()
    print("冻结模型候选：", *[str(x) for x in candidates], sep="\n", flush=True)
    if len(candidates) != 1:
        raise RuntimeError("需要唯一冻结模型；请 export FROZEN_MODEL_DIR=正式冻结目录")
    return candidates[0]


def process_start_ticks(pid):
    try:
        raw = Path("/proc/%s/stat" % pid).read_text()
        return raw[raw.rfind(")") + 2:].split()[19]
    except Exception:
        return None


def status():
    RUN.mkdir(parents=True, exist_ok=True)
    print("RUN_DIR =", RUN)
    print("SUBSET_SIZE =", SUBSET_SIZE)
    print("SUBSET_SEED =", SUBSET_SEED)
    sr = RUN / "selection_report.json"
    if sr.exists():
        x = read_json(sr)
        print("selection frozen =", len(x.get("selected_sample_ids") or []))
    else:
        print("selection frozen = NO")
    sm = RUN / "subset_manifest.csv"
    if sm.exists():
        with sm.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        ready = sum(check_semantic(SEM_ROOT, r)["ready"] for r in rows)
        print("semantic ready = %d/%d" % (ready, len(rows)))
    snap = RUN / "snapshot.json"
    print("snapshot =", "READY" if snap.exists() else "NOT_READY")
    progress = RUN / "pipeline_progress.json"
    if progress.exists():
        print("pipeline =", json.dumps(read_json(progress), ensure_ascii=False))
    for f in ("candidate_rankings.csv", "candidate_rankings.jsonl", "rank_summary.json"):
        p = RUN / f
        print(f, "READY" if p.is_file() and p.stat().st_size else "NOT_YET")
    jd = RUN / "background_jobs"
    if jd.exists():
        for p in sorted(jd.glob("*/state.json"))[-6:]:
            st = read_json(p)
            alive = bool(st.get("pid_start_ticks") and process_start_ticks(st.get("pid")) == st["pid_start_ticks"])
            print(p.parent.name, "alive="+str(alive), json.dumps(st, ensure_ascii=False))
    return 0


def run_rank(require_prepared=False):
    if not (RUN / "snapshot.json").is_file():
        if require_prepared:
            raise RuntimeError("snapshot 未准备；先运行 prepare 或 all")
        prepare_semantics()
    frozen = choose_frozen()
    from pipeline import run_pipeline
    run_pipeline(RUN, UP, frozen, mode="rank")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["status", "models", "select", "prepare", "all", "continue"])
    ap.add_argument("--job-dir", type=Path)
    args = ap.parse_args()
    if args.mode == "status":
        return status()
    if args.mode == "models":
        print(json.dumps([str(x) for x in models()], ensure_ascii=False, indent=2))
        return 0
    if args.mode == "select":
        rows = freeze_selection()
        print("selected =", len(rows))
        return 0
    jd = args.job_dir or RUN / "background_jobs" / ("foreground_" + stamp() + "_" + str(os.getpid()))
    jd.mkdir(parents=True, exist_ok=True)
    state = {
        "mode": args.mode,
        "pid": os.getpid(),
        "pid_start_ticks": process_start_ticks(os.getpid()),
        "started_utc": stamp(),
        "status": "RUNNING",
        "run_dir": str(RUN),
        "subset_size": SUBSET_SIZE,
        "subset_seed": SUBSET_SEED,
    }
    dump(jd / "state.json", state)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    rc = 1
    try:
        if args.mode == "prepare":
            prepare_semantics()
        elif args.mode == "all":
            prepare_semantics()
            run_rank(require_prepared=True)
        elif args.mode == "continue":
            run_rank(require_prepared=True)
        rc = 0
    except Exception:
        state["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        state.update(status="DONE" if rc == 0 else "FAILED_OR_INCOMPLETE",
                     returncode=rc, finished_utc=stamp())
        dump(jd / "state.json", state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
