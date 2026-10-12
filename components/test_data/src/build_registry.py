"""Conservative identity registry for FINAL_TEST_V2 provenance preflight.

This is identity-only: it never opens model predictions or evaluation metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


HEX64 = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
REPO = re.compile(r"(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?![A-Za-z0-9_.-])")
BASE = Path("/data/chaoni/mjy/resym_exp")
RUNS = BASE / "training_runs/seraph_local"
VAL = BASE / "validation_runs"
RAW = Path("/data/chaoni/mjy/source_recovery/resym_rawdata/ReSym_rawdata")
SOURCES = Path("/data/chaoni/mjy/source_recovery/source_subset_v1/repos_redownload_v1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add_source(files: list[Path], path: Path) -> None:
    if path.is_file() and path not in files:
        files.append(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")

    metadata_path = RAW / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    reverse: dict[str, set[str]] = defaultdict(set)
    for repo, ids in metadata.items():
        for binary_id in ids:
            reverse[binary_id].add(repo)
    all_ids = set(reverse)
    all_repos = set(metadata)

    files: list[Path] = []
    # Stage-1 generations, including the old contaminated adapter input.
    for root in sorted(RUNS.glob("first_batch_v0_1_*")):
        add_source(files, root / "pairs.jsonl")
    for rel in (
        "stage1_clean_v0_1_5/pairs.jsonl",
        "stage1_clean_v0_1_5/STAGE1_CLEAN_DATA_AUDIT.json",
        "joint_v0_2_7_global_split/global_split_contract.json",
        "joint_v0_2_7_global_split/global_identity_inventory.jsonl",
        "joint_v0_2_7_global_split/joint_manifest_global_clean_v0_2_7.jsonl",
        "clean_hsym_v0_2_8/data/encoder_inner_split_manifest.jsonl",
        "clean_hsym_v0_2_8/data/outer_denylist.json",
        "clean_hsym_v0_2_8/data/train_nodes.jsonl",
        "clean_joint_v0_2_9/v0_2_9_contract.json",
        "clean_joint_v0_2_10/v0_2_10_contract.json",
        "ssq_layout_dev_supplement_v0_3_1a/NEW_LAYOUT_FUNCTION_PROVENANCE.json",
    ):
        add_source(files, RUNS / rel)
    for root in sorted(RUNS.glob("local_ssq_v0_3_*")) + [RUNS / "ssq_layout_dev_supplement_v0_3_1a"]:
        if not root.exists():
            continue
        for p in root.iterdir():
            lower = p.name.lower()
            if p.is_file() and p.suffix in {".json", ".jsonl", ".csv"} and any(
                word in lower for word in ("manifest", "provenance", "split")
            ):
                add_source(files, p)
        for sub in ("gold_zone", "layout_records"):
            if (root / sub).exists():
                for pattern in ("*manifest*.jsonl", "*manifest*.csv", "*provenance*.json"):
                    for p in (root / sub).glob(pattern):
                        add_source(files, p)
    for root in sorted(VAL.glob("blocksem_subset*_v7_1_0")):
        for name in ("subset_manifest.csv", "inference_manifest.csv"):
            add_source(files, root / name)
    for root in sorted(VAL.glob("blocksem_v7_selection_*")):
        for p in root.glob("*manifest*.csv"):
                add_source(files, p)

    by_id: dict[str, set[str]] = defaultdict(set)
    by_repo: dict[str, set[str]] = defaultdict(set)
    source_audit = []
    for path in files:
        found_ids: set[str] = set()
        observed_sample_ids: set[str] = set()
        found_repos: set[str] = set()
        # Stream lines; never materialize pseudocode/Gold contents in output.
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                found_ids.update(h for h in HEX64.findall(line) if h in all_ids)
                found_repos.update(r for r in REPO.findall(line) if r in all_repos)
                if path.suffix == ".jsonl":
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        row = None
                    if isinstance(row, dict):
                        for key in ("sample_id", "binary_id", "full_sha256", "stripped_sha256"):
                            value = row.get(key)
                            if isinstance(value, str) and HEX64.fullmatch(value):
                                observed_sample_ids.add(value)
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
                for row in csv.DictReader(fh):
                    for key in ("sample_id", "binary_id", "full_sha256", "stripped_sha256"):
                        value = row.get(key)
                        if isinstance(value, str) and HEX64.fullmatch(value):
                            observed_sample_ids.add(value)
        # Some frozen contracts store sample IDs as object keys.
        if path.name == "global_split_contract.json":
            observed_sample_ids.update(json.loads(path.read_text()).get("sample_roles", {}))
        found_ids.update(observed_sample_ids)
        try:
            label = str(path.relative_to(RUNS))
        except ValueError:
            label = str(path)
        for h in found_ids:
            by_id[h].add(label)
        for repo in found_repos:
            by_repo[repo].add(label)
        source_audit.append({"path": str(path), "sha256": sha256(path),
                             "binary_or_sample_ids": len(found_ids),
                             "sample_ids_extracted_by_field": len(observed_sample_ids),
                             "metadata_repos": len(found_repos)})

    # Historical source downloads are excluded conservatively even if not
    # known to have produced a pair or score.
    downloaded = []
    if SOURCES.exists():
        for path in SOURCES.iterdir():
            if path.is_dir() and "__" in path.name and not path.name.startswith("."):
                repo = path.name.replace("__", "/", 1)
                downloaded.append(repo)
                by_repo[repo].add("historical_source_download")
    by_repo["janet-lang/janet"].add("explicit_v0_3_1a_development_exclusion")
    # Local source-recovery attempts in this task are treated as historical
    # candidates, never promoted to a fresh final-test project by accident.
    for repo in (
        "cloudwu/skynet", "etlegacy/etlegacy", "bluekitchen/btstack",
        "pq-crystals/kyber", "openthread/ot-efr32", "xmos/xcore_iot",
        "Matheus-Garbelini/esp32_bluetooth_classic_sniffer",
        "BitBoxSwiss/bitbox02-firmware", "RobLoach/raylib-nuklear",
    ):
        by_repo[repo].add("local_source_recovery_candidate_excluded_conservatively")

    for h, evidence in by_id.items():
        for repo in reverse.get(h, ()):
            by_repo[repo].update(evidence)

    registry = {
        "schema": "seraph_all_used_projects_and_binaries_v2",
        "policy": "conservative_union_of_historical_sample_ids_and_repo_mentions",
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "source_audit": source_audit,
        "projects": {k: {"evidence": sorted(v),
                          "metadata_binary_ids_seen": sorted(h for h in metadata.get(k, []) if h in by_id)}
                     for k, v in sorted(by_repo.items())},
        "binary_ids": {k: {"metadata_repos": sorted(reverse.get(k, ())), "evidence": sorted(v)}
                       for k, v in sorted(by_id.items())},
        "sample_ids": {k: {"metadata_repos": sorted(reverse.get(k, ())), "evidence": sorted(v)}
                       for k, v in sorted(by_id.items())},
        "downloaded_repos_excluded_conservatively": sorted(downloaded),
        "counts": {"projects": len(by_repo), "binary_ids": len(by_id),
                   "sample_ids": len(by_id), "source_files": len(source_audit)},
        "limitation": "This union scans declared historical manifests/reports, not arbitrary unpublished private artifacts. Any selected candidate still requires exact repo/binary/sample preflight and independent provenance.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as fh:
        json.dump(registry, fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")
    print(json.dumps(registry["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
