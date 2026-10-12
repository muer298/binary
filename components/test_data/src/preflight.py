"""Identity-only FINAL_TEST_V2 provenance preflight; no model imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    out = Path(cfg["output_root"]) / "FINAL_TEST_V2_PROVENANCE_PREFLIGHT.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    registry_path = Path(cfg["registry"])
    registry = json.loads(registry_path.read_text())
    metadata_path = Path(cfg["metadata"])
    metadata = json.loads(metadata_path.read_text())
    reverse = {}
    for repo, ids in metadata.items():
        for sample_id in ids:
            reverse.setdefault(sample_id, set()).add(repo)
    banned_repos = {repo.casefold() for repo in registry["projects"]}
    banned_ids = set(registry["binary_ids"]) | set(registry["sample_ids"])
    seen_repos = set()
    seen_ids = set()
    candidates = []
    for item in cfg["candidates"]:
        repo, sample_id = item["repo"], item["binary_id"]
        binary = Path(cfg["raw_root"]) / "bin" / sample_id
        aliases = sorted(reverse.get(sample_id, ()))
        section = subprocess.run(["readelf", "-SW", str(binary)],
                                 capture_output=True, text=True, check=False)
        checks = {
            "repo_not_historical": repo.casefold() not in banned_repos,
            "sample_id_not_historical": sample_id not in banned_ids,
            "all_metadata_aliases_not_historical": all(a.casefold() not in banned_repos for a in aliases),
            "metadata_exact_single_project": aliases == [repo],
            "binary_present": binary.is_file(),
            "binary_sha256_matches_id": binary.is_file() and digest(binary) == sample_id,
            "dwarf_debug_info_and_line": section.returncode == 0 and
                ".debug_info" in section.stdout and ".debug_line" in section.stdout,
            "independent_within_candidates": repo.casefold() not in seen_repos and sample_id not in seen_ids,
            "fixed_source_commit_and_tree_recorded": len(item["source_commit"]) == 40 and len(item["source_tree"]) == 40,
        }
        seen_repos.add(repo.casefold()); seen_ids.add(sample_id)
        candidates.append({
            "repo": repo, "binary_id": sample_id, "sample_id": sample_id,
            "metadata_project_aliases": aliases, "binary_path": str(binary),
            "binary_size_bytes": binary.stat().st_size if binary.exists() else None,
            "binary_sha256": digest(binary) if binary.is_file() else None,
            "source_url": item["source_url"], "source_commit": item["source_commit"],
            "source_tree": item["source_tree"], "source_snapshot_time_utc": cfg["source_snapshot_time_utc"],
            "source_commit_matches_historical_binary_build": "UNVERIFIED_METADATA_DOES_NOT_PIN_COMMIT",
            "checks": checks, "identity_status": "PASSED" if all(checks.values()) else "REJECTED",
        })
    report = {
        "schema": "seraph_final_test_v2_provenance_preflight_v1",
        "status": "IDENTITY_PREFLIGHT_PASSED" if len(candidates) >= 2 and all(
            r["identity_status"] == "PASSED" for r in candidates) else "BLOCKED",
        "selection_policy": cfg["selection_policy"],
        "historical_registry_sha256": digest(registry_path),
        "metadata_sha256": digest(metadata_path),
        "config_sha256": digest(cfg_path),
        "historical_registry_counts": registry["counts"],
        "candidates": candidates,
        "candidate_project_count": len(candidates),
        "candidate_binary_count": len(candidates),
        "model_score_or_performance_consulted": False,
        "stage2_checkpoint_loaded": False,
        "test_prediction_count": 0,
        "test_performance_evaluation_count": 0,
    }
    with out.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, sort_keys=True, indent=2)
        fh.write("\n")
    print(json.dumps({"status": report["status"], "projects": len(candidates),
                      "rejected": [r["repo"] for r in candidates if r["identity_status"] != "PASSED"]}))
    if report["status"] != "IDENTITY_PREFLIGHT_PASSED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
