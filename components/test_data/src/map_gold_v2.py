"""Repair only the frozen DWARF-oracle cache filename contract.

The original four oracle files and failed v1 map/audit remain untouched.
Aliases are symlinks to byte-identical frozen oracles, not modified labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


PYTHON = "/home/cn/miniconda3/envs/seraph_llm/bin/python"
MAPPER = Path("/data/chaoni/mjy/resym_exp/packages/seraph_ssq_layout_dev_supplement_v0_3_1a/src/map_symgen_gold_indexed_v031a.py")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["output_root"])
    inputs = root / "final_test_v2_inputs"
    gold = root / "final_test_v2_gold_sealed"
    if (root / "map_gold.exit").read_text().strip() != "1":
        raise RuntimeError("repair applies only to recorded v1 cache-name failure")
    previous = json.loads((gold / "mapped_strict_gold_audit.json").read_text())
    if previous.get("counts", {}).get("missing_oracle_cache") != len(cfg["candidates"]):
        raise RuntimeError("v1 failure was not exclusively missing frozen oracle cache")
    output = gold / "mapped_strict_gold_v2.jsonl"
    audit = gold / "mapped_strict_gold_audit_v2.json"
    report = root / "ORACLE_CACHE_FILENAME_REPAIR_V2.json"
    if any(path.exists() for path in (output, audit, report)):
        raise FileExistsError("refusing to overwrite v2 repair outputs")
    aliases = []
    for item in cfg["candidates"]:
        sample = item["binary_id"]
        original = gold / "oracles" / (sample + ".oracle.json")
        alias = gold / "oracles" / (sample + ".oracle_symgen_v1.json")
        if not original.is_file() or alias.exists():
            raise RuntimeError("oracle missing or alias already exists: " + sample)
        payload = json.loads(original.read_text())
        if (payload.get("schema") != "task2_symgen_elf_dwarf_oracle_v1"
                or not isinstance(payload.get("callsite_to_symbol"), dict)
                or not isinstance(payload.get("callsite_extraction"), dict)):
            raise RuntimeError("frozen oracle schema failed: " + sample)
        alias.symlink_to(original.name)
        if sha(alias) != sha(original):
            raise RuntimeError("oracle alias bytes differ: " + sample)
        aliases.append({"sample_id": sample, "original": str(original),
                        "alias": str(alias), "sha256": sha(original)})
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(cfg["research2_package"]) / "symgen_external") + os.pathsep + env.get("PYTHONPATH", "")
    command = [PYTHON, "-u", str(MAPPER),
               "--usg-index", str(inputs / "graphs/usg_index.jsonl"),
               "--nodes", str(inputs / "graphs/usg_nodes_raw.jsonl"),
               "--edges", str(inputs / "graphs/usg_edges.jsonl"),
               "--split-manifest", str(gold / "split_manifest.jsonl"),
               "--out", str(output), "--audit-out", str(audit),
               "--oracle-cache-dir", str(gold / "oracles"),
               "--require-complete-oracle-coverage", "--min-mapping-rate-per-module", "0"]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)
    updated = json.loads(audit.read_text())
    if output.stat().st_size == 0 or updated.get("oracle_cache_coverage") != 1.0:
        raise RuntimeError("v2 mapped Gold empty or oracle cache incomplete")
    with report.open("x", encoding="utf-8") as handle:
        json.dump({"status": "PASSED", "reason": "frozen mapper cache filename suffix mismatch",
                   "original_gold_content_modified": False, "aliases": aliases,
                   "v1_map_and_audit_preserved": True, "v2_output": str(output),
                   "v2_output_sha256": sha(output), "v2_audit": str(audit),
                   "test_prediction_count": 0, "performance_evaluation_count": 0},
                  handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "PASSED", "v2_gold_bytes": output.stat().st_size,
                      "oracle_cache_coverage": updated["oracle_cache_coverage"]}), flush=True)


if __name__ == "__main__": main()
