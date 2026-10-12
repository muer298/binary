"""Frozen upstream data stages for FINAL_TEST_V2; no Stage-2 inference."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

PYTHON = "/home/cn/miniconda3/envs/seraph_llm/bin/python"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

def require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")


def dump(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")


def run(argv: list[str], *, env: dict | None = None) -> None:
    print("RUN", " ".join(argv), flush=True)
    subprocess.run(argv, check=True, env=env)


def checked_config(config: str) -> tuple[dict, Path, Path, Path]:
    cfg = json.loads(Path(config).read_text())
    out = Path(cfg["output_root"])
    inputs = out / "final_test_v2_inputs"
    gold = out / "final_test_v2_gold_sealed"
    if not inputs.is_dir() or not gold.is_dir():
        raise RuntimeError("prepared inputs/Gold missing")
    if any(cfg[key] for key in (
        "final_stage2_checkpoint_load_allowed", "stage1_training_allowed",
        "hsym_training_allowed", "stage2_training_allowed", "ssq_allowed",
        "calibration_allowed", "test_prediction_allowed", "performance_evaluation_allowed",
    )):
        raise RuntimeError("forbidden operation enabled")
    return cfg, out, inputs, gold


def ghidra(cfg: dict, out: Path, inputs: Path, gold: Path) -> None:
    pkg = Path(cfg["research2_package"])
    semantic = inputs / "semantic_json_stripped_heap32g"
    work = inputs / "ghidra_projects_heap32g"
    report = out / "GHIDRA_GOLD_FIREWALL.json"
    for p in (semantic, work, report): require_absent(p)
    semantic.mkdir(); work.mkdir()
    env = dict(os.environ)
    env["JAVA_HOME"] = cfg["java_home"]
    env["PATH"] = str(Path(cfg["java_home"]) / "bin") + os.pathsep + env.get("PATH", "")
    run([
        PYTHON, "-u", str(pkg / "run_usg_headless_stream_v2.py"),
        "--manifest", str(inputs / "inference_manifest.csv"),
        "--ghidra-support", str(Path(cfg["ghidra_home"]) / "support"),
        "--jython-script", str(pkg / "over_v11_typesem_plus_paramflow.py"),
        "--out-root", str(semantic), "--work-root", str(work),
        "--project-prefix", "seraph_final_test_v2", "--only-status", "ok",
        "--timeout-sec", "2400", "--analysis-timeout-per-file", "1200", "--no-stream",
    ], env=env)
    actual = len(list(semantic.glob("*_semantic_dict.json")))
    if actual != len(cfg["candidates"]):
        raise RuntimeError(f"semantic export count {actual} != {len(cfg['candidates'])}")
    run([
        PYTHON, "-u", str(pkg / "symgen_external/audit_gold_firewall_v1.py"),
        "--pair-manifest", str(gold / "pair_manifest.csv"),
        "--inference-manifest", str(inputs / "inference_manifest.csv"),
        "--semantic-json-root", str(semantic),
        "--gold-root", str(gold), "--inference-root", str(inputs),
        "--report", str(report),
    ])
    dump(out / "GHIDRA_STAGE_STATUS.json", {
        "schema": "seraph_final_test_v2_ghidra_stage", "status": "PASSED",
        "semantic_json_count": actual, "full_elf_opened_by_ghidra": 0,
        "stage2_checkpoint_loaded": False, "test_prediction_count": 0,
        "performance_evaluation_count": 0,
    })


def usg(cfg: dict, out: Path, inputs: Path, gold: Path) -> None:
    pkg = Path(cfg["research2_package"])
    semantic = inputs / "semantic_json_stripped_heap32g"
    graph = inputs / "graphs"
    require_absent(graph)
    graph.mkdir()
    run([PYTHON, "-u", str(pkg / "01_build_usg_index.py"),
         "--json-root", str(semantic), "--out", str(graph / "usg_index.jsonl"),
         "--stats", str(out / "USG_INDEX_STATUS.json")])
    run([PYTHON, "-u", str(pkg / "02_extract_nodes_edges.py"),
         "--index", str(graph / "usg_index.jsonl"),
         "--nodes-out", str(graph / "usg_nodes_raw.jsonl"),
         "--edges-out", str(graph / "usg_edges.jsonl"),
         "--stats", str(out / "USG_GRAPH_STATUS.json"), "--fail-on-error"])
    if not (graph / "usg_nodes_raw.jsonl").stat().st_size or not (graph / "usg_edges.jsonl").stat().st_size:
        raise RuntimeError("empty USG graph")


def map_gold(cfg: dict, out: Path, inputs: Path, gold: Path) -> None:
    pkg = Path(cfg["research2_package"])
    indexed_mapper = Path("/data/chaoni/mjy/resym_exp/packages/seraph_ssq_layout_dev_supplement_v0_3_1a/src/map_symgen_gold_indexed_v031a.py")
    mapped = gold / "mapped_strict_gold.jsonl"
    audit = gold / "mapped_strict_gold_audit.json"
    for p in (mapped, audit): require_absent(p)
    graph = inputs / "graphs"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pkg / "symgen_external") + os.pathsep + env.get("PYTHONPATH", "")
    run([
        PYTHON, "-u", str(indexed_mapper),
        "--usg-index", str(graph / "usg_index.jsonl"),
        "--nodes", str(graph / "usg_nodes_raw.jsonl"),
        "--edges", str(graph / "usg_edges.jsonl"),
        "--split-manifest", str(gold / "split_manifest.jsonl"),
        "--out", str(mapped), "--audit-out", str(audit),
        "--oracle-cache-dir", str(gold / "oracles"),
        "--require-complete-oracle-coverage", "--min-mapping-rate-per-module", "0",
    ], env=env)
    if not mapped.stat().st_size:
        raise RuntimeError("mapped strict Gold empty")


def clean(cfg: dict, out: Path, inputs: Path, gold: Path) -> None:
    pkg = Path(cfg["clean_package"])
    graph = inputs / "graphs"
    clean_dir = inputs / "clean"
    require_absent(clean_dir)
    clean_dir.mkdir()
    features = clean_dir / "clean_node_features.pt"
    embeddings = clean_dir / "node_embeddings_clean.pt"
    index = clean_dir / "node_embedding_index_clean.json"
    export = clean_dir / "clean_embedding_export_report.json"
    run([PYTHON, "-u", str(pkg / "src/build_clean_features.py"),
         "--config", str(PACKAGE_ROOT / "config.json"),
         "--nodes", str(graph / "usg_nodes_raw.jsonl"),
         "--edges", str(graph / "usg_edges.jsonl"),
         "--output", str(features), "--scope", "coverage_only"])
    run([PYTHON, "-u", str(pkg / "src/export_clean_embeddings.py"),
         "--features", str(features),
         "--schema", str(out / "clean_feature_schema_coverage_only.json"),
         "--vocab", str(out / "clean_feature_vocab_coverage_only.json"),
         "--checkpoint", cfg["frozen_clean_hsym_checkpoint"],
         "--output", str(embeddings), "--index-output", str(index),
         "--report", str(export), "--device", "cpu", "--batch-size", "8192"])
    report = json.loads(export.read_text())
    if not report.get("passed"):
        raise RuntimeError("frozen clean h_sym export failed")
    dump(out / "CLEAN_HSYM_STAGE_STATUS.json", {
        "schema": "seraph_final_test_v2_clean_hsym_stage", "status": "PASSED",
        "checkpoint": cfg["frozen_clean_hsym_checkpoint"],
        "forward_only": True, "device": "cpu", "optimizer_steps": 0,
        "gradient_steps": 0, "stage2_checkpoint_loaded": False,
        "test_prediction_count": 0, "performance_evaluation_count": 0,
    })


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("stage", choices=["ghidra", "usg", "map-gold", "clean"])
    args = ap.parse_args()
    cfg, out, inputs, gold = checked_config(args.config)
    globals()[args.stage.replace("-", "_")](cfg, out, inputs, gold)


if __name__ == "__main__": main()
