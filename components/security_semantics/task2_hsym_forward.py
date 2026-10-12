"""Forward-only frozen clean h_sym export for new Juliet USG nodes.

Runs Task1 v0_2_8 featurizer/model code read-only with a Task2-only config and
Task2-only output paths. This does not infer frozen Stage2 semantic labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PYTHON = "/home/cn/miniconda3/envs/seraph_llm/bin/python"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with log.open("x") as stream:
        stream.write("COMMAND " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        stream.write(f"\nEXIT {result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"read-only frozen h_sym forward failed; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task2-package", type=Path, required=True)
    parser.add_argument("--task2-output", type=Path, required=True)
    parser.add_argument("--wait-timeout", type=int, default=7200)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--expected-samples", type=int, default=100)
    args = parser.parse_args()
    pkg, out = args.task2_package, args.task2_output
    cfg = json.loads((pkg / "task2_config.json").read_text())
    task1_pkg = Path(cfg["clean_hsym_package"])
    task1_cfg = json.loads((task1_pkg / "config.json").read_text())
    checkpoint = Path(cfg["clean_hsym_checkpoint"])
    if not checkpoint.is_file():
        raise RuntimeError("frozen clean h_sym checkpoint missing")
    checkpoint_sha = sha(checkpoint)
    target = out / "juliet_clean_hsym_forward"
    target.mkdir(exist_ok=True)
    config_path = target / "TASK2_HSYM_FORWARD_CONFIG.json"
    task1_cfg["output_root"] = str(target)
    with config_path.open("x") as stream:
        json.dump(task1_cfg, stream, indent=2, sort_keys=True)
        stream.write("\n")
    graph = args.graph_root or (out / "juliet_usg_final")
    ready_log = out / "pipeline_logs/02_extract.log"
    if not args.no_wait:
        deadline = time.monotonic() + args.wait_timeout
        while time.monotonic() < deadline:
            if (out / "TASK2_PIPELINE_BLOCKED.json").exists():
                raise RuntimeError("Task2 structural pipeline blocked before final Juliet USG was ready")
            if ready_log.is_file() and "EXIT 0" in ready_log.read_text() and (graph / "graph_schema_stats.json").is_file():
                break
            time.sleep(20)
        else:
            raise RuntimeError("timed out waiting for final Juliet USG; no Task1 artifact touched")
    stats = json.loads((graph / "graph_schema_stats.json").read_text())
    if stats["successful_sample_count"] != args.expected_samples or stats["error_count"]:
        raise RuntimeError("Juliet USG coverage does not match expected sample count")
    feature = target / "clean_node_features_coverage_only.pt"
    run([PYTHON, str(task1_pkg / "src/build_clean_features.py"),
         "--config", str(config_path), "--nodes", str(graph / "usg_nodes.jsonl"),
         "--edges", str(graph / "usg_edges.jsonl"), "--output", str(feature),
         "--scope", "coverage_only"], target / "featurize.log")
    schema = target / "clean_feature_schema_coverage_only.json"
    vocab = target / "clean_feature_vocab_coverage_only.json"
    old_schema = Path(cfg["task1_hsym_status"]).parent / "clean_feature_schema_coverage_only.json"
    old_vocab = Path(cfg["task1_hsym_status"]).parent / "clean_feature_vocab_coverage_only.json"
    new_schema_data, old_schema_data = json.loads(schema.read_text()), json.loads(old_schema.read_text())
    new_vocab_data, old_vocab_data = json.loads(vocab.read_text()), json.loads(old_vocab.read_text())
    if new_schema_data["columns"] != old_schema_data["columns"] or new_vocab_data["features"] != old_vocab_data["features"]:
        raise RuntimeError("Juliet clean feature schema/vocab differs from frozen Task1")
    output_embeddings = target / "node_embeddings_clean_juliet.pt"
    output_index = target / "node_embedding_index_clean_juliet.json"
    output_report = target / "clean_hsym_forward_report.json"
    run([PYTHON, str(task1_pkg / "src/export_clean_embeddings.py"),
         "--features", str(feature), "--schema", str(schema), "--vocab", str(vocab),
         "--checkpoint", str(checkpoint), "--output", str(output_embeddings),
         "--index-output", str(output_index), "--report", str(output_report),
         "--device", "cpu", "--batch-size", "1024"], target / "export.log")
    report = json.loads(output_report.read_text())
    if not report["passed"] or report["shape"][0] != stats["total_nodes"] or report["shape"][1] != 64:
        raise RuntimeError("frozen clean h_sym Juliet forward shape/coverage failed")
    if sha(checkpoint) != checkpoint_sha:
        raise RuntimeError("frozen Task1 h_sym checkpoint changed during forward")
    md = ["# Juliet clean h_sym forward-only export", "",
          "Status: PASSED. The frozen v0_2_8 model was reloaded on CPU with unchanged feature schema/vocab.",
          f"- Checkpoint: `{checkpoint}` (SHA256 `{checkpoint_sha}`)",
          f"- Juliet node embeddings: `{output_embeddings}`; shape `{report['shape']}`; NaN/Inf `{report['nan_count']}/{report['inf_count']}`.",
          f"- Node index: `{output_index}`; entries `{report['index_count']}`.",
          "- This is h_sym forward inference only. Frozen Stage2 TypeSem/InterfaceSem/LayoutSem predictions on Juliet are still unavailable and are not fabricated.",
          "- No Task1 checkpoint, feature schema, or training result was modified.", ""]
    with (target / "JULIET_CLEAN_HSYM_FORWARD_REPORT.md").open("x") as stream:
        stream.write("\n".join(md))
    print(json.dumps({"status": "PASSED", "shape": report["shape"],
                      "checkpoint_sha256": checkpoint_sha, "Stage2_Juliet_predictions": 0}), flush=True)


if __name__ == "__main__":
    main()
