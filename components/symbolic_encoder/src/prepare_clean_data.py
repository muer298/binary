#!/usr/bin/env python3
import argparse
import collections
import json
from pathlib import Path

from evidence_objective_v2 import extract_node_ids
from io_utils import dump, load_config, rows, sha256


def write_jsonl(path, iterable):
    count = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in iterable:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_root"])
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)

    actual_sha = sha256(cfg["global_manifest"])
    if actual_sha != cfg["expected_manifest_sha256"]:
        raise RuntimeError("global manifest SHA256 mismatch: %s" % actual_sha)

    split_contract = json.loads(Path(cfg["global_split_contract"]).read_text())
    retrain = json.loads(Path(cfg["clean_retrain_contract"]).read_text())
    sample_roles = split_contract["sample_roles"]
    train_samples = {sample for sample, role in sample_roles.items() if role == "TRAIN"}
    if len(train_samples) != 16:
        raise RuntimeError("expected 16 GLOBAL TRAIN samples, got %d" % len(train_samples))

    sample_groups = {}
    function_groups = collections.defaultdict(set)
    for row in rows(cfg["global_inventory"]):
        sample = str(row["sample_id"])
        group = str(row["partition_group_id"])
        previous = sample_groups.setdefault(sample, group)
        if previous != group:
            raise RuntimeError("sample maps to multiple isolation groups: %s" % sample)
        function_groups[(sample, str(row["function_entry"]))].add(group)

    encoder_train_groups = set(retrain["encoder_train_groups"])
    inner_dev_groups = set(retrain["encoder_inner_dev_groups"])
    deny_groups = set(retrain["stage2_DEV_CALIBRATION_TEST_deny_groups"])
    if encoder_train_groups & inner_dev_groups:
        raise RuntimeError("encoder train/inner-dev group overlap")
    if (encoder_train_groups | inner_dev_groups) & deny_groups:
        raise RuntimeError("outer deny group appears in encoder split")

    manifest_rows = []
    split_samples = {"train": set(), "validation": set()}
    for sample in sorted(train_samples):
        group = sample_groups[sample]
        if group in encoder_train_groups:
            split = "train"
        elif group in inner_dev_groups:
            split = "validation"
        else:
            raise RuntimeError("GLOBAL TRAIN sample has no encoder assignment: %s" % sample)
        split_samples[split].add(sample)
        manifest_rows.append({
            "sample_id": sample,
            "split": split,
            "group_id": group,
            "global_role": "TRAIN",
        })
    write_jsonl(data / "encoder_inner_split_manifest.jsonl", manifest_rows)

    outer_samples = {sample for sample, role in sample_roles.items() if role != "TRAIN"}
    deny = {
        "schema": "seraph_hsym_outer_denylist_v0_2_8",
        "outer_sample_ids": sorted(outer_samples),
        "outer_groups": sorted(deny_groups),
        "roles": {role: sorted(s for s, r in sample_roles.items() if r == role)
                  for role in ("DEV", "CALIBRATION", "TEST")},
    }
    dump(data / "outer_denylist.json", deny)

    eligible_ids = set()
    eligible_by_sample = collections.Counter()
    for node in rows(cfg["eligible_nodes"]):
        node_id = str(node.get("node_id") or "")
        if node_id:
            eligible_ids.add(node_id)
            eligible_by_sample[str(node.get("sample_id") or node_id.split("::", 1)[0])] += 1

    raw_by_sample = collections.Counter()
    raw_type_by_role = collections.Counter()
    for node in rows(cfg["raw_nodes"]):
        sample = str(node.get("sample_id") or "")
        raw_by_sample[sample] += 1
        raw_type_by_role[(sample_roles.get(sample, "UNKNOWN"), str(node.get("node_type") or "unknown"))] += 1

    def train_nodes():
        for node in rows(cfg["eligible_nodes"]):
            sample = str(node.get("sample_id") or "")
            if sample in train_samples:
                yield node

    train_node_count = write_jsonl(data / "train_nodes.jsonl", train_nodes())

    edge_stats = collections.Counter()
    def train_edges():
        for edge in rows(cfg["raw_edges"]):
            edge_stats["raw_edges"] += 1
            sample = str(edge.get("sample_id") or "")
            if sample not in train_samples:
                edge_stats["excluded_outer_sample"] += 1
                continue
            if edge.get("source_node") not in eligible_ids or edge.get("target_node") not in eligible_ids:
                edge_stats["excluded_ineligible_endpoint"] += 1
                continue
            edge_stats["retained"] += 1
            yield edge
    train_edge_count = write_jsonl(data / "train_edges.jsonl", train_edges())

    neighborhood_stats = collections.Counter()
    def compact_train_neighborhoods():
        for row in rows(cfg["neighborhoods"]):
            neighborhood_stats["raw_records"] += 1
            sample = str(row.get("sample_id") or "")
            if sample not in train_samples:
                neighborhood_stats["excluded_outer_sample"] += 1
                continue
            low_ids = [x for x in extract_node_ids(row.get("low_nodes") or row.get("low_node_ids")) if x in eligible_ids]
            high_ids = [x for x in extract_node_ids(row.get("high_nodes") or row.get("high_node_ids")) if x in eligible_ids]
            source = str(row.get("source_node") or "")
            target = str(row.get("target_node") or "")
            if source in eligible_ids and source not in low_ids:
                low_ids.append(source)
            if target in eligible_ids and target not in high_ids:
                high_ids.append(target)
            compact = {
                "anchor_id": row.get("anchor_id"),
                "sample_id": sample,
                "evidence_type": row.get("evidence_type"),
                "strength": row.get("strength"),
                "source_node": source,
                "target_node": target,
                "source_type": row.get("source_type"),
                "target_type": row.get("target_type"),
                "low_node_ids": low_ids,
                "high_node_ids": high_ids,
            }
            if not low_ids or not high_ids:
                neighborhood_stats["retained_but_unusable"] += 1
            else:
                neighborhood_stats["retained_usable"] += 1
            yield compact
    train_neighborhood_count = write_jsonl(
        data / "train_neighborhoods_compact.jsonl", compact_train_neighborhoods()
    )

    report = {
        "schema": "seraph_clean_data_preparation_v0_2_8",
        "global_manifest_sha256": actual_sha,
        "global_train_sample_count": len(train_samples),
        "encoder_train_sample_count": len(split_samples["train"]),
        "encoder_inner_dev_sample_count": len(split_samples["validation"]),
        "encoder_train_groups": sorted(encoder_train_groups),
        "encoder_inner_dev_groups": sorted(inner_dev_groups),
        "outer_deny_groups": sorted(deny_groups),
        "outer_sample_count": len(outer_samples),
        "raw_node_count_global_train": sum(raw_by_sample[s] for s in train_samples),
        "raw_node_count_all": sum(raw_by_sample.values()),
        "eligible_node_count_global_train": sum(eligible_by_sample[s] for s in train_samples),
        "eligible_node_count_all": len(eligible_ids),
        "excluded_node_count_global_train": sum(raw_by_sample[s] for s in train_samples) - train_node_count,
        "train_node_file_count": train_node_count,
        "train_edge_file_count": train_edge_count,
        "train_neighborhood_file_count": train_neighborhood_count,
        "edge_stats": dict(edge_stats),
        "neighborhood_stats": dict(neighborhood_stats),
        "raw_node_types_by_role": {"%s/%s" % key: value for key, value in sorted(raw_type_by_role.items())},
        "outer_data_training_files": 0,
        "passed": True,
    }
    dump(out / "data_preparation_report.json", report)

    lines = [
        "# Encoder inner split report", "",
        "The encoder split is derived only from GLOBAL TRAIN and is frozen before training.", "",
        "| Partition | Isolation groups | Samples |",
        "|---|---:|---:|",
        "| encoder_train | %d | %d |" % (len(encoder_train_groups), len(split_samples["train"])),
        "| encoder_inner_dev | %d | %d |" % (len(inner_dev_groups), len(split_samples["validation"])),
        "| outer deny-list | %d | %d |" % (len(deny_groups), len(outer_samples)), "",
        "Project/binary groups are disjoint. Because every function belongs to exactly one sample/group, function isolation follows from the same firewall.", "",
        "Outer DEV/CALIBRATION/TEST rows are absent from all trainer inputs.",
    ]
    (out / "ENCODER_INNER_SPLIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
