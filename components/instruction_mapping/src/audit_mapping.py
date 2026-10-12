from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from common import address_int, centered_window, dump_json, load_json, norm_entry, rows, write_rows


LEVELS = ("L0_EXACT_TOKEN", "L1_SYNTAX_EXACT_PC_SET", "L2_VERIFIED_NODE_RANGE", "L3_UNKNOWN")


def pct(numerator, denominator):
    return (float(numerator) / denominator) if denominator else 0.0


def select_smallest(items, key):
    return min(items, key=key) if items else None


def char_token_indices(offsets, spans):
    indices = []
    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if any(end > span["start"] and start < span["end"] for span in spans):
            indices.append(index)
    return indices


def node_span(node, text):
    start, end = node.get("char_start"), node.get("char_end")
    if start is None or end is None or not (0 <= start < end <= len(text)):
        return None
    return {"start": start, "end": end, "text": text[start:end]}


def classify(function, pc):
    text = function["function_text"]
    flat_tokens = [token for line in function["lines"] for token in line["tokens"]]
    direct = []
    for token in flat_tokens:
        if any(address_int(anchor.get("address")) == pc for anchor in token.get("anchors", [])):
            if token["char_end"] > token["char_start"]:
                direct.append(token)
    if direct:
        spans = [
            {"start": token["char_start"], "end": token["char_end"],
             "text": text[token["char_start"]:token["char_end"]]}
            for token in sorted(direct, key=lambda token: (token["char_start"], token["char_end"]))
        ]
        return "L0_EXACT_TOKEN", spans, {
            "method": "EXACT_GHIDRA_TOKEN_ANCHOR",
            "anchored_flat_token_indices": [token["flat_token_index"] for token in direct],
            "anchors": [anchor for token in direct for anchor in token.get("anchors", []) if address_int(anchor.get("address")) == pc],
        }, None

    nonroot = [node for node in function["syntax_nodes"] if not node.get("is_root")]
    exact_nodes = [
        node for node in nonroot
        if any(address_int(value) == pc for value in node.get("direct_pcode_addresses", []))
        and node_span(node, text) is not None
    ]
    node = select_smallest(exact_nodes, lambda item: (
        item["char_end"] - item["char_start"],
        len(item.get("descendant_token_indices", [])), item["node_id"],
    ))
    if node:
        span = node_span(node, text)
        return "L1_SYNTAX_EXACT_PC_SET", [span], {
            "method": "CLANG_SYNTAX_NODE_EXACT_PC_SET",
            "node_id": node["node_id"], "node_class": node["class"],
            "node_min_address": node["min_address"], "node_max_address": node["max_address"],
            "node_exact_pc_set": node["direct_pcode_addresses"],
            "descendant_token_indices": node["descendant_token_indices"],
        }, None

    listing = {address_int(item["address"]): item for item in function.get("address_catalog", [])}
    pc_is_instruction = pc in listing
    range_nodes = []
    if pc_is_instruction:
        for candidate in nonroot:
            minimum, maximum = address_int(candidate.get("min_address")), address_int(candidate.get("max_address"))
            if minimum is not None and maximum is not None and minimum <= pc <= maximum and node_span(candidate, text):
                range_nodes.append(candidate)
    node = select_smallest(range_nodes, lambda item: (
        address_int(item["max_address"]) - address_int(item["min_address"]),
        item["char_end"] - item["char_start"],
        len(item.get("descendant_token_indices", [])), item["node_id"],
    ))
    if node:
        span = node_span(node, text)
        return "L2_VERIFIED_NODE_RANGE", [span], {
            "method": "CLANG_SYNTAX_NODE_VERIFIED_ADDRESS_RANGE",
            "node_id": node["node_id"], "node_class": node["class"],
            "node_min_address": node["min_address"], "node_max_address": node["max_address"],
            "listing_exact_pc": listing[pc],
            "descendant_token_indices": node["descendant_token_indices"],
        }, None

    if not pc_is_instruction:
        reason = "PC_NOT_IN_GHIDRA_FUNCTION_LISTING"
    else:
        roots = [node for node in function["syntax_nodes"] if node.get("is_root")]
        root_contains = any(
            address_int(node.get("min_address")) is not None
            and address_int(node.get("min_address")) <= pc <= address_int(node.get("max_address"))
            for node in roots
        )
        nonroot_contains = any(
            address_int(node.get("min_address")) is not None
            and address_int(node.get("min_address")) <= pc <= address_int(node.get("max_address"))
            for node in nonroot
        )
        if nonroot_contains:
            reason = "NONROOT_RANGE_HAS_NO_NONEMPTY_PSEUDOC_SPAN"
        elif root_contains:
            reason = "ONLY_FUNCTION_ROOT_RANGE_CONTAINS_PC"
        else:
            reason = "DECOMPILER_ELIDED_NO_TRACEABLE_SYNTAX_NODE"
    return "L3_UNKNOWN", [], {"method": None, "listing_exact_pc": listing.get(pc)}, reason


def counter_table(records, field):
    counter = Counter(record[field] for record in records)
    return {key: counter.get(key, 0) for key in LEVELS}


def outcome_by_level(items):
    counts = Counter(item["mapping_level"] for item in items)
    total = len(items)
    return {
        "n": total,
        "by_level": {level: {"n": counts[level], "proportion": pct(counts[level], total)} for level in LEVELS},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    output_root = Path(cfg["output_root"])
    export_report = load_json(output_root / "ghidra_syntax_ancestry_export_report.json")
    if export_report.get("status") != "PASSED" or export_report.get("sample_count") != int(cfg["sample_count"]):
        raise SystemExit("Ghidra ancestry export did not pass")

    functions = {}
    for sample in export_report["samples"]:
        export = load_json(sample["output"])
        for function in export["functions"]:
            key = (sample["sample_id"], norm_entry(function["function_entry"]))
            if key in functions:
                raise ValueError("duplicate exported function: %r" % (key,))
            functions[key] = function

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"], local_files_only=True, use_fast=True)
    tokenized = {}
    for key, function in functions.items():
        encoded = tokenizer(function["function_text"], add_special_tokens=False, return_offsets_mapping=True)
        tokenized[key] = {"ids": encoded["input_ids"], "offsets": encoded["offset_mapping"]}

    legacy = list(rows(cfg["v021_mapping"]))
    legacy_by_query = {}
    for record in legacy:
        key = (record["sample_id"], record["query_id"])
        prior = legacy_by_query.setdefault(key, record)
        if (prior["mapping_status"], prior["instruction_address"], prior["function_entry"]) != (
            record["mapping_status"], record["instruction_address"], record["function_entry"]
        ):
            raise ValueError("per-module legacy mapping disagreement: %r" % (key,))
    if len(legacy_by_query) * 4 != len(legacy):
        raise ValueError("expected four module rows per instruction query")

    query_objects = {}
    samples_root = Path(cfg["task1_dir"]) / "samples"
    for query_path in samples_root.glob("*/queries.jsonl"):
        for query in rows(query_path):
            query_objects.setdefault((query["sample_id"], query["query_id"]), query)

    unique_results = {}
    for key, old in sorted(legacy_by_query.items()):
        function_key = (old["sample_id"], norm_entry(old["function_entry"]))
        function = functions.get(function_key)
        if function is None or function.get("status") != "OK":
            raise ValueError("missing successful function export: %r" % (function_key,))
        pc = address_int(old["instruction_address"])
        level, spans, evidence, reason = classify(function, pc)
        tok = tokenized[function_key]
        indices = char_token_indices(tok["offsets"], spans)
        if level != "L3_UNKNOWN" and not indices:
            level, spans, evidence, reason = "L3_UNKNOWN", [], evidence, "VERIFIED_CHAR_SPAN_HAS_NO_MODEL_TOKEN_OVERLAP"
        window = centered_window(len(tok["ids"]), indices, int(cfg["max_length"])) if indices else None
        if level != "L3_UNKNOWN" and window is None:
            reason = "VERIFIED_TARGET_SPAN_EXCEEDS_MAX_LENGTH"
        unique_results[key] = {
            "sample_id": old["sample_id"], "query_id": old["query_id"],
            "function_entry": norm_entry(old["function_entry"]),
            "instruction_address": old["instruction_address"],
            "mapping_level": level, "mapping_status": "VERIFIED" if level != "L3_UNKNOWN" else "UNKNOWN",
            "char_spans": spans, "token_indices": indices, "token_count": len(tok["ids"]),
            "window": window, "unknown_reason": reason, "evidence": evidence,
            "legacy_mapping_status": old["mapping_status"],
            "legacy_first_512_usable": bool(old.get("token_indices_within_reference_512")),
        }

    output_rows = []
    for record in legacy:
        unique = unique_results[(record["sample_id"], record["query_id"])]
        output_rows.append(dict(unique, schema="seraph_instruction_mapping_v0_2_2", module=record["module"]))
    write_rows(output_root / "instruction_mapping_v0_2_2.jsonl", output_rows)

    unique = list(unique_results.values())
    total = len(unique)
    counts = Counter(item["mapping_level"] for item in unique)
    verified = sum(counts[level] for level in LEVELS[:-1])
    centered_usable = sum(item["window"] is not None for item in unique)
    old_exact = sum(item["legacy_mapping_status"] == "EXACT" for item in unique)
    old_first = sum(item["legacy_first_512_usable"] for item in unique)
    old_unknown = [item for item in unique if item["legacy_mapping_status"] == "UNKNOWN"]

    module_counts = {}
    for module in sorted({row["module"] for row in output_rows}):
        subset = [row for row in output_rows if row["module"] == module]
        module_counts[module] = {
            "total": len(subset), "levels": counter_table(subset, "mapping_level"),
            "verified": sum(row["mapping_level"] != "L3_UNKNOWN" for row in subset),
            "verified_coverage": pct(sum(row["mapping_level"] != "L3_UNKNOWN" for row in subset), len(subset)),
            "centered_window_usable": sum(row["window"] is not None for row in subset),
            "centered_window_coverage": pct(sum(row["window"] is not None for row in subset), len(subset)),
        }

    mapping_report = {
        "schema": "seraph_instruction_mapping_report_v0_2_2", "total_unique_instruction_queries": total,
        "legacy": {"exact": old_exact, "unknown": total - old_exact, "first_512_usable": old_first},
        "levels": {level: counts[level] for level in LEVELS},
        "verified": verified, "verified_coverage": pct(verified, total),
        "module_counts": module_counts,
        "mapping_policy": [
            "L0 exact token-local P-code/address anchor", "L1 non-root syntax node exact descendant PC set",
            "L2 non-root syntax node real address range containing an exact listing PC", "L3 unknown",
        ],
        "forbidden_fallbacks_used": [], "test_evaluated": False,
    }
    dump_json(output_root / "instruction_mapping_report_v0_2_2.json", mapping_report)

    span_report = {
        "schema": "seraph_span_centered_window_report_v0_2_2", "max_length": int(cfg["max_length"]),
        "legacy_first_512_usable": old_first, "legacy_first_512_coverage": pct(old_first, total),
        "legacy_exact_total": old_exact,
        "legacy_exact_centered_usable": sum(
            item["window"] is not None for item in unique if item["legacy_mapping_status"] == "EXACT"
        ),
        "verified_mapping_total": verified, "centered_window_usable": centered_usable,
        "centered_window_coverage": pct(centered_usable, total),
        "window_contract": "target span is fully enclosed; global/local indices and [start,end) are recorded",
        "test_evaluated": False,
    }
    dump_json(output_root / "span_centered_window_report.json", span_report)

    reason_counts = Counter(
        item["unknown_reason"] if item["mapping_level"] == "L3_UNKNOWN"
        else "RECOVERED_AS_" + item["mapping_level"]
        for item in old_unknown
    )
    final_unknown_reasons = Counter(
        item["unknown_reason"] for item in old_unknown if item["mapping_level"] == "L3_UNKNOWN"
    )
    recovery_levels = Counter(item["mapping_level"] for item in old_unknown)
    inventory = {
        "schema": "seraph_v021_unknown_reason_inventory_v0_2_2", "legacy_unknown_total": len(old_unknown),
        "reclassified_levels": {level: recovery_levels[level] for level in LEVELS},
        "classification_of_all_legacy_unknown": dict(sorted(reason_counts.items())),
        "final_unknown_reasons": dict(sorted(final_unknown_reasons.items())),
        "final_unknown": recovery_levels["L3_UNKNOWN"],
        "recovered": len(old_unknown) - recovery_levels["L3_UNKNOWN"],
        "test_evaluated": False,
    }
    dump_json(output_root / "unmapped_reason_inventory.json", inventory)

    buckets = defaultdict(list)
    for item in unique:
        buckets[item["mapping_level"]].append(item)
    examples = []
    per_level_target = 8
    for level in LEVELS:
        for item in buckets[level][:per_level_target]:
            function = functions[(item["sample_id"], norm_entry(item["function_entry"]))]
            query = query_objects.get((item["sample_id"], item["query_id"]), {})
            catalog = {address_int(row["address"]): row for row in function["address_catalog"]}
            instruction = catalog.get(address_int(item["instruction_address"]))
            examples.append({
                "sample_id": item["sample_id"], "query_id": item["query_id"],
                "function_entry": item["function_entry"], "pc": item["instruction_address"],
                "assembly": instruction.get("assembly") if instruction else None,
                "ghidra_pcode": instruction.get("pcode") if instruction else [],
                "query_n_pcode": query.get("instruction"),
                "full_pseudo_c_context": function["function_text"],
                "mapped_substrings": item["char_spans"], "global_token_indices": item["token_indices"],
                "centered_window": item["window"], "mapping_level": level,
                "evidence": item["evidence"], "unknown_reason": item["unknown_reason"],
            })
    if len(examples) < 30:
        already = {(x["sample_id"], x["query_id"]) for x in examples}
        for item in unique:
            if (item["sample_id"], item["query_id"]) in already:
                continue
            function = functions[(item["sample_id"], norm_entry(item["function_entry"]))]
            query = query_objects.get((item["sample_id"], item["query_id"]), {})
            catalog = {address_int(row["address"]): row for row in function["address_catalog"]}
            instruction = catalog.get(address_int(item["instruction_address"]))
            examples.append({
                "sample_id": item["sample_id"], "query_id": item["query_id"], "function_entry": item["function_entry"],
                "pc": item["instruction_address"], "assembly": instruction.get("assembly") if instruction else None,
                "ghidra_pcode": instruction.get("pcode") if instruction else [], "query_n_pcode": query.get("instruction"),
                "full_pseudo_c_context": function["function_text"], "mapped_substrings": item["char_spans"],
                "global_token_indices": item["token_indices"], "centered_window": item["window"],
                "mapping_level": item["mapping_level"], "evidence": item["evidence"],
                "unknown_reason": item["unknown_reason"],
            })
            if len(examples) >= 30:
                break
    dump_json(output_root / "mapping_manual_examples.json", {
        "schema": "seraph_mapping_manual_examples_v0_2_2", "n": len(examples),
        "level_counts": dict(Counter(item["mapping_level"] for item in examples)), "examples": examples,
    })
    dump_json(output_root / "unmapped_reason_examples.json", {
        "schema": "seraph_unmapped_reason_examples_v0_2_2",
        "examples": [item for item in examples if item["mapping_level"] == "L3_UNKNOWN"],
    })

    symbolic = load_json(Path(cfg["v020_arms_root"]) / "symbolic_only" / "training_report.json")
    frozen = load_json(Path(cfg["v020_arms_root"]) / "frozen_llm_gate" / "training_report.json")
    symbolic_type = {item["manifest_key"]: item for item in symbolic["validation_details"] if item["module"] == "TypeSem"}
    frozen_type = {item["manifest_key"]: item for item in frozen["validation_details"] if item["module"] == "TypeSem"}
    if set(symbolic_type) != set(frozen_type):
        raise ValueError("TypeSem validation manifests differ")
    type_mapping = {(row["sample_id"], row["query_id"]): row for row in output_rows if row["module"] == "TypeSem"}
    compared, harmed, corrected = [], [], []
    for manifest_key in sorted(symbolic_type):
        left, right = symbolic_type[manifest_key], frozen_type[manifest_key]
        mapping = type_mapping[(left["sample_id"], left["query_id"])]
        item = {
            "manifest_key": manifest_key, "sample_id": left["sample_id"], "query_id": left["query_id"],
            "symbolic_rank": left["rank"], "frozen_llm_rank": right["rank"],
            "mapping_level": mapping["mapping_level"], "mapping_status": mapping["mapping_status"],
        }
        compared.append(item)
        if left["rank"] == 1 and right["rank"] != 1:
            harmed.append(item)
        if left["rank"] != 1 and right["rank"] == 1:
            corrected.append(item)
    granularity = {
        "schema": "seraph_typesem_mapping_granularity_audit_v0_2_2",
        "definition": {"harmed": "symbolic rank=1 and frozen-LLM rank!=1", "corrected": "symbolic rank!=1 and frozen-LLM rank=1"},
        "all_typesem_validation": outcome_by_level(compared), "harmed": outcome_by_level(harmed),
        "corrected": outcome_by_level(corrected), "harmed_expected_reference": 95,
        "corrected_expected_reference": 26, "test_evaluated": False,
    }
    dump_json(output_root / "typesem_granularity_audit.json", granularity)

    threshold = float(cfg["verified_coverage_threshold"])
    type_report = module_counts["TypeSem"]
    ready = (
        type_report["verified_coverage"] >= threshold
        and type_report["centered_window_coverage"] >= threshold
        and len(examples) >= 30
        and export_report["status"] == "PASSED"
    )
    status = "READY_FOR_V0_2_3_SMOKE" if ready else "BLOCKED"
    readiness = {
        "schema": "seraph_v0_2_2_mapping_readiness", "status": status, "threshold": threshold,
        "checks": {
            "ghidra_export_passed": export_report["status"] == "PASSED",
            "typesem_verified_coverage": type_report["verified_coverage"],
            "typesem_centered_window_coverage": type_report["centered_window_coverage"],
            "manual_examples": len(examples), "harmed_count_matches_reference": len(harmed) == 95,
            "corrected_count_matches_reference": len(corrected) == 26,
            "test_evaluated": False,
        },
        "APISem": cfg["apisem_status"], "constraints": cfg["constraint_status"],
    }
    dump_json(output_root / "v0_2_2_readiness.json", readiness)

    summary = """# SERAPH v0.2.2 instruction-mapping audit

## Boundary

- Mapping-only audit; no joint training, LoRA update, gate/LR tuning, SSQ, Stage-3, or test evaluation.
- APISem: `{apisem}`
- Constraints: `{constraints}`
- Forbidden heuristic fallbacks used: none.

## Ghidra ancestry export

- Samples: {samples}/20
- Functions: {functions}/{functions_ok} successful
- Syntax nodes: {nodes:,}
- Nodes with exact PC sets: {exact_nodes:,}
- Nodes with address ranges: {range_nodes:,}

## Mapping levels (unique instruction queries)

| Level | Count | Meaning |
|---|---:|---|
| L0 | {l0:,} | exact token-local P-code/address anchor |
| L1 | {l1:,} | non-root syntax node exact descendant PC set |
| L2 | {l2:,} | verified non-root syntax-node address range containing an exact listing PC |
| L3 | {l3:,} | unknown; no permitted evidence path |

- Total: {total:,}
- Verified L0+L1+L2: {verified:,} ({coverage:.2%})
- Legacy v0.2.1 exact: {old_exact:,}; unknown: {old_unknown:,}
- Recovered from legacy unknown: {recovered:,}

## 512-token window

- Legacy first-512 usable: {old_first:,} ({old_first_cov:.2%})
- Verified mappings usable with span-centred window: {centered:,} ({centered_cov:.2%})
- Every accepted window records global/local token indices and fully encloses the verified target span.

## Per-module coverage

| Module | Total | L0 | L1 | L2 | L3 | Verified |
|---|---:|---:|---:|---:|---:|---:|
{module_rows}

## Legacy unknown reasons

{reason_rows}

## TypeSem validation granularity correlation (v0.2.0, no test)

- Compared validation records: {type_n}
- Harmed (symbolic R1 correct, frozen-LLM not R1): {harmed} (reference 95)
- Corrected (symbolic not R1, frozen-LLM R1): {corrected} (reference 26)
- Harmed by level: `{harmed_levels}`
- Corrected by level: `{corrected_levels}`

This is an association audit, not evidence that mapping granularity caused the rank changes.

## Manual evidence

- Stratified examples: {examples} (full pseudo-C, PC, assembly, query N-pcode, Ghidra pcode, spans, token indices, level, evidence).

## Final status

`{status}`
""".format(
        apisem=cfg["apisem_status"], constraints=cfg["constraint_status"], samples=export_report["sample_count"],
        functions=export_report["aggregate"]["functions"], functions_ok=export_report["aggregate"]["functions_ok"],
        nodes=export_report["aggregate"]["syntax_nodes"], exact_nodes=export_report["aggregate"]["syntax_nodes_with_exact_pc_set"],
        range_nodes=export_report["aggregate"]["syntax_nodes_with_address_range"],
        l0=counts[LEVELS[0]], l1=counts[LEVELS[1]], l2=counts[LEVELS[2]], l3=counts[LEVELS[3]],
        total=total, verified=verified, coverage=pct(verified, total), old_exact=old_exact, old_unknown=total-old_exact,
        recovered=len(old_unknown)-recovery_levels["L3_UNKNOWN"], old_first=old_first, old_first_cov=pct(old_first,total),
        centered=centered_usable, centered_cov=pct(centered_usable,total),
        module_rows="\n".join(
            "| {0} | {1:,} | {2:,} | {3:,} | {4:,} | {5:,} | {6:.2%} |".format(
                module, data["total"], data["levels"][LEVELS[0]], data["levels"][LEVELS[1]],
                data["levels"][LEVELS[2]], data["levels"][LEVELS[3]], data["verified_coverage"]
            ) for module, data in sorted(module_counts.items())
        ),
        reason_rows="\n".join("- `{}`: {:,}".format(key, value) for key, value in sorted(reason_counts.items())),
        type_n=len(compared), harmed=len(harmed), corrected=len(corrected),
        harmed_levels=json.dumps(outcome_by_level(harmed)["by_level"], sort_keys=True),
        corrected_levels=json.dumps(outcome_by_level(corrected)["by_level"], sort_keys=True),
        examples=len(examples), status=status,
    )
    (output_root / "V0_2_2_MAPPING_SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": status, "mapping": mapping_report, "span_window": span_report,
                      "unknown_inventory": inventory, "typesem": granularity, "manual_examples": len(examples)},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
