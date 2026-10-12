#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safely enrich an existing selected NODES JSONL with LayoutSem evidence.

The script never reads Gold and never overwrites its input.  It follows each
node's ``json_path`` and ``source_path`` back to the original semantic JSON,
recovers fixed ``InstrSem.effects.memory_accesses`` offsets, aggregates them by
function-local base object, and recovers candidate field offsets from bounded
LayoutSem schemas.  Node IDs, row order, and all unrelated attributes are
preserved, so an existing 2.2.2 reasoning scope can be reused after contract
checks.

v6.7.4 adds a conservative escape hatch for an irrecoverably missing historic
semantic JSON.  Missing sources must be named explicitly in an allow-list and
are left completely unchanged.  A second unlisted missing source, a malformed
JSON, a permission error, or an unused/typoed allow-list entry still aborts the
run.  This is not synthetic evidence: it merely lets the available sources be
enriched while retaining the unavailable sample in the evaluation denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


IMPLEMENTATION_REVISION = (
    "v6_7_4_controlled_missing_source_layout_evidence_bridge"
)

NODE_ID_KEYS = (
    "id", "node_id", "semantic_id", "func_id", "function_id",
    "block_id", "api_id", "layout_id", "_dict_key",
)
QUERY_OFFSET_KEYS = (
    "observed_field_offsets", "field_offsets_observed", "observed_offsets",
    "accessed_offsets", "memory_offsets", "access_offsets",
)
CANDIDATE_OFFSET_KEYS = (
    "field_offsets", "offsets", "offset_set", "observed_field_offsets",
    "field_offsets_observed", "inferred_field_offsets",
    "field_offsets_inferred", "inferred_offsets",
)
WIDTH_KEYS = (
    "access_widths", "field_widths", "widths", "access_width",
    "field_width", "byte_width", "width", "byte_size", "size_bytes",
    "size",
)
MEMORY_ACCESS_CONTAINERS = ("memory_accesses", "memory_access_patterns")
FIELD_CONTAINERS = ("fields", "members", "layout_fields", "field_layouts")
OFFSET_KEYS = ("offset", "byte_offset", "displacement", "disp")
NESTED_MAP_KEYS = (
    "attrs", "metadata", "candidate_metadata", "payload", "raw", "layout",
    "memory_layout", "effects", "propagated_constraints",
)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    "%s:%d is not a JSON object" % (path, line_number)
                )
            yield row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_path(obj: Any, path: str) -> Any:
    current = obj
    if not path:
        return current
    for part in str(path).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                copied = dict(item)
                copied.setdefault("_dict_key", key)
                items.append(copied)
            else:
                items.append(item)
        return items
    return [value]


def object_raw_id(obj: Any, node_type: str, index: int) -> str:
    if isinstance(obj, Mapping):
        for key in NODE_ID_KEYS:
            if obj.get(key):
                return str(obj.get(key))
        for key in (
            "function_entry", "func_entry", "entry", "address", "location",
        ):
            if obj.get(key) not in (None, ""):
                name = obj.get("function_name") or obj.get("func_name") or obj.get("name") or ""
                return "%s_%s_%s" % (node_type, obj.get(key), name)
    return "%s_%d" % (node_type, index)


def remap_path(path: str, remaps: Sequence[Tuple[str, str]]) -> Path:
    text = str(path)
    for old, new in remaps:
        if text == old or text.startswith(old.rstrip("/") + "/"):
            text = new.rstrip("/") + text[len(old.rstrip("/")):]
            break
    return Path(text)


def parse_remaps(values: Sequence[str]) -> List[Tuple[str, str]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--json-root-remap must use OLD=NEW: %r" % value
            )
        old, new = value.split("=", 1)
        if not old or not new:
            raise ValueError(
                "--json-root-remap must use non-empty OLD=NEW"
            )
        result.append((old.rstrip("/"), new.rstrip("/")))
    return result


def normalise_path_text(value: Any) -> str:
    """Return a stable non-strict absolute spelling for path comparison."""
    return str(Path(str(value)).expanduser().resolve(strict=False))


def load_allowed_missing_sources(
    direct_values: Sequence[str],
    list_files: Sequence[str],
) -> set[str]:
    """Load exact source paths from CLI values and newline-delimited files."""
    values: List[str] = []
    values.extend(str(value) for value in direct_values if str(value).strip())
    for raw_file in list_files:
        path = Path(raw_file).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(
                "allowed-missing-source file missing or empty: %s" % path
            )
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                text = line.strip()
                if text and not text.startswith("#"):
                    values.append(text)
    return {normalise_path_text(value) for value in values}


def finite_integer(
    value: Any,
    minimum: int = 0,
    maximum: int = 65536,
) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return None
            number = int(text, 0)
        else:
            floating = float(value)
            if not math.isfinite(floating):
                return None
            number = int(floating)
            if abs(floating - number) > 1e-6:
                return None
    except (TypeError, ValueError, OverflowError):
        return None
    if number < minimum or number > maximum:
        return None
    return number


def explicit_false(value: Any) -> bool:
    if value is False or value == 0:
        return True
    return isinstance(value, str) and value.strip().lower() in {
        "false", "no", "off", "0", "dynamic", "unknown",
    }


def numbers(value: Any) -> List[Any]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (int, float, str)):
        return [value]
    if isinstance(value, Mapping):
        result = []
        for key in ("offset", "value", "width", "size", "bytes"):
            if key in value:
                result.extend(numbers(value.get(key)))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(numbers(item))
        return result
    return []


def normalise_values(
    values: Iterable[Any],
    minimum: int,
    maximum: int,
) -> List[int]:
    result = set()
    for value in values:
        for item in numbers(value):
            number = finite_integer(item, minimum=minimum, maximum=maximum)
            if number is not None:
                result.add(number)
    return sorted(result)


def nested_maps(record: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    seen = set()
    stack: List[Mapping[str, Any]] = [record]
    while stack:
        current = stack.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        yield current
        for key in NESTED_MAP_KEYS:
            child = current.get(key)
            if isinstance(child, Mapping):
                stack.append(child)


def container_records(
    record: Mapping[str, Any],
    names: Sequence[str],
) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    seen = set()
    for mapping in nested_maps(record):
        for name in names:
            value = mapping.get(name)
            if isinstance(value, Mapping):
                items = (value,)
            elif isinstance(value, list):
                items = value
            else:
                continue
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                identity = id(item)
                if identity in seen:
                    continue
                seen.add(identity)
                yield name, item


def collect_direct(
    record: Mapping[str, Any],
    keys: Sequence[str],
    minimum: int,
    maximum: int,
) -> List[int]:
    values = []
    for mapping in nested_maps(record):
        for key in keys:
            if key in mapping:
                values.append(mapping.get(key))
    return normalise_values(values, minimum, maximum)


def extract_instr_evidence(
    obj: Mapping[str, Any],
    maximum: int,
) -> Dict[str, Any]:
    offsets = set(collect_direct(obj, QUERY_OFFSET_KEYS, 0, maximum))
    widths = set(collect_direct(obj, WIDTH_KEYS, 1, maximum))
    bases = set()
    kinds = set()
    compact = []

    for container, access in container_records(
        obj, MEMORY_ACCESS_CONTAINERS,
    ):
        if explicit_false(access.get("is_fixed")):
            continue
        offset = None
        for key in OFFSET_KEYS:
            if key in access:
                offset = finite_integer(access.get(key), 0, maximum)
                if offset is not None:
                    break
        if offset is None:
            continue
        width = None
        for key in WIDTH_KEYS:
            if key in access:
                width = finite_integer(access.get(key), 1, maximum)
                if width is not None:
                    break
        base = str(access.get("base") or "").strip()
        kind = str(
            access.get("access") or access.get("kind") or ""
        ).strip().lower()
        offsets.add(offset)
        if width is not None:
            widths.add(width)
        if base:
            bases.add(base)
        if kind:
            kinds.add(kind)
        compact.append({
            "site": access.get("site"),
            "base": base or None,
            "offset": offset,
            "access": kind or None,
            "width": width,
            "byte_width": width,
            "is_fixed": True,
            "source": "%s[].offset" % container,
        })

    function_entry = str(
        obj.get("function_entry") or obj.get("func_entry") or ""
    ).strip().lower()
    return {
        "function_entry": function_entry,
        "local_offsets": sorted(offsets),
        "widths": sorted(widths),
        "bases": sorted(bases),
        "kinds": sorted(kinds),
        "accesses": compact,
    }


def extract_layout_candidate_evidence(
    obj: Mapping[str, Any],
    maximum: int,
) -> Dict[str, Any]:
    offsets = set(collect_direct(
        obj, CANDIDATE_OFFSET_KEYS, 0, maximum,
    ))
    widths = set(collect_direct(obj, WIDTH_KEYS, 1, maximum))
    member_sources = set()
    for container, member in container_records(obj, FIELD_CONTAINERS):
        for key in OFFSET_KEYS:
            if key not in member:
                continue
            offset = finite_integer(member.get(key), 0, maximum)
            if offset is not None:
                offsets.add(offset)
                member_sources.add("%s[].%s" % (container, key))
                break
        for key in WIDTH_KEYS:
            if key not in member:
                continue
            width = finite_integer(member.get(key), 1, maximum)
            if width is not None:
                widths.add(width)
                break
    return {
        "offsets": sorted(offsets),
        "widths": sorted(widths),
        "member_sources": sorted(member_sources),
    }


def build_source_objects(
    data: Mapping[str, Any],
    source_path: str,
    node_type: str,
) -> Dict[str, Mapping[str, Any]]:
    result = {}
    for index, obj in enumerate(as_list(get_path(data, source_path)), 1):
        if not isinstance(obj, Mapping):
            continue
        result[object_raw_id(obj, node_type, index)] = obj
    return result


def recursive_find(
    value: Any,
    raw_id: str,
    node_type: str,
    limit: int = 200000,
) -> Optional[Mapping[str, Any]]:
    stack = [value]
    visited = 0
    while stack and visited < limit:
        current = stack.pop()
        visited += 1
        if isinstance(current, Mapping):
            for key in NODE_ID_KEYS:
                if str(current.get(key) or "") == raw_id:
                    return current
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return None


def merge_ints(
    current: Any,
    new_values: Iterable[Any],
    minimum: int,
    maximum: int,
) -> List[int]:
    return normalise_values(
        list(numbers(current)) + list(new_values), minimum, maximum,
    )


def enrich_query_attrs(
    attrs: MutableMapping[str, Any],
    evidence: Mapping[str, Any],
    clusters: Mapping[Tuple[str, str], Mapping[str, Any]],
    maximum: int,
) -> bool:
    before = bool(collect_direct(attrs, QUERY_OFFSET_KEYS, 0, maximum))
    local_offsets = merge_ints(
        attrs.get("local_observed_field_offsets"),
        evidence.get("local_offsets") or [], 0, maximum,
    )
    widths = merge_ints(
        attrs.get("access_widths"),
        evidence.get("widths") or [], 1, maximum,
    )
    offsets = set(merge_ints(
        attrs.get("observed_field_offsets"),
        local_offsets, 0, maximum,
    ))
    cluster_keys = []
    cluster_members = 1 if local_offsets else 0
    function_entry = str(evidence.get("function_entry") or "")
    bases = sorted(set(
        str(value)
        for value in (
            list(attrs.get("memory_access_bases") or [])
            + list(evidence.get("bases") or [])
        )
        if str(value)
    ))
    for base in bases:
        key = (function_entry, str(base))
        cluster = clusters.get(key)
        if not cluster:
            continue
        offsets.update(cluster.get("offsets") or [])
        widths = merge_ints(
            widths, cluster.get("widths") or [], 1, maximum,
        )
        cluster_members = max(
            cluster_members, int(cluster.get("members") or 0),
        )
        cluster_keys.append("%s|%s" % key)

    attrs["layout_query_evidence_export_revision"] = IMPLEMENTATION_REVISION
    attrs["local_observed_field_offsets"] = local_offsets
    attrs["observed_field_offsets"] = sorted(offsets)
    attrs["access_widths"] = widths
    attrs["memory_access_bases"] = bases
    attrs["memory_access_kinds"] = sorted(set(
        list(attrs.get("memory_access_kinds") or [])
        + list(evidence.get("kinds") or [])
    ))
    attrs["memory_access_evidence"] = list(
        evidence.get("accesses") or []
    )
    attrs["layout_evidence_cluster_keys"] = sorted(set(cluster_keys))
    attrs["layout_evidence_cluster_member_count"] = cluster_members
    attrs["layout_evidence_scope"] = (
        "base_cluster" if cluster_keys and offsets
        else "instruction_local" if offsets
        else "unavailable"
    )
    return (not before) and bool(offsets)


def enrich_candidate_attrs(
    attrs: MutableMapping[str, Any],
    evidence: Mapping[str, Any],
    maximum: int,
) -> bool:
    before = bool(collect_direct(
        attrs, CANDIDATE_OFFSET_KEYS, 0, maximum,
    ))
    offsets = merge_ints(
        collect_direct(attrs, CANDIDATE_OFFSET_KEYS, 0, maximum),
        evidence.get("offsets") or [], 0, maximum,
    )
    widths = merge_ints(
        attrs.get("field_widths"),
        evidence.get("widths") or [], 1, maximum,
    )
    attrs["layout_candidate_evidence_export_revision"] = IMPLEMENTATION_REVISION
    if offsets:
        attrs["field_offsets"] = offsets
    if widths:
        attrs["field_widths"] = widths
    attrs["layout_member_offset_sources"] = list(
        evidence.get("member_sources") or []
    )
    return (not before) and bool(offsets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes-in", required=True)
    parser.add_argument("--nodes-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--max-abs-offset", type=int, default=65536)
    parser.add_argument(
        "--json-root-remap",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Optional repeatable remap for stale json_path roots.",
    )
    parser.add_argument(
        "--allow-missing-source",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Exact historic semantic JSON path allowed to be absent; "
            "repeatable. The corresponding NODES rows remain unchanged."
        ),
    )
    parser.add_argument(
        "--allow-missing-source-file",
        action="append",
        default=[],
        metavar="FILE",
        help=(
            "Newline-delimited exact missing-source allow-list; repeatable."
        ),
    )
    parser.add_argument(
        "--max-allowed-missing-sources",
        type=int,
        default=0,
        help="Hard cap for explicitly allowed missing semantic sources.",
    )
    parser.add_argument(
        "--fail-on-source-error",
        action="store_true",
        help=(
            "Compatibility flag; v6.7.4 always fails every source error "
            "that is not an exact approved missing file."
        ),
    )
    args = parser.parse_args()

    nodes_in = Path(args.nodes_in).resolve()
    nodes_out = Path(args.nodes_out).resolve()
    report_out = Path(args.report_out).resolve()
    maximum = int(args.max_abs_offset)
    if maximum <= 0:
        raise SystemExit("ERROR: --max-abs-offset must be positive")
    if args.max_allowed_missing_sources < 0:
        raise SystemExit(
            "ERROR: --max-allowed-missing-sources must be non-negative"
        )
    if not nodes_in.is_file() or nodes_in.stat().st_size <= 0:
        raise SystemExit("ERROR: nodes input missing or empty: %s" % nodes_in)
    if nodes_out == nodes_in:
        raise SystemExit("ERROR: refusing to overwrite --nodes-in")
    for output in (nodes_out, report_out):
        if output.exists():
            raise SystemExit("ERROR: output already exists: %s" % output)
        output.parent.mkdir(parents=True, exist_ok=True)

    remaps = parse_remaps(args.json_root_remap)
    try:
        allowed_missing_sources = load_allowed_missing_sources(
            args.allow_missing_source,
            args.allow_missing_source_file,
        )
    except (OSError, ValueError) as error:
        raise SystemExit("ERROR: %s" % error)
    if len(allowed_missing_sources) > args.max_allowed_missing_sources:
        raise SystemExit(
            "ERROR: missing-source allow-list has %d entries; cap is %d"
            % (
                len(allowed_missing_sources),
                args.max_allowed_missing_sources,
            )
        )
    requests_by_path: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    total_rows = 0
    node_type_counts = Counter()
    query_before = 0
    candidate_before = 0

    for row in read_jsonl(nodes_in):
        total_rows += 1
        node_type = str(row.get("node_type") or "")
        node_type_counts[node_type] += 1
        attrs = row.get("attrs") if isinstance(row.get("attrs"), Mapping) else {}
        if node_type == "InstrSem":
            query_before += int(bool(collect_direct(
                attrs, QUERY_OFFSET_KEYS, 0, maximum,
            )))
        elif node_type == "LayoutSem":
            candidate_before += int(bool(collect_direct(
                attrs, CANDIDATE_OFFSET_KEYS, 0, maximum,
            )))
        else:
            continue
        json_path = str(row.get("json_path") or "")
        if not json_path:
            continue
        requests_by_path[json_path].append({
            "node_id": str(row.get("node_id") or ""),
            "node_type": node_type,
            "source_path": str(row.get("source_path") or ""),
            "raw_id": str(row.get("raw_id") or ""),
        })

    requested_path_spellings = set()
    for original_path in requests_by_path:
        requested_path_spellings.add(normalise_path_text(original_path))
        requested_path_spellings.add(normalise_path_text(
            remap_path(original_path, remaps)
        ))
    unknown_allowlist_entries = sorted(
        allowed_missing_sources - requested_path_spellings
    )
    if unknown_allowlist_entries:
        raise SystemExit(
            "ERROR: allowed-missing-source entries are not requested by NODES: %s"
            % unknown_allowlist_entries[:5]
        )

    enrichment: Dict[str, Dict[str, Any]] = {}
    source_errors = []
    matched_allowlist_entries = set()
    source_files_loaded = 0
    raw_objects_resolved = 0

    for original_path, requests in requests_by_path.items():
        source_path = remap_path(original_path, remaps)
        try:
            data = json.loads(source_path.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping):
                raise ValueError("semantic JSON root is not an object")
            source_files_loaded += 1
        except Exception as error:
            original_key = normalise_path_text(original_path)
            resolved_key = normalise_path_text(source_path)
            source_is_absent = not source_path.exists()
            matching_entries = {
                value
                for value in (original_key, resolved_key)
                if value in allowed_missing_sources
            }
            approved_missing = bool(
                source_is_absent and matching_entries
            )
            if approved_missing:
                matched_allowlist_entries.update(matching_entries)
            source_errors.append({
                "json_path": original_path,
                "resolved_path": str(source_path),
                "error": str(error),
                "error_kind": (
                    "missing_file" if source_is_absent
                    else type(error).__name__
                ),
                "approved_missing": approved_missing,
            })
            if not approved_missing:
                break
            continue

        cache: Dict[Tuple[str, str], Dict[str, Mapping[str, Any]]] = {}
        instr_objects: Dict[str, Mapping[str, Any]] = {}
        for request in requests:
            key = (request["source_path"], request["node_type"])
            if key not in cache:
                cache[key] = build_source_objects(
                    data, request["source_path"], request["node_type"],
                )
            if request["node_type"] == "InstrSem":
                instr_objects.update(cache[key])

        instr_evidence = {
            raw_id: extract_instr_evidence(obj, maximum)
            for raw_id, obj in instr_objects.items()
        }
        clusters: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {"offsets": set(), "widths": set(), "members": 0}
        )
        for evidence in instr_evidence.values():
            for base in evidence.get("bases") or []:
                key = (str(evidence.get("function_entry") or ""), str(base))
                clusters[key]["offsets"].update(evidence.get("local_offsets") or [])
                clusters[key]["widths"].update(evidence.get("widths") or [])
                clusters[key]["members"] += 1

        for request in requests:
            key = (request["source_path"], request["node_type"])
            obj = cache.get(key, {}).get(request["raw_id"])
            if obj is None and request["raw_id"]:
                obj = recursive_find(
                    data, request["raw_id"], request["node_type"],
                )
            if obj is None:
                continue
            raw_objects_resolved += 1
            if request["node_type"] == "InstrSem":
                evidence = extract_instr_evidence(obj, maximum)
                enrichment[request["node_id"]] = {
                    "kind": "query",
                    "evidence": evidence,
                    "clusters": clusters,
                }
            else:
                enrichment[request["node_id"]] = {
                    "kind": "candidate",
                    "evidence": extract_layout_candidate_evidence(
                        obj, maximum,
                    ),
                }

    approved_source_errors = [
        item for item in source_errors
        if item.get("approved_missing") is True
    ]
    unapproved_source_errors = [
        item for item in source_errors
        if item.get("approved_missing") is not True
    ]
    approved_missing_requests = []
    approved_missing_node_types = Counter()
    for item in approved_source_errors:
        requests = requests_by_path.get(item["json_path"], [])
        approved_missing_requests.extend(requests)
        approved_missing_node_types.update(
            request.get("node_type") or "unknown"
            for request in requests
        )
    unused_allowlist_entries = sorted(
        allowed_missing_sources - matched_allowlist_entries
    )
    if (
        len(approved_source_errors)
        > args.max_allowed_missing_sources
    ):
        raise SystemExit(
            "ERROR: approved missing sources exceed configured cap"
        )
    if unapproved_source_errors:
        first = unapproved_source_errors[0]
        raise SystemExit(
            "ERROR: semantic source read failed: %s: %s"
            % (first["resolved_path"], first["error"])
        )
    if unused_allowlist_entries:
        raise SystemExit(
            "ERROR: missing-source allow-list entries were not observed as "
            "missing files: %s" % unused_allowlist_entries[:5]
        )

    query_after = 0
    candidate_after = 0
    query_newly_enriched = 0
    candidate_newly_enriched = 0
    rows_with_enrichment = 0

    with nodes_out.open("x", encoding="utf-8") as output:
        for row in read_jsonl(nodes_in):
            node_type = str(row.get("node_type") or "")
            attrs = row.get("attrs")
            if not isinstance(attrs, dict):
                attrs = {}
                row["attrs"] = attrs
            payload = enrichment.get(str(row.get("node_id") or ""))
            if payload and payload["kind"] == "query":
                rows_with_enrichment += 1
                query_newly_enriched += int(enrich_query_attrs(
                    attrs,
                    payload["evidence"],
                    payload["clusters"],
                    maximum,
                ))
            elif payload and payload["kind"] == "candidate":
                rows_with_enrichment += 1
                candidate_newly_enriched += int(enrich_candidate_attrs(
                    attrs, payload["evidence"], maximum,
                ))

            if node_type == "InstrSem":
                query_after += int(bool(collect_direct(
                    attrs, QUERY_OFFSET_KEYS, 0, maximum,
                )))
            elif node_type == "LayoutSem":
                candidate_after += int(bool(collect_direct(
                    attrs, CANDIDATE_OFFSET_KEYS, 0, maximum,
                )))
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    instr_count = int(node_type_counts.get("InstrSem", 0))
    layout_count = int(node_type_counts.get("LayoutSem", 0))
    report = {
        "schema": "task2_layout_evidence_enrichment_report_v6_7_4",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "inputs": {
            "nodes_in": str(nodes_in),
            "nodes_in_sha256": sha256_file(nodes_in),
            "json_root_remaps": [
                {"old": old, "new": new} for old, new in remaps
            ],
        },
        "outputs": {
            "nodes_out": str(nodes_out),
            "nodes_out_sha256": sha256_file(nodes_out),
        },
        "total_rows": total_rows,
        "node_type_counts": dict(node_type_counts),
        "source_files_requested": len(requests_by_path),
        "source_files_loaded": source_files_loaded,
        "source_error_count": len(source_errors),
        "approved_missing_source_count": len(approved_source_errors),
        "unapproved_source_error_count": len(unapproved_source_errors),
        "approved_missing_sources": [
            item["resolved_path"] for item in approved_source_errors
        ],
        "approved_missing_requested_rows": len(
            approved_missing_requests
        ),
        "approved_missing_requested_node_types": dict(
            approved_missing_node_types
        ),
        "allowed_missing_source_cap": (
            args.max_allowed_missing_sources
        ),
        "semantic_source_load_coverage": (
            source_files_loaded / len(requests_by_path)
            if requests_by_path else None
        ),
        "controlled_missing_source_mode": bool(
            approved_source_errors
        ),
        "source_errors": source_errors[:20],
        "raw_objects_requested": sum(len(v) for v in requests_by_path.values()),
        "raw_objects_resolved": raw_objects_resolved,
        "rows_with_enrichment": rows_with_enrichment,
        "instrsem_with_offsets_before": query_before,
        "instrsem_with_offsets_after": query_after,
        "instrsem_newly_enriched": query_newly_enriched,
        "instrsem_offset_coverage_after": (
            query_after / instr_count if instr_count else None
        ),
        "layoutsem_with_offsets_before": candidate_before,
        "layoutsem_with_offsets_after": candidate_after,
        "layoutsem_newly_enriched": candidate_newly_enriched,
        "layoutsem_offset_coverage_after": (
            candidate_after / layout_count if layout_count else None
        ),
        "node_ids_preserved": True,
        "row_order_preserved": True,
        "gold_used": False,
        "test_used": False,
    }
    with report_out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("LAYOUT EVIDENCE ENRICHMENT: COMPLETED")


if __name__ == "__main__":
    main()
