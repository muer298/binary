#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate sample-aware, semantic-type-aware hypotheses for Task 2.2.2.

Compared with v2, candidates from APISem/TypeSem/LayoutSem/InterfaceSem no
longer compete in one undifferentiated pool. Direct USG evidence is always
preserved, while embedding neighbours are retrieved within the same sample and
semantic type.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

from module_common_v5_3 import (
    HARD_API_SOURCES,
    SEMANTIC_TYPES,
    api_source_evidence_from_objects,
    clamp01,
    cosine_to_probability,
    edge_family,
    edge_kind,
    edge_weight,
    endpoints,
    ensure_output_path,
    evidence_type_from_objects,
    load_node_metadata,
    noisy_or,
    normalize_api_source,
    normalize_evidence_type,
    normalize_semantic_type,
    semantic_type_for_evidence,
)


def extract_index_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "items", "nodes", "index", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise RuntimeError("Index must be a JSON list or contain records/items/nodes")


def extract_embedding_tensor(payload: Any) -> torch.Tensor:
    if torch is None:
        raise RuntimeError("PyTorch is required for candidate generation")
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        for key in (
            "embeddings",
            "node_embeddings",
            "features",
            "vectors",
            "tensor",
        ):
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise RuntimeError(
        "Embedding file must contain a Tensor or a dictionary Tensor field"
    )


def load_sample_filter(path: str, split: str) -> set[str]:
    """Load sample IDs from a frozen, dataset-independent split manifest."""
    if bool(path) != bool(split):
        raise ValueError(
            "--sample-split-manifest and --sample-split must be supplied together"
        )
    if not path:
        return set()
    wanted = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            if str(row.get("split") or "").lower() != split.lower():
                continue
            sample_id = str(row.get("sample_id") or "")
            if sample_id:
                wanted.add(sample_id)
    if not wanted:
        raise RuntimeError(
            "No samples for split %r were found in %s" % (split, path)
        )
    return wanted


def load_query_filter(path: str) -> set[str]:
    """Load independently labelled query IDs for an aligned evaluation run."""
    if not path:
        return set()
    wanted = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            query_id = str(row.get("query_id") or "")
            if query_id:
                wanted.add(query_id)
    if not wanted:
        raise RuntimeError("No query_id was found in %s" % path)
    return wanted


def resolve_device(requested: str) -> str:
    if torch is None:
        raise RuntimeError("PyTorch is required for candidate generation")
    requested = str(requested).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def topk_chunked(
    query_matrix: torch.Tensor,
    target_indices: Sequence[int],
    embeddings: torch.Tensor,
    top_k: int,
    target_batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cosine values and global embedding indices without target OOM."""
    if torch is None:
        raise RuntimeError("PyTorch is required for candidate generation")
    if not target_indices or top_k <= 0:
        shape = (query_matrix.shape[0], 0)
        return torch.empty(shape), torch.empty(shape, dtype=torch.long)

    keep = min(int(top_k), len(target_indices))
    best_values = torch.empty(
        (query_matrix.shape[0], 0),
        dtype=query_matrix.dtype,
        device=device,
    )
    best_indices = torch.empty(
        (query_matrix.shape[0], 0),
        dtype=torch.long,
        device=device,
    )

    for start in range(0, len(target_indices), target_batch_size):
        chunk_indices = list(
            target_indices[start : start + target_batch_size]
        )
        target_matrix = embeddings[chunk_indices].to(device)
        similarities = query_matrix @ target_matrix.T
        local_keep = min(keep, similarities.shape[1])
        values, locations = torch.topk(
            similarities,
            k=local_keep,
            dim=1,
        )
        index_tensor = torch.tensor(
            chunk_indices,
            dtype=torch.long,
            device=device,
        )
        global_indices = index_tensor[locations]

        combined_values = torch.cat((best_values, values), dim=1)
        combined_indices = torch.cat(
            (best_indices, global_indices),
            dim=1,
        )
        combined_keep = min(keep, combined_values.shape[1])
        best_values, selected = torch.topk(
            combined_values,
            k=combined_keep,
            dim=1,
        )
        best_indices = torch.gather(combined_indices, 1, selected)

    return best_values.cpu(), best_indices.cpu()


def choose_evidence_type(
    query_row: Mapping[str, Any],
    edge_evidence: Sequence[Tuple[str, float]],
) -> Tuple[str, str]:
    explicit = evidence_type_from_objects(query_row)
    if explicit != "unknown":
        return explicit, "query_index"
    scores: Dict[str, float] = defaultdict(float)
    for evidence_type, weight in edge_evidence:
        normalized = normalize_evidence_type(evidence_type)
        if normalized != "unknown":
            scores[normalized] += max(0.0, float(weight))
    if not scores:
        return "unknown", "missing"
    selected = max(
        scores,
        key=lambda name: (
            scores[name],
            -list(
                ("api_call", "deref_chain", "fixed_offset", "def_use")
            ).index(name),
        ),
    )
    return selected, "direct_usg_edge"


def round_robin_non_direct(
    candidates: Sequence[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_type[str(candidate["target_semantic_type"])].append(candidate)
    for values in by_type.values():
        values.sort(
            key=lambda item: (
                -float(item.get("confidence", 0.0)),
                str(item.get("target_id") or ""),
            )
        )

    result: List[Dict[str, Any]] = []
    cursor = 0
    ordered_types = [
        semantic_type
        for semantic_type in SEMANTIC_TYPES
        if semantic_type in by_type
    ]
    while len(result) < max(0, limit):
        added = False
        for semantic_type in ordered_types:
            values = by_type[semantic_type]
            if cursor < len(values) and len(result) < limit:
                result.append(values[cursor])
                added = True
        if not added:
            break
        cursor += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument(
        "--nodes",
        help=(
            "Complete USG node JSONL used to recover API recognition "
            "provenance omitted from the compact embedding index"
        ),
    )
    parser.add_argument(
        "--api-provenance",
        help=(
            "Optional compact APISem provenance sidecar. When supplied, "
            "it is used instead of rescanning the complete nodes JSONL."
        ),
    )
    parser.add_argument("--edges", required=True)
    parser.add_argument("--query-types", default="InstrSem")
    parser.add_argument(
        "--target-types",
        default="TypeSem,LayoutSem,InterfaceSem,APISem",
    )
    parser.add_argument("--top-k-per-type", type=int, default=3)
    parser.add_argument("--max-total-candidates", type=int, default=12)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="0 means all queries; use 5000 only for a smoke run",
    )
    parser.add_argument(
        "--sample-split-manifest",
        default="",
        help="Frozen split JSONL produced by 15_make_frozen_split_v5_4.py",
    )
    parser.add_argument(
        "--sample-split",
        choices=("train", "validation", "test"),
        default="",
        help="Only generate queries/targets from this frozen split",
    )
    parser.add_argument(
        "--query-whitelist-jsonl",
        default="",
        help=(
            "Optional gold/evaluation JSONL whose query_id values define the "
            "query subset; labels are not read by candidate scoring"
        ),
    )
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--target-batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if torch is None or F is None:
        raise RuntimeError(
            "PyTorch is required for 08_generate_hypotheses_v5_3.py"
        )
    if args.top_k_per_type <= 0:
        raise ValueError("--top-k-per-type must be positive")
    if args.max_total_candidates <= 0:
        raise ValueError("--max-total-candidates must be positive")
    if args.query_batch_size <= 0 or args.target_batch_size <= 0:
        raise ValueError("query/target batch sizes must be positive")
    device = resolve_device(args.device)
    embedding_payload = torch.load(args.embeddings, map_location="cpu")
    embeddings = F.normalize(
        extract_embedding_tensor(embedding_payload).float(),
        dim=-1,
    )
    index = extract_index_records(
        json.loads(Path(args.index).read_text(encoding="utf-8"))
    )
    if len(index) != len(embeddings):
        raise RuntimeError(
            "Index/embedding length mismatch: %d != %d"
            % (len(index), len(embeddings))
        )

    query_types = {
        normalize_semantic_type(value)
        for value in args.query_types.split(",")
        if value
    }
    target_types = [
        normalize_semantic_type(value)
        for value in args.target_types.split(",")
        if value
    ]
    target_set = set(target_types)
    allowed_samples = load_sample_filter(
        args.sample_split_manifest, args.sample_split
    )
    allowed_queries = load_query_filter(args.query_whitelist_jsonl)

    id_to_index: Dict[str, int] = {}
    sample_by_id: Dict[str, str] = {}
    type_by_id: Dict[str, str] = {}
    metadata_by_id: Dict[str, Dict[str, Any]] = {}
    sample_targets: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sample_queries: Dict[str, List[int]] = defaultdict(list)
    query_ids = set()
    query_count = 0
    missing_sample_ids: List[str] = []

    for index_number, row in enumerate(index):
        identity = str(row.get("node_id") or row.get("id") or "")
        semantic_type = normalize_semantic_type(
            row.get("node_type") or row.get("semantic_type") or ""
        )
        sample_id = str(
            row.get("sample_id")
            or row.get("binary_id")
            or row.get("program_id")
            or ""
        )
        if allowed_samples and sample_id not in allowed_samples:
            continue
        if not identity:
            continue
        id_to_index[identity] = index_number
        sample_by_id[identity] = sample_id
        type_by_id[identity] = semantic_type
        metadata_by_id[identity] = row
        if semantic_type in target_set.union(query_types) and not sample_id:
            missing_sample_ids.append(identity)
        if semantic_type in target_set:
            sample_targets[sample_id][semantic_type].append(index_number)
        if (
            semantic_type in query_types
            and (not allowed_queries or identity in allowed_queries)
            and (args.max_queries <= 0 or query_count < args.max_queries)
        ):
            sample_queries[sample_id].append(index_number)
            query_ids.add(identity)
            query_count += 1

    if missing_sample_ids:
        raise RuntimeError(
            "sample_id is required for leakage-safe retrieval; missing on "
            "%d query/target nodes (first: %s)"
            % (
                len(missing_sample_ids),
                ", ".join(missing_sample_ids[:10]),
            )
        )

    api_node_ids = {
        identity
        for identity, semantic_type in type_by_id.items()
        if semantic_type == "APISem"
    }
    provenance_input = args.api_provenance or args.nodes
    complete_node_metadata = load_node_metadata(
        provenance_input,
        wanted_ids=api_node_ids,
    )
    for identity, full_record in complete_node_metadata.items():
        # Retain the complete record while keeping canonical index identity,
        # type and sample fields authoritative.
        merged = dict(full_record)
        merged.update(metadata_by_id.get(identity, {}))
        metadata_by_id[identity] = merged

    node_api_source_counts: Counter = Counter()
    for identity in api_node_ids:
        evidence = api_source_evidence_from_objects(
            metadata_by_id.get(identity, {}),
            object_labels=("usg_node",),
        )
        source = str(evidence.get("source") or "unknown")
        if source != "unknown":
            node_api_source_counts[source] += 1

    direct: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    query_edge_evidence: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    edges_seen = 0
    for edge in _iter_edges(args.edges):
        edges_seen += 1
        source, target = endpoints(edge)
        if source not in query_ids:
            continue
        evidence_type = evidence_type_from_objects(edge)
        weight = edge_weight(edge)
        if evidence_type != "unknown":
            query_edge_evidence[source].append((evidence_type, weight))
        family = edge_family(edge)
        if (
            target not in id_to_index
            or type_by_id.get(target) not in target_set
            or sample_by_id.get(source) != sample_by_id.get(target)
            or family == "constraint"
        ):
            continue
        record = direct[source].setdefault(
            target,
            {
                "weights": [],
                "kinds": [],
                "families": [],
                "api_sources": [],
                "api_source_evidence": [],
                "evidence_types": [],
            },
        )
        record["weights"].append(weight)
        record["kinds"].append(edge_kind(edge))
        record["families"].append(family)
        source_evidence = api_source_evidence_from_objects(
            edge,
            metadata_by_id.get(target, {}),
            object_labels=("direct_edge", "usg_node"),
        )
        api_source = str(source_evidence.get("source") or "unknown")
        if api_source != "unknown":
            record["api_sources"].append(api_source)
            record["api_source_evidence"].append(source_evidence)
        if evidence_type != "unknown":
            record["evidence_types"].append(evidence_type)
        if edges_seen % 1000000 == 0:
            print(
                json.dumps(
                    {
                        "phase": "scan_edges",
                        "edges_seen": edges_seen,
                        "queries_with_direct": len(direct),
                    }
                ),
                flush=True,
            )

    output = ensure_output_path(args.out, overwrite=args.overwrite)
    written = 0
    route_counts: Counter = Counter()
    candidate_type_counts: Counter = Counter()
    hard_api_candidates = 0
    queries_with_direct = 0

    with output.open("w", encoding="utf-8") as writer:
        for sample_id, query_indices in sample_queries.items():
            targets = sample_targets.get(sample_id, {})
            for start in range(0, len(query_indices), args.query_batch_size):
                batch_query_indices = query_indices[
                    start : start + args.query_batch_size
                ]
                query_matrix = embeddings[batch_query_indices].to(device)
                per_type: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
                for semantic_type in target_types:
                    target_indices = targets.get(semantic_type, [])
                    if not target_indices:
                        continue
                    per_type[semantic_type] = topk_chunked(
                        query_matrix,
                        target_indices,
                        embeddings,
                        args.top_k_per_type,
                        args.target_batch_size,
                        device,
                    )

                for row_offset, query_index in enumerate(batch_query_indices):
                    query_row = index[query_index]
                    query_id = str(
                        query_row.get("node_id")
                        or query_row.get("id")
                        or ""
                    )
                    candidate_map: Dict[str, Dict[str, Any]] = {}

                    for target_id, direct_info in direct.get(
                        query_id, {}
                    ).items():
                        target_index = id_to_index[target_id]
                        cosine = float(
                            embeddings[query_index] @ embeddings[target_index]
                        )
                        api_sources = [
                            normalize_api_source(value)
                            for value in direct_info["api_sources"]
                            if normalize_api_source(value) != "unknown"
                        ]
                        source_evidence = (
                            direct_info["api_source_evidence"][0]
                            if direct_info["api_source_evidence"]
                            else api_source_evidence_from_objects(
                                metadata_by_id.get(target_id, {}),
                                object_labels=("usg_node",),
                            )
                        )
                        api_source = str(
                            source_evidence.get("source") or "unknown"
                        )
                        if api_source == "unknown" and api_sources:
                            api_source = api_sources[0]
                        semantic_type = type_by_id[target_id]
                        is_hard = (
                            semantic_type == "APISem"
                            and api_source in HARD_API_SOURCES
                        )
                        hard_api_candidates += int(is_hard)
                        candidate_map[target_id] = {
                            "target_id": target_id,
                            "target_semantic_type": semantic_type,
                            "confidence": cosine_to_probability(cosine),
                            "representation_score": cosine_to_probability(
                                cosine
                            ),
                            "embedding_cosine": cosine,
                            "candidate_source": "usg_direct_evidence",
                            "same_sample": True,
                            "direct_edge_count": len(
                                direct_info["weights"]
                            ),
                            "direct_edge_weight": noisy_or(
                                direct_info["weights"]
                            ),
                            "direct_edge_weights": [
                                clamp01(value)
                                for value in direct_info["weights"][:16]
                            ],
                            "direct_edge_kinds": direct_info["kinds"][:16],
                            "direct_edge_families": direct_info[
                                "families"
                            ][:16],
                            "direct_evidence_types": direct_info[
                                "evidence_types"
                            ][:16],
                            "api_source": api_source,
                            "api_source_origin": source_evidence.get(
                                "origin", "none"
                            ),
                            "api_source_path": source_evidence.get(
                                "path", ""
                            ),
                            "api_source_raw_value": source_evidence.get(
                                "raw_value", ""
                            ),
                            "api_source_signal_kind": source_evidence.get(
                                "signal_kind", "none"
                            ),
                            "deterministic_api_source": is_hard,
                            "initial_status": (
                                "confirmed_hard"
                                if is_hard
                                else "hypothesis"
                            ),
                        }

                    if direct.get(query_id):
                        queries_with_direct += 1

                    for semantic_type, (values, target_indices) in per_type.items():
                        for cosine_tensor, target_index_tensor in zip(
                            values[row_offset],
                            target_indices[row_offset],
                        ):
                            target_index = int(target_index_tensor)
                            target_id = str(
                                index[target_index].get("node_id")
                                or index[target_index].get("id")
                                or ""
                            )
                            if not target_id or target_id in candidate_map:
                                continue
                            cosine = float(cosine_tensor)
                            source_evidence = (
                                api_source_evidence_from_objects(
                                    metadata_by_id.get(target_id, {}),
                                    object_labels=("usg_node",),
                                )
                            )
                            api_source = str(
                                source_evidence.get("source") or "unknown"
                            )
                            # A known APISem node reached only through nearest
                            # neighbour retrieval is not a deterministic match
                            # for this query. Hard confirmation additionally
                            # requires the direct query->API evidence checked
                            # by 09_validate_evidence_v5_3.py.
                            is_hard = False
                            candidate_map[target_id] = {
                                "target_id": target_id,
                                "target_semantic_type": semantic_type,
                                "confidence": cosine_to_probability(cosine),
                                "representation_score": cosine_to_probability(
                                    cosine
                                ),
                                "embedding_cosine": cosine,
                                "candidate_source": "same_sample_embedding",
                                "same_sample": True,
                                "direct_edge_count": 0,
                                "direct_edge_weight": 0.0,
                                "direct_edge_weights": [],
                                "direct_edge_kinds": [],
                                "direct_edge_families": [],
                                "direct_evidence_types": [],
                                "api_source": api_source,
                                "api_source_origin": source_evidence.get(
                                    "origin", "none"
                                ),
                                "api_source_path": source_evidence.get(
                                    "path", ""
                                ),
                                "api_source_raw_value": source_evidence.get(
                                    "raw_value", ""
                                ),
                                "api_source_signal_kind": source_evidence.get(
                                    "signal_kind", "none"
                                ),
                                "deterministic_api_source": is_hard,
                                "initial_status": (
                                    "confirmed_hard"
                                    if is_hard
                                    else "hypothesis"
                                ),
                            }

                    direct_candidates = [
                        candidate
                        for candidate in candidate_map.values()
                        if candidate["candidate_source"]
                        == "usg_direct_evidence"
                    ]
                    embedding_candidates = [
                        candidate
                        for candidate in candidate_map.values()
                        if candidate["candidate_source"]
                        != "usg_direct_evidence"
                    ]
                    direct_candidates.sort(
                        key=lambda item: (
                            -float(item.get("direct_edge_weight", 0.0)),
                            -float(item.get("confidence", 0.0)),
                            str(item.get("target_id") or ""),
                        )
                    )
                    remaining = max(
                        0,
                        args.max_total_candidates
                        - len(direct_candidates),
                    )
                    candidates = direct_candidates + round_robin_non_direct(
                        embedding_candidates,
                        remaining,
                    )

                    evidence_type, route_source = choose_evidence_type(
                        query_row,
                        query_edge_evidence.get(query_id, []),
                    )
                    routed_semantic_type = semantic_type_for_evidence(
                        evidence_type
                    )
                    route_counts[evidence_type] += 1
                    for candidate in candidates:
                        candidate_type_counts[
                            candidate["target_semantic_type"]
                        ] += 1

                    writer.write(
                        json.dumps(
                            {
                                "schema": "task2_222_hypotheses_v5_3",
                                "query_id": query_id,
                                "query_semantic_type": normalize_semantic_type(
                                    query_row.get("node_type")
                                    or query_row.get("semantic_type")
                                    or ""
                                ),
                                "sample_id": sample_id,
                                "candidate_scope": (
                                    "same_sample+semantic_type+direct_evidence"
                                ),
                                "evidence_type": evidence_type,
                                "route_source": route_source,
                                "routed_semantic_type": routed_semantic_type,
                                "candidates": candidates,
                                "status": "hypothesis",
                                "confirmation_level": 1,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
                    if written % 1000 == 0:
                        print(
                            json.dumps(
                                {
                                    "phase": "generate",
                                    "records": written,
                                    "device": device,
                                }
                            ),
                            flush=True,
                        )

    print(
        json.dumps(
            {
                "done": True,
                "schema": "task2_222_hypotheses_summary_v5_3",
                "output": str(output),
                "records": written,
                "edges_seen": edges_seen,
                "queries_with_direct_candidates": queries_with_direct,
                "complete_api_node_records_loaded": len(
                    complete_node_metadata
                ),
                "api_provenance_input": str(provenance_input or ""),
                "api_provenance_sidecar_used": bool(
                    args.api_provenance
                ),
                "api_sources_recovered_from_nodes": dict(
                    node_api_source_counts
                ),
                "candidate_counts_by_semantic_type": dict(
                    candidate_type_counts
                ),
                "evidence_route_counts": dict(route_counts),
                "hard_api_candidates": hard_api_candidates,
                "sample_split_manifest": args.sample_split_manifest,
                "sample_split": args.sample_split,
                "allowed_sample_count": len(allowed_samples),
                "query_whitelist_jsonl": args.query_whitelist_jsonl,
                "allowed_query_count": len(allowed_queries),
                "same_sample_ratio_expected": 1.0,
                "device": device,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _iter_edges(path: str) -> Iterable[Dict[str, Any]]:
    from module_common_v5_3 import read_jsonl

    return read_jsonl(path)


if __name__ == "__main__":
    main()
