from __future__ import annotations

import hashlib
import math
import statistics


def _permutation(candidate_ids, seed, record_id, index):
    return sorted(candidate_ids, key=lambda candidate: hashlib.sha256(
        (str(seed) + "\0" + str(record_id) + "\0" + str(index) + "\0" + candidate).encode("utf-8")
    ).hexdigest())


def _ranks(score_map):
    order = sorted(score_map, key=lambda candidate: (-score_map[candidate], candidate))
    return {candidate: index + 1 for index, candidate in enumerate(order)}


def _spearman(left, right):
    keys = sorted(left)
    n = len(keys)
    if n < 2:
        return 1.0
    squared = sum((left[key] - right[key]) ** 2 for key in keys)
    return 1.0 - 6.0 * squared / (n * (n * n - 1))


def evaluate_permutation_robustness(records, score_fn, seed, permutations=5):
    """Evaluate candidate-identity stability under deterministic order changes.

    ``score_fn(record, ordered_candidate_ids)`` returns scores in the same
    position order. Scores are immediately mapped back to semantic identity.
    """
    if permutations < 5:
        raise ValueError("at least five deterministic permutations are required")
    query_reports = []
    for record in records:
        original_ids = list(record["candidate_ids"])
        original_scores = score_fn(record, original_ids)
        if len(original_scores) != len(original_ids):
            raise RuntimeError("score count mismatch")
        reference = dict(zip(original_ids, map(float, original_scores)))
        reference_ranks = _ranks(reference)
        reference_top1 = min(reference, key=lambda candidate: (-reference[candidate], candidate))
        outcomes = []
        for index in range(permutations):
            ordered = _permutation(original_ids, seed, record["record_id"], index)
            values = score_fn(record, ordered)
            mapped = dict(zip(ordered, map(float, values)))
            ranks = _ranks(mapped)
            top1 = min(mapped, key=lambda candidate: (-mapped[candidate], candidate))
            outcomes.append({
                "permutation": index,
                "order": ordered,
                "top1_candidate_id": top1,
                "top1_identity_agreement": float(top1 == reference_top1),
                "rank_correlation": _spearman(reference_ranks, ranks),
                "R1": float(top1 == record["gold_candidate_id"]),
            })
        query_reports.append({
            "record_id": record["record_id"],
            "module": record["module"],
            "permutations": outcomes,
            "top1_identity_agreement": statistics.mean(row["top1_identity_agreement"] for row in outcomes),
            "mean_rank_correlation": statistics.mean(row["rank_correlation"] for row in outcomes),
            "R1_variance": statistics.pvariance(row["R1"] for row in outcomes),
        })
    return {
        "queries": len(query_reports),
        "permutations_per_query": permutations,
        "top1_identity_agreement": statistics.mean(row["top1_identity_agreement"] for row in query_reports) if query_reports else None,
        "mean_rank_correlation": statistics.mean(row["mean_rank_correlation"] for row in query_reports) if query_reports else None,
        "mean_R1_variance": statistics.mean(row["R1_variance"] for row in query_reports) if query_reports else None,
        "details": query_reports,
    }
