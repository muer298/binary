#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train small per-module calibration heads with strict split isolation.

These heads operate on auditable specialised features and the frozen 2.2.1
representation score.  They are rank calibrators, not duplicate encoders.
Train labels update parameters; Validation selects hyperparameters; Test is
neither accepted nor read by this program.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from module_common_223_v5_3 import (
    SEMANTIC_TYPES, clamp01, normalize_semantic_type, read_jsonl,
)
from module_specialized_v6_6 import (
    FEATURE_NAMES, RULE_WEIGHTS, GraphContext, collect_wanted_ids, sigmoid,
    specialised_features,
)


Group = Tuple[List[List[float]], List[int], List[str], List[float]]


def parse_blend_grid(text: str) -> List[float]:
    values = sorted({float(item) for item in str(text).split(",") if item})
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("--blend-grid must be a non-empty list in [0,1]")
    return values


def load_gold(path: str, required_split: str) -> Dict[Tuple[str, str], str]:
    result: Dict[Tuple[str, str], str] = {}
    for row in read_jsonl(path):
        split = str(row.get("split") or required_split).lower()
        if split != required_split:
            raise ValueError(
                "%s contains split=%r; expected only %s (Test is never allowed)"
                % (path, split, required_split)
            )
        semantic_type = normalize_semantic_type(
            row.get("target_semantic_type") or row.get("semantic_type")
        )
        query_id = str(row.get("query_id") or "")
        target_id = str(row.get("target_id") or "")
        if query_id and target_id and semantic_type in SEMANTIC_TYPES:
            key = (query_id, semantic_type)
            previous = result.get(key)
            if previous is not None and previous != target_id:
                raise ValueError("Ambiguous gold for %r" % (key,))
            result[key] = target_id
    return result


def make_groups(
    reasoning: str,
    gold: Mapping[Tuple[str, str], str],
    context: GraphContext,
    max_groups_per_module: int,
) -> Tuple[Dict[str, List[Group]], Dict[str, Any]]:
    groups: Dict[str, List[Group]] = defaultdict(list)
    audit = Counter()
    for row in read_jsonl(reasoning):
        query_id = str(row.get("query_id") or "")
        by_module = defaultdict(list)
        for candidate in row.get("candidates", []) or []:
            semantic_type = normalize_semantic_type(candidate.get("target_semantic_type"))
            if semantic_type in SEMANTIC_TYPES:
                by_module[semantic_type].append(candidate)
        for semantic_type, candidates in by_module.items():
            key = (query_id, semantic_type)
            target = gold.get(key)
            if not target:
                continue
            audit["gold_groups_seen"] += 1
            if len(groups[semantic_type]) >= max_groups_per_module:
                audit["groups_capped"] += 1
                continue
            ids = [str(candidate.get("target_id") or "") for candidate in candidates]
            if target not in ids:
                audit["gold_missing_from_candidates"] += 1
                continue
            names = FEATURE_NAMES[semantic_type]
            vectors = []
            labels = []
            rule_scores = []
            for candidate in candidates:
                target_id = str(candidate.get("target_id") or "")
                feature_map = specialised_features(
                    semantic_type,
                    row,
                    candidate,
                    context.nodes.get(target_id, {}),
                    context.incident.get(target_id, []),
                )
                vector = [feature_map[name] for name in names]
                vectors.append(vector)
                rule_weights = RULE_WEIGHTS[semantic_type]
                rule_scores.append(clamp01(
                    sum(weight * value for weight, value in zip(rule_weights, vector))
                    / max(1e-12, sum(rule_weights))
                ))
                labels.append(int(target_id == target))
            groups[semantic_type].append((vectors, labels, ids, rule_scores))
            audit["usable_groups"] += 1
    return dict(groups), dict(audit)


def evaluate(
    groups: Sequence[Group], weights: Sequence[float], bias: float,
    blend_weight: float,
) -> Dict[str, float]:
    if not groups:
        return {"n": 0, "recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0}
    r1 = r5 = 0
    rr = 0.0
    for vectors, labels, ids, rule_scores in groups:
        order = sorted(
            range(len(vectors)),
            key=lambda index: (
                -(
                    (1.0 - blend_weight) * rule_scores[index]
                    + blend_weight * sigmoid(
                        bias + sum(
                            w * x for w, x in zip(weights, vectors[index])
                        )
                    )
                ),
                ids[index],
            ),
        )
        rank = next(position for position, index in enumerate(order, 1) if labels[index])
        r1 += int(rank <= 1)
        r5 += int(rank <= 5)
        rr += 1.0 / rank
    return {
        "n": len(groups),
        "recall_at_1": r1 / len(groups),
        "recall_at_5": r5 / len(groups),
        "mrr": rr / len(groups),
    }


def train_one(
    train_groups: Sequence[Group],
    validation_groups: Sequence[Group],
    dimension: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    blend_grid: Sequence[float],
) -> Dict[str, Any]:
    candidates = []
    for l2 in (0.0, 1e-4, 1e-3, 1e-2):
        rng = random.Random(seed)
        weights = [0.0] * dimension
        bias = 0.0
        flat = []
        for vectors, labels, _, _ in train_groups:
            positives = [item for item in zip(vectors, labels) if item[1] == 1]
            negatives = [item for item in zip(vectors, labels) if item[1] == 0]
            rng.shuffle(negatives)
            flat.extend(positives + negatives[: max(1, min(10, len(negatives)))])
        for epoch in range(epochs):
            rng.shuffle(flat)
            rate = learning_rate / math.sqrt(1.0 + epoch)
            for vector, label in flat:
                probability = sigmoid(bias + sum(w * x for w, x in zip(weights, vector)))
                error = probability - label
                for index, value in enumerate(vector):
                    weights[index] -= rate * (error * value + l2 * weights[index])
                bias -= rate * error
        for blend_weight in blend_grid:
            metrics = evaluate(
                validation_groups, weights, bias, blend_weight
            )
            candidates.append((
                metrics["mrr"],
                metrics["recall_at_1"],
                -blend_weight,
                -l2,
                weights,
                bias,
                metrics,
                l2,
                blend_weight,
            ))
    best = max(candidates, key=lambda item: item[:4])
    return {
        "weights": best[4],
        "bias": best[5],
        "validation_metrics": best[6],
        "selected_l2": best[7],
        "blend_weight": best[8],
        "blend_selection": "per_module_validation_grid_search",
        "training_groups": len(train_groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning", required=True)
    parser.add_argument("--nodes")
    parser.add_argument("--edges")
    parser.add_argument("--gold-train", required=True)
    parser.add_argument("--gold-validation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-groups-per-module", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=22366)
    parser.add_argument(
        "--blend-grid",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: %s" % out)
    if args.epochs <= 0 or args.max_groups_per_module <= 0:
        raise ValueError("epochs and max-groups-per-module must be positive")
    blend_grid = parse_blend_grid(args.blend_grid)

    train_gold = load_gold(args.gold_train, "train")
    validation_gold = load_gold(args.gold_validation, "validation")
    overlap = set(train_gold) & set(validation_gold)
    if overlap:
        raise ValueError("Train/Validation query overlap: %d" % len(overlap))
    wanted = collect_wanted_ids(args.reasoning)
    context = GraphContext.load(args.nodes, args.edges, wanted)
    train_groups, train_audit = make_groups(
        args.reasoning, train_gold, context, args.max_groups_per_module,
    )
    validation_groups, validation_audit = make_groups(
        args.reasoning, validation_gold, context, args.max_groups_per_module,
    )

    modules = {}
    for semantic_type in SEMANTIC_TYPES:
        train_values = train_groups.get(semantic_type, [])
        validation_values = validation_groups.get(semantic_type, [])
        if not train_values or not validation_values:
            continue
        module = train_one(
            train_values,
            validation_values,
            len(FEATURE_NAMES[semantic_type]),
            args.epochs,
            args.learning_rate,
            args.seed,
            blend_grid,
        )
        module.update({
            "feature_names": list(FEATURE_NAMES[semantic_type]),
            "validation_groups": len(validation_values),
        })
        modules[semantic_type] = module

    payload = {
        "schema": "task2_223_specialised_heads_v6_7",
        "implementation_revision": "v6_7_validation_guarded",
        "modules": modules,
        "train_gold_labels": len(train_gold),
        "validation_gold_labels": len(validation_gold),
        "train_validation_query_overlap": 0,
        "test_used_for_training_or_selection": False,
        "train_audit": train_audit,
        "validation_audit": validation_audit,
        "representation_encoder_frozen": True,
        "blend_grid": blend_grid,
        "module_blend_selected_on_validation": True,
        "role": "lightweight module-specific calibration; not a duplicate deep encoder",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if len(modules) != len(SEMANTIC_TYPES):
        raise SystemExit("ERROR: not every module has Train and Validation groups")


if __name__ == "__main__":
    main()
