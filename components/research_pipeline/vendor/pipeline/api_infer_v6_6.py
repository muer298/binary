#!/usr/bin/env python3
"""APIInfer facade: strict provenance tiers, template and def-use evidence."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from module_specialized_v6_6 import GraphContext, score_candidate

SEMANTIC_TYPE = "APISem"
ALGORITHM = {
    "hard_sources": ("import_table", "symbol_table", "exact_signature"),
    "soft_sources": ("flirt", "byte_signature", "behavior_match"),
    "template": ("intent", "parameter roles", "pre/postconditions", "side effects"),
    "hard_seed_policy": "never fabricate; preserve verified 2.2.2 lineage only",
}

def infer(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], context: GraphContext, heads: Mapping[str, Mapping[str, Any]]):
    return [score_candidate(SEMANTIC_TYPE, row, candidate, context, heads) for candidate in candidates]

