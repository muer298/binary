#!/usr/bin/env python3
"""TypeInfer facade: hierarchy, instruction, kNN and inherited constraints."""
from __future__ import annotations
from typing import Any, Dict, Mapping, Sequence
from module_specialized_v6_6 import GraphContext, score_candidate

SEMANTIC_TYPE = "TypeSem"
ALGORITHM = {
    "candidate_space": "base/pointer/array/struct/function hierarchy",
    "evidence": ("instruction rules", "kNN", "API/Interface", "Layout feedback"),
    "compatibility_filter": True,
    "neural_component": "optional lightweight multilabel calibration head",
}

def infer(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], context: GraphContext, heads: Mapping[str, Mapping[str, Any]]):
    return [score_candidate(SEMANTIC_TYPE, row, candidate, context, heads) for candidate in candidates]

