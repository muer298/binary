#!/usr/bin/env python3
"""LayoutInfer facade: offsets, widths, alias completion and Type feedback."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from module_specialized_v6_6 import GraphContext, score_candidate

SEMANTIC_TYPE = "LayoutSem"
ALGORITHM = {
    "evidence": ("fixed offsets", "access widths", "stride", "alignment/padding"),
    "classification": ("struct", "array"),
    "cross_function_alias_completion": True,
    "maximum_nesting_depth": 3,
    "type_feedback": True,
}

def infer(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], context: GraphContext, heads: Mapping[str, Mapping[str, Any]]):
    return [score_candidate(SEMANTIC_TYPE, row, candidate, context, heads) for candidate in candidates]

