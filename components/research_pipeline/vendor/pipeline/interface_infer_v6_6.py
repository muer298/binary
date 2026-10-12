#!/usr/bin/env python3
"""InterfaceInfer facade: ABI, entry def-use, roles and callsite consensus."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from module_specialized_v6_6 import GraphContext, score_candidate

SEMANTIC_TYPE = "InterfaceSem"
ALGORITHM = {
    "evidence": ("calling convention", "read-before-def", "parameter/return locations", "Type feedback"),
    "role_classes": ("buffer", "size", "fd", "output_ptr", "callback"),
    "cross_callsite_consistency": True,
    "neural_component": "optional lightweight role/rank calibration head",
}

def infer(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], context: GraphContext, heads: Mapping[str, Mapping[str, Any]]):
    return [score_candidate(SEMANTIC_TYPE, row, candidate, context, heads) for candidate in candidates]

