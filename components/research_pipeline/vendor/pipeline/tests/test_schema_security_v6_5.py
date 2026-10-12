#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


HERE = Path(__file__).resolve().parent
SRC = HERE if (HERE / "conflict_resolver_v6_5.py").is_file() else HERE.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if TORCH_AVAILABLE:
    from conflict_resolver_v6_5 import (  # noqa: E402
        LEVEL_DETERMINISTIC,
        LEVEL_HIGH_CONFIDENCE,
        LEVEL_STATISTICAL,
        ConflictResolver,
        ConstraintCompiler,
        ConstraintSpec,
        UnsupportedConstraint,
        deduplicate_constraints,
        rank_gate,
    )


class FormalWritebackGateTest(unittest.TestCase):
    def test_formal_override_entrypoints_are_fail_closed(self):
        result = subprocess.run(
            [sys.executable, str(SRC / "14_block_direct_semantic_writeback_v6_5.py")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn('"allowed": false', result.stderr.lower())


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for resolver security tests")
class ResolverSchemaSecurityTest(unittest.TestCase):
    def test_missing_rank_or_rr_fails_requested_filter(self):
        self.assertFalse(rank_gate({}, require_rank1=True))
        self.assertFalse(rank_gate({"rank": 1}, min_rr=0.5))
        self.assertTrue(rank_gate({"rank": 1, "reciprocal_rank": 1.0}, True, 0.5))

    def test_api_role_and_source_are_required_for_level3(self):
        api = "s::APISem::malloc"
        target = "s::TypeSem::ret"
        nodes = {
            api: {"node_id": api, "node_type": "APISem", "api_name": "malloc", "api_source": "symbol_table"},
            target: {"node_id": target, "node_type": "TypeSem", "api_role": "return"},
        }
        compiler = ConstraintCompiler(nodes, {
            api: torch.tensor([1.0, 0.0]),
            target: torch.tensor([0.0, 1.0]),
        })
        raw = {"edge_id": "api-ret", "edge_kind": "evidence", "evidence_type": "api_call"}
        resolved = {
            "edge_id": "api-ret",
            "edge_class": "evidence",
            "source_node": api,
            "target_node": target,
            "strength": 1.0,
        }
        compiled = compiler.compile_edge(raw, resolved)
        self.assertEqual(len(compiled), 1)
        self.assertEqual(compiled[0].level, LEVEL_DETERMINISTIC)
        self.assertEqual(compiled[0].allowed_candidates, ("ptr",))
        self.assertTrue(compiled[0].hard_source_verified)

        no_role_target = "s::TypeSem::unknown"
        compiler_missing = ConstraintCompiler(
            {api: nodes[api], no_role_target: {"node_id": no_role_target, "node_type": "TypeSem"}},
            {api: torch.tensor([1.0, 0.0]), no_role_target: torch.tensor([0.0, 1.0])},
        )
        resolved_missing = dict(resolved, target_node=no_role_target, edge_id="api-no-role")
        self.assertEqual(compiler_missing.compile_edge(raw, resolved_missing), [])
        self.assertEqual(compiler_missing.audit["api_role_missing"], 1)

        unverified_nodes = dict(nodes)
        unverified_nodes[api] = dict(nodes[api], api_source="behavior_match")
        compiler_unverified = ConstraintCompiler( unverified_nodes, {
            api: torch.tensor([1.0, 0.0]), target: torch.tensor([0.0, 1.0])
        })
        downgraded = compiler_unverified.compile_edge(raw, resolved)[0]
        self.assertEqual(downgraded.level, LEVEL_HIGH_CONFIDENCE)
        self.assertFalse(downgraded.hard_source_verified)

    def test_rank_filter_missing_data_downgrades_model_evidence(self):
        target = "s::LayoutSem::l1"
        source = "s::InstrSem::i1"
        compiler = ConstraintCompiler(
            {
                source: {"node_id": source, "node_type": "InstrSem"},
                target: {"node_id": target, "node_type": "LayoutSem"},
            },
            {source: torch.tensor([1.0, 0.0]), target: torch.tensor([0.0, 1.0])},
            require_model_rank1=True,
        )
        raw = {"edge_id": "fixed", "edge_kind": "evidence", "evidence_type": "fixed_offset"}
        resolved = {
            "edge_id": "fixed",
            "edge_class": "evidence",
            "source_node": source,
            "target_node": target,
            "strength": 0.95,
        }
        compiled = compiler.compile_edge(raw, resolved)
        self.assertEqual(compiled[0].level, LEVEL_STATISTICAL)
        self.assertEqual(compiler.audit["model_rank_gate_downgraded"], 1)

    def test_deduplication_keeps_higher_level_lock(self):
        node = "s::TypeSem::t"
        common = dict(
            node_id=node,
            weight=0.8,
            predicate_name="selected_in_allowed",
            allowed_candidates=("ptr",),
            raw_hash="same",
        )
        soft = ConstraintSpec(constraint_id="soft", level=1, **common)
        hard = ConstraintSpec(constraint_id="hard", level=3, **common)
        output, audit = deduplicate_constraints([soft, hard])
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].level, 3)
        self.assertEqual(audit["semantic_duplicates_suppressed"], 1)

    def test_unregistered_operations_are_not_executed(self):
        node = "s::TypeSem::t"
        unknown = ConstraintSpec(
            constraint_id="unknown",
            node_id=node,
            level=LEVEL_HIGH_CONFIDENCE,
            weight=1.0,
            predicate_name="__import___os_system",
            projection_name="eval",
            constraint_vector=torch.tensor([1.0, 0.0]),
            raw_hash="unknown",
        )
        result = ConflictResolver(max_iterations=2).resolve(
            node, {"int": 1.0}, torch.tensor([0.0, 1.0]), [unknown]
        )
        self.assertGreater(result["active_conflict_count"], 0)
        with self.assertRaises(UnsupportedConstraint):
            ConflictResolver(max_iterations=1, fail_on_unsupported=True).resolve(
                node, {"int": 1.0}, torch.tensor([0.0, 1.0]), [unknown]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
