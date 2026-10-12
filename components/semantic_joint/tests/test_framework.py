from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from checkpointing import reload_checkpoint, save_checkpoint  # noqa: E402
from common import dump_json, sha256_file  # noqa: E402
from model import CleanJointRanker, masked_module_loss  # noqa: E402
from permutation_eval import evaluate_permutation_robustness  # noqa: E402
from preflight import run_preflight  # noqa: E402
from protocol import SplitFirewall, load_train_dev_without_deserializing_sealed  # noqa: E402


class FrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((PACKAGE / "config.json").read_text(encoding="utf-8"))

    def test_config_is_formally_bound_and_fixed(self):
        self.assertNotIn("REQUIRED_PLACEHOLDER", json.dumps(self.cfg))
        self.assertTrue(self.cfg["execution"]["authorized_after_v028_handoff"])
        self.assertEqual(len(self.cfg["fixed_seeds"]), 3)
        self.assertEqual(len(set(self.cfg["fixed_seeds"])), 3)
        self.assertEqual(set(self.cfg["arms"]), {
            "symbolic_only_clean", "unified_llm_gate_clean", "module_aware_clean"})
        self.assertTrue(self.cfg["apisem_learned"]["enabled"])
        self.assertEqual(self.cfg["apisem_learned"]["classification"],
                         "LOW_SAMPLE_AUXILIARY_MODULE")
        self.assertFalse(self.cfg["constraint_loss"]["enabled"])
        self.assertEqual(self.cfg["constraint_loss"]["weight"], 0.0)
        locked = copy.deepcopy(self.cfg)
        locked["clean_embedding_path"] = "REQUIRED_PLACEHOLDER"
        locked["execution"]["authorized_after_v028_handoff"] = False
        result = run_preflight(locked)
        self.assertEqual(result["status"], "STOP")
        self.assertIn("no_required_placeholder_remaining", result["failed_checks"])
        self.assertIn("clean_embedding_paths_resolved", result["failed_checks"])

    def test_deterministic_component_initialization(self):
        symbolic = CleanJointRanker("symbolic_only_clean", 1337)
        unified = CleanJointRanker("unified_llm_gate_clean", 1337)
        repeat = CleanJointRanker("symbolic_only_clean", 1337)
        different = CleanJointRanker("symbolic_only_clean", 3407)
        left = symbolic.initialization_hashes()
        right = unified.initialization_hashes()
        self.assertEqual(left["W_s.weight"], right["W_s.weight"])
        self.assertEqual(left["heads.TypeSem.weight"], right["heads.TypeSem.weight"])
        self.assertEqual(left, repeat.initialization_hashes())
        self.assertNotEqual(left["W_s.weight"], different.initialization_hashes()["W_s.weight"])

    def test_disabled_module_is_masked_not_zero_averaged(self):
        query = torch.tensor([[1.0, 0.0]])
        gold = torch.tensor([[1.0, 0.0]])
        type_item = {"module": "TypeSem", "logits": torch.tensor([3.0, -1.0], requires_grad=True),
                     "gold_index": 0, "query": query, "gold_candidate": gold}
        api_item = {"module": "APISem", "logits": torch.tensor([-100.0, 100.0], requires_grad=True),
                    "gold_index": 0, "query": query, "gold_candidate": gold}
        total, components = masked_module_loss([type_item, api_item], ["TypeSem"], 0.1)
        self.assertEqual(set(components), {"TypeSem"})
        self.assertAlmostEqual(float(total), float(components["TypeSem"]["total"]), places=7)

    def test_permutation_maps_back_to_identity(self):
        row = {"record_id": "r", "module": "LayoutSem",
               "candidate_ids": ["a", "b", "c"], "gold_candidate_id": "b"}
        scores = {"a": 0.2, "b": 0.9, "c": -0.1}
        report = evaluate_permutation_robustness(
            [row], lambda unused, ordered: [scores[value] for value in ordered], 1337, 5)
        self.assertEqual(report["permutations_per_query"], 5)
        self.assertEqual(report["top1_identity_agreement"], 1.0)
        self.assertEqual(report["mean_rank_correlation"], 1.0)
        self.assertEqual(report["mean_R1_variance"], 0.0)

    def test_split_firewall_never_deserializes_sealed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            rows = [
                {"record_id": "train", "global_split": "TRAIN"},
                {"record_id": "dev", "global_split": "DEV"},
                {"record_id": "cal", "global_split": "CALIBRATION", "secret": "sealed"},
                {"record_id": "test", "global_split": "TEST", "secret": "sealed"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            selected, audit = load_train_dev_without_deserializing_sealed(path)
            self.assertEqual([row["record_id"] for row in selected], ["train", "dev"])
            self.assertEqual(audit["sealed_rows_deserialized"], 0)
        firewall = SplitFirewall()
        with self.assertRaises(RuntimeError):
            firewall.claim_test_evaluation()
        with self.assertRaises(RuntimeError):
            firewall.assert_sealed()

    def test_checkpoint_unique_path_hash_and_real_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = CleanJointRanker("symbolic_only_clean", 1337, symbolic_dim=4,
                                     llm_dim=8, bridge_dim=3, modules=("TypeSem",))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            metadata = {"parameter_initialization_hashes": model.initialization_hashes(),
                        "manifest_sha256": "m", "candidate_order_sha256": "c",
                        "clean_embedding_sha256": "e", "test_evaluation_count": 0,
                        "calibration_evaluation_count": 0, "training_config_sha256": "t",
                        "trainer_code_sha256": "code"}
            path, manifest = save_checkpoint(tmp, "symbolic_only_clean", 1337,
                                             model, optimizer, metadata)
            self.assertEqual(manifest["checkpoint_sha256"], sha256_file(path / "ranker.pt"))

            def factory(payload):
                return CleanJointRanker(payload["arm"], payload["seed"],
                                        payload["symbolic_dim"], payload["llm_dim"],
                                        payload["bridge_dim"], tuple(payload["modules"]),
                                        payload["apisem_enabled"], payload["temperature"])

            def forward(reloaded):
                fused, _, _ = reloaded.fuse("TypeSem", torch.randn(1, 4))
                return reloaded.score("TypeSem", fused, torch.randn(2, 4))[0]

            result = reload_checkpoint(path, factory, forward)
            self.assertTrue(result["passed"])
            with self.assertRaises(RuntimeError):
                save_checkpoint(tmp, "symbolic_only_clean", 1337, model, optimizer, metadata)

    def test_mock_clean_embedding_preflight_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = copy.deepcopy(self.cfg)
            manifest = root / "manifest.jsonl"
            manifest.write_text('{"global_split":"TRAIN"}\n', encoding="utf-8")
            embedding = root / "clean.pt"
            index = root / "index.json"
            checkpoint = root / "checkpoint.pt"
            schema = root / "schema.json"
            vocab = root / "vocab.json"
            torch.save(torch.randn(12, 64), embedding)
            dump_json(index, {"id_%d" % value: value for value in range(12)})
            checkpoint.write_bytes(b"checkpoint")
            dump_json(schema, {"clean": True})
            dump_json(vocab, {"clean": True})
            final = root / "final.json"
            training = root / "training.json"
            leakage = root / "leakage.json"
            coverage = root / "coverage.json"
            export = root / "export.json"
            v027 = root / "v027.json"
            alignment = root / "training" / "alignment_metrics.json"
            clean_contract = {"checkpoint": str(checkpoint), "shape": [12, 64],
                              "index_count": 12, "nan_count": 0, "inf_count": 0,
                              "duplicate_node_id_count": 0, "reload_passed": True,
                              "real_forward_passed": True}
            dump_json(final, {"status": "READY_FOR_V0_2_9_CLEAN_JOINT",
                              "clean_hsym": clean_contract})
            dump_json(training, {"outer_metrics_computed": False})
            dump_json(leakage, {"passed": True, "forbidden_identity_feature_count": 0,
                                "hash_feature_count": 0})
            by_role_module = {}
            for role in ("TRAIN", "DEV"):
                by_role_module[role] = {}
                for module in ("TypeSem", "InterfaceSem", "LayoutSem", "APISem"):
                    by_role_module[role][module] = {"query_coverage": 1.0,
                                                    "candidate_coverage": 1.0,
                                                    "gold_coverage": 1.0}
            dump_json(coverage, {"performance_metrics_computed": False,
                                 "by_role_module": by_role_module})
            dump_json(export, {**clean_contract, "passed": True,
                               "outer_performance_metrics_computed": False})
            dump_json(v027, {"test_performance_read": False})
            dump_json(alignment, {
                "dataset": {"test_records": 0, "split_info": {"pair_counts": {"test": 0}}},
                "model": {"loss_weight_schedule": {"lambda_constraint_final": 0.0,
                                                     "lambda_constraint_start": 0.0}},
                "training_truth": {"optimizer_steps": 1, "gradient_steps": 1,
                                   "changed_parameter_tensors": 1,
                                   "max_parameter_abs_delta": 0.1,
                                   "nan_or_inf_loss": False},
                "evaluation_audit": {"test_performance_evaluations": 0,
                                     "outer_dev_records_opened": 0,
                                     "outer_cal_records_opened": 0,
                                     "outer_test_records_opened": 0}})
            cfg.update({
                "global_manifest": str(manifest),
                "expected_global_manifest_sha256": sha256_file(manifest),
                "v027_report": str(v027), "v028_root": str(root),
                "v028_final_report": str(final), "v028_training_report": str(training),
                "v028_feature_leakage_audit": str(leakage), "v028_coverage_report": str(coverage),
                "v028_export_report": str(export),
                "clean_embedding_path": str(embedding), "clean_embedding_index": str(index),
                "clean_alignment_checkpoint": str(checkpoint),
                "clean_feature_schema": str(schema), "clean_feature_vocab": str(vocab),
                "expected_clean_embedding_shape": [12, 64],
                "expected_clean_index_entries": 12,
            })
            cfg["expected_clean_artifact_sha256"] = {
                "checkpoint": sha256_file(checkpoint), "embedding": sha256_file(embedding),
                "index": sha256_file(index), "schema": sha256_file(schema),
                "vocab": sha256_file(vocab)}
            result = run_preflight(cfg)
            self.assertEqual(result["status"], "PASS", result)
            self.assertEqual(result["embedding_shape"], [12, 64])
            self.assertEqual(result["test_evaluation_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
