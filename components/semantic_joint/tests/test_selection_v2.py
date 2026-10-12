import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def module(name):
    spec = importlib.util.spec_from_file_location(name, SRC / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


SELECT = module("select_final")
PROTOCOL = module("protocol")


class CleanV2ContractTests(unittest.TestCase):
    def test_dev_mean_then_stability_selection(self):
        rule = {"fixed_arm_order": ["symbolic_only_clean", "unified_llm_gate_clean", "module_aware_clean"],
                "tie_tolerance": 1e-12}
        metrics = {
            "symbolic_only_clean": {"R@1": {"mean": 0.5, "std_population": 0.1}, "MRR": {"mean": 0.6}},
            "unified_llm_gate_clean": {"R@1": {"mean": 0.6, "std_population": 0.2}, "MRR": {"mean": 0.7}},
            "module_aware_clean": {"R@1": {"mean": 0.6, "std_population": 0.1}, "MRR": {"mean": 0.65}},
        }
        self.assertEqual(SELECT.choose_arm(list(rule["fixed_arm_order"]), metrics, rule), "module_aware_clean")

    def test_llm_contract_deserializes_selected_identity_only(self):
        with tempfile.TemporaryDirectory() as dirname:
            source = Path(dirname) / "source.jsonl"
            mapping = Path(dirname) / "mapping.jsonl"
            selected = json.dumps({"manifest_key": "selected", "pseudo_c": "safe"})
            # The sealed line is intentionally invalid JSON after the key. A
            # parser that deserializes it would fail; the clean loader must skip.
            sealed = '{"manifest_key": "sealed", "pseudo_c": INVALID}\n'
            source.write_text(selected + "\n" + sealed)
            mapping.write_text(json.dumps({"manifest_key": "selected", "span": [1, 2]}) + "\n" + sealed)
            src, spans = PROTOCOL.load_llm_contract(source, mapping, {"selected"})
        self.assertEqual(set(src), {"selected"})
        self.assertEqual(set(spans), {"selected"})


if __name__ == "__main__":
    unittest.main()
