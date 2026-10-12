import importlib.util
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
SPEC = importlib.util.spec_from_file_location("train_clean", SRC / "train_clean.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class CleanStage1Tests(unittest.TestCase):
    def rows(self):
        return [
            {"split": "train", "global_role": "TRAIN", "source_function_id": "a",
             "source_code": "int a(){return 1;}", "pseudoc": "int a() {return 1;}"},
            {"split": "val", "global_role": "TRAIN", "source_function_id": "b",
             "source_code": "int b(){return 2;}", "pseudoc": "int b() {return 2;}"},
        ]

    def test_two_way_inner_split(self):
        audit, split = TRAIN.audit_pairs(self.rows())
        self.assertEqual(audit["rows"], {"train": 1, "val": 1})
        self.assertEqual(len(split["train"]), 1)

    def test_outer_role_is_rejected(self):
        rows = self.rows()
        rows[1]["global_role"] = "TEST"
        with self.assertRaises(RuntimeError):
            TRAIN.audit_pairs(rows)

    def test_exact_source_overlap_is_rejected(self):
        rows = self.rows()
        rows[1]["source_code"] = rows[0]["source_code"]
        with self.assertRaises(RuntimeError):
            TRAIN.audit_pairs(rows)


if __name__ == "__main__":
    unittest.main()
