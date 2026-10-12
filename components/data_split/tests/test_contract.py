import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from build_global_split import canonical_candidate, balanced_order, norm_entry


def test_norm_entry():
    assert norm_entry("0x00401000") == "401000"


def test_canonical_candidate_drops_identity():
    node = {
        "node_type": "LayoutSem",
        "attrs": {
            "layout_kind": "struct",
            "offsets": [8, 0],
            "address": "deadbeef",
            "function_name": "secret_name",
            "raw_id": "unique",
        },
    }
    result = canonical_candidate(node)
    encoded = str(result)
    assert "deadbeef" not in encoded
    assert "secret_name" not in encoded
    assert "unique" not in encoded
    assert result["semantic"]["offsets"] == [0, 8]


def test_balanced_order_preserves_membership_and_gold():
    rows = []
    for index in range(7):
        rows.append({
            "record_id": str(index), "global_split": "DEV", "module": "TypeSem",
            "candidate_instance_ids": ["g" + str(index), "a" + str(index), "b" + str(index)],
            "gold_candidate_instance_id": "g" + str(index),
        })
    balanced_order(rows, "seed")
    assert sorted(row["gold_position"] for row in rows) == [0, 0, 0, 1, 1, 2, 2]
    for index, row in enumerate(rows):
        assert set(row["candidate_order"]) == {"g" + str(index), "a" + str(index), "b" + str(index)}
