import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("common", ROOT / "src" / "common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def test_centered_window_preserves_late_target():
    result = common.centered_window(2000, [1200, 1201, 1202], 512)
    assert result["window_start"] > 0
    assert result["window_start"] <= 1200
    assert result["window_end"] >= 1203
    assert result["target_local_start"] == 1200 - result["window_start"]


def test_centered_window_preserves_discontinuous_target_envelope():
    result = common.centered_window(900, [400, 402, 405], 512)
    assert result["target_global_start"] == 400
    assert result["target_global_end"] == 406
    assert result["target_local_indices"] == [x - result["window_start"] for x in [400, 402, 405]]


def test_centered_window_rejects_too_wide_target():
    assert common.centered_window(2000, [100, 700], 512) is None


def test_address_is_hex():
    assert common.address_int("0x401000") == 0x401000
    assert common.address_int("00401000") == 0x401000


def test_exporter_has_real_ancestry_apis_and_no_distance_heuristic():
    text = (ROOT / "ghidra_scripts" / "ExportSeraphMappingV022.py").read_text()
    for required in ("getPcodeOp", "getSeqnum", "getMinAddress", "getMaxAddress", "Parent", "getVarnode"):
        assert required in text
    assert "distance" not in text.lower()
