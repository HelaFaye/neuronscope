import json
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))

from quantization_lab import (
    CapabilityInput,
    QuantizationLabError,
    build_plan,
    build_weighted_calibration,
    parse_weighted_specs,
    tensor_regex_for_layers,
)


def write_profile(path, layers, neurons=64, n_layers=4):
    path.write_text(json.dumps({
        "version": 1,
        "config_name": path.stem,
        "fingerprint": "test",
        "n_layers": n_layers,
        "n_neurons": neurons,
        "by_layer": {str(k): v for k, v in layers.items()},
    }))


def test_tensor_regex():
    assert tensor_regex_for_layers([3, 1, 3]) == r"blk\.(1|3)\.ffn_down(_exps)?\.weight"
    assert tensor_regex_for_layers([2], include_shared_expert=True) == r"blk\.(2)\.ffn_down(_exps|_shexp)?\.weight"


def test_profile_validation_rejects_duplicates_and_bad_layer():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.json"
        write_profile(p, {0: [1, 1]})
        with pytest.raises(QuantizationLabError, match="duplicate"):
            parse_weighted_specs([str(p)]) and __import__("quantization_lab").load_profiles(parse_weighted_specs([str(p)]))

        p2 = Path(d) / "bad_layer.json"
        write_profile(p2, {9: [1]}, n_layers=4)
        with pytest.raises(QuantizationLabError, match="outside"):
            __import__("quantization_lab").load_profiles(parse_weighted_specs([str(p2)]))


def test_weighted_calibration_is_deterministic_and_bounded():
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "a.txt"; b = Path(d) / "b.txt"; out1 = Path(d) / "o1.txt"; out2 = Path(d) / "o2.txt"
        a.write_text("a1\na2\na3\na4\n")
        b.write_text("b1\nb2\nb3\nb4\n")
        inputs = [CapabilityInput(str(a), 2.0, "a"), CapabilityInput(str(b), 1.0, "b")]
        m1 = build_weighted_calibration(inputs, str(out1), 6)
        m2 = build_weighted_calibration(inputs, str(out2), 6)
        assert m1["line_count"] == 6
        assert out1.read_text() == out2.read_text()


def test_build_plan_selects_highest_scoring_layer():
    with tempfile.TemporaryDirectory() as d:
        model = Path(d) / "model-F16.gguf"
        model.write_bytes(b"dummy")
        p1 = Path(d) / "math.json"; p2 = Path(d) / "general.json"
        write_profile(p1, {2: list(range(20)), 0: [1]})
        write_profile(p2, {0: list(range(10))})
        plan = build_plan(
            str(model), "Q4_K_M", "Q6_K", 25,
            [CapabilityInput(str(p1), 2.0, "math"), CapabilityInput(str(p2), 1.0, "general")],
        )
        assert plan["selected_layers"] == [2]
        assert plan["tensor_regex"] == r"blk\.(2)\.ffn_down(_exps)?\.weight"
