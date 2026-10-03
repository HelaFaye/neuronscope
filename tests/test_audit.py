import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from profiles import Profile, SuppressionHandle, fingerprint
from gguf_utils import field_value


CFG = types.SimpleNamespace(
    model_type="qwen3",
    num_hidden_layers=4,
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    vocab_size=1000,
    _name_or_path="fake/model",
)


class FakeModel:
    def __init__(self, cfg, layers=4, inter=64, vision=True):
        self.config = cfg
        self._m = {}

        for i in range(layers):
            self._m[f"model.layers.{i}.mlp.down_proj"] = torch.nn.Linear(
                inter, 32
            )

        if vision:
            self._m[
                "vision_tower.encoder.layers.0.mlp.down_proj"
            ] = torch.nn.Linear(16, 99)

    def named_modules(self):
        return list(self._m.items())


@pytest.fixture
def profile():
    fp, geom = fingerprint(CFG)
    return Profile.create(
        fp,
        geom,
        "fake/model",
        {"0": [1, 2], "2": [5]},
        scale=0.1,
        n_layers=4,
        n_neurons=64,
        config_name="t1",
    )


def test_profile_carries_dimensions(profile):
    assert profile.data["n_neurons"] == 64
    assert profile.data["n_layers"] == 4

    spec = profile.data
    by_layer = {int(k): v for k, v in spec["by_layer"].items()}
    total = sum(len(v) for v in by_layer.values())
    pct = total / (spec["n_layers"] * spec["n_neurons"]) * 100
    assert abs(pct - 3 / 256 * 100) < 1e-9


def test_profile_roundtrip(profile):
    with tempfile.TemporaryDirectory() as d:
        path = profile.save(d)
        assert os.path.basename(path) == "t1.json"

        loaded = Profile.load(path)
        assert loaded.by_layer == {0: [1, 2], 2: [5]}
        assert loaded.data["n_neurons"] == 64
        assert loaded.data["n_layers"] == 4


def test_profile_rejects_geometry_mismatch(profile):
    other = types.SimpleNamespace(
        **{**vars(CFG), "intermediate_size": 128}
    )
    with pytest.raises(ValueError):
        profile.check(FakeModel(other, inter=128))


def test_profile_accepts_matching_geometry(profile):
    profile.check(FakeModel(CFG))


def test_hooks_only_target_text_layers(profile):
    model = FakeModel(CFG)
    handle = SuppressionHandle(model, profile)

    assert len(handle._targets) == 2
    assert {layer for layer, _ in handle._targets} == {0, 2}

    vision = model._m[
        "vision_tower.encoder.layers.0.mlp.down_proj"
    ]
    assert len(vision._forward_pre_hooks) == 0

    assert handle.scale == 0.1
    handle.set_scale(0.5)
    assert handle.scale == 0.5

    handle.remove()
    assert all(len(module._forward_pre_hooks) == 0 for module in model._m.values())


def _fingerprint_geometry():
    return fingerprint(CFG)


def test_rejects_out_of_range_neuron_index():
    fp, geom = _fingerprint_geometry()

    with pytest.raises(
        ValueError,
        match=r"references neuron 999",
    ):
        Profile.create(
            fp,
            geom,
            "fake/model",
            {"0": [999]},
            0.1,
            4,
            64,
        )


def test_rejects_negative_neuron_index():
    fp, geom = _fingerprint_geometry()

    with pytest.raises(
        ValueError,
        match=r"references neuron -1",
    ):
        Profile.create(
            fp,
            geom,
            "fake/model",
            {"0": [-1]},
            0.1,
            4,
            64,
        )


def test_rejects_duplicate_neuron_index():
    fp, geom = _fingerprint_geometry()

    with pytest.raises(
        ValueError,
        match=r"duplicate neuron 1",
    ):
        Profile.create(
            fp,
            geom,
            "fake/model",
            {"0": [1, 1]},
            0.1,
            4,
            64,
        )


def test_rejects_out_of_range_layer():
    fp, geom = _fingerprint_geometry()

    with pytest.raises(
        ValueError,
        match=r"layer 4 is outside",
    ):
        Profile.create(
            fp,
            geom,
            "fake/model",
            {"4": [1]},
            0.1,
            4,
            64,
        )


def test_rejects_nonexistent_layer():
    fp, geom = _fingerprint_geometry()

    with pytest.raises(
        ValueError,
        match=r"layer 9 is outside",
    ):
        Profile.create(
            fp,
            geom,
            "fake/model",
            {"9": [1]},
            0.1,
            4,
            64,
        )


def test_gguf_string_field():
    class FakeField:
        def __init__(self, value):
            self._value = value
            self.parts = {0: [value]}
            self.data = [0]

        def contents(self):
            return self._value

    assert field_value(FakeField(b"chat-template")) == "chat-template"


def test_gguf_numeric_field():
    class FakeField:
        def __init__(self, value):
            self._value = value
            self.parts = {0: [value]}
            self.data = [0]

        def contents(self):
            return self._value

    assert field_value(FakeField(32)) == 32


def test_layer_regex_defined_once():
    matches = []

    for path in SCRIPTS.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for lineno, line in enumerate(text.splitlines(), 1):
            if "TEXT_LAYER_RE =" in line:
                matches.append(f"{path}:{lineno}")

    assert len(matches) == 1, matches


def test_extractor_has_no_merge_conflict_markers():
    path = SCRIPTS / "extract_activations_gguf.py"
    text = path.read_text(encoding="utf-8")

    for marker in ("<" * 7, "=" * 7, ">" * 7, "|" * 7):
        assert marker not in text


def test_repository_has_no_merge_conflict_markers():
    files = [
        ROOT / ".gitignore",
        SCRIPTS / "extract_activations_gguf.py",
        SCRIPTS / "gguf_tokenizer.py",
        SCRIPTS / "vram_budget.py",
        ROOT / "tests" / "test_audit.py",
        ROOT / "viz" / "weights.py",
    ]

    for path in files:
        text = path.read_text(encoding="utf-8")
        for marker in ("<" * 7, "=" * 7, ">" * 7, "|" * 7):
            assert marker not in text, path


def test_git_diff_check():
    result = subprocess.run(
        ["git", "diff", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
