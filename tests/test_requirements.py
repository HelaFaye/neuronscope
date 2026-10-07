"""ns_requirements: the feature catalog, version checks, per-OS fixes and
environment snapshots."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ns_requirements as nr  # noqa: E402


def test_specs_come_from_the_requirements_files():
    specs = nr.requirement_specs()
    assert specs["scikit-learn"]["spec"] == ">=1.4" and specs["scikit-learn"]["file"] == "requirements-core.txt"
    assert specs["accelerate"]["file"] == "requirements.txt"
    assert "torch" not in specs                                   # install.sh decides torch
    # Every catalog feature names packages and tools the catalog knows.
    for f in nr.FEATURES.values():
        for p in f.get("packages", []) + f.get("optional_packages", []):
            assert p in nr.PACKAGES
        for t in f.get("tools", []) + f.get("optional_tools", []):
            assert t in nr.TOOLS


@pytest.mark.parametrize("v,spec,ok", [("1.5.2", ">=1.4", True), ("1.3", ">=1.4", False), ("3.11.0", ">=3.10,<4", True),
                                       ("4.0", ">=3.10,<4", False), ("2.3.1", "", True), ("24.1", ">=24", True)])
def test_satisfies(v, spec, ok):
    assert nr.satisfies(v, spec) is ok


def test_install_commands_per_os():
    assert nr.install_cmd("apt", "cmake") == "sudo apt install cmake"
    assert nr.install_cmd("pacman", "shaderc vulkan-headers") == "sudo pacman -S shaderc vulkan-headers"
    assert nr.install_cmd("dnf", "cuda (NVIDIA repo)") == "cuda (NVIDIA repo)"
    assert nr.install_cmd("apt", None) == ""


def test_check_reports_features_and_skips_absent_vendors(monkeypatch):
    rep = nr.check(hw={"ram_gib": 32.0, "devices": []}, vendors={"amd"})
    f = rep["features"]
    assert f["core"]["status"] in ("ok", "warn")                  # this test suite runs on the core install
    assert f["nvidia_cuda"]["status"] == "n/a" and "NVIDIA" in f["nvidia_cuda"]["note"]
    rocm = {i["name"]: i for i in f["amd_rocm"]["items"]}
    assert "/dev/kfd readable and writable (ROCm compute)" in rocm and "rocminfo" in rocm
    assert rocm["llama-server with hip"]["optional"]
    # A missing package names the pip command with the files' version floor.
    monkeypatch.setattr(nr, "dist_version", lambda name: None)
    rep = nr.check(["core"], hw={"ram_gib": 32.0}, vendors=set())
    sk = next(i for i in rep["features"]["core"]["items"] if i["name"] == "scikit-learn")
    assert sk["status"] == "missing" and sk["fix"] == "pip install 'scikit-learn>=1.4'" and sk["installable"]
    rep = nr.check(["torch_pipeline"], hw={"ram_gib": 8.0}, vendors=set())
    items = {i["name"]: i for i in rep["features"]["torch_pipeline"]["items"]}
    assert not items["torch"]["installable"] and "install.sh" in items["torch"]["fix"]
    assert items["memory"]["status"] == "warn"
    with pytest.raises(ValueError):
        nr.check(["nonsense"])


def test_snapshot_diff_and_freeze(tmp_path):
    a = {"when": "1", "os": "Linux 6.1", "python": "3.11", "packages": {"numpy": "1.26", "old": "1"},
         "tools": {"cmake": "3.28"}, "llama": {"llama-server": {"built": "2026-01-01", "backends": []}},
         "hardware": {"devices": [{"id": "cpu:0", "name": "x", "backend": "cpu", "memory_gib": 16}]}}
    b = json.loads(json.dumps(a))
    b.update(when="2", os="Linux 6.8")
    b["packages"] = {"numpy": "2.0", "new": "3"}
    b["llama"]["llama-server"] = {"built": "2026-02-01", "backends": ["vulkan"]}
    b["hardware"]["devices"].append({"id": "vulkan:0", "name": "gfx90c", "backend": "vulkan", "memory_gib": 16})
    d = nr.diff(a, b)
    assert d["packages"]["changed"] == {"numpy": ["1.26", "2.0"]} and d["packages"]["added"] == {"new": "3"}
    assert d["packages"]["removed"] == {"old": "1"} and d["system"]["os"] == ["Linux 6.1", "Linux 6.8"]
    assert "vulkan" in d["llama"]["changed"]["llama-server"][1] and "vulkan:0" in d["devices"]["added"]
    assert not d["unchanged"] and nr.diff(a, a)["unchanged"]
    p = nr.save_snapshot({**a, "time": 1767225600.0}, tmp_path)
    assert [s["id"] for s in nr.list_snapshots(tmp_path)] == [p.stem]
    assert nr.load_snapshot(p.stem, tmp_path)["os"] == "Linux 6.1"
    with pytest.raises(ValueError):
        nr.load_snapshot("../../etc/passwd", tmp_path)
    assert "numpy==" in nr.freeze()


def test_install_refuses_outside_the_catalog():
    with pytest.raises(SystemExit, match="not in the catalog"):
        nr.pip_install("evil-package")
    with pytest.raises(SystemExit, match="install.sh"):
        nr.pip_install("torch")


def test_cli_json():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "ns_requirements.py"), "check", "--feature", "core",
                        "--json"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["features"]["core"]["title"].startswith("Core")
