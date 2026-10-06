"""CUDA decisions from nvidia-smi output: wheels, build archs, precision, multi-GPU."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cuda_info  # noqa: E402

M10 = "".join(f"{i}, Tesla M10, 5.0, 8192, 8100, 580.95.05, 00000000:0{4 + i}:00.0\n" for i in range(4))


def test_m10_needs_legacy_stack():
    a = cuda_info.advise(cuda_info.parse_smi(M10), (13, 0))
    assert a["count"] == 4 and a["total_vram"] == 32 * 2**30
    assert a["torch"] == {**a["torch"], "index": "https://download.pytorch.org/whl/cu126", "spec": "torch<2.15"}
    assert a["llama_cpp"]["cmake_archs"] == "50-real" and "CUDA 12" in a["llama_cpp"]["error"]
    assert all(p["dtype"] == "fp32" for p in a["precision"].values())
    assert a["qlora"]["ok"] is False
    assert a["multi_gpu"]["llama_server"] == "-sm layer -ts 1,1,1,1"
    assert cuda_info.advise(cuda_info.parse_smi(M10), (12, 9))["llama_cpp"].get("error") is None


def test_modern_and_mixed():
    a = cuda_info.advise(cuda_info.parse_smi("0, NVIDIA GeForce RTX 4090, 8.9, 24564, 23000, 590.10, x\n"), (13, 1))
    assert a["torch"]["index"] is None and a["precision"][0]["dtype"] == "bf16" and a["qlora"]["ok"] is True
    assert "error" not in a["llama_cpp"] and "multi_gpu" not in a
    mixed = cuda_info.advise(cuda_info.parse_smi(
        "0, NVIDIA GeForce RTX 3060, 8.6, 12288, 12000, 580.1, a\n1, Tesla P40, 6.1, 24576, 24000, 580.1, b\n"))
    assert mixed["torch"]["index"].endswith("cu126")
    assert mixed["llama_cpp"]["cmake_archs"] == "61-real;86-real"
    assert mixed["precision"][1]["dtype"] == "fp32" and mixed["precision"][0]["dtype"] == "bf16"
    assert mixed["multi_gpu"]["llama_server"] == "-sm layer -ts 1,2"
    assert cuda_info.precision(60)["dtype"] == "fp16" and cuda_info.precision(70)["dtype"] == "fp16"


def test_no_gpu_and_cli(tmp_path, capsys):
    assert cuda_info.advise([]) == {"gpus": [], "nvidia": False}
    f = tmp_path / "smi.csv"
    f.write_text(M10)
    cuda_info.main(["--smi-file", str(f), "--torch-pip"])
    assert capsys.readouterr().out.strip() == "torch<2.15 --index-url https://download.pytorch.org/whl/cu126"
    cuda_info.main(["--smi-file", str(f), "--cmake-archs"])
    assert capsys.readouterr().out.strip() == "50-real"


def test_doctor_flags_a_wheel_without_kernels(monkeypatch):
    import doctor
    from types import SimpleNamespace
    fake_torch = SimpleNamespace(__version__="2.14.1+cu130", version=SimpleNamespace(hip=None),
                                 cuda=SimpleNamespace(is_available=lambda: True,
                                                      get_arch_list=lambda: ["sm_75", "sm_80", "sm_86"]))
    monkeypatch.setattr(cuda_info, "nvcc_version", lambda: (13, 0))
    doctor.results.clear()
    doctor.check_cuda(cuda_info.parse_smi(M10), fake_torch)
    rows = {r["name"]: r for r in doctor.results}
    assert rows["torch kernels"]["status"] == "fail" and "whl/cu126" in rows["torch kernels"]["fix"]
    assert rows["CUDA toolkit"]["status"] == "fail" and "12" in rows["CUDA toolkit"]["fix"]
    assert sum(1 for r in doctor.results if r["name"].startswith("GPU ")) == 4
