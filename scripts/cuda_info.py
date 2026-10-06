#!/usr/bin/env python3
"""
NVIDIA/CUDA detection and the decisions that follow from it.

Everything here reads `nvidia-smi` (and `nvcc` if present), so it works before
PyTorch is installed and on machines where torch was built for the wrong
architecture. It answers, per machine:

  * which PyTorch wheel to install. Turing (sm_75) and newer use the current
    wheels. Maxwell, Pascal and Volta (sm_50-sm_72) need the CUDA 12.6 wheels,
    and PyTorch 2.14 is the last release that publishes them;
  * which CUDA architectures to build llama.cpp for, and whether the installed
    toolkit can: CUDA 13 cannot target anything below sm_75;
  * which training precision each GPU can use (bf16 needs sm_80; fast fp16
    needs sm_60 (P100) or sm_70+; consumer Pascal and Maxwell should train in
    fp32), and whether QLoRA (bitsandbytes) is available;
  * how to spread a model over several GPUs (a Tesla M10 is four 8 GB GPUs).

    python scripts/cuda_info.py             # report
    python scripts/cuda_info.py --json
    python scripts/cuda_info.py --torch-pip # what install.sh should pip install
    python scripts/cuda_info.py --cmake-archs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

GIB = 1024 ** 3

# Last driver branch for Maxwell/Pascal/Volta is R580 (NVIDIA, 2025). Newer
# branches do not load on these cards at all.
LEGACY_MAX_CC = 72          # Volta / Xavier and older
TORCH_LEGACY_INDEX = "https://download.pytorch.org/whl/cu126"
TORCH_LEGACY_SPEC = "torch<2.15"     # 2.14 is the last release with cu126 wheels

ARCH_NAMES = [(50, "Maxwell"), (60, "Pascal"), (70, "Volta"), (75, "Turing"), (80, "Ampere"),
              (89, "Ada"), (90, "Hopper"), (100, "Blackwell")]


def arch_name(cc: int) -> str:
    name = "unknown"
    for floor, n in ARCH_NAMES:
        if cc >= floor:
            name = n
    return name


def parse_smi(text: str) -> list[dict]:
    """Parse `nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.free,driver_version,pci.bus_id
    --format=csv,noheader,nounits` (MiB)."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        try:
            major, _, minor = parts[2].partition(".")
            cc = int(major) * 10 + int(minor or 0)
        except ValueError:
            cc = 0
        def mib(v):
            try:
                return int(float(v)) * 1024 * 1024
            except ValueError:
                return 0
        gpus.append({"index": int(parts[0]), "name": parts[1], "cc": cc, "arch": arch_name(cc),
                     "memory_total": mib(parts[3]), "memory_free": mib(parts[4]), "driver": parts[5],
                     "bus_id": parts[6] if len(parts) > 6 else ""})
    return gpus


def query_gpus() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,memory.free,"
                              "driver_version,pci.bus_id", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_smi(out.stdout) if out.returncode == 0 else []


def nvcc_version() -> tuple[int, int] | None:
    exe = shutil.which("nvcc")
    if not exe:
        if os.path.exists("/usr/local/cuda/bin/nvcc"):
            exe = "/usr/local/cuda/bin/nvcc"
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def precision(cc: int) -> dict:
    """Training precision for one GPU."""
    if cc >= 80:
        return {"dtype": "bf16", "why": "bf16 is native from Ampere (sm_80)"}
    if cc >= 70 or cc == 60:
        return {"dtype": "fp16", "why": "fast fp16 (Volta/Turing tensor cores, or the P100's 2x fp16 rate); no bf16"}
    return {"dtype": "fp32", "why": f"sm_{cc} has no fast fp16 (Maxwell and consumer Pascal run fp16 at a "
                                    "fraction of fp32 speed); train in fp32"}


def advise(gpus: list[dict], nvcc: tuple[int, int] | None = None) -> dict:
    """Decisions for this machine. Pure function of the inputs, so it is testable."""
    if not gpus:
        return {"gpus": [], "nvidia": False}
    ccs = sorted({g["cc"] for g in gpus if g["cc"]})
    legacy = [c for c in ccs if c <= LEGACY_MAX_CC]
    out = {"nvidia": True, "gpus": gpus, "count": len(gpus),
           "total_vram": sum(g["memory_total"] for g in gpus), "compute_caps": ccs,
           "legacy": bool(legacy), "notes": []}
    # ---- PyTorch
    if legacy:
        out["torch"] = {"index": TORCH_LEGACY_INDEX, "spec": TORCH_LEGACY_SPEC,
                        "why": f"sm_{legacy[0]} ({arch_name(legacy[0])}) is only in the CUDA 12.6 wheels; "
                               "PyTorch 2.14 is the last release that has them"}
        if len(legacy) != len(ccs):
            out["notes"].append("Mixed generations: the cu126 wheel covers every GPU here, the newer wheels "
                                "would not run on the older cards.")
    else:
        out["torch"] = {"index": None, "spec": "torch", "why": "Turing or newer: the default PyPI wheel"}
    # ---- llama.cpp
    archs = ";".join(f"{c}-real" for c in ccs)
    out["llama_cpp"] = {"cmake_archs": archs, "toolkit_max_major": 12 if legacy else None}
    if legacy:
        out["llama_cpp"]["why"] = (f"sm_{legacy[0]} needs a CUDA 12.x toolkit (12.9 is the newest); "
                                   "CUDA 13 cannot compile for anything below sm_75")
    if nvcc:
        out["nvcc"] = f"{nvcc[0]}.{nvcc[1]}"
        if legacy and nvcc[0] >= 13:
            out["llama_cpp"]["error"] = (f"nvcc {nvcc[0]}.{nvcc[1]} cannot build for sm_{legacy[0]}; install a "
                                         "CUDA 12.x toolkit (12.9) or build in the nvidia/cuda:12.9.x-devel "
                                         "container (docker/llama-cuda.Dockerfile)")
    # ---- driver
    if legacy:
        drivers = sorted({g["driver"] for g in gpus})
        major = max(int(d.split(".")[0]) for d in drivers if d.split(".")[0].isdigit()) if drivers else 0
        out["driver"] = {"installed": drivers, "note": "R580 is the last driver branch for Maxwell/Pascal/Volta; "
                                                       "pin it (newer branches drop these GPUs)"}
        if major > 580:
            out["notes"].append(f"driver {drivers[-1]} is newer than the R580 branch, which is the last that "
                                "supports these GPUs")
    # ---- training
    out["precision"] = {g["index"]: precision(g["cc"]) for g in gpus}
    worst = min(ccs) if ccs else 0
    if worst < 60:
        out["qlora"] = {"ok": False, "why": "bitsandbytes dropped Maxwell (0.49+); use --method lora in fp32"}
    elif worst < 75:
        out["qlora"] = {"ok": None, "why": "bitsandbytes on Pascal/Volta depends on the version; finetune.py "
                                           "--check reports whether it imports, and lora is the safe fallback"}
    else:
        out["qlora"] = {"ok": True, "why": "supported"}
    # ---- multi-GPU serving
    if len(gpus) > 1:
        mem = [g["memory_total"] for g in gpus]
        ts = ",".join(str(round(m / min(mem), 2)).rstrip("0").rstrip(".") for m in mem)
        out["multi_gpu"] = {
            "llama_server": f"-sm layer -ts {ts}",
            "why": "layer split puts whole layers on each GPU and passes one activation between them per "
                   "token, so it needs no fast GPU-to-GPU link",
            "row": "-sm row splits each matrix across GPUs; it needs fast peer links and is slower on PCIe "
                   "boards like the M10"}
        if worst <= LEGACY_MAX_CC and len(set(g["name"] for g in gpus)) == 1 and "M10" in gpus[0]["name"]:
            out["notes"].append("Tesla M10: four GPUs with 8 GB each on one board. Spread a model across all "
                                "four with layer split, or run up to four small models side by side "
                                "(Studio: visible GPUs per model).")
    return out


def report(a: dict) -> str:
    if not a.get("nvidia"):
        return "no NVIDIA GPU visible to nvidia-smi"
    L = [f"{a['count']} NVIDIA GPU(s), {a['total_vram'] / GIB:.1f} GiB total"]
    for g in a["gpus"]:
        L.append(f"  [{g['index']}] {g['name']}  sm_{g['cc']} ({g['arch']})  {g['memory_total'] / GIB:.1f} GiB  "
                 f"driver {g['driver']}  -> train in {a['precision'][g['index']]['dtype']}")
    t = a["torch"]
    L.append(f"PyTorch: pip install '{t['spec']}'" + (f" --index-url {t['index']}" if t["index"] else "")
             + f"\n  {t['why']}")
    lc = a["llama_cpp"]
    L.append(f"llama.cpp: -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='{lc['cmake_archs']}'"
             + (f"\n  {lc['why']}" if lc.get("why") else ""))
    if a.get("nvcc"):
        L.append(f"  nvcc {a['nvcc']}" + (f"  ERROR: {lc['error']}" if lc.get("error") else "  ok"))
    else:
        L.append("  nvcc not found (needed to build llama.cpp with CUDA)")
    if a.get("driver"):
        L.append(f"driver: {', '.join(a['driver']['installed'])}. {a['driver']['note']}")
    L.append(f"QLoRA: {'yes' if a['qlora']['ok'] else 'no' if a['qlora']['ok'] is False else 'maybe'} "
             f"({a['qlora']['why']})")
    if a.get("multi_gpu"):
        L.append(f"multi-GPU serving: llama-server {a['multi_gpu']['llama_server']}\n  {a['multi_gpu']['why']}")
    for n in a["notes"]:
        L.append("note: " + n)
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--json", action="store_true")
    g.add_argument("--torch-pip", action="store_true", help="print the pip arguments for torch (empty: no GPU)")
    g.add_argument("--cmake-archs", action="store_true", help="print CMAKE_CUDA_ARCHITECTURES for these GPUs")
    p.add_argument("--smi-file", help="read nvidia-smi CSV output from a file (testing, or another machine)")
    a = p.parse_args(argv)
    gpus = parse_smi(open(a.smi_file).read()) if a.smi_file else query_gpus()
    adv = advise(gpus, nvcc_version())
    if a.json:
        print(json.dumps(adv, indent=1))
    elif a.torch_pip:
        if adv.get("nvidia"):
            t = adv["torch"]
            print(t["spec"] + (f" --index-url {t['index']}" if t["index"] else ""))
    elif a.cmake_archs:
        if adv.get("nvidia"):
            print(adv["llama_cpp"]["cmake_archs"])
    else:
        print(report(adv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
