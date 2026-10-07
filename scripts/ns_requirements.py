#!/usr/bin/env python3
"""
What each NeuronScope feature needs (Python packages, programs, llama.cpp
builds, drivers, device access and hardware), what this machine has, and the
command that fixes each gap on this OS. Also snapshots of the environment, so
"it worked last week" can be answered with a diff.

    python scripts/ns_requirements.py                    # every feature, short
    python scripts/ns_requirements.py check --feature studio --feature amd_rocm
    python scripts/ns_requirements.py check --json
    python scripts/ns_requirements.py install scikit-learn   # catalog packages only
    python scripts/ns_requirements.py snapshot               # -> ~/.neuronscope/env/<time>.json
    python scripts/ns_requirements.py diff                   # the last two snapshots
    python scripts/ns_requirements.py freeze > my-env.txt    # exact versions of the catalog's packages

The package version floors come from requirements*.txt, so the files stay the
one place a version is decided. doctor.py answers "can I run the pipeline
now?"; this answers "what does each feature need, and what changed?".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
SNAP_DIR = Path.home() / ".neuronscope" / "env"
OK, WARN, MISSING = "ok", "warn", "missing"

# ---------------------------------------------------------------- catalog
# Python packages: import name, why, and which requirements file decides the version.
# torch is deliberately not installable from here: the wheel depends on the GPU
# (install.sh picks CUDA, ROCm, Metal or CPU).
PACKAGES = {
    "numpy": ("numpy", "arrays, everywhere"),
    "requests": ("requests", "HTTP for evaluation and downloads"),
    "tqdm": ("tqdm", "progress bars"),
    "transformers": ("transformers", "tokenizers, chat templates and configs"),
    "safetensors": ("safetensors", "reading model weights"),
    "scikit-learn": ("sklearn", "the hallucination classifier and the subject classifier"),
    "pandas": ("pandas", "reading benchmark parquet files"),
    "pyarrow": ("pyarrow", "parquet support for pandas"),
    "openai": ("openai", "TestQA and answer collection through OpenAI-compatible servers"),
    "gguf": ("gguf", "GGUF metadata, expert sweeps, LoRA export"),
    "jinja2": ("jinja2", "chat templates"),
    "pillow": ("PIL", "images for the vision bank and CLIP"),
    "accelerate": ("accelerate", "loading models in PyTorch for extraction and fine-tuning"),
    "torch": ("torch", "the PyTorch path: extraction, CLIP neurons, fine-tuning, merging"),
    "aiohttp": ("aiohttp", "the transfer GUI's signalling server"),
    "mcp": ("mcp", "MCP client library (NeuronScope's own MCP server needs nothing)"),
    "keyring": ("keyring", "keeping MCP target tokens in the OS credential store"),
    "cryptography": ("cryptography", "self-signed TLS certificates for LAN Studio"),
    "fastplotlib": ("fastplotlib", "the desktop 2D/3D explorer"),
    "pygfx": ("pygfx", "the desktop 3D viewer"),
    "peft": ("peft", "LoRA fine-tuning"),
    "playwright": ("playwright", "rendering the web UIs to images (docs only)"),
}
NOT_PIP = {"torch": "run ./install.sh (it picks the wheel for your GPU), or see pytorch.org/get-started"}

# Programs: binaries to look for, a version command, and the package that provides
# them per package manager.
TOOLS = {
    "git": (["git"], ["--version"], {"apt": "git", "dnf": "git", "pacman": "git", "zypper": "git", "brew": "git"}),
    "cmake": (["cmake"], ["--version"], {"apt": "cmake", "dnf": "cmake", "pacman": "cmake", "zypper": "cmake",
                                          "brew": "cmake"}),
    "c++ compiler": (["c++", "g++", "clang++"], ["--version"],
                     {"apt": "build-essential", "dnf": "gcc-c++", "pacman": "base-devel", "zypper": "gcc-c++",
                      "brew": None}),
    "hipcc": (["hipcc", "/opt/rocm/bin/hipcc"], ["--version"],
              {"apt": "rocm-hip-sdk (from repo.radeon.com)", "dnf": "rocm-hip-devel", "pacman": "rocm-hip-sdk",
               "zypper": "rocm-hip-devel"}),
    "rocminfo": (["rocminfo", "/opt/rocm/bin/rocminfo"], None,
                 {"apt": "rocminfo", "dnf": "rocminfo", "pacman": "rocminfo", "zypper": "rocminfo"}),
    "glslc": (["glslc"], ["--version"], {"apt": "glslc libvulkan-dev", "dnf": "glslc vulkan-loader-devel",
                                         "pacman": "shaderc vulkan-headers", "zypper": "shaderc vulkan-devel",
                                         "brew": "shaderc"}),
    "vulkaninfo": (["vulkaninfo"], None, {"apt": "vulkan-tools mesa-vulkan-drivers",
                                          "dnf": "vulkan-tools mesa-vulkan-drivers",
                                          "pacman": "vulkan-tools vulkan-radeon", "zypper": "vulkan-tools Mesa-vulkan-device-select",
                                          "brew": "vulkan-tools"}),
    "nvidia-smi": (["nvidia-smi"], None, {"apt": "nvidia-driver (ubuntu-drivers install)", "dnf": "akmod-nvidia (RPM Fusion)",
                                          "pacman": "nvidia", "zypper": "nvidia drivers (opi nvidia)"}),
    "nvcc": (["nvcc", "/usr/local/cuda/bin/nvcc"], ["--version"],
             {"apt": "nvidia-cuda-toolkit, or CUDA from developer.nvidia.com", "dnf": "cuda (NVIDIA repo)",
              "pacman": "cuda", "zypper": "cuda (NVIDIA repo)"}),
    "docker or podman": (["podman", "docker"], ["--version"],
                         {"apt": "podman", "dnf": "podman", "pacman": "podman", "zypper": "podman", "brew": "podman"}),
    "godot": (["godot", "godot4"], ["--version"], {"apt": "godot (or the official download)", "dnf": "godot",
                                                   "pacman": "godot", "zypper": "godot", "brew": "--cask godot"}),
    "node": (["node"], ["--version"], {"apt": "nodejs", "dnf": "nodejs", "pacman": "nodejs", "zypper": "nodejs",
                                       "brew": "node"}),
}

# Features: what a person wants to do -> what it needs. "any" lists are satisfied by one.
FEATURES = {
    "core": {"title": "Core (Studio, TestQA, Projects, review)", "python": "3.10",
             "packages": ["numpy", "requests", "tqdm", "transformers", "safetensors", "scikit-learn", "pandas",
                          "pyarrow", "openai", "gguf", "jinja2", "pillow"]},
    "studio_models": {"title": "Run GGUF models in Studio", "llama": ["llama-server"],
                      "hardware": {"ram_gib": 8}},
    "activation_checks": {"title": "Per-reply checks, live scoring, neuron review", "llama": ["llama-cett-dump"]},
    "build_llama": {"title": "Build llama.cpp (Setup → Build)", "tools": ["git", "cmake", "c++ compiler"]},
    "amd_rocm": {"title": "AMD GPUs through ROCm (HIP)", "vendor": "amd", "tools": ["rocminfo"], "devices": ["kfd", "render_group"],
                 "llama_backend": "hip", "optional_tools": ["hipcc"]},
    "amd_vulkan": {"title": "AMD (and Intel) GPUs through Vulkan, incl. APUs like gfx90c",
                   "vendor": "amd/intel", "tools": ["vulkaninfo"], "devices": ["dri_render", "render_group"], "llama_backend": "vulkan",
                   "optional_tools": ["glslc"]},
    "nvidia_cuda": {"title": "NVIDIA GPUs through CUDA", "vendor": "nvidia", "tools": ["nvidia-smi"], "devices": ["nvidia"],
                    "llama_backend": "cuda", "optional_tools": ["nvcc"]},
    "torch_pipeline": {"title": "PyTorch pipeline (extraction, CLIP, fine-tuning, merging)",
                       "packages": ["torch", "accelerate"], "optional_packages": ["peft"],
                       "hardware": {"ram_gib": 16}},
    "testqa_sandbox": {"title": "TestQA code tasks in a container", "tools": ["docker or podman"]},
    "transfer": {"title": "Model transfer GUI and MCP extras", "packages": ["aiohttp"],
                 "optional_packages": ["mcp", "keyring", "cryptography"]},
    "desktop_viewers": {"title": "Desktop viewers (pygfx / fastplotlib)", "packages": ["fastplotlib", "pygfx"]},
    "godot_viewer": {"title": "Godot 3D client", "tools": ["godot"]},
}

DEVICE_CHECKS = {
    "kfd": ("/dev/kfd readable and writable (ROCm compute)",
            "sudo usermod -aG render,video $USER, then log out and in"),
    "dri_render": ("a /dev/dri/renderD* node you can open (Vulkan compute)",
                   "sudo usermod -aG render,video $USER, then log out and in"),
    "render_group": ("you are in the render (or video) group",
                     "sudo usermod -aG render,video $USER, then log out and in"),
    "nvidia": ("an NVIDIA GPU the driver can see", "install the NVIDIA driver for your distribution"),
}


# ---------------------------------------------------------------- probes

def package_manager() -> str | None:
    if sys.platform == "darwin":
        return "brew"
    try:
        info = dict(re.findall(r'^(\w+)="?([^"\n]*)"?', Path("/etc/os-release").read_text(), re.M))
    except OSError:
        info = {}
    ids = (info.get("ID", "") + " " + info.get("ID_LIKE", "")).split()
    for pm, names in (("apt", {"debian", "ubuntu", "linuxmint", "pop"}), ("dnf", {"fedora", "rhel", "centos", "rocky"}),
                      ("pacman", {"arch", "manjaro", "endeavouros", "cachyos"}), ("zypper", {"opensuse", "suse", "sles"})):
        if names & set(ids):
            return pm
    for pm in ("apt", "dnf", "pacman", "zypper"):
        if shutil.which(pm):
            return pm
    return None


def install_cmd(pm: str | None, pkg: str | None) -> str:
    if not pkg:
        return ""
    if "(" in pkg or "," in pkg:          # instructions, not a plain package name
        return pkg
    return {"apt": f"sudo apt install {pkg}", "dnf": f"sudo dnf install {pkg}", "pacman": f"sudo pacman -S {pkg}",
            "zypper": f"sudo zypper install {pkg}", "brew": f"brew install {pkg}"}.get(pm or "", pkg)


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_specs(root: Path = ROOT) -> dict:
    """{package: {"spec": ">=1.4", "file": "requirements-core.txt"}} from every requirements*.txt."""
    out = {}
    for f in sorted(root.glob("requirements*.txt")):
        for line in f.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            m = re.match(r"([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(.*)", line)
            if m:
                out.setdefault(_norm(m.group(1)), {"spec": m.group(3).replace(" ", ""), "file": f.name})
    return out


def _vtuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v.split("+")[0])[:4])


def satisfies(version: str, spec: str) -> bool:
    if not spec:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        return SpecifierSet(spec).contains(version, prereleases=True)
    except Exception:
        pass
    for part in spec.split(","):
        m = re.match(r"(>=|<=|==|!=|<|>|~=)(.+)", part)
        if not m:
            continue
        op, want = m.group(1), _vtuple(m.group(2))
        have = _vtuple(version)[:len(want)]
        if not {"<": have < want, "<=": have <= want, ">": have > want, ">=": have >= want,
                "==": have == want, "!=": have != want, "~=": have >= want}[op]:
            return False
    return True


def dist_version(name: str) -> str | None:
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def tool_version(binaries, vcmd) -> tuple[str | None, str | None]:
    for b in binaries:
        p = b if os.path.isabs(b) and os.access(b, os.X_OK) else shutil.which(b)
        if not p:
            continue
        if not vcmd:
            return p, ""
        try:
            r = subprocess.run([p, *vcmd], capture_output=True, text=True, timeout=8)
            line = next((ln.strip() for ln in (r.stdout + r.stderr).splitlines() if re.search(r"\d+\.\d+", ln)), "")
            m = re.search(r"\d+\.\d+(\.\d+)?", line)
            return p, m.group(0) if m else line[:60]
        except Exception:
            return p, ""
    return None, None


def llama_dirs() -> list[Path]:
    dirs = []
    for d in [os.environ.get("NS_LLAMA"), "~/llama.cpp"]:
        if d:
            p = Path(os.path.expanduser(d))
            dirs += [p / "build" / "bin", p]
    seen, out = set(), []
    for d in dirs:
        if d not in seen and d.is_dir():
            seen.add(d)
            out.append(d)
    return out


def find_llama(name: str, configured: dict | None = None) -> str | None:
    if configured and configured.get(name) and os.path.isfile(configured[name]):
        return configured[name]
    for d in llama_dirs():
        if (d / name).is_file():
            return str(d / name)
    return shutil.which(name)


def llama_backends(binary: str | None) -> list[str]:
    """Which GPU backends a llama.cpp build has, from the ggml libraries next to it
    (or linked into it)."""
    if not binary:
        return []
    d = Path(binary).resolve().parent
    found = set()
    for pat, be in (("*ggml-hip*", "hip"), ("*ggml-vulkan*", "vulkan"), ("*ggml-cuda*", "cuda"),
                    ("*ggml-metal*", "metal"), ("*ggml-sycl*", "sycl")):
        if glob.glob(str(d / pat)) or glob.glob(str(d.parent / "lib" / pat)):
            found.add(be)
    if not found:
        try:
            r = subprocess.run(["ldd", binary], capture_output=True, text=True, timeout=5)
            for key, be in (("amdhip", "hip"), ("libvulkan", "vulkan"), ("libcuda", "cuda")):
                if key in r.stdout:
                    found.add(be)
        except Exception:
            pass
    return sorted(found)


def gpu_vendors() -> set[str]:
    """GPU vendors on this machine, from sysfs and the NVIDIA driver."""
    out = set()
    try:
        import accelerators
        out |= {c["vendor"] for c in accelerators.drm_cards()}
    except Exception:
        pass
    if os.path.exists("/proc/driver/nvidia") or shutil.which("nvidia-smi"):
        out.add("nvidia")
    return out


def device_check(key: str) -> tuple[bool, str, str | None]:
    """-> (ok, what was found, a fix that differs from the default, if any)."""
    if key == "kfd":
        if not os.path.exists("/dev/kfd"):
            return False, "no /dev/kfd", ("the amdgpu kernel driver provides it: use a kernel with amdgpu "
                                          "(any recent distro kernel), and check `lsmod | grep amdgpu`")
        ok = os.access("/dev/kfd", os.R_OK | os.W_OK)
        return ok, "present and usable" if ok else "present but not accessible", None
    if key == "dri_render":
        nodes = sorted(glob.glob("/dev/dri/renderD*"))
        ok = [n for n in nodes if os.access(n, os.R_OK | os.W_OK)]
        return bool(ok), ", ".join(ok) if ok else ("not accessible: " + ", ".join(nodes) if nodes else
                                                   "no render nodes"), None if nodes or ok else "no GPU driver loaded"
    if key == "render_group":
        if sys.platform == "darwin":
            return True, "not needed on macOS", None
        try:
            import grp
            names = {grp.getgrgid(g).gr_name for g in os.getgroups()}
        except Exception:
            names = set()
        ok = bool(names & {"render", "video"}) or os.geteuid() == 0
        return ok, ", ".join(sorted(names & {"render", "video"})) or ("root" if ok else "in neither"), None
    if key == "nvidia":
        p, _ = tool_version(["nvidia-smi"], None)
        if not p:
            return False, "no nvidia-smi", None
        try:
            r = subprocess.run([p, "-L"], capture_output=True, text=True, timeout=8)
            gpus = [ln for ln in r.stdout.splitlines() if ln.startswith("GPU")]
            return bool(gpus), f"{len(gpus)} GPU(s)" if gpus else "the driver sees no GPU", None
        except Exception as e:
            return False, str(e), None
    return False, "unknown check", None


def hardware() -> dict:
    out = {"cpu": platform.processor() or platform.machine(), "machine": platform.machine(),
           "cores": os.cpu_count(), "ram_gib": None, "devices": [], "disk_free_gib": None}
    try:
        import accelerators
        r = accelerators._ram()
        if r:
            out["ram_gib"] = round(r[0] / 2**30, 1)
        out["devices"] = [{"id": d["id"], "name": d["name"], "backend": d["backend"], "vendor": d.get("vendor"),
                           "memory_gib": round((d.get("memory_total") or 0) / 2**30, 1), "unified": d.get("unified"),
                           "enabled": d.get("enabled", True)}
                          for d in accelerators.detect(accelerators.load_config())]
    except Exception as e:
        out["devices_error"] = str(e)
    try:
        home = Path.home() / ".neuronscope"
        out["disk_free_gib"] = round(shutil.disk_usage(home if home.exists() else Path.home()).free / 2**30, 1)
    except OSError:
        pass
    try:
        cpu = Path("/proc/cpuinfo").read_text()
        m = re.search(r"model name\s*:\s*(.+)", cpu)
        if m:
            out["cpu"] = m.group(1).strip()
    except OSError:
        pass
    return out


# ---------------------------------------------------------------- report

def check(features=None, configured: dict | None = None, hw: dict | None = None,
          vendors: set | None = None) -> dict:
    """-> {os, package_manager, features: {name: {title, status, items: [...]}}, hardware}.
    configured: binaries Studio was given ({"llama-server": path, "llama-cett-dump": path})."""
    pm = package_manager()
    specs = requirement_specs()
    hw = hw if hw is not None else hardware()
    out = {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
           "package_manager": pm, "hardware": hw, "features": {}}
    names = features or list(FEATURES)
    unknown = [f for f in names if f not in FEATURES]
    if unknown:
        raise ValueError(f"unknown feature(s) {unknown}; known: {', '.join(FEATURES)}")
    cache: dict = {}
    vendors = gpu_vendors() if vendors is None else vendors
    out["gpu_vendors"] = sorted(vendors)
    for fname in names:
        f = FEATURES[fname]
        items = []
        if f.get("vendor") and not set(f["vendor"].split("/")) & vendors:
            out["features"][fname] = {"title": f["title"], "status": "n/a", "items": [],
                                      "note": "no " + " or ".join({"amd": "AMD", "intel": "Intel", "nvidia": "NVIDIA"}[v]
                                                          for v in f["vendor"].split("/")) + " GPU on this machine"}
            continue
        if f.get("python"):
            ok = sys.version_info[:2] >= tuple(int(x) for x in f["python"].split("."))
            items.append({"kind": "python", "name": "python", "need": ">=" + f["python"],
                          "have": platform.python_version(), "status": OK if ok else MISSING,
                          "fix": "install Python " + f["python"] + " or newer" if not ok else ""})
        for opt, key in ((False, "packages"), (True, "optional_packages")):
            for p in f.get(key, []):
                v = dist_version(p)
                sp = specs.get(_norm(p), {})
                need = sp.get("spec", "")
                good = v is not None and satisfies(v, need)
                st = OK if good else (WARN if opt or v is not None else MISSING)
                if opt and v is None:
                    st = WARN
                fix = "" if good else NOT_PIP.get(p) or (f"pip install '{p}{need}'" if need else f"pip install {p}")
                items.append({"kind": "package", "name": p, "need": need or "any", "have": v,
                              "why": PACKAGES.get(p, ("", ""))[1], "from": sp.get("file"), "optional": opt,
                              "status": st, "fix": fix, "installable": not good and p not in NOT_PIP and p in PACKAGES})
        for opt, key in ((False, "tools"), (True, "optional_tools")):
            for t in f.get(key, []):
                if t not in cache:
                    cache[t] = tool_version(TOOLS[t][0], TOOLS[t][1])
                path, v = cache[t]
                items.append({"kind": "tool", "name": t, "need": "installed", "have": (v or "found") if path else None,
                              "path": path, "optional": opt, "status": OK if path else (WARN if opt else MISSING),
                              "fix": "" if path else install_cmd(pm, TOOLS[t][2].get(pm or "", None))
                              or f"install {t}"})
        for b in f.get("llama", []):
            p = find_llama(b, configured)
            items.append({"kind": "llama.cpp", "name": b, "need": "built", "have": p and "found", "path": p,
                          "backends": llama_backends(p), "status": OK if p else MISSING,
                          "fix": "" if p else "Setup → Build llama.cpp, or bash scripts/build_llama_tools.sh"})
        if f.get("llama_backend"):
            be = f["llama_backend"]
            srv = find_llama("llama-server", configured)
            have = llama_backends(srv)
            flag = {"hip": "--backend hip", "vulkan": "--backend vulkan", "cuda": "--backend cuda"}[be]
            items.append({"kind": "llama.cpp", "name": f"llama-server with {be}", "need": be,
                          "have": ", ".join(have) or ("cpu only" if srv else None), "optional": True,
                          "status": OK if be in have else WARN,
                          "fix": "" if be in have else f"Setup → Build llama.cpp with {be}, or "
                                                       f"bash scripts/build_llama_tools.sh {flag}"})
        for d in f.get("devices", []):
            ok, detail, special = device_check(d)
            what, fix = DEVICE_CHECKS[d]
            items.append({"kind": "device", "name": what, "have": detail, "status": OK if ok else MISSING,
                          "fix": "" if ok else special or fix})
        for k, v in (f.get("hardware") or {}).items():
            if k == "ram_gib" and hw.get("ram_gib") is not None:
                ok = hw["ram_gib"] >= v * 0.9
                items.append({"kind": "hardware", "name": "memory", "need": f"{v} GiB", "have": f"{hw['ram_gib']} GiB",
                              "status": OK if ok else WARN,
                              "fix": "" if ok else "smaller models or quantizations; more RAM"})
        req = [i for i in items if not i.get("optional")]
        status = (MISSING if any(i["status"] == MISSING for i in req) else
                  WARN if any(i["status"] != OK for i in items) else OK)
        out["features"][fname] = {"title": f["title"], "status": status, "items": items}
    return out


# ---------------------------------------------------------------- snapshots

def snapshot(configured: dict | None = None) -> dict:
    from importlib import metadata
    pk = {}
    for d in metadata.distributions():
        n = d.metadata.get("Name")
        if n:
            pk[_norm(n)] = d.version
    tools = {}
    for t, (bins, vcmd, _) in TOOLS.items():
        p, v = tool_version(bins, vcmd)
        if p:
            tools[t] = v or "found"
    llama = {}
    for b in ("llama-server", "llama-cett-dump"):
        p = find_llama(b, configured)
        if p:
            st = os.stat(p)
            llama[b] = {"path": p, "built": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
                        "size": st.st_size, "backends": llama_backends(p)}
    try:
        git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5).stdout.strip() or None
    except Exception:
        git = None
    return {"time": time.time(), "when": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
            "os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "executable": sys.executable, "neuronscope": git, "packages": pk, "tools": tools, "llama": llama,
            "hardware": hardware()}


def save_snapshot(snap: dict, d: Path | None = None) -> Path:
    d = Path(d) if d else SNAP_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / (time.strftime("%Y%m%d-%H%M%S", time.localtime(snap["time"])) + ".json")
    p.write_text(json.dumps(snap, indent=1))
    return p


def list_snapshots(d: Path | None = None) -> list[dict]:
    d = Path(d) if d else SNAP_DIR
    out = []
    for p in sorted(d.glob("*.json"), reverse=True) if d.is_dir() else []:
        try:
            s = json.loads(p.read_text())
            out.append({"id": p.stem, "when": s.get("when"), "host": s.get("host"), "packages": len(s.get("packages", {})),
                        "neuronscope": s.get("neuronscope")})
        except (OSError, ValueError):
            continue
    return out


def load_snapshot(sid: str, d: Path | None = None) -> dict:
    d = Path(d) if d else SNAP_DIR
    if not re.fullmatch(r"\d{8}-\d{6}", sid or ""):
        raise ValueError("snapshot ids look like 20260101-120000")
    return json.loads((d / f"{sid}.json").read_text())


def _dict_diff(a: dict, b: dict) -> dict:
    return {"added": {k: b[k] for k in sorted(set(b) - set(a))},
            "removed": {k: a[k] for k in sorted(set(a) - set(b))},
            "changed": {k: [a[k], b[k]] for k in sorted(set(a) & set(b)) if a[k] != b[k]}}


def diff(a: dict, b: dict) -> dict:
    """What changed from snapshot a to snapshot b."""
    dev = lambda s: {d["id"]: f"{d['name']} {d['backend']} {d['memory_gib']} GiB"   # noqa: E731
                     for d in (s.get("hardware") or {}).get("devices", [])}
    lla = lambda s: {k: f"{v['built']} {'+'.join(v['backends']) or 'cpu'}"          # noqa: E731
                     for k, v in (s.get("llama") or {}).items()}
    out = {"from": a.get("when"), "to": b.get("when"),
           "packages": _dict_diff(a.get("packages", {}), b.get("packages", {})),
           "tools": _dict_diff(a.get("tools", {}), b.get("tools", {})),
           "llama": _dict_diff(lla(a), lla(b)), "devices": _dict_diff(dev(a), dev(b)),
           "system": {k: [a.get(k), b.get(k)] for k in ("os", "python", "neuronscope", "host") if a.get(k) != b.get(k)}}
    out["unchanged"] = not any(v for sec in ("packages", "tools", "llama", "devices") for v in out[sec].values()) \
        and not out["system"]
    return out


def freeze() -> str:
    lines = [f"# NeuronScope environment, {time.strftime('%Y-%m-%d')}, Python {platform.python_version()}",
             "# Exact versions of the packages NeuronScope uses; pip install -r this file to reproduce."]
    for p in PACKAGES:
        v = dist_version(p)
        if v:
            lines.append(f"{p}=={v}" + ("   # install from the wheel index for your GPU" if p == "torch" else ""))
    return "\n".join(lines) + "\n"


def pip_install(pkg: str) -> int:
    """Install one catalog package at the version requirements*.txt asks for."""
    if pkg not in PACKAGES:
        raise SystemExit(f"{pkg} is not in the catalog ({', '.join(p for p in PACKAGES if p not in NOT_PIP)})")
    if pkg in NOT_PIP:
        raise SystemExit(f"{pkg}: {NOT_PIP[pkg]}")
    spec = requirement_specs().get(_norm(pkg), {}).get("spec", "")
    cmd = [sys.executable, "-m", "pip", "install", f"{pkg}{spec}"]
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


# ---------------------------------------------------------------- CLI

def _print(rep: dict, verbose: bool) -> None:
    mark = {OK: " ok ", WARN: "warn", MISSING: "MISS", "n/a": "n/a "}
    hw = rep["hardware"]
    print(f"{rep['os']} · Python {rep['python']} · package manager: {rep['package_manager'] or 'unknown'}")
    print(f"{hw.get('cpu')} · {hw.get('cores')} threads · {hw.get('ram_gib')} GiB RAM · "
          f"{hw.get('disk_free_gib')} GiB free")
    for d in hw.get("devices", []):
        print(f"  {d['id']:<10} {d['name']} ({d['memory_gib']} GiB{' shared' if d.get('unified') else ''})")
    for name, f in rep["features"].items():
        print(f"\n[{mark[f['status']]}] {f['title']}  ({name})" + (f": {f['note']}" if f.get("note") else ""))
        for i in f["items"]:
            if i["status"] == OK and not verbose:
                continue
            have = i.get("have") or "not found"
            opt = " (optional)" if i.get("optional") else ""
            print(f"    {mark[i['status']]}  {i['name']}{opt}: {have}" + (f", needs {i['need']}" if i.get('need') not in
                                                                          (None, 'installed', 'built', 'any') else ""))
            if i.get("fix"):
                print(f"          fix: {i['fix']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd")
    c = sub.add_parser("check", help="what each feature needs and what is missing (the default)")
    c.add_argument("--feature", action="append", choices=list(FEATURES))
    c.add_argument("--json", action="store_true")
    c.add_argument("-v", "--verbose", action="store_true", help="also list what is fine")
    i = sub.add_parser("install", help="pip install catalog packages at the versions requirements*.txt asks for")
    i.add_argument("package", nargs="+")
    s = sub.add_parser("snapshot", help="record packages, programs, llama.cpp builds and devices")
    s.add_argument("--print", action="store_true", help="print it instead of saving")
    d = sub.add_parser("diff", help="what changed between two snapshots (default: the last two)")
    d.add_argument("a", nargs="?")
    d.add_argument("b", nargs="?")
    sub.add_parser("snapshots", help="list saved snapshots")
    sub.add_parser("freeze", help="exact versions of the catalog's packages, as a requirements file")
    a = p.parse_args(argv)
    if a.cmd in (None, "check"):
        rep = check(getattr(a, "feature", None))
        if getattr(a, "json", False):
            print(json.dumps(rep, indent=1))
        else:
            _print(rep, getattr(a, "verbose", False))
        return 1 if rep["features"].get("core", {}).get("status") == MISSING else 0
    if a.cmd == "install":
        for pkg in a.package:
            if pip_install(pkg):
                return 1
        return 0
    if a.cmd == "snapshot":
        snap = snapshot()
        if a.print:
            print(json.dumps(snap, indent=1))
        else:
            print(f"saved {save_snapshot(snap)}")
        return 0
    if a.cmd == "snapshots":
        for s in list_snapshots():
            print(f"{s['id']}  {s['host']}  {s['packages']} packages  neuronscope {s['neuronscope']}")
        return 0
    if a.cmd == "diff":
        snaps = [x["id"] for x in list_snapshots()]
        ids = [a.a or (snaps[1] if len(snaps) > 1 else None), a.b or (snaps[0] if snaps else None)]
        if not all(ids):
            raise SystemExit("need two snapshots: run `snapshot` now and again after a change")
        r = diff(load_snapshot(ids[0]), load_snapshot(ids[1]))
        print(f"{r['from']} -> {r['to']}")
        if r["unchanged"]:
            print("  nothing changed")
        for k, v in r["system"].items():
            print(f"  {k}: {v[0]} -> {v[1]}")
        for sec in ("packages", "tools", "llama", "devices"):
            for k, v in r[sec]["changed"].items():
                print(f"  {sec[:-1] if sec != 'llama' else 'llama.cpp'} {k}: {v[0]} -> {v[1]}")
            for k, v in r[sec]["added"].items():
                print(f"  + {k} {v}")
            for k, v in r[sec]["removed"].items():
                print(f"  - {k} {v}")
        return 0
    if a.cmd == "freeze":
        sys.stdout.write(freeze())
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
