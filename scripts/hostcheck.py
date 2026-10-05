#!/usr/bin/env python3
"""
Host capability checks: refuse or warn before a mode exceeds this machine.

Several stages here have costs that are obvious in hindsight and invisible in
advance -- a MoE classifier matrix is 30GB on a model whose weights are 20GB,
and you find out an hour into loading. This estimates first.

    python scripts/hostcheck.py                     # what this machine has
    python scripts/hostcheck.py --mode classifier --layers 48 \\
        --experts 128 --neurons 768 --pairs 400 --train-mode 3-vs-1

As a library:

    from hostcheck import Host, check
    check("classifier", n_features=4_718_592, n_rows=1600)

Estimates are deliberately conservative and label their assumptions. A warning
you can override beats a crash four steps in, and beats a hard limit that is
wrong about your machine.
"""

import argparse
import os
import shutil
import subprocess

GIB = 1024 ** 3


class Host:
    def __init__(self):
        self.ram = self._ram()
        self.ram_available = self._ram_available()
        self.cpus = os.cpu_count() or 1
        self.vram, self.gpu_name = self._gpu()
        self.swap = self._swap()

    @staticmethod
    def _meminfo(key):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith(key):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        return None

    def _ram(self):
        v = self._meminfo("MemTotal:")
        if v:
            return v
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            return None

    def _ram_available(self):
        # MemAvailable, not MemFree: page cache is reclaimable and counting it
        # as used would understate what a big allocation can actually get.
        return self._meminfo("MemAvailable:")

    def _swap(self):
        return self._meminfo("SwapTotal:")

    def _gpu(self):
        """-> (bytes, name). Integrated GPUs report a shared-memory carve-out,
        which is a real limit for llama.cpp but comes out of system RAM."""
        try:
            out = subprocess.run(["vulkaninfo", "--summary"],
                                 capture_output=True, text=True, timeout=20).stdout
            name = None
            for line in out.splitlines():
                if "deviceName" in line:
                    name = line.split("=", 1)[-1].strip()
                    break
            if name:
                return None, name       # size not reported by --summary
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            import torch
            if torch.cuda.is_available():
                _, total = torch.cuda.mem_get_info()
                return total, torch.cuda.get_device_name(0)
        except Exception:
            pass
        return None, None

    def disk_free(self, path="."):
        try:
            return shutil.disk_usage(path).free
        except OSError:
            return None

    def report(self):
        def g(v):
            return f"{v / GIB:.1f} GiB" if v else "unknown"
        print(f"CPU cores      : {self.cpus}")
        print(f"RAM total      : {g(self.ram)}")
        print(f"RAM available  : {g(self.ram_available)}")
        print(f"swap           : {g(self.swap)}")
        print(f"GPU            : {self.gpu_name or 'none detected'}"
              + (f"  {g(self.vram)}" if self.vram else ""))
        print(f"disk free here : {g(self.disk_free())}")
        if self.gpu_name and "RADV" in self.gpu_name.upper():
            print("\nRADV detected. ROCm does not support Vega-class integrated "
                  "GPUs, so\nPyTorch runs on CPU here; use the llama.cpp Vulkan "
                  "path for extraction.")


def _fmt(v):
    return f"{v / GIB:.1f} GiB"


def check(mode, host=None, strict=False, **kw):
    """Estimate a mode's peak cost and warn. Returns (ok, [messages])."""
    host = host or Host()
    msgs, need_ram, need_disk, notes = [], 0, 0, []

    if mode == "classifier":
        n_features = kw["n_features"]
        n_rows = kw["n_rows"]
        need_ram = n_rows * n_features * 4
        notes.append(f"{n_rows} rows x {n_features:,} features, float32")
        if kw.get("solver") == "liblinear":
            need_ram *= 2
            notes.append("liblinear promotes to float64: doubled")

    elif mode == "extract":
        model_bytes = kw["model_bytes"]
        need_ram = int(model_bytes * 1.2)
        notes.append("model weights plus ~20% for activations and KV")
        if kw.get("n_samples") and kw.get("n_features"):
            need_disk = kw["n_samples"] * kw["n_features"] * 2 * kw.get("locations", 2)
            notes.append(f"{kw['n_samples']} samples x "
                         f"{kw['n_features']:,} features, fp16, "
                         f"{kw.get('locations', 2)} locations")

    elif mode == "merge":
        model_bytes = kw["model_bytes"]
        need_ram = int(model_bytes * 2.1)
        need_disk = model_bytes
        notes.append("both models resident in fp32-equivalent, plus output")

    elif mode == "train":
        params = kw.get("params", 9e9)
        need_ram = int(params * 2 * 3.5)
        notes.append("QLoRA rule of thumb: weights, gradients, optimizer state")
        if not host.vram:
            msgs.append("REFUSE: no trainable GPU detected. Training a 9B needs "
                        "24GB VRAM;\n  rent a card rather than attempting this "
                        "locally.")
            return False, msgs

    else:
        return True, [f"no estimate for mode {mode!r}"]

    for n in notes:
        msgs.append(f"  {n}")
    if need_ram:
        msgs.append(f"  peak RAM estimate: {_fmt(need_ram)}")
    if need_disk:
        msgs.append(f"  disk needed: {_fmt(need_disk)}")

    ok = True
    avail = host.ram_available or host.ram
    if need_ram and avail:
        if need_ram > avail:
            ok = False
            msgs.append(f"EXCEEDS HOST: needs {_fmt(need_ram)}, "
                        f"{_fmt(avail)} available.")
            if mode == "classifier":
                msgs.append("  Reduce --num_samples, use --train_mode 1-vs-1, "
                            "or for MoE\n  use --top-experts to cut the feature "
                            "count.")
            elif mode == "extract":
                msgs.append("  Use a smaller quant, or the llama.cpp path "
                            "instead of PyTorch.")
            elif mode == "merge":
                msgs.append("  Merge on a larger machine; this is a one-off "
                            "cost, unlike inference.")
        elif need_ram > avail * 0.8:
            msgs.append(f"TIGHT: needs {_fmt(need_ram)} of {_fmt(avail)} "
                        "available. Close other applications.")
    if need_disk:
        free = host.disk_free(kw.get("path", "."))
        if free and need_disk > free:
            ok = False
            msgs.append(f"EXCEEDS DISK: needs {_fmt(need_disk)}, "
                        f"{_fmt(free)} free.")

    return (ok or not strict), msgs


def guard(mode, strict=False, **kw):
    """Print the estimate, and exit if it will not fit and strict is set."""
    ok, msgs = check(mode, strict=strict, **kw)
    print(f"host check [{mode}]:")
    for m in msgs:
        print(m if m.startswith("  ") else f"  {m}")
    if not ok:
        if strict:
            raise SystemExit("  refusing to start; see above")
        print("  continuing anyway -- expect it to fail or swap heavily")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["classifier", "extract", "merge", "train"])
    p.add_argument("--layers", type=int)
    p.add_argument("--experts", type=int, default=1)
    p.add_argument("--neurons", type=int)
    p.add_argument("--pairs", type=int, default=400)
    p.add_argument("--train-mode", default="3-vs-1")
    p.add_argument("--model-gb", type=float)
    p.add_argument("--samples", type=int)
    a = p.parse_args()

    host = Host()
    host.report()
    if not a.mode:
        return

    print()
    kw = {}
    if a.mode == "classifier":
        if not (a.layers and a.neurons):
            raise SystemExit("--layers and --neurons required")
        kw["n_features"] = a.layers * a.experts * a.neurons
        kw["n_rows"] = a.pairs * 2 * (2 if a.train_mode == "3-vs-1" else 1)
    elif a.mode in ("extract", "merge"):
        if not a.model_gb:
            raise SystemExit("--model-gb required")
        kw["model_bytes"] = int(a.model_gb * GIB)
        if a.mode == "extract" and a.samples and a.layers and a.neurons:
            kw["n_samples"] = a.samples
            kw["n_features"] = a.layers * a.experts * a.neurons
    guard(a.mode, host=host, **kw)


if __name__ == "__main__":
    main()
