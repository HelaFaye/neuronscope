#!/usr/bin/env python3
"""
NeuronScope Studio: model manager, server supervisor and chat, over llama-server.

Scope, deliberately narrow. This does not implement inference, tokenization,
sampling or GGUF loading -- llama.cpp does all of that, better than a
reimplementation would. What this adds is the layer around it: finding models,
remembering per-model settings, starting and stopping the server, a chat
window, and the runtime controls LM Studio does not expose.

That last part is the reason to build it at all:

    LoRA scale slider     live suppression strength via /lora-adapters
    expert count          --override-kv <arch>.expert_used_count for MoE
    host fit estimate     before loading, not after it fails

Everything else here is table stakes that any front end needs.

    python viz/studio.py --models-dir ~/.models --server ~/llama.cpp/build/bin/llama-server
    python viz/studio.py --models-dir A --models-dir B --host 0.0.0.0

Stdlib only, except the optional `gguf` package for reading model metadata.
No build step, no bundled browser, and it works over the LAN because the GPU is
often on another machine.

It also exposes an OpenAI-compatible API at /v1 (models, chat, completions,
embeddings) with just-in-time loading: a request naming another model loads
it, `"model": "auto"` picks one from the subject classifier and each model's
rolling performance stats (models without stats are never auto-picked), and --idle-ttl unloads after inactivity. Vision models are paired with
their mmproj file automatically and accept image attachments in chat.

Binding beyond loopback requires --token (or NS_STUDIO_TOKEN), and TLS unless
--allow-plaintext is passed for a VPN / reverse-proxy deployment.
"""

import argparse
import glob
import hashlib
import hmac
import http.cookies
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))
import gguf_utils
import ns_security as sec
import model_stats
import jobs as ns_jobs
import rag as ns_rag
import mcp_client as ns_mcp
import ns_pairing
try:
    from hostcheck import Host, check as host_check
except ImportError:
    Host, host_check = None, None

STATE = {"models_dirs": [], "server_bin": None, "settings_path": None,
         "download_dir": None, "offline": False, "token": None,
         "chats_dir": None, "stats": None, "min_graded": 20, "min_subject": 5,
         "halluc_cost": 1.0, "traces_dir": None, "cett": None, "score_ngl": 0, "score_every": 1,
         "idle_ttl": 0, "jit": True,
         "backend_port": 8080, "tls": False}
ACTIVITY = {"last": time.time(), "active": 0}
ACTIVITY_LOCK = threading.Lock()
MAX_BODY = 64 * 1024 * 1024        # chat bodies may carry base64 images
MAX_SMALL_BODY = 256 * 1024
THROTTLE = sec.FailureThrottle()
JOBS = {}          # download id -> progress record
JOBS_LOCK = threading.Lock()

HF_API = "https://huggingface.co/api"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"
LOCK = threading.Lock()
PROC = {"proc": None, "model": None, "port": None, "args": None,
        "started": None, "lora": None, "lora_scale": 1.0}


# ------------------------------------------------------------------ metadata

def _gguf_meta(path):
    """Architecture, parameter and quant info. Falls back to filename cues."""
    out = {"arch": None, "n_layers": None, "n_experts": None,
           "context": None, "quant": None, "params": None}
    m = re.search(r"(Q\d+_[A-Z0-9_]+|IQ\d+_[A-Z]+|F16|BF16|F32)",
                  os.path.basename(path), re.I)
    if m:
        out["quant"] = m.group(1).upper()
    try:
        import gguf
    except ImportError:
        return out
    try:
        r = gguf.GGUFReader(path)

        def kv(key):
            return gguf_utils.read_kv(r, key)

        arch = kv("general.architecture")
        out["arch"] = arch
        if arch:
            out["n_layers"] = kv(f"{arch}.block_count")
            out["n_experts"] = kv(f"{arch}.expert_count")
            out["n_experts_used"] = kv(f"{arch}.expert_used_count")
            out["context"] = kv(f"{arch}.context_length")
        out["params"] = kv("general.parameter_count") or kv("general.size_label")
    except Exception as e:
        out["error"] = str(e)[:120]
    return out


_META_CACHE = {}


MMPROJ_RE = re.compile(r"(^|[-_.])mmproj([-_.]|$)", re.I)


def is_mmproj(path):
    return bool(MMPROJ_RE.search(os.path.basename(path)))


def model_id(path):
    """Stable OpenAI-style id: <repo dir>/<file stem>, multi-part suffix removed."""
    stem = re.sub(r"-\d{5}-of-\d{5}$", "", os.path.basename(path)[:-5])
    return f"{os.path.basename(os.path.dirname(path))}/{stem}".lower()


def scan_models():
    """Every .gguf under the configured directories, with cached metadata.

    Multi-part files (`-00001-of-0000N`) are listed once, by their first part,
    which is what llama.cpp wants passed to -m. Vision projector files
    (mmproj-*.gguf) are not models; they are attached to the models in the
    same directory so vision just works when loaded.
    """
    found, seen = [], set()
    for root in STATE["models_dirs"]:
        for path in sorted(glob.glob(os.path.join(root, "**", "*.gguf"),
                                     recursive=True)):
            if is_mmproj(path):
                continue
            base = os.path.basename(path)
            part = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", base)
            if part:
                if part.group(1) != "00001":
                    continue
                key = base[:part.start()]
                if key in seen:
                    continue
                seen.add(key)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            st = os.stat(path)
            ck = (path, st.st_mtime, size)
            if ck not in _META_CACHE:
                _META_CACHE[ck] = _gguf_meta(path)
            meta = _META_CACHE[ck]
            projectors = sorted(p for p in glob.glob(os.path.join(os.path.dirname(path), "*.gguf"))
                                if is_mmproj(p))
            found.append({
                "path": path, "name": base, "size": size, "id": model_id(path),
                "publisher": os.path.basename(os.path.dirname(
                    os.path.dirname(path))),
                "mmproj": projectors[0] if projectors else None,
                **meta,
            })
    return found


def find_model(ref, models=None):
    """Resolve an id, file name, path or served alias to a scanned model."""
    if not ref:
        return None
    ref_l = ref.lower()
    st = load_settings()
    for m in models or scan_models():
        alias = (st.get(_key(m["path"]), {}).get("served_name") or "").lower()
        if ref_l in (m["id"], m["name"].lower(), m["path"].lower(), alias) or \
                ref_l == m["name"].lower()[:-5]:
            return m
    return None


_GPU_CACHE = {"t": 0.0, "gpus": []}


def nvidia_gpus():
    """NVIDIA GPUs via nvidia-smi (cached 10 s); [] elsewhere."""
    if time.time() - _GPU_CACHE["t"] > 10:
        try:
            import cuda_info
            _GPU_CACHE["gpus"] = cuda_info.query_gpus()
        except Exception:
            _GPU_CACHE["gpus"] = []
        _GPU_CACHE["t"] = time.time()
    return _GPU_CACHE["gpus"]


def fit_estimate(size_bytes):
    """Will this load, on this machine? Reuses the shared host checks."""
    if Host is None:
        return {"ok": None, "note": "hostcheck unavailable"}
    gpus = nvidia_gpus()
    if gpus:
        # Offloaded weights live in VRAM, summed over every GPU llama-server can split across.
        avail = sum(g["memory_free"] or g["memory_total"] for g in gpus)
        where = f"VRAM on {len(gpus)} GPUs" if len(gpus) > 1 else "VRAM"
        need = int(size_bytes * 1.15)
        ratio = need / avail if avail else 9
        if ratio > 1.0:
            return {"ok": False, "note": f"needs ~{need / 2**30:.1f} GiB, {avail / 2**30:.1f} GiB free {where} "
                                         "(lower GPU layers to keep some on the CPU)"}
        return {"ok": True, "note": f"{'tight: ' if ratio > 0.85 else ''}~{need / 2**30:.1f} of "
                                    f"{avail / 2**30:.1f} GiB {where}"}
    h = Host()
    avail = h.ram_available or h.ram
    if not avail:
        return {"ok": None, "note": "cannot read memory"}
    # Weights plus a modest KV allowance. Deliberately rough: the real number
    # depends on context length, which the user is about to choose.
    need = int(size_bytes * 1.15)
    ratio = need / avail
    if ratio > 1.0:
        return {"ok": False, "note": f"needs ~{need / 2**30:.1f} GiB, "
                                     f"{avail / 2**30:.1f} GiB available"}
    if ratio > 0.85:
        return {"ok": True, "note": f"tight: ~{need / 2**30:.1f} of "
                                    f"{avail / 2**30:.1f} GiB"}
    return {"ok": True, "note": f"~{need / 2**30:.1f} of "
                                f"{avail / 2**30:.1f} GiB"}


# ------------------------------------------------------------- hub search

def _hf_get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "neuronscope"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def hub_search(query, limit=20):
    """GGUF repos matching a query. Read-only, no dependency on huggingface_hub."""
    q = urllib.parse.urlencode({"search": query, "filter": "gguf",
                                "limit": limit, "sort": "downloads",
                                "direction": -1})
    out = []
    for m in _hf_get(f"{HF_API}/models?{q}"):
        out.append({"repo": m.get("modelId") or m.get("id"),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "gated": bool(m.get("gated"))})
    return out


def hub_files(repo):
    """The .gguf files in a repo, with sizes where the API reports them.

    Multi-part files are grouped: downloading one part of a split model is
    useless, so the UI offers the set rather than the pieces.
    """
    info = _hf_get(f"{HF_API}/models/{urllib.parse.quote(repo)}?blobs=true")
    files = []
    for sib in info.get("siblings", []):
        name = sib.get("rfilename", "")
        if not name.endswith(".gguf"):
            continue
        files.append({"path": name, "size": sib.get("size"),
                      "quant": (re.search(r"(Q\d+_[A-Z0-9_]+|IQ\d+_[A-Z]+|"
                                          r"BF16|F16|F32)", name, re.I).group(1).upper()
                                if re.search(r"(Q\d+_[A-Z0-9_]+|IQ\d+_[A-Z]+|"
                                             r"BF16|F16|F32)", name, re.I) else None)})
    groups = {}
    for f in files:
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", f["path"])
        key = f["path"][:m.start()] if m else f["path"]
        g = groups.setdefault(key, {"name": key, "parts": [], "size": 0,
                                    "quant": f["quant"]})
        g["parts"].append(f["path"])
        if f["size"]:
            g["size"] += f["size"]
    for g in groups.values():
        g["parts"].sort()
    return sorted(groups.values(), key=lambda g: g["name"])


def _download_one(repo, path, dest, job):
    url = HF_RESOLVE.format(repo=urllib.parse.quote(repo),
                            path=urllib.parse.quote(path))
    # The directory has to exist before the .part file is opened, not just
    # before the final rename.
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    # Resume: the Hub honours Range, and these files are large enough that
    # restarting from zero after a dropped connection is not acceptable.
    start = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "neuronscope"})
    if start:
        req.add_header("Range", f"bytes={start}-")
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0) + start
        job["total"] = max(job.get("total", 0), total)
        mode = "ab" if start and r.status == 206 else "wb"
        if mode == "wb":
            start = 0
        with open(tmp, mode) as f:
            done = start
            while not job.get("cancel"):
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                job["done"] = job.get("base", 0) + done
    if job.get("cancel"):
        return False
    os.replace(tmp, dest)
    return True


def start_download(repo, group, root):
    job_id = hashlib.sha256(f"{repo}/{group['name']}".encode()).hexdigest()[:12]
    with JOBS_LOCK:
        if job_id in JOBS and JOBS[job_id]["state"] == "running":
            return job_id
        JOBS[job_id] = {"id": job_id, "repo": repo, "name": group["name"],
                        "state": "running", "done": 0,
                        "total": group.get("size") or 0, "error": None,
                        "base": 0, "cancel": False}
    job = JOBS[job_id]
    target_dir = os.path.join(root, *repo.split("/"))

    def run():
        try:
            for part in group["parts"]:
                dest = os.path.join(target_dir, os.path.basename(part))
                if os.path.exists(dest):
                    job["base"] = job.get("base", 0) + os.path.getsize(dest)
                    job["done"] = job["base"]
                    continue
                if not _download_one(repo, part, dest, job):
                    job["state"] = "cancelled"
                    return
                job["base"] = job["done"]
            job["state"] = "done"
            _META_CACHE.clear()
        except Exception as e:
            job["state"] = "error"
            job["error"] = str(e)[:200]

    threading.Thread(target=run, daemon=True).start()
    return job_id


# ------------------------------------------------------------------ settings

def _key(path):
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def load_settings():
    p = STATE["settings_path"]
    if p and os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_settings(all_settings):
    p = STATE["settings_path"]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(all_settings, f, indent=2)
    os.replace(tmp, p)


DEFAULTS = {"ngl": 99, "ctx": 8192, "batch": 2048, "threads": 0,
            "parallel": 1, "flash_attn": True, "lora": "", "lora_scale": 1.0,
            "experts": 0, "extra": "",
            # Speculative decoding. A small draft model proposes tokens that the
            # large one verifies in a batch, so on a bandwidth-bound host the
            # win can be large -- this is the biggest speed lever available on
            # an iGPU, where generation is limited by weight reads per token.
            "draft_model": "", "draft_max": 16, "draft_min": 4,
            # Load the model's mmproj projector when one sits next to it.
            "vision": True,
            # classifier.npz for this model: enables per-reply activation stats
            "classifier": "",
            # Several GPUs (a Tesla M10 is four): which ones this model may use
            # (CUDA_VISIBLE_DEVICES), how llama-server splits it (layer | row |
            # none), the per-GPU proportions ("1,1,1,1") and the main GPU.
            "gpus": "", "split_mode": "", "tensor_split": "", "main_gpu": -1}

# A named config is a complete, reusable setup: model, load settings, preset,
# visualizer and hardware limits, under a name. The name is also what the model
# is served as, so Cline sees "model-suppressed" rather than a filename.
CONFIG_DEFAULTS = {
    "name": "", "path": "", "settings": {}, "preset": "default",
    "viz": "pygfx", "viz_device": "auto",
    "cache_type": "f16", "cache_type_v": "",
    "turboquant": False,          # no implementation exists; see warnings
    "cpu_cores": 0, "gpu_reserve_gib": 0.75, "vram_gib": 0.0,
    # Which machine this config is meant for. Empty means "wherever it runs",
    # audited against the current host.
    "host": "",
}


def load_configs():
    return load_settings().get("__configs__", {})


def save_config(cfg):
    st = load_settings()
    st.setdefault("__configs__", {})[cfg["name"]] = {
        **CONFIG_DEFAULTS, **{k: v for k, v in cfg.items()
                              if k in CONFIG_DEFAULTS}}
    save_settings(st)


def delete_config(name):
    st = load_settings()
    st.get("__configs__", {}).pop(name, None)
    save_settings(st)


def host_profiles():
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
        import hostprofiles as HP
        cur, _ = HP.current_profile()
        return HP.load_all(), cur
    except Exception:
        return {}, None


def audit_config(cfg, model):
    """-> [{level, field, message}]. Everything the user should see in orange.

    Warnings are attached to the field that caused them, so the UI can put the
    marker where the setting is rather than in a list at the bottom.
    """
    out = []
    s = {**DEFAULTS, **(cfg.get("settings") or {})}

    def warn(field, msg, level="warn"):
        out.append({"level": level, "field": field, "message": msg})

    if cfg.get("turboquant"):
        warn("turboquant",
             "TurboQuant has no implementation in llama.cpp. It is a research "
             "method for KV-cache compression, not a switch. Use cache_type "
             "q8_0 for the same lever at lower sophistication.", "error")

    if cfg.get("cache_type") in ("q4_0", "q4_1") or \
            cfg.get("cache_type_v") in ("q4_0", "q4_1"):
        warn("cache_type", "4-bit KV cache measurably degrades long-context "
                           "recall. q8_0 is the usual safe choice.")

    if s.get("ctx", 0) > 32768:
        warn("ctx", f"context {s['ctx']} makes the KV cache the dominant "
                    "memory cost. Check the budget below before loading.")

    if cfg.get("cpu_cores") and cfg["cpu_cores"] > (os.cpu_count() or 1):
        warn("cpu_cores", f"{cfg['cpu_cores']} threads on "
                          f"{os.cpu_count()} cores will thrash.")

    if s.get("parallel", 1) > 1 and s.get("ctx", 0) > 16384:
        warn("parallel", "concurrent slots each get ctx/parallel tokens; at "
                         "this context that may truncate requests.")

    if cfg.get("viz_device") == "gpu" and cfg.get("viz") in ("godot", "threejs"):
        warn("viz_device", "forcing the visualizer onto the GPU competes with "
                           "the model for VRAM. 'auto' evicts it to CPU when "
                           "memory is short.")

    if not model:
        return out

    # The real numbers, from the model's own architecture.
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
        from vram_budget import GIB, detect_vram, plan, read_gguf_shape
        shape = read_gguf_shape(model["path"])
        # A config can name the machine it is for, so you can plan for another
        # host from this one. Falls back to whatever this machine has.
        reserve_mult = 1.0
        vram = int((cfg.get("vram_gib") or 0) * GIB)
        hosts, current = host_profiles()
        target = cfg.get("host") or current
        if not vram and target and target in hosts:
            h = hosts[target]
            vram = int((h.get("vram_gib") or 0) * GIB)
            if not h.get("dedicated"):
                reserve_mult = 1.6   # the desktop draws from the same pool
            if target != current:
                warn("host", f"planned for host '{target}', not the machine "
                             "you are on.", "info")
        vram = vram or detect_vram()
        if vram:
            r = plan(shape, s.get("ctx", 8192), vram,
                     cfg.get("viz", "pygfx"),
                     cfg.get("cache_type", "f16"),
                     cfg.get("cache_type_v") or None,
                     None,
                     (cfg.get("gpu_reserve_gib") or 0.75) * GIB * reserve_mult)
            for w in r["warnings"]:
                warn("ctx", w)
            if r["viz_device"] == "cpu" and cfg.get("viz_device") != "cpu":
                warn("viz", "not enough VRAM for model, cache and visualizer; "
                            "the visualizer will run on CPU.")
            if r["layers_on_gpu"] < r["n_layers"]:
                warn("ngl", f"only {r['layers_on_gpu']} of {r['n_layers']} "
                            f"layers fit; the rest run on CPU and generation "
                            f"will be much slower.")
            out.append({"level": "info", "field": "budget", "message":
                        f"weights {r['model_bytes'] / GIB:.1f} + KV "
                        f"{r['kv_bytes'] / GIB:.1f} + viz "
                        f"{r['viz_bytes'] / GIB:.1f} GiB, "
                        f"{r['gpu_used'] / GIB:.1f} of "
                        f"{r['budget_bytes'] / GIB:.1f} GiB usable",
                        "plan": {k: r[k] for k in
                                 ("layers_on_gpu", "n_layers", "viz_device",
                                  "fits")}})
    except Exception as e:
        warn("budget", f"could not compute the budget: {e}", "info")
    return out


# Presets are chat-time, load settings are load-time. Keeping them apart means
# switching a system prompt does not restart the server.
PRESET_DEFAULTS = {"system": "", "temperature": 0.7, "top_p": 0.95,
                   "max_tokens": 2048,
                   # Structured output: either a JSON schema or a GBNF grammar.
                   # Both are constrained decoding in llama.cpp, so they are
                   # guarantees about the output shape, not requests.
                   "json_schema": "", "grammar": ""}
BUILTIN_PRESETS = {
    "default": {},
    "deterministic": {"temperature": 0.0, "top_p": 1.0},
    "reasoning": {"temperature": 0.6, "top_p": 0.95},
    "terse": {"system": "Answer concisely. No preamble.", "temperature": 0.3},
    "json object": {"temperature": 0.2,
                    "json_schema": '{"type":"object"}'},
}


def load_presets():
    st = load_settings()
    user = st.get("__presets__", {})
    out = {k: {**PRESET_DEFAULTS, **v} for k, v in BUILTIN_PRESETS.items()}
    out.update({k: {**PRESET_DEFAULTS, **v} for k, v in user.items()})
    return out


def save_preset(name, cfg):
    st = load_settings()
    st.setdefault("__presets__", {})[name] = {
        k: v for k, v in cfg.items() if k in PRESET_DEFAULTS}
    save_settings(st)


# -------------------------------------------------------------------- server

def server_running():
    p = PROC["proc"]
    return p is not None and p.poll() is None


def port_busy(port) -> bool:
    import socket
    with socket.socket() as sk:
        sk.settimeout(0.5)
        return sk.connect_ex(("127.0.0.1", port)) == 0


def wait_healthy(port, proc, timeout=600):
    url = f"http://127.0.0.1:{port}/health"
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False, f"exited with code {proc.returncode}"
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True, "ready"
        except Exception:
            pass
        time.sleep(1)
    return False, f"not healthy after {timeout}s"


def start_server(model, settings, port):
    if not STATE["server_bin"]:
        return False, "no --server binary configured"
    stop_server()
    if port_busy(port):
        # Something else (often an orphaned llama-server) holds the port and would
        # answer our health check while our own server fails to bind.
        return False, f"port {port} is already in use; stop whatever holds it or pass --backend-port"
    cmd = [STATE["server_bin"], "-m", model["path"],
           "-ngl", str(settings["ngl"]), "-c", str(settings["ctx"]),
           "-b", str(settings["batch"]),
           "--port", str(port), "--host", "127.0.0.1"]
    if settings.get("threads"):
        cmd += ["-t", str(settings["threads"])]
    if settings.get("parallel", 1) > 1:
        cmd += ["-np", str(settings["parallel"])]
    if settings.get("flash_attn"):
        # llama.cpp now requires a value: -fa on|off|auto
        cmd += ["-fa", "on"]
    if settings.get("cache_type"):
        cmd += ["--cache-type-k", settings["cache_type"]]
        cmd += ["--cache-type-v", settings.get("cache_type_v")
                or settings["cache_type"]]
    # What Cline and every other client sees in the model list.
    cmd += ["--alias", settings.get("served_name") or model.get("id") or model["name"]]
    if model.get("mmproj") and settings.get("vision", True):
        cmd += ["--mmproj", model["mmproj"]]
    if settings.get("cpu_cores"):
        cmd += ["-t", str(settings["cpu_cores"])]
    draft = settings.get("draft_model")
    if draft:
        if not os.path.exists(draft):
            return False, f"draft model not found: {draft}"
        cmd += ["-md", draft,
                "--draft-max", str(settings.get("draft_max", 16)),
                "--draft-min", str(settings.get("draft_min", 4))]
    if settings.get("lora"):
        cmd += ["--lora-scaled", settings["lora"],
                str(settings.get("lora_scale", 1.0))]
    # MoE only, and only when the model declares experts. The override key is
    # architecture-prefixed, so it is read from the file rather than assumed.
    if settings.get("experts") and model.get("n_experts"):
        arch = model.get("arch")
        if arch:
            cmd += ["--override-kv",
                    f"{arch}.expert_used_count=int:{settings['experts']}"]
    env = None
    gpus = str(settings.get("gpus") or "").replace(" ", "")
    if gpus:
        if not re.fullmatch(r"\d+(,\d+)*", gpus):
            return False, "visible GPUs must look like 0,1,2"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus}
    if settings.get("split_mode"):
        if settings["split_mode"] not in ("layer", "row", "none"):
            return False, "split mode must be layer, row or none"
        cmd += ["-sm", settings["split_mode"]]
    ts = str(settings.get("tensor_split") or "").replace(" ", "")
    if ts:
        if not re.fullmatch(r"\d+(\.\d+)?(,\d+(\.\d+)?)*", ts):
            return False, "tensor split must look like 1,1,1,1 or 3,1"
        cmd += ["-ts", ts]
    mg = settings.get("main_gpu")
    if mg not in (None, "") and int(mg) >= 0:
        cmd += ["-mg", str(int(mg))]
    if settings.get("extra"):
        cmd += settings["extra"].split()

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, env=env)
    ok, why = wait_healthy(port, proc)
    if not ok:
        err = ""
        try:
            proc.terminate()
            err = (proc.stderr.read() or b"").decode(errors="replace")[-800:]
        except Exception:
            pass
        return False, f"{why}\n{err}"
    PROC.update({"proc": proc, "model": model, "port": port,
                 "args": cmd, "started": time.time(),
                 "lora": settings.get("lora") or None,
                 "lora_scale": settings.get("lora_scale", 1.0)})
    return True, "ready"


def stop_server():
    p = PROC["proc"]
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
        time.sleep(1)
    PROC.update({"proc": None, "model": None, "port": None, "args": None,
                 "started": None, "lora": None})


# ---------------------------------------------------------------- linked hosts

# What a paired device may POST. Everything else needs the owner's token.
DEVICE_POSTS = {"/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/api/chat", "/api/chats",
                "/api/chats/delete", "/api/rag/search", "/api/tools/approve", "/api/trace"}

LINKS = {"cache": {}, "lock": threading.Lock()}
LINK_SEP = ":"


def load_links() -> list[dict]:
    try:
        return json.loads(open(STATE["links_path"]).read()).get("links", [])
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return []


def save_links(links: list[dict]) -> None:
    os.makedirs(os.path.dirname(STATE["links_path"]), exist_ok=True)
    sec.write_secret_file(Path(STATE["links_path"]), json.dumps({"links": links}, indent=1))


def link_models(refresh: bool = False) -> list[dict]:
    """Models of every linked host, as '<link>:<remote id>' (cached 30 s)."""
    out = []
    for ln in load_links():
        with LINKS["lock"]:
            hit = LINKS["cache"].get(ln["name"])
        if hit and not refresh and time.time() - hit[0] < 30:
            out += hit[1]
            continue
        if ln.get("expires") and ln["expires"] <= time.time():
            with LINKS["lock"]:
                LINKS["cache"][ln["name"]] = (time.time(), [], "access expired " + time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(ln["expires"])) + "; ask that host for a new pairing link")
            continue
        try:
            r = ns_pairing.request(ln["url"], "GET", "/v1/models", token=ln["token"],
                                   fingerprint=ln.get("fingerprint", ""), timeout=5)
            ms = [{"id": f"{ln['name']}{LINK_SEP}{m['id']}", "object": "model", "owned_by": f"link:{ln['name']}",
                   "remote_id": m["id"], "link": ln["name"], "vision": m.get("vision", False)}
                  for m in r.get("data", []) if m.get("id") != "auto"]
            err = None
        except Exception as e:
            ms, err = [], f"{type(e).__name__}: {e}"[:200]
        with LINKS["lock"]:
            LINKS["cache"][ln["name"]] = (time.time(), ms, err)
        out += ms
    return out


def link_target(model_id: str | None):
    """-> (upstream dict, remote model id) when model_id names a linked host's model."""
    if not model_id or LINK_SEP not in model_id:
        return None
    name, _, rid = model_id.partition(LINK_SEP)
    ln = next((x for x in load_links() if x["name"] == name), None)
    if ln is None:
        return None
    return {"base": ln["url"], "token": ln["token"], "fingerprint": ln.get("fingerprint", "")}, rid


def _upstream_stream(upstream: dict, path: str, payload: dict):
    r = ns_pairing.request(upstream["base"], "POST", path, payload, token=upstream["token"],
                           fingerprint=upstream["fingerprint"], timeout=900, stream=True)
    if r.status >= 400:
        raise RuntimeError(f"linked host: HTTP {r.status}: {r.read()[:300].decode(errors='replace')}")
    return r


# ---------------------------------------------------------------- MCP tools

APPROVALS: dict = {}            # call key -> {"event": Event, "allow": bool}
APPROVALS_LOCK = threading.Lock()
MAX_TOOL_ROUNDS = 8


def mcp_hub():
    hub = STATE.get("mcp")
    if hub is None and STATE.get("mcp_config"):
        hub = STATE["mcp"] = ns_mcp.MCPHub(STATE["mcp_config"])
        hub.connect()
    return hub


def wait_approval(key: str, timeout: float = 300) -> bool:
    ev = threading.Event()
    with APPROVALS_LOCK:
        APPROVALS[key] = {"event": ev, "allow": False}
    try:
        ev.wait(timeout)
        with APPROVALS_LOCK:
            return APPROVALS[key]["allow"]
    finally:
        with APPROVALS_LOCK:
            APPROVALS.pop(key, None)


def sse(wfile, obj) -> None:
    wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
    wfile.flush()


def chat_with_tools(wfile, payload: dict, msgs: list, hub, upstream=None) -> dict:
    """Stream a chat in which the model may call MCP tools. Each call is shown
    to the user and, unless auto-approved, waits for Allow/Deny."""
    payload = {**payload, "tools": hub.openai_tools()}
    out = {"text": ""}
    for _ in range(MAX_TOOL_ROUNDS):
        payload["messages"] = msgs
        out = proxy("/v1/chat/completions", payload, stream_to=wfile, hold_done=True, upstream=upstream)
        calls = out.get("tool_calls") or []
        if not calls:
            break
        msgs = msgs + [{"role": "assistant", "content": out["text"] or None, "tool_calls": [
            {"id": c["id"] or f"call_{i}", "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}} for i, c in enumerate(calls)]}]
        for i, c in enumerate(calls):
            cid = c["id"] or f"call_{i}"
            key = os.urandom(8).hex()
            try:
                args = json.loads(c["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except ValueError as e:
                args, err = None, f"invalid arguments: {e}"
            else:
                err = None
            need = err is None and not hub.auto_approved(c["name"])
            sse(wfile, {"tool_call": {"key": key, "name": c["name"], "arguments": args if args is not None
                                      else c["arguments"], "needs_approval": need}})
            if err is None and need and not wait_approval(key):
                err = "the user declined this tool call"
            if err is None:
                try:
                    ok, text = hub.call(c["name"], args)
                except Exception as e:
                    ok, text = False, f"{type(e).__name__}: {e}"
            else:
                ok, text = False, err
            sse(wfile, {"tool_result": {"key": key, "ok": ok, "text": text[:4000]}})
            msgs = msgs + [{"role": "tool", "tool_call_id": cid, "content": text if ok else f"Error: {text}"}]
    wfile.write(b"data: [DONE]\n\n")
    wfile.flush()
    return out


# ---------------------------------------------------------------- RAG

EMBED = {"proc": None, "lock": threading.Lock()}


def rag_embedder():
    """The embedder for dense retrieval: --rag-embed URL@model, or a llama-server
    --embedding sidecar for --rag-embed-gguf, started on first use."""
    if STATE.get("rag_embed"):
        return ns_rag.Embedder(STATE["rag_embed"])
    gguf = STATE.get("rag_embed_gguf")
    if not gguf:
        return None
    port = STATE["backend_port"] + 1
    with EMBED["lock"]:
        p = EMBED["proc"]
        if p is None or p.poll() is not None:
            if not STATE["server_bin"]:
                raise RuntimeError("--rag-embed-gguf needs --server")
            if port_busy(port):
                raise RuntimeError(f"port {port} (embedding sidecar) is already in use")
            cmd = [STATE["server_bin"], "-m", gguf, "--embedding", "--pooling", "mean", "-ngl",
                   str(STATE.get("rag_embed_ngl", 0)), "-c", "2048", "-b", "2048", "-ub", "2048",
                   "--port", str(port), "--host", "127.0.0.1", "--alias", "rag-embed"]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            ok, why = wait_healthy(port, proc)
            if not ok:
                proc.terminate()
                raise RuntimeError(f"embedding server did not start: {why}")
            EMBED["proc"] = proc
    return ns_rag.Embedder(f"http://127.0.0.1:{port}/v1@rag-embed")


def rag_store():
    st = STATE.get("rag")
    if st is None:
        st = STATE["rag"] = ns_rag.RagStore(STATE.get("rag_dir"))
    return st


def rag_context(req: dict, msgs: list) -> tuple[list, list]:
    """Retrieve for the last user message; returns (messages with context, sources)."""
    spec = req.get("rag") or {}
    name = spec.get("collection")
    if not name:
        return msgs, []
    last = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
    query = last["content"] if last and isinstance(last.get("content"), str) else \
        " ".join(c.get("text", "") for c in (last or {}).get("content") or [] if isinstance(c, dict))
    store = rag_store()
    coll = store.get(name)
    emb = rag_embedder() if coll.dense is not None else None
    hits = coll.search(query, max(1, min(int(spec.get("k", 4)), 12)), emb)
    if not hits:
        return msgs, []
    ctx = {"role": "system", "content": ns_rag.context_message(hits)}
    i = 1 if msgs and msgs[0].get("role") == "system" else 0
    return msgs[:i] + [ctx] + msgs[i:], hits


def proxy(path, payload, stream_to=None, hold_done=False, upstream=None):
    """Forward to the running llama-server. Streams SSE when asked.

    Streaming matters here: on an iGPU at a few tokens per second, a
    non-streaming chat window looks indistinguishable from a hang.
    With hold_done the final [DONE] is not forwarded (a tool round follows),
    and streamed tool calls are assembled and returned.
    """
    if upstream is not None:
        # a linked host's model: same OpenAI API, its device token, its pinned certificate
        if stream_to is None:
            return ns_pairing.request(upstream["base"], "POST", path, payload, token=upstream["token"],
                                      fingerprint=upstream["fingerprint"], timeout=900)
        opener = lambda: _upstream_stream(upstream, path, payload)  # noqa: E731
    else:
        if not server_running():
            raise RuntimeError("no model loaded")
        url = f"http://127.0.0.1:{PROC['port']}{path}"
        req = urllib.request.Request(
            url, method="POST", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        if stream_to is None:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())
        opener = lambda: urllib.request.urlopen(req, timeout=900)  # noqa: E731
    # Stream through untouched, keeping a copy of the text for stats.
    text, calls, finish = [], {}, None
    r = opener()
    with r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if hold_done and line.startswith("data:") and line[5:].strip() == "[DONE]":
                continue
            stream_to.write(raw)
            stream_to.flush()
            if line.startswith("data:") and "[DONE]" not in line:
                try:
                    ch = json.loads(line[5:])["choices"][0]
                    delta = ch.get("delta") or {}
                    text.append(delta.get("content") or "")
                    finish = ch.get("finish_reason") or finish
                    for tc in delta.get("tool_calls") or []:
                        c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        c["id"] = tc.get("id") or c["id"]
                        fn = tc.get("function") or {}
                        c["name"] += fn.get("name") or ""
                        c["arguments"] += fn.get("arguments") or ""
                except (ValueError, KeyError, IndexError, TypeError):
                    pass
    return {"text": "".join(text), "tool_calls": [calls[k] for k in sorted(calls)], "finish_reason": finish}


# ------------------------------------------------------- OpenAI-compatible API

def touch(delta=0):
    with ACTIVITY_LOCK:
        ACTIVITY["last"] = time.time()
        ACTIVITY["active"] += delta


def idle_reaper():
    """Unload the model after --idle-ttl seconds without requests (LM Studio's TTL)."""
    while True:
        time.sleep(5)
        ttl = STATE["idle_ttl"]
        if not ttl or not server_running():
            continue
        with ACTIVITY_LOCK:
            idle = ACTIVITY["active"] == 0 and time.time() - ACTIVITY["last"] > ttl
        if idle:
            with LOCK:
                if server_running():
                    print(f"idle for {ttl}s: unloading {PROC['model']['name']}")
                    stop_server()


def _last_user_text(body):
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            return str(c or "")
    return str(body.get("prompt") or "")


def _has_image(body):
    for m in body.get("messages") or []:
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(x, dict) and x.get("type") == "image_url" for x in c):
            return True
    return False


def stats_ids(m, st=None):
    """Every name a model's stats may have been recorded under."""
    st = st if st is not None else load_settings()
    alias = st.get(_key(m["path"]), {}).get("served_name")
    return [x for x in (m["id"], alias, m["name"], m["name"][:-5]) if x]


def model_summary(m, st=None):
    s = STATE["stats"].summary(stats_ids(m, st), size=m["size"])
    n = s["graded"].get("n", 0)
    s["eligible"] = n >= STATE["min_graded"]
    s["why_not"] = None if s["eligible"] else (
        "No performance stats for this model file yet." if n == 0 else
        f"Only {n} graded results; auto routing needs {STATE['min_graded']}.")
    return s


def auto_pick(body, models):
    """Stats-based routing. Only models with enough graded results compete."""
    from subject_classifier import default_classifier
    text = _last_user_text(body)
    pool = models
    if _has_image(body):
        text += " image photo picture"
        pool = [m for m in models if m.get("mmproj")]
    proba = default_classifier().predict_proba(text)
    st = load_settings()
    by_id = {m["id"]: m for m in pool}
    sums = {m["id"]: model_summary(m, st) for m in pool}
    pick = model_stats.rank(proba, sums, STATE["min_graded"], STATE["min_subject"], STATE["halluc_cost"])
    pick["proba"] = dict(list(proba.items())[:3])
    return by_id.get(pick["model"]), pick


def resolve_request_model(body):
    """Which scanned model should serve this request? -> (model or None, route info).

    "auto" ranks only models with performance stats; any other model name is a
    manual choice and is honoured whether or not stats exist."""
    ref = (body.get("model") or "").strip()
    current = PROC["model"]
    models = scan_models()
    if ref.lower() == "auto":
        target, pick = auto_pick(body, models)
        if target is None:
            excluded = "; ".join(f"{k}: {v}" for k, v in list(pick.get("excluded", {}).items())[:6])
            return None, {"mode": "auto", "error": (
                "auto: no model has enough performance stats to choose from "
                f"(need {STATE['min_graded']} graded results). Run scripts/testqa.py "
                f"--publish-stats against your models, or name a model explicitly. {excluded}")}
        return target, {"mode": "auto", "model": target["id"], "reason": pick["reason"],
                        "candidates": pick["candidates"][:5]}
    if not ref or (current and find_model(ref, [current])):
        return current, {"mode": "loaded", "model": current["id"] if current else None}
    target = find_model(ref, models)
    return target, {"mode": "manual", "model": target["id"] if target else ref}


# ---------------------------------------------------------- reply statistics

SCORE_JOBS = queue.Queue(maxsize=4)
_SCORERS = {}
_REPLIES = {"n": 0}


def note_reply(model, messages, text):
    """Record an ungraded live reply, and queue an activation score if this
    model has a classifier configured. Never blocks the response."""
    if not model or STATE["stats"] is None:
        return
    try:
        from subject_classifier import default_classifier
        subject = default_classifier().predict(_last_user_text({"messages": messages}))
    except Exception:
        subject = None
    STATE["stats"].record(model["id"], "live", size=model["size"], subject=subject,
                          abstained=model_stats.looks_abstained(text), chars=len(text or ""))
    clf = load_settings().get(_key(model["path"]), {}).get("classifier")
    _REPLIES["n"] += 1
    if clf and STATE["cett"] and text and _REPLIES["n"] % max(1, STATE["score_every"]) == 0:
        try:
            SCORE_JOBS.put_nowait((model, clf, messages, text, subject))
        except queue.Full:
            pass        # never queue behind scoring; the next reply matters more


def score_worker():
    while True:
        model, clf, messages, text, subject = SCORE_JOBS.get()
        # Activation scoring is an extra prefill; wait for a quiet moment.
        while ACTIVITY["active"] > 0:
            time.sleep(1)
        try:
            res = _scorer(model, clf).score([m for m in messages if m.get("role") != "system"] or messages, text)
            STATE["stats"].record(model["id"], "activation", size=model["size"], subject=subject,
                                  h_score=res["score"], prob=res["prob"], n_tokens=res["n_tokens"],
                                  threshold=0.0)
        except Exception as e:
            print(f"[studio] activation scoring failed for {model['name']}: {e}", file=sys.stderr)


# ------------------------------------------------------- per-reply checks

TRACES = {"lock": threading.Lock(), "payloads": {}}
TRACE_ID = re.compile(r"[A-Za-z0-9_-]{6,64}")


def _scorer(model, clf):
    key = (model["path"], clf)
    if key not in _SCORERS:
        from hscore import HScorer
        _SCORERS[key] = HScorer(STATE["cett"], model["path"], clf, ngl=STATE["score_ngl"])
    return _SCORERS[key]


def trace_reply(model, messages, text, threshold=0.5):
    """Score one reply token by token and save it as a trace session that the
    3D view (viz/bloom.py) and timeline.py can open. -> summary for the chat."""
    clf = load_settings().get(_key(model["path"]), {}).get("classifier")
    if not clf:
        raise ValueError(f"{model['id']} has no classifier set (model settings: classifier)")
    if not STATE["cett"]:
        raise ValueError("activation checks need llama-cett-dump (studio --cett)")
    msgs = [m for m in messages if m.get("role") != "system"] or messages
    r = _scorer(model, clf).trace(msgs, text)
    frames = r["frames"]
    T, L, N = frames.shape
    tid = time.strftime("%Y%m%d-%H%M%S-") + hashlib.sha1(text.encode()).hexdigest()[:8]
    from records import Recorder
    with Recorder(os.path.join(STATE["traces_dir"], tid),
                  {"model": model["name"], "n_layers": L, "n_neurons": N, "kind": "trace",
                   "stride": r["stride"], "source": "studio"}, resume=False) as rec:
        rec.add("reply", agg=frames.reshape(-1, N), tokens=r["tokens"], scores=r["scores"],
                kind="trace", n_frames=T, n_layers=L, pieces=r["pieces"], stride=1,
                question=(_last_user_text({"messages": msgs}) or "")[:500], verdict=None,
                h_cells=r["h_cells"], col_weight=r["col_weight"].tolist())
    prob = [round(float(p), 4) for p in r["prob"]]
    return {"id": tid, "pieces": r["pieces"], "prob": prob,
            "flagged": [i for i, p in enumerate(prob) if p >= threshold], "threshold": threshold,
            "max": max(prob), "mean": round(sum(prob) / len(prob), 4),
            "url": f"viz/{tid}/"}


def trace_payload(tid):
    """bloom payload for a saved trace, built once and cached (a few per process)."""
    with TRACES["lock"]:
        hit = TRACES["payloads"].get(tid)
    if hit:
        return hit
    path = os.path.join(STATE["traces_dir"], tid)
    if not TRACE_ID.fullmatch(tid) or not os.path.isdir(path):
        return None
    import bloom
    blob, meta = bloom.build_payload(path, None, 97.0, 40000)
    out = {"blob": blob, "meta": meta, "theme": bloom.THEMES.get(STATE.get("viz_theme") or "dark")
           or next(iter(bloom.THEMES.values()))}
    with TRACES["lock"]:
        if len(TRACES["payloads"]) >= 8:
            TRACES["payloads"].pop(next(iter(TRACES["payloads"])))
        TRACES["payloads"][tid] = out
    return out


def list_traces(limit=200):
    d = STATE.get("traces_dir")
    out = []
    if d and os.path.isdir(d):
        for name in sorted(os.listdir(d), reverse=True)[:limit]:
            try:
                meta = json.load(open(os.path.join(d, name, "manifest.json")))
            except (OSError, ValueError):
                continue
            out.append({"id": name, "model": meta.get("model"), "created": meta.get("created")})
    return out


def ensure_loaded(model, own=1):
    """JIT: load `model` with its saved settings unless it is already serving.
    `own` is how many in-flight requests belong to the caller."""
    if server_running() and PROC["model"] and PROC["model"]["path"] == model["path"]:
        return True, "loaded"
    if not STATE["jit"]:
        return False, f"{model['id']} is not loaded and JIT loading is disabled"
    # One llama-server at a time: let in-flight requests (other than this
    # one) finish before swapping the model out from under them.
    deadline = time.time() + 600
    while ACTIVITY["active"] > own and time.time() < deadline:
        time.sleep(0.5)
    settings = {**DEFAULTS, **load_settings().get(_key(model["path"]), {})}
    return start_server(model, settings, STATE["backend_port"])


def openai_models():
    st = load_settings()
    data = []
    any_eligible = False
    for m in scan_models():
        alias = st.get(_key(m["path"]), {}).get("served_name")
        sm = model_summary(m, st)
        g = sm["graded"]
        any_eligible |= sm["eligible"]
        data.append({"id": alias or m["id"], "object": "model", "owned_by": "local",
                     "stats": {"graded": g.get("n", 0), "accuracy": g.get("accuracy"),
                               "hallucination_rate": g.get("hallucination_rate"),
                               "abstention_rate": g.get("abstention_rate"),
                               "auto_eligible": sm["eligible"]},
                     "created": int(os.path.getmtime(m["path"])),
                     "loaded": bool(PROC["model"] and PROC["model"]["path"] == m["path"]),
                     "vision": bool(m.get("mmproj")), "arch": m.get("arch"),
                     "quant": m.get("quant"), "size": m["size"]})
    if any_eligible:
        data.insert(0, {"id": "auto", "object": "model", "owned_by": "neuronscope",
                        "description": "per prompt: subject classifier + rolling performance stats"})
    data += [{k: v for k, v in m.items() if k != "remote_id"} for m in link_models()]
    return {"object": "list", "data": data}


# ---------------------------------------------------------------- chat history

CHAT_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")


def _chat_path(cid):
    if not CHAT_ID_RE.match(cid or ""):
        raise ValueError("bad chat id")
    return os.path.join(STATE["chats_dir"], f"{cid}.json")


def list_chats():
    d = STATE["chats_dir"]
    out = []
    if d and os.path.isdir(d):
        for f in glob.glob(os.path.join(d, "*.json")):
            try:
                with open(f) as fh:
                    c = json.load(fh)
                out.append({k: c.get(k) for k in ("id", "title", "updated", "model")} |
                           {"n": len(c.get("messages", []))})
            except Exception:
                continue
    return sorted(out, key=lambda c: -(c.get("updated") or 0))


def save_chat(chat):
    cid = chat.get("id") or hashlib.sha256(os.urandom(16)).hexdigest()[:16]
    path = _chat_path(cid)
    os.makedirs(STATE["chats_dir"], exist_ok=True)
    rec = {"id": cid, "title": str(chat.get("title") or "New chat")[:120],
           "model": chat.get("model"), "preset": chat.get("preset"),
           "messages": chat.get("messages", []), "updated": time.time()}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return rec


# ---------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # -- auth ---------------------------------------------------------------
    # This endpoint loads models, downloads files and runs inference. On a LAN
    # that is not something to leave open, so --token gates everything except
    # the login page itself.
    _role = None      # "owner" (master token, or no token configured) or "device" (a paired device)

    def _presented(self) -> list[str]:
        out = []
        hdr = self.headers.get("Authorization", "")
        if hdr.startswith("Bearer "):
            out.append(hdr[7:])
        raw = self.headers.get("Cookie", "")
        if raw:
            try:
                c = http.cookies.SimpleCookie(raw)
                if "ns_token" in c:
                    out.append(c["ns_token"].value)
            except Exception:
                pass
        return out

    def _authed(self):
        self._role = None
        tok = STATE["token"]
        if not tok:
            self._role = "owner"
            return True
        presented = self._presented()
        if any(hmac.compare_digest(p, tok) for p in presented):
            self._role = "owner"
            return True
        reg = STATE.get("devices")
        if reg is not None and any(reg.check(p) for p in presented):
            self._role = "device"
            return True
        return False

    def _owner_only(self):
        """Pairing, device management and links are for the owner, not for paired devices."""
        if self._role != "owner":
            self._json(403, {"error": "only the owner (master token) can do this"})
            return False
        return True

    def _deny(self):
        if self.path == "/" or self.path.startswith("/login"):
            body = LOGIN.encode()
            self.send_response(401)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(401, {"error": "unauthorized"})

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        if self._route:
            self.send_header("X-NeuronScope-Route", json.dumps(self._route)[:4000])
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self, limit=MAX_SMALL_BODY):
        n = int(self.headers.get("Content-Length", 0))
        if n < 0 or n > limit:
            raise ValueError(f"request body too large (limit {limit} bytes)")
        return json.loads(self.rfile.read(n) or b"{}")

    _route = None

    def _send_headers_sse(self):
        self.send_response(200)
        if self._route:
            self.send_header("X-NeuronScope-Route", json.dumps(self._route)[:4000])
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _v1(self, path):
        """OpenAI-compatible passthrough with JIT model loading."""
        body = self._read(MAX_BODY)
        touch(+1)
        try:
            lt = link_target(body.get("model"))
            if lt is not None:
                upstream, rid = lt
                body = {**body, "model": rid}
                if body.get("stream"):
                    self._send_headers_sse()
                    proxy(path, body, stream_to=self.wfile, upstream=upstream)
                else:
                    self._json(200, proxy(path, body, upstream=upstream))
                return
            with LOCK:
                target, route = resolve_request_model(body)
                if target is None:
                    if route.get("error"):
                        return self._json(409, {"error": {"message": route["error"], "type": "invalid_request_error"}})
                    return self._json(404, {"error": {"message": f"model not found: {body.get('model')}",
                                                      "type": "invalid_request_error"}})
                ok, msg = ensure_loaded(target)
            if not ok:
                return self._json(503, {"error": {"message": f"could not load {target['id']}: {msg}",
                                                  "type": "server_error"}})
            if _has_image(body) and not target.get("mmproj"):
                return self._json(400, {"error": {"message": f"{target['id']} has no mmproj vision projector",
                                                  "type": "invalid_request_error"}})
            self._route = route
            if body.get("stream"):
                self._send_headers_sse()
                out = proxy(path, body, stream_to=self.wfile)
                text = out["text"]
            else:
                out = proxy(path, body)
                text = ((out.get("choices") or [{}])[0].get("message") or {}).get("content") \
                    if isinstance(out, dict) else None
                self._json(200, out)
            if path == "/v1/chat/completions":
                note_reply(target, body.get("messages") or [], text or "")
        finally:
            touch(-1)

    def _pairing_post(self):
        if not self._owner_only():
            return
        req = self._read()
        reg = STATE["devices"]
        if self.path == "/api/pair/start":
            if not STATE["token"]:
                return self._json(400, {"error": "pairing needs Studio to run with a token (and TLS off loopback); "
                                                 "without one, nothing is protected to pair into"})
            try:
                code, exp, access = reg.new_code(req.get("persistent"), req.get("ttl"))
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            host = self.headers.get("Host") or "127.0.0.1"
            scheme = "https" if STATE["tls"] else "http"
            # `a` only tells the claiming page what it will get; the host enforces it.
            frag = (f"c={code[:4]}-{code[4:8]}-{code[8:]}"
                    + (f"&fp={STATE['fingerprint']}" if STATE.get("fingerprint") else "")
                    + f"&a={'p' if access is None else ns_pairing.fmt_duration(access)}")
            return self._json(200, {"code": code, "expires": exp, "link": f"{scheme}://{host}/pair#{frag}",
                                    "fingerprint": STATE.get("fingerprint"), "persistent": access is None,
                                    "access_seconds": access})
        if self.path == "/api/devices/revoke":
            return self._json(200, {"ok": reg.revoke(str(req.get("id", "")))})
        if self.path == "/api/devices/update":
            try:
                return self._json(200, reg.update(str(req.get("id", "")), req.get("persistent"), req.get("ttl")))
            except KeyError:
                return self._json(404, {"error": "no such device"})
            except ValueError as e:
                return self._json(400, {"error": str(e)})
        if self.path == "/api/links/add":
            name = re.sub(r"[^A-Za-z0-9_.-]", "-", str(req.get("name") or "remote"))[:32].strip("-") or "remote"
            links = [x for x in load_links() if x["name"] != name]
            try:
                r = ns_pairing.claim(str(req.get("link", "")), str(req.get("device_name") or "studio"))
            except (ValueError, RuntimeError, OSError) as e:
                return self._json(400, {"error": f"could not pair: {e}"})
            links.append({"name": name, "url": r["url"], "token": r["token"], "fingerprint": r["fingerprint"],
                          "device_id": r["device_id"], "added": time.time(), "expires": r.get("expires")})
            save_links(links)
            with LINKS["lock"]:
                LINKS["cache"].pop(name, None)
            return self._json(200, {"name": name, "models": [m["id"] for m in link_models(refresh=True)
                                                             if m["link"] == name]})
        if self.path == "/api/links/remove":
            name = str(req.get("name", ""))
            save_links([x for x in load_links() if x["name"] != name])
            with LINKS["lock"]:
                LINKS["cache"].pop(name, None)
            return self._json(200, {"ok": True})

    def _jobs_off(self):
        return self._json(403, {"error": STATE.get("jobs_off") or "jobs are disabled"})

    def _jobs_get(self):
        runner = STATE.get("jobs")
        if self.path == "/api/jobs/specs":
            return self._json(200, ns_jobs.public_specs())
        if runner is None:
            return self._jobs_off()
        if self.path == "/api/jobs":
            return self._json(200, runner.list())
        m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})/log", self.path)
        if m:
            try:
                return self._json(200, {"log": runner.log(m.group(1))})
            except (ValueError, FileNotFoundError):
                return self._json(404, {"error": "no such job"})
        return self._json(404, {"error": "not found"})

    def _jobs_post(self):
        runner = STATE.get("jobs")
        if runner is None:
            return self._jobs_off()
        if "application/json" not in self.headers.get("Content-Type", ""):
            return self._json(415, {"error": "JSON only"})     # no cross-site form posts
        req = self._read()
        try:
            if self.path == "/api/jobs/cancel":
                return self._json(200, runner.cancel(str(req.get("id", ""))))
            return self._json(200, runner.start(str(req.get("kind", "")), dict(req.get("values") or {})))
        except (ValueError, FileNotFoundError) as e:
            return self._json(400, {"error": str(e)})
        except RuntimeError as e:
            return self._json(429, {"error": str(e)})

    def _rag_post(self):
        import base64
        store = rag_store()
        try:
            if self.path == "/api/rag/upload":
                req = self._read(48 * 1024 * 1024)
                data = base64.b64decode(str(req.get("data", "")).split(",", 1)[-1], validate=False)
                if len(data) > 32 * 1024 * 1024:
                    return self._json(413, {"error": "file larger than 32 MB"})
                coll = store.get(str(req.get("collection", "")), create=True)
                emb = rag_embedder() if (coll.dense is not None or (not coll.chunks and
                                                                    req.get("dense", True))) else None
                return self._json(200, coll.add(str(req.get("filename", "upload.txt")), data, emb))
            req = self._read()
            name = str(req.get("collection", ""))
            if self.path == "/api/rag/create":
                return self._json(200, store.get(name, create=True).info())
            if self.path == "/api/rag/delete":
                if req.get("doc"):
                    return self._json(200, {"removed": store.get(name).remove(str(req["doc"]))})
                store.drop(name)
                return self._json(200, {"ok": True})
            if self.path == "/api/rag/search":
                coll = store.get(name)
                emb = rag_embedder() if coll.dense is not None else None
                return self._json(200, coll.search(str(req.get("query", "")), int(req.get("k", 4)), emb))
        except FileNotFoundError as e:
            return self._json(404, {"error": str(e)})
        except (ValueError, RuntimeError, OSError) as e:
            return self._json(400, {"error": str(e)[:300]})
        return self._json(404, {"error": "not found"})

    def do_GET(self):
        self._route = None
        if self.path.split("?")[0] == "/pair":
            body = PAIR_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if not self._authed():
            return self._deny()
        if self.path == "/link":
            body = LINK_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/devices":
            if not self._owner_only():
                return
            return self._json(200, {"devices": STATE["devices"].list(),
                                    "policy": STATE["devices"].policy.describe(),
                                    "links": [{"name": x["name"], "url": x["url"],
                                               "pinned": bool(x.get("fingerprint")),
                                               "expires": x.get("expires"),
                                               "error": (LINKS["cache"].get(x["name"]) or (0, [], None))[2],
                                               "models": len((LINKS["cache"].get(x["name"]) or (0, []))[1])}
                                              for x in load_links()]})
        if self.path == "/api/links/models":
            return self._json(200, link_models())
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        m = re.fullmatch(r"/viz/([A-Za-z0-9_-]+)/(|api/meta|api/theme|api/trace)", self.path)
        if m:
            pl = trace_payload(m.group(1))
            if pl is None:
                return self._json(404, {"error": "no such trace"})
            import bloom
            kind = m.group(2)
            body, ctype = ((bloom.PAGE.encode(), "text/html; charset=utf-8") if kind == "" else
                           (pl["blob"], "application/octet-stream") if kind == "api/trace" else
                           (json.dumps(pl["meta" if kind == "api/meta" else "theme"]).encode(),
                            "application/json"))
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/traces":
            return self._json(200, {"traces": list_traces()})
        if self.path.startswith("/api/jobs") or self.path == "/jobs":
            if self._role != "owner":
                return self._json(403, {"error": "jobs need the owner"})
            if self.path == "/jobs":
                body = ns_jobs.PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return self._jobs_get()
        if self.path == "/api/gpus":
            import cuda_info
            return self._json(200, cuda_info.advise(nvidia_gpus()))
        if self.path == "/api/mcp":
            hub = mcp_hub()
            return self._json(200, {"servers": hub.status() if hub else [], "config": STATE.get("mcp_config"),
                                    "tools": len(hub.openai_tools()) if hub else 0})
        if self.path == "/api/rag":
            return self._json(200, {"collections": rag_store().list(),
                                    "embedder": bool(STATE.get("rag_embed") or STATE.get("rag_embed_gguf"))})
        m = re.fullmatch(r"/api/rag/docs\?c=(.+)", self.path)
        if m:
            try:
                return self._json(200, rag_store().get(urllib.parse.unquote(m.group(1))).docs())
            except (ValueError, FileNotFoundError) as e:
                return self._json(404, {"error": str(e)})
        if self.path == "/v1/models":
            return self._json(200, openai_models())
        if self.path == "/api/chats":
            return self._json(200, list_chats())
        if self.path.startswith("/api/chats/"):
            try:
                with open(_chat_path(self.path.rsplit("/", 1)[-1])) as f:
                    return self._json(200, json.load(f))
            except (ValueError, FileNotFoundError):
                return self._json(404, {"error": "no such chat"})
        if self.path == "/api/models":
            ms = scan_models()
            st = load_settings()
            for m in ms:
                m["settings"] = {**DEFAULTS, **st.get(_key(m["path"]), {})}
                m["fit"] = fit_estimate(m["size"])
                m["stats"] = model_summary(m, st)
            return self._json(200, ms)
        if self.path == "/api/stats":
            return self._json(200, {"min_graded": STATE["min_graded"], "min_subject": STATE["min_subject"],
                                    "hallucination_cost": STATE["halluc_cost"],
                                    "window": STATE["stats"].window, "scoring": bool(STATE["cett"]),
                                    "models": {m["id"]: model_summary(m) for m in scan_models()}})
        if self.path == "/api/hosts":
            hosts, current = host_profiles()
            return self._json(200, {"hosts": hosts, "current": current})
        if self.path == "/api/configs":
            models = {m["path"]: m for m in scan_models()}
            out = []
            for name, cfg in load_configs().items():
                c = {**CONFIG_DEFAULTS, **cfg, "name": name}
                out.append({**c, "audit": audit_config(c, models.get(c["path"])),
                            "missing": c["path"] not in models})
            return self._json(200, out)
        if self.path == "/api/presets":
            return self._json(200, load_presets())
        if self.path == "/api/downloads":
            with JOBS_LOCK:
                return self._json(200, [
                    {k: v for k, v in j.items() if k not in ("cancel", "base")}
                    for j in JOBS.values()])
        if self.path.startswith("/api/search"):
            q = urllib.parse.parse_qs(self.path.split("?", 1)[-1]).get("q", [""])[0]
            if not q:
                return self._json(400, {"error": "q required"})
            try:
                return self._json(200, hub_search(q))
            except Exception as e:
                return self._json(502, {"error": f"hub unreachable: {e}"})
        if self.path.startswith("/api/files"):
            repo = urllib.parse.parse_qs(
                self.path.split("?", 1)[-1]).get("repo", [""])[0]
            if not repo:
                return self._json(400, {"error": "repo required"})
            try:
                return self._json(200, hub_files(repo))
            except Exception as e:
                return self._json(502, {"error": f"hub unreachable: {e}"})
        if self.path == "/api/status":
            return self._json(200, {
                "running": server_running(),
                "model": PROC["model"]["name"] if PROC["model"] else None,
                "path": PROC["model"]["path"] if PROC["model"] else None,
                "port": PROC["port"],
                "uptime": int(time.time() - PROC["started"]) if PROC["started"] else 0,
                "lora": PROC["lora"], "lora_scale": PROC["lora_scale"],
                "arch": PROC["model"].get("arch") if PROC["model"] else None,
                "n_experts": PROC["model"].get("n_experts") if PROC["model"] else None,
                "id": PROC["model"].get("id") if PROC["model"] else None,
                "vision": bool(PROC["model"] and PROC["model"].get("mmproj")
                               and "--mmproj" in (PROC["args"] or [])),
                "idle_ttl": STATE["idle_ttl"], "jit": STATE["jit"],
                "auto_ready": any(model_summary(m)["eligible"] for m in scan_models()),
                "idle_for": int(time.time() - ACTIVITY["last"]),
            })
        self._json(404, {"error": "not found"})

    def do_POST(self):
        self._route = None
        if self.path == "/api/login":
            req = self._read()
            if STATE["token"] and not THROTTLE.blocked(self.client_address[0]) and hmac.compare_digest(
                    str(req.get("token", "")), STATE["token"]):
                body = json.dumps({"ok": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Set-Cookie",
                                 f"ns_token={STATE['token']}; Path=/; "
                                 "HttpOnly; SameSite=Strict; Max-Age=604800"
                                 + ("; Secure" if STATE["tls"] else ""))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                THROTTLE.fail(self.client_address[0])
                self._json(401, {"error": "bad token"})
            return
        if self.path == "/api/pair/claim":
            ip = self.client_address[0]
            if THROTTLE.blocked(ip):
                return self._json(429, {"error": "too many attempts; wait a few minutes"})
            req = self._read()
            r = STATE["devices"].claim(str(req.get("code", "")), str(req.get("name", "device")))
            if r is None:
                THROTTLE.fail(ip)
                return self._json(403, {"error": "pairing code is wrong, used or expired"})
            r["fingerprint"] = STATE.get("fingerprint") or ""
            if req.get("cookie"):
                body = json.dumps(r).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                # The cookie lives as long as the access: a year for persistent pairing (the
                # host can still revoke it), exactly the granted time for temporary access.
                age = 31536000 if r["expires"] is None else max(1, int(r["expires"] - time.time()))
                self.send_header("Set-Cookie", f"ns_token={r['token']}; Path=/; HttpOnly; SameSite=Strict; "
                                 f"Max-Age={age}" + ("; Secure" if STATE["tls"] else ""))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return self._json(200, r)
        if not self._authed():
            return self._deny()
        if self._role == "device" and self.path not in DEVICE_POSTS:
            # A paired device may use models, not administer this machine: no jobs (they
            # run code), no load flags or downloads, no MCP reloads, no pairing.
            return self._json(403, {"error": "paired devices can chat and use /v1; this needs the owner"})
        try:
            if self.path in ("/api/pair/start", "/api/devices/revoke", "/api/devices/update", "/api/links/add",
                             "/api/links/remove"):
                return self._pairing_post()
            if self.path in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings"):
                return self._v1(self.path)

            if self.path == "/api/chats":
                return self._json(200, save_chat(self._read(MAX_BODY)))

            if self.path in ("/api/jobs", "/api/jobs/cancel"):
                return self._jobs_post()

            if self.path == "/api/trace":
                req = self._read(MAX_BODY)
                ref = req.get("model")
                model = find_model(ref) if ref and ref != "auto" else PROC["model"]
                if model is None:
                    return self._json(404, {"error": "checks run on local models only"})
                text = str(req.get("text") or "")
                if not text.strip():
                    return self._json(400, {"error": "nothing to check"})
                try:
                    return self._json(200, trace_reply(model, req.get("messages") or [], text,
                                                       float(req.get("threshold", 0.5))))
                except ValueError as e:
                    return self._json(400, {"error": str(e)})

            if self.path.startswith("/api/rag/"):
                return self._rag_post()

            if self.path == "/api/tools/approve":
                req = self._read()
                with APPROVALS_LOCK:
                    a = APPROVALS.get(str(req.get("key", "")))
                    if a:
                        a["allow"] = bool(req.get("allow"))
                        a["event"].set()
                return self._json(200 if a else 404, {"ok": bool(a)})

            if self.path == "/api/mcp/reload":
                hub = mcp_hub()
                if hub is None:
                    return self._json(400, {"error": "no MCP config (--mcp-config)"})
                hub.connect()
                return self._json(200, {"servers": hub.status(), "config": str(hub.config_path)})

            if self.path == "/api/stats/ingest":
                req = self._read(8 * 1024 * 1024)
                target = find_model(str(req.get("model", "")))
                if target is None:
                    return self._json(404, {"error": f"no local model matches {req.get('model')!r}"})
                n = 0
                for r in req.get("records", [])[:20000]:
                    if not isinstance(r, dict):
                        continue
                    rec = {k: r[k] for k in ("subject", "task_kind", "verdict", "task", "source") if k in r}
                    if STATE["stats"].record(target["id"], "graded", size=target["size"], **rec):
                        n += 1
                return self._json(200, {"model": target["id"], "recorded": n})

            if self.path == "/api/chats/delete":
                try:
                    os.remove(_chat_path(self._read().get("id")))
                except FileNotFoundError:
                    pass
                return self._json(200, {"ok": True})

            if self.path == "/api/load":
                req = self._read()
                target = next((m for m in scan_models()
                               if m["path"] == req["path"]), None)
                if not target:
                    return self._json(404, {"error": "model not found"})
                settings = {**DEFAULTS, **req.get("settings", {})}
                allset = load_settings()
                allset[_key(target["path"])] = settings
                save_settings(allset)
                with LOCK:
                    ok, why = start_server(target, settings, req.get("port", 8080))
                return self._json(200 if ok else 500,
                                  {"ok": ok, "message": why})

            if self.path == "/api/unload":
                with LOCK:
                    stop_server()
                return self._json(200, {"ok": True})

            if self.path == "/api/lora":
                req = self._read()
                scale = float(req["scale"])
                proxy("/lora-adapters", [{"id": 0, "scale": scale}])
                PROC["lora_scale"] = scale
                return self._json(200, {"ok": True, "scale": scale})

            if self.path == "/api/config":
                req = self._read()
                if not req.get("name"):
                    return self._json(400, {"error": "name required"})
                save_config(req)
                models = {m["path"]: m for m in scan_models()}
                return self._json(200, {"ok": True,
                                        "audit": audit_config(
                                            {**CONFIG_DEFAULTS, **req},
                                            models.get(req.get("path")))})

            if self.path == "/api/config/delete":
                delete_config(self._read()["name"])
                return self._json(200, {"ok": True})

            if self.path == "/api/preset":
                req = self._read()
                save_preset(req["name"], req["config"])
                return self._json(200, {"ok": True})

            if self.path == "/api/download":
                req = self._read()
                root = STATE["download_dir"] or STATE["models_dirs"][0]
                if not os.path.isdir(root):
                    return self._json(400,
                                      {"error": f"{root} does not exist"})
                jid = start_download(req["repo"], req["group"], root)
                return self._json(200, {"ok": True, "id": jid})

            if self.path == "/api/download/cancel":
                req = self._read()
                with JOBS_LOCK:
                    j = JOBS.get(req["id"])
                if j:
                    j["cancel"] = True
                return self._json(200, {"ok": bool(j)})

            if self.path == "/api/chat":
                req = self._read(MAX_BODY)
                lt = link_target(req.get("model"))
                upstream = None
                if lt is not None:
                    upstream, rid = lt
                    target, route = None, {"mode": "link", "model": req["model"]}
                else:
                    with LOCK:
                        target, route = resolve_request_model({"model": req.get("model", ""),
                                                               "messages": req.get("messages", [])})
                        if target is None:
                            msg = route.get("error") or (f"model not found: {req.get('model')}" if req.get("model")
                                                         else "no model loaded")
                            return self._json(409, {"error": msg})
                        ok, why = ensure_loaded(target, own=0)
                    if not ok:
                        return self._json(503, {"error": f"could not load {target['id']}: {why}"})
                if target is not None and _has_image(req) and not target.get("mmproj"):
                    return self._json(400, {"error": f"{target['id']} has no vision projector (mmproj)"})
                preset = load_presets().get(req.get("preset", "default"),
                                            PRESET_DEFAULTS)
                msgs = list(req["messages"])
                if preset.get("system") and not (
                        msgs and msgs[0].get("role") == "system"):
                    msgs.insert(0, {"role": "system",
                                    "content": preset["system"]})
                try:
                    msgs, sources = rag_context(req, msgs)
                except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
                    return self._json(400, {"error": f"retrieval failed: {e}"})
                payload = {"messages": msgs,
                           "temperature": preset.get("temperature", 0.7),
                           "top_p": preset.get("top_p", 0.95),
                           "max_tokens": preset.get("max_tokens", 2048),
                           "stream": True,
                           "stream_options": {"include_usage": True}}
                if preset.get("json_schema"):
                    try:
                        schema = json.loads(preset["json_schema"])
                    except json.JSONDecodeError as e:
                        return self._json(400, {"error": f"bad json_schema: {e}"})
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "response", "schema": schema},
                    }
                elif preset.get("grammar"):
                    payload["grammar"] = preset["grammar"]
                if upstream is not None:
                    payload["model"] = rid
                self._route = route
                self._send_headers_sse()
                if sources:
                    self.wfile.write(f"data: {json.dumps({'sources': sources})}\n\n".encode())
                touch(+1)
                try:
                    # MCP tools run with the host owner's permissions: owner sessions only
                    hub = mcp_hub() if req.get("tools") and self._role == "owner" else None
                    if hub is not None and hub.openai_tools():
                        out = chat_with_tools(self.wfile, payload, msgs, hub, upstream=upstream)
                    else:
                        out = proxy("/v1/chat/completions", payload, stream_to=self.wfile, upstream=upstream)
                    if target is not None:
                        note_reply(target, msgs, out["text"])
                except Exception as e:
                    self.wfile.write(
                        f"data: {json.dumps({'error': str(e)})}\n\n".encode())
                finally:
                    touch(-1)
                return

            self._json(404, {"error": "not found"})
        except BrokenPipeError:
            pass
        except ValueError as e:
            self._json(400, {"error": str(e)[:300]})
        except Exception as e:
            try:
                self._json(500, {"error": str(e)[:300]})
            except Exception:
                pass


_PAIR_STYLE = """<style>
:root{--bg:#f7f7f5;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b64;--line:#e3e2dd;--acc:#2c5f8a;--accfg:#fff;--ok:#2f7d4f;--no:#b23c2e;--code:#f3f2ee;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--panel:#1e1e1c;--fg:#ecebe6;--mut:#9b9a93;--line:#34332f;--acc:#7aa7d6;--accfg:#0f0f0e;--ok:#5fb27f;--no:#e0705f;--code:#262522;color-scheme:dark}}
:root[data-theme=dark]{--bg:#161615;--panel:#1e1e1c;--fg:#ecebe6;--mut:#9b9a93;--line:#34332f;--acc:#7aa7d6;--accfg:#0f0f0e;--ok:#5fb27f;--no:#e0705f;--code:#262522;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:15px;margin:0}a{color:var(--acc)}
main{max-width:900px;margin:0 auto;padding:1rem;display:grid;gap:1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.9rem}
h2{font-size:13px;margin:0 0 .5rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
input{font:13px ui-monospace,monospace;padding:.35rem .45rem;border:1px solid var(--line);border-radius:5px;width:100%;background:var(--bg);color:var(--fg)}
select{font:13px ui-sans-serif,system-ui;padding:.3rem .4rem;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg)}
#access{width:100%}
button{font:500 13px ui-sans-serif,system-ui;padding:.4rem .8rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
.note{font-size:12.5px;color:var(--mut)}.err{color:var(--no)}.ok{color:var(--ok)}
code,.code{font:12.5px ui-monospace,monospace;background:var(--code);padding:.15rem .35rem;border-radius:4px;word-break:break-all}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:.3rem .25rem;border-bottom:1px solid var(--line)}
.row{display:flex;gap:.5rem;align-items:flex-end;flex-wrap:wrap}.row>div{flex:1;min-width:180px}
label{display:block;font-size:12px;color:var(--mut);margin:.3rem 0 .1rem}
</style><script>try{document.documentElement.dataset.theme=localStorage.getItem('ns-theme')||'dark'}catch{document.documentElement.dataset.theme='dark'}</script>"""

PAIR_PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pair device</title>""" + _PAIR_STYLE + r"""
<header><h1>NeuronScope Studio · Pair this device</h1></header>
<main><div class="card"><h2>Pair</h2><div id="msg" class="note">Reading the pairing code…</div>
<div id="form" style="display:none"><label>Name for this device</label><input id="name">
<button class="pri" id="go" style="margin-top:.6rem">Pair and open Studio</button></div></div></main>
<script>
const p=new URLSearchParams(location.hash.slice(1)), code=p.get('c'), acc=p.get('a'), $=s=>document.querySelector(s);
const accText=acc==='p'?'Access lasts until the owner revokes it.':acc?`Access is temporary: it ends ${acc} after pairing.`:'';
history.replaceState(null,'',location.pathname);         // keep the code out of history
if(!code){ $('#msg').innerHTML='<span class="err">This link has no pairing code. Ask the owner for a new one.</span>'; }
else { $('#msg').textContent='Pairing code '+code+'. This browser will get its own access, which the owner can revoke. '+accText;
  $('#name').value=(navigator.userAgentData?.platform||navigator.platform||'browser')+' browser'; $('#form').style.display=''; }
$('#go').onclick=async()=>{ const r=await fetch('/api/pair/claim',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({code,name:$('#name').value,cookie:true})}); const j=await r.json().catch(()=>({}));
  if(r.ok){ if(j.expires) alert('Paired. Access ends '+new Date(j.expires*1000).toLocaleString()+'.'); location.href='/'; } else { $('#msg').innerHTML='<span class="err">'+(j.error||r.status)+'</span>'; } };
</script>"""

LINK_PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Link</title>""" + _PAIR_STYLE + r"""
<header><h1>Studio · Link</h1><a href="/">← Studio</a></header>
<main>
<div class="card"><h2>Pair a device with this Studio</h2>
 <div class="note">Creates a one-time link (<span id="codettl">5 minutes</span>). Open it on a phone or laptop, or paste it into another Studio below. Each device gets its own token, revocable here; the master token is never shared. The link carries this server's certificate fingerprint, so the other side pins it instead of trusting any certificate.</div>
 <div class="row" style="margin-top:.5rem"><div style="max-width:260px"><label>Access</label><select id="access"></select></div>
 <div id="customBox" style="max-width:160px;display:none"><label>Duration (e.g. 12h, 3d)</label><input id="custom"></div>
 <button class="pri" id="start">Create pairing link</button></div>
 <div id="policy" class="note" style="margin-top:.3rem"></div>
 <div id="pairout" style="margin-top:.6rem"></div></div>
<div class="card"><h2>Paired devices</h2><div id="devs" class="note">loading…</div></div>
<div class="card"><h2>Use another machine's models here</h2>
 <div class="note">On the other machine's Studio, open Link → Create pairing link, and paste it here. Its models then appear in this Studio's model picker and its <code>/v1</code> API as <code>name:model</code>; requests are served by that machine.</div>
 <div class="row"><div><label>Pairing link from the other Studio</label><input id="lnk" placeholder="https://host:7870/pair#c=…&fp=…"></div>
 <div style="max-width:200px"><label>Name here</label><input id="lname" placeholder="gpu-box"></div><button class="pri" id="add">Link</button></div>
 <div id="addmsg" class="note" style="margin-top:.4rem"></div>
 <div id="links" style="margin-top:.6rem"></div></div>
</main>
<script>
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
const when=t=>t?new Date(t*1000).toLocaleString():'never';
let policy=null;
const left=t=>{ const s=t-Date.now()/1000; if(s<=0) return 'expired';
  return s>86400?`${Math.floor(s/86400)}d ${Math.floor(s%86400/3600)}h left`:s>3600?`${Math.floor(s/3600)}h ${Math.floor(s%3600/60)}m left`:`${Math.ceil(s/60)}m left`; };
const accessCell=d=>d.persistent?'persistent':d.expired?`<span class="err">expired ${when(d.expires)}</span>`:`until ${when(d.expires)} <span class="note">(${left(d.expires)})</span>`;
function accessChoice(){ const v=$('#access').value; if(v==='persistent') return {persistent:true};
  if(v==='custom') return {persistent:false, ttl:$('#custom').value.trim()}; return {persistent:false, ttl:v}; }
function renderPolicy(pol){ policy=pol; const keep=$('#access').value;
  $('#access').innerHTML=(pol.allow_persistent?'<option value="persistent">Persistent (until revoked)</option>':'')+
    pol.presets.map(p=>`<option value="${p}">Temporary: ${p}</option>`).join('')+`<option value="custom">Temporary: custom…</option>`;
  $('#access').value=keep||(pol.default==='persistent'?'persistent':pol.default);
  if(!$('#access').value) $('#access').selectedIndex=0;
  $('#codettl').textContent=pol.code_ttl>=120?`${Math.round(pol.code_ttl/60)} minutes`:`${pol.code_ttl} seconds`;
  $('#policy').textContent=`Host policy: temporary access up to ${pol.max_ttl}`+(pol.allow_persistent?'; persistent allowed.':'; persistent pairing is disabled on this host.'); }
$('#access').onchange=()=>{ $('#customBox').style.display=$('#access').value==='custom'?'':'none'; };
async function load(){ const r=await fetch('/api/devices'); const j=await r.json();
  if(!r.ok){ $('#devs').innerHTML='<span class="err">'+esc(j.error)+'</span>'; $('#start').disabled=true; $('#add').disabled=true; return; }
  renderPolicy(j.policy);
  $('#devs').innerHTML=j.devices.length?'<table><tr><th>device</th><th>paired</th><th>last seen</th><th>access</th><th></th></tr>'+j.devices.map(d=>`<tr><td>${esc(d.name)}</td><td>${when(d.created)}</td><td>${when(d.last_seen)}</td><td>${accessCell(d)}</td><td style="white-space:nowrap"><select data-u="${esc(d.id)}"><option value="">change…</option>${policy.allow_persistent&&!d.persistent?'<option value="persistent">make persistent</option>':''}${policy.presets.map(p=>`<option value="${p}">${d.persistent?'expire in':d.expired?'renew for':'reset to'} ${p}</option>`).join('')}</select> <button data-r="${esc(d.id)}">Revoke</button></td></tr>`).join('')+'</table>':'No paired devices.';
  document.querySelectorAll('[data-r]').forEach(b=>b.onclick=async()=>{ if(confirm('Revoke this device? It loses access immediately.')){ await post('/api/devices/revoke',{id:b.dataset.r}); load(); } });
  document.querySelectorAll('[data-u]').forEach(sel=>sel.onchange=async()=>{ const v=sel.value; if(!v) return;
    const r=await post('/api/devices/update', v==='persistent'?{id:sel.dataset.u,persistent:true}:{id:sel.dataset.u,persistent:false,ttl:v});
    if(!r.ok) alert((await r.json()).error); load(); });
  $('#links').innerHTML=j.links.length?'<table><tr><th>name</th><th>host</th><th>access</th><th>models</th><th></th></tr>'+j.links.map(l=>`<tr><td>${esc(l.name)}</td><td><code>${esc(l.url)}</code> ${l.pinned?'🔒 pinned':''}</td><td>${l.expires?accessCell({expires:l.expires,expired:l.expires<Date.now()/1000}):'persistent'}</td><td>${l.error?'<span class="err">'+esc(l.error)+'</span>':l.models}</td><td><button data-l="${esc(l.name)}">Unlink</button></td></tr>`).join('')+'</table>':'';
  document.querySelectorAll('[data-l]').forEach(b=>b.onclick=async()=>{ await post('/api/links/remove',{name:b.dataset.l}); load(); }); }
$('#start').onclick=async()=>{ const r=await post('/api/pair/start',accessChoice()); const j=await r.json();
  const grants=j.persistent?'persistent access (until revoked)':`temporary access for ${Math.round(j.access_seconds/3600*10)/10} h after pairing`;
  $('#pairout').innerHTML=r.ok?`<div><span class="code" id="plink">${esc(j.link)}</span> <button id="cp">Copy</button></div><div class="note">Grants ${grants}. Code <b>${esc(j.code)}</b>, claimable until ${new Date(j.expires*1000).toLocaleTimeString()}, single use.${j.fingerprint?'':' <span class="err">No TLS on this Studio: use the link only inside a VPN.</span>'}</div>`:'<span class="err">'+esc(j.error)+'</span>';
  if(r.ok) $('#cp').onclick=()=>navigator.clipboard.writeText(j.link); };
$('#add').onclick=async()=>{ $('#addmsg').textContent='pairing…'; const r=await post('/api/links/add',{link:$('#lnk').value.trim(),name:$('#lname').value.trim()||'remote',device_name:location.host});
  const j=await r.json(); $('#addmsg').innerHTML=r.ok?`<span class="ok">Linked ${esc(j.name)}: ${j.models.length} models.</span>`:'<span class="err">'+esc(j.error)+'</span>'; if(r.ok){ $('#lnk').value=''; } load(); };
load();
</script>"""

PAGE = r"""<!DOCTYPE html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuronScope Studio</title><link rel="icon" href="data:,"><script>try{document.documentElement.dataset.theme=localStorage.getItem('ns-theme')||'dark'}catch{document.documentElement.dataset.theme='dark'}</script>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b64;--line:#e3e2dd;--ok:#2f7d4f;--no:#b23c2e;--warn:#a8701c;
  --acc:#2c5f8a;--accfg:#fff;--soft:#f0f5fa;--code:#f3f2ee;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111316;--panel:#181b20;--fg:#e6e8eb;--mut:#9aa1ab;
  --line:#2a2f37;--ok:#5fcf8f;--no:#ff8b7e;--warn:#e7b65c;--acc:#6ea8e0;--accfg:#0d1117;--soft:#1d2733;--code:#20242b;color-scheme:dark}}
:root[data-theme=dark]{--bg:#111316;--panel:#181b20;--fg:#e6e8eb;--mut:#9aa1ab;--line:#2a2f37;--ok:#5fcf8f;--no:#ff8b7e;
  --warn:#e7b65c;--acc:#6ea8e0;--accfg:#0d1117;--soft:#1d2733;--code:#20242b;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
header{padding:.55rem 1rem;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:.8rem;background:var(--panel);flex-wrap:wrap}
h1{margin:0;font-size:15px;font-weight:600}
.status{font-size:13px;color:var(--mut);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--line);margin-right:.35rem}.dot.on{background:var(--ok)}
.chip{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:999px;padding:.1rem .6rem;color:var(--mut);cursor:pointer;background:var(--bg)}
main{flex:1;display:grid;grid-template-columns:330px 1fr;min-height:0}
@media(max-width:800px){main{grid-template-columns:1fr;grid-template-rows:auto 1fr}aside{max-height:30vh;border-right:none;border-bottom:1px solid var(--line)}
  .dials{gap:.45rem .7rem;padding:.4rem .8rem}.dials input[type=range]{width:90px}#export{display:none}
  form{padding:.5rem .8rem}#log{padding:.7rem .8rem}header{gap:.5rem}#api{display:none}}
aside{border-right:1px solid var(--line);overflow-y:auto;padding:.7rem;background:var(--panel)}
.tabs{display:flex;gap:.25rem;margin-bottom:.6rem;flex-wrap:wrap}.tabs button{flex:1 1 auto;min-width:0;padding:.35rem .45rem;font-size:12.5px}
section.chat{display:flex;flex-direction:column;min-height:0;min-width:0}
.m{border:1px solid var(--line);border-radius:8px;padding:.5rem .65rem;margin-bottom:.4rem;cursor:pointer;background:var(--panel);position:relative}
.m:hover{background:var(--soft)}.m.sel{border-color:var(--acc);background:var(--soft)}
.mn{font-weight:500;font-size:13px;word-break:break-all}.mm{font-size:12px;color:var(--mut);margin-top:.1rem}
.x{position:absolute;right:.4rem;top:.3rem;border:none;background:none;color:var(--mut);cursor:pointer;padding:0 .3rem}
.tag{display:inline-block;font-size:11px;padding:0 .35rem;border:1px solid var(--line);border-radius:4px;margin-right:.25rem}
.bad{color:var(--no)}.warn{color:var(--warn)}.okc{color:var(--ok)}
label{display:block;font-size:12px;color:var(--mut);margin:.4rem 0 .1rem}
input,select,textarea{font:12px ui-monospace,monospace;padding:.3rem .4rem;border:1px solid var(--line);border-radius:5px;width:100%;background:var(--bg);color:var(--fg)}
input[type=checkbox]{width:auto}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}
button{font:500 13px ui-sans-serif,system-ui;padding:.4rem .8rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button:hover{background:var(--soft)}button:disabled{opacity:.45;cursor:default}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
#log{flex:1;overflow-y:auto;padding:1rem 1.2rem}
.msg{max-width:800px;margin:0 auto 1.1rem}
.who{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin-bottom:.2rem;display:flex;gap:.6rem;align-items:center}
.who button{font-size:11px;padding:0 .4rem;border:none;background:none;color:var(--mut)}
.body{white-space:pre-wrap;word-wrap:break-word}
.body pre{background:var(--code);padding:.6rem .7rem;border-radius:6px;overflow-x:auto;white-space:pre;font:12.5px ui-monospace,monospace}
.body code{background:var(--code);padding:0 .25rem;border-radius:3px;font:12.5px ui-monospace,monospace}
details.think{color:var(--mut);font-size:13px;border-left:2px solid var(--line);padding-left:.7rem;margin-bottom:.5rem}
details.think summary{cursor:pointer}
.stats{font-size:11px;color:var(--mut);margin-top:.3rem}
.risk{font-size:12px;margin-top:.35rem}.risk:empty{display:none}
.risk .sum{color:var(--mut);margin-bottom:.25rem}.risk .sum b{color:var(--fg)}
.risk .rt{white-space:pre-wrap;word-wrap:break-word;line-height:1.7;font-size:13px}
.risk .rt span{border-radius:2px}.risk .rt .fl{text-decoration:underline 2px var(--no);text-underline-offset:3px}
.imgs{display:flex;gap:.4rem;flex-wrap:wrap;margin:.3rem 0}.imgs img{max-height:120px;max-width:200px;border-radius:6px;border:1px solid var(--line)}
form{border-top:1px solid var(--line);padding:.6rem 1.2rem;display:flex;gap:.5rem;background:var(--panel);align-items:flex-end;flex-wrap:wrap}
#in{flex:1;min-width:200px;font:14px ui-sans-serif,system-ui;padding:.5rem .6rem;resize:none}
#pending{width:100%}
.dials{border-top:1px solid var(--line);padding:.5rem 1.2rem;font-size:12px;color:var(--mut);display:flex;gap:1rem;align-items:center;background:var(--panel);flex-wrap:wrap}
.dials select{width:auto}.dials input[type=range]{width:140px}
.note{font-size:12px;color:var(--mut);margin:.4rem 0}
.empty{max-width:560px;margin:15vh auto;text-align:center;color:var(--mut)}
.warnico{color:#e8890c;cursor:help;font-weight:700;margin-left:.3rem}
.stat{font-size:11px;color:var(--mut);cursor:help}
.route{font-size:11px;color:var(--mut);margin-top:.2rem}
table.st{width:100%;border-collapse:collapse;font-size:11.5px;margin-top:.4rem}
table.st th,table.st td{text-align:left;padding:.15rem .3rem;border-bottom:1px solid var(--line)}
#modelPick{width:auto;max-width:260px}
.tool{font-size:12px;border:1px solid var(--line);border-radius:6px;padding:.35rem .5rem;margin:.3rem 0;background:var(--code)}
.tool .tn{font:600 12px ui-monospace,monospace}.tool pre{white-space:pre-wrap;margin:.2rem 0;font:12px ui-monospace,monospace;max-height:220px;overflow:auto}
.tool button{font-size:11px;padding:.1rem .5rem;margin-right:.3rem}
.src{font-size:12px;color:var(--mut);margin-top:.3rem}.src details{margin:.1rem 0}.src summary{cursor:pointer}
.src pre{white-space:pre-wrap;font:12px ui-monospace,monospace;background:var(--code);padding:.4rem;border-radius:5px;margin:.2rem 0}
</style>
<header>
  <h1>NeuronScope Studio</h1>
  <div class="status"><span id="dot" class="dot"></span><span id="st">no model loaded</span></div>
  <span class="chip" id="api" title="OpenAI-compatible endpoint; click to copy"></span>
  <a href="/link" class="chip" style="margin-left:auto;text-decoration:none" title="pair devices and link other machines' models">Link</a>
  <a href="/jobs" class="chip" style="text-decoration:none" title="evaluation, retraining and benchmark jobs">Jobs</a>
  <button id="theme" title="toggle theme">◐</button>
  <button id="unload" disabled>Unload</button>
</header>
<main>
<aside>
  <div class="tabs">
    <button data-tab="chats" class="pri">Chats</button>
    <button data-tab="local">Models</button>
    <button data-tab="hub">Hub</button>
    <button data-tab="docs">Docs</button>
    <button data-tab="tools">Tools</button>
  </div>
  <div id="tab-tools" style="display:none">
    <div class="note">MCP servers whose tools the model may call when <b>tools</b> is on under the chat. Every call asks first unless the server's <code>autoApprove</code> allows it.</div>
    <div id="mcpcfg" class="note"></div>
    <button id="mcpreload" style="width:100%;margin:.4rem 0">Reconnect / reload config</button>
    <div id="mcplist"></div>
  </div>
  <div id="tab-docs" style="display:none">
    <div class="note">Document collections for chat. Pick one under the chat ("docs") and replies cite the passages they used.</div>
    <div style="display:flex;gap:.4rem;margin:.4rem 0"><input id="newcoll" placeholder="new collection name"><button id="mkcoll">Create</button></div>
    <div id="colls"></div>
    <div id="collpanel" style="display:none">
      <hr style="border:none;border-top:1px solid var(--line);margin:.6rem 0">
      <b id="collname"></b> <span id="colldense" class="note"></span>
      <input type="file" id="docfile" multiple hidden>
      <button id="adddocs" style="width:100%;margin:.4rem 0">+ Add files (txt, md, code, html, pdf, docx)</button>
      <div id="docmsg" class="note"></div>
      <div id="docs"></div>
      <label>Try a search</label><input id="ragq" placeholder="query, Enter to search"><div id="raghits" class="note"></div>
      <button id="dropcoll" style="margin-top:.6rem;width:100%;color:var(--no)">Delete collection</button>
    </div>
  </div>
  <div id="tab-chats">
    <button id="newchat" style="width:100%;margin-bottom:.5rem">+ New chat</button>
    <div id="chats"></div>
  </div>
  <div id="tab-hub" style="display:none">
    <input id="q" placeholder="search GGUF models on Hugging Face">
    <button id="go" style="width:100%;margin-top:.4rem">Search</button>
    <div id="hits" class="note"></div>
    <div id="jobs"></div>
  </div>
  <div id="tab-local" style="display:none">
  <input id="filter" placeholder="filter models" style="margin-bottom:.5rem">
  <div id="list">scanning…</div>
  <div id="panel" style="display:none">
    <hr style="border:none;border-top:1px solid var(--line);margin:.7rem 0">
    <div class="row2">
      <div><label>GPU layers</label><input id="ngl" value="99"></div>
      <div><label>Context</label><input id="ctx" value="8192"></div>
    </div>
    <div class="row2">
      <div><label>Batch</label><input id="batch" value="2048"></div>
      <div><label>Threads (0=auto)</label><input id="threads" value="0"></div>
    </div>
    <div id="gpubox" style="display:none">
      <label>GPUs <span id="gpuhint" class="note"></span></label>
      <div class="row2">
        <div><label>Visible GPUs</label><input id="gpus" placeholder="all, or 0,1"></div>
        <div><label>Split</label><select id="split_mode"><option value="">default (layer)</option><option>layer</option><option>row</option><option>none</option></select></div>
      </div>
      <div class="row2">
        <div><label>Tensor split</label><input id="tensor_split" placeholder="1,1,1,1"></div>
        <div><label>Main GPU (-1 = auto)</label><input id="main_gpu" value="-1"></div>
      </div>
    </div>
    <div id="moebox" style="display:none">
      <label>Active experts <span id="moehint" class="note"></span></label>
      <input id="experts" value="0">
    </div>
    <label>Served name (API model id)</label><input id="served_name" placeholder="defaults to the model id">
    <label>LoRA adapter (optional)</label><input id="lora" placeholder="/path/suppress-lora.gguf">
    <label>Draft model (speculative decoding)</label>
    <input id="draft_model" placeholder="/path/small-Q4_K_M.gguf">
    <label>Extra llama-server flags</label><input id="extra" placeholder="--override-tensor '\.ffn_.*_exps\.=CPU'">
    <label id="visionrow" style="display:none"><input type="checkbox" id="vision" checked> load vision projector <span id="mmname"></span></label>
    <label>H-Neuron classifier (optional, enables activation stats)</label><input id="classifier" placeholder="models/classifier.npz for this model">
    <button id="load" class="pri" style="margin-top:.7rem;width:100%">Load</button>
    <div id="statsbox"></div>
    <div id="loadmsg" class="note"></div>
  </div>
  </div>
</aside>
<section class="chat">
  <div id="log"></div>
  <div class="dials">
    <span>preset</span><select id="preset"></select>
    <span title="retrieve passages from a document collection for each message">docs</span><select id="ragPick"><option value="">none</option></select>
    <label id="toolsBox" style="display:none;margin:0" title="let the model call MCP tools (each call asks first)"><input type="checkbox" id="toolsOn"> tools</label>
    <label style="margin:0" title="after each reply, score every token with the model's hallucination classifier (needs studio --cett and a classifier in the model's settings)"><input type="checkbox" id="autoCheck"> check replies</label>
    <span>suppression α</span>
    <input type="range" id="alpha" min="0" max="1" step="0.05" value="1" disabled>
    <span id="av">1.00</span>
    <span id="adesc">no adapter loaded</span>
    <button id="export" style="margin-left:auto">Export .md</button>
  </div>
  <form id="f">
    <div id="pending" class="imgs"></div>
    <input type="file" id="file" accept="image/*" multiple hidden>
    <select id="modelPick" title="which model answers"></select><span id="pickWarn" class="warnico" style="display:none">⚠</span>
    <button type="button" id="attach" title="attach image (vision models)" disabled>📎</button>
    <textarea id="in" rows="2" placeholder="Message… (Enter to send, Shift+Enter for a new line)"></textarea>
    <button class="pri" id="send">Send</button>
  </form>
</section>
</main>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let models=[], sel=null, busy=false, ctrl=null, status_={}, pendingImgs=[];
let chat={id:null,title:'New chat',messages:[]};
const F=["ngl","ctx","batch","threads","experts","served_name","lora","draft_model","extra","classifier","gpus","split_mode","tensor_split","main_gpu"];
const TEXT_FIELDS=["served_name","lora","extra","draft_model","classifier","gpus","split_mode","tensor_split"];
const pct=x=>x==null?'–':(100*x).toFixed(0)+'%';
const NOSTATS_HELP=' Auto routing skips it: it only picks models whose answers have been graded. You can still load it or pick it by name. To add stats, run scripts/testqa.py against it with --publish-stats (see docs/TESTQA.md).';
function statBadge(m){ const s=m.stats; if(!s) return '';
  if(!s.eligible) return `<span class="warnico" title="${esc(s.why_not+NOSTATS_HELP)}">⚠</span>`;
  return ''; }
function statTitle(s){ const g=s.graded; let t=`Rolling stats over the last ${g.n} graded answers:\n`+
  `correct ${pct(g.accuracy)}, hallucinated (answered wrong) ${pct(g.hallucination_rate)}, abstained ${pct(g.abstention_rate)}\n`;
  for(const [k,v] of Object.entries(s.subjects||{})) t+=`  ${k}: ${pct(v.accuracy)} right, ${pct(v.hallucination_rate)} wrong, ${pct(v.abstention_rate)} abstained (n ${v.n})\n`;
  if(s.activation&&s.activation.n) t+=`H-Neuron activation over ${s.activation.n} replies: mean ${s.activation.mean.toFixed(2)}, flagged ${pct(s.activation.flagged_rate)}\n`;
  if(s.live&&s.live.n) t+=`Live traffic: ${s.live.n} replies, ${pct(s.live.abstention_rate)} declined\n`;
  if(g.sources) t+=`Graded by: ${Object.entries(g.sources).map(([k,v])=>k+' '+v).join(', ')}\n`;
  const refs=Object.entries(s.reference||{}); if(refs.length) t+=`Published scores (reference only, not used by auto): `+refs.map(([k,v])=>`${k} ${pct(v.score)}`).join(', ');
  return t; }
function statLine(m){ const s=m.stats; if(!s||!s.graded.n) return '';
  const g=s.graded; return `<div class="stat" title="${esc(statTitle(s))}">✓ ${pct(g.accuracy)} · halluc ${pct(g.hallucination_rate)} · abst ${pct(g.abstention_rate)} · n ${g.n}</div>`; }
function statsTable(s){ if(!s) return '';
  if(!s.graded.n && !(s.activation&&s.activation.n) && !(s.live&&s.live.n)) return `<div class="note"><span class="warnico">⚠</span> ${esc(s.why_not+NOSTATS_HELP)}</div>`;
  let h='<table class="st"><tr><th>subject</th><th>n</th><th>right</th><th>halluc</th><th>abstain</th></tr>';
  const rows=[['all',s.graded],...Object.entries(s.subjects||{})];
  for(const [k,v] of rows) if(v.n) h+=`<tr><td>${esc(k)}</td><td>${v.n}</td><td>${pct(v.accuracy)}</td><td>${pct(v.hallucination_rate)}</td><td>${pct(v.abstention_rate)}</td></tr>`;
  h+='</table>';
  if(s.activation&&s.activation.n) h+=`<div class="note">H-Neuron activation: ${s.activation.n} replies, mean score ${s.activation.mean.toFixed(2)}, p95 ${s.activation.p95.toFixed(2)}, flagged ${pct(s.activation.flagged_rate)}</div>`;
  if(s.live&&s.live.n) h+=`<div class="note">Live: ${s.live.n} replies, ${pct(s.live.abstention_rate)} declined</div>`;
  if(!s.eligible) h+=`<div class="note"><span class="warnico">⚠</span> ${esc(s.why_not)}</div>`;
  return h; }
function human(b){const u=["B","KB","MB","GB","TB"];let i=0;while(b>=1024&&i<4){b/=1024;i++}return b.toFixed(1)+' '+u[i]}
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b??{})});

// theme
// Dark by default (set in <head>); the toggle remembers the choice.
$('#theme').onclick=()=>{const cur=document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
  const nx=cur==='dark'?'light':'dark'; document.documentElement.dataset.theme=nx; try{localStorage.setItem('ns-theme',nx)}catch{}};
$('#api').textContent=location.origin+'/v1';
$('#api').onclick=async()=>{try{await navigator.clipboard.writeText(location.origin+'/v1'); $('#api').textContent='copied';
  setTimeout(()=>$('#api').textContent=location.origin+'/v1',1200)}catch{}};

// tabs
$$('.tabs button').forEach(b=>b.onclick=()=>{ $$('.tabs button').forEach(x=>x.classList.toggle('pri',x===b));
  ['chats','local','hub','docs','tools'].forEach(t=>$('#tab-'+t).style.display=t===b.dataset.tab?'block':'none'); });

// ---------- models
let linked=[];
async function refresh(){
  models=await (await fetch('/api/models')).json();
  try{ linked=await (await fetch('/api/links/models')).json(); }catch{ linked=[]; }
  renderModels(); renderPicker();
}
function renderPicker(){
  const cur=$('#modelPick').value; const anyOk=models.some(m=>m.stats&&m.stats.eligible);
  let o=`<option value="">Loaded model</option><option value="auto"${anyOk?'':' disabled'}>Auto (by performance stats)${anyOk?'':' - no models have stats'}</option>`;
  o+=models.map(m=>`<option value="${esc(m.id)}">${m.stats&&!m.stats.eligible?'⚠ ':''}${esc(m.id)}${m.stats&&!m.stats.eligible?' (no stats)':''}</option>`).join('');
  if(linked.length) o+=`<optgroup label="linked hosts">`+linked.map(m=>`<option value="${esc(m.id)}">⇢ ${esc(m.id)}</option>`).join('')+`</optgroup>`;
  $('#modelPick').innerHTML=o; if([...$('#modelPick').options].some(x=>x.value===cur&&!x.disabled)) $('#modelPick').value=cur;
  pickChanged();
}
function pickChanged(){
  const m=models.find(x=>x.id===$('#modelPick').value); const w=$('#pickWarn');
  if(m&&m.stats&&!m.stats.eligible){ w.style.display='inline'; w.title=m.stats.why_not+' You picked it manually, so it will be used.'+NOSTATS_HELP; }
  else w.style.display='none';
  $('#attach').disabled=!( status_.vision || (m&&m.mmproj) || $('#modelPick').value==='auto');
}
$('#modelPick').onchange=pickChanged;
function renderModels(){
  const q=$('#filter').value.toLowerCase();
  const rows=models.map((m,i)=>[m,i]).filter(([m])=>!q||m.name.toLowerCase().includes(q)||(m.arch||'').includes(q));
  $('#list').innerHTML = rows.length ? rows.map(([m,i])=>{
    const f=m.fit||{}; const cls=f.ok===false?'bad':(/tight/.test(f.note||'')?'warn':'');
    return `<div class="m${sel&&sel.path===m.path?' sel':''}" data-i="${i}">
      <div class="mn">${esc(m.name)}${statBadge(m)}</div>
      <div class="mm">${m.quant?`<span class="tag">${esc(m.quant)}</span>`:''}${m.arch?`<span class="tag">${esc(m.arch)}</span>`:''}
        ${m.n_experts?`<span class="tag">MoE ${m.n_experts}</span>`:''}${m.mmproj?'<span class="tag">vision</span>':''}
        <span class="tag">${human(m.size)}</span></div>
      <div class="mm ${cls}">${esc(f.note||'')}</div>${statLine(m)}</div>`;
  }).join('') : '<div class="note">No .gguf files found under the configured directories.</div>';
  $$('#list .m').forEach(el=>el.onclick=()=>pick(+el.dataset.i));
}
$('#filter').oninput=renderModels;
function pick(i){
  sel=models[i]; renderModels(); $('#panel').style.display='block';
  const s=sel.settings||{};
  F.forEach(k=>{ if($('#'+k)) $('#'+k).value = s[k] ?? ''; });
  $('#moebox').style.display = sel.n_experts?'block':'none';
  if(sel.n_experts) $('#moehint').textContent = `0 = model default (of ${sel.n_experts})`;
  $('#visionrow').style.display = sel.mmproj?'block':'none';
  $('#statsbox').innerHTML = statsTable(sel.stats);
  $('#vision').checked = s.vision!==false; $('#mmname').textContent = sel.mmproj?'('+sel.mmproj.split('/').pop()+')':'';
}
$('#load').onclick=async()=>{
  if(!sel) return;
  const s={}; F.forEach(k=>{ const el=$('#'+k); if(!el) return; s[k] = TEXT_FIELDS.includes(k) ? el.value : (parseFloat(el.value)||0); });
  if($('#main_gpu').value.trim()==='' || isNaN(parseFloat($('#main_gpu').value))) s.main_gpu=-1; else s.main_gpu=parseInt($('#main_gpu').value);
  s.flash_attn=true; s.lora_scale=1.0; s.vision=$('#vision').checked;
  $('#load').disabled=true; $('#loadmsg').textContent='starting llama-server…';
  const r=await (await post('/api/load',{path:sel.path,settings:s})).json();
  $('#load').disabled=false;
  $('#loadmsg').innerHTML = r.ok?'<span class="okc">loaded</span>':'<span class="bad">'+esc(r.message||r.error||'failed')+'</span>';
  status();
};
$('#unload').onclick=async()=>{ await post('/api/unload'); status(); };
async function status(){
  const s=status_=await (await fetch('/api/status')).json();
  $('#dot').className='dot'+(s.running?' on':'');
  $('#st').textContent = s.running ? `${s.id||s.model}${s.vision?' · vision':''} · up ${s.uptime}s${s.idle_ttl?` · TTL ${s.idle_ttl}s`:''}` : 'no model loaded';
  $('#unload').disabled=!s.running; pickChanged();
  const hasLora=!!s.lora; $('#alpha').disabled=!hasLora;
  $('#adesc').textContent = hasLora ? s.lora.split('/').pop() : 'no adapter loaded';
  if(hasLora && document.activeElement!==$('#alpha')){ $('#alpha').value=s.lora_scale; $('#av').textContent=(+s.lora_scale).toFixed(2); }
}
$('#alpha').oninput=async e=>{ const v=parseFloat(e.target.value); $('#av').textContent=v.toFixed(2); await post('/api/lora',{scale:v}); };

// ---------- chats
async function loadChats(){
  const cs=await (await fetch('/api/chats')).json();
  $('#chats').innerHTML = cs.length ? cs.map(c=>`<div class="m${c.id===chat.id?' sel':''}" data-id="${esc(c.id)}">
    <button class="x" data-del="${esc(c.id)}" title="delete">×</button>
    <div class="mn">${esc(c.title||'Untitled')}</div>
    <div class="mm">${c.n} messages · ${new Date(c.updated*1000).toLocaleString()}</div></div>`).join('')
    : '<div class="note">Conversations are saved here automatically.</div>';
  $$('#chats .m').forEach(el=>el.onclick=async e=>{
    if(e.target.dataset.del){ e.stopPropagation(); if(confirm('Delete this chat?')){ await post('/api/chats/delete',{id:e.target.dataset.del});
      if(chat.id===e.target.dataset.del) newChat(); loadChats(); } return; }
    chat=await (await fetch('/api/chats/'+el.dataset.id)).json(); renderChat(); loadChats(); });
}
function newChat(){ chat={id:null,title:'New chat',messages:[]}; renderChat(); loadChats(); }
$('#newchat').onclick=newChat;
async function persist(){
  if(!chat.messages.length) return;
  if(chat.title==='New chat'){ const first=chat.messages.find(m=>m.role==='user'); chat.title=textOf(first?.content).slice(0,60)||'New chat'; }
  chat.model=status_.id||null; chat.preset=$('#preset').value;
  const r=await (await post('/api/chats',chat)).json(); chat.id=r.id; loadChats();
}
const textOf=c=>Array.isArray(c)?c.filter(x=>x.type==='text').map(x=>x.text).join(' '):String(c||'');
const imagesOf=c=>Array.isArray(c)?c.filter(x=>x.type==='image_url').map(x=>x.image_url.url):[];

// minimal, safe markdown: fenced code blocks and inline code only
function render(text){
  const parts=String(text).split(/```/); let out='';
  parts.forEach((p,i)=>{ if(i%2){ const nl=p.indexOf('\n'); out+='<pre>'+esc(nl>=0?p.slice(nl+1):p)+'</pre>'; }
    else out+=esc(p).replace(/`([^`\n]+)`/g,'<code>$1</code>'); });
  return out;
}
function bubble(role,content,extra={}){
  if(!$('#log .msg')) $('#log').innerHTML='';
  const d=document.createElement('div'); d.className='msg';
  d.innerHTML=`<div class="who">${role==='user'?'you':'assistant'}<span class="acts"></span></div>`+
    (extra.think!==undefined?`<details class="think"><summary>reasoning</summary><div></div></details>`:'')+
    `<div class="imgs"></div><div class="tools"></div><div class="body"></div><div class="src"></div><div class="stats"></div><div class="risk"></div>`;
  d.querySelector('.imgs').innerHTML=imagesOf(content).map(u=>`<img src="${esc(u)}" alt="attached image">`).join('');
  d.querySelector('.body').innerHTML=render(textOf(content));
  if(extra.think) d.querySelector('.think div').textContent=extra.think;
  if(extra.stats) d.querySelector('.stats').textContent=extra.stats;
  if(extra.sources) showSources(d.querySelector('.src'),extra.sources);
  for(const t of extra.tools||[]) toolCard(d.querySelector('.tools'),t);
  if(role==='assistant'){ const a=d.querySelector('.acts');
    a.innerHTML='<button data-a="copy">copy</button><button data-a="regen">regenerate</button><button data-a="check" title="score each token of this reply for hallucination risk">check</button>';
    a.querySelector('[data-a=copy]').onclick=()=>navigator.clipboard.writeText(d.querySelector('.body').textContent);
    a.querySelector('[data-a=regen]').onclick=regenerate;
    a.querySelector('[data-a=check]').onclick=()=>checkReply(d);
    if(extra.check) showRisk(d.querySelector('.risk'),extra.check); }
  $('#log').appendChild(d); $('#log').scrollTop=1e9;
  return {root:d, think:d.querySelector('.think div'), thinkBox:d.querySelector('.think'), body:d.querySelector('.body'), stats:d.querySelector('.stats'), src:d.querySelector('.src'), tools:d.querySelector('.tools')};
}
function toolCard(box,t){
  let el=box.querySelector(`[data-k="${t.key}"]`);
  if(!el){ el=document.createElement('div'); el.className='tool'; el.dataset.k=t.key; box.appendChild(el); }
  const args=typeof t.arguments==='string'?t.arguments:JSON.stringify(t.arguments,null,1);
  const state=t.result?(t.result.ok?'<span class="okc">done</span>':'<span class="bad">failed</span>'):(t.needs_approval&&!t.decided?'<span class="warn">waiting for you</span>':'<span>running…</span>');
  el.innerHTML=`<span class="tn">🔧 ${esc(t.name)}</span> ${state}<details ${t.needs_approval&&!t.decided?'open':''}><summary>arguments</summary><pre>${esc(args)}</pre></details>`+
    (t.needs_approval&&!t.decided&&!t.result?`<button class="pri" data-a="1">Allow</button><button data-a="0">Deny</button>`:'')+
    (t.result?`<details><summary>result</summary><pre>${esc(t.result.text)}</pre></details>`:'');
  el.querySelectorAll('button[data-a]').forEach(b=>b.onclick=async()=>{ t.decided=true; toolCard(box,t); await post('/api/tools/approve',{key:t.key,allow:b.dataset.a==='1'}); });
}
function showSources(el,src){ el.innerHTML='sources: '+src.map((h,i)=>`<details><summary>[${i+1}] ${esc(h.source)}${h.page?' p.'+h.page:''}</summary><pre>${esc(h.text)}</pre></details>`).join(''); }
function renderChat(){
  $('#log').innerHTML='';
  if(!chat.messages.length){ $('#log').innerHTML='<div class="empty"><h2>Start a conversation</h2><p>Load a model from the Models tab, or point any OpenAI client at <code>'+esc(location.origin)+'/v1</code>. Requests name a model and it loads on demand.</p></div>'; return; }
  for(const m of chat.messages){ const b=bubble(m.role,m.content,{think:m.think,stats:m.stats,sources:m.sources,tools:m.tools,check:m.check}); b.root._msg=m; }
}
// ---------- per-reply check: token risk from the activation classifier
function showRisk(el,c){
  if(c.error){ el.innerHTML=`<div class="sum bad">check failed: ${esc(c.error)}</div>`; return; }
  const n=c.flagged.length, T=c.prob.length;
  const verdict=n?`<b class="bad">${n} of ${T} tokens flagged</b>`:`<b class="okc">no tokens flagged</b>`;
  el.innerHTML=`<div class="sum">risk: ${verdict} · peak ${(c.max*100).toFixed(0)}% · mean ${(c.mean*100).toFixed(0)}% `+
    `(flag at ${(c.threshold*100).toFixed(0)}%) · <a href="${esc(c.url)}" target="_blank" rel="noopener">open 3D view ↗</a>`+
    ` · <a href="#" data-a="tog">${n?'hide':'show'} tokens</a></div><div class="rt"${n?'':' hidden'}></div>`;
  // Shade each token by its risk; underline the ones over the threshold.
  // Shading starts at half the threshold, so tokens the classifier calls clean stay unshaded.
  el.querySelector('.rt').innerHTML=c.pieces.map((p,i)=>{ const r=c.prob[i]||0, a=Math.max(0,Math.min(1,(r-c.threshold/2)/(c.threshold/2)));
    return `<span class="${r>=c.threshold?'fl':''}" title="risk ${(r*100).toFixed(0)}%" style="background:color-mix(in srgb,var(--no) ${Math.round(a*(r>=c.threshold?50:30))}%,transparent)">${esc(p)}</span>`; }).join('');
  el.querySelector('[data-a=tog]').onclick=e=>{ e.preventDefault(); const rt=el.querySelector('.rt'); rt.hidden=!rt.hidden; e.target.textContent=(rt.hidden?'show':'hide')+' tokens'; };
}
async function checkReply(d){
  const m=d._msg; if(!m) return;
  const idx=chat.messages.indexOf(m), el=d.querySelector('.risk');
  el.innerHTML='<div class="sum">checking… (one extra pass over the reply)</div>';
  let c;
  try{ const r=await post('/api/trace',{model:m.model||$('#modelPick').value||undefined,messages:chat.messages.slice(0,idx).map(x=>({role:x.role,content:textOf(x.content)})),text:textOf(m.content)});
    c=await r.json(); if(!r.ok) c={error:c.error||r.statusText}; }catch(e){ c={error:String(e)}; }
  showRisk(el,c);
  if(!c.error){ m.check={id:c.id,pieces:c.pieces,prob:c.prob,flagged:c.flagged,threshold:c.threshold,max:c.max,mean:c.mean,url:c.url}; await persist(); }
}
try{ $('#autoCheck').checked=localStorage.getItem('ns_autocheck')==='1'; }catch{}
$('#autoCheck').onchange=()=>{ try{ localStorage.setItem('ns_autocheck',$('#autoCheck').checked?'1':'0'); }catch{} };

// ---------- images
$('#attach').onclick=()=>$('#file').click();
$('#file').onchange=async()=>{ for(const f of $('#file').files){ if(f.size>20e6){ alert(f.name+' is larger than 20 MB'); continue; }
    pendingImgs.push(await new Promise(r=>{const fr=new FileReader(); fr.onload=()=>r(fr.result); fr.readAsDataURL(f);})); }
  $('#file').value=''; drawPending(); };
function drawPending(){ $('#pending').innerHTML=pendingImgs.map((u,i)=>`<img src="${u}" title="click to remove" data-i="${i}">`).join('');
  $$('#pending img').forEach(im=>im.onclick=()=>{pendingImgs.splice(+im.dataset.i,1); drawPending();}); }

// ---------- send / stream
$('#in').onkeydown=e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); $('#f').requestSubmit(); } };
$('#f').onsubmit=async e=>{
  e.preventDefault();
  if(busy){ ctrl?.abort(); return; }
  const text=$('#in').value.trim(); if(!text&&!pendingImgs.length) return;
  const content = pendingImgs.length ? [{type:'text',text}, ...pendingImgs.map(u=>({type:'image_url',image_url:{url:u}}))] : text;
  $('#in').value=''; pendingImgs=[]; drawPending();
  chat.messages.push({role:'user',content}); bubble('user',content);
  await stream();
};
async function regenerate(){ if(busy) return; while(chat.messages.length && chat.messages.at(-1).role==='assistant') chat.messages.pop(); renderChat(); await stream(); }
async function stream(){
  busy=true; $('#send').textContent='Stop'; ctrl=new AbortController();
  const out=bubble('assistant','',{think:''}); let acc='', think='', t0=performance.now(), tFirst=0, usage=null, timings=null, routeNote='', sources=null, tools=[], servedBy=null;
  try{
    const r=await fetch('/api/chat',{method:'POST',signal:ctrl.signal,headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:chat.messages.map(m=>({role:m.role,content:m.content})),preset:$('#preset').value,model:$('#modelPick').value,
        rag:$('#ragPick').value?{collection:$('#ragPick').value,k:4}:undefined, tools:$('#toolsOn').checked||undefined})});
    if(!r.ok){ let e=await r.text(); try{e=JSON.parse(e).error||e}catch{} out.body.textContent='error: '+e; throw 0; }
    try{ const rt=JSON.parse(r.headers.get('X-NeuronScope-Route')||'null');
      if(rt) servedBy=rt.model;
      if(rt){ routeNote = rt.mode==='auto' ? `auto → ${rt.model}: ${rt.reason}` : `model: ${rt.model}${rt.mode==='manual'?' (chosen manually)':''}`;
        const m=models.find(x=>x.id===rt.model); if(m&&m.stats&&!m.stats.eligible) routeNote+=' ⚠ no performance stats'; } }catch{}
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){
      const {done,value}=await rd.read(); if(done) break;
      buf+=dec.decode(value,{stream:true}); let idx;
      while((idx=buf.indexOf('\n'))>=0){
        const line=buf.slice(0,idx).trim(); buf=buf.slice(idx+1);
        if(!line.startsWith('data:')) continue;
        const d=line.slice(5).trim(); if(d==='[DONE]') continue;
        let j; try{ j=JSON.parse(d) }catch{ continue }
        if(j.error){ out.body.textContent='error: '+(j.error.message||j.error); continue; }
        if(j.sources){ sources=j.sources; showSources(out.src,sources); continue; }
        if(j.tool_call){ tools.push(j.tool_call); toolCard(out.tools,j.tool_call); $('#log').scrollTop=1e9; continue; }
        if(j.tool_result){ const t=tools.find(x=>x.key===j.tool_result.key); if(t){ t.result=j.tool_result; t.decided=true; toolCard(out.tools,t); } continue; }
        if(j.usage) usage=j.usage; if(j.timings) timings=j.timings;
        const delta=j.choices?.[0]?.delta||{};
        if((delta.reasoning_content||delta.content) && !tFirst) tFirst=performance.now();
        if(delta.reasoning_content){ think+=delta.reasoning_content; out.think.textContent=think; }
        if(delta.content){ acc+=delta.content; out.body.innerHTML=render(acc); }
        $('#log').scrollTop=1e9;
      }
    }
  }catch(err){ if(err && err.name==='AbortError') out.body.innerHTML=render(acc+'\n[stopped]'); else if(err) out.body.textContent='error: '+err; }
  if(!think) out.thinkBox.remove();
  const secs=(performance.now()-(tFirst||t0))/1000;
  const ntok=usage?.completion_tokens ?? timings?.predicted_n;
  const tps=timings?.predicted_per_second ?? (ntok?ntok/secs:null);
  const stats=[ntok?`${ntok} tokens`:null, tps?`${tps.toFixed(1)} tok/s`:null, tFirst?`first token ${((tFirst-t0)/1000).toFixed(2)}s`:null].filter(Boolean).join(' · ');
  const fullStats=[routeNote,stats].filter(Boolean).join(' · ');
  out.stats.textContent=fullStats;
  let msg=null;
  if(acc||think){ msg={role:'assistant',content:acc,model:servedBy||undefined,think:think||undefined,stats:fullStats,sources:sources||undefined,tools:tools.length?tools:undefined};
    chat.messages.push(msg); out.root._msg=msg; await persist(); }
  status(); refresh();
  busy=false; $('#send').textContent='Send';
  if(msg&&acc&&$('#autoCheck').checked) checkReply(out.root);
}
$('#export').onclick=()=>{
  const md=`# ${chat.title}\n\n`+chat.messages.map(m=>`**${m.role}**\n\n${textOf(m.content)}${imagesOf(m.content).length?`\n\n_[${imagesOf(m.content).length} image(s)]_`:''}\n`).join('\n');
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([md],{type:'text/markdown'}));
  a.download=(chat.title||'chat').replace(/[^\w -]+/g,'_').slice(0,50)+'.md'; a.click(); URL.revokeObjectURL(a.href);
};

// ---------- hub
$('#go').onclick=async()=>{
  const q=$('#q').value.trim(); if(!q) return;
  $('#hits').textContent='searching…';
  const r=await (await fetch('/api/search?q='+encodeURIComponent(q))).json();
  if(r.error){ $('#hits').innerHTML='<span class="bad">'+esc(r.error)+'</span>'; return; }
  $('#hits').innerHTML=r.map(m=>`<div class="m" data-repo="${esc(m.repo)}"><div class="mn">${esc(m.repo)}</div>
    <div class="mm">${m.downloads.toLocaleString()} downloads ${m.gated?'<span class="tag warn">gated</span>':''}</div></div>`).join('');
  $$('#hits .m').forEach(el=>el.onclick=()=>files(el.dataset.repo));
};
async function files(repo){
  $('#hits').innerHTML='<div class="note">loading files…</div>';
  const r=await (await fetch('/api/files?repo='+encodeURIComponent(repo))).json();
  if(r.error){ $('#hits').innerHTML='<span class="bad">'+esc(r.error)+'</span>'; return; }
  $('#hits').innerHTML=`<div class="note"><b>${esc(repo)}</b> · click a file to download</div>`+r.map((g,i)=>
    `<div class="m" data-i="${i}"><div class="mn">${esc(g.name)}</div>
     <div class="mm">${g.quant?`<span class="tag">${esc(g.quant)}</span>`:''}${g.size?human(g.size):'size unknown'}
     ${g.parts.length>1?`<span class="tag">${g.parts.length} parts</span>`:''}</div></div>`).join('');
  $$('#hits .m[data-i]').forEach(el=>el.onclick=async()=>{ await post('/api/download',{repo,group:r[+el.dataset.i]}); jobs(); });
}
async function jobs(){
  const js=await (await fetch('/api/downloads')).json();
  $('#jobs').innerHTML = js.map(j=>{ const pct=j.total?Math.min(100,100*j.done/j.total):0;
    return `<div class="m"><div class="mn">${esc(j.name)}</div>
      <div class="mm">${esc(j.state)}${j.error?': <span class="bad">'+esc(j.error)+'</span>':''}${j.total?` · ${pct.toFixed(1)}% of ${human(j.total)}`:''}</div>
      <div style="height:4px;background:var(--line);border-radius:2px;margin-top:.3rem"><div style="height:4px;width:${pct}%;background:var(--acc);border-radius:2px"></div></div>
      ${j.state==='running'?`<button data-c="${esc(j.id)}" style="margin-top:.35rem;font-size:11px;padding:.15rem .5rem">Cancel</button>`:''}</div>`;
  }).join('');
  $$('#jobs button[data-c]').forEach(b=>b.onclick=async()=>{ await post('/api/download/cancel',{id:b.dataset.c}); jobs(); });
  if(js.some(j=>j.state==='running')) setTimeout(jobs,1200); else if(js.some(j=>j.state==='done')) refresh();
}
async function presets(){
  const p=await (await fetch('/api/presets')).json();
  $('#preset').innerHTML=Object.keys(p).map(k=>`<option>${esc(k)}</option>`).join('');
}
// ---------- docs (RAG)
let curColl=null;
async function colls(){
  const r=await (await fetch('/api/rag')).json(), cs=r.collections||[];
  const keep=$('#ragPick').value;
  $('#ragPick').innerHTML='<option value="">none</option>'+cs.map(c=>`<option ${c.name===keep?'selected':''}>${esc(c.name)}</option>`).join('');
  $('#colls').innerHTML=cs.length?cs.map(c=>`<div class="m" data-c="${esc(c.name)}"><div class="mn">${esc(c.name)}</div><div class="mm">${c.docs} files · ${c.chunks} passages${c.dense?' · <span class="tag">dense</span>':''}</div></div>`).join(''):'<div class="note">No collections yet.</div>';
  $$('#colls .m').forEach(el=>el.onclick=()=>openColl(el.dataset.c));
  return r;
}
async function openColl(name){ curColl=name; $('#collpanel').style.display=''; $('#collname').textContent=name; $('#raghits').textContent='';
  const ds=await (await fetch('/api/rag/docs?c='+encodeURIComponent(name))).json();
  $('#docs').innerHTML=(ds.length?ds:[]).map(d=>`<div class="m"><div class="mn">${esc(d.source)}</div><div class="mm">${d.chunks} passages</div><button class="x" data-d="${esc(d.doc)}" title="remove">×</button></div>`).join('')||'<div class="note">empty</div>';
  $$('#docs button[data-d]').forEach(b=>b.onclick=async()=>{ await post('/api/rag/delete',{collection:name,doc:b.dataset.d}); openColl(name); colls(); }); }
$('#mkcoll').onclick=async()=>{ const n=$('#newcoll').value.trim(); if(!n) return;
  const r=await post('/api/rag/create',{collection:n}); if(!r.ok){ alert((await r.json()).error); return; } $('#newcoll').value=''; await colls(); openColl(n); };
$('#adddocs').onclick=()=>$('#docfile').click();
$('#docfile').onchange=async()=>{ const fs=[...$('#docfile').files]; $('#docfile').value='';
  for(const f of fs){ if(f.size>32e6){ $('#docmsg').textContent=f.name+' is larger than 32 MB'; continue; }
    $('#docmsg').textContent='indexing '+f.name+'…';
    const data=await new Promise(r=>{const fr=new FileReader(); fr.onload=()=>r(fr.result); fr.readAsDataURL(f);});
    const r=await post('/api/rag/upload',{collection:curColl,filename:f.name,data}); const j=await r.json();
    $('#docmsg').textContent=r.ok?`${f.name}: ${j.chunks} passages`:`${f.name}: ${j.error}`; }
  openColl(curColl); colls(); };
$('#ragq').onkeydown=async e=>{ if(e.key!=='Enter') return; const r=await post('/api/rag/search',{collection:curColl,query:$('#ragq').value,k:4}); const j=await r.json();
  $('#raghits').innerHTML=r.ok?(j.map((h,i)=>`<details><summary>[${i+1}] ${esc(h.source)}${h.page?' p.'+h.page:''}</summary><pre style="white-space:pre-wrap">${esc(h.text)}</pre></details>`).join('')||'no match'):esc(j.error); };
$('#dropcoll').onclick=async()=>{ if(!confirm('Delete collection '+curColl+'?')) return; await post('/api/rag/delete',{collection:curColl}); $('#collpanel').style.display='none'; colls(); };

// ---------- GPUs
async function gpus(){ try{ const r=await (await fetch('/api/gpus')).json(); if(!r.nvidia||!r.count) return;
  $('#gpubox').style.display='';
  $('#gpuhint').textContent=`${r.count} × ${r.gpus[0].name}, ${(r.total_vram/2**30).toFixed(0)} GiB`+(r.multi_gpu?` · suggested: ${r.multi_gpu.llama_server}`:'');
  $('#gpuhint').title=(r.multi_gpu?r.multi_gpu.why+'\n\n':'')+(r.notes||[]).join('\n');
}catch{} }

// ---------- MCP tools
async function mcp(reload){
  const r=reload?await (await post('/api/mcp/reload',{})).json():await (await fetch('/api/mcp')).json();
  if(r.error){ $('#mcplist').innerHTML='<span class="bad">'+esc(r.error)+'</span>'; return; }
  $('#mcpcfg').innerHTML=r.config?'config: <code>'+esc(r.config)+'</code>':'No MCP config. Create <code>~/.neuronscope/mcp.json</code> (see docs/STUDIO.md) and restart, or pass --mcp-config.';
  const sv=r.servers||[]; const n=sv.reduce((a,x)=>a+(x.connected?x.tools.length:0),0);
  $('#toolsBox').style.display=n?'':'none';
  $('#mcplist').innerHTML=sv.map(x=>`<div class="m"><div class="mn">${esc(x.name)} <span class="tag">${esc(x.transport)}</span>${x.disabled?'<span class="tag">disabled</span>':x.connected?'<span class="tag okc">connected</span>':'<span class="tag bad">down</span>'}${x.auto_approve===true?'<span class="tag warn">auto-approve all</span>':''}</div>
    <div class="mm">${x.error?'<span class="bad">'+esc(x.error)+'</span>':x.tools.map(t=>`<div title="${esc(t.description)}">${esc(t.name)}${Array.isArray(x.auto_approve)&&x.auto_approve.includes(t.name)?' <span class="tag">auto</span>':''}</div>`).join('')}</div></div>`).join('');
}
$('#mcpreload').onclick=()=>mcp(true);

refresh(); status(); presets(); jobs(); loadChats(); renderChat(); colls(); mcp(); gpus(); setInterval(status,4000);
</script>"""


LOGIN = """<!DOCTYPE html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuronScope Studio</title>
<style>:root{color-scheme:dark}body{font:14px ui-sans-serif,system-ui,sans-serif;background:#111316;color:#e6e8eb;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
form{background:#181b20;border:1px solid #2a2f37;border-radius:10px;padding:1.6rem;width:320px}
h1{font-size:15px;margin:0 0 1rem}input{width:100%;padding:.5rem;border:1px solid #2a2f37;background:#20242b;color:#e6e8eb;
border-radius:6px;font:13px ui-monospace,monospace}
button{width:100%;margin-top:.7rem;padding:.5rem;border:1px solid #6ea8e0;background:#6ea8e0;
color:#0d1117;border-radius:6px;cursor:pointer;font:500 13px ui-sans-serif,system-ui}
.e{color:#ff8b7e;font-size:13px;margin-top:.5rem;min-height:1em}</style>
<form id="f"><h1>NeuronScope Studio</h1>
<input id="t" type="password" placeholder="access token" autofocus>
<button>Unlock</button><div class="e" id="e"></div></form>
<script>document.getElementById('f').onsubmit=async e=>{e.preventDefault();
const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({token:document.getElementById('t').value})});
if(r.ok) location.reload(); else document.getElementById('e').textContent='Incorrect token.';};
</script>"""

def main(argv=None):
    p = argparse.ArgumentParser(description="NeuronScope Studio: model manager, chat and OpenAI-compatible server")
    p.add_argument("--models-dir", action="append", default=[],
                   help="repeatable; searched recursively for .gguf")
    p.add_argument("--server", help="path to llama-server (or $NS_LLAMA_SERVER)")
    p.add_argument("--port", type=int, default=7870)
    p.add_argument("--host", default="127.0.0.1",
                   help="non-loopback binds require a token, and TLS unless --allow-plaintext")
    p.add_argument("--backend-port", type=int, default=8080, help="local port for llama-server")
    p.add_argument("--download-dir",
                   help="where hub downloads land (default: first --models-dir)")
    p.add_argument("--settings",
                   default=os.path.expanduser("~/.neuronscope/studio.json"))
    p.add_argument("--chats-dir", default=os.path.expanduser("~/.neuronscope/chats"))
    p.add_argument("--stats-dir", default=str(model_stats.DEFAULT_DIR),
                   help="rolling per-model stats (shared with testqa.py --record-stats)")
    p.add_argument("--stats-window", type=int, default=500, help="most recent observations per kind that count")
    p.add_argument("--min-graded", type=int, default=20,
                   help="graded results a model needs before model \"auto\" will consider it")
    p.add_argument("--hallucination-cost", type=float, default=1.0,
                   help="how much a wrong answer costs relative to a right one when auto ranks models")
    p.add_argument("--cett", default=os.environ.get("NS_CETT"),
                   help="llama-cett-dump binary; enables activation scoring for models with a classifier set")
    p.add_argument("--score-ngl", type=int, default=0,
                   help="GPU layers for activation scoring (0 keeps it off the GPU llama-server is using)")
    p.add_argument("--score-every", type=int, default=1, help="score one reply in N")
    p.add_argument("--traces-dir", default=os.path.expanduser("~/.neuronscope/traces"),
                   help="where per-reply checks are saved (open them in the 3D view or timeline.py)")
    p.add_argument("--idle-ttl", type=int, default=0, help="unload the model after N idle seconds (0 = never)")
    p.add_argument("--no-jit", action="store_true", help="/v1 requests never load or swap models")
    p.add_argument("--jobs-dir", default=os.path.expanduser("~/.neuronscope/jobs"),
                   help="logs and state of evaluation/retraining jobs started from /jobs")
    p.add_argument("--max-jobs", type=int, default=2, help="jobs that may run at once")
    p.add_argument("--allow-remote-jobs", action="store_true",
                   help="allow /jobs on a non-loopback bind (jobs can train models and run model-written code)")
    p.add_argument("--no-jobs", action="store_true", help="disable /jobs entirely")
    p.add_argument("--devices", default=os.path.expanduser("~/.neuronscope/devices.json"),
                   help="paired devices (token hashes only)")
    p.add_argument("--pair-code-ttl", default="5m",
                   help="how long a pairing link can be claimed (e.g. 90s, 5m, 1h; max 1d)")
    p.add_argument("--pair-durations", default="1h,8h,1d,7d,30d",
                   help="temporary-access durations offered when pairing")
    p.add_argument("--pair-max", default="90d", help="longest temporary access a pairing may grant")
    p.add_argument("--pair-default", default="persistent",
                   help="what a pairing grants unless chosen otherwise: persistent, or a duration like 8h")
    p.add_argument("--no-persistent-pairing", action="store_true",
                   help="only grant temporary access; every paired device expires")
    p.add_argument("--links", default=os.path.expanduser("~/.neuronscope/links.json"),
                   help="linked hosts whose models this Studio serves (holds their device tokens; 0600)")
    p.add_argument("--mcp-config", default=str(ns_mcp.DEFAULT_CONFIG),
                   help="MCP servers for chat tools (mcpServers JSON, as in LM Studio / Claude Desktop)")
    p.add_argument("--rag-dir", default=os.path.expanduser("~/.neuronscope/rag"), help="document collections")
    p.add_argument("--rag-embed", help="URL[@model] of an OpenAI-compatible embeddings endpoint for dense retrieval")
    p.add_argument("--rag-embed-gguf", help="embedding GGUF; Studio runs it with llama-server --embedding on demand")
    p.add_argument("--rag-embed-ngl", type=int, default=0, help="GPU layers for the embedding model")
    sec.add_server_security_args(p)
    a = p.parse_args(argv)

    STATE["models_dirs"] = [os.path.expanduser(d) for d in a.models_dir] or [
        os.path.expanduser("~/.lmstudio/models"),
        os.path.expanduser("~/.cache/lm-studio/models"),
    ]
    STATE["server_bin"] = a.server or os.environ.get("NS_LLAMA_SERVER")
    STATE["settings_path"] = os.path.expanduser(a.settings)
    STATE["chats_dir"] = os.path.expanduser(a.chats_dir)
    STATE["traces_dir"] = os.path.expanduser(a.traces_dir)
    os.environ.setdefault(sec.TOKEN_ENV, os.environ.get("NS_STUDIO_TOKEN", ""))
    STATE["token"] = sec.resolve_token(a.token, a.token_file) or None
    STATE["download_dir"] = (os.path.expanduser(a.download_dir)
                             if a.download_dir else None)
    STATE["idle_ttl"] = max(0, a.idle_ttl)
    STATE["jit"] = not a.no_jit
    STATE["backend_port"] = a.backend_port
    STATE["stats"] = model_stats.StatsStore(a.stats_dir, a.stats_window)
    STATE.update(min_graded=a.min_graded, halluc_cost=a.hallucination_cost, score_ngl=a.score_ngl,
                 score_every=a.score_every,
                 cett=a.cett if a.cett and os.path.exists(a.cett) else None)
    tls = bool(a.tls_cert and a.tls_key)
    STATE["tls"] = tls
    try:
        for w in sec.check_bind(a.host, STATE["token"] or "", tls=tls,
                                allow_plaintext=a.allow_plaintext):
            print("warning:", w)
    except sec.SecurityConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        policy = ns_pairing.PairingPolicy(a.pair_code_ttl, a.pair_durations.split(","), a.pair_max,
                                          not a.no_persistent_pairing, a.pair_default)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    STATE.update(devices=ns_pairing.DeviceRegistry(a.devices, policy), links_path=os.path.expanduser(a.links),
                 fingerprint=ns_pairing.cert_fingerprint(a.tls_cert) if a.tls_cert and a.tls_key else None)
    STATE.update(mcp_config=a.mcp_config if os.path.exists(os.path.expanduser(a.mcp_config)) else None, mcp=None)
    STATE.update(rag_dir=a.rag_dir, rag_embed=a.rag_embed, rag_embed_gguf=a.rag_embed_gguf,
                 rag_embed_ngl=a.rag_embed_ngl, rag=None)
    if a.no_jobs:
        STATE["jobs"], STATE["jobs_off"] = None, "jobs are disabled (--no-jobs)"
    elif not sec.is_loopback(a.host) and not a.allow_remote_jobs:
        STATE["jobs"] = None
        STATE["jobs_off"] = ("jobs are off on a network bind: they can train models and run model-written code. "
                             "Restart Studio with --allow-remote-jobs to enable them.")
    else:
        STATE["jobs"] = ns_jobs.JobRunner(a.jobs_dir, a.max_jobs)

    scheme = "https" if tls else "http"
    print(f"NeuronScope Studio on {scheme}://{a.host}:{a.port}")
    notes = ["JIT " + ("on" if STATE["jit"] else "off")]
    if STATE["idle_ttl"]:
        notes.append(f"idle TTL {STATE['idle_ttl']}s")
    notes.append(f'model "auto" over models with >= {STATE["min_graded"]} graded results')
    if STATE["cett"]:
        notes.append("activation scoring available")
    print(f"OpenAI-compatible API: {scheme}://{a.host}:{a.port}/v1  ({', '.join(notes)})")
    print("model dirs:")
    for d in STATE["models_dirs"]:
        print(f"  {d}{'' if os.path.isdir(d) else '   (missing)'}")
    if not STATE["server_bin"]:
        print("\nno --server given: models can be listed but not loaded")
    elif not os.path.exists(STATE["server_bin"]):
        print(f"\n!! {STATE['server_bin']} does not exist")
    if STATE["token"]:
        print("authentication enabled")

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    if tls:
        srv.socket = sec.server_ssl_context(a.tls_cert, a.tls_key).wrap_socket(
            srv.socket, server_side=True, do_handshake_on_connect=False)
    threading.Thread(target=idle_reaper, daemon=True).start()
    threading.Thread(target=score_worker, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_server()
        if EMBED["proc"] is not None and EMBED["proc"].poll() is None:
            EMBED["proc"].terminate()
        if STATE.get("mcp") is not None:
            STATE["mcp"].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
