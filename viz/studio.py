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

No authentication. --host 0.0.0.0 puts model loading and chat on your network;
trusted networks only.
"""

import argparse
import glob
import hashlib
import hmac
import http.cookies
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))
try:
    from hostcheck import Host, check as host_check
except ImportError:
    Host, host_check = None, None

STATE = {"models_dirs": [], "server_bin": None, "settings_path": None,
         "download_dir": None, "offline": False, "token": None}
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
            f = r.fields.get(key)
            if f is None:
                return None
            try:
                return f.parts[f.data[0]][0].item()
            except Exception:
                try:
                    return bytes(f.parts[f.data[0]]).decode()
                except Exception:
                    return None

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


def scan_models():
    """Every .gguf under the configured directories, with cached metadata.

    Multi-part files (`-00001-of-0000N`) are listed once, by their first part,
    which is what llama.cpp wants passed to -m.
    """
    found, seen = [], set()
    for root in STATE["models_dirs"]:
        for path in sorted(glob.glob(os.path.join(root, "**", "*.gguf"),
                                     recursive=True)):
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
            found.append({
                "path": path, "name": base, "size": size,
                "publisher": os.path.basename(os.path.dirname(
                    os.path.dirname(path))),
                **meta,
            })
    return found


def fit_estimate(size_bytes):
    """Will this load, on this machine? Reuses the shared host checks."""
    if Host is None:
        return {"ok": None, "note": "hostcheck unavailable"}
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
            "draft_model": "", "draft_max": 16, "draft_min": 4}

# A named config is a complete, reusable setup: model, load settings, preset,
# visualizer and hardware limits, under a name. The name is also what the model
# is served as, so Cline sees "ornith-suppressed" rather than a filename.
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
        # A config can name the machine it is for, so you can check the
        # friend's rig from the laptop. Falls back to whatever this box has.
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
    "ornith recommended": {"temperature": 0.6, "top_p": 0.95},
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
    if settings.get("served_name"):
        # What Cline and every other client sees in the model list.
        cmd += ["--alias", settings["served_name"]]
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
    if settings.get("extra"):
        cmd += settings["extra"].split()

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
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


def proxy(path, payload, stream_to=None):
    """Forward to the running llama-server. Streams SSE when asked.

    Streaming matters here: on an iGPU at a few tokens per second, a
    non-streaming chat window looks indistinguishable from a hang.
    """
    if not server_running():
        raise RuntimeError("no model loaded")
    url = f"http://127.0.0.1:{PROC['port']}{path}"
    req = urllib.request.Request(
        url, method="POST", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    if stream_to is None:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            stream_to.write(raw)
            stream_to.flush()
    return None


# ---------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # -- auth ---------------------------------------------------------------
    # This endpoint loads models, downloads files and runs inference. On a LAN
    # that is not something to leave open, so --token gates everything except
    # the login page itself.
    def _authed(self):
        tok = STATE["token"]
        if not tok:
            return True
        hdr = self.headers.get("Authorization", "")
        if hdr.startswith("Bearer ") and hmac.compare_digest(hdr[7:], tok):
            return True
        raw = self.headers.get("Cookie", "")
        if raw:
            try:
                c = http.cookies.SimpleCookie(raw)
                if "ns_token" in c and hmac.compare_digest(
                        c["ns_token"].value, tok):
                    return True
            except Exception:
                pass
        return False

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
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if not self._authed():
            return self._deny()
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/models":
            ms = scan_models()
            st = load_settings()
            for m in ms:
                m["settings"] = {**DEFAULTS, **st.get(_key(m["path"]), {})}
                m["fit"] = fit_estimate(m["size"])
            return self._json(200, ms)
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
            })
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/login":
            req = self._read()
            if STATE["token"] and hmac.compare_digest(
                    str(req.get("token", "")), STATE["token"]):
                body = json.dumps({"ok": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Set-Cookie",
                                 f"ns_token={STATE['token']}; Path=/; "
                                 "HttpOnly; SameSite=Strict; Max-Age=604800")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(401, {"error": "bad token"})
            return
        if not self._authed():
            return self._deny()
        try:
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
                req = self._read()
                preset = load_presets().get(req.get("preset", "default"),
                                            PRESET_DEFAULTS)
                msgs = list(req["messages"])
                if preset.get("system") and not (
                        msgs and msgs[0].get("role") == "system"):
                    msgs.insert(0, {"role": "system",
                                    "content": preset["system"]})
                payload = {"messages": msgs,
                           "temperature": preset.get("temperature", 0.7),
                           "top_p": preset.get("top_p", 0.95),
                           "max_tokens": preset.get("max_tokens", 2048),
                           "stream": True}
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
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    proxy("/v1/chat/completions", payload, stream_to=self.wfile)
                except Exception as e:
                    self.wfile.write(
                        f"data: {json.dumps({'error': str(e)})}\n\n".encode())
                return

            self._json(404, {"error": "not found"})
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._json(500, {"error": str(e)[:300]})
            except Exception:
                pass


PAGE = r"""<!DOCTYPE html><meta charset="utf-8"><title>NeuronScope Studio</title>
<style>
:root{--bg:#faf9f7;--fg:#1c1c1a;--mut:#6b6b64;--line:#e3e2dd;--ok:#2f7d4f;--no:#b23c2e;--warn:#a8701c;--acc:#2c5f8a}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
header{padding:.7rem 1rem;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:1rem;background:#fff}
h1{margin:0;font-size:15px;font-weight:600}
.status{font-size:13px;color:var(--mut)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#ccc;margin-right:.35rem}
.dot.on{background:var(--ok)}
main{flex:1;display:grid;grid-template-columns:340px 1fr;min-height:0}
aside{border-right:1px solid var(--line);overflow-y:auto;padding:.8rem;background:#fff}
section.chat{display:flex;flex-direction:column;min-height:0}
.m{border:1px solid var(--line);border-radius:8px;padding:.55rem .7rem;margin-bottom:.45rem;cursor:pointer;background:#fff}
.m:hover{background:#f4f3ef}.m.sel{border-color:var(--acc);background:#f0f5fa}
.mn{font-weight:500;font-size:13px;word-break:break-all}
.mm{font-size:12px;color:var(--mut);margin-top:.15rem}
.tag{display:inline-block;font-size:11px;padding:.05rem .35rem;border:1px solid var(--line);border-radius:4px;margin-right:.25rem}
.bad{color:var(--no)}.warn{color:var(--warn)}
label{display:block;font-size:12px;color:var(--mut);margin:.45rem 0 .1rem}
input,select{font:12px ui-monospace,monospace;padding:.3rem .4rem;border:1px solid var(--line);border-radius:5px;width:100%;background:#fff}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}
button{font:500 13px ui-sans-serif,system-ui;padding:.45rem .9rem;border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer}
button:hover{background:#f2f1ec}button:disabled{opacity:.45;cursor:default}
button.pri{background:var(--acc);color:#fff;border-color:var(--acc)}
#log{flex:1;overflow-y:auto;padding:1rem 1.2rem}
.msg{max-width:760px;margin:0 auto 1rem}
.who{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin-bottom:.2rem}
.body{white-space:pre-wrap;word-wrap:break-word}
.think{color:var(--mut);font-size:13px;border-left:2px solid var(--line);padding-left:.7rem;margin-bottom:.5rem}
form{border-top:1px solid var(--line);padding:.7rem 1.2rem;display:flex;gap:.5rem;background:#fff}
#in{flex:1;font:14px ui-sans-serif,system-ui;padding:.5rem .6rem;border:1px solid var(--line);border-radius:6px;resize:none}
.dials{border-top:1px solid var(--line);padding:.6rem 1.2rem;font-size:12px;color:var(--mut);display:flex;gap:1.2rem;align-items:center;background:#fff}
.dials input[type=range]{width:150px}
.note{font-size:12px;color:var(--mut);margin:.5rem 0}
</style>
<header>
  <h1>NeuronScope Studio</h1>
  <div class="status"><span id="dot" class="dot"></span><span id="st">no model loaded</span></div>
  <button id="unload" style="margin-left:auto" disabled>Unload</button>
</header>
<main>
<aside>
  <div style="display:flex;gap:.3rem;margin-bottom:.6rem">
    <button id="tabLocal" class="pri" style="flex:1">Local</button>
    <button id="tabHub" style="flex:1">Hub</button>
  </div>
  <div id="hub" style="display:none">
    <input id="q" placeholder="search GGUF models on Hugging Face">
    <button id="go" style="width:100%;margin-top:.4rem">Search</button>
    <div id="hits" class="note"></div>
    <div id="jobs"></div>
  </div>
  <div id="local">
  <div id="list">scanning…</div>
  <div id="panel" style="display:none">
    <hr style="border:none;border-top:1px solid var(--line);margin:.8rem 0">
    <div class="row2">
      <div><label>GPU layers</label><input id="ngl" value="99"></div>
      <div><label>Context</label><input id="ctx" value="8192"></div>
    </div>
    <div class="row2">
      <div><label>Batch</label><input id="batch" value="2048"></div>
      <div><label>Threads (0=auto)</label><input id="threads" value="0"></div>
    </div>
    <div id="moebox" style="display:none">
      <label>Active experts <span id="moehint" class="note"></span></label>
      <input id="experts" value="0">
    </div>
    <label>LoRA adapter (optional)</label><input id="lora" placeholder="/path/suppress-lora.gguf">
    <label>Draft model (speculative decoding)</label>
    <input id="draft_model" placeholder="/path/small-Q4_K_M.gguf">
    <label>Extra llama-server flags</label><input id="extra" placeholder="--override-tensor '\.ffn_.*_exps\.=CPU'">
    <button id="load" class="pri" style="margin-top:.7rem;width:100%">Load</button>
    <div id="loadmsg" class="note"></div>
  </div>
  </div>
</aside>
<section class="chat">
  <div id="log"></div>
  <div class="dials">
    <span>preset</span><select id="preset" style="width:auto"></select>
    <span>suppression α</span>
    <input type="range" id="alpha" min="0" max="1" step="0.05" value="1" disabled>
    <span id="av">1.00</span>
    <span id="adesc">no adapter loaded</span>
  </div>
  <form id="f"><textarea id="in" rows="2" placeholder="Message…"></textarea>
    <button class="pri" id="send">Send</button></form>
</section>
</main>
<script>
const $=s=>document.querySelector(s); let models=[], sel=null, busy=false;
const F=["ngl","ctx","batch","threads","experts","lora","draft_model","extra"];

function human(b){const u=["B","KB","MB","GB"];let i=0;while(b>1024&&i<3){b/=1024;i++}return b.toFixed(1)+u[i]}

async function refresh(){
  models=await (await fetch('/api/models')).json();
  $('#list').innerHTML = models.length ? models.map((m,i)=>{
    const f=m.fit||{}; const cls=f.ok===false?'bad':(/tight/.test(f.note||'')?'warn':'');
    return `<div class="m" data-i="${i}">
      <div class="mn">${m.name}</div>
      <div class="mm">
        ${m.quant?`<span class="tag">${m.quant}</span>`:''}
        ${m.arch?`<span class="tag">${m.arch}</span>`:''}
        ${m.n_experts?`<span class="tag">MoE ${m.n_experts_used||'?'}/${m.n_experts}</span>`:''}
        <span class="tag">${human(m.size)}</span>
      </div>
      <div class="mm ${cls}">${f.note||''}</div></div>`;
  }).join('') : '<div class="note">No .gguf files found under the configured directories.</div>';
  document.querySelectorAll('.m').forEach(el=>el.onclick=()=>pick(+el.dataset.i));
}
function pick(i){
  sel=models[i];
  document.querySelectorAll('.m').forEach((e,j)=>e.classList.toggle('sel',j===i));
  $('#panel').style.display='block';
  const s=sel.settings||{};
  F.forEach(k=>{ if($('#'+k)) $('#'+k).value = s[k] ?? ''; });
  const moe = !!sel.n_experts;
  $('#moebox').style.display = moe?'block':'none';
  if(moe) $('#moehint').textContent = `0 = model default (${sel.n_experts_used} of ${sel.n_experts})`;
}
$('#load').onclick=async()=>{
  if(!sel) return;
  const s={}; F.forEach(k=>{ const el=$('#'+k); if(!el) return;
    s[k] = ["lora","extra","draft_model"].includes(k) ? el.value : (parseFloat(el.value)||0); });
  s.flash_attn=true; s.lora_scale=1.0;
  $('#load').disabled=true; $('#loadmsg').textContent='starting…';
  const r=await (await fetch('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:sel.path,settings:s})})).json();
  $('#load').disabled=false;
  $('#loadmsg').innerHTML = r.ok?'':'<span class="bad">'+(r.message||'failed').replace(/</g,'&lt;')+'</span>';
  status();
};
$('#unload').onclick=async()=>{ await fetch('/api/unload',{method:'POST'}); status(); };
async function status(){
  const s=await (await fetch('/api/status')).json();
  $('#dot').className='dot'+(s.running?' on':'');
  $('#st').textContent = s.running ? `${s.model} · port ${s.port} · ${s.uptime}s` : 'no model loaded';
  $('#unload').disabled=!s.running;
  const hasLora=!!s.lora;
  $('#alpha').disabled=!hasLora;
  $('#adesc').textContent = hasLora ? s.lora.split('/').pop() : 'no adapter loaded';
  if(hasLora){ $('#alpha').value=s.lora_scale; $('#av').textContent=(+s.lora_scale).toFixed(2); }
}
$('#alpha').oninput=async e=>{
  const v=parseFloat(e.target.value); $('#av').textContent=v.toFixed(2);
  await fetch('/api/lora',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({scale:v})});
};
let history=[];
function bubble(who,text,think){
  const d=document.createElement('div'); d.className='msg';
  d.innerHTML=`<div class="who">${who}</div>`+(think?`<div class="think"></div>`:'')+`<div class="body"></div>`;
  $('#log').appendChild(d); $('#log').scrollTop=1e9;
  return {think:d.querySelector('.think'), body:d.querySelector('.body')};
}
$('#f').onsubmit=async e=>{
  e.preventDefault(); if(busy) return;
  const text=$('#in').value.trim(); if(!text) return;
  $('#in').value=''; bubble('you',text).body.textContent=text;
  history.push({role:'user',content:text});
  busy=true; $('#send').disabled=true;
  const out=bubble('assistant','',true); let acc='', inThink=false;
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:history,preset:$('#preset').value})});
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){
      const {done,value}=await rd.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      let idx;
      while((idx=buf.indexOf('\n'))>=0){
        const line=buf.slice(0,idx).trim(); buf=buf.slice(idx+1);
        if(!line.startsWith('data:')) continue;
        const d=line.slice(5).trim(); if(d==='[DONE]') continue;
        let j; try{ j=JSON.parse(d) }catch{ continue }
        if(j.error){ out.body.textContent='error: '+j.error; break; }
        const delta=j.choices?.[0]?.delta||{};
        const piece=(delta.reasoning_content||'')+(delta.content||'');
        if(delta.reasoning_content){ out.think.textContent+=delta.reasoning_content; }
        if(delta.content){ acc+=delta.content; out.body.textContent=acc; }
        $('#log').scrollTop=1e9;
      }
    }
  }catch(err){ out.body.textContent='error: '+err; }
  if(acc) history.push({role:'assistant',content:acc});
  busy=false; $('#send').disabled=false;
};

$('#tabLocal').onclick=()=>{$('#local').style.display='block';$('#hub').style.display='none';
  $('#tabLocal').className='pri';$('#tabHub').className='';};
$('#tabHub').onclick=()=>{$('#local').style.display='none';$('#hub').style.display='block';
  $('#tabHub').className='pri';$('#tabLocal').className='';};

$('#go').onclick=async()=>{
  const q=$('#q').value.trim(); if(!q) return;
  $('#hits').textContent='searching…';
  const r=await (await fetch('/api/search?q='+encodeURIComponent(q))).json();
  if(r.error){ $('#hits').innerHTML='<span class="bad">'+r.error+'</span>'; return; }
  $('#hits').innerHTML=r.map(m=>`<div class="m" data-repo="${m.repo}">
    <div class="mn">${m.repo}</div>
    <div class="mm">${m.downloads.toLocaleString()} downloads
      ${m.gated?'<span class="tag warn">gated</span>':''}</div></div>`).join('');
  document.querySelectorAll('#hits .m').forEach(el=>el.onclick=()=>files(el.dataset.repo));
};
async function files(repo){
  $('#hits').innerHTML='<div class="note">loading files…</div>';
  const r=await (await fetch('/api/files?repo='+encodeURIComponent(repo))).json();
  if(r.error){ $('#hits').innerHTML='<span class="bad">'+r.error+'</span>'; return; }
  $('#hits').innerHTML=`<div class="note"><b>${repo}</b></div>`+r.map((g,i)=>
    `<div class="m" data-i="${i}"><div class="mn">${g.name}</div>
     <div class="mm">${g.quant?`<span class="tag">${g.quant}</span>`:''}
     ${g.size?human(g.size):'size unknown'}
     ${g.parts.length>1?`<span class="tag">${g.parts.length} parts</span>`:''}</div></div>`
  ).join('');
  document.querySelectorAll('#hits .m[data-i]').forEach(el=>el.onclick=async()=>{
    const g=r[+el.dataset.i];
    await fetch('/api/download',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({repo,group:g})});
    jobs();
  });
}
async function jobs(){
  const js=await (await fetch('/api/downloads')).json();
  $('#jobs').innerHTML = js.length ? js.map(j=>{
    const pct = j.total ? Math.min(100,100*j.done/j.total) : 0;
    return `<div class="m"><div class="mn">${j.name}</div>
      <div class="mm">${j.state}${j.error?': <span class="bad">'+j.error+'</span>':''}
      ${j.total?` · ${pct.toFixed(1)}% of ${human(j.total)}`:''}</div>
      <div style="height:4px;background:var(--line);border-radius:2px;margin-top:.3rem">
        <div style="height:4px;width:${pct}%;background:var(--acc);border-radius:2px"></div></div>
      ${j.state==='running'?`<button data-c="${j.id}" style="margin-top:.35rem;font-size:11px;padding:.15rem .5rem">Cancel</button>`:''}
      </div>`;
  }).join('') : '';
  document.querySelectorAll('#jobs button[data-c]').forEach(b=>b.onclick=async()=>{
    await fetch('/api/download/cancel',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({id:b.dataset.c})}); jobs();
  });
  if(js.some(j=>j.state==='running')) setTimeout(jobs,1200);
  else if(js.some(j=>j.state==='done')) refresh();
}
async function presets(){
  const p=await (await fetch('/api/presets')).json();
  $('#preset').innerHTML=Object.keys(p).map(k=>`<option>${k}</option>`).join('');
}
refresh(); status(); presets(); jobs(); setInterval(status,4000);
</script>"""



LOGIN = """<!DOCTYPE html><meta charset="utf-8"><title>NeuronScope Studio</title>
<style>body{font:14px ui-sans-serif,system-ui,sans-serif;background:#faf9f7;color:#1c1c1a;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
form{background:#fff;border:1px solid #e3e2dd;border-radius:10px;padding:1.6rem;width:320px}
h1{font-size:15px;margin:0 0 1rem}input{width:100%;padding:.5rem;border:1px solid #e3e2dd;
border-radius:6px;font:13px ui-monospace,monospace}
button{width:100%;margin-top:.7rem;padding:.5rem;border:1px solid #2c5f8a;background:#2c5f8a;
color:#fff;border-radius:6px;cursor:pointer;font:500 13px ui-sans-serif,system-ui}
.e{color:#b23c2e;font-size:13px;margin-top:.5rem;min-height:1em}</style>
<form id="f"><h1>NeuronScope Studio</h1>
<input id="t" type="password" placeholder="access token" autofocus>
<button>Unlock</button><div class="e" id="e"></div></form>
<script>document.getElementById('f').onsubmit=async e=>{e.preventDefault();
const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({token:document.getElementById('t').value})});
if(r.ok) location.reload(); else document.getElementById('e').textContent='Incorrect token.';};
</script>"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", action="append", default=[],
                   help="repeatable; searched recursively for .gguf")
    p.add_argument("--server", help="path to llama-server")
    p.add_argument("--port", type=int, default=7870)
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 exposes model loading and chat to your LAN "
                        "with no authentication")
    p.add_argument("--token",
                   help="require this token. Without it the server is open to "
                        "anyone who can reach the port.")
    p.add_argument("--download-dir",
                   help="where hub downloads land (default: first --models-dir)")
    p.add_argument("--settings",
                   default=os.path.expanduser("~/.neuronscope/studio.json"))
    a = p.parse_args()

    STATE["models_dirs"] = [os.path.expanduser(d) for d in a.models_dir] or [
        os.path.expanduser("~/.lmstudio/models"),
        os.path.expanduser("~/.cache/lm-studio/models"),
    ]
    STATE["server_bin"] = a.server or os.environ.get("NS_LLAMA_SERVER")
    STATE["settings_path"] = os.path.expanduser(a.settings)
    STATE["token"] = a.token or os.environ.get("NS_STUDIO_TOKEN")
    STATE["download_dir"] = (os.path.expanduser(a.download_dir)
                             if a.download_dir else None)

    print(f"NeuronScope Studio on http://{a.host}:{a.port}")
    print("model dirs:")
    for d in STATE["models_dirs"]:
        print(f"  {d}{'' if os.path.isdir(d) else '   (missing)'}")
    if not STATE["server_bin"]:
        print("\nno --server given: models can be listed but not loaded")
    elif not os.path.exists(STATE["server_bin"]):
        print(f"\n!! {STATE['server_bin']} does not exist")
    if a.host == "0.0.0.0" and not STATE["token"]:
        print("\n!! exposed to the LAN with NO AUTHENTICATION.")
        print("   Anyone who can reach this port can load models, download")
        print("   files and run inference. Pass --token to gate it.")
    elif STATE["token"]:
        print("\nauthentication enabled")

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_server()


if __name__ == "__main__":
    main()
