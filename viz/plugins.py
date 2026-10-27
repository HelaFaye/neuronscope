"""Studio plugins: folders that add job kinds, reply checks, a page and MCP tools.

A plugin is a folder with a plugin.json (docs/PLUGINS.md):

    {"name": "word-count", "version": "1.0", "description": "...",
     "jobs":   [{"kind": "count", "title": "...", "script": "count.py", "fields": [...]}],
     "checks": [{"name": "length", "title": "...", "script": "check.py"}],
     "page":   "page",
     "mcp":    {"command": "python", "args": ["server.py"]},
     "nodes":  [{"type": "word_count", "title": "...", "script": "node.py", "inputs": [["text", "text", true]],
                 "outputs": [["count", "number"]], "params": [{"name": "min", "kind": "int", "default": 1}]}]}

Plugins run code, so none is used until the owner turns it on in Settings ->
Plugins. Turning one on records a SHA-256 of its files; if they change
afterwards, the plugin is off again until it is reviewed and turned on anew.
Scripts run as separate processes with the plugin's folder as working directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
FIELD_KINDS = ("str", "int", "float", "bool")
CHECK_TIMEOUT = 60


class PluginError(ValueError):
    pass


def _inside(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if p != root.resolve() and root.resolve() not in p.parents:
        raise PluginError(f"{rel}: outside the plugin's folder")
    return p


def digest(folder: Path) -> str:
    """SHA-256 over every file's path and content (caches and VCS folders aside)."""
    h = hashlib.sha256()
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder).as_posix()
        if not p.is_file() or any(part in ("__pycache__", ".git") for part in p.relative_to(folder).parts):
            continue
        h.update(rel.encode() + b"\0")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def read_manifest(folder: Path) -> dict:
    """plugin.json, checked. Raises PluginError with the reason."""
    try:
        m = json.loads((folder / "plugin.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PluginError(f"plugin.json: {e}") from None
    if not isinstance(m, dict) or not NAME.match(str(m.get("name", ""))):
        raise PluginError("plugin.json needs a name: lower-case letters, digits, - and _")
    for j in m.get("jobs") or []:
        if not NAME.match(str(j.get("kind", ""))) or not j.get("script"):
            raise PluginError(f"job {j.get('kind')!r}: needs a kind (like a name) and a script")
        _inside(folder, j["script"])
        for f in j.get("fields") or []:
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", str(f.get("name", ""))):
                raise PluginError(f"job {j['kind']}: bad field name {f.get('name')!r}")
            if f.get("kind", "str") not in FIELD_KINDS:
                raise PluginError(f"job {j['kind']}: field {f['name']}: kind must be one of {FIELD_KINDS}")
    for c in m.get("checks") or []:
        if not NAME.match(str(c.get("name", ""))) or not c.get("script"):
            raise PluginError(f"check {c.get('name')!r}: needs a name and a script")
        _inside(folder, c["script"])
    for n in m.get("nodes") or []:
        if not NAME.match(str(n.get("type", ""))) or not n.get("script"):
            raise PluginError(f"node {n.get('type')!r}: needs a type (like a name) and a script")
        _inside(folder, n["script"])
        for port in (n.get("inputs") or []) + (n.get("outputs") or []):
            if not (isinstance(port, list) and port and re.fullmatch(r"[a-z][a-z0-9_]{0,30}", str(port[0]))):
                raise PluginError(f"node {n['type']}: ports are [name, type(, required)]")
    if m.get("page"):
        _inside(folder, m["page"])
    if m.get("mcp") is not None and not (isinstance(m["mcp"], dict) and (m["mcp"].get("command") or m["mcp"].get("url"))):
        raise PluginError("mcp: needs a command (stdio) or a url")
    return m


class Plugins:
    """The plugins found in some folders, and which of them the owner turned on."""

    def __init__(self, dirs: list, state_path: str | Path):
        self.dirs = [Path(d).expanduser() for d in dirs if d]
        self.state_path = Path(state_path).expanduser()
        self.lock = threading.Lock()

    # -- state: name -> {"sha256": digest when turned on}
    def _state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, st: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(self.state_path)

    def scan(self) -> list[dict]:
        """Every plugin folder: manifest, folder, digest, status (on / off / changed / broken)."""
        st, out, seen = self._state(), [], set()
        for d in self.dirs:
            if not d.is_dir():
                continue
            for folder in sorted(p for p in d.iterdir() if (p / "plugin.json").is_file()):
                rec = {"folder": str(folder)}
                try:
                    m = read_manifest(folder)
                except PluginError as e:
                    out.append({**rec, "name": folder.name, "status": "broken", "error": str(e)})
                    continue
                if m["name"] in seen:
                    out.append({**rec, "name": m["name"], "status": "broken",
                                "error": "another plugin of this name was found first"})
                    continue
                seen.add(m["name"])
                sha = digest(folder)
                on = st.get(m["name"])
                status = "off" if not on else ("on" if on.get("sha256") == sha and on.get("folder") == str(folder)
                                                 else "changed")
                out.append({**rec, "name": m["name"], "manifest": m, "sha256": sha, "status": status})
        return out

    def enabled(self) -> list[dict]:
        return [p for p in self.scan() if p["status"] == "on"]

    def get(self, name: str, need_on: bool = True) -> dict:
        p = next((x for x in self.scan() if x["name"] == name), None)
        if p is None:
            raise PluginError(f"no plugin {name!r}")
        if need_on and p["status"] != "on":
            raise PluginError(f"plugin {name} is {p['status']}: turn it on in Settings -> Plugins")
        return p

    def set_enabled(self, name: str, on: bool) -> dict:
        with self.lock:
            p = self.get(name, need_on=False)
            if p["status"] == "broken":
                raise PluginError(p["error"])
            st = self._state()
            if on:
                st[name] = {"sha256": p["sha256"], "folder": p["folder"]}
            else:
                st.pop(name, None)
            self._save(st)
        return self.get(name, need_on=False)

    # -- what plugins add
    def job_specs(self) -> dict:
        """Job kinds from enabled plugins, in viz/jobs.py's SPECS shape."""
        out = {}
        for p in self.enabled():
            folder = Path(p["folder"])
            for j in p["manifest"].get("jobs") or []:
                fields = []
                for f in j.get("fields") or []:
                    fields.append({"name": f["name"], "flag": f.get("flag", "--" + f["name"].replace("_", "-")),
                                   "kind": f.get("kind", "str"), "required": bool(f.get("required")),
                                   "default": f.get("default"), "help": f.get("help", ""),
                                   "choices": f.get("choices"), "multi": bool(f.get("multi")), "repeat": False})
                out[f"plugin:{p['name']}:{j['kind']}"] = {
                    "title": j.get("title") or j["kind"], "group": "Plugins: " + p["name"],
                    "argv": [_inside(folder, j["script"])], "help": j.get("help", ""), "fields": fields,
                    **({"interp": j["interp"]} if isinstance(j.get("interp"), list) else {})}
        return out

    def checks(self) -> list[dict]:
        return [{"plugin": p["name"], "name": c["name"], "title": c.get("title") or c["name"],
                 "help": c.get("help", ""), "auto": bool(c.get("auto"))}
                for p in self.enabled() for c in p["manifest"].get("checks") or []]

    def run_check(self, plugin: str, check: str, payload: dict, timeout: float = CHECK_TIMEOUT) -> dict:
        """A reply check: the script reads JSON on stdin ({model, messages, text}) and prints
        JSON ({label, score, flag, note, spans: [[start, end, note], ...]})."""
        p = self.get(plugin)
        c = next((x for x in p["manifest"].get("checks") or [] if x["name"] == check), None)
        if c is None:
            raise PluginError(f"plugin {plugin} has no check {check!r}")
        folder = Path(p["folder"])
        script = _inside(folder, c["script"])
        interp = c["interp"] if isinstance(c.get("interp"), list) else ([sys.executable] if script.suffix == ".py" else [])
        try:
            r = subprocess.run([*interp, str(script)], input=json.dumps(payload), capture_output=True, text=True,
                               timeout=timeout, cwd=str(folder), env=dict(os.environ, PYTHONUNBUFFERED="1"))
        except subprocess.TimeoutExpired:
            raise PluginError(f"{plugin}/{check} took longer than {timeout:.0f}s") from None
        if r.returncode != 0:
            raise PluginError(f"{plugin}/{check} failed: {(r.stderr or r.stdout).strip()[-400:]}")
        try:
            out = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise PluginError(f"{plugin}/{check} printed no JSON") from None
        if not isinstance(out, dict):
            raise PluginError(f"{plugin}/{check} printed {type(out).__name__}, not an object")
        res = {"plugin": plugin, "check": check, "title": c.get("title") or check,
               "label": str(out.get("label", ""))[:200], "note": str(out.get("note", ""))[:2000],
               "flag": bool(out.get("flag"))}
        if isinstance(out.get("score"), (int, float)):
            res["score"] = float(out["score"])
        spans = []
        for s in out.get("spans") or []:
            if isinstance(s, list) and len(s) >= 2 and all(isinstance(x, int) for x in s[:2]):
                spans.append([s[0], s[1], str(s[2])[:200] if len(s) > 2 else ""])
        res["spans"] = spans[:500]
        return res

    def graph_nodes(self) -> list[dict]:
        """Node types from enabled plugins, for graph_engine.py: one script per node type."""
        return [{"plugin": p["name"], **n} for p in self.enabled() for n in p["manifest"].get("nodes") or []]

    def run_node(self, plugin: str, node_type: str, payload: dict, timeout: float = 300) -> dict:
        """A graph node: the script reads {"params", "inputs"} as JSON on stdin and prints its outputs as a
        JSON object (a missing output is skipped; {"error": "..."} fails the node)."""
        p = self.get(plugin)
        n = next((x for x in p["manifest"].get("nodes") or [] if x["type"] == node_type), None)
        if n is None:
            raise PluginError(f"plugin {plugin} has no node {node_type!r}")
        folder = Path(p["folder"])
        script = _inside(folder, n["script"])
        interp = n["interp"] if isinstance(n.get("interp"), list) else ([sys.executable] if script.suffix == ".py" else [])
        try:
            r = subprocess.run([*interp, str(script)], input=json.dumps(payload, default=str), capture_output=True,
                               text=True, timeout=timeout, cwd=str(folder), env=dict(os.environ, PYTHONUNBUFFERED="1"))
        except subprocess.TimeoutExpired:
            raise PluginError(f"{plugin}/{node_type} took longer than {timeout:.0f}s") from None
        if r.returncode != 0:
            raise PluginError(f"{plugin}/{node_type} failed: {(r.stderr or r.stdout).strip()[-400:]}")
        try:
            out = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise PluginError(f"{plugin}/{node_type} printed no JSON") from None
        if not isinstance(out, dict):
            raise PluginError(f"{plugin}/{node_type} printed {type(out).__name__}, not an object")
        if out.get("error"):
            raise PluginError(str(out["error"])[:1000])
        return out

    def page_file(self, plugin: str, rel: str) -> Path | None:
        p = self.get(plugin)
        page = p["manifest"].get("page")
        if not page:
            return None
        root = _inside(Path(p["folder"]), page)
        try:
            f = _inside(root, rel or "index.html")
        except PluginError:
            return None
        if f.is_dir():
            f = f / "index.html"
        return f if f.is_file() else None

    def mcp_servers(self) -> dict:
        """MCP servers from enabled plugins, for Studio's MCP client (run in the plugin's folder)."""
        out = {}
        for p in self.enabled():
            spec = p["manifest"].get("mcp")
            if spec:
                out["plugin-" + p["name"]] = {**spec, "cwd": spec.get("cwd") or p["folder"]}
        return out

    def summary(self) -> list[dict]:
        rows = []
        for p in self.scan():
            m = p.get("manifest") or {}
            rows.append({"name": p["name"], "status": p["status"], "error": p.get("error"), "folder": p["folder"],
                         "version": m.get("version"), "description": m.get("description", ""),
                         "author": m.get("author", ""), "sha256": p.get("sha256"),
                         "adds": {"jobs": [j.get("title") or j["kind"] for j in m.get("jobs") or []],
                                  "checks": [c.get("title") or c["name"] for c in m.get("checks") or []],
                                  "page": bool(m.get("page")), "mcp": bool(m.get("mcp")),
                                  "nodes": [n.get("title") or n["type"] for n in m.get("nodes") or []]}})
        return rows
