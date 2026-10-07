#!/usr/bin/env python3
"""Optional MCP control plane for NeuronScope transfers (stdio only).

Secrets handling:
* the state file holds target identity only (url, CA file, hints), mode 0600;
* each target's bearer token lives in the OS keyring when ``keyring`` is
  installed, otherwise in a 0600 file under ~/.config/neuronscope/secrets/;
* tokens reach the CLI through the NS_TRANSFER_TOKEN environment variable of
  the child process, never argv, and are never returned by any tool.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_security as sec  # noqa: E402

MIN_MCP = (2, 3)

try:
    # mcp 2.x renamed FastMCP to MCPServer; the decorator API is unchanged.
    from mcp.server.mcpserver import MCPServer as FastMCP
    IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on optional package
    try:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x, refused by ensure()
        IMPORT_ERROR = None
    except Exception:
        FastMCP = None
        IMPORT_ERROR = e

ROOT = Path(__file__).resolve().parent
CLI = ROOT / "ns_transfer.py"
CONFIG = Path(os.environ.get("NS_CONFIG_DIR", str(Path.home() / ".config" / "neuronscope")))
STATE = Path(os.environ.get("NS_TRANSFER_MCP_STATE", str(CONFIG / "transfer-mcp.json")))
SECRETS = CONFIG / "secrets"
KEYRING_SERVICE = "neuronscope-transfer"
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def mcp_version_ok() -> tuple[bool, str]:
    try:
        from importlib.metadata import version
        v = version("mcp")
    except Exception:
        return False, "not installed"
    nums = tuple(int(x) for x in re.findall(r"\d+", v)[:2])
    return nums >= MIN_MCP, v


# ---------------------------------------------------------------- secrets

def _keyring():
    try:
        import keyring  # type: ignore
        keyring.get_keyring()
        return keyring
    except Exception:
        return None


def store_token(name: str, token: str) -> str:
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(KEYRING_SERVICE, name, token)
            return "keyring"
        except Exception:
            pass
    sec.write_secret_file(SECRETS / f"{name}.token", token)
    return "file"


def load_token(name: str) -> str:
    kr = _keyring()
    if kr is not None:
        try:
            v = kr.get_password(KEYRING_SERVICE, name)
            if v:
                return v
        except Exception:
            pass
    p = SECRETS / f"{name}.token"
    return sec.read_secret_file(p) if p.exists() else ""


def delete_token(name: str) -> None:
    kr = _keyring()
    if kr is not None:
        try:
            kr.delete_password(KEYRING_SERVICE, name)
        except Exception:
            pass
    (SECRETS / f"{name}.token").unlink(missing_ok=True)


# ---------------------------------------------------------------- state

def load() -> dict:
    if STATE.exists():
        try:
            d = json.loads(STATE.read_text())
            # Migrate state written by older versions that stored tokens inline.
            changed = False
            for name, t in d.get("targets", {}).items():
                if t.get("token"):
                    store_token(name, t.pop("token"))
                    changed = True
            if changed:
                save(d)
            return d
        except Exception:
            pass
    return {"targets": {}}


def save(d: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, STATE)


def get_target(name: str) -> dict:
    t = load()["targets"].get(name)
    if t is None:
        raise KeyError(f"unknown target {name!r}; call register_target first")
    return t


def _redact(text: str, token: str) -> str:
    return text.replace(token, "***") if token else text


def run(name: str, args: list[str], timeout: int = 30) -> dict:
    """Run the transfer CLI against a target with the token in the environment."""
    t = get_target(name)
    token = load_token(name)
    cmd = [sys.executable, str(CLI), *args, "--url", t["url"]]
    if t.get("cafile"):
        cmd += ["--cafile", t["cafile"]]
    env = {**os.environ, sec.TOKEN_ENV: token}
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return {"returncode": p.returncode,
            "stdout": _redact(p.stdout[-12000:], token),
            "stderr": _redact(p.stderr[-12000:], token)}


def public(name: str, t: dict) -> dict:
    return {**t, "has_token": bool(load_token(name))}


def ensure() -> None:
    if FastMCP is None:
        raise RuntimeError(f"mcp package unavailable: {IMPORT_ERROR}  (pip install 'mcp>=2.3,<3')")
    ok, v = mcp_version_ok()
    if not ok:
        raise RuntimeError(f"mcp {v} is older than {'.'.join(map(str, MIN_MCP))}, which has known "
                           "security advisories; pip install -U 'mcp>=2.3,<3'")


if FastMCP:
    mcp = FastMCP("neuronscope-transfer")

    @mcp.tool()
    def register_target(name: str, url: str, token: str = "", cafile: str = "",
                        destination_hint: str = "") -> dict:
        """Register a remote NeuronScope transfer receiver. The token is moved to
        the OS keyring (or a 0600 file) and is never echoed back."""
        if not NAME_RE.match(name):
            raise ValueError("name must match [A-Za-z0-9_.-]{1,64}")
        if not url.startswith(("https://", "http://")):
            raise ValueError("url must be http(s)://")
        d = load()
        d["targets"][name] = {"url": url, "cafile": cafile, "destination_hint": destination_hint}
        save(d)
        where = store_token(name, token) if token else "none"
        warn = []
        if url.startswith("http://") and not sec.is_loopback(url.split("/")[2].rsplit(":", 1)[0]):
            warn.append("plaintext http:// target: only safe through a VPN tunnel")
        return {"ok": True, "name": name, "token_store": where, "warnings": warn,
                "target": public(name, d["targets"][name])}

    @mcp.tool()
    def remove_target(name: str) -> dict:
        """Forget a target and delete its stored token."""
        d = load()
        d["targets"].pop(name, None)
        save(d)
        delete_token(name)
        return {"ok": True}

    @mcp.tool()
    def list_targets() -> dict:
        """List configured transfer targets (tokens are never returned)."""
        return {"targets": {k: public(k, v) for k, v in load()["targets"].items()}}

    @mcp.tool()
    def probe_target(name: str) -> dict:
        """Check a target receiver."""
        return run(name, ["health"])

    @mcp.tool()
    def send_file(target: str, file_path: str, subdir: str = "") -> dict:
        """Start one reliable resumable file transfer."""
        return run(target, ["send", "--file", file_path, "--subdir", subdir], timeout=86400)

    @mcp.tool()
    def send_batch(target: str, directory: str, glob_pattern: str = "*.gguf", workers: int = 2,
                   subdir: str = "") -> dict:
        """Transfer a batch of GGUFs to a receiver."""
        return run(target, ["batch", "--dir", directory, "--glob", glob_pattern,
                            "--workers", str(max(1, min(workers, 8))), "--subdir", subdir],
                   timeout=86400)

    @mcp.tool()
    def start_watch(target: str, directory: str, glob_pattern: str = "*.gguf", workers: int = 1,
                    subdir: str = "") -> dict:
        """Start a background watcher that sends new GGUFs as they appear."""
        t = get_target(target)
        logdir = Path.home() / ".local" / "state" / "neuronscope-transfer"
        logdir.mkdir(parents=True, exist_ok=True)
        log = logdir / f"{target}.log"
        cmd = [sys.executable, str(CLI), "watch", "--url", t["url"], "--dir", directory,
               "--glob", glob_pattern, "--workers", str(max(1, min(workers, 8))), "--subdir", subdir]
        if t.get("cafile"):
            cmd += ["--cafile", t["cafile"]]
        env = {**os.environ, sec.TOKEN_ENV: load_token(target)}
        with log.open("a") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        return {"ok": True, "pid": p.pid, "log": str(log), "command": cmd}

    @mcp.tool()
    def transfer_status(target: str) -> dict:
        """List receiver-side transfers."""
        return run(target, ["status"])

    @mcp.tool()
    def pull_completed(target: str, transfer_id: str, out_directory: str) -> dict:
        """Pull a completed remote transfer back to the controller."""
        return run(target, ["pull", "--id", transfer_id, "--out", out_directory], timeout=86400)
else:
    mcp = None


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="MCP server (stdio) for NeuronScope transfers. An MCP client starts it; "
                    "it reads requests on stdin, so run on its own it waits silently.",
        epilog='client config: {"mcpServers": {"neuronscope-transfer": {"command": "python", '
               '"args": ["/path/to/neuronscope/scripts/ns_transfer_mcp.py"]}}}')
    p.parse_args()
    ensure()
    mcp.run("stdio")  # no network listener


if __name__ == "__main__":
    main()
