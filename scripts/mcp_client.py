#!/usr/bin/env python3
"""
MCP client for Studio: connect to MCP servers and expose their tools to chat.

Servers are configured in the format LM Studio and Claude Desktop use
(default ~/.neuronscope/mcp.json):

    {"mcpServers": {
        "files":  {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]},
        "search": {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer ..."}},
        "clock":  {"command": "python", "args": ["clock_server.py"], "autoApprove": ["now"]},
        "old":    {"command": "...", "disabled": true}}}

Each enabled server keeps one connection (stdio subprocess or streamable HTTP)
in a background event loop. Tools are offered to the model as OpenAI
functions named `<server>__<tool>`. Studio asks before every call unless the
server sets `"autoApprove": true` or lists the tool names it may run unasked.

    python scripts/mcp_client.py tools               # what the configured servers offer
    python scripts/mcp_client.py call clock__now '{}'
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import threading
from concurrent.futures import Future
from pathlib import Path

DEFAULT_CONFIG = Path.home() / ".neuronscope" / "mcp.json"
SEP = "__"
MAX_RESULT_CHARS = 20000


def _attr(obj, *names):
    """mcp 2.x uses snake_case model fields; older releases used camelCase."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


class _Conn:
    """One server: a task that holds the connection and serves queued calls."""

    def __init__(self, name: str, spec: dict, log_dir: Path):
        self.name, self.spec, self.log_dir = name, spec, log_dir
        self.errlog = None
        self.queue: asyncio.Queue | None = None
        self.tools: list = []
        self.error: str | None = None
        self.ready = asyncio.Event()
        self.task: asyncio.Task | None = None

    def transport(self):
        if self.spec.get("url"):
            from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
            return streamable_http_client(self.spec["url"],
                                          http_client=create_mcp_http_client(headers=self.spec.get("headers") or None))
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        env = None
        if self.spec.get("env"):
            env = {**os.environ, **{k: str(v) for k, v in self.spec["env"].items()}}
        params = StdioServerParameters(command=self.spec["command"], args=[str(a) for a in self.spec.get("args", [])],
                                       env=env, cwd=self.spec.get("cwd"))
        # The server's stderr (its logs) goes to a file, not into the host's console.
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.errlog = open(self.log_dir / f"{self.name}.log", "a", encoding="utf-8")
        return stdio_client(params, errlog=self.errlog)

    async def run(self):
        from mcp import Client
        self.queue = asyncio.Queue()
        try:
            async with Client(self.transport(), read_timeout_seconds=float(self.spec.get("timeout", 120))) as c:
                self.tools = list((await c.list_tools()).tools)
                self.error = None
                self.ready.set()
                while True:
                    op, args, fut = await self.queue.get()
                    if op == "stop":
                        fut.set_result(None)
                        return
                    try:
                        if op == "call":
                            fut.set_result(await c.call_tool(*args))
                        elif op == "list":
                            self.tools = list((await c.list_tools()).tools)
                            fut.set_result(self.tools)
                    except Exception as e:           # a failing call must not drop the connection
                        fut.set_exception(e)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"[:500]
            self.ready.set()
            while self.queue is not None and not self.queue.empty():
                _, _, fut = self.queue.get_nowait()
                if not fut.done():
                    fut.set_exception(RuntimeError(self.error))


class MCPHub:
    def __init__(self, config: str | Path | None = None):
        self.config_path = Path(config or DEFAULT_CONFIG).expanduser()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True, name="mcp-hub")
        self.thread.start()
        self.conns: dict[str, _Conn] = {}
        self.specs: dict[str, dict] = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------ config
    def load(self) -> dict:
        if not self.config_path.exists():
            return {}
        cfg = json.loads(self.config_path.read_text())
        servers = cfg.get("mcpServers", cfg if isinstance(cfg, dict) else {})
        out = {}
        for name, spec in servers.items():
            if not isinstance(spec, dict) or not (spec.get("command") or spec.get("url")):
                continue
            out[_safe(name)] = spec
        return out

    def _run(self, coro, timeout: float):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def connect(self, timeout: float = 60) -> None:
        """(Re)read the config; start new or changed servers, stop removed ones."""
        specs = {n: s for n, s in self.load().items() if not s.get("disabled")}
        with self.lock:
            for name in list(self.conns):
                if name not in specs or specs[name] != self.specs.get(name):
                    self._stop(name)
            for name, spec in specs.items():
                if name not in self.conns:
                    conn = _Conn(name, spec, self.config_path.parent / "mcp-logs")
                    self.conns[name] = conn
                    self.specs[name] = spec

                    async def start(c=conn):
                        c.task = asyncio.ensure_future(c.run())
                    self._run(start(), 5)
            conns = list(self.conns.values())
        for c in conns:
            try:
                self._run(asyncio.wait_for(c.ready.wait(), timeout), timeout + 5)
            except Exception:
                c.error = c.error or f"no answer within {timeout}s"

    def _stop(self, name: str) -> None:
        c = self.conns.pop(name, None)
        self.specs.pop(name, None)
        if c is None or c.queue is None or c.task is None or c.task.done():
            return
        fut = Future()

        async def put():
            await c.queue.put(("stop", None, _Bridge(fut)))
        try:
            self._run(put(), 5)
            fut.result(10)
        except Exception:
            self.loop.call_soon_threadsafe(c.task.cancel)

    def close(self) -> None:
        with self.lock:
            for name in list(self.conns):
                self._stop(name)

    # ------------------------------------------------------------ tools
    def status(self) -> list[dict]:
        out = []
        for name, spec in self.load().items():
            c = self.conns.get(name)
            out.append({"name": name, "transport": "http" if spec.get("url") else "stdio",
                        "disabled": bool(spec.get("disabled")), "connected": bool(c and not c.error and c.ready.is_set()),
                        "error": c.error if c else None, "auto_approve": spec.get("autoApprove", False),
                        "tools": [{"name": t.name, "description": (t.description or "")[:300]} for t in (c.tools if c else [])]})
        return out

    def openai_tools(self) -> list[dict]:
        out = []
        for name, c in self.conns.items():
            if c.error:
                continue
            for t in c.tools:
                schema = _attr(t, "input_schema", "inputSchema")
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}
                out.append({"type": "function", "function": {
                    "name": f"{name}{SEP}{_safe(t.name)}"[:64],
                    "description": (t.description or f"{t.name} from MCP server {name}")[:1024],
                    "parameters": schema}})
        return out

    def resolve(self, fn_name: str) -> tuple[str, str]:
        server, _, tool = fn_name.partition(SEP)
        c = self.conns.get(server)
        if c is None:
            raise KeyError(f"no MCP server {server!r}")
        for t in c.tools:
            if _safe(t.name) == tool:
                return server, t.name
        raise KeyError(f"MCP server {server!r} has no tool {tool!r}")

    def auto_approved(self, fn_name: str) -> bool:
        try:
            server, tool = self.resolve(fn_name)
        except KeyError:
            return False
        aa = self.specs.get(server, {}).get("autoApprove", False)
        return aa is True or (isinstance(aa, list) and tool in aa)

    def call(self, fn_name: str, arguments: dict, timeout: float = 120) -> tuple[bool, str]:
        """-> (ok, text). Text content is joined; other content is summarised."""
        server, tool = self.resolve(fn_name)
        c = self.conns[server]
        fut: Future = Future()

        async def put():
            await c.queue.put(("call", (tool, arguments), _Bridge(fut)))
        self._run(put(), 5)
        res = fut.result(timeout)
        parts = []
        for item in getattr(res, "content", None) or []:
            kind = getattr(item, "type", "")
            if kind == "text":
                parts.append(item.text)
            elif kind == "resource" and getattr(getattr(item, "resource", None), "text", None):
                parts.append(item.resource.text)
            else:
                parts.append(f"[{kind or 'content'} omitted]")
        sc = _attr(res, "structured_content", "structuredContent")
        if not parts and sc is not None:
            parts.append(json.dumps(sc, ensure_ascii=False))
        text = "\n".join(parts)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"\n[truncated {len(text) - MAX_RESULT_CHARS} characters]"
        return not _attr(res, "is_error", "isError"), text


class _Bridge:
    """Lets the event loop resolve a concurrent.futures.Future from another thread."""

    def __init__(self, fut: Future):
        self.fut = fut

    def done(self):
        return self.fut.done()

    def set_result(self, v):
        if not self.fut.done():
            self.fut.set_result(v)

    def set_exception(self, e):
        if not self.fut.done():
            self.fut.set_exception(e)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tools")
    c = sub.add_parser("call")
    c.add_argument("name")
    c.add_argument("arguments", nargs="?", default="{}")
    a = p.parse_args(argv)
    hub = MCPHub(a.config)
    hub.connect()
    try:
        if a.cmd == "tools":
            print(json.dumps(hub.status(), indent=1))
        else:
            ok, text = hub.call(a.name, json.loads(a.arguments))
            print(("" if ok else "error: ") + text)
    finally:
        hub.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
