#!/usr/bin/env python3
"""
NeuronScope as an MCP server: the harness and lab as tools for Cline, Claude
Desktop, LM Studio or any other MCP client.

Every tool is a call to Studio's HTTP API, so an agent sees exactly what the
UI sees and is held to the same rules: Studio's token, owner-only actions,
paired-device limits, jobs only from typed forms. Two ways to connect:

  * streamable HTTP, served by Studio itself at  http://127.0.0.1:7870/mcp
    (nothing to install; send Studio's token as a Bearer header if it has one);
  * stdio, for clients that start a process:
        python scripts/ns_mcp.py --studio http://127.0.0.1:7870 [--token-file F]

No MCP library is needed: the protocol (JSON-RPC 2.0: initialize, tools/list,
tools/call, ping) is implemented here, and tests check it against the
official `mcp` client.

    python scripts/ns_mcp.py --list        # the tools, and what each does
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request

SERVER_NAME = "neuronscope"
SERVER_VERSION = "1.0"
PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
INSTRUCTIONS = (
    "NeuronScope Studio: local models (llama.cpp), graded evaluation (TestQA), hallucination checks from "
    "neuron activations, retraining and benchmark jobs, and projects split into tasks for worker models. "
    "Start with studio_status and list_models. Model ids from list_models go in every 'model' argument. "
    "Jobs and projects run on the user's machine and can take hours; start them only when asked, then "
    "poll job_status or get_project. Results of check_reply are a risk signal (classifier AUROC ~0.7), "
    "not a verdict.")


class ToolError(Exception):
    pass


class Studio:
    """HTTP client for Studio's API, carrying the caller's token."""

    def __init__(self, base: str, token: str | None = None, insecure_loopback: bool = False):
        self.base = base.rstrip("/")
        self.token = token
        self.ctx = None
        if base.startswith("https://") and insecure_loopback:
            # Studio calling itself over TLS with a self-signed certificate.
            self.ctx = ssl.create_default_context()
            self.ctx.check_hostname = False
            self.ctx.verify_mode = ssl.CERT_NONE

    def call(self, method: str, path: str, body=None, timeout: float = 600):
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=self.ctx) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                msg = json.loads(raw).get("error")
                msg = msg.get("message") if isinstance(msg, dict) else msg
            except Exception:
                msg = raw[:300].decode(errors="replace")
            raise ToolError(f"Studio said {e.code}: {msg}")
        except urllib.error.URLError as e:
            raise ToolError(f"cannot reach Studio at {self.base}: {e.reason}. Is it running?")
        try:
            return json.loads(raw)
        except ValueError:
            return {"text": raw.decode(errors="replace")}


# ------------------------------------------------------------------ tools

def _s(props: dict, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


STR = {"type": "string"}
INT = {"type": "integer"}
BOOL = {"type": "boolean"}
READ = {"readOnlyHint": True, "openWorldHint": False}
WRITE = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}
RISKY = {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False}


def _chat(st, a):
    msgs = a.get("messages") or [{"role": "user", "content": a.get("prompt", "")}]
    if a.get("system"):
        msgs = [{"role": "system", "content": a["system"]}] + msgs
    r = st.call("POST", "/v1/chat/completions", {
        "model": a.get("model") or "", "messages": msgs, "stream": False,
        "max_tokens": int(a.get("max_tokens") or 1024), "temperature": float(a.get("temperature", 0.7))})
    msg = (r.get("choices") or [{}])[0].get("message", {})
    return {"model": r.get("model") or a.get("model"), "reply": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or None, "usage": r.get("usage")}


def _models(st, a):
    r = st.call("GET", "/v1/models")
    out = []
    for m in r.get("data", []):
        s = m.get("stats") or {}
        out.append({"id": m["id"], "loaded": m.get("loaded"), "vision": m.get("vision"), "quant": m.get("quant"),
                    "graded": s.get("graded"), "accuracy": s.get("accuracy"),
                    "hallucination_rate": s.get("hallucination_rate"), "auto_eligible": s.get("auto_eligible")})
    return {"models": out}


def _load(st, a):
    ms = st.call("GET", "/api/models")
    m = next((x for x in ms if a["model"] in (x["id"], x["name"], x["path"])), None)
    if m is None:
        raise ToolError(f"no model {a['model']!r}; list_models shows the ids")
    return st.call("POST", "/api/load", {"path": m["path"], "settings": m.get("settings") or {}}, timeout=900)


def _check(st, a):
    msgs = a.get("messages") or [{"role": "user", "content": a.get("prompt", "")}]
    r = st.call("POST", "/api/trace", {"model": a.get("model"), "messages": msgs, "text": a["reply"]}, timeout=1800)
    flagged = [r["pieces"][i] for i in r.get("flagged", [])]
    return {"flagged_tokens": len(flagged), "of_tokens": len(r.get("prob", [])), "peak_risk": r.get("max"),
            "mean_risk": r.get("mean"), "threshold": r.get("threshold"), "flagged_text": flagged[:80],
            "view_3d": st.base + "/" + r["url"], "trace_id": r.get("id")}


def _job_status(st, a):
    jobs = st.call("GET", "/api/jobs")
    if a.get("id"):
        j = next((x for x in jobs if x.get("id") == a["id"]), None)
        if j is None:
            raise ToolError(f"no job {a['id']}")
        return j
    return {"jobs": jobs[: int(a.get("limit") or 20)]}


def _job_log(st, a):
    log = st.call("GET", f"/api/jobs/{a['id']}/log").get("log", "")
    n = int(a.get("tail_chars") or 4000)
    return {"id": a["id"], "log": log[-n:]}


def _project(st, a):
    p = st.call("GET", f"/api/projects/{a['id']}")
    for t in p.get("tasks", []):
        for at in t.get("attempts", []):           # keep replies short; the full text is in the UI
            if len(at.get("text") or "") > 3000:
                at["text"] = at["text"][:3000] + f"… [{len(at['text'])} chars; full text on /projects]"
    p.pop("history", None)
    return p


def _pact(action):
    def f(st, a):
        body = {k: v for k, v in a.items() if k != "id"}
        return st.call("POST", f"/api/projects/{a['id']}/{action}", body)
    return f


TOOLS = [
    ("studio_status", "What Studio is serving right now: the loaded model, its uptime, whether auto routing "
     "has stats to work with.", _s({}), READ, lambda st, a: st.call("GET", "/api/status")),
    ("list_models", "Local GGUF models Studio can serve, with their measured accuracy and hallucination rate "
     "when TestQA has graded them. The 'id' is what every other tool's 'model' takes.", _s({}), READ, _models),
    ("load_model", "Load a model into Studio's chat server (unloads the current one). Chat requests also load "
     "models on demand, so this is only needed to preload.", _s({"model": STR}, ["model"]), WRITE, _load),
    ("unload_model", "Unload Studio's chat model to free memory.", _s({}), WRITE,
     lambda st, a: st.call("POST", "/api/unload", {})),
    ("chat", "Ask one of the user's local models. 'model' may be 'auto' to let Studio pick the model with the "
     "best measured record for the prompt's subject.",
     _s({"model": STR, "prompt": STR, "system": STR,
         "messages": {"type": "array", "items": {"type": "object"}}, "max_tokens": INT,
         "temperature": {"type": "number"}}, ["model"]), WRITE, _chat),
    ("check_reply", "Score each token of a reply for hallucination risk from the model's own neuron activations "
     "(needs a classifier for that model). Returns the flagged text and a link to a 3D view.",
     _s({"model": STR, "prompt": STR, "reply": STR, "messages": {"type": "array", "items": {"type": "object"}}},
        ["model", "reply"]), WRITE, _check),
    ("route_prompt", "Which subjects a prompt or task involves, and which model 'auto' would pick for it and "
     "why.", _s({"text": STR}, ["text"]), READ, lambda st, a: st.call("POST", "/api/route", {"text": a["text"]})),
    ("model_stats", "Per-model, per-subject results from graded runs: accuracy, hallucination and abstention "
     "rates with confidence intervals.", _s({}), READ, lambda st, a: st.call("GET", "/api/stats")),
    ("devices", "Accelerators on this machine (ROCm, Vulkan, CUDA, Metal, CPU) with free memory, and the "
     "worker models running on them.", _s({}), READ, lambda st, a: st.call("GET", "/api/workers")),
    ("doctor", "Check the installation: packages, GPUs, llama.cpp build, model, pipeline progress, with a fix "
     "for each problem.", _s({}), READ, lambda st, a: st.call("GET", "/api/doctor", timeout=180)),
    ("job_kinds", "Jobs Studio can run (TestQA, retraining, benchmarks, pipeline stages, builds) and the "
     "typed fields each takes.", _s({}), READ, lambda st, a: st.call("GET", "/api/jobs/specs")),
    ("start_job", "Start a job of one of the kinds from job_kinds, with its fields as 'values'. Runs on the "
     "user's machine; long jobs can take hours.",
     _s({"kind": STR, "values": {"type": "object"}}, ["kind"]), RISKY,
     lambda st, a: st.call("POST", "/api/jobs", {"kind": a["kind"], "values": a.get("values") or {}})),
    ("job_status", "One job's state, or the most recent jobs.", _s({"id": STR, "limit": INT}), READ, _job_status),
    ("job_log", "The end of a job's output.", _s({"id": STR, "tail_chars": INT}, ["id"]), READ, _job_log),
    ("cancel_job", "Stop a running job.", _s({"id": STR}, ["id"]), RISKY,
     lambda st, a: st.call("POST", "/api/jobs/cancel", {"id": a["id"]})),
    ("list_projects", "Projects the director is planning or running.", _s({}), READ,
     lambda st, a: st.call("GET", "/api/projects")),
    ("create_project", "Turn a description (notes, a task list, a README) into a draft plan of skill-labelled "
     "tasks. Nothing runs until a person approves it.",
     _s({"title": STR, "goal": STR, "text": STR, "repo": STR}, ["text"]), WRITE,
     lambda st, a: st.call("POST", "/api/projects", a)),
    ("get_project", "A project's plan: tasks, their status, assigned models, results awaiting review, pending "
     "proposals.", _s({"id": STR}, ["id"]), READ, _project),
    ("edit_project", "Change a plan: changes is a list of {op: add|update|drop|reopen|assign, ...} (see "
     "docs/DIRECTOR.md). Logged as the person's edit.",
     _s({"id": STR, "changes": {"type": "array", "items": {"type": "object"}}, "reason": STR}, ["id", "changes"]),
     WRITE, _pact("edit")),
    ("approve_project", "Approve a draft plan: assigns a model to every task and freezes it.",
     _s({"id": STR}, ["id"]), WRITE, _pact("approve")),
    ("start_project", "Start (or resume) an approved project: worker models begin on ready tasks.",
     _s({"id": STR}, ["id"]), RISKY, _pact("start")),
    ("pause_project", "Pause a running project.", _s({"id": STR}, ["id"]), WRITE, _pact("pause")),
    ("review_task", "Accept a task's result, or send it back with feedback for the next attempt.",
     _s({"id": STR, "task": STR, "accept": BOOL, "feedback": STR}, ["id", "task", "accept"]), WRITE,
     _pact("review")),
    ("decide_proposal", "Accept or reject a change the director proposed.",
     _s({"id": STR, "proposal": STR, "accept": BOOL, "note": STR}, ["id", "proposal", "accept"]), WRITE,
     _pact("decide")),
    ("answer_task", "Answer a blocked task's questions; it goes back in the queue.",
     _s({"id": STR, "task": STR, "answers": {"type": "array", "items": STR}}, ["id", "task", "answers"]), WRITE,
     _pact("answer")),
    ("list_traces", "Saved per-reply checks and traces, viewable in 3D at <studio>/viz/<id>/.", _s({}), READ,
     lambda st, a: st.call("GET", "/api/traces")),
]
BY_NAME = {t[0]: t for t in TOOLS}


def tool_list() -> list[dict]:
    return [{"name": n, "description": d, "inputSchema": s, "annotations": {"title": n.replace("_", " "), **ann}}
            for n, d, s, ann, _ in TOOLS]


# ------------------------------------------------------------------ protocol

def _err(mid, code, msg):
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": msg}}


def handle(msg, studio: Studio):
    """One JSON-RPC message -> response dict, or None for a notification."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
        return _err(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
    mid, method, params = msg.get("id"), msg["method"], msg.get("params") or {}
    if "id" not in msg:                       # notifications (initialized, cancelled, …) need no answer
        return None
    try:
        if method == "initialize":
            want = params.get("protocolVersion")
            res = {"protocolVersion": want if want in PROTOCOLS else PROTOCOLS[0],
                   "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                   "instructions": INSTRUCTIONS}
        elif method == "ping":
            res = {}
        elif method == "tools/list":
            res = {"tools": tool_list()}
        elif method == "tools/call":
            name = params.get("name")
            if name not in BY_NAME:
                return _err(mid, -32602, f"unknown tool {name!r}")
            args = params.get("arguments") or {}
            if not isinstance(args, dict):
                return _err(mid, -32602, "arguments must be an object")
            schema = BY_NAME[name][2]
            missing = [k for k in schema["required"] if k not in args]
            if missing:
                return _err(mid, -32602, f"{name}: missing {', '.join(missing)}")
            try:
                out = BY_NAME[name][4](studio, args)
                res = {"content": [{"type": "text", "text": json.dumps(out, indent=1, default=str)}],
                       "structuredContent": out if isinstance(out, dict) else {"result": out}, "isError": False}
            except ToolError as e:
                res = {"content": [{"type": "text", "text": str(e)}], "isError": True}
        elif method in ("resources/list", "prompts/list"):
            res = {method.split("/")[0]: []}
        else:
            return _err(mid, -32601, f"method not found: {method}")
    except Exception as e:                     # a bug must not kill the session
        return _err(mid, -32603, f"{type(e).__name__}: {e}")
    return {"jsonrpc": "2.0", "id": mid, "result": res}


def handle_payload(payload, studio: Studio):
    """A message or a batch -> response(s), None when nothing needs an answer."""
    if isinstance(payload, list):
        out = [r for r in (handle(m, studio) for m in payload) if r is not None]
        return out or None
    return handle(payload, studio)


def serve_stdio(studio: Studio) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            resp = _err(None, -32700, "parse error")
        else:
            resp = handle_payload(payload, studio)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="NeuronScope MCP server (stdio). An MCP client starts this; "
                                            "run alone it waits for requests on stdin.")
    p.add_argument("--studio", default=os.environ.get("NS_STUDIO_URL", "http://127.0.0.1:7870"),
                   help="Studio URL (default $NS_STUDIO_URL or http://127.0.0.1:7870)")
    p.add_argument("--token-file", help="file holding Studio's token (or a paired device's)")
    p.add_argument("--list", action="store_true", help="print the tools and exit")
    a = p.parse_args(argv)
    if a.list:
        for n, d, s, ann, _ in TOOLS:
            kind = "read" if ann.get("readOnlyHint") else ("changes things" if not ann.get("destructiveHint")
                                                           else "starts work / stops work")
            print(f"{n:<16} [{kind}] {d}")
        return 0
    token = os.environ.get("NS_STUDIO_TOKEN") or None
    if a.token_file:
        token = open(os.path.expanduser(a.token_file)).read().strip()
    serve_stdio(Studio(a.studio, token))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
