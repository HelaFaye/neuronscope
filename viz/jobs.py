#!/usr/bin/env python3
"""
Background jobs for Studio: evaluation, retraining and benchmarks from the
browser, without a free-form command line.

Each job kind is a fixed script plus a schema of typed fields. The argv is
built from the schema only: unknown fields are ignored, booleans become bare
flags, numbers must parse, and no value may start with "-" (so a value can
never smuggle in an extra flag). Jobs run as subprocesses with their output in
a log file under ~/.neuronscope/jobs/<id>/, survive the browser closing, and
can be cancelled.

Studio serves the page at /jobs and the API at /api/jobs. Because a job can run
training or execute model-written code (TestQA --allow-exec), Studio allows
jobs only on a loopback bind unless started with --allow-remote-jobs.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts"


def F(name, flag=None, kind="str", req=False, default=None, help="", choices=None, multi=False, repeat=False):
    """multi: several values after one flag (nargs); repeat: the flag once per value (action=append)."""
    return {"name": name, "flag": flag if flag is not None else "--" + name.replace("_", "-"), "kind": kind,
            "required": req, "default": default, "help": help, "choices": choices, "multi": multi or repeat,
            "repeat": repeat}


# kind -> (title, group, script argv prefix, fields)
SPECS: dict[str, dict] = {
    "testqa": {
        "title": "TestQA run", "group": "Evaluate", "argv": [S / "testqa.py"],
        "help": "Graded bank against an endpoint; publish stats so auto routing sees them.",
        "fields": [
            F("endpoint", req=True, repeat=True, help="label=URL@model, e.g. m=http://127.0.0.1:7870/v1@my-model"),
            F("per_subject", kind="int", help="items per subject (0 = all)"),
            F("subject", multi=True, help="limit to these subjects"),
            F("allow_exec", kind="bool", help="run model-written code"),
            F("sandbox", choices=["none", "docker", "podman"], default="none",
              help="contain that code in a container (recommended)"),
            F("cache", help="cache directory (needed by deficits)"),
            F("out", req=True, help="results JSON"),
            F("publish_stats", help="Studio URL to publish stats to, e.g. http://127.0.0.1:7870"),
        ]},
    "deficits": {
        "title": "Deficit dataset", "group": "Retrain",
        "argv": [S / "deficits.py"], "help": "Failures -> verified SFT/DPO data with replay and a holdout.",
        "fields": [
            F("results", req=True, help="TestQA results JSON"), F("label"), F("cache"),
            F("teacher", help="URL@model of a stronger model"),
            F("expand", kind="int", default=0, help="verified variations per failure"),
            F("allow_exec", kind="bool"), F("sandbox", choices=["none", "docker", "podman"], default="none"),
            F("deficit_fraction", kind="float", default=0.25),
            F("general", multi=True, help="extra anchor JSONL files"),
            F("out", req=True)]},
    "finetune": {
        "title": "Fine-tune", "group": "Retrain", "argv": [S / "finetune.py"],
        "help": "QLoRA / LoRA / full SFT, then optional DPO. HF weights, not GGUF.",
        "fields": [
            F("model", req=True, help="HF id or local path"), F("data", req=True, help="deficits output dir"),
            F("out", req=True), F("method", choices=["qlora", "lora", "full"], default="qlora"),
            F("dpo", kind="bool"), F("vision", kind="bool"), F("freeze_projector", kind="bool"),
            F("launch", kind="int", help="GPUs (accelerate)"), F("fsdp", kind="bool"),
            F("lora_r", kind="int", default=16), F("epochs", kind="float", default=2),
            F("max_steps", kind="int", default=-1), F("max_length", kind="int", default=2048),
            F("gradient_checkpointing", kind="bool")]},
    "merge": {
        "title": "Merge and convert", "group": "Retrain", "argv": [S / "merge_export.py"],
        "help": "Adapter -> merged model -> (quantized) GGUF, or a GGUF LoRA.",
        "fields": [
            F("base", req=True), F("adapter", req=True), F("out", req=True),
            F("gguf", help="quant type, e.g. Q4_K_M (empty: no GGUF)"), F("lora_gguf", kind="bool"),
            F("vision", kind="bool"), F("llama", help="llama.cpp checkout (default $NS_LLAMA)")]},
    "swe_predict": {
        "title": "SWE-bench predict", "group": "Benchmarks", "argv": [S / "benchmarks.py", "swebench", "predict"],
        "help": "Single-shot retrieval patches; score with the Docker harness.",
        "fields": [F("endpoint", req=True, help="URL@model"),
                   F("dataset", default="princeton-nlp/SWE-bench_Lite_bm25_13K"), F("split", default="test"),
                   F("limit", kind="int", default=50), F("out", req=True)]},
    "swe_evaluate": {
        "title": "SWE-bench evaluate", "group": "Benchmarks", "argv": [S / "benchmarks.py", "swebench", "evaluate"],
        "help": "Runs swebench.harness (needs Docker).",
        "fields": [F("predictions", req=True), F("dataset", default="princeton-nlp/SWE-bench_Lite"),
                   F("run_id", req=True), F("max_workers", kind="int", default=2)]},
    "swe_import": {
        "title": "SWE-bench import", "group": "Benchmarks", "argv": [S / "benchmarks.py", "swebench", "import"],
        "help": "Harness report -> stats.",
        "fields": [F("report", req=True), F("model", req=True), F("publish_stats")]},
    "swe_train": {
        "title": "SWE-bench train data", "group": "Retrain", "argv": [S / "benchmarks.py", "swebench", "train-data"],
        "help": "Train split gold patches -> SFT/DPO, prioritised by a test report.",
        "fields": [F("dataset", default="princeton-nlp/SWE-bench_bm25_13K"), F("predictions"), F("report"),
                   F("exclude", multi=True), F("anchors"), F("limit", kind="int", default=0), F("out", req=True)]},
    "livebench": {
        "title": "LiveBench run", "group": "Benchmarks", "argv": [S / "benchmarks.py", "livebench", "run"],
        "help": "Needs a LiveBench checkout.",
        "fields": [F("livebench", req=True), F("endpoint", req=True, help="URL@model"),
                   F("bench", multi=True, default=["live_bench"]), F("publish_stats")]},
    "clip_bench": {
        "title": "CLIP_benchmark", "group": "Benchmarks", "argv": [S / "clip_bench.py", "eval"],
        "help": "Zero-shot classification at each suppression scale.",
        "fields": [F("model", req=True), F("dataset", req=True, multi=True, help="e.g. imagenetv2 imagenet_sketch"),
                   F("dataset_root", req=True), F("h_neurons", flag="--h_neurons"), F("scales", multi=True, default=["1"]),
                   F("out", req=True)]},
}

SAFE = re.compile(r"^[^\x00-\x1f]*$")


def build_argv(kind: str, values: dict) -> list[str]:
    if kind not in SPECS:
        raise ValueError(f"unknown job kind {kind!r}")
    spec = SPECS[kind]
    argv = [sys.executable, *[str(x) for x in spec["argv"]]]
    for f in spec["fields"]:
        v = values.get(f["name"], f["default"])
        if f["kind"] == "bool":
            if v is True or v in ("true", "on", "1", 1):
                argv.append(f["flag"])
            continue
        if v is None or v == "" or v == []:
            if f["required"]:
                raise ValueError(f"{f['name']} is required")
            continue
        vals = v if isinstance(v, list) else (shlex.split(v) if f["multi"] and isinstance(v, str) else [v])
        out = []
        for x in vals:
            x = str(x).strip()
            if not x:
                continue
            if f["kind"] == "int":
                x = str(int(x))
            elif f["kind"] == "float":
                x = repr(float(x))
            elif x.startswith("-") or not SAFE.match(x) or len(x) > 4096:
                raise ValueError(f"{f['name']}: values may not start with '-' or contain control characters")
            if f["choices"] is not None and x not in f["choices"]:
                raise ValueError(f"{f['name']} must be one of {', '.join(c for c in f['choices'] if c)}")
            out.append(x)
        if out and f["repeat"]:
            for x in out:
                argv += [f["flag"], x]
        elif out:
            argv += [f["flag"], *out]
        elif f["required"]:
            raise ValueError(f"{f['name']} is required")
    return argv


class JobRunner:
    def __init__(self, root: str | Path, max_running: int = 2):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_running = max_running
        self.lock = threading.Lock()
        self.procs: dict[str, subprocess.Popen] = {}

    def _meta_path(self, jid: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", jid):
            raise ValueError("bad job id")
        return self.root / jid / "job.json"

    def _write(self, meta: dict) -> None:
        p = self._meta_path(meta["id"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(p)

    def start(self, kind: str, values: dict) -> dict:
        argv = build_argv(kind, values)
        with self.lock:
            running = sum(p.poll() is None for p in self.procs.values())
            if running >= self.max_running:
                raise RuntimeError(f"{running} jobs already running; wait or cancel one")
            jid = uuid.uuid4().hex[:12]
            d = self.root / jid
            d.mkdir(parents=True)
            log = open(d / "log.txt", "wb")
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env,
                                    start_new_session=True)
            log.close()
            self.procs[jid] = proc
            meta = {"id": jid, "kind": kind, "title": SPECS[kind]["title"], "values": values, "argv": argv[1:],
                    "status": "running", "pid": proc.pid, "started": time.time(), "ended": None, "returncode": None}
            self._write(meta)
        threading.Thread(target=self._wait, args=(jid, proc), daemon=True).start()
        return meta

    def _wait(self, jid: str, proc: subprocess.Popen) -> None:
        rc = proc.wait()
        meta = self.get(jid)
        if meta.get("status") == "running":
            meta["status"] = "done" if rc == 0 else "failed"
        meta.update(returncode=rc, ended=time.time())
        self._write(meta)

    def get(self, jid: str) -> dict:
        return json.loads(self._meta_path(jid).read_text())

    def list(self, limit: int = 50) -> list[dict]:
        out = []
        for p in self.root.glob("*/job.json"):
            try:
                m = json.loads(p.read_text())
            except Exception:
                continue
            if m.get("status") == "running" and m["id"] not in self.procs:
                m["status"] = "lost"        # Studio restarted while it ran
            out.append(m)
        out.sort(key=lambda m: -m.get("started", 0))
        return out[:limit]

    def log(self, jid: str, tail: int = 20000) -> str:
        p = self._meta_path(jid).parent / "log.txt"
        if not p.exists():
            return ""
        with open(p, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - tail))
            return f.read().decode(errors="replace")

    def cancel(self, jid: str) -> dict:
        meta = self.get(jid)
        proc = self.procs.get(jid)
        if proc is not None and proc.poll() is None:
            meta["status"] = "cancelled"
            self._write(meta)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, AttributeError):
                proc.terminate()
        return meta


def public_specs() -> dict:
    return {k: {"title": v["title"], "group": v["group"], "help": v["help"],
                "fields": [{x: f[x] for x in ("name", "kind", "required", "default", "help", "choices", "multi")}
                           for f in v["fields"]]} for k, v in SPECS.items()}


PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Studio Jobs</title>
<style>
:root{--bg:#fafaf9;--panel:#fff;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e4df;--acc:#2d5bd7;--accfg:#fff;--ok:#1f8a4c;--no:#c0392b;--code:#f3f3f0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--code:#262624}}
:root[data-theme=dark]{--bg:#151514;--panel:#1d1d1b;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#6f93ff;--accfg:#0b0b0a;--ok:#4cc27e;--no:#ff7a6b;--code:#262624}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 ui-sans-serif,system-ui;background:var(--bg);color:var(--fg)}
header{display:flex;gap:1rem;align-items:center;padding:.6rem 1rem;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:15px;margin:0}header a{color:var(--acc);font-size:13px}
main{display:grid;grid-template-columns:minmax(260px,380px) 1fr;gap:1rem;padding:1rem;max-width:1400px;margin:0 auto}
@media (max-width:800px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.8rem}
h2{font-size:13px;margin:.2rem 0 .5rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
label{display:block;font-size:12px;color:var(--mut);margin:.45rem 0 .1rem}
input,select{font:12px ui-monospace,monospace;padding:.3rem .4rem;border:1px solid var(--line);border-radius:5px;width:100%;background:var(--bg);color:var(--fg)}
input[type=checkbox]{width:auto;margin-right:.3rem}
button{font:500 13px ui-sans-serif,system-ui;padding:.4rem .8rem;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
.note{font-size:12px;color:var(--mut)}.err{color:var(--no);font-size:12px;white-space:pre-wrap}
.job{border-bottom:1px solid var(--line);padding:.45rem 0;cursor:pointer;display:flex;gap:.6rem;align-items:baseline}
.job:hover{background:var(--code)}.st{font-size:11px;padding:0 .4rem;border-radius:4px;border:1px solid var(--line)}
.st.running{color:var(--acc)}.st.done{color:var(--ok)}.st.failed,.st.lost{color:var(--no)}
pre{background:var(--code);padding:.6rem;border-radius:6px;max-height:60vh;overflow:auto;font:12px ui-monospace,monospace;white-space:pre-wrap;word-break:break-all;margin:.5rem 0 0}
</style>
<header><h1>Studio · Jobs</h1><a href="/">← Studio</a><span class="note" id="hint"></span></header>
<main>
 <section class="card"><h2>New job</h2>
  <select id="kind"></select><div id="khelp" class="note" style="margin-top:.3rem"></div>
  <form id="form"></form>
  <button class="pri" id="run" style="margin-top:.7rem;width:100%">Run</button>
  <div id="ferr" class="err"></div>
 </section>
 <section class="card"><h2>Jobs</h2><div id="jobs" class="note">none yet</div>
  <div id="detail" style="display:none;margin-top:.8rem">
   <div style="display:flex;gap:.5rem;align-items:center"><b id="dtitle"></b><span id="dst" class="st"></span>
    <button id="cancel" style="margin-left:auto">Cancel</button></div>
   <div class="note" id="dargv" style="word-break:break-all;margin-top:.3rem"></div>
   <pre id="dlog"></pre></div>
 </section>
</main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let specs={}, cur=null;
async function api(p,o){const r=await fetch(p,o);const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.status);return j}
function renderForm(){const k=$('#kind').value, s=specs[k]; $('#khelp').textContent=s.help;
 $('#form').innerHTML=s.fields.map(f=>{const id='f_'+f.name, lab=esc(f.name.replace(/_/g,' '))+(f.required?' *':'');
  const d=Array.isArray(f.default)?f.default.join(' '):(f.default??'');
  if(f.kind==='bool')return `<label><input type="checkbox" id="${id}">${lab}${f.help?` <span class="note">${esc(f.help)}</span>`:''}</label>`;
  const inp=f.choices?`<select id="${id}">${f.choices.map(c=>`<option ${c===d?'selected':''}>${esc(c)}</option>`).join('')}</select>`
   :`<input id="${id}" value="${esc(d)}" placeholder="${esc(f.help||'')}">`;
  return `<label for="${id}">${lab}${f.multi?' <span class="note">(space-separated)</span>':''}</label>${inp}`}).join('');}
async function load(){specs=await api('/api/jobs/specs');const g={};for(const[k,s]of Object.entries(specs))(g[s.group]??=[]).push([k,s]);
 $('#kind').innerHTML=Object.entries(g).map(([n,xs])=>`<optgroup label="${esc(n)}">${xs.map(([k,s])=>`<option value="${k}">${esc(s.title)}</option>`).join('')}</optgroup>`).join('');
 const want=location.hash.slice(1); if(specs[want])$('#kind').value=want; renderForm(); refresh();}
$('#kind').onchange=()=>{renderForm();location.hash=$('#kind').value};
$('#run').onclick=async()=>{const k=$('#kind').value, v={}; $('#ferr').textContent='';
 for(const f of specs[k].fields){const el=$('#f_'+f.name); v[f.name]=f.kind==='bool'?el.checked:el.value.trim()}
 try{const j=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:k,values:v})});cur=j.id;refresh()}
 catch(e){$('#ferr').textContent=e.message}};
$('#cancel').onclick=async()=>{if(cur){await api('/api/jobs/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:cur})});refresh()}};
async function refresh(){const js=await api('/api/jobs');
 $('#jobs').innerHTML=js.length?js.map(j=>`<div class="job" data-id="${j.id}"><span class="st ${j.status}">${j.status}</span><span>${esc(j.title)}</span><span class="note" style="margin-left:auto">${new Date(j.started*1000).toLocaleString()}</span></div>`).join(''):'none yet';
 document.querySelectorAll('.job').forEach(e=>e.onclick=()=>{cur=e.dataset.id;refresh()});
 if(cur){const j=js.find(x=>x.id===cur); if(j){$('#detail').style.display='';$('#dtitle').textContent=j.title;$('#dst').textContent=j.status;$('#dst').className='st '+j.status;
  $('#dargv').textContent=j.argv.join(' ');$('#cancel').disabled=j.status!=='running';
  const l=await api('/api/jobs/'+cur+'/log');const pre=$('#dlog'),atEnd=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-20;pre.textContent=l.log||'(no output yet)';if(atEnd)pre.scrollTop=pre.scrollHeight}}}
setInterval(()=>refresh().catch(()=>{}),2000);
load().catch(e=>{$('#ferr').textContent=e.message});
</script>"""
