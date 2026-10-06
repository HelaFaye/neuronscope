"""Studio's OpenAI-compatible API, JIT loading, routing, TTL and chat storage
against a fake llama-server."""
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import model_stats  # noqa: E402
import studio  # noqa: E402

CODER = "coder-gguf/coder-7b-q4_k_m"
VLM = "vision-gguf/vlm-3b-q8_0"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def studio_srv(tmp_path):
    models = tmp_path / "models"
    (models / "acme" / "Coder-GGUF").mkdir(parents=True)
    (models / "acme" / "Vision-GGUF").mkdir(parents=True)
    (models / "acme" / "Coder-GGUF" / "coder-7b-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    (models / "acme" / "Vision-GGUF" / "vlm-3b-Q8_0.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    (models / "acme" / "Vision-GGUF" / "mmproj-vlm-3b-f16.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    studio.STATE.update({
        "models_dirs": [str(models)], "server_bin": str(ROOT / "tests" / "fake_llama_server.py"),
        "settings_path": str(tmp_path / "studio.json"), "chats_dir": str(tmp_path / "chats"),
        "token": None, "stats": model_stats.StatsStore(tmp_path / "stats"), "min_graded": 20,
        "min_subject": 5, "halluc_cost": 1.0, "cett": None, "idle_ttl": 0, "jit": True, "backend_port": free_port(), "tls": False,
        "jobs": studio.ns_jobs.JobRunner(tmp_path / "jobs", 1), "jobs_off": None,
        "rag": None, "rag_dir": str(tmp_path / "rag"), "rag_embed": None, "rag_embed_gguf": None,
        "devices": studio.ns_pairing.DeviceRegistry(tmp_path / "devices.json"),
        "links_path": str(tmp_path / "links.json"), "fingerprint": None})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), studio.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    studio.stop_server()


def call(base, path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_models_skip_mmproj_and_pair_it(studio_srv):
    _, ms = call(studio_srv, "/v1/models")
    ids = [m["id"] for m in ms["data"]]
    assert "auto" not in ids                      # nothing has stats yet
    assert CODER in ids and VLM in ids
    assert not any("mmproj" in i for i in ids)
    assert next(m for m in ms["data"] if m["id"] == VLM)["vision"] is True


def seed(base, model, subject, correct, wrong, abstained=0):
    recs = ([{"subject": subject, "verdict": "correct"}] * correct + [{"subject": subject, "verdict": "wrong"}] * wrong
            + [{"subject": subject, "verdict": "abstained"}] * abstained)
    code, r = call(base, "/api/stats/ingest", {"model": model, "records": recs})
    assert code == 200
    return r


def test_jit_load_swap_and_vision(studio_srv):
    code, r = call(studio_srv, "/v1/chat/completions",
                   {"model": "coder-gguf/coder-7b-q4_k_m", "messages": [{"role": "user", "content": "hi"}]})
    assert code == 200 and "model=coder-gguf/coder-7b-q4_k_m" in r["choices"][0]["message"]["content"]
    img = [{"type": "text", "text": "what is this"},
           {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    code, r = call(studio_srv, "/v1/chat/completions", {"model": "coder-gguf/coder-7b-q4_k_m",
                                                        "messages": [{"role": "user", "content": img}]})
    assert code == 400 and "mmproj" in r["error"]["message"]
    code, r = call(studio_srv, "/v1/chat/completions", {"model": "vision-gguf/vlm-3b-q8_0",
                                                        "messages": [{"role": "user", "content": img}]})
    text = r["choices"][0]["message"]["content"]
    assert code == 200 and "mmproj=yes" in text and "image=yes" in text
    _, st = call(studio_srv, "/api/status")
    assert st["id"] == "vision-gguf/vlm-3b-q8_0" and st["vision"]
    code, r = call(studio_srv, "/v1/chat/completions", {"model": "nope", "messages": []})
    assert code == 404


def test_auto_refuses_without_stats_but_manual_still_works(studio_srv):
    code, r = call(studio_srv, "/v1/chat/completions", {"model": "auto", "messages": [
        {"role": "user", "content": "Write a Python function"}]})
    assert code == 409 and "performance stats" in r["error"]["message"]
    code, r = call(studio_srv, "/v1/chat/completions", {"model": CODER, "messages": [
        {"role": "user", "content": "hi"}]})
    assert code == 200


def test_auto_routes_on_stats(studio_srv):
    assert seed(studio_srv, CODER, "code", 40, 5)["recorded"] == 45
    seed(studio_srv, CODER, "writing", 2, 18)
    seed(studio_srv, VLM, "writing", 18, 2)
    seed(studio_srv, VLM, "vision", 20, 2)
    _, ms = call(studio_srv, "/v1/models")
    assert ms["data"][0]["id"] == "auto"
    _, r = call(studio_srv, "/v1/chat/completions", {"model": "auto", "messages": [
        {"role": "user", "content": "Write a Python function that parses JSON and fix this bug"}]})
    assert f"model={CODER}" in r["choices"][0]["message"]["content"]
    _, r = call(studio_srv, "/v1/chat/completions", {"model": "auto", "messages": [
        {"role": "user", "content": "Write a short poem about autumn, in the tone of a letter"}]})
    assert f"model={VLM}" in r["choices"][0]["message"]["content"]
    _, r = call(studio_srv, "/v1/chat/completions", {"model": "auto", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Fix this Python bug, see the screenshot"},
                                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]})
    assert f"model={VLM}" in r["choices"][0]["message"]["content"]   # only vision models are candidates


def test_stats_surface_and_warning_reason(studio_srv):
    seed(studio_srv, CODER, "code", 10, 2)
    _, models = call(studio_srv, "/api/models")
    by = {m["id"]: m["stats"] for m in models}
    assert not by[CODER]["eligible"] and "12 graded" in by[CODER]["why_not"]
    assert by[VLM]["why_not"].startswith("No performance stats")
    code, r = call(studio_srv, "/api/stats/ingest", {"model": "nope", "records": []})
    assert code == 404


def test_live_and_activation_recording(studio_srv, monkeypatch):
    import hscore

    class FakeScorer:
        def __init__(self, *a, **k):
            pass

        def score(self, messages, text):
            return {"score": 1.5, "prob": 0.8, "n_tokens": 7}
    monkeypatch.setattr(hscore, "HScorer", FakeScorer)
    studio.STATE["cett"] = "/bin/true"
    threading.Thread(target=studio.score_worker, daemon=True).start()
    settings = {studio._key(m["path"]): {"classifier": "x.npz"} for m in studio.scan_models()}
    studio.save_settings(settings)
    for _ in range(3):
        call(studio_srv, "/v1/chat/completions", {"model": CODER, "messages": [{"role": "user", "content": "hi"}]})
    deadline = time.time() + 10
    while time.time() < deadline:
        _, st = call(studio_srv, "/api/stats")
        if st["models"][CODER]["activation"]["n"] >= 1:
            break
        time.sleep(0.2)
    s = st["models"][CODER]
    studio.STATE["cett"] = None
    assert s["live"]["n"] == 3 and s["live"]["abstention_rate"] == 0.0
    assert s["activation"]["n"] >= 1 and s["activation"]["mean"] == 1.5


def test_streaming_passthrough(studio_srv):
    req = urllib.request.Request(studio_srv + "/v1/chat/completions", method="POST",
                                 data=json.dumps({"model": "coder-gguf/coder-7b-q4_k_m", "stream": True,
                                                  "messages": [{"role": "user", "content": "x"}]}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode()
    assert body.count("data:") >= 3 and "[DONE]" in body and "predicted_per_second" in body


def test_idle_ttl_unloads(studio_srv):
    call(studio_srv, "/v1/chat/completions", {"model": "coder-gguf/coder-7b-q4_k_m",
                                              "messages": [{"role": "user", "content": "x"}]})
    assert studio.server_running()
    studio.STATE["idle_ttl"] = 1
    studio.ACTIVITY["last"] = time.time() - 10
    threading.Thread(target=studio.idle_reaper, daemon=True).start()
    deadline = time.time() + 15
    while studio.server_running() and time.time() < deadline:
        time.sleep(0.5)
    studio.STATE["idle_ttl"] = 0
    assert not studio.server_running()


def test_chat_history_roundtrip(studio_srv):
    _, saved = call(studio_srv, "/api/chats", {"title": "t1", "messages": [{"role": "user", "content": "hi"}]})
    _, lst = call(studio_srv, "/api/chats")
    assert [c["id"] for c in lst] == [saved["id"]] and lst[0]["n"] == 1
    _, got = call(studio_srv, f"/api/chats/{saved['id']}")
    assert got["messages"][0]["content"] == "hi"
    code, _ = call(studio_srv, "/api/chats/../../etc")
    assert code == 404
    call(studio_srv, "/api/chats/delete", {"id": saved["id"]})
    assert call(studio_srv, "/api/chats")[1] == []


def test_refuses_open_lan_bind():
    assert studio.main(["--host", "0.0.0.0", "--port", "0"]) == 2


def test_jobs_run_log_and_reject_injection(studio_srv, tmp_path):
    code, specs = call(studio_srv, "/api/jobs/specs")
    assert code == 200 and {"testqa", "deficits", "finetune", "swe_train"} <= set(specs)
    rep = tmp_path / "rep.json"
    rep.write_text(json.dumps({"resolved_ids": ["a"], "unresolved_ids": ["b"], "empty_patch_ids": []}))
    code, job = call(studio_srv, "/api/jobs", {"kind": "swe_import", "values": {"report": str(rep), "model": "m"}})
    assert code == 200 and job["status"] == "running"
    for _ in range(100):
        _, js = call(studio_srv, "/api/jobs")
        if js[0]["status"] != "running":
            break
        time.sleep(0.1)
    assert js[0]["status"] == "done"
    _, log = call(studio_srv, f"/api/jobs/{job['id']}/log")
    assert "1/2 resolved" in log["log"]
    code, err = call(studio_srv, "/api/jobs", {"kind": "swe_import", "values": {"report": "--help", "model": "m"}})
    assert code == 400 and "may not start" in err["error"]
    code, _ = call(studio_srv, "/api/jobs", {"kind": "rm", "values": {}})
    assert code == 400
    # a cross-site form post cannot start a job
    req = urllib.request.Request(studio_srv + "/api/jobs", data=b"kind=swe_import", method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 415


def test_jobs_cancel_and_off_on_network_bind(studio_srv, tmp_path, monkeypatch):
    runner = studio.STATE["jobs"]
    monkeypatch.setitem(studio.ns_jobs.SPECS, "sleep", {"title": "sleep", "group": "t", "help": "",
                        "argv": ["-c", "import time; time.sleep(30)"], "fields": []})
    code, job = call(studio_srv, "/api/jobs", {"kind": "sleep", "values": {}})
    assert code == 200
    code, err = call(studio_srv, "/api/jobs", {"kind": "sleep", "values": {}})
    assert code == 429                                        # --max-jobs 1
    call(studio_srv, "/api/jobs/cancel", {"id": job["id"]})
    for _ in range(50):
        if runner.get(job["id"])["ended"]:
            break
        time.sleep(0.1)
    assert runner.get(job["id"])["status"] == "cancelled"
    monkeypatch.setitem(studio.STATE, "jobs", None)
    monkeypatch.setitem(studio.STATE, "jobs_off", "jobs are off on a network bind")
    code, err = call(studio_srv, "/api/jobs")
    assert code == 403 and "network bind" in err["error"]


def sse_chat(base, body):
    req = urllib.request.Request(base + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=30) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                events.append(json.loads(line[5:]))
    return events


def _upload(base, coll, name, text):
    import base64
    return call(base, "/api/rag/upload", {"collection": coll, "filename": name,
                                          "data": base64.b64encode(text.encode()).decode()})


def test_rag_collections_and_chat_citations(studio_srv):
    code, _ = call(studio_srv, "/api/rag/create", {"collection": "kb"})
    assert code == 200
    code, r = _upload(studio_srv, "kb", "falcon.md", "The Falcon 9 booster lands on a drone ship.\n\n"
                                                        "Paris is the capital of France.")
    assert code == 200 and r["chunks"] >= 1
    _upload(studio_srv, "kb", "cats.txt", "Cats sleep for most of the day.")
    _, info = call(studio_srv, "/api/rag")
    (kb,) = info["collections"]
    assert kb["docs"] == 2 and not kb["dense"]
    code, hits = call(studio_srv, "/api/rag/search", {"collection": "kb", "query": "where does the booster land"})
    assert hits[0]["source"] == "falcon.md"
    ev = sse_chat(studio_srv, {"model": CODER, "rag": {"collection": "kb", "k": 2},
                               "messages": [{"role": "user", "content": "Where does the Falcon booster land?"}]})
    assert ev[0]["sources"][0]["source"] == "falcon.md"
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev if "choices" in e)
    assert "ctx=yes" in text                              # the passages reached the model
    ev = sse_chat(studio_srv, {"model": CODER, "messages": [{"role": "user", "content": "hi"}]})
    assert "sources" not in ev[0]
    # bad names cannot escape the collection root
    code, err = call(studio_srv, "/api/rag/create", {"collection": "../etc"})
    assert code == 400
    _, docs = call(studio_srv, "/api/rag/docs?c=kb")
    call(studio_srv, "/api/rag/delete", {"collection": "kb", "doc": next(d["doc"] for d in docs if d["source"] == "cats.txt")})
    _, info = call(studio_srv, "/api/rag")
    assert info["collections"][0]["docs"] == 1
    call(studio_srv, "/api/rag/delete", {"collection": "kb"})
    assert call(studio_srv, "/api/rag")[1]["collections"] == []


def test_rag_dense_with_embedding_sidecar(studio_srv, monkeypatch):
    monkeypatch.setitem(studio.STATE, "rag_embed_gguf", "/models/embed.gguf")   # the fake server ignores -m
    try:
        _upload(studio_srv, "dense", "a.txt", "alpha bravo charlie")
        _upload(studio_srv, "dense", "b.txt", "rocket booster landing")
        _, info = call(studio_srv, "/api/rag")
        assert info["embedder"] and info["collections"][0]["dense"]
        _, hits = call(studio_srv, "/api/rag/search", {"collection": "dense", "query": "booster landing", "k": 1})
        assert hits[0]["source"] == "b.txt"
    finally:
        p = studio.EMBED["proc"]
        if p is not None:
            p.terminate()
            studio.EMBED["proc"] = None


def _mcp_cfg(tmp_path, auto):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"calc": {"command": sys.executable,
                                                        "args": [str(ROOT / "tests" / "fake_mcp_server.py")],
                                                        "autoApprove": auto}}}))
    return str(cfg)


@pytest.mark.parametrize("auto,allow", [(["add"], None), ([], True), ([], False)])
def test_mcp_tool_calls_with_approval(studio_srv, tmp_path, monkeypatch, auto, allow):
    pytest.importorskip("mcp")
    monkeypatch.setitem(studio.STATE, "mcp_config", _mcp_cfg(tmp_path, auto))
    monkeypatch.setitem(studio.STATE, "mcp", None)
    try:
        _, st = call(studio_srv, "/api/mcp")
        (srv,) = st["servers"]
        assert srv["connected"] and {t["name"] for t in srv["tools"]} == {"add", "echo", "boom"}
        approver = None
        if allow is not None:
            def approve():
                for _ in range(200):
                    with studio.APPROVALS_LOCK:
                        keys = list(studio.APPROVALS)
                    if keys:
                        call(studio_srv, "/api/tools/approve", {"key": keys[0], "allow": allow})
                        return
                    time.sleep(0.05)
            approver = threading.Thread(target=approve)
            approver.start()
        ev = sse_chat(studio_srv, {"model": CODER, "tools": True,
                                   "messages": [{"role": "user", "content": "what is 2+40?"}]})
        if approver:
            approver.join()
        (tc,) = [e["tool_call"] for e in ev if "tool_call" in e]
        (tr,) = [e["tool_result"] for e in ev if "tool_result" in e]
        assert tc["name"] == "calc__add" and tc["arguments"] == {"a": 2, "b": 40}
        assert tc["needs_approval"] is (allow is not None)
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev if "choices" in e)
        if allow is False:
            assert not tr["ok"] and "declined" in tr["text"] and "declined" in text
        else:
            assert tr["ok"] and tr["text"] == "42" and text == "tool said: 42"
    finally:
        if studio.STATE.get("mcp"):
            studio.STATE["mcp"].close()


@pytest.fixture()
def remote_studio(tmp_path):
    """A second Studio in its own process: token + self-signed TLS, one model."""
    import subprocess
    pytest.importorskip("cryptography")
    d = tmp_path / "remote"
    (d / "models" / "acme" / "Big-GGUF").mkdir(parents=True)
    (d / "models" / "acme" / "Big-GGUF" / "big-70b-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    sec = __import__("ns_security")
    token = sec.generate_token()
    sec.write_secret_file(d / "token", token)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "ns_security.py"), "selfsigned", "--host", "127.0.0.1",
                    "--cert", str(d / "c.pem"), "--key", str(d / "k.pem")], check=True, capture_output=True)
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "viz" / "studio.py"), "--models-dir", str(d / "models"),
                             "--server", str(ROOT / "tests" / "fake_llama_server.py"), "--port", str(port),
                             "--backend-port", str(free_port()), "--token-file", str(d / "token"),
                             "--tls-cert", str(d / "c.pem"), "--tls-key", str(d / "k.pem"),
                             "--devices", str(d / "devices.json"), "--links", str(d / "links.json"),
                             "--mcp-config", str(d / "none.json"), "--rag-dir", str(d / "rag"),
                             "--jobs-dir", str(d / "jobs"), "--settings", str(d / "s.json"),
                             "--chats-dir", str(d / "chats"), "--stats-dir", str(d / "stats")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    base = f"https://127.0.0.1:{port}"
    import ns_pairing
    fp = ns_pairing.cert_fingerprint(str(d / "c.pem"))
    for _ in range(100):
        try:
            ns_pairing.request(base, "GET", "/api/status", token=token, fingerprint=fp, timeout=2)
            break
        except Exception:
            time.sleep(0.1)
    yield base, token, fp
    proc.terminate()
    proc.wait(10)


def test_pairing_and_linked_host(studio_srv, remote_studio):
    import ns_pairing
    base, token, fp = remote_studio
    # the owner creates a link; it carries the certificate fingerprint
    r = ns_pairing.request(base, "POST", "/api/pair/start", {}, token=token, fingerprint=fp)
    assert r["fingerprint"] == fp and f"fp={fp}" in r["link"]
    link = r["link"].replace(r["link"].split("/pair")[0], base)
    # a wrong fingerprint is refused before the code is ever sent
    bad = link.replace(fp, "0" * 64)
    code, err = call(studio_srv, "/api/links/add", {"link": bad, "name": "gpu"})
    assert code == 400 and "fingerprint" in err["error"]
    code, r2 = call(studio_srv, "/api/links/add", {"link": link, "name": "gpu"})
    assert code == 200 and r2["models"] == ["gpu:big-gguf/big-70b-q4_k_m"], r2
    _, ms = call(studio_srv, "/v1/models")
    remote_ids = [m["id"] for m in ms["data"] if m.get("owned_by") == "link:gpu"]
    assert len(remote_ids) == 1 and remote_ids[0].startswith("gpu:")
    # chat through the link: the remote Studio loads and answers
    ev = sse_chat(studio_srv, {"model": remote_ids[0], "messages": [{"role": "user", "content": "hi"}]})
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev if "choices" in e)
    assert text.startswith("model=") and "big" in text
    code, out = call(studio_srv, "/v1/chat/completions", {"model": remote_ids[0],
                                                          "messages": [{"role": "user", "content": "hi"}]})
    assert code == 200 and "big" in out["choices"][0]["message"]["content"]
    # the code was single use
    with pytest.raises(RuntimeError, match="403"):
        ns_pairing.claim(link, "again")
    # the device token works for models but not for owner actions
    links = json.loads(open(studio.STATE["links_path"]).read())["links"]
    dev_tok = links[0]["token"]
    assert ns_pairing.request(base, "GET", "/v1/models", token=dev_tok, fingerprint=fp)["data"]
    with pytest.raises(RuntimeError, match="403"):
        ns_pairing.request(base, "POST", "/api/pair/start", {}, token=dev_tok, fingerprint=fp)
    for path, body in [("/api/jobs", {"kind": "swe_import", "values": {}}), ("/api/load", {"id": "x"}),
                       ("/api/download", {"repo": "a/b"}), ("/api/mcp/reload", {})]:
        with pytest.raises(RuntimeError, match="403"):
            ns_pairing.request(base, "POST", path, body, token=dev_tok, fingerprint=fp)
    with pytest.raises(RuntimeError, match="403"):
        ns_pairing.request(base, "GET", "/api/jobs", token=dev_tok, fingerprint=fp)
    # revoking the device cuts the link off
    devs = ns_pairing.request(base, "GET", "/api/devices", token=token, fingerprint=fp)["devices"]
    assert [d["name"] for d in devs] == ["studio"]
    ns_pairing.request(base, "POST", "/api/devices/revoke", {"id": devs[0]["id"]}, token=token, fingerprint=fp)
    with pytest.raises(RuntimeError, match="401"):
        ns_pairing.request(base, "GET", "/v1/models", token=dev_tok, fingerprint=fp)
    # the token file on this side is private
    assert oct(os.stat(studio.STATE["links_path"]).st_mode & 0o777) == "0o600"


def test_device_registry_codes_expire_and_are_hashed(tmp_path, monkeypatch):
    import ns_pairing
    reg = ns_pairing.DeviceRegistry(tmp_path / "d.json")
    code, _, _ = reg.new_code()
    assert reg.claim("WRONGCODE000", "x") is None
    r = reg.claim(code.lower()[:4] + "-" + code[4:], "laptop")          # dashes and case do not matter
    assert r and reg.check(r["token"])["name"] == "laptop"
    assert r["token"] not in (tmp_path / "d.json").read_text()          # only the hash is stored
    code2, _, _ = reg.new_code()
    monkeypatch.setattr(ns_pairing.time, "time", lambda: 10 ** 12)      # far future
    assert reg.claim(code2, "late") is None


def test_multi_gpu_flags_and_vram_fit(studio_srv, monkeypatch):
    seen = {}

    class P:
        def __init__(self, cmd, **kw):
            seen["cmd"], seen["env"] = cmd, kw.get("env")
        def poll(self):
            return None
        def terminate(self):
            pass
        def wait(self, timeout=None):
            return 0
        kill = terminate
    monkeypatch.setattr(studio.subprocess, "Popen", P)
    monkeypatch.setattr(studio.time, "sleep", lambda s: None)
    monkeypatch.setattr(studio, "wait_healthy", lambda port, proc: (True, "ready"))
    monkeypatch.setattr(studio, "port_busy", lambda port: False)
    m = {"path": "/m.gguf", "id": "m", "name": "m.gguf"}
    ok, _ = studio.start_server(m, {**studio.DEFAULTS, "gpus": "0,1,2,3", "split_mode": "layer",
                                    "tensor_split": "1,1,1,1", "main_gpu": 0}, 9999)
    assert ok and seen["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    c = seen["cmd"]
    assert c[c.index("-sm") + 1] == "layer" and c[c.index("-ts") + 1] == "1,1,1,1" and c[c.index("-mg") + 1] == "0"
    for bad in ({"gpus": "0;rm"}, {"split_mode": "fast"}, {"tensor_split": "1,a"}):
        ok, why = studio.start_server(m, {**studio.DEFAULTS, **bad}, 9999)
        assert not ok, bad
    studio.PROC.update({"proc": None, "model": None, "port": None})
    import cuda_info
    m10 = cuda_info.parse_smi("".join(f"{i}, Tesla M10, 5.0, 8192, 8000, 580.95.05, b{i}\n" for i in range(4)))
    monkeypatch.setattr(studio, "nvidia_gpus", lambda: m10)
    assert studio.fit_estimate(20 * 2**30)["ok"] is True and "4 GPUs" in studio.fit_estimate(20 * 2**30)["note"]
    assert studio.fit_estimate(40 * 2**30)["ok"] is False
    code, adv = call(studio_srv, "/api/gpus")
    assert code == 200 and adv["count"] == 4 and adv["multi_gpu"]["llama_server"] == "-sm layer -ts 1,1,1,1"


def test_pairing_policy_persistent_and_temporary(tmp_path, monkeypatch):
    import json as _json
    import ns_pairing
    clock = [1_000_000.0]
    monkeypatch.setattr(ns_pairing.time, "time", lambda: clock[0])
    pol = ns_pairing.PairingPolicy(code_ttl="2m", presets=["1h", "8h", "7d"], max_ttl="30d")
    reg = ns_pairing.DeviceRegistry(tmp_path / "d.json", pol)
    # persistent (the default) and temporary pairings side by side
    c1, code_exp, acc = reg.new_code()
    assert acc is None and code_exp == clock[0] + 120
    keep = reg.claim(c1, "desktop")
    c2, _, acc = reg.new_code(persistent=False, ttl="8h")
    assert acc == 8 * 3600
    temp = reg.claim(c2, "phone")
    assert keep["expires"] is None and temp["expires"] == clock[0] + 8 * 3600
    clock[0] += 8 * 3600 - 1
    assert reg.check(temp["token"]) and reg.check(keep["token"])
    clock[0] += 2
    assert reg.check(temp["token"]) is None                  # temporary access ran out
    assert reg.check(keep["token"])                          # persistent access did not
    rows = {d["name"]: d for d in reg.list()}
    assert rows["phone"]["expired"] and rows["desktop"]["persistent"]
    # the owner renews, extends or makes persistent; the token is unchanged
    reg.update(temp["device_id"], persistent=False, ttl="1h")
    assert reg.check(temp["token"])
    reg.update(temp["device_id"], persistent=True)
    clock[0] += 400 * 86400
    assert reg.check(temp["token"])
    # host limits
    for kw in ({"persistent": False, "ttl": "31d"}, {"persistent": False, "ttl": "10s"}):
        try:
            reg.new_code(**kw)
            raise AssertionError(kw)
        except ValueError:
            pass
    strict = ns_pairing.DeviceRegistry(tmp_path / "s.json", ns_pairing.PairingPolicy(allow_persistent=False))
    try:
        strict.new_code(persistent=True)
        raise AssertionError("persistent allowed")
    except ValueError as e:
        assert "temporary" in str(e)
    assert strict.policy.describe()["default"] == "30d"      # default falls back to a temporary preset
    _, _, acc = strict.new_code()
    assert acc == 30 * 86400
    # pairing codes expire on the host's schedule
    c3, _, _ = reg.new_code(persistent=False, ttl="1h")
    clock[0] += 121
    assert reg.claim(c3, "late") is None
    # long-expired devices are pruned on the next pairing
    c4, _, _ = reg.new_code(persistent=False, ttl="1h")
    gone = reg.claim(c4, "short")
    clock[0] += 3600 + ns_pairing.EXPIRED_KEEP + 1
    reg.claim(reg.new_code()[0], "next")
    assert "short" not in {d["name"] for d in reg.list()}
    # device files from before expiry existed load as persistent
    (tmp_path / "old.json").write_text(_json.dumps({"devices": [{"id": "a", "name": "old", "token_sha256": "x",
                                                                 "created": 1, "last_seen": None}]}))
    assert ns_pairing.DeviceRegistry(tmp_path / "old.json").list()[0]["persistent"]
    assert ns_pairing.parse_duration("90m") == 5400 and ns_pairing.fmt_duration(7 * 86400) == "7d"


def test_temporary_pairing_end_to_end(remote_studio):
    import ns_pairing
    base, token, fp = remote_studio
    r = ns_pairing.request(base, "POST", "/api/pair/start", {"persistent": False, "ttl": "2h"},
                           token=token, fingerprint=fp)
    assert r["access_seconds"] == 7200 and not r["persistent"] and "&a=2h" in r["link"]
    code = ns_pairing.parse_link(r["link"].replace(r["link"].split("/pair")[0], base))[1]
    resp = ns_pairing.request(base, "POST", "/api/pair/claim", {"code": code, "name": "tablet", "cookie": True},
                              fingerprint=fp, stream=True)
    cookie = resp.getheader("Set-Cookie")
    claimed = json.loads(resp.read())
    assert 7100 < int(cookie.split("Max-Age=")[1].split(";")[0]) <= 7200      # the cookie ends with the access
    assert claimed["expires"] and ns_pairing.request(base, "GET", "/v1/models", token=claimed["token"],
                                                     fingerprint=fp)["data"]
    devs = ns_pairing.request(base, "GET", "/api/devices", token=token, fingerprint=fp)
    (dev,) = [d for d in devs["devices"] if d["name"] == "tablet"]
    assert not dev["persistent"] and devs["policy"]["allow_persistent"]
    up = ns_pairing.request(base, "POST", "/api/devices/update", {"id": dev["id"], "persistent": True},
                            token=token, fingerprint=fp)
    assert up["persistent"] and up["expires"] is None
    with pytest.raises(RuntimeError, match="400"):          # beyond the host's --pair-max (90d)
        ns_pairing.request(base, "POST", "/api/pair/start", {"persistent": False, "ttl": "400d"},
                           token=token, fingerprint=fp)
    with pytest.raises(RuntimeError, match="403"):          # a device cannot change its own access
        ns_pairing.request(base, "POST", "/api/devices/update", {"id": dev["id"], "persistent": True},
                           token=claimed["token"], fingerprint=fp)


def test_expired_link_is_reported_not_called(studio_srv):
    import json as _json
    studio.LINKS["cache"].clear()
    with open(studio.STATE["links_path"], "w") as f:
        _json.dump({"links": [{"name": "old", "url": "https://192.0.2.1:7870", "token": "t", "fingerprint": "",
                               "expires": time.time() - 60}]}, f)
    assert studio.link_models(refresh=True) == []          # no request to a host that would refuse it
    _, d = call(studio_srv, "/api/devices")
    (ln,) = d["links"]
    assert "expired" in ln["error"] and ln["expires"]
