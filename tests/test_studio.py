"""Studio's OpenAI-compatible API, JIT loading, routing, TTL and chat storage
against a fake llama-server."""
import json
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
        "min_subject": 5, "halluc_cost": 1.0, "cett": None, "idle_ttl": 0, "jit": True, "backend_port": free_port(), "tls": False})
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
