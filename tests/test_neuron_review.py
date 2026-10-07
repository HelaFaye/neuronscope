"""Neuron review: the observation store, its statistics over time and the 3D
payload, and ingest from TestQA results and graded JSONL."""
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "viz"))

import neuron_review as nr  # noqa: E402

DAY = 86400.0
T0 = time.mktime((2026, 3, 2, 12, 0, 0, 0, 0, -1))      # a Monday, noon local time


def fill(store, model="m", n=24, L=3, B=8, seed=0):
    """Wrong answers light up layer 1 bin 2; right answers layer 0 bin 5.
    Half the observations are a week later."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        wrong = i % 2 == 0
        v = rng.normal(1.0, 0.05, (L, B))
        if wrong:
            v[1, 2] += 1.0
        else:
            v[0, 5] += 1.0
        store.record(model, v, source="testqa" if i % 3 else "livebench", kind="test" if i % 3 else "benchmark",
                     subjects=["math"] if i % 4 < 2 else ["code"], verdict="wrong" if wrong else "correct",
                     risk=0.9 if wrong else 0.1, t=T0 + (7 * DAY if i >= n // 2 else 0) + i)


def test_record_query_label_and_facets(tmp_path):
    s = nr.ReviewStore(tmp_path)
    fill(s)
    assert len(s.query("m")) == 24
    assert {r["source"] for r in s.query("m", sources=["livebench"])} == {"livebench"}
    assert all(r["kind"] == "test" for r in s.query("m", kinds=["test"]))
    assert all("code" in r["subjects"] for r in s.query("m", subjects=["code"]))
    assert len(s.query("m", since=T0 + 3 * DAY)) == 12
    f = s.facets("m")
    assert f["n"] == 24 and f["sources"]["livebench"]["kind"] == "benchmark" and set(f["subjects"]) == {"math", "code"}
    oid = s.query("m", verdicts=["correct"])[0]["id"]
    s.label("m", oid, "wrong")
    r = next(r for r in s.query("m") if r["id"] == oid)
    assert r["verdict"] == "wrong" and r["labelled"]
    with pytest.raises(KeyError):
        s.label("m", "nope", "wrong")
    with pytest.raises(ValueError):
        s.record("m", np.zeros((2, 2)), source="x", kind="nonsense")
    assert s.models()[0]["observations"] == 24


def test_association_finds_the_planted_neurons(tmp_path):
    s = nr.ReviewStore(tmp_path)
    fill(s)
    r = nr.summary(s, "m", "association", "week")
    assert r["n_wrong"] == 12 and r["n_right"] == 12 and r["note"] == ""
    top = {(t["layer"], t["bin"]): t["value"] for t in r["top"][:2]}
    assert top[(1, 2)] > 3 and top[(0, 5)] < -3
    assert [b["label"] for b in r["timeline"]] == ["2026-W10", "2026-W11"]
    assert r["timeline"][0]["error_rate"] == 0.5
    # Filters narrow it; too few labels say why instead of guessing.
    few = nr.summary(s, "m", "association", sources=["livebench"], subjects=["code"])
    assert few["n"] < 24
    one = nr.summary(s, "m", "association", verdicts=["wrong"])
    assert one["note"].startswith("needs at least 2 right") and one["top"] == []
    risk = nr.summary(s, "m", "risk")
    assert {(t["layer"], t["bin"]) for t in risk["top"][:2]} == {(1, 2), (0, 5)}
    fire = nr.summary(s, "m", "firing")
    assert fire["grid"] is not None
    assert nr.summary(s, "nobody")["n"] == 0


def test_buckets():
    rows = [{"t": T0}, {"t": T0 + 1}, {"t": T0 + 40 * DAY}]
    assert [len(i) for _, i in nr.buckets(rows, "month")] == [2, 1]
    assert [len(i) for _, i in nr.buckets(rows, "day")] == [2, 1]
    assert [len(i) for _, i in nr.buckets(rows, "all")] == [3]
    assert [lab for lab, _ in nr.buckets(rows, "week", every=2)] == ["#1–2", "#3–3"]


def test_view_payload_decodes_like_bloom(tmp_path):
    s = nr.ReviewStore(tmp_path)
    fill(s)
    blob, meta = nr.view_payload(s, "m", "association", "week")
    T, n, L = struct.unpack_from("<iii", blob)
    assert (T, L) == (2, 3) and meta["frames"] == 2 and meta["cells"] == n and meta["mode"] == "review"
    o = 12
    ly = np.frombuffer(blob, "<i4", n, o); o += 4 * n
    nx = np.frombuffer(blob, "<i4", n, o); o += 4 * n
    inten = np.frombuffer(blob, "<f4", T * n, o).reshape(T, n); o += 4 * T * n
    state = np.frombuffer(blob, np.uint8, T * n, o).reshape(T, n)
    assert o + T * n == len(blob)
    cells = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(ly, nx))}
    assert (state[:, cells[(1, 2)]] == 2).all() and (state[:, cells[(0, 5)]] == 1).all()
    assert inten.max() <= 1.0
    assert meta["prob"] == [0.5, 0.5] and meta["legend"]["halluc"].startswith("fires more on halluc")
    with pytest.raises(ValueError):
        nr.view_payload(s, "m", sources=["nothing"])


def test_mixed_shapes_are_refused(tmp_path):
    s = nr.ReviewStore(tmp_path)
    s.record("m", np.zeros((2, 4)), source="a", kind="test")
    s.record("m", np.zeros((3, 4)), source="a", kind="test")
    with pytest.raises(ValueError, match="different shapes"):
        nr.summary(s, "m")


def test_bin_profile():
    v = nr.bin_profile(np.arange(2 * 2048, dtype=np.float32).reshape(2, 2048))
    assert v.shape == (2, 512) and v.dtype == np.float16 and float(v[0, 0]) == 3.0


class FakeScorer:
    gguf = "/models/Tiny-Model.Q4.gguf"     # Studio's id: models/tiny-model.q4

    def __init__(self):
        self.seen = []

    def profile(self, messages, response):
        self.seen.append((messages[-1]["content"], response))
        v = np.ones((2, 4), np.float32) * (2.0 if "wrong" in response else 1.0)
        return {"cett": v, "n_tokens": len(response), "prob": 0.7}


def test_testqa_items_and_cli(tmp_path, monkeypatch):
    bank = tmp_path / "bank"
    bank.mkdir()
    tasks = [{"id": "a1", "kind": "factual", "subject": "history", "prompt": "When?", "answer": "1066"},
             {"id": "a2", "kind": "reasoning", "subject": "math", "prompt": "2+2?", "answer": "4"},
             {"id": "v1", "kind": "factual", "subject": "vision", "prompt": "What is shown?", "answer": "cat",
              "image": "x.png"}]
    (bank / "t.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    res = {"endpoints": {"mine": {"url": "http://x", "model": "tiny-model"}},
           "raw": {"mine": [{"id": "a1", "verdict": "correct", "subject": "history", "text": "1066"},
                            {"id": "a2", "verdict": "wrong", "subject": "math", "text": "wrong: 5"},
                            {"id": "v1", "verdict": "correct", "text": "cat"},
                            {"id": "a3", "verdict": "error", "text": ""}]}}
    rp = tmp_path / "run.json"
    rp.write_text(json.dumps(res))
    items, model = nr.testqa_items(str(rp), None, [str(bank)])
    assert model == "tiny-model" and [i["id"] for i in items] == ["a1", "a2"]
    assert items[1]["prompt"].startswith("2+2?") and items[1]["verdict"] == "wrong"

    fake = FakeScorer()
    monkeypatch.setattr(nr, "_scorer", lambda a: fake)
    monkeypatch.setattr(nr, "testqa_items", lambda path, ep, bank=None: (items, model))
    assert nr.main(["ingest-testqa", "--root", str(tmp_path / "rv"), "--results", str(rp)]) == 0
    s = nr.ReviewStore(tmp_path / "rv")
    rows = s.query("models/tiny-model.q4")
    assert [r["verdict"] for r in rows] == ["correct", "wrong"] and rows[0]["kind"] == "test"
    assert rows[1]["subjects"] == ["math"] and rows[0]["risk"] == 0.7
    assert s.vecs("models/tiny-model.q4", rows).shape == (2, 2, 4)

    graded = tmp_path / "livebench.jsonl"
    graded.write_text(json.dumps({"prompt": "p", "response": "ok", "verdict": "PASS", "subject": "code"}) + "\n" +
                      json.dumps({"prompt": "q", "response": "r", "verdict": "maybe"}) + "\n")
    assert nr.main(["ingest-items", "--root", str(tmp_path / "rv"), "--items", str(graded)]) == 0
    rows = s.query("models/tiny-model.q4", sources=["livebench"])
    assert len(rows) == 1 and rows[0]["kind"] == "benchmark" and rows[0]["verdict"] == "correct"


def test_studio_id_matches_studio():
    import studio
    for p in ["/m/acme/Coder-GGUF/coder-7b-Q4_K_M.gguf", "/m/x/big-00001-of-00003.gguf"]:
        assert nr.studio_id(p) == studio.model_id(p)


def test_noise_stays_dark(tmp_path):
    """No real effect: the noise floor keeps chance differences from lighting up."""
    s = nr.ReviewStore(tmp_path)
    rng = np.random.default_rng(3)
    for i in range(40):
        s.record("m", rng.gamma(2.0, 0.05, (8, 64)), source="t", kind="test",
                 verdict="wrong" if i % 2 else "correct", t=T0 + (i // 10) * 31 * DAY)
    r = nr.summary(s, "m", "association", "month")
    assert r["floor"] >= 0.6 and len(r["top"]) <= 3
    _, meta = nr.view_payload(s, "m", "association", "month")
    assert meta["cells"] <= 0.01 * 8 * 64
