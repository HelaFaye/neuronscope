import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import model_stats as ms  # noqa: E402


def fill(store, model, subject, correct, wrong, abstained, size=None):
    for v, k in (("correct", correct), ("wrong", wrong), ("abstained", abstained)):
        for _ in range(k):
            store.record(model, "graded", size=size, subject=subject, verdict=v)


def test_summary_rates_and_window(tmp_path):
    st = ms.StatsStore(tmp_path, window=50)
    fill(st, "m", "code", 40, 5, 5)
    s = st.summary("m")["graded"]
    assert (s["n"], s["correct"], s["wrong"], s["abstained"]) == (50, 40, 5, 5)
    assert s["hallucination_rate"] == 0.1 and s["accuracy_ci"][0] < 0.8 < s["accuracy_ci"][1]
    fill(st, "m", "code", 0, 50, 0)            # newer evidence pushes the old out of the window
    assert st.summary("m")["graded"]["wrong"] == 50


def test_stats_are_tied_to_the_file(tmp_path):
    st = ms.StatsStore(tmp_path)
    fill(st, "m", "math", 10, 0, 0, size=100)
    fill(st, "m", "math", 0, 3, 0, size=200)   # a different file under the same name
    assert st.summary("m", size=100)["graded"]["n"] == 10
    assert st.summary("m", size=200)["graded"]["n"] == 3


def test_max_age(tmp_path):
    st = ms.StatsStore(tmp_path, max_age_days=1)
    rec = st.record("m", "graded", verdict="correct", subject="x")
    p = tmp_path / "m.jsonl"
    old = dict(rec, t=time.time() - 3 * 86400)
    p.write_text(json.dumps(old) + "\n" + p.read_text())
    assert st.summary("m")["graded"]["n"] == 1


def test_verdict_mapping_skips_ungraded(tmp_path):
    st = ms.StatsStore(tmp_path)
    assert st.record("m", "graded", verdict="answered") == {}
    assert st.record("m", "graded", verdict="timeout")["verdict"] == "wrong"


def test_rank_excludes_models_without_stats_and_prefers_low_hallucination(tmp_path):
    st = ms.StatsStore(tmp_path)
    fill(st, "careful", "factual", 60, 2, 38)     # answers less, rarely wrong
    fill(st, "reckless", "factual", 68, 32, 0)    # answers more, often wrong
    fill(st, "newbie", "factual", 5, 0, 0)        # too little evidence
    sums = {m: st.summary(m) for m in ("careful", "reckless", "newbie", "unknown")}
    pick = ms.rank({"factual": 1.0}, sums, min_graded=20)
    assert pick["model"] == "careful"
    assert set(pick["excluded"]) == {"newbie", "unknown"}
    # If wrong answers are free, raw accuracy wins.
    assert ms.rank({"factual": 1.0}, sums, min_graded=20, hallucination_cost=0.0)["model"] == "reckless"


def test_rank_uses_subject_specific_stats(tmp_path):
    st = ms.StatsStore(tmp_path)
    fill(st, "coder", "code", 45, 5, 0)
    fill(st, "coder", "writing", 2, 8, 0)
    fill(st, "writer", "writing", 18, 2, 0)
    fill(st, "writer", "code", 10, 20, 0)
    sums = {m: st.summary(m) for m in ("coder", "writer")}
    assert ms.rank({"code": 0.9, "writing": 0.1}, sums)["model"] == "coder"
    assert ms.rank({"writing": 0.9, "code": 0.1}, sums)["model"] == "writer"
    assert ms.rank({"math": 1.0}, sums)["candidates"][0]["subject_fallback"] == ["math"]


def test_no_eligible_models(tmp_path):
    pick = ms.rank({"code": 1.0}, {"a": ms.StatsStore(tmp_path).summary("a")})
    assert pick["model"] is None and "a" in pick["excluded"]


def test_hscore_with_fake_cett(tmp_path):
    import hscore

    class Tok:
        has_template = True

        def render_chat(self, messages, add_generation_prompt=True):
            return "<u>" + messages[-1]["content"] + "</u><a>"

        def decode(self, ids):
            return "".join(chr(int(i)) for i in ids)

    n_layers, n_ff = 3, 4
    coef = np.zeros(n_layers * n_ff, dtype=np.float32)
    coef[5] = 1.0                                     # layer 1, neuron 1
    np.savez(tmp_path / "clf.npz", coef=coef, intercept=-2.0, n_layers=n_layers, n_neurons=n_ff)
    sc = hscore.HScorer(str(ROOT / "tests" / "fake_cett_dump.py"), "model.gguf", str(tmp_path / "clf.npz"),
                        tokenizer=Tok())
    sc._norms = np.full((n_layers, n_ff), 2.0, dtype=np.float32)
    prompt = "<u>hi</u><a>"
    res = sc.score([{"role": "user", "content": "hi"}], "abcdef")
    t = np.arange(len(prompt), len(prompt) + 6)
    want = 2.0 * (1 + (t + 1 + 1) % 3).mean() - 2.0
    assert res["n_tokens"] == 6 and res["score"] == pytest.approx(want, rel=1e-3)
    assert 0 < res["prob"] < 1
