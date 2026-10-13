"""The LM Studio evaluator for the adaptive tuner: finding the candidate in
LM Studio's list, scoring, and a run against a stand-in server."""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_lmstudio as ev  # noqa: E402


def test_match_id():
    ids = ["huihui/ornith-9b-supp3500", "huihui/ornith-9b-supp350", "other/model@q6_k", "ornith-9b-base100"]
    assert ev.match_id(ids, "ornith-9b-supp350") == "huihui/ornith-9b-supp350"
    assert ev.match_id(ids, "Ornith-9B-base100") == "ornith-9b-base100"
    assert ev.match_id(["pub/My_Model.Q4"], "my-model-q4") == "pub/My_Model.Q4"
    assert ev.match_id(ids, "nothing-like-it") is None
    assert ev.stem_of("/x/y/cand-supp350-00001-of-00002.gguf") == "cand-supp350"


def test_score_metrics():
    rows = [{"verdict": v} for v in ["correct"] * 6 + ["wrong"] * 2 + ["abstained"] * 2 + ["error"]]
    acc = ev.score(rows, "accuracy")
    assert acc["score"] == 0.6 and acc["correct"] == 6 and acc["total"] == 10 and acc["request_errors"] == 1
    nh = ev.score(rows, "not_hallucinated")
    assert nh["score"] == 0.8 and nh["correct"] == 8
    ut = ev.score(rows, "utility", cost=2.0)
    assert ut["score"] == pytest.approx((0.6 - 0.4 + 2) / 3, abs=1e-6) and "total" not in ut
    with pytest.raises(ValueError):
        ev.score([{"verdict": "error"}], "accuracy")


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_wait_times_out_with_a_reason():
    with pytest.raises(TimeoutError, match="cannot reach LM Studio"):
        ev.wait_for(f"http://127.0.0.1:{_port()}", "x", None, wait=0.2, poll=0.1, log=lambda *a, **k: None)


def test_end_to_end_against_a_stand_in(tmp_path):
    # The stand-in answers every prompt with "model=<alias> mmproj=no image=no";
    # a one-item bank whose answer is "image=no" makes that a right answer.
    bank = tmp_path / "bank.jsonl"
    bank.write_text(json.dumps({"id": "t1", "kind": "qa", "subject": "factual", "prompt": "Image?",
                                "aliases": ["image=no"]}) + "\n" +
                    json.dumps({"id": "t2", "kind": "qa", "subject": "factual", "prompt": "Capital of France?",
                                "aliases": ["Paris"]}) + "\n")
    port = _port()
    srv = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_llama_server.py"), "--port", str(port),
                            "--alias", "pub/cand-supp350"])
    try:
        for _ in range(50):
            try:
                ev.list_models(f"http://127.0.0.1:{port}", None, timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_lmstudio.py"), "--lmstudio",
                            f"http://127.0.0.1:{port}", "--model", str(tmp_path / "cand-supp350.gguf"),
                            "--tasks", str(bank), "--per-subject", "0", "--keep", str(tmp_path / "keep"),
                            "--wait", "5"], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        res = json.loads(r.stdout.strip().splitlines()[-1])
        assert res["lmstudio_id"] == "pub/cand-supp350" and res["total"] == 2 and res["score"] == 0.5
        # The adaptive tuner's parser takes it as is.
        import adaptive_tuner
        parsed = adaptive_tuner.parse_eval(res)
        assert parsed["score"] == 0.5 and 0 <= parsed["lower_ci"] < parsed["upper_ci"] <= 1
    finally:
        srv.terminate()
