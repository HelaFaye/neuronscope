"""Deficit -> dataset -> LoRA SFT + DPO -> merge -> GGUF, on a tiny real model."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import deficits  # noqa: E402
import testqa as tq  # noqa: E402

LLAMA = Path(os.environ.get("NS_LLAMA", "")).expanduser()


class Teacher(BaseHTTPRequestHandler):
    """Answers bank prompts correctly (gold answers), and writes variations."""
    answers: dict = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        if "Write 2 NEW items" in prompt:
            reply = json.dumps([{"prompt": "What is 6 times 7?", "answer": "42", "answer_type": "number"},
                                {"prompt": "What is 9 plus 10?", "answer": "19", "answer_type": "number"}])
        elif prompt.startswith("What is 6 times 7?"):
            reply = "6*7 = 42\nAnswer: 42"
        elif prompt.startswith("What is 9 plus 10?"):
            reply = "Answer: 19"
        else:
            reply = self.answers.get(prompt, "no idea")
        out = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A fake TestQA result: the model fails every 3rd item, answers the rest."""
    d = tmp_path_factory.mktemp("run")
    tasks = [t for t in tq.load_tasks([str(ROOT / "qa" / "bank")], None, None, 0)
             if t["kind"] in ("reasoning", "qa", "constraints") and not t.get("image")]
    rows, cache, answers = [], [], {}
    for i, t in enumerate(tasks):
        if t["kind"] == "reasoning":
            good = f"Answer: {t['answer']}"
        elif t["kind"] == "qa":
            good = deficits.ABSTAIN_TARGET if t.get("expect_abstain") else t["aliases"][0]
        else:
            good = None
        answers[tq.prompt_for(t)] = good or "x"
        fail = i % 3 == 0 or good is None
        text = "Answer: 999999" if fail else good
        verdict = tq.grade(t, text, type("A", (), {"allow_exec": False, "exec_timeout": 5})())[0]
        rows.append({"id": t["id"], "kind": t["kind"], "subject": t["subject"], "verdict": verdict, "text": text})
        cache.append({"id": t["id"], "text": text})
    (d / "res.json").write_text(json.dumps({"reference": "m", "raw": {"m": rows}}))
    (d / "cache").mkdir()
    (d / "cache" / "m.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cache))
    Teacher.answers = answers
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Teacher)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield d, f"http://127.0.0.1:{srv.server_port}/v1@teacher", tasks
    srv.shutdown()


def test_deficit_dataset(run):
    d, teacher, tasks = run
    plan = deficits.build(deficits.main.__globals__["argparse"].Namespace(
        results=str(d / "res.json"), label="m", cache=str(d / "cache"), tasks=[str(ROOT / "qa" / "bank")],
        teacher=teacher, teacher_tries=1, teacher_temperature=0.0, teacher_max_tokens=256, api_key="",
        expand=2, allow_exec=False, deficit_fraction=0.25, general=None, holdout_frac=0.5, seed=0,
        out=str(d / "retrain")))
    out = d / "retrain"
    sft = [json.loads(l) for l in (out / "sft.jsonl").read_text().splitlines()]
    dpo = [json.loads(l) for l in (out / "dpo.jsonl").read_text().splitlines()]
    hold = set(json.loads((out / "holdout_ids.json").read_text())["ids"])
    assert {"factual_gap", "reasoning_failure", "format_violation"} & set(plan["deficits_by_category"])
    # every target passes the grader that failed the model
    by_id = {t["id"]: t for t in tasks}
    for r in sft:
        t = by_id.get(r["task"])
        if t is not None:
            assert tq.grade(t, r["messages"][-1]["content"], type("A", (), {"allow_exec": False, "exec_timeout": 5})())[0] == "correct"
    # nothing from the holdout is trained on, directly or through a variation
    assert not hold & {r["task"].split("~")[0] for r in sft}
    assert any(r["source"] == "synthetic" for r in sft)
    # replay buffer keeps deficits near the requested share
    assert 0.2 <= plan["deficit_fraction"] <= 0.3
    assert dpo and all(p["chosen"][0]["content"] != p["rejected"][0]["content"] for p in dpo)
    # constraints items without a teacher answer are reported, not invented
    assert plan["skipped"]


def test_lora_sft_dpo_merge_gguf(run, tmp_path):
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    pytest.importorskip("sentencepiece")
    import tiny_models
    d, _, _ = run
    if not (d / "retrain" / "sft.jsonl").exists():
        pytest.skip("dataset test did not run")
    hf = tiny_models.make_tiny_llama(tmp_path / "tiny")
    adapter = tmp_path / "adapter"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "finetune.py"), "--model", str(hf),
                        "--data", str(d / "retrain"), "--out", str(adapter), "--method", "lora", "--lora-r", "4",
                        "--max-steps", "3", "--batch-size", "2", "--grad-accum", "1", "--max-length", "128",
                        "--dpo", "--dpo-max-steps", "2", "--logging-steps", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert (adapter / "adapter_model.safetensors").exists()
    assert json.loads((adapter / "neuronscope-finetune.json").read_text())["dpo"] is True
    if not (LLAMA / "convert_hf_to_gguf.py").exists():
        pytest.skip("set NS_LLAMA for the GGUF half")
    out = tmp_path / "merged"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "merge_export.py"), "--base", str(hf),
                        "--adapter", str(adapter), "--out", str(out), "--gguf", "Q8_0", "--dtype", "float32",
                        "--llama", str(LLAMA)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    meta = json.loads((out / "neuronscope-export.json").read_text())
    assert Path(meta["gguf"]).exists() and meta["gguf"].endswith("Q8_0.gguf")
    # the merged GGUF loads and tokenizes in llama.cpp
    probe = tmp_path / "m.jsonl"
    probe.write_text(json.dumps({"id": "p", "text": "the cat sat"}) + "\n")
    r = subprocess.run([str(LLAMA / "build" / "bin" / "llama-cett-dump"), "-m", meta["gguf"], "--tokenize-only",
                        "-ngl", "0", "-c", "256", "--manifest", str(probe), "--outdir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and (tmp_path / "p.toks").exists(), r.stderr[-1500:]
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "merge_export.py"), "--base", str(hf),
                        "--adapter", str(adapter), "--out", str(tmp_path / "lora"), "--lora-gguf",
                        "--llama", str(LLAMA)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
