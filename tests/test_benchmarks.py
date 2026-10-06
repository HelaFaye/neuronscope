import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmarks as bm  # noqa: E402
import model_stats  # noqa: E402

PATCH = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"


def test_livebench_import_records_per_category(tmp_path):
    d = tmp_path / "LiveBench" / "livebench" / "data" / "live_bench" / "math" / "AMPS_Hard" / "model_judgment"
    d.mkdir(parents=True)
    rows = [{"question_id": f"q{i}", "task": "AMPS_Hard", "model": "my-model", "score": 1.0 if i < 3 else 0.5,
             "category": "math"} for i in range(5)]
    rows.append({"question_id": "z", "task": "AMPS_Hard", "model": "other", "score": 1, "category": "math"})
    (d / "ground_truth_judgment.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert bm.main(["livebench", "import", "--livebench", str(tmp_path / "LiveBench"), "--model", "my-model",
                    "--record-stats", str(tmp_path / "stats")]) == 0
    s = model_stats.StatsStore(tmp_path / "stats").summary("my-model")
    assert s["subjects"]["math"]["n"] == 5 and s["subjects"]["math"]["correct"] == 3
    assert s["graded"]["sources"] == {"livebench": 5}


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        reply = f"Here you go:\n<patch>\n{PATCH}</patch>" if "issue A" in prompt else "I am not sure."
        out = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_swebench_predict_and_import(tmp_path):
    ds = tmp_path / "lite.jsonl"
    ds.write_text("".join(json.dumps(r) + "\n" for r in [
        {"instance_id": "org__a-1", "repo": "org/a", "problem_statement": "issue A", "text": "issue A with code"},
        {"instance_id": "org__a-2", "repo": "org/a", "problem_statement": "issue B"}]))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = tmp_path / "preds.jsonl"
    try:
        assert bm.main(["swebench", "predict", "--endpoint", f"http://127.0.0.1:{srv.server_port}/v1@m",
                        "--dataset", str(ds), "--out", str(out)]) == 0
        # resumable: a second run adds nothing
        assert bm.main(["swebench", "predict", "--endpoint", f"http://127.0.0.1:{srv.server_port}/v1@m",
                        "--dataset", str(ds), "--out", str(out)]) == 0
    finally:
        srv.shutdown()
    preds = [json.loads(l) for l in out.read_text().splitlines()]
    assert [p["instance_id"] for p in preds] == ["org__a-1", "org__a-2"]
    assert preds[0]["model_patch"] == PATCH and preds[1]["model_patch"] == ""
    rep = tmp_path / "m.run.json"
    rep.write_text(json.dumps({"resolved_ids": ["org__a-1"], "unresolved_ids": ["org__a-3"],
                               "empty_patch_ids": ["org__a-2"], "error_ids": []}))
    assert bm.main(["swebench", "import", "--report", str(rep), "--model", "m",
                    "--record-stats", str(tmp_path / "stats")]) == 0
    g = model_stats.StatsStore(tmp_path / "stats").summary("m")["graded"]
    assert (g["correct"], g["wrong"], g["abstained"]) == (1, 1, 1)


def test_patch_extraction():
    assert bm.extract_patch(f"```diff\n{PATCH}```") == PATCH
    assert bm.extract_patch("no diff here") == ""
    assert bm.extract_patch(f"<think>hmm</think><patch>{PATCH}</patch>") == PATCH


def test_leaderboard_import_is_reference_only(tmp_path):
    f = tmp_path / "board.csv"
    f.write_text("model,benchmark,score\nQwen3-8B,GPQA Diamond,62.0\nQwen3-8B,SWE-bench Verified,0.41\nOther,MMLU,90\n")
    assert bm.main(["leaderboard", "import", "--file", str(f), "--map", "qwen3-8b-gguf/qwen3-8b-q6_k=Qwen3-8B",
                    "--record-stats", str(tmp_path / "stats")]) == 0
    s = model_stats.StatsStore(tmp_path / "stats").summary("qwen3-8b-gguf/qwen3-8b-q6_k")
    assert s["reference"]["GPQA Diamond"] == {"score": 0.62, "subject": "science", "source": "benchlm"}
    assert s["reference"]["SWE-bench Verified"]["subject"] == "code"
    assert s["graded"]["n"] == 0          # never counted toward auto routing


def _diff(path, old, new):
    return (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1,1 +1,1 @@\n-{old}\n+{new}\n")


def test_swebench_train_data(tmp_path):
    import benchmarks
    rows = [{"instance_id": f"org__r-{i}", "repo": "org/r", "problem_statement": f"bug {i}",
             "text": f"issue {i} with code", "patch": _diff(f"m{i}.py", "x = 1", "x = 2")} for i in range(8)]
    rows.append({"instance_id": "org__r-bad", "repo": "org/r", "problem_statement": "p", "patch": "not a diff"})
    rows.append({"instance_id": "lite__t-1", "repo": "x/y", "problem_statement": "p", "patch": _diff("a.py", "1", "2")})
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "lite.jsonl").write_text(json.dumps({"instance_id": "lite__t-1"}) + "\n")
    preds = {"org__r-0": rows[0]["patch"],                       # solved: an anchor
             "org__r-1": "",                                      # no patch
             "org__r-2": "--- a/m2.py\n+++ b/m2.py\nnonsense",    # malformed
             "org__r-3": _diff("other.py", "a", "b"),             # wrong file
             "org__r-4": _diff("m4.py", "x = 1", "x = 3")}        # wrong fix
    (tmp_path / "preds.jsonl").write_text("".join(json.dumps({"instance_id": k, "model_patch": v}) + "\n"
                                                  for k, v in preds.items()))
    (tmp_path / "report.json").write_text(json.dumps({"resolved_ids": [], "empty_patch_ids": ["a", "b", "c"],
                                                      "unresolved_ids": ["d"], "error_ids": []}))
    (tmp_path / "anchors.jsonl").write_text("".join(json.dumps({"messages": [
        {"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}) + "\n" for i in range(40)))
    out = tmp_path / "out"
    assert benchmarks.main(["swebench", "train-data", "--dataset", str(tmp_path / "train.jsonl"),
                            "--predictions", str(tmp_path / "preds.jsonl"), "--report", str(tmp_path / "report.json"),
                            "--exclude", str(tmp_path / "lite.jsonl"), "--anchors", str(tmp_path / "anchors.jsonl"),
                            "--limit", "4", "--out", str(out)]) == 0
    plan = json.loads((out / "plan.json").read_text())
    assert plan["train_failure_modes"] == {"no_patch": 1, "malformed_patch": 1, "wrong_files": 1, "wrong_fix": 1}
    assert plan["skipped"] == {"in an excluded (test) set": 1, "gold patch is not a well-formed diff": 1}
    sft = [json.loads(l) for l in (out / "sft.jsonl").read_text().splitlines()]
    dpo = [json.loads(l) for l in (out / "dpo.jsonl").read_text().splitlines()]
    deficit = [r for r in sft if r.get("category", "").startswith("swebench")]
    assert len(deficit) == 4 and plan["deficit_examples"] == 4
    # the test run mostly failed to produce a diff, so that failure mode comes first
    assert any(r["category"] == "swebench_no_patch" for r in deficit)
    assert all("lite__t-1" not in r.get("task", "") for r in sft)
    for r in deficit:
        assert r["messages"][1]["content"].startswith("<patch>\n")
        assert benchmarks.patch_ok(benchmarks.extract_patch(r["messages"][1]["content"]))
    assert dpo and all(p["chosen"][0]["content"] != p["rejected"][0]["content"] for p in dpo)
    assert 0.2 <= 4 / len(sft) <= 0.3
