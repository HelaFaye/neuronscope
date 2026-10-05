#!/usr/bin/env python3
"""
External benchmarks for local models: LiveBench, SWE-bench, and published
leaderboard scores (e.g. an export from BenchLM.ai). Results can be recorded
as rolling model stats, so Studio's `auto` routing and the retraining
pipeline see them next to TestQA.

LiveBench (https://github.com/LiveBench/LiveBench). Contamination-limited,
objectively graded questions in six categories, refreshed over time.

    git clone https://github.com/LiveBench/LiveBench && pip install -e LiveBench
    python scripts/benchmarks.py livebench run --livebench LiveBench \\
        --endpoint http://127.0.0.1:7870/v1@my-model --bench live_bench/math live_bench/coding
    python scripts/benchmarks.py livebench import --livebench LiveBench --model my-model \\
        --publish-stats http://127.0.0.1:7870

SWE-bench (https://github.com/SWE-bench/SWE-bench). Resolve real GitHub issues.
Predictions are generated here (single-shot retrieval baseline: the model
reads the issue plus retrieved files and writes a patch); scoring runs in
SWE-bench's own Docker harness.

    python scripts/benchmarks.py swebench predict --endpoint http://127.0.0.1:7870/v1@my-model \\
        --dataset princeton-nlp/SWE-bench_Lite_bm25_13K --limit 50 --out runs/swe/preds.jsonl
    python scripts/benchmarks.py swebench evaluate --predictions runs/swe/preds.jsonl \\
        --dataset princeton-nlp/SWE-bench_Lite --run-id my-model-1
    python scripts/benchmarks.py swebench import --report my-model.my-model-1.json --model my-model \\
        --publish-stats http://127.0.0.1:7870

Published scores (BenchLM.ai and similar leaderboards): CSV or JSON rows of
model, benchmark, score, recorded as *reference* stats. They describe the
vendor's model, not your local quant or edit, so they are shown but never
used for auto routing.

    python scripts/benchmarks.py leaderboard import --file benchlm.csv --map my-model=Qwen3-8B
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testqa as tq  # noqa: E402

LIVEBENCH_SUBJECT = {"coding": "code", "agentic_coding": "code", "agentic_coding_v2": "code", "math": "math",
                     "data_analysis": "math", "reasoning": "logic", "language": "writing",
                     "instruction_following": "writing"}
# Common leaderboard benchmark names -> NeuronScope subjects (case-insensitive substring match).
BENCHMARK_SUBJECT = [
    ("swe-bench", "code"), ("swebench", "code"), ("humaneval", "code"), ("mbpp", "code"), ("livecodebench", "code"),
    ("terminal-bench", "code"), ("aider", "code"), ("codeforces", "code"),
    ("aime", "math"), ("math", "math"), ("gsm8k", "math"), ("hmmt", "math"),
    ("gpqa", "science"), ("scicode", "science"), ("chemistry", "science"), ("physics", "science"),
    ("mmlu", "factual"), ("simpleqa", "factual"), ("triviaqa", "factual"), ("humanity", "factual"),
    ("arc", "logic"), ("bbh", "logic"), ("big-bench", "logic"), ("zebra", "logic"), ("reasoning", "logic"),
    ("ifeval", "writing"), ("arena", "writing"), ("writing", "writing"), ("instruction", "writing"),
    ("mmmu", "vision"), ("mathvista", "vision"), ("chartqa", "vision"), ("docvqa", "vision"), ("vision", "vision"),
]


def subject_for_benchmark(name: str) -> str | None:
    low = name.lower()
    for key, subj in BENCHMARK_SUBJECT:
        if key in low:
            return subj
    return None


def record(model_id: str, rows: list[dict], a, source: str) -> int:
    """Rows shaped like TestQA results (id, kind, subject, verdict)."""
    if a.record_stats is None and not a.publish_stats:
        return 0
    args = SimpleNamespace(record_stats=a.record_stats, publish_stats=a.publish_stats, api_key=a.api_key)
    for r in rows:
        r.setdefault("source", source)
    return tq.record_stats(model_id, rows, args, source=source)


def add_stats_args(p):
    p.add_argument("--record-stats", nargs="?", const="", metavar="DIR", help="append to local rolling stats")
    p.add_argument("--publish-stats", metavar="STUDIO_URL", help="send to a running Studio")
    p.add_argument("--api-key", default="")


# ---------------------------------------------------------------- LiveBench

def livebench_run(a) -> int:
    root = Path(a.livebench).resolve()
    script = root / "livebench" / "run_livebench.py"
    if not script.exists():
        raise SystemExit(f"{script} not found; git clone https://github.com/LiveBench/LiveBench and pip install -e it")
    _, url, model = tq.parse_endpoint("m=" + a.endpoint)
    model = model or a.model
    if not model:
        raise SystemExit("give the endpoint as URL@model")
    cmd = [sys.executable, str(script), "--model", model, "--api-base", url, "--bench-name", *a.bench,
           "--model-display-name", a.display_name or model, "--parallel-requests", str(a.parallel)]
    if a.api_key:
        cmd += ["--api-key", a.api_key]
    if a.max_tokens:
        cmd += ["--max-tokens", str(a.max_tokens)]
    if a.release:
        cmd += ["--livebench-release-option", a.release]
    print("running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=root / "livebench")
    if rc == 0 and (a.record_stats is not None or a.publish_stats):
        a.model = a.display_name or model
        return livebench_import(a)
    return rc


def livebench_judgments(root: Path, model: str) -> list[dict]:
    out = []
    for f in sorted((root / "livebench" / "data").rglob("model_judgment/ground_truth_judgment.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("model", "").lower() == model.lower() or r.get("model", "").lower().endswith("/" + model.lower()):
                out.append(r)
    return out


def livebench_import(a) -> int:
    root = Path(a.livebench).resolve()
    js = livebench_judgments(root, a.model)
    if not js:
        raise SystemExit(f"no LiveBench judgments for {a.model!r} under {root}/livebench/data")
    by_cat = defaultdict(list)
    rows = []
    for r in js:
        cat = r.get("category") or r.get("task", "unknown")
        score = float(r.get("score", 0) or 0)
        by_cat[cat].append(score)
        subj = LIVEBENCH_SUBJECT.get(cat)
        if subj is None:
            continue
        # LiveBench scores are 0..1, some with partial credit; it has no abstention
        # notion. Full marks count as correct, anything less as wrong.
        rows.append({"id": f"livebench:{r['question_id']}", "kind": "livebench", "subject": subj,
                     "verdict": "correct" if score >= a.full_marks else "wrong"})
    print(f"LiveBench, {a.model}: {len(js)} judged questions")
    for cat, s in sorted(by_cat.items()):
        print(f"  {cat:<24} n {len(s):>4}   mean {sum(s) / len(s):.3f}   -> {LIVEBENCH_SUBJECT.get(cat, '(unmapped)')}")
    n = record(a.stats_model or a.model, rows, a, "livebench")
    if n:
        print(f"recorded {n} results as stats for {a.stats_model or a.model}")
    return 0


# ---------------------------------------------------------------- SWE-bench

PATCH_RE = [re.compile(r"<patch>\s*(.*?)\s*</patch>", re.S), re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.S)]
SWE_INSTRUCTIONS = ("You will be given a GitHub issue and parts of the repository. Write a patch that resolves "
                    "the issue. Reply with a single unified diff (as produced by `git diff`) inside <patch></patch> "
                    "tags and nothing else.\n\n")


def extract_patch(text: str) -> str:
    body = tq.strip_think(text)
    for rx in PATCH_RE:
        m = rx.search(body)
        if m:
            body = m.group(1)
            break
    body = body.strip("\n")
    if not re.search(r"^(diff --git |--- )", body, re.M):
        return ""
    return body + "\n"


def load_rows(spec: str, split: str) -> list[dict]:
    p = Path(spec)
    if p.exists():
        if p.suffix == ".jsonl":
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if p.suffix == ".json":
            return json.loads(p.read_text())
        if p.suffix == ".parquet":
            import pandas as pd
            return pd.read_parquet(p).to_dict("records")
    from datasets import load_dataset
    return list(load_dataset(spec, split=split))


def swebench_predict(a) -> int:
    label, url, model = tq.parse_endpoint("m=" + a.endpoint)
    rows = load_rows(a.dataset, a.split)
    if a.instance_ids:
        rows = [r for r in rows if r["instance_id"] in set(a.instance_ids)]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["instance_id"] for l in out.read_text().splitlines() if l.strip()}
    ask_args = SimpleNamespace(temperature=0.0, max_tokens=a.max_tokens, api_key=a.api_key,
                               request_timeout=a.request_timeout)
    name = a.model_name or model or label
    n_patch = 0
    for r in rows:
        if r["instance_id"] in done:
            continue
        # *_bm25_* and *_oracle variants carry a ready prompt with retrieved code in "text";
        # the plain datasets only have the issue, which rarely suffices.
        prompt = r.get("text") or (SWE_INSTRUCTIONS + "Repository: " + r["repo"] + "\n\nIssue:\n" + r["problem_statement"])
        try:
            reply = tq.ask(url, model, prompt, ask_args)
        except Exception as e:
            print(f"{r['instance_id']}: request failed: {e}", file=sys.stderr)
            continue
        patch = extract_patch(reply)
        n_patch += bool(patch)
        with out.open("a") as f:
            f.write(json.dumps({"instance_id": r["instance_id"], "model_name_or_path": name,
                                "model_patch": patch}) + "\n")
        print(f"{r['instance_id']}: {'patch' if patch else 'no patch'}")
    print(f"wrote {out}; {n_patch} patches this run")
    return 0


def swebench_evaluate(a) -> int:
    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation", "--dataset_name", a.dataset,
           "--predictions_path", a.predictions, "--max_workers", str(a.max_workers), "--run_id", a.run_id]
    if a.split:
        cmd += ["--split", a.split]
    print("running:", " ".join(cmd), "\n(needs Docker; see the SWE-bench README for disk and memory requirements)")
    return subprocess.call(cmd)


def swebench_import(a) -> int:
    rep = json.loads(Path(a.report).read_text())
    resolved = set(rep.get("resolved_ids", []))
    empty = set(rep.get("empty_patch_ids", []))
    failed = set(rep.get("unresolved_ids", [])) | set(rep.get("error_ids", []))
    rows = ([{"id": f"swebench:{i}", "kind": "swebench", "subject": "code", "verdict": "correct"} for i in resolved]
            + [{"id": f"swebench:{i}", "kind": "swebench", "subject": "code", "verdict": "abstained"} for i in empty]
            + [{"id": f"swebench:{i}", "kind": "swebench", "subject": "code", "verdict": "wrong"}
               for i in failed - resolved - empty])
    total = len(rows)
    print(f"SWE-bench, {a.model}: {len(resolved)}/{total} resolved ({len(resolved) / max(1, total):.1%}), "
          f"{len(empty)} without a patch")
    n = record(a.stats_model or a.model, rows, a, "swebench")
    if n:
        print(f"recorded {n} results as stats for {a.stats_model or a.model}")
    return 0


# ---------------------------------------------------------------- leaderboards

def leaderboard_import(a) -> int:
    from model_stats import StatsStore
    p = Path(a.file)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        rows = data if isinstance(data, list) else data.get("rows", [])
    else:
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
    alias = dict(m.split("=", 1) for m in a.map)          # local id -> leaderboard model name
    wanted = {v.lower(): k for k, v in alias.items()}
    store = StatsStore(a.record_stats or None)
    n = 0
    for r in rows:
        name = str(r.get(a.model_col, "")).strip()
        local = wanted.get(name.lower())
        if local is None:
            continue
        bench = str(r.get(a.benchmark_col, "")).strip()
        try:
            score = float(str(r.get(a.score_col, "")).rstrip("%"))
        except ValueError:
            continue
        if score > 1.0:
            score /= 100.0
        store.record(local, "reference", benchmark=bench, subject=subject_for_benchmark(bench), score=score,
                     source=a.source, leaderboard_model=name)
        n += 1
    print(f"recorded {n} published scores as reference stats ({a.source})")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    top = p.add_subparsers(dest="bench", required=True)

    lb = top.add_parser("livebench").add_subparsers(dest="cmd", required=True)
    r = lb.add_parser("run")
    r.add_argument("--livebench", required=True, help="LiveBench checkout")
    r.add_argument("--endpoint", required=True, help="URL@model (OpenAI-compatible)")
    r.add_argument("--bench", nargs="+", default=["live_bench"], help="e.g. live_bench/math live_bench/coding")
    r.add_argument("--display-name")
    r.add_argument("--model")
    r.add_argument("--release", help="LiveBench release, e.g. 2025-11-25")
    r.add_argument("--parallel", type=int, default=2)
    r.add_argument("--max-tokens", type=int, default=0)
    r.add_argument("--stats-model")
    r.add_argument("--full-marks", type=float, default=1.0)
    add_stats_args(r)
    i = lb.add_parser("import")
    i.add_argument("--livebench", required=True)
    i.add_argument("--model", required=True, help="model name as LiveBench recorded it")
    i.add_argument("--stats-model", help="record under this id instead (e.g. Studio's model id)")
    i.add_argument("--full-marks", type=float, default=1.0, help="score that counts as correct")
    add_stats_args(i)

    sw = top.add_parser("swebench").add_subparsers(dest="cmd", required=True)
    pr = sw.add_parser("predict")
    pr.add_argument("--endpoint", required=True)
    pr.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite_bm25_13K",
                    help="HF dataset or local .jsonl/.json/.parquet; *_bm25_* / *_oracle variants include code context")
    pr.add_argument("--split", default="test")
    pr.add_argument("--instance-ids", nargs="*")
    pr.add_argument("--limit", type=int, default=0)
    pr.add_argument("--model-name")
    pr.add_argument("--max-tokens", type=int, default=4096)
    pr.add_argument("--request-timeout", type=float, default=1800)
    pr.add_argument("--api-key", default="")
    pr.add_argument("--out", required=True)
    ev = sw.add_parser("evaluate")
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    ev.add_argument("--split", default="test")
    ev.add_argument("--run-id", required=True)
    ev.add_argument("--max-workers", type=int, default=2)
    im = sw.add_parser("import")
    im.add_argument("--report", required=True, help="<model>.<run_id>.json written by the harness")
    im.add_argument("--model", required=True)
    im.add_argument("--stats-model")
    add_stats_args(im)

    ld = top.add_parser("leaderboard").add_subparsers(dest="cmd", required=True)
    li = ld.add_parser("import", help="published scores (CSV/JSON) as reference stats")
    li.add_argument("--file", required=True)
    li.add_argument("--map", action="append", default=[], required=True, metavar="LOCAL_ID=LEADERBOARD_NAME")
    li.add_argument("--model-col", default="model")
    li.add_argument("--benchmark-col", default="benchmark")
    li.add_argument("--score-col", default="score")
    li.add_argument("--source", default="benchlm")
    li.add_argument("--record-stats", nargs="?", const="", metavar="DIR")
    a = p.parse_args(argv)

    if a.bench == "livebench":
        return livebench_run(a) if a.cmd == "run" else livebench_import(a)
    if a.bench == "swebench":
        return {"predict": swebench_predict, "evaluate": swebench_evaluate, "import": swebench_import}[a.cmd](a)
    return leaderboard_import(a)


if __name__ == "__main__":
    raise SystemExit(main())
