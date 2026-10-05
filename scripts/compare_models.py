#!/usr/bin/env python3
"""
Does suppression help? Ask the original and the edited model the same fresh
questions and compare, question by question.

    python scripts/compare_models.py \\
        --target base=http://GPU-HOST:1234/v1@my-model \\
        --target s025=http://GPU-HOST:1234/v1@my-model-supp25 \\
        --n 200 --out runs/compare-s025

Targets are name=URL[@model]. Any OpenAI-compatible server works: LM Studio
(give the model identifier after @; it swaps models on demand) or one
llama-server per model (omit @model). Targets run one after another, so two
models never have to fit in VRAM together.

Questions come from the TriviaQA parquet, skipping every qid already in
consistency_samples.jsonl: the classifier never saw these, so the result is
not flattered by the neurons having been chosen on the same questions.

Results are cached per target in --out, so an interrupted run resumes.

What a useful edit looks like, against the first target:
  wrong goes down          the point
  abstained goes up        the price; fine for trivia, maybe not for code
  correct stays put        if it drops, the edit is removing knowledge
"""
import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_scale_server import classify, strip_think  # noqa: E402

SUFFIX = " Respond with the answer only, without any explanation."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", action="append", required=True,
                   help="name=URL[@model]; first one is the baseline")
    p.add_argument("--data_path",
                   default="data/TriviaQA/rc.nocontext/train-00000-of-00001.parquet")
    p.add_argument("--exclude", nargs="*",
                   default=["data/consistency_samples.jsonl"],
                   help="jsonl files whose qids are not reused")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max_tokens", type=int, default=4096,
                   help="the model reasons first; too low reads as abstention")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "lm"))
    p.add_argument("--out", required=True)
    return p.parse_args()


def parse_target(s):
    if "=" not in s:
        raise SystemExit(f"--target {s!r}: expected name=URL[@model]")
    name, rest = s.split("=", 1)
    url, _, model = rest.partition("@")
    return name, url.rstrip("/"), model or None


def pick_questions(args):
    seen = set()
    for path in args.exclude or []:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        seen.add(next(iter(json.loads(line))))
    from collect_responses_lmstudio import load_questions
    rows = [r for r in load_questions(args.data_path, None)
            if r["qid"] not in seen and r["aliases"]]
    random.Random(args.seed).shuffle(rows)
    print(f"{len(rows):,} unused questions available, {len(seen)} excluded")
    return rows[:args.n]


def ask(url, model, key, question, max_tokens):
    body = {"messages": [{"role": "user", "content": question + SUFFIX}],
            "temperature": 0.0, "max_tokens": max_tokens}
    if model:
        body["model"] = model
    r = requests.post(f"{url}/chat/completions", json=body, timeout=1800,
                      headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    ch = r.json()["choices"][0]
    return strip_think(ch["message"].get("content") or ""), \
        ch.get("finish_reason") == "length"


def run_target(name, url, model, qs, args):
    path = os.path.join(args.out, f"{name}.jsonl")
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["qid"]] = r
    todo = [q for q in qs if q["qid"] not in done]
    if todo:
        print(f"\n{name}: {len(todo)} to ask ({len(done)} cached) at {url}"
              f"{' model ' + model if model else ''}")
    with open(path, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(args.concurrency) as pool:
        futs = {pool.submit(ask, url, model, args.api_key, q["question"],
                            args.max_tokens): q for q in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=name,
                        disable=not todo):
            q = futs[fut]
            try:
                ans, trunc = fut.result()
            except Exception as e:      # keep going; report, do not cache
                print(f"  {q['qid']}: {type(e).__name__}: {e}")
                continue
            v = "truncated" if trunc else classify(ans, q["aliases"])
            rec = {"qid": q["qid"], "answer": ans[:300], "verdict": v}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            done[q["qid"]] = rec
    return {q["qid"]: done[q["qid"]]["verdict"] for q in qs if q["qid"] in done}


def ci(k, n):
    """Wilson 95% interval for a proportion."""
    if not n:
        return 0.0, 0.0
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(b, c):
    """Exact two-sided p for b vs c discordant pairs."""
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    targets = [parse_target(t) for t in args.target]
    qs = pick_questions(args)
    with open(os.path.join(args.out, "questions.json"), "w") as f:
        json.dump([q["qid"] for q in qs], f)

    res = {}
    for name, url, model in targets:
        res[name] = run_target(name, url, model, qs, args)

    common = set.intersection(*(set(r) for r in res.values()))
    n = len(common)
    print(f"\n{n} questions answered by every target\n")
    cats = ["correct", "abstained", "wrong", "truncated"]
    print(f"{'target':<12}" + "".join(f"{c:>20}" for c in cats))
    for name in res:
        row = f"{name:<12}"
        for c in cats:
            k = sum(1 for q in common if res[name][q] == c)
            lo, hi = ci(k, n)
            row += f"{k / max(n, 1):>8.1%} [{lo:.0%}-{hi:.0%}]"
        print(row)

    base = targets[0][0]
    for name in list(res)[1:]:
        b, s = res[base], res[name]
        moved = {}
        for q in common:
            if b[q] != s[q]:
                moved[(b[q], s[q])] = moved.get((b[q], s[q]), 0) + 1
        fixed = sum(v for (x, y), v in moved.items() if x == "wrong" and y != "wrong")
        broke = sum(v for (x, y), v in moved.items() if y == "wrong" and x != "wrong")
        lost = sum(v for (x, y), v in moved.items() if x == "correct" and y != "correct")
        gained = sum(v for (x, y), v in moved.items() if y == "correct" and x != "correct")
        print(f"\n{name} vs {base}:")
        for (x, y), v in sorted(moved.items(), key=lambda kv: -kv[1]):
            print(f"  {x:>9} -> {y:<9} {v}")
        print(f"  wrong answers:   {fixed} removed, {broke} introduced  "
              f"(McNemar p = {mcnemar(fixed, broke):.3f})")
        print(f"  correct answers: {gained} gained, {lost} lost  "
              f"(McNemar p = {mcnemar(gained, lost):.3f})")
        if fixed > broke and lost <= gained + max(2, n // 50):
            print("  -> fewer wrong answers without losing correct ones. "
                  "Check p before trusting it.")
        elif lost > gained + max(2, n // 50):
            print("  -> losing correct answers: the edit removes knowledge. "
                  "Try a milder scale or fewer neurons (lower --C).")
        else:
            print("  -> no clear benefit at this sample size.")


if __name__ == "__main__":
    main()
