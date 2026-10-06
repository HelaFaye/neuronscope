#!/usr/bin/env python3
"""
Compare two judges on the same responses before trusting either one.

Label quality is the weakest link in this pipeline. Every mislabelled pair is
noise the L1 classifier fits, and it puts the wrong neurons in your profile --
which no amount of scale tuning can undo. So before swapping the rule judge for
a stronger model judge, measure how far apart they actually are, and read the
disagreements yourself.

Reports Cohen's kappa (agreement corrected for chance, since these labels are
badly imbalanced and raw agreement flatters), the confusion matrix, and a sample
of disagreements for human review.

What to do with the result:

    kappa > 0.8   judges substantially agree; the rule judge is fine, keep it,
                  it is free and deterministic
    0.6 - 0.8     read the disagreements. Usually the rule judge missing a
                  correct answer phrased unusually -- fixable with better
                  aliases, no model judge needed
    < 0.6         your labels are noise. Fix this before extracting anything

Adjudicate by reading, not by assuming the model judge is right. A model judge
has its own failure modes and is not ground truth; it is a second opinion whose
disagreements point at where your labels are ambiguous.

    python scripts/judge_agreement.py \\
        --input_path data/consistency_samples.jsonl \\
        --base_url http://GPU-HOST:8080/v1 --model <judge-model> \\
        --n 150 --show 15
"""

import argparse
import json
import random
import re
import string
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

THINK_CLOSE = "</think>"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--base_url", required=True, help="OpenAI-compatible root + /v1")
    p.add_argument("--api_key", default="none")
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--show", type=int, default=15,
                   help="How many disagreements to print for review")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="judge_agreement.json")
    return p.parse_args()


def normalize_answer(s):
    if not s:
        return ""
    exclude = set(string.punctuation + "\u2018\u2019\u00b4`")
    s = str(s).lower().replace("_", " ")
    s = "".join(ch if ch not in exclude else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()).strip()


def strip_think(t):
    return t.split(THINK_CLOSE, 1)[1].strip() if THINK_CLOSE in t else t.strip()


def rule_judge(answer, aliases):
    na = normalize_answer(answer)
    if not na:
        return "uncertain"
    return "true" if any(normalize_answer(a) == na or normalize_answer(a) in na
                         for a in aliases if a) else "false"


def model_judge(args, question, answer, aliases):
    prompt = (
        f"Question: {question}\n"
        f"Reference answers (any one is correct): {aliases}\n"
        f"Model response: {answer}\n\n"
        "Does the response give one of the reference answers? Ignore phrasing, "
        "verbosity and extra commentary; judge only whether the factual content "
        "matches. Reply with exactly one word: true or false."
    )
    for _ in range(3):
        try:
            r = requests.post(
                f"{args.base_url}/chat/completions", timeout=300,
                headers={"Authorization": f"Bearer {args.api_key}"},
                json={"model": args.model, "temperature": 0.0, "max_tokens": 512,
                      "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            v = strip_think(
                r.json()["choices"][0]["message"].get("content") or "").lower()
            if "true" in v and "false" not in v:
                return "true"
            if "false" in v:
                return "false"
        except Exception:
            continue
    return "uncertain"


def kappa(a, b):
    """Cohen's kappa. Raw agreement flatters badly imbalanced labels; this
    corrects for the agreement you would get by chance alone."""
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main():
    args = parse_args()
    random.seed(args.seed)

    rows = []
    with open(args.input_path, encoding="utf-8") as f:
        for line in f:
            d = next(iter(json.loads(line).values()))
            if d.get("aliases") and d.get("answer") is not None:
                rows.append(d)
    if not rows:
        raise SystemExit(
            "no entries with gold aliases and a stored answer; re-run the "
            "collector (versions before the alias fix did not persist them)")
    if len(rows) > args.n:
        rows = random.sample(rows, args.n)
    print(f"comparing judges on {len(rows)} responses\n")

    rule = [rule_judge(r["answer"], r["aliases"]) for r in rows]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        model = list(tqdm(
            pool.map(lambda r: model_judge(args, r["question"], r["answer"],
                                           r["aliases"]), rows),
            total=len(rows), desc="model judge"))

    labels = ["true", "false", "uncertain"]
    print("\n           model judge")
    print("rule       " + "".join(f"{l:>11}" for l in labels))
    for rl in labels:
        cells = [sum(1 for x, y in zip(rule, model) if x == rl and y == ml)
                 for ml in labels]
        print(f"{rl:<11}" + "".join(f"{c:>11}" for c in cells))

    agree = sum(1 for x, y in zip(rule, model) if x == y)
    k = kappa(rule, model)
    print(f"\nraw agreement : {agree / len(rows):.1%}")
    print(f"Cohen's kappa : {k:.3f}")
    if k > 0.8:
        print("  Judges substantially agree. Keep the rule judge: free, "
              "deterministic, reproducible.")
    elif k > 0.6:
        print("  Moderate. Read the disagreements below -- usually the rule "
              "judge missing a correct\n  answer phrased unusually, which "
              "better aliases fix without a model judge.")
    else:
        print("  Poor. Your labels are noise and the neuron set will inherit "
              "it. Fix this\n  before extracting anything.")

    disagreements = [(r, x, y) for r, x, y in zip(rows, rule, model) if x != y]
    print(f"\n{len(disagreements)} disagreements "
          f"({len(disagreements) / len(rows):.1%}). "
          f"Showing {min(args.show, len(disagreements))} for review:\n")
    for r, x, y in disagreements[:args.show]:
        print(f"  rule={x:<9} model={y}")
        print(f"    Q: {r['question'][:110]}")
        print(f"    A: {r['answer'][:110]}")
        print(f"    gold: {r['aliases'][:4]}\n")

    print("Adjudicate by reading these, not by assuming the model judge is "
          "right. It has\nits own failure modes; disagreement marks where your "
          "labels are ambiguous.")

    with open(args.out, "w") as f:
        json.dump({"n": len(rows), "kappa": k,
                   "raw_agreement": agree / len(rows),
                   "disagreements": [
                       {"question": r["question"], "answer": r["answer"],
                        "aliases": r["aliases"], "rule": x, "model": y}
                       for r, x, y in disagreements]}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
