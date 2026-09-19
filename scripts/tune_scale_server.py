#!/usr/bin/env python3
"""
Stage 7, server path: tune suppression strength over HTTP.

tune_scale.py needs PyTorch and a full-precision model in memory. On a Vega
iGPU neither is available, and generation at ~4 tok/s makes it hopeless anyway.
This does the same sweep against a running llama-server with the LoRA adapter
loaded, so the work happens wherever the model is actually fast.

It sweeps the *adapter* scale a. Because the delta is linear in (s - 1), the
effective neuron scale is

    1 + a * (s - 1)

where s is the scale the adapter was exported at. a=0 is the untouched model,
a=1 is the profile as exported. So a single adapter covers the whole range and
nothing is re-exported between points.

    # serve, on whichever machine has the fast GPU
    llama-server -m ornith-Q6_K.gguf --lora-scaled suppress-lora.gguf 1.0 \\
        -ngl 99 --port 8080

    # tune, from anywhere
    python scripts/tune_scale_server.py \\
        --base_url http://192.168.41.171:8080 \\
        --eval_path data/consistency_samples.jsonl \\
        --profile profiles/<fp>/trivia-q6.json \\
        --alphas 0 0.25 0.5 0.75 1.0 --n_eval 200 --concurrency 4 --save
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

THINK_CLOSE = "</think>"
ABSTAIN_MARKERS = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "not certain", "cannot determine", "can't determine", "no information",
    "unable to answer", "unsure", "i don't have", "i do not have",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", required=True, help="llama-server root, e.g. http://host:8080")
    p.add_argument("--eval_path", required=True)
    p.add_argument("--profile", help="Records the sweep into this profile")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="Adapter scales. 0 = no suppression.")
    p.add_argument("--adapter_id", type=int, default=0)
    p.add_argument("--n_eval", type=int, default=200,
                   help="Below ~200 the confidence interval is wider than the "
                        "effect you are trying to measure.")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def strip_think(t):
    return t.split(THINK_CLOSE, 1)[1].strip() if THINK_CLOSE in t else t.strip()


def norm(s):
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s.lower())
    return " ".join(s.split())


def classify(answer, aliases):
    if not answer.strip():
        return "abstained"
    low = answer.lower()
    if any(m in low for m in ABSTAIN_MARKERS):
        return "abstained"
    na = norm(answer)
    return "correct" if any(norm(a) and norm(a) in na for a in aliases) else "wrong"


def set_alpha(base_url, adapter_id, alpha):
    r = requests.post(f"{base_url}/lora-adapters", timeout=30,
                      json=[{"id": adapter_id, "scale": alpha}])
    if r.status_code != 200:
        raise SystemExit(
            f"could not set adapter scale ({r.status_code}): {r.text[:200]}\n"
            "Is llama-server running with --lora-scaled? LM Studio does not "
            "expose this endpoint; use llama-server for tuning."
        )


def ask(base_url, question, max_tokens):
    r = requests.post(f"{base_url}/v1/chat/completions", timeout=600, json={
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.0,          # greedy: no sampling variance to average out
        "max_tokens": max_tokens,
    })
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    content = msg.get("content") or ""
    if msg.get("reasoning_content"):
        return strip_think(content)
    return strip_think(content)


def main():
    args = parse_args()

    cases, no_gold = [], 0
    with open(args.eval_path, encoding="utf-8") as f:
        for line in f:
            d = next(iter(json.loads(line).values()))
            aliases = d.get("aliases") or []
            if not aliases:
                no_gold += 1
                continue
            cases.append({"question": d["question"], "aliases": aliases})
            if len(cases) >= args.n_eval:
                break
    if not cases:
        raise SystemExit(f"no gold aliases in {args.eval_path} ({no_gold} skipped)")
    if len(cases) < 200:
        print(f"warning: {len(cases)} questions gives a 95% CI of roughly "
              f"+/-{1.96 * (0.25 / len(cases)) ** 0.5 * 100:.0f} points; "
              "that may be wider than the effect you are looking for")
    print(f"evaluating {len(cases)} questions at {len(args.alphas)} scales")

    s = None
    if args.profile:
        with open(args.profile) as f:
            prof = json.load(f)
        s = prof["scale"]

    results = []
    print(f"\n{'alpha':>6} {'neuron':>7} {'correct':>8} {'abstain':>8} {'wrong':>7}")
    print("-" * 40)
    for alpha in args.alphas:
        set_alpha(args.base_url, args.adapter_id, alpha)
        counts = {"correct": 0, "abstained": 0, "wrong": 0}
        per_case = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            answers = list(tqdm(
                pool.map(lambda c: ask(args.base_url, c["question"], args.max_tokens),
                         cases),
                total=len(cases), leave=False, desc=f"  a={alpha}"))
        for c, a in zip(cases, answers):
            v = classify(a, c["aliases"])
            counts[v] += 1
            per_case.append(v)
        n = sum(counts.values())
        eff = None if s is None else 1 + alpha * (s - 1)
        results.append({"alpha": alpha, "effective_scale": eff,
                        "per_case": per_case, "n": n, **counts})
        print(f"{alpha:>6.2f} {('%.3f' % eff) if eff is not None else '?':>7} "
              f"{counts['correct'] / n:>8.1%} {counts['abstained'] / n:>8.1%} "
              f"{counts['wrong'] / n:>7.1%}")

    base = next((r for r in results if r["alpha"] == 0.0), results[0])

    # Paired comparison. The same questions are asked at every scale with greedy
    # decoding, so only the questions that changed verdict carry information --
    # McNemar, not two independent proportions.
    print("\nvs alpha=0, counting only questions whose verdict changed:")
    for r in results:
        if r is base:
            continue
        fixed = sum(1 for b, x in zip(base["per_case"], r["per_case"])
                    if b == "wrong" and x != "wrong")
        broken = sum(1 for b, x in zip(base["per_case"], r["per_case"])
                     if b != "wrong" and x == "wrong")
        lost = sum(1 for b, x in zip(base["per_case"], r["per_case"])
                   if b == "correct" and x == "abstained")
        disc = fixed + broken
        verdict = ""
        if disc >= 10:
            # McNemar without continuity correction; chi2 > 3.84 is p < 0.05
            chi2 = (fixed - broken) ** 2 / disc
            verdict = f"  chi2={chi2:.1f}{' significant' if chi2 > 3.84 else ''}"
        elif disc:
            verdict = "  too few changes to call"
        print(f"  a={r['alpha']:.2f}: {fixed} wrong->not, {broken} not->wrong, "
              f"{lost} correct->abstained{verdict}")

    if args.save and args.profile:
        prof.setdefault("evaluations", []).append({
            "kind": "server_alpha_sweep",
            "base_url": args.base_url,
            "n_eval": len(cases),
            "sweep": [{k: v for k, v in r.items() if k != "per_case"}
                      for r in results],
        })
        with open(args.profile, "w") as f:
            json.dump(prof, f, indent=2)
        print(f"\nrecorded the sweep in {args.profile}")
        print("The profile's own `scale` is unchanged: alpha is a serving-time "
              "dial, not a property of the neuron set.")


if __name__ == "__main__":
    main()
