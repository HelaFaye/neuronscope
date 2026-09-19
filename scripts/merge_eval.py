#!/usr/bin/env python3
"""
Compare models item by item: what improved, what regressed.

Aggregate accuracy hides the thing you actually need to know. A merge that goes
from 60% to 64% might have fixed 12 items and broken 8, or fixed 4 and broken 0.
Those are completely different outcomes and the headline number is identical.
This reports both directions and names the regressions.

Endpoint-based, so it does not care how the models were produced -- merged with
mergekit, merged with merge_selective.py, fine-tuned, or just two different
checkpoints. Serve each one and point at it.

    # serve variants on different ports, then
    python scripts/merge_eval.py \\
        --endpoint base=http://127.0.0.1:8080 \\
        --endpoint merged=http://127.0.0.1:8081 \\
        --endpoint donor=http://127.0.0.1:8082 \\
        --tasks data/eval_tasks.jsonl --reference base --out cmp.json

Task file, one object per line. Two gradeable kinds, mixable:

    {"id": "qa-1",  "prompt": "...", "aliases": ["Paris"]}
    {"id": "cod-1", "prompt": "...", "modules": ["requests"]}

A third kind, with neither field, is a CANARY: ungraded, recorded only as
answered or refused. Put general-capability prompts here. A merge that improves
your target metric while the canary refusal rate climbs has not improved the
model, it has narrowed it, and that is the failure the aggregate number hides.
"""

import argparse
import ast
import importlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

THINK_CLOSE = "</think>"
ABSTAIN = ["i don't know", "i do not know", "i'm not sure", "i am not sure",
           "cannot determine", "can't determine", "unable to answer",
           "no information", "i don't have", "unsure"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", action="append", required=True,
                   metavar="LABEL=URL", help="repeatable")
    p.add_argument("--tasks", required=True)
    p.add_argument("--reference", help="label to compare others against "
                                       "(default: the first endpoint)")
    p.add_argument("--model", help="model field for OpenAI-compatible servers")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--show", type=int, default=10,
                   help="how many regressions to print in full")
    p.add_argument("--out")
    return p.parse_args()


def strip_think(t):
    return t.split(THINK_CLOSE, 1)[1] if THINK_CLOSE in t else t


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", s.lower()).split())


# ------------------------------------------------------------------ graders

def grade_qa(text, task):
    body = strip_think(text).strip()
    if not body:
        return "abstained"
    low = body.lower()
    if any(m in low for m in ABSTAIN):
        return "abstained"
    n = norm(body)
    return "correct" if any(norm(a) and norm(a) in n
                            for a in task["aliases"]) else "wrong"


_mods = {}


def _resolves(module, attr):
    if module not in _mods:
        try:
            _mods[module] = importlib.import_module(module)
        except Exception:
            _mods[module] = None
    m = _mods[module]
    return None if m is None else hasattr(m, attr)


def _code_blocks(text):
    out, cur, inside = [], [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if inside:
                out.append("\n".join(cur)); cur, inside = [], False
            else:
                inside = True
            continue
        if inside:
            cur.append(line)
    if inside and cur:
        out.append("\n".join(cur))
    if out:
        return "\n\n".join(out)
    try:
        ast.parse(text); return text
    except SyntaxError:
        return ""


def grade_code(text, task):
    code = _code_blocks(strip_think(text))
    if not code.strip():
        return "abstained"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "unparsable"
    allowed = set(task["modules"])
    alias, froms, assigned = {}, [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                if al.name.split(".")[0] in allowed:
                    alias[al.asname or al.name.split(".")[0]] = al.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in allowed \
                    and node.level == 0:
                froms += [(node.module, al.name) for al in node.names
                          if al.name != "*"]
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for s in ast.walk(t):
                    if isinstance(s, ast.Name):
                        assigned.add(s.id)
    syms = set(froms)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in alias and node.value.id not in assigned:
                syms.add((alias[node.value.id], node.attr))
    bad = [f"{m}.{a}" for m, a in sorted(syms) if _resolves(m, a) is False]
    return "wrong" if bad else "correct"


def grade_canary(text, task):
    body = strip_think(text).strip()
    low = body.lower()
    if not body or any(m in low for m in ABSTAIN):
        return "refused"
    return "answered"


def kind_of(task):
    if task.get("aliases"):
        return "qa"
    if task.get("modules"):
        return "code"
    return "canary"


GRADERS = {"qa": grade_qa, "code": grade_code, "canary": grade_canary}


# ----------------------------------------------------------------- querying

def ask(url, task, model, max_tokens):
    body = {"messages": [{"role": "user", "content": task["prompt"]}],
            "temperature": 0.0, "max_tokens": max_tokens}
    if model:
        body["model"] = model
    r = requests.post(f"{url.rstrip('/')}/v1/chat/completions",
                      json=body, timeout=900)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    txt = msg.get("content") or ""
    if msg.get("reasoning_content") and THINK_CLOSE not in txt:
        txt = f"<think>{msg['reasoning_content']}{THINK_CLOSE}{txt}"
    return txt


def run_endpoint(label, url, tasks, args):
    def one(t):
        try:
            text = ask(url, t, args.model, args.max_tokens)
        except Exception as e:
            return {"id": t["id"], "verdict": "error", "note": str(e)[:120]}
        k = kind_of(t)
        return {"id": t["id"], "kind": k, "verdict": GRADERS[k](text, t),
                "text": strip_think(text).strip()[:400]}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        return list(tqdm(pool.map(one, tasks), total=len(tasks),
                         desc=f"  {label}", leave=False))


# ----------------------------------------------------------------- analysis

GOOD = {"correct", "answered"}


def compare(ref, var, tasks):
    """-> gains, regressions, and canary movement, per item."""
    by_id = {t["id"]: t for t in tasks}
    r = {x["id"]: x for x in ref}
    v = {x["id"]: x for x in var}
    gained, lost, canary_lost, canary_gained = [], [], [], []
    for tid in r:
        if tid not in v:
            continue
        a, b = r[tid]["verdict"], v[tid]["verdict"]
        if a == b or "error" in (a, b):
            continue
        kind = kind_of(by_id[tid])
        rec = {"id": tid, "kind": kind, "from": a, "to": b,
               "prompt": by_id[tid]["prompt"][:110],
               "answer": v[tid].get("text", "")[:160]}
        if kind == "canary":
            (canary_lost if b == "refused" else canary_gained).append(rec)
        elif a not in GOOD and b in GOOD:
            gained.append(rec)
        elif a in GOOD and b not in GOOD:
            lost.append(rec)
    return gained, lost, canary_gained, canary_lost


def mcnemar(n_gain, n_loss):
    d = n_gain + n_loss
    if d == 0:
        return 0.0, d
    return (n_gain - n_loss) ** 2 / d, d


def main():
    args = parse_args()
    endpoints = []
    for spec in args.endpoint:
        if "=" not in spec:
            raise SystemExit(f"--endpoint needs LABEL=URL, got {spec!r}")
        label, url = spec.split("=", 1)
        endpoints.append((label, url))

    tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8")
             if l.strip()]
    if not tasks:
        raise SystemExit("no tasks")
    kinds = {}
    for t in tasks:
        kinds[kind_of(t)] = kinds.get(kind_of(t), 0) + 1
    print(f"{len(tasks)} tasks: " +
          ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    if "canary" not in kinds:
        print("  no canary tasks. Add prompts with neither 'aliases' nor "
              "'modules' to\n  catch a merge that improves the target metric "
              "by narrowing the model.")

    results = {}
    for label, url in endpoints:
        print(f"\nquerying {label} ({url})")
        results[label] = run_endpoint(label, url, tasks, args)

    print(f"\n{'endpoint':<14} " +
          " ".join(f"{k:>10}" for k in ("correct", "wrong", "abstained",
                                        "refused", "error")))
    for label, _ in endpoints:
        c = {}
        for x in results[label]:
            c[x["verdict"]] = c.get(x["verdict"], 0) + 1
        print(f"{label:<14} " + " ".join(
            f"{c.get(k, 0):>10}" for k in ("correct", "wrong", "abstained",
                                           "refused", "error")))

    ref_label = args.reference or endpoints[0][0]
    if ref_label not in results:
        raise SystemExit(f"--reference {ref_label} is not an endpoint")

    out = {"tasks": len(tasks), "reference": ref_label, "comparisons": []}
    for label, _ in endpoints:
        if label == ref_label:
            continue
        g, l, cg, cl = compare(results[ref_label], results[label], tasks)
        chi2, disc = mcnemar(len(g), len(l))
        verdict = ("significant" if chi2 > 3.84 else "not significant") \
            if disc >= 10 else "too few changes to call"
        print(f"\n{label} vs {ref_label}")
        print(f"  gained {len(g)}, regressed {len(l)}   "
              f"net {len(g) - len(l):+d}   chi2={chi2:.1f} ({verdict})")
        if cg or cl:
            print(f"  canary: {len(cl)} newly refused, {len(cg)} newly answered")
            if len(cl) > len(cg):
                print("  !! the model refuses more general prompts than before. "
                      "A target-metric\n     gain bought with this is a "
                      "narrower model, not a better one.")
        if l:
            print(f"\n  regressions ({min(args.show, len(l))} of {len(l)}):")
            for r in l[:args.show]:
                print(f"    [{r['kind']}] {r['id']}  {r['from']} -> {r['to']}")
                print(f"      {r['prompt']}")
                if r["answer"]:
                    print(f"      got: {r['answer'][:100]}")
        out["comparisons"].append({
            "endpoint": label, "gained": g, "regressed": l,
            "canary_newly_refused": cl, "canary_newly_answered": cg,
            "chi2": chi2, "discordant": disc})

    if args.out:
        with open(args.out, "w") as f:
            json.dump({**out, "raw": results}, f, indent=2)
        print(f"\nwrote {args.out}")

    print("\nNet gain alone does not settle it. +4 from 12 gained and 8 "
          "regressed is a\ndifferent model from +4 with 4 gained and 0 "
          "regressed; read the lists.")


if __name__ == "__main__":
    main()
