#!/usr/bin/env python3
"""
Which neurons a model recruits for which kind of task.

Train one classifier per task type, then ask whether the resulting H-Neuron
sets are the same neurons. Three outcomes, all informative:

  mostly shared      hallucination is a general mechanism in this model; one
                     profile should transfer, and delegating to a specialist
                     buys you nothing on this axis
  mostly disjoint    the model has task-specific failure modes; you need a
                     profile per domain, and a specialist model is plausible
  partly shared      the interesting case; the shared core is the general
                     mechanism and the specific neurons are where a domain
                     profile earns its keep

    python scripts/task_neurons.py \\
        --acts_root data/activations --ids data/train_qids.json \\
        --samples data/consistency_samples.jsonl --task-field task \\
        --C 1.0 --out_dir models/by_task

Overlap is reported against chance, not raw. Two sets of 400 drawn from 458,752
neurons overlap by about 0.35 cells by accident; two sets of 40,000 overlap by
3,500. Raw Jaccard between small sets always looks low and between large sets
always looks high, so neither number means anything on its own.
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import collect_paths, load_matrix  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts_root", required=True)
    p.add_argument("--ids", required=True, help="train_qids.json")
    p.add_argument("--samples", required=True,
                   help="jsonl carrying the task label per qid")
    p.add_argument("--task-field", default="task",
                   help="field in the sample record holding the task type")
    p.add_argument("--tasks-json",
                   help="alternative: {qid: task} mapping")
    p.add_argument("--ans_dir", default="answer_tokens")
    p.add_argument("--other_dir", default="all_except_answer_tokens")
    p.add_argument("--train_mode", default="3-vs-1")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--solver", default="saga")
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--min-per-task", type=int, default=60,
                   help="skip a task with fewer balanced samples than this")
    p.add_argument("--out_dir", default="models/by_task")
    return p.parse_args()


def load_tasks(args):
    if args.tasks_json:
        with open(args.tasks_json) as f:
            return json.load(f)
    tasks = {}
    with open(args.samples, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = next(iter(rec))
            t = rec[qid].get(args.task_field)
            if t:
                tasks[qid] = str(t)
    return tasks


def expected_overlap(a, b, total):
    """Cells two independent sets of size a and b share by chance."""
    return a * b / total if total else 0.0


def jaccard(x, y):
    u = len(x | y)
    return len(x & y) / u if u else 0.0


def enrichment(x, y, total):
    """Observed intersection over chance. 1.0 is indistinguishable from
    independent selection; the ratio is what carries meaning, not Jaccard."""
    exp = expected_overlap(len(x), len(y), total)
    return (len(x & y) / exp) if exp > 0 else float("nan")


def layer_profile(neurons, n_layers, n_neurons, bins=10):
    """Where in depth a set sits, on a common basis so models and tasks with
    different layer counts stay comparable."""
    counts = np.zeros(n_layers, dtype=float)
    for flat in neurons:
        counts[flat // n_neurons] += 1
    if counts.sum() == 0:
        return np.zeros(bins)
    x = np.linspace(0, 1, n_layers)
    out = np.interp(np.linspace(0, 1, bins), x, counts)
    tot = out.sum()
    return out / tot if tot else out


def train_one(paths, n_features, args, n_layers, n_experts, n_neurons):
    X, y = load_matrix(paths, n_features, None,
                       (n_layers, n_experts, n_neurons))
    clf = LogisticRegression(penalty="l1", C=args.C, solver=args.solver,
                             max_iter=args.max_iter, random_state=42)
    clf.fit(X, y)
    coef = clf.coef_[0].astype(np.float32)
    del X, y
    return coef


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.acts_root, "neuron_index.json")) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    n_neurons = meta["n_neurons"]
    n_experts = meta.get("n_experts", 1)
    n_features = n_layers * n_experts * n_neurons
    print(f"{n_features:,} features")

    with open(args.ids) as f:
        ids = json.load(f)
    tasks = load_tasks(args)
    if not tasks:
        raise SystemExit(
            f"no task labels found. Add a '{args.task_field}' field to the "
            "sample records, or pass --tasks-json.")

    by_task = {}
    for label in ("t", "f"):
        for qid in ids[label]:
            t = tasks.get(qid)
            if t:
                by_task.setdefault(t, {"t": [], "f": []})[label].append(qid)

    print(f"\n{len(by_task)} task types:")
    usable = {}
    for t, d in sorted(by_task.items()):
        n = min(len(d["t"]), len(d["f"]))
        mark = ""
        if n * 2 < args.min_per_task:
            mark = f"  skipped, {n} balanced pairs < --min-per-task"
        else:
            usable[t] = d
        print(f"  {t:<18} {len(d['t']):>4} correct / {len(d['f']):>4} "
              f"hallucinated{mark}")
    if len(usable) < 2:
        raise SystemExit("need at least two usable task types to compare")

    sets, coefs = {}, {}
    for t, d in usable.items():
        tmp = os.path.join(args.out_dir, f"_{t}_ids.json")
        with open(tmp, "w") as f:
            json.dump(d, f)
        paths = collect_paths(tmp, args.acts_root, args.ans_dir,
                              args.other_dir, args.train_mode)
        os.remove(tmp)
        if not paths:
            print(f"  {t}: no activation files, skipping")
            continue
        print(f"\ntraining {t} on {len(paths)} rows")
        coef = train_one(paths, n_features, args, n_layers, n_experts,
                         n_neurons)
        pos = set(np.where(coef > 0)[0].tolist())
        sets[t], coefs[t] = pos, coef
        print(f"  {len(pos):,} H-Neurons "
              f"({len(pos) / n_features * 100:.4f}% of all)")

    if len(sets) < 2:
        raise SystemExit("fewer than two tasks trained")

    names = sorted(sets)
    print(f"\n{'pair':<32} {'shared':>8} {'chance':>8} {'ratio':>7} "
          f"{'jaccard':>8}")
    print("-" * 68)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sets[names[i]], sets[names[j]]
            exp = expected_overlap(len(a), len(b), n_features)
            rat = enrichment(a, b, n_features)
            jac = jaccard(a, b)
            label = f"{names[i]} / {names[j]}"
            print(f"  {label:<30} {len(a & b):>8,} {exp:>8.1f} {rat:>7.1f}x "
                  f"{jac:>8.3f}")
            pairs.append({"a": names[i], "b": names[j],
                          "shared": len(a & b), "expected": round(exp, 2),
                          "ratio": None if np.isnan(rat) else round(rat, 2),
                          "jaccard": round(jac, 4)})

    ratios = [p["ratio"] for p in pairs if p["ratio"] is not None]
    if ratios:
        med = float(np.median(ratios))
        print(f"\nmedian enrichment {med:.1f}x over chance")
        if med > 20:
            print("  Largely the same neurons. Hallucination looks like a "
                  "general mechanism\n  here: one profile should transfer, and "
                  "a task specialist buys nothing\n  on this axis.")
        elif med < 3:
            print("  Largely different neurons. The model has task-specific "
                  "failure modes,\n  so a profile per domain is warranted and "
                  "delegating to a specialist\n  has something underneath it.")
        else:
            print("  Partly shared. The intersection is the general mechanism; "
                  "the rest is\n  where a domain profile earns its keep.")

    print(f"\ndepth profile (share of H-Neurons by relative depth, 10 bins)")
    for t in names:
        prof = layer_profile(sets[t], n_layers, n_experts * n_neurons)
        print(f"  {t:<18} " + " ".join(f"{v:.2f}" for v in prof))

    out = {"n_features": n_features, "n_layers": n_layers,
           "n_neurons": n_neurons, "n_experts": n_experts,
           "tasks": {t: {"count": len(s),
                         "by_layer": {}} for t, s in sets.items()},
           "pairs": pairs}
    for t, s in sets.items():
        by_layer = {}
        for flat in sorted(s):
            layer, neuron = divmod(flat, n_experts * n_neurons)
            by_layer.setdefault(str(layer), []).append(neuron)
        out["tasks"][t]["by_layer"] = by_layer
        with open(os.path.join(args.out_dir, f"h_neurons_{t}.json"), "w") as f:
            json.dump({"n_layers": n_layers,
                       "n_neurons": n_experts * n_neurons,
                       "total": len(s), "task": t,
                       "by_layer": by_layer}, f, indent=2)

    with open(os.path.join(args.out_dir, "task_overlap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_dir}/h_neurons_<task>.json and task_overlap.json")
    print("Each is a normal profile: tune, export and evaluate it the same way.")


if __name__ == "__main__":
    main()
