#!/usr/bin/env python3
"""
Stage 5: train the sparse classifier and export the H-Neuron set.

Two changes from the upstream scripts/classifier.py:

  1. Preallocates a float32 matrix and fills it in place, and defaults to the
     saga solver. Upstream builds a Python list of arrays then calls np.array
     (one full copy), and liblinear then promotes to float64 (another copy at
     double width). On a 32-layer model with ~14k intermediate that is the
     difference between ~5GB and ~20GB.
  2. Exports h_neurons.json with explicit (layer, neuron) pairs read from
     neuron_index.json, so the intervention step never has to infer the
     flat-index layout.

    python scripts/classifier.py \
        --acts_root data/activations \
        --train_ids data/train_qids.json \
        --train_mode 3-vs-1 --penalty l1 --C 1.0 \
        --test_ids data/test_qids.json --test_acts_root data/activations_test \
        --out_dir models
"""

import argparse
import json
import os

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hostcheck import guard  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts_root", required=True,
                   help="Directory containing answer_tokens/ etc. and neuron_index.json")
    p.add_argument("--train_ids")
    p.add_argument("--test_ids")
    p.add_argument("--test_acts_root", help="Defaults to --acts_root")
    p.add_argument("--ans_dir", default="answer_tokens")
    p.add_argument("--other_dir", default="all_except_answer_tokens")

    p.add_argument("--train_mode", choices=["1-vs-1", "3-vs-1"], default="3-vs-1")
    p.add_argument("--penalty", choices=["l1", "l2"], default="l1")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--solver", default="saga",
                   help="saga preserves float32. liblinear promotes to float64.")
    p.add_argument("--max_iter", type=int, default=1000)

    p.add_argument("--out_dir", default="models")
    p.add_argument("--strict-host", action="store_true",
                   help="Refuse to start if the estimate exceeds this machine, "
                        "instead of warning")
    p.add_argument("--load", help="Skip training, evaluate this classifier.npz")
    p.add_argument("--top-experts", type=int,
                   help="MoE only: keep the N most-activated experts per layer. "
                        "A 48x128x768 model is 4.7M features and ~30GB of "
                        "classifier matrix; --top-experts 16 cuts that to ~3.8GB.")
    return p.parse_args()


def load_qids(path):
    with open(path) as f:
        ids = json.load(f)
    return set(ids.get("t", [])) | set(ids.get("f", []))


def collect_paths(ids_path, acts_root, ans_dir, other_dir, mode):
    """Returns [(path, label)] without loading anything."""
    with open(ids_path) as f:
        ids = json.load(f)

    entries = []
    for qid in ids["f"]:
        entries.append((os.path.join(acts_root, ans_dir, f"act_{qid}.npy"), 1))
    for qid in ids["t"]:
        entries.append((os.path.join(acts_root, ans_dir, f"act_{qid}.npy"), 0))
    if mode == "3-vs-1":
        for key in ("t", "f"):
            for qid in ids[key]:
                entries.append(
                    (os.path.join(acts_root, other_dir, f"act_{qid}.npy"), 0)
                )
    return [(p, y) for p, y in entries if os.path.exists(p)]


def load_matrix(entries, n_features, keep=None, shape=None):
    """Fill a preallocated float32 array. No intermediate list of arrays."""
    n = len(entries)
    gb = n * n_features * 4 / 1e9
    print(f"allocating {n} x {n_features:,} float32 = {gb:.1f} GB")
    X = np.empty((n, n_features), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    for i, (path, label) in enumerate(tqdm(entries, desc="loading")):
        a = np.load(path)
        if keep is not None:
            a = a.reshape(shape)[keep]        # [n_layers*top_experts, n_neurons]
        X[i] = a.ravel().astype(np.float32, copy=False)
        y[i] = label
    return X, y


def evaluate(coef, intercept, X, y, name):
    scores = X @ coef + intercept
    preds = (scores >= 0).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )
    print(f"\n--- {name} ---")
    print(f"accuracy : {accuracy_score(y, preds):.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall   : {rec:.4f}")
    print(f"f1       : {f1:.4f}")
    if len(set(y.tolist())) > 1:
        print(f"auroc    : {roc_auc_score(y, scores):.4f}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.train_ids and args.test_ids:
        overlap = load_qids(args.train_ids) & load_qids(args.test_ids)
        if overlap:
            sample = ", ".join(sorted(overlap)[:5])
            suffix = "..." if len(overlap) > 5 else ""
            raise SystemExit(
                f"train/test qid overlap: {len(overlap)} ids ({sample}{suffix}). "
                "Use disjoint qid files before reporting test metrics."
            )

    with open(os.path.join(args.acts_root, "neuron_index.json")) as f:
        idx_meta = json.load(f)
    n_layers = idx_meta["n_layers"]
    n_neurons = idx_meta["n_neurons"]
    n_experts = idx_meta.get("n_experts", 1)
    n_features = n_layers * n_experts * n_neurons
    if n_experts > 1:
        print(f"{n_layers} layers x {n_experts} experts x {n_neurons} neurons "
              f"= {n_features:,} features (MoE)")
    else:
        print(f"{n_layers} layers x {n_neurons} neurons = {n_features:,} features")

    keep = None
    if args.top_experts:
        if n_experts <= 1:
            raise SystemExit("--top-experts only applies to MoE sessions")
        if args.top_experts >= n_experts:
            print(f"  --top-experts {args.top_experts} >= {n_experts}, ignoring")
        else:
            # Rank experts by mean activation across the training set, then keep
            # the busiest. Rarely-routed experts contribute little signal and a
            # lot of noise: few tokens reach them, so their per-cell estimates
            # are the least reliable in the matrix.
            probe = collect_paths(args.train_ids, args.acts_root, args.ans_dir,
                                  args.other_dir, "1-vs-1")[:64]
            if not probe:
                raise SystemExit("no files to rank experts from")
            acc = np.zeros((n_layers, n_experts), dtype=np.float64)
            for path, _ in probe:
                a = np.load(path).reshape(n_layers, n_experts, n_neurons)
                acc += np.abs(a).mean(axis=2)
            order = np.argsort(-acc, axis=1)[:, :args.top_experts]
            keep = np.zeros((n_layers, n_experts), dtype=bool)
            for l in range(n_layers):
                keep[l, order[l]] = True
            n_features = n_layers * args.top_experts * n_neurons
            print(f"  keeping {args.top_experts}/{n_experts} experts per layer "
                  f"-> {n_features:,} features "
                  f"({n_features / (n_layers * n_experts * n_neurons) * 100:.0f}%)")

    if args.load:
        blob = np.load(args.load)
        coef, intercept = blob["coef"], float(blob["intercept"])
    else:
        if not args.train_ids:
            raise SystemExit("need --train_ids or --load")
        entries = collect_paths(
            args.train_ids, args.acts_root, args.ans_dir,
            args.other_dir, args.train_mode,
        )
        if not entries:
            raise SystemExit("no activation files found; check --acts_root")
        guard("classifier", n_features=n_features, n_rows=len(entries),
              solver=args.solver, strict=args.strict_host)
        X, y = load_matrix(entries, n_features, keep,
                           (n_layers, n_experts, n_neurons))

        print(f"fitting {args.penalty} / C={args.C} / solver={args.solver} "
              f"on {len(y)} rows ({int(y.sum())} positive)")
        clf = LogisticRegression(
            penalty=args.penalty, C=args.C, solver=args.solver,
            max_iter=args.max_iter, random_state=42, verbose=1,
        )
        clf.fit(X, y)
        coef = clf.coef_[0].astype(np.float32)
        intercept = float(clf.intercept_[0])

        evaluate(coef, intercept, X, y, "train")
        del X, y

        np.savez(os.path.join(args.out_dir, "classifier.npz"),
                 coef=coef, intercept=intercept,
                 n_layers=n_layers, n_neurons=n_neurons)

    # H-Neurons are the positively weighted ones: higher contribution pushes
    # the prediction toward "hallucinated".
    pos = np.where(coef > 0)[0]
    nonzero = int((coef != 0).sum())
    print(f"\nnonzero weights : {nonzero:,} ({nonzero / n_features * 100:.4f}%)")
    print(f"H-Neurons (w>0) : {len(pos):,} ({len(pos) / n_features * 100:.4f}%)")
    if args.penalty == "l1" and nonzero > n_features * 0.01:
        print("  Sparsity looks low for L1. Lower --C for a tighter set;\n"
              "  the paper reports under 0.1% of neurons.")

    by_layer = {}
    if n_experts > 1:
        # Columns are (layer, expert) pairs; with --top-experts only the kept
        # ones are present, so map back through the same ordering.
        pairs = ([(l, e) for l in range(n_layers) for e in range(n_experts)
                  if keep is None or keep[l, e]])
        by_expert = {}
        for flat in pos.tolist():
            slot, neuron = divmod(flat, n_neurons)
            layer, expert = pairs[slot]
            by_expert.setdefault(f"{layer}:{expert}", []).append(neuron)
            by_layer.setdefault(str(layer), []).append(expert * n_neurons + neuron)
        with open(os.path.join(args.out_dir, "h_neurons_moe.json"), "w") as f:
            json.dump({"n_layers": n_layers, "n_experts": n_experts,
                       "n_neurons": n_neurons, "total": len(pos),
                       "by_layer_expert": by_expert}, f, indent=2)
        print(f"wrote {args.out_dir}/h_neurons_moe.json "
              f"({len(by_expert)} (layer, expert) cells)")
    else:
        for flat in pos.tolist():
            layer, neuron = divmod(flat, n_neurons)
            by_layer.setdefault(str(layer), []).append(neuron)

    with open(os.path.join(args.out_dir, "h_neurons.json"), "w") as f:
        json.dump({
            "model_path": idx_meta.get("model_path"),
            "n_layers": n_layers,
            "n_experts": n_experts,
            "n_neurons": n_neurons if n_experts == 1 else n_experts * n_neurons,
            "total": len(pos),
            "by_layer": by_layer,
            # Carried through so downstream tools know which modules these
            # indices address (e.g. a CLIP vision tower vs a text decoder).
            **{k: idx_meta[k] for k in ("arch", "tower", "module_template")
               if k in idx_meta},
        }, f, indent=2)

    counts = {int(k): len(v) for k, v in by_layer.items()}
    if counts:
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        print("densest layers  : " +
              ", ".join(f"L{k}={v}" for k, v in top))

    if args.test_ids:
        root = args.test_acts_root or args.acts_root
        entries = collect_paths(
            args.test_ids, root, args.ans_dir, args.other_dir, "1-vs-1"
        )
        if entries:
            Xt, yt = load_matrix(entries, n_features, keep,
                                 (n_layers, n_experts, n_neurons))
            evaluate(coef, intercept, Xt, yt, "test")

    print(f"\nwrote {args.out_dir}/classifier.npz and {args.out_dir}/h_neurons.json")


if __name__ == "__main__":
    main()
