#!/usr/bin/env python3
"""
Cross-validated hallucination detection, plus the checks that say whether the
number means what it appears to.

One held-out test set of ~38 questions gives an AUROC with a +/-0.17 interval.
This pools every extracted question (train and test roots), rotates each one
through the held-out role, and reports:

  1. out-of-fold AUROC with a bootstrap 95% CI, repeated over several shuffles
  2. a label-permutation test: the same pipeline on shuffled labels, so the
     p-value accounts for the 393k-feature search, not just the final score
  3. trivial baselines: response length, reasoning length, answer length.
     If one of these matches the classifier, the neurons may be reading length
  4. AUROC within answer types (number / name / other), so "it detects
     obscure names" can be checked against data you already have

Fitting uses liblinear L1, which on ~150 rows takes seconds rather than the
25 minutes saga needed. The solution differs slightly from classifier.py's,
so treat this as an estimate of the method, not of that exact .npz.

    python scripts/cv.py --roots data/activations data/activations_test \\
        --ids data/train_qids.json data/test_qids.json \\
        --samples data/answer_tokens.jsonl --C 10
"""
import argparse
import json
import os
import re
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True,
                   help="activation roots, paired with --ids in order")
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--samples", default="data/answer_tokens.jsonl",
                   help="for the length and answer-type checks")
    p.add_argument("--ans_dir", default="answer_tokens")
    p.add_argument("--C", type=float, nargs="+", default=[10.0],
                   help="several values are all reported; picking the best "
                        "one afterwards is optimistic")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--permutations", type=int, default=50,
                   help="0 to skip. Each costs one full CV run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write per-question out-of-fold scores here")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="folds fitted in parallel (threads; liblinear releases "
                        "the GIL). Default: half your logical cores")
    return p.parse_args()


def load(args):
    if len(args.roots) != len(args.ids):
        raise SystemExit("--roots and --ids must pair up one to one")
    rows, missing = [], 0
    for root, ids_path in zip(args.roots, args.ids):
        with open(ids_path) as f:
            ids = json.load(f)
        for key, label in (("f", 1), ("t", 0)):
            for qid in ids[key]:
                path = os.path.join(root, args.ans_dir, f"act_{qid}.npy")
                if os.path.exists(path):
                    rows.append((qid, path, label))
                else:
                    missing += 1
    seen, uniq = set(), []
    for r in rows:                       # a qid in both splits counts once
        if r[0] not in seen:
            seen.add(r[0])
            uniq.append(r)
    if missing:
        print(f"note: {missing} ids have no activation file (not extracted)")
    if len(uniq) < len(rows):
        print(f"note: {len(rows) - len(uniq)} duplicate qids across splits dropped")
    first = np.load(uniq[0][1])
    X = np.empty((len(uniq), first.size), dtype=np.float32)
    for i, (_, path, _) in enumerate(uniq):
        X[i] = np.load(path).ravel()
    y = np.array([r[2] for r in uniq], dtype=np.int8)
    qids = [r[0] for r in uniq]
    return qids, X, y, first.shape


def fit_predict(Xtr, ytr, Xte, C):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                             max_iter=5000)
    clf.fit(Xtr, ytr)
    return clf.decision_function(Xte), int((clf.coef_ != 0).sum())


JOBS = 1


def oof_scores(X, y, C, folds, seed):
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedKFold
    splits = list(StratifiedKFold(folds, shuffle=True,
                                  random_state=seed).split(X, y))
    res = Parallel(n_jobs=JOBS, prefer="threads")(
        delayed(fit_predict)(X[tr], y[tr], X[te], C) for tr, te in splits)
    oof = np.zeros(len(y))
    for (_, te), (s, _) in zip(splits, res):
        oof[te] = s
    return oof, [k for _, k in res]


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if len(set(y.tolist())) > 1 else float("nan")


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i].tolist())) > 1:
            out.append(auroc(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def answer_type(text):
    t = (text or "").strip().strip(".\"'")
    if not t:
        return "empty"
    if re.search(r"\d", t):
        return "number/date"
    words = t.split()
    if all(w[:1].isupper() for w in words if w[:1].isalpha()):
        return "name"
    return "other"


def split_response(resp):
    resp = resp or ""
    if "</think>" in resp:
        think, ans = resp.split("</think>", 1)
        return think, ans.strip()
    return "", resp.strip()


def main():
    args = parse_args()
    global JOBS
    JOBS = args.jobs
    t0 = time.time()
    qids, X, y, shape = load(args)
    print(f"{len(y)} questions ({int(y.sum())} hallucinated, "
          f"{int(len(y) - y.sum())} correct), {X.shape[1]:,} features "
          f"[{shape[0]} layers x {shape[-1]} neurons]")
    if min(y.sum(), len(y) - y.sum()) < args.folds:
        raise SystemExit("too few of one class for this many folds")

    # ---- 1. repeated cross-validation ------------------------------------
    best = None
    for C in args.C:
        runs = []
        for r in range(args.repeats):
            oof, nnz = oof_scores(X, y, C, args.folds, args.seed + r)
            runs.append(oof)
        aucs = [auroc(y, o) for o in runs]
        mean_oof = np.mean(runs, axis=0)
        lo, hi = boot_ci(y, mean_oof, seed=args.seed)
        a = auroc(y, mean_oof)
        print(f"\nC={C:g}: out-of-fold AUROC {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
              f"   per-shuffle {', '.join(f'{v:.3f}' for v in aucs)}"
              f"   ~{int(np.median(nnz))} nonzero weights/fold")
        if best is None or a > best[1]:
            best = (C, a, mean_oof)
    C, a_real, oof = best
    if len(args.C) > 1:
        print(f"(best of {len(args.C)} C values reported below; choosing it "
              f"after seeing the results inflates it slightly)")

    # ---- 2. permutation test ---------------------------------------------
    if args.permutations:
        rng = np.random.default_rng(args.seed + 1000)
        null = []
        print(f"\npermutation test: {args.permutations} shuffled-label runs "
              f"at C={C:g} ...", flush=True)
        for k in range(args.permutations):
            yp = rng.permutation(y)
            o, _ = oof_scores(X, yp, C, args.folds, args.seed + 2000 + k)
            null.append(auroc(yp, o))
        null = np.array(null)
        p = (1 + (null >= a_real).sum()) / (1 + len(null))
        print(f"  shuffled labels: mean AUROC {null.mean():.3f}, "
              f"95th pct {np.percentile(null, 95):.3f}, max {null.max():.3f}")
        print(f"  real labels {a_real:.3f}  ->  p = {p:.3f}"
              f"{'  (resolution limited by permutation count)' if p <= 1 / len(null) * 1.01 + 1e-9 else ''}")

    # ---- 3 & 4. confounds from the text ----------------------------------
    meta = {}
    if args.samples and os.path.exists(args.samples):
        want = set(qids)
        with open(args.samples, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                qid = next(iter(rec))
                if qid in want:
                    meta[qid] = rec[qid]
    if meta:
        have = np.array([q in meta for q in qids])
        think_len, ans_len, kinds = [], [], []
        for q in qids:
            th, an = split_response(meta.get(q, {}).get("response"))
            think_len.append(len(th))
            ans_len.append(len(an))
            kinds.append(answer_type(an))
        think_len, ans_len = np.array(think_len), np.array(ans_len)
        yh, oh = y[have], oof[have]

        print("\nbaselines (AUROC from one number alone, no neurons):")
        base = {"reasoning length": think_len[have],
                "answer length": ans_len[have]}
        for name, v in base.items():
            b = auroc(yh, v)
            print(f"  {name:<18} {b:.3f}"
                  f"{'   <- comparable to the classifier' if max(b, 1 - b) >= a_real - 0.05 else ''}")
        r = np.corrcoef(oh, think_len[have])[0, 1]
        print(f"  classifier score vs reasoning length: r = {r:+.2f}")

        # Does the classifier add anything once length is known? Compare
        # within length halves.
        med = np.median(think_len[have])
        for name, m in (("short reasoning", think_len[have] <= med),
                        ("long reasoning", think_len[have] > med)):
            ys = yh[m]
            if min(ys.sum(), len(ys) - ys.sum()) >= 5:
                print(f"  within {name:<16} (n={m.sum():>3}): classifier AUROC "
                      f"{auroc(ys, oh[m]):.3f}")

        print("\nby answer type (model's final answer):")
        kinds = np.array(kinds)[have]
        for k in sorted(set(kinds)):
            m = kinds == k
            ys = yh[m]
            nh, nc = int(ys.sum()), int(len(ys) - ys.sum())
            line = f"  {k:<12} n={m.sum():>3} ({nh} halluc / {nc} correct)"
            if min(nh, nc) >= 5:
                line += f"   AUROC {auroc(ys, oh[m]):.3f}"
            else:
                line += "   too few of one class to score"
            print(line)
        print("  Within-type AUROC near the overall figure means the signal is "
              "not just\n  'this answer is a name'. Near 0.5 means it might be.")

    if args.out:
        with open(args.out, "w") as f:
            for q, lab, s in zip(qids, y.tolist(), oof.tolist()):
                f.write(json.dumps({"qid": q, "hallucinated": lab,
                                    "oof_score": round(s, 4)}) + "\n")
        print(f"\nper-question scores -> {args.out}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
