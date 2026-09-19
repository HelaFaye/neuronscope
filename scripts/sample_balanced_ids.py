#!/usr/bin/env python3
"""
Stage 3: pick an equal number of faithful and hallucinated question ids.

Unchanged in behaviour from the upstream script; included so the package is
self-contained. Use --exclude to keep the test split disjoint from train.

    python scripts/sample_balanced_ids.py \
        --input_path data/answer_tokens.jsonl \
        --output_path data/train_qids.json --num_samples 400
"""
import argparse, json, random

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--num_samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude", help="Another qids.json to exclude (for a test split)")
    a = p.parse_args()
    random.seed(a.seed)

    taken = set()
    if a.exclude:
        with open(a.exclude) as f:
            prev = json.load(f)
        taken = set(prev["t"]) | set(prev["f"])

    t, f_ = [], []
    with open(a.input_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            qid = next(iter(rec))
            if qid in taken:
                continue
            lab = rec[qid].get("judge")
            (t if lab == "true" else f_ if lab == "false" else []).append(qid)

    print(f"available - faithful: {len(t)}, hallucinated: {len(f_)}")
    n = min(a.num_samples, len(t), len(f_))
    if n < a.num_samples:
        print(f"warning: only {n} per class available")
    if n == 0:
        raise SystemExit("no balanced pairs; collect more responses")

    with open(a.output_path, "w") as fh:
        json.dump({"t": random.sample(t, n), "f": random.sample(f_, n)}, fh, indent=2)
    print(f"wrote {n * 2} ids to {a.output_path}")

if __name__ == "__main__":
    main()
