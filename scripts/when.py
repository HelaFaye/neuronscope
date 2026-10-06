#!/usr/bin/env python3
"""
Where in the response does the hallucination score spike?

    python scripts/when.py runs/trace-tc32                    # one trace
    python scripts/when.py runs/trace-tc32 runs/trace-tc3     # compare

Pass a hallucinated trace and a correct one together: a per-token pattern only
means something if it shows up in the first and NOT in the second.

Frame 0 and chat control tokens (<|im_start|> etc.) are excluded from peaks.
The first position of a sequence carries very large activations in almost every
transformer regardless of content (the "attention sink"), so it tops the chart
on every trace and says nothing about this one.
"""
import os
import re
import sys
from collections import Counter

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(_here, "..", "viz"), os.path.join(_here, "viz"), _here):
    if os.path.exists(os.path.join(_c, "records.py")):
        sys.path.insert(0, _c)
        break
else:
    raise SystemExit("cannot find viz/records.py -- run from the repo")
from records import Session  # noqa: E402

_CONTROL = re.compile(r"^<\|.*\|>$")


def load(path):
    s = Session(path)
    qid = s.ids[0]
    d = s.get(qid)
    f = d.get("fields") or {}
    if d.get("scores") is None:
        raise SystemExit(f"{path}: no scores -- rerun trace_sample.py with "
                         f"--classifier")
    return qid, f, np.asarray(d["scores"], dtype=np.float32)


def analyse(path):
    qid, f, scores = load(path)
    pieces = f.get("pieces") or []
    stride = int(f.get("stride", 1))
    T = int(f["n_frames"])
    verdict = str(f.get("verdict") or "unknown").lower()
    text = lambda i: "".join(pieces[i * stride:(i + 1) * stride])  # noqa: E731

    print(f"== {path}  ({qid}, verdict: {verdict})")
    print(f"   {T} frames, stride {stride}; score mean {scores.mean():+.2f} "
          f"sd {scores.std():.2f}")

    sink = np.zeros(T, bool)
    sink[0] = True
    for i in range(T):
        if _CONTROL.match(text(i).strip() or "x"):
            sink[i] = True

    close = next((i for i, p in enumerate(pieces) if "</think>" in p), None)
    out = {"path": path, "qid": qid, "verdict": verdict}
    warnings = []

    if close is None:
        print("   no </think>; treating the whole trace as one region")
        ans_idx = np.array([], int)
        rea_idx = np.where(~sink)[0]
    else:
        c = close // stride
        rea_idx = np.array([i for i in range(0, c) if not sink[i]], int)
        ans_idx = np.array([i for i in range(c + 1, T)
                            if not sink[i] and text(i).strip()], int)
        print(f"   </think> at piece {close} "
              f"({close / max(len(pieces), 1) * 100:.0f}% through)")

    if len(rea_idx):
        r = scores[rea_idx]
        print(f"   reasoning  {len(rea_idx):>4} frames  mean {r.mean():+.2f}  "
              f"max {r.max():+.2f}")
        out["reasoning"] = float(r.mean())
    if len(ans_idx):
        a = scores[ans_idx]
        ans_text = "".join(text(i) for i in ans_idx).strip()
        pct = (r < a.mean()).mean() * 100 if len(rea_idx) else float("nan")
        print(f"   answer     {len(ans_idx):>4} frames  mean {a.mean():+.2f}  "
              f"({pct:.0f}th pct of reasoning)  {ans_text[:40]!r}")
        for i in ans_idx[:12]:
            print(f"      frame {i:>4}  {scores[i]:+6.2f}  {text(i)!r}")
        out.update(answer=float(a.mean()), answer_pct=float(pct),
                   answer_text=ans_text)
        if len(ans_idx) < 3:
            warnings.append(
                f"SHORT ANSWER: {len(ans_idx)} content token(s). One token's "
                f"per-token score is too noisy to compare on its own.")

    cand = np.where(~sink)[0]
    k = min(8, len(cand))
    top = cand[np.argsort(-scores[cand])[:k]]
    print(f"   top {k} frames (frame 0 and control tokens excluded):")
    for i in sorted(top):
        where = ("" if close is None else
                 "reasoning" if (i + 1) * stride <= close else
                 "boundary" if i * stride <= close else "ANSWER")
        print(f"      frame {i:>4}  {scores[i]:+6.2f}  {where:<9} "
              f"{text(i)[:44]!r}")
    top_txt = [text(i).strip() for i in top]
    out["top"] = top_txt

    common, n = Counter(top_txt).most_common(1)[0] if top_txt else ("", 0)
    if n >= max(3, k // 2):
        warnings.append(f"LEXICAL: {n} of the top {k} frames are {common!r}.")

    if (verdict in ("false", "hallucinated", "0") and "answer_pct" in out
            and out["answer_pct"] < 25):
        warnings.append(
            f"INVERTED: the fabricated answer sits at the "
            f"{out['answer_pct']:.0f}th percentile of this trace's reasoning; "
            f"it should be near the top.")
    for w in warnings:
        print(f"   ! {w}")
    out["warnings"] = warnings
    print()
    return out


def main():
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        return
    paths = sys.argv[1:] or ["runs/trace-tc32"]
    rows = [analyse(p) for p in paths]

    print("Per-token scores are extrapolation: the classifier was trained on "
          "activations\naveraged over an answer span, so scale and sign here "
          "are not the training scale.")

    if len(rows) < 2:
        print("\nPass a correct trace alongside this one to see whether its "
              "pattern is\nspecific to hallucination.")
        return

    print(f"\n{'trace':<14}{'verdict':<10}{'answer':>8}{'pct':>6}"
          f"{'reason':>8}  answer text")
    for r in rows:
        print(f"{r['qid']:<14}{r['verdict']:<10}"
              f"{r.get('answer', float('nan')):>+8.2f}"
              f"{r.get('answer_pct', float('nan')):>5.0f}%"
              f"{r.get('reasoning', float('nan')):>+8.2f}  "
              f"{r.get('answer_text', '')[:30]!r}")

    bad = [r for r in rows if r["verdict"] in ("false", "hallucinated", "0")]
    good = [r for r in rows if r["verdict"] in ("true", "correct", "1")]
    if bad and good and all("answer" in r for r in bad + good):
        gap = (np.mean([r["answer"] for r in bad])
               - np.mean([r["answer"] for r in good]))
        print(f"\nhallucinated minus correct, answer region: {gap:+.2f}")
        if gap <= 0:
            print("The hallucinated answer does not score higher than the "
                  "correct one per-token.\nNo per-token timing claim is "
                  "supported by these traces.")
        else:
            print("Direction is right. Two traces are an anecdote; repeat on "
                  "a dozen of each.")


if __name__ == "__main__":
    main()
