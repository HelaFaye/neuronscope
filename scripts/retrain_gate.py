#!/usr/bin/env python3
"""
Keep or reject a retrained model: the before/after check on items it never
trained on.

    python scripts/testqa.py --endpoint before=...@my-model --endpoint after=...@my-model-retrained \\
        --allow-exec --out runs/after.json
    python scripts/retrain_gate.py --results runs/after.json --before before --after after \\
        --holdout runs/retrain/holdout_ids.json --out runs/gate.json

Two results files work too: `--before-results runs/base.json --results runs/after.json`.

The same items are compared before and after:

- **holdout** (never trained on, not even through variations) decides;
- **trained** (the rest) is reported, to see memorisation: a big gain there
  with none on the holdout means the model learned the test, not the skill.

For each split and each subject:
- items fixed and items broken;
- an exact McNemar test on those changes;
- wrong-answer (hallucination) and abstention rates, before and after.

The verdict:
  keep          the holdout improved significantly, hallucination did not rise
                significantly, and no subject regressed significantly
  reject        the holdout got significantly worse, hallucination rose
                significantly, or some subject regressed significantly
  inconclusive  anything else; the report says roughly how many items it
                would take to tell

Exit status is 0 unless --strict, which returns 3 for inconclusive and 4 for
reject (for scripts that chain retraining runs).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

OK = ("correct", "answered")
SKIP = ("error", "skipped")


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided McNemar p for k of n discordant changes going one way."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(path: str, label: str | None) -> tuple[str, dict]:
    d = json.loads(Path(path).read_text())
    raw = d["raw"]
    label = label or d.get("reference") or next(iter(raw))
    if label not in raw:
        raise SystemExit(f"{path}: no endpoint {label!r} (have: {', '.join(raw)})")
    return label, {r["id"]: r for r in raw[label]}


def compare(before: dict, after: dict, ids) -> dict:
    """Paired comparison over `ids` (graded in both runs, canaries apart)."""
    n = fixed = broke = 0
    w_before = w_after = a_before = a_after = ok_before = ok_after = 0
    new_wrong = gone_wrong = 0
    canary = {"n": 0, "refused_before": 0, "refused_after": 0}
    examples = {"fixed": [], "broke": []}
    for i in ids:
        b, a = before.get(i), after.get(i)
        if b is None or a is None or b["verdict"] in SKIP or a["verdict"] in SKIP:
            continue
        if b.get("kind") == "canary":
            canary["n"] += 1
            canary["refused_before"] += b["verdict"] == "refused"
            canary["refused_after"] += a["verdict"] == "refused"
            continue
        n += 1
        bo, ao = b["verdict"] in OK, a["verdict"] in OK
        ok_before += bo
        ok_after += ao
        if ao and not bo:
            fixed += 1
            examples["fixed"].append(i)
        elif bo and not ao:
            broke += 1
            examples["broke"].append(i)
        bw, aw = b["verdict"] == "wrong", a["verdict"] == "wrong"
        w_before += bw
        w_after += aw
        new_wrong += aw and not bw
        gone_wrong += bw and not aw
        a_before += b["verdict"] == "abstained"
        a_after += a["verdict"] == "abstained"
    rate = (lambda x: round(x / n, 4)) if n else (lambda x: None)
    return {"n": n, "fixed": fixed, "broke": broke, "net": fixed - broke,
            "p": round(binom_two_sided(fixed, fixed + broke), 5),
            "accuracy_before": rate(ok_before), "accuracy_after": rate(ok_after),
            "wrong_before": rate(w_before), "wrong_after": rate(w_after),
            "abstained_before": rate(a_before), "abstained_after": rate(a_after),
            "newly_wrong": new_wrong, "no_longer_wrong": gone_wrong,
            "p_wrong": round(binom_two_sided(new_wrong, new_wrong + gone_wrong), 5),
            "canary": canary, "examples": {k: v[:20] for k, v in examples.items()}}


def items_needed(c: dict, alpha: float) -> int | None:
    """Roughly how many paired items it takes to detect a change of this size: the
    discordant count where this split of fixed and broken reaches p < alpha,
    scaled by the share of items that changed."""
    d = c["fixed"] + c["broke"]
    if d == 0 or c["fixed"] == c["broke"] or not c["n"]:
        return None
    share = max(c["fixed"], c["broke"]) / d
    for m in range(d, 20000):
        if binom_two_sided(round(m * (1 - share)), m) < alpha:
            return math.ceil(m * c["n"] / d)
    return None


def gate(before: dict, after: dict, holdout: set, alpha: float = 0.05) -> dict:
    common = set(before) & set(after)
    hold = sorted(common & holdout) if holdout else sorted(common)
    trained = sorted(common - holdout) if holdout else []
    tasks_of = lambda ids: {i: (before[i].get("subject") or "?") for i in ids}  # noqa: E731
    out = {"alpha": alpha, "holdout": compare(before, after, hold), "trained": compare(before, after, trained),
           "subjects": {}}
    by_subj = defaultdict(list)
    for i, s in tasks_of(hold).items():
        by_subj[s].append(i)
    for s, ids in sorted(by_subj.items()):
        out["subjects"][s] = compare(before, after, ids)
    h = out["holdout"]
    reasons = []
    regressed = [s for s, c in out["subjects"].items() if c["broke"] > c["fixed"] and c["p"] < alpha]
    halluc_up = h["newly_wrong"] > h["no_longer_wrong"] and h["p_wrong"] < alpha
    if h["n"] == 0:
        verdict = "inconclusive"
        reasons.append("no holdout items were graded in both runs (pass --holdout and run TestQA on them)")
    elif h["broke"] > h["fixed"] and h["p"] < alpha:
        verdict = "reject"
        reasons.append(f"the holdout got worse: {h['broke']} broken vs {h['fixed']} fixed (p={h['p']})")
    elif halluc_up:
        verdict = "reject"
        reasons.append(f"hallucination rose on the holdout: {h['newly_wrong']} newly wrong vs "
                       f"{h['no_longer_wrong']} no longer wrong (p={h['p_wrong']})")
    elif regressed:
        verdict = "reject"
        reasons.append("significant regression in " + ", ".join(regressed))
    elif h["fixed"] > h["broke"] and h["p"] < alpha:
        verdict = "keep"
        reasons.append(f"the holdout improved: {h['fixed']} fixed vs {h['broke']} broken (p={h['p']})")
    else:
        verdict = "inconclusive"
        need = items_needed(h, alpha)
        reasons.append(f"{h['fixed']} fixed vs {h['broke']} broken on {h['n']} holdout items (p={h['p']})"
                       + (f"; a change this size needs about {need} holdout items to tell" if need else ""))
    t = out["trained"]
    if t["n"] and h["n"]:
        tg, hg = t["net"] / t["n"], h["net"] / h["n"]
        if tg > 0.05 and (hg <= 0 or tg >= 3 * hg) and not (verdict == "keep"):
            reasons.append(f"memorisation: trained items gained {tg:+.0%}, the holdout {hg:+.0%}; the model learned "
                           "the training items more than the skill")
            out["memorisation"] = True
    hc = h["canary"]
    if hc["n"] and hc["refused_after"] > hc["refused_before"]:
        reasons.append(f"canaries refused: {hc['refused_before']} -> {hc['refused_after']} of {hc['n']}")
    out.update(verdict=verdict, reasons=reasons)
    return out


def report(g: dict, before: str, after: str) -> str:
    def row(name, c):
        if not c["n"]:
            return f"{name:<14} {'-':>5}"
        pct = lambda x: "  -  " if x is None else f"{x:5.0%}"  # noqa: E731
        return (f"{name:<14} {c['n']:>5}  {pct(c['accuracy_before'])} -> {pct(c['accuracy_after'])}  "
                f"fixed {c['fixed']:>3}  broke {c['broke']:>3}  p={c['p']:<7.3g} "
                f"wrong {pct(c['wrong_before'])} -> {pct(c['wrong_after'])}  "
                f"abstain {pct(c['abstained_before'])} -> {pct(c['abstained_after'])}")
    lines = [f"{after} vs {before}", "", f"{'':<14} {'n':>5}  accuracy         changes", row("holdout", g["holdout"]),
             row("trained", g["trained"]), "", "holdout by subject:"]
    lines += [row("  " + s, c) for s, c in g["subjects"].items()]
    lines += ["", f"VERDICT: {g['verdict'].upper()}"] + [f"  - {r}" for r in g["reasons"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", required=True, help="testqa.py --out JSON with the retrained model")
    p.add_argument("--after", help="its endpoint label (default: the file's reference)")
    p.add_argument("--before-results", help="a separate results JSON for the original model (default: --results)")
    p.add_argument("--before", help="the original model's endpoint label")
    p.add_argument("--holdout", help="deficits.py holdout_ids.json (default: every item counts as holdout)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", help="write the full comparison as JSON")
    p.add_argument("--strict", action="store_true", help="exit 3 when inconclusive, 4 when rejected")
    a = p.parse_args(argv)
    bl, before = load(a.before_results or a.results, a.before)
    al, after = load(a.results, a.after)
    if not a.before_results and bl == al:
        raise SystemExit("--before and --after name the same endpoint; give both labels, or --before-results")
    holdout = set(json.loads(Path(a.holdout).read_text())["ids"]) if a.holdout else set()
    g = gate(before, after, holdout, a.alpha)
    g.update(before=bl, after=al)
    print(report(g, bl, al))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(g, indent=1))
    if a.strict:
        return {"keep": 0, "inconclusive": 3, "reject": 4}[g["verdict"]]
    return 0


if __name__ == "__main__":
    sys.exit(main())
