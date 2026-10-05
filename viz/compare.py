#!/usr/bin/env python3
"""
Compare activations across sessions, with the comparability question enforced
rather than left to the user.

The naive version of this is meaningless. Layer 5 neuron 1234 in one model has no
relationship to layer 5 neuron 1234 in Qwen3.5: independently trained networks
have no neuron correspondence, so differencing their maps yields noise that
looks convincingly like structure. This module sorts comparisons into tiers and
refuses index-level operations outside the ones where indices actually mean the
same thing.

    IDENTICAL      same fingerprint, same quant
                   -> everything. This is a repeat measurement.

    REQUANTIZED    same fingerprint, different quant
                   -> index-level comparison is valid: same weights, same
                      neurons, perturbed values. The interesting question here
                      is whether Q4 recruits different neurons than Q6, which
                      is the mechanistic version of "low quants hallucinate
                      more" and is testable rather than folklore.

    LINEAGE        same geometry, different weights, and you assert one was
                   post-trained from the other (a fine-tune from its base checkpoint,
                   say). Post-training does not permute neurons, so indices
                   plausibly still correspond -- but that is a hypothesis, not
                   a guarantee, so it must be asserted with --assert-lineage
                   and the correlation reported alongside is how you check it.
                   A near-zero correlation means the assertion was wrong.

    UNRELATED      different geometry or unasserted different weights
                   -> distribution and behaviour only. No index arithmetic.

Distribution-level comparisons (depth profiles, sparsity, concentration) and
behavioural ones (which items each model got wrong) are valid across every
tier, because neither depends on indices lining up.

    python viz/compare.py runs/model-q6 runs/model-q4
    python viz/compare.py runs/base runs/finetune --assert-lineage
    python viz/compare.py a b c --json out.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from records import Session  # noqa: E402

IDENTICAL, REQUANTIZED, LINEAGE, UNRELATED = (
    "IDENTICAL", "REQUANTIZED", "LINEAGE", "UNRELATED")


def comparability(a, b, assert_lineage=False):
    """-> (tier, reason). Geometry is the hard gate; the rest is provenance."""
    ga = (a.get("n_layers"), a.get("n_neurons"))
    gb = (b.get("n_layers"), b.get("n_neurons"))
    if ga != gb or None in ga:
        return UNRELATED, f"geometry differs: {ga} vs {gb}"
    fa, fb = a.get("fingerprint"), b.get("fingerprint")
    qa, qb = a.get("quant"), b.get("quant")
    if fa and fa == fb:
        if qa == qb:
            return IDENTICAL, f"same model and quant ({qa})"
        return REQUANTIZED, f"same weights, {qa} vs {qb}"
    if assert_lineage:
        return LINEAGE, ("same geometry, lineage asserted by the user "
                         "(unverified)")
    return UNRELATED, ("same geometry but different weights; pass "
                       "--assert-lineage only if one was post-trained from "
                       "the other")


def depth_profile(mean_map, bins=10):
    """Per-layer mean, resampled onto a fixed number of relative-depth bins.

    Resampling is what makes this comparable across models with different layer
    counts: a 32-layer and a 48-layer model both land on 10 depth fractions.
    """
    prof = mean_map.mean(axis=1)
    x = np.linspace(0, 1, len(prof))
    out = np.interp(np.linspace(0, 1, bins), x, prof)
    # Normalize AFTER resampling. Interpolating onto a different bin count does
    # not preserve the sum, so normalizing first would leave a 32-layer and a
    # 48-layer profile on different scales -- exactly the comparison this
    # function exists to make valid.
    total = out.sum()
    return out / total if total > 0 else out


def concentration(counts):
    """Gini over per-layer counts. 0 = spread evenly across layers, ->1 =
    concentrated in a few. Comparable across models; a raw count is not."""
    c = np.sort(np.asarray(counts, dtype=float))
    n = len(c)
    if n == 0 or c.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * c).sum()) / (n * c.sum()) - (n + 1) / n)


def summarize(session, max_samples=300):
    stack, ids = session.stack(max_samples=max_samples)
    if stack.size == 0:
        raise SystemExit(f"{session.path} has no samples with an 'agg' array")
    mean_map = stack.mean(0)
    per_layer = mean_map.mean(axis=1)
    verdicts = {}
    for qid in ids:
        f = (session.get(qid).get("fields") or {})
        if f.get("verdict"):
            verdicts[qid] = f["verdict"]
    return {
        "path": session.path,
        "meta": session.meta,
        "n_samples": len(ids),
        "mean_map": mean_map,
        "per_layer": per_layer,
        "depth_profile": depth_profile(mean_map),
        "concentration": concentration(per_layer),
        "sparsity_p99": float(np.percentile(mean_map, 99)),
        "verdicts": verdicts,
    }


def behavioural_overlap(a, b):
    """Which items each model got wrong, on the items both were asked.

    Never depends on indices, so it is valid at every tier -- and for two
    different models it is usually the only thing worth computing.
    """
    shared = set(a["verdicts"]) & set(b["verdicts"])
    if not shared:
        return None
    wa = {q for q in shared if a["verdicts"][q] == "wrong"}
    wb = {q for q in shared if b["verdicts"][q] == "wrong"}
    union = wa | wb
    return {
        "shared_items": len(shared),
        "wrong_a": len(wa), "wrong_b": len(wb),
        "both_wrong": len(wa & wb),
        "only_a": len(wa - wb), "only_b": len(wb - wa),
        "jaccard": (len(wa & wb) / len(union)) if union else 0.0,
    }


def index_level(a, b):
    """Only called when the tier permits it."""
    fa, fb = a["mean_map"].ravel(), b["mean_map"].ravel()
    corr = float(np.corrcoef(fa, fb)[0, 1])
    diff = b["mean_map"] - a["mean_map"]
    k = min(10, diff.size)
    flat = np.argsort(np.abs(diff), axis=None)[-k:][::-1]
    rows, cols = np.unravel_index(flat, diff.shape)
    return {
        "correlation": corr,
        "top_shifts": [{"layer": int(r), "neuron": int(c),
                        "delta": float(diff[r, c])}
                       for r, c in zip(rows, cols)],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sessions", nargs="+")
    p.add_argument("--assert-lineage", action="store_true",
                   help="Assert same-geometry sessions share neuron identity "
                        "(one post-trained from the other). Unverified.")
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--json", help="write the full comparison here")
    a = p.parse_args()

    if len(a.sessions) < 2:
        raise SystemExit("need at least two sessions")

    S = [summarize(Session(p_), a.max_samples) for p_ in a.sessions]
    for s in S:
        m = s["meta"]
        print(f"{os.path.basename(s['path']):<24} {m.get('model','?')} "
              f"{m.get('quant','?')}  {s['n_samples']} samples  "
              f"{m.get('n_layers')}x{m.get('n_neurons')}")

    print("\ndistribution (valid across every tier):")
    print(f"  {'session':<24} {'concentration':>13} {'p99 activation':>15}")
    for s in S:
        print(f"  {os.path.basename(s['path']):<24} "
              f"{s['concentration']:>13.3f} {s['sparsity_p99']:>15.5f}")

    print("\n  depth profile (activation mass by relative depth, 10 bins):")
    for s in S:
        bars = " ".join(f"{v:.3f}" for v in s["depth_profile"])
        print(f"  {os.path.basename(s['path']):<24} {bars}")

    out = {"sessions": [{"path": s["path"], "meta": s["meta"],
                         "concentration": s["concentration"],
                         "depth_profile": s["depth_profile"].tolist()}
                        for s in S], "pairs": []}

    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            x, y = S[i], S[j]
            tier, reason = comparability(x["meta"], y["meta"], a.assert_lineage)
            name = (f"{os.path.basename(x['path'])} vs "
                    f"{os.path.basename(y['path'])}")
            print(f"\n{name}\n  tier: {tier} -- {reason}")

            rec = {"pair": name, "tier": tier, "reason": reason}

            cos = float(np.dot(x["depth_profile"], y["depth_profile"]) /
                        ((np.linalg.norm(x["depth_profile"]) *
                          np.linalg.norm(y["depth_profile"])) or 1))
            rec["depth_profile_similarity"] = cos
            print(f"  depth profile similarity: {cos:.3f}")

            bo = behavioural_overlap(x, y)
            if bo:
                rec["behaviour"] = bo
                print(f"  behaviour: {bo['wrong_a']} vs {bo['wrong_b']} wrong "
                      f"of {bo['shared_items']} shared; {bo['both_wrong']} both, "
                      f"Jaccard {bo['jaccard']:.3f}")
            else:
                print("  behaviour: no shared items with recorded verdicts")

            if tier == UNRELATED:
                print("  index-level comparison skipped: neuron indices do not "
                      "correspond.\n  Differencing these maps would produce "
                      "noise that looks like structure.")
            else:
                il = index_level(x, y)
                rec["index_level"] = il
                print(f"  map correlation: {il['correlation']:.4f}")
                if tier == LINEAGE and il["correlation"] < 0.3:
                    print("  !! low correlation for an asserted lineage. The "
                          "assertion is probably\n     wrong, and the shifts "
                          "below are meaningless if so.")
                print("  largest shifts:")
                for t in il["top_shifts"][:5]:
                    print(f"    L{t['layer']:<3} n{t['neuron']:<6} "
                          f"{t['delta']:+.5f}")
            out["pairs"].append(rec)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
