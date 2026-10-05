#!/usr/bin/env python3
"""
Prompt subject classifier: which kind of question is this?

Used in two places:
  * testqa.py reports results per subject, so "the edit helped" can be read
    as "helped on factual recall, hurt on code" instead of one blended number;
  * routing (later): pick the model whose measured strengths match the
    subject of an incoming prompt. `route` already works against a small
    JSON routing table so the idea can be tried today.

Deliberately dependency-free: a multinomial naive Bayes over word and
character n-grams with keyword priors, trained in milliseconds on the
labelled bank in qa/bank plus qa/subject_seed.jsonl. It is a router hint, not
an oracle; `evaluate` prints leave-one-out accuracy so you know how far to
trust it, and you can add labelled lines to the seed file to sharpen it.

    python scripts/subject_classifier.py train --out models/subject.json
    python scripts/subject_classifier.py predict "Write a function that ..."
    python scripts/subject_classifier.py evaluate
    python scripts/subject_classifier.py route --table qa/routing.example.json "prompt"
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "qa" / "bank"
SEED = ROOT / "qa" / "subject_seed.jsonl"
SUBJECTS = ["code", "math", "logic", "science", "factual", "writing", "vision"]

# Strong lexical cues. They act as a prior bump, not a hard rule, so a
# "write a poem about Python" prompt can still land in writing.
KEYWORDS = {
    "code": ["python", "function", "def ", "class ", "bug", "compile", "javascript", "sql", "regex", "api",
             "```", "stack trace", "git ", "http", "json", "algorithm", "refactor", "unit test", "rust", "c++"],
    "math": ["solve", "equation", "probability", "integral", "derivative", "sum of", "percent", "%",
             "how many", "calculate", "fraction", "prime", "average", "area", "angle"],
    "logic": ["puzzle", "riddle", "if all", "must", "conclude", "who is", "taller", "yes or no", "deduce",
              "brother", "sister", "day of the week"],
    "science": ["atom", "molecule", "cell", "planet", "energy", "physics", "chemistry", "biology", "light",
                "photosynthesis", "dna", "gravity", "evolution", "element"],
    "factual": ["capital", "who wrote", "who painted", "what year", "in which", "country", "president",
                "invented", "founded", "history", "born"],
    "writing": ["write a", "poem", "haiku", "essay", "email", "summarise", "summarize", "story", "rewrite",
                "draft", "tone", "letter", "recipe"],
    "vision": ["image", "photo", "picture", "diagram", "screenshot", "what is shown", "describe this",
               "in this figure", "ocr"],
}

TOKEN_RE = re.compile(r"[a-z0-9_+#]+|[^\sa-z0-9]")
STOP = set("a an the of to in is are be this that what which how do does i my me you your for and "
           "or with on it as by at from".split())


def features(text: str) -> list[str]:
    # Unigrams only: on a bank this small, bigrams and character n-grams
    # overfit (leave-one-out accuracy drops), and the keyword prior does
    # more work than either.
    return [f"w:{w}" for w in TOKEN_RE.findall(text.lower()) if w not in STOP]


def load_labelled(paths: list[Path] | None = None) -> list[tuple[str, str]]:
    rows = []
    for p in paths or [*sorted(BANK.glob("*.jsonl")), SEED]:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("subject") and r.get("prompt"):
                    rows.append((r["prompt"], r["subject"]))
    return rows


class SubjectClassifier:
    def __init__(self, alpha: float = 0.5, keyword_weight: float = 3.0):
        self.alpha = alpha
        self.keyword_weight = keyword_weight
        self.counts: dict[str, Counter] = {}
        self.totals: dict[str, int] = {}
        self.docs: Counter = Counter()
        self.vocab: set[str] = set()

    def fit(self, rows: list[tuple[str, str]]) -> "SubjectClassifier":
        self.counts = defaultdict(Counter)
        self.docs = Counter()
        for text, label in rows:
            f = features(text)
            self.counts[label].update(f)
            self.docs[label] += 1
            self.vocab.update(f)
        self.counts = dict(self.counts)
        self.totals = {k: sum(v.values()) for k, v in self.counts.items()}
        return self

    def scores(self, text: str) -> dict[str, float]:
        f = features(text)
        low = text.lower()
        n_docs = sum(self.docs.values()) or 1
        v = len(self.vocab) + 1
        out = {}
        for label in self.counts:
            s = math.log((self.docs[label] + 1) / (n_docs + len(self.counts)))
            c, tot = self.counts[label], self.totals[label]
            for x in f:
                s += math.log((c.get(x, 0) + self.alpha) / (tot + self.alpha * v))
            s += self.keyword_weight * sum(1 for k in KEYWORDS.get(label, []) if k in low)
            out[label] = s
        return out

    def predict_proba(self, text: str) -> dict[str, float]:
        s = self.scores(text)
        if not s:
            return {}
        m = max(s.values())
        e = {k: math.exp(v - m) for k, v in s.items()}
        z = sum(e.values())
        return dict(sorted(((k, v / z) for k, v in e.items()), key=lambda kv: -kv[1]))

    def predict(self, text: str) -> str:
        p = self.predict_proba(text)
        return next(iter(p)) if p else "unknown"

    def to_json(self) -> dict:
        return {"alpha": self.alpha, "keyword_weight": self.keyword_weight,
                "counts": {k: dict(v) for k, v in self.counts.items()}, "docs": dict(self.docs),
                "vocab_size": len(self.vocab)}

    @classmethod
    def from_json(cls, d: dict) -> "SubjectClassifier":
        c = cls(d["alpha"], d["keyword_weight"])
        c.counts = {k: Counter(v) for k, v in d["counts"].items()}
        c.totals = {k: sum(v.values()) for k, v in c.counts.items()}
        c.docs = Counter(d["docs"])
        c.vocab = set().union(*[set(v) for v in c.counts.values()]) if c.counts else set()
        return c


def default_classifier(model_path: str | None = None) -> SubjectClassifier:
    if model_path and Path(model_path).exists():
        return SubjectClassifier.from_json(json.loads(Path(model_path).read_text()))
    return SubjectClassifier().fit(load_labelled())


def leave_one_out(rows) -> tuple[float, dict]:
    hits, confusion = 0, defaultdict(Counter)
    for i, (text, label) in enumerate(rows):
        clf = SubjectClassifier().fit(rows[:i] + rows[i + 1:])
        pred = clf.predict(text)
        hits += pred == label
        confusion[label][pred] += 1
    return hits / max(1, len(rows)), {k: dict(v) for k, v in confusion.items()}


def route(text: str, table: dict, clf: SubjectClassifier) -> dict:
    """Pick a model for a prompt from {"models": {name: {"skills": {subject: score}}}, "default": name}."""
    proba = clf.predict_proba(text)
    best, best_score = table.get("default"), -1.0
    for name, spec in table.get("models", {}).items():
        skills = spec.get("skills", {})
        score = sum(p * float(skills.get(s, 0.0)) for s, p in proba.items())
        if score > best_score:
            best, best_score = name, score
    return {"subject": next(iter(proba), "unknown"), "proba": proba, "model": best, "fit": best_score}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--data", nargs="*", help="labelled JSONL files (default: qa/bank + seed)")
    pr = sub.add_parser("predict")
    pr.add_argument("text")
    pr.add_argument("--model")
    sub.add_parser("evaluate")
    r = sub.add_parser("route")
    r.add_argument("text")
    r.add_argument("--table", required=True)
    r.add_argument("--model")
    a = p.parse_args(argv)

    if a.cmd == "train":
        rows = load_labelled([Path(x) for x in a.data] if a.data else None)
        clf = SubjectClassifier().fit(rows)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(clf.to_json()))
        print(f"trained on {len(rows)} prompts, {len(clf.counts)} subjects -> {a.out}")
    elif a.cmd == "predict":
        print(json.dumps(default_classifier(a.model).predict_proba(a.text), indent=2))
    elif a.cmd == "evaluate":
        rows = load_labelled()
        acc, conf = leave_one_out(rows)
        print(f"leave-one-out accuracy on {len(rows)} prompts: {acc:.3f}")
        for label, row in sorted(conf.items()):
            print(f"  {label:<8} " + ", ".join(f"{k}={v}" for k, v in sorted(row.items(), key=lambda kv: -kv[1])))
    elif a.cmd == "route":
        print(json.dumps(route(a.text, json.loads(Path(a.table).read_text()), default_classifier(a.model)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
