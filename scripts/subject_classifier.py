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
# Subjects with a graded question bank (qa/bank), which testqa.py can score.
QA_SUBJECTS = ["code", "math", "logic", "science", "factual", "writing", "vision"]
# Kinds of work that show up when a project is split into tasks. They have
# labelled seed lines but no graded bank yet, so routing uses a model's overall
# numbers for them (model_stats.rank says so when it does).
TASK_SUBJECTS = ["graphics", "systems", "reverse-engineering"]
SUBJECTS = QA_SUBJECTS + TASK_SUBJECTS
UNKNOWN = "unknown"

# analyze(): a label is returned when its share of the probability reaches CUTOFF, and
# nothing is returned (subject unknown) when the text has fewer than
# MIN_EVIDENCE words the classifier has seen often enough to mean something.
# Chosen by 10-fold cross-validation on the labelled lines (top label right
# 79% of the time, 4% unknown) and checked on held-out task descriptions.
CUTOFF = 0.3
MIN_EVIDENCE = 2
TEMPERATURE = 0.5
KEYWORD_CAP = 1

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
    # Not "write a": tasks say "write a shader" or "write a parser" as often as "write a letter".
    "writing": ["poem", "haiku", "essay", "email", "summarise", "summarize", "story", "rewrite",
                "draft", "tone", "letter", "recipe"],
    "vision": ["image", "photo", "picture", "diagram", "screenshot", "what is shown", "describe this",
               "in this figure", "ocr"],
    "graphics": ["shader", "glsl", "hlsl", "vulkan", "opengl", "webgl", "three.js", "mesh", "texture",
                 "render", "voxel", "vertex", "fragment", "gpu", "lighting", "uv "],
    "systems": ["build system", "cmake", "makefile", "meson", "bazel", "linker", "toolchain",
                "dockerfile", "docker", "ci ", "cross-compile", "package", "systemd", "dependencies"],
    "reverse-engineering": ["decompil", "disassembl", "ghidra", "reverse engineer", "reverse-engineer",
                            "firmware", "rom", "hex dump", "assembly", "memory address", "binary",
                            "file format", "debugger"],
}

def keyword_hits(text: str, label: str) -> int:
    """Keywords match at the start of a word ("shader" matches "shaders",
    "rom" does not match "from"); ones that start with punctuation match anywhere."""
    low = text.lower()
    n = 0
    for k in KEYWORDS.get(label, []):
        if k[0].isalnum():
            n += bool(re.search(r"(?<![a-z0-9])" + re.escape(k), low))
        else:
            n += k in low
    return n


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
            # Capped: a keyword says the subject is involved; three of them do
            # not make it three times more involved than a subject with one.
            s += self.keyword_weight * min(KEYWORD_CAP, keyword_hits(text, label))
            out[label] = s
        return out

    def evidence(self, text: str) -> int:
        """Words that carry information: seen at least twice in training, plus
        keyword hits. Text made of words the classifier has never seen has
        nothing to classify on, whatever the probabilities say."""
        seen = {x for x in features(text) if sum(c.get(x, 0) for c in self.counts.values()) >= 2}
        return len(seen) + sum(keyword_hits(text, lb) for lb in self.counts)

    def label_proba(self, text: str) -> dict[str, float]:
        """predict_proba softened for length: the naive-Bayes log scores are
        divided by TEMPERATURE * sqrt(informative words) before the softmax.
        Plain naive Bayes is near-certain of one subject after a dozen words;
        softened, a task that is half graphics and half build work shows both."""
        s = self.scores(text)
        if not s:
            return {}
        n = len([x for x in features(text) if any(x in c for c in self.counts.values())])
        t = TEMPERATURE * math.sqrt(max(1, n))
        m = max(s.values())
        e = {k: math.exp((v - m) / t) for k, v in s.items()}
        z = sum(e.values())
        return dict(sorted(((k, v / z) for k, v in e.items()), key=lambda kv: -kv[1]))

    def analyze(self, text: str, cutoff: float = CUTOFF, min_evidence: int = MIN_EVIDENCE) -> dict:
        """-> {labels: [subjects whose own probability >= cutoff, strongest
        first], proba: per-subject probabilities, evidence, unknown, why}.
        `unknown` is True when the evidence is thin or no subject reaches the
        cutoff; callers should then not route on subject at all."""
        ev = self.evidence(text)
        proba = self.label_proba(text) if self.counts else {}
        labels = [k for k, p in proba.items() if p >= cutoff]
        if ev < min_evidence:
            why = f"only {ev} informative word(s) (need {min_evidence})"
            labels = []
        elif not labels:
            top = next(iter(proba), None)
            why = f"no subject reaches {cutoff:.0%}" + (f" (closest: {top} {proba[top]:.0%})" if top else "")
        else:
            why = ", ".join(f"{k} {proba[k]:.0%}" for k in labels)
        return {"labels": labels, "proba": {k: round(v, 4) for k, v in proba.items()},
                "evidence": ev, "unknown": not labels, "why": why}

    def route_weights(self, text: str) -> dict[str, float]:
        """Subject weights for routing: the returned labels, normalised. Empty
        when the subject is unknown, meaning "use the best model overall"."""
        a = self.analyze(text)
        tot = sum(a["proba"][k] for k in a["labels"])
        return {k: a["proba"][k] / tot for k in a["labels"]} if tot else {}

    def predict_proba(self, text: str) -> dict[str, float]:
        s = self.scores(text)
        if not s:
            return {}
        m = max(s.values())
        e = {k: math.exp(v - m) for k, v in s.items()}
        z = sum(e.values())
        return dict(sorted(((k, v / z) for k, v in e.items()), key=lambda kv: -kv[1]))

    def predict(self, text: str) -> str:
        """The strongest subject, or "unknown" when the evidence is thin or no
        subject is likely enough (see analyze)."""
        a = self.analyze(text)
        return a["labels"][0] if a["labels"] else UNKNOWN

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


def score_tasks(clf: SubjectClassifier, items: list[dict]) -> dict:
    """Multi-label check. An item passes when every label in `need` (or one of
    `need_any`) is returned and nothing outside need/need_any/ok is; an item
    with unknown=true passes only when the subject comes back unknown."""
    rows = []
    for it in items:
        a = clf.analyze(it["text"])
        got = a["labels"]
        if it.get("unknown"):
            ok = a["unknown"]
        else:
            allowed = set(it.get("need", [])) | set(it.get("need_any", [])) | set(it.get("ok", []))
            ok = (all(x in got for x in it.get("need", []))
                  and (not it.get("need_any") or any(x in got for x in it["need_any"]))
                  and set(got) <= allowed and bool(got))
        rows.append({"id": it.get("id", "?"), "pass": ok, "got": got or [UNKNOWN], "why": a["why"]})
    return {"n": len(rows), "passed": sum(r["pass"] for r in rows), "rows": rows}


def route(text: str, table: dict, clf: SubjectClassifier) -> dict:
    """Pick a model for a prompt from {"models": {name: {"skills": {subject: score}}}, "default": name}.
    An unknown subject goes to the table's default rather than to a guess."""
    a = clf.analyze(text)
    weights = clf.route_weights(text)
    if not weights:
        return {"subject": UNKNOWN, "labels": [], "why": a["why"], "model": table.get("default"), "fit": None}
    best, best_score = table.get("default"), -1.0
    for name, spec in table.get("models", {}).items():
        skills = spec.get("skills", {})
        score = sum(p * float(skills.get(s, 0.0)) for s, p in weights.items())
        if score > best_score:
            best, best_score = name, score
    return {"subject": a["labels"][0], "labels": a["labels"], "why": a["why"], "model": best, "fit": best_score}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--data", nargs="*", help="labelled JSONL files (default: qa/bank + seed)")
    pr = sub.add_parser("predict")
    pr.add_argument("text")
    pr.add_argument("--model")
    pr.add_argument("--cutoff", type=float, default=CUTOFF)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--tasks", nargs="*", help="also score labelled task files: JSONL with text, need/ok "
                                              "label lists or unknown=true")
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
        print(json.dumps(default_classifier(a.model).analyze(a.text, a.cutoff), indent=2))
    elif a.cmd == "evaluate":
        rows = load_labelled()
        acc, conf = leave_one_out(rows)
        unk = sum(r.get(UNKNOWN, 0) for r in conf.values())
        print(f"leave-one-out on {len(rows)} prompts: {acc:.3f} correct, {unk / max(1, len(rows)):.3f} unknown")
        for label, row in sorted(conf.items()):
            print(f"  {label:<20} " + ", ".join(f"{k}={v}" for k, v in sorted(row.items(), key=lambda kv: -kv[1])))
        if a.tasks:
            clf = default_classifier()
            for path in a.tasks:
                res = score_tasks(clf, [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()])
                print(f"\n{path}: {res['passed']}/{res['n']} pass")
                for r in res["rows"]:
                    print(f"  {'ok  ' if r['pass'] else 'FAIL'} {r['id']:<10} {r['got']}  ({r['why']})")
    elif a.cmd == "route":
        print(json.dumps(route(a.text, json.loads(Path(a.table).read_text()), default_classifier(a.model)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
