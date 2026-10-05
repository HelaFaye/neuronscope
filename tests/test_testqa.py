import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import subject_classifier as sc  # noqa: E402
import testqa as tq  # noqa: E402

REFERENCE = {
    "is_palindrome": "def is_palindrome(s):\n    t=[c.lower() for c in s if c.isalnum()]\n    return t==t[::-1]",
    "fizzbuzz": "def fizzbuzz(n):\n    return ['FizzBuzz' if i%15==0 else 'Fizz' if i%3==0 else 'Buzz' if i%5==0 else str(i) for i in range(1,n+1)]",
    "two_sum": "def two_sum(nums,target):\n    seen={}\n    for j,x in enumerate(nums):\n        if target-x in seen: return (seen[target-x],j)\n        seen[x]=j",
    "flatten": "def flatten(x):\n    out=[]\n    for e in x:\n        out.extend(flatten(e) if isinstance(e,list) else [e])\n    return out",
    "roman_to_int": "def roman_to_int(s):\n    v=dict(I=1,V=5,X=10,L=50,C=100,D=500,M=1000); t=0\n    for i,c in enumerate(s):\n        t+= -v[c] if i+1<len(s) and v[s[i+1]]>v[c] else v[c]\n    return t",
    "merge_intervals": "def merge_intervals(iv):\n    out=[]\n    for s,e in sorted(iv):\n        if out and s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)\n        else: out.append([s,e])\n    return out",
    "word_count": "import re\nfrom collections import Counter\ndef word_count(text):\n    return dict(Counter(re.findall(r\"[a-z0-9']+\", text.lower())))",
    "is_balanced": "def is_balanced(s):\n    st=[]; m={')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in '([{': st.append(c)\n        elif c in m:\n            if not st or st.pop()!=m[c]: return False\n    return not st",
    "fib": "def fib(n):\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a",
    "binary_search": "import bisect\ndef binary_search(arr,x):\n    i=bisect.bisect_left(arr,x)\n    return i if i<len(arr) and arr[i]==x else -1",
    "rotate": "def rotate(m):\n    return [list(r) for r in zip(*m[::-1])]",
    "lcs_length": "def lcs_length(a,b):\n    d=[[0]*(len(b)+1) for _ in range(len(a)+1)]\n    for i in range(len(a)):\n        for j in range(len(b)):\n            d[i+1][j+1]=d[i][j]+1 if a[i]==b[j] else max(d[i][j+1],d[i+1][j])\n    return d[-1][-1]",
    "parse_duration": "import re\ndef parse_duration(s):\n    m=re.fullmatch(r'(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?',s or '')\n    if not s or not m: raise ValueError(s)\n    h,mi,se=(int(x or 0) for x in m.groups())\n    return h*3600+mi*60+se",
    "chunk": "def chunk(xs,n):\n    if n<1: raise ValueError(n)\n    return [xs[i:i+n] for i in range(0,len(xs),n)]",
    "dedupe": "def dedupe(xs):\n    return list(dict.fromkeys(xs))",
    "caesar": "def caesar(s,k):\n    o=[]\n    for c in s:\n        if c.isascii() and c.isalpha():\n            b=ord('A') if c.isupper() else ord('a'); o.append(chr((ord(c)-b+k)%26+b))\n        else: o.append(c)\n    return ''.join(o)",
    "median": "def median(xs):\n    if not xs: raise ValueError('empty')\n    s=sorted(xs); n=len(s)\n    return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2",
    "top_k_frequent": "from collections import Counter\ndef top_k_frequent(words,k):\n    c=Counter(words)\n    return sorted(c,key=lambda w:(-c[w],w))[:k]",
}


def bank(kind):
    return [json.loads(l) for l in (ROOT / "qa" / "bank" / f"{kind}.jsonl").read_text().splitlines() if l.strip()]


def test_reference_solutions_pass_every_coding_task():
    tasks = bank("coding")
    assert {t["entry_point"] for t in tasks} == set(REFERENCE)
    for t in tasks:
        verdict, detail = tq.run_code(REFERENCE[t["entry_point"]], t["tests"])
        assert verdict == "correct", (t["id"], detail)


def test_interpreter_rejects_wrong_and_runaway_code():
    tests = ["assert f(2) == 4"]
    assert tq.run_code("def f(x): return x + 1", tests)[0] == "wrong"
    assert tq.run_code("def f(x):\n    while True: pass", tests, timeout=2)[0] == "timeout"
    bomb = "def f(x):\n    a = bytearray(8 * 1024**3)\n    return 4"
    assert tq.run_code(bomb, tests, mem_mb=256)[0] in ("wrong", "timeout")
    # The child gets a stripped environment.
    leak = "import os\ndef f(x): return 4 if 'SECRET_TESTQA' not in os.environ else 0"
    import os
    os.environ["SECRET_TESTQA"] = "1"
    try:
        assert tq.run_code(leak, tests)[0] == "correct"
    finally:
        del os.environ["SECRET_TESTQA"]


def test_code_exec_requires_opt_in():
    t = bank("coding")[0]
    reply = "```python\n" + REFERENCE[t["entry_point"]] + "\n```"
    assert tq.grade_code_exec(reply, t, allow_exec=False, timeout=5)[0] == "skipped"
    assert tq.grade_code_exec(reply, t, allow_exec=True, timeout=5)[0] == "correct"
    assert tq.grade_code_exec("I can't help with that.", t, True, 5)[0] == "abstained"
    assert tq.grade_code_exec("```python\ndef broken(:\n```", t, True, 5)[0] == "unparsable"


@pytest.mark.parametrize("reply,task,verdict", [
    ("blah\nAnswer: 72", {"answer": "72", "answer_type": "number"}, "correct"),
    ("Answer: $2.50", {"answer": "2.5", "answer_type": "number"}, "correct"),
    ("so the answer is: 3/28", {"answer": "3/28", "answer_type": "number"}, "correct"),
    ("Answer: \\frac{2}{3}", {"answer": "2/3", "answer_type": "number"}, "correct"),
    ("Answer: 1,000", {"answer": "1000", "answer_type": "number"}, "correct"),
    ("<think>maybe 41</think>Answer: 42", {"answer": "42", "answer_type": "number"}, "correct"),
    ("Answer: 41", {"answer": "42", "answer_type": "number"}, "wrong"),
    ("Answer: (A)", {"answer": "A", "answer_type": "choice"}, "correct"),
    ("Answer: B", {"answer": "A", "answer_type": "choice"}, "wrong"),
    ("Answer: Friday.", {"answer": "Friday", "answer_type": "text"}, "correct"),
    ("Answer: No, we cannot.", {"answer": "no", "answer_type": "text"}, "correct"),
    ("Answer: Thursday", {"answer": "Friday", "answer_type": "text"}, "wrong"),
    ("", {"answer": "1", "answer_type": "number"}, "abstained"),
])
def test_reasoning_grader(reply, task, verdict):
    assert tq.grade_reasoning(reply, task) == verdict


def test_expect_abstain_qa():
    t = {"aliases": ["i don't know", "does not exist"], "expect_abstain": True}
    assert tq.grade_qa("I don't know.", t) == "correct"
    assert tq.grade_qa("Freloniaville", t) == "wrong"


def test_bank_ids_unique_and_subjects_known():
    tasks = tq.load_tasks([str(ROOT / "qa" / "bank")], None, None, 0)
    assert len(tasks) == len({t["id"] for t in tasks}) >= 70
    assert {t["subject"] for t in tasks} <= set(sc.SUBJECTS)


def test_subject_classifier_and_routing():
    clf = sc.default_classifier()
    assert clf.predict("Write a Python function that reverses a linked list") == "code"
    assert clf.predict("Write a haiku about the sea") == "writing"
    assert clf.predict("What objects are visible in this photo?") == "vision"
    table = json.loads((ROOT / "qa" / "routing.example.json").read_text())
    assert sc.route("Fix this Python bug in my function", table, clf)["model"] == "coder"


class Oracle(BaseHTTPRequestHandler):
    answers: dict = {}
    sabotage: set = set()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        model = body.get("model", "")
        reply = self.answers.get(prompt, "I don't know.")
        if model == "bad" and prompt in self.sabotage:
            reply = "Answer: 0" if "Answer:" in reply else "```python\ndef nope(): pass\n```"
        out = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_end_to_end_against_fake_endpoints(tmp_path):
    tasks = tq.load_tasks([str(ROOT / "qa" / "bank")], {"reasoning", "code_exec"}, None, 0)
    answers, sabotage = {}, set()
    for t in tasks:
        p = tq.prompt_for(t)
        if t["kind"] == "reasoning":
            answers[p] = f"Working...\nAnswer: {t['answer']}"
        else:
            answers[p] = "```python\n" + REFERENCE[t["entry_point"]] + "\n```"
        if len(sabotage) < 12:
            sabotage.add(p)
    Oracle.answers, Oracle.sabotage = answers, sabotage
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Oracle)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    out = tmp_path / "r.json"
    try:
        assert tq.main(["--endpoint", f"good={url}@good", "--endpoint", f"bad={url}/v1@bad",
                        "--kind", "reasoning", "code_exec", "--allow-exec", "--concurrency", "4",
                        "--cache", str(tmp_path / "cache"), "--out", str(out)]) == 0
    finally:
        srv.shutdown()
    rep = json.loads(out.read_text())
    assert rep["summary"]["good"]["kind"]["reasoning"]["score"] == 1.0
    assert rep["summary"]["good"]["kind"]["code_exec"]["score"] == 1.0
    cmp = rep["comparisons"][0]
    assert len(cmp["regressed"]) == 12 and not cmp["gained"]
    assert rep["skills"]["good"]["code"] == 1.0
    assert (tmp_path / "cache" / "good.jsonl").exists()
