import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from collections import Counter

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
    "is_anagram": "def is_anagram(a,b):\n    f=lambda s: sorted(s.replace(' ','').lower())\n    return f(a)==f(b)",
    "matmul": "def matmul(a,b):\n    if not a or not b or len(a[0])!=len(b): raise ValueError('shape')\n    return [[sum(x*y for x,y in zip(r,c)) for c in zip(*b)] for r in a]",
    "run_length_encode": "from itertools import groupby\ndef run_length_encode(s):\n    return ''.join(f'{k}{len(list(g))}' for k,g in groupby(s))",
    "valid_ipv4": "def valid_ipv4(s):\n    p=s.split('.')\n    return len(p)==4 and all(x.isdigit() and str(int(x))==x and int(x)<=255 for x in p)",
    "to_snake_case": "import re\ndef to_snake_case(n):\n    n=re.sub(r'([A-Z]+)([A-Z][a-z])',r'\\1_\\2',n)\n    return re.sub(r'([a-z\\d])([A-Z])',r'\\1_\\2',n).lower()",
}

# A passing and a failing answer for every writing task, so a grader change that
# silently accepts everything (or nothing) fails here.
WRITING = {
    "w-haiku": ("Waves fold into foam\nsalt wind carries gull voices\nthe tide keeps its time", "Waves fold into foam, salt wind carries gull voices."),
    "w-two-sent": ("The cat naps in the sun. It wakes only for dinner.", "The cat naps."),
    "w-slogan": ("Every adventure starts with a sip.", "Stay hydrated."),
    "w-tips": ("- Keep a schedule\n- Avoid screens late\n- Keep the room cool\n- Skip late caffeine", "- Keep a schedule\n- Avoid screens"),
    "w-summary": ("Regular exercise strengthens the heart and lungs, helps control weight, lifts mood through endorphins, improves sleep, and lowers the risk of chronic diseases such as diabetes.", "It is good."),
    "w-json": ('{"title": "Bread at Home", "tags": ["baking", "bread"]}', "Title: Bread at Home"),
    "w-limerick": ("A coder who worked through the night\nkept fixing a bug out of sight\nshe changed just one line\nand all ran fine\nthen the tests went from red into bright", "A coder who worked through the night"),
    "w-subject": ("Meeting request for next Tuesday", "Can we meet sometime next week to talk about the roadmap and the budget please"),
    "w-passive": ("The meal was cooked by the chef.", "The chef cooked the meal."),
    "w-caps": ("HELLO", "Hello"),
    "w-synonyms": ("joyful, cheerful, content", "joyful, cheerful"),
    "w-couplet": ("The winter wind is cold and bright,\nit wraps the town in silver light.", "The winter wind is cold."),
    "w-acrostic": ("Clever minds at work\nOpen loops of thought\nDebugging late\nEvery line is taught", "Clever minds\nDebugging\nOpen\nEvery"),
    "w-french": ("Bonjour", "Good day"),
    "w-tweet": ("Our library now opens at 8am on weekdays! #ReadMore", "Our library now opens at 8am on weekdays!"),
    "w-one-sentence": ("The meeting moved to Wednesday because the projector broke, so bring laptops.", "The meeting moved. Bring laptops."),
    "w-steps": ("1. Boil water\n2. Steep the tea\n3. Pour and enjoy", "1. Boil water\n3. Pour"),
    "w-although": ("Although it rained all day, we still enjoyed our long walk outside.", "Although it rained, we walked."),
    "w-title": ("The Robot Who Lost Its Way", "the robot who lost its way!"),
    "w-question": ("Where can I find books about local history?", "I want books about local history."),
    "w-then-question": ("It is raining hard today. Will it stop by noon?", "It is raining hard today. It will stop by noon."),
    "w-fruits": ("1. Apple\n2. Banana\n3. Cherry\n4. Mango\n5. Pear", "1. Apple\n2. Banana\n3. Cherry"),
    "w-five-words": ("Keep going, you are close.", "Keep going."),
    "w-csv": ("name,age\nAlice,30\nBob,25", "name,age\nAlice,30"),
    "w-alpha": ("apple\nbanana\ncherry", "cherry\napple\nbanana"),
    "w-yesno": ("Yes, water makes other things wet.", "Water is wet, yes."),
    "w-markdown": ("# Notes\n- buy milk\n- call Sam", "## Notes\n- buy milk\n- call Sam"),
    "w-quote": ('She said "see you soon" and left.', "She said see you soon and left."),
    "w-lowercase": ("coffee tastes best in the morning.", "Coffee tastes best in the morning."),
    "w-paragraphs": ("The Moon orbits Earth.\n\nIt has phases.\n\nIt causes tides.", "The Moon orbits Earth. It has phases. It causes tides."),
}


def bank(kind):
    return [json.loads(l) for l in (ROOT / "qa" / "bank" / f"{kind}.jsonl").read_text().splitlines() if l.strip()]


def test_reference_solutions_pass_every_coding_task():
    tasks = [t for t in bank("code") if t["kind"] == "code_exec"]
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
    t = next(t for t in bank("code") if t["kind"] == "code_exec")
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
    ("Answer: 9 - 4 = 5", {"answer": "5", "answer_type": "number"}, "correct"),
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
    assert len(tasks) == len({t["id"] for t in tasks}) >= 210
    assert {t["subject"] for t in tasks} <= set(sc.QA_SUBJECTS)


def test_every_subject_has_enough_graded_items():
    tasks = tq.load_tasks([str(ROOT / "qa" / "bank")], None, None, 0)
    graded = Counter(t["subject"] for t in tasks if t["kind"] != "canary")
    assert set(graded) == set(sc.QA_SUBJECTS)
    assert min(graded.values()) >= 30, graded


def test_writing_checks_accept_good_and_reject_bad():
    tasks = {t["id"]: t for t in bank("writing")}
    assert set(tasks) == set(WRITING)
    for tid, (good, bad) in WRITING.items():
        assert tq.grade_constraints(good, tasks[tid]) == ("correct", ""), (tid, tq.check_constraints(good, tasks[tid]["checks"]))
        assert tq.grade_constraints(bad, tasks[tid])[0] == "wrong", tid
    assert tq.grade_constraints("I can't help with that.", tasks["w-haiku"])[0] == "abstained"


def test_vision_tasks_render():
    pytest.importorskip("PIL")
    for t in bank("vision"):
        msg = tq.message_for({**t, "kind": t["kind"]})
        url = msg["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,") and len(url) > 200


def test_per_subject_sampling_is_balanced():
    tasks = tq.load_tasks([str(ROOT / "qa" / "bank")], None, None, 0)
    picked = tq.per_subject_sample(tasks, 8, seed=1)
    graded = Counter(t["subject"] for t in picked if t["kind"] != "canary")
    assert set(graded.values()) == {8}
    code_kinds = Counter(t["kind"] for t in picked if t["subject"] == "code")
    assert len(code_kinds) >= 3          # round-robin across kinds, not 8 of one
    assert tq.per_subject_sample(tasks, 8, seed=1) == picked


def test_subject_classifier_and_routing():
    clf = sc.default_classifier()
    assert clf.predict("Write a Python function that reverses a linked list") == "code"
    assert clf.predict("Write a haiku about the sea") == "writing"
    assert clf.predict("What objects are visible in this photo?") == "vision"
    table = json.loads((ROOT / "qa" / "routing.example.json").read_text())
    assert sc.route("Fix this Python bug in my function", table, clf)["model"] == "coder"


def key_of(message):
    """Same question about different images must map to different answers."""
    c = message["content"]
    if isinstance(c, str):
        return c
    text = next(x["text"] for x in c if x.get("type") == "text")
    img = next((x["image_url"]["url"] for x in c if x.get("type") == "image_url"), "")
    return text + "|" + str(hash(img))


class Oracle(BaseHTTPRequestHandler):
    answers: dict = {}
    sabotage: set = set()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = key_of(body["messages"][0])
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
    tasks = tq.load_tasks([str(ROOT / "qa" / "bank")], {"reasoning", "code_exec", "constraints"}, None, 0)
    answers, sabotage = {}, set()
    for t in tasks:
        p = key_of(tq.message_for(t))
        if t["kind"] == "reasoning":
            answers[p] = f"Working...\nAnswer: {t['answer']}"
        elif t["kind"] == "constraints":
            answers[p] = WRITING[t["id"]][0]
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
                        "--kind", "reasoning", "code_exec", "constraints", "--allow-exec", "--concurrency", "4",
                        "--cache", str(tmp_path / "cache"), "--out", str(out)]) == 0
    finally:
        srv.shutdown()
    rep = json.loads(out.read_text())
    assert rep["summary"]["good"]["kind"]["reasoning"]["score"] == 1.0
    assert rep["summary"]["good"]["kind"]["code_exec"]["score"] == 1.0
    assert rep["summary"]["good"]["kind"]["constraints"]["score"] == 1.0
    assert rep["summary"]["good"]["subject"]["vision"]["accuracy"] == 1.0
    cmp = rep["comparisons"][0]
    assert len(cmp["regressed"]) == 12 and not cmp["gained"]
    assert rep["skills"]["good"]["code"] == 1.0
    assert (tmp_path / "cache" / "good.jsonl").exists()


def test_optional_word_bans_pack():
    pack = [json.loads(l) for l in (ROOT / "qa" / "bank" / "optional" / "word-bans.jsonl").read_text().splitlines() if l.strip()]
    good = {"w-no-and": "The sky glowed orange as the sun slipped below the sandy horizon.",
            "w-lipogram": "Rain falls softly on a dark city road.",
            "w-no-very": "The meal was wonderful. Every bite tasted fresh.",
            "w-no-i": "Welcome to the old town. My name is Sam. Let us explore together."}
    bad = {"w-no-and": "The sky turned orange and pink.", "w-lipogram": "Rain falls softly on the quiet street.",
           "w-no-very": "The meal was very good. It was really tasty.", "w-no-i": "I am Sam. I guide tours. I love it."}
    for t in pack:
        assert tq.grade_constraints(good[t["id"]], t)[0] == "correct", (t["id"], tq.check_constraints(good[t["id"]], t["checks"]))
        assert tq.grade_constraints(bad[t["id"]], t)[0] == "wrong", t["id"]
    # Not in the default bank; added only with --with word-bans.
    default = tq.load_tasks([str(ROOT / "qa" / "bank")], None, None, 0)
    assert not {t["id"] for t in pack} & {t["id"] for t in default}


def _docker_ok():
    import shutil
    import subprocess
    if not shutil.which("docker"):
        return False
    r = subprocess.run(["docker", "image", "inspect", "python:3.12-slim"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _docker_ok(), reason="needs a running docker with python:3.12-slim")
def test_docker_sandbox_contains_hostile_code():
    import testqa as tq
    tq.configure_sandbox("docker")
    try:
        t = ["assert add(2, 3) == 5"]
        ok = "def add(a, b): return a + b"
        assert tq.run_code(ok, t, timeout=3)[0] == "correct"
        assert tq.run_code("def add(a, b): return a - b", t, timeout=3)[0] == "wrong"
        net = tq.run_code("import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=2)\n" + ok, t, timeout=3)
        assert net[0] == "wrong" and "unreachable" in net[1].lower()
        assert "Read-only" in tq.run_code("open('/work/x', 'w').write('x')\n" + ok, t, timeout=3)[1]
        assert "65534" in tq.run_code("import os\nassert os.getuid() == 0, os.getuid()\n" + ok, t, timeout=3)[1]
        assert tq.run_code("while True: pass\n" + ok, t, timeout=2)[0] == "timeout"
        assert tq.run_code("import os\nfor _ in range(200):\n    os.fork()\n" + ok, t, timeout=3)[0] == "wrong"
    finally:
        tq.configure_sandbox(None)


def test_subject_unknown_when_evidence_is_thin_and_routes_to_default():
    clf = sc.default_classifier()
    for text in ("Handle it.", "Blorf the quandle snivets."):
        a = clf.analyze(text)
        assert a["unknown"] and a["labels"] == [] and clf.predict(text) == sc.UNKNOWN
        assert clf.route_weights(text) == {}
    table = json.loads((ROOT / "qa" / "routing.example.json").read_text())
    r = sc.route("Handle it.", table, clf)
    assert r["subject"] == sc.UNKNOWN and r["model"] == table["default"]


def test_subject_returns_every_label_above_the_cutoff():
    clf = sc.default_classifier()
    a = clf.analyze("Set up the CMake build and write the GLSL shader for the terrain mesh.")
    assert a["labels"][:2] == ["graphics", "systems"]
    assert all(a["proba"][k] >= sc.CUTOFF for k in a["labels"])
    w = clf.route_weights("Set up the CMake build and write the GLSL shader for the terrain mesh.")
    assert set(w) == {"graphics", "systems"} and abs(sum(w.values()) - 1) < 1e-9


def test_task_subjects_and_word_start_keywords():
    clf = sc.default_classifier()
    assert clf.predict("Use Ghidra to find the routine that decrypts the save file.") == "reverse-engineering"
    assert clf.predict("Get the project building with Meson and fix the linker errors.") == "systems"
    assert clf.predict("Write a vertex shader that skins the character mesh.") == "graphics"
    assert sc.keyword_hits("Copy the data from the server", "reverse-engineering") == 0   # "rom" in "from"
    assert sc.keyword_hits("Dump the ROM and the firmware", "reverse-engineering") == 2


def test_rank_uses_overall_numbers_when_subject_is_unknown():
    import model_stats as ms
    good = {"graded": {"n": 50, "correct": 40, "wrong": 5, "abstained": 5, "accuracy": 0.8,
                       "hallucination_rate": 0.1, "abstention_rate": 0.1}, "subjects": {}}
    weak = {"graded": {"n": 50, "correct": 20, "wrong": 25, "abstained": 5, "accuracy": 0.4,
                       "hallucination_rate": 0.5, "abstention_rate": 0.1}, "subjects": {}}
    pick = ms.rank({}, {"good": good, "weak": weak}, 20, 5)
    assert pick["model"] == "good" and pick["subject"] == "unknown" and "best overall" in pick["reason"]
