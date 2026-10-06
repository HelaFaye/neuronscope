"""Director core: analysis, the versioned plan, proposals, review loop and
assignment, without HTTP or models."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import director as d  # noqa: E402

DESC = """# Renderer rewrite

Some background prose about why this matters.

- Shaders: write GLSL vertex and fragment shaders for the terrain mesh and compile them for Vulkan.
- Build: set up the CMake build with the toolchain and dependencies for Linux and Windows.
- Firmware: use Ghidra to disassemble the controller firmware and recover its file format.
- Handle it.

```
code blocks are skipped
```

| Item | Notes |
| --- | --- |
| Port the parser to async | keep the public interface |
"""


def test_analyze_splits_by_skill():
    tasks = d.analyze_text(DESC)
    titles = [t["title"] for t in tasks]
    assert titles[:3] == ["Shaders", "Build", "Firmware"]
    by = {t["title"]: t["labels"] for t in tasks}
    assert by["Shaders"][0] == "graphics" and by["Build"][0] == "systems"
    assert by["Firmware"][0] == "reverse-engineering"
    assert not any("code blocks" in t["detail"] for t in tasks)
    assert any(t["source"] == "row" and "parser" in t["detail"] for t in tasks)
    assert "Handle it." not in [t["detail"] for t in tasks]            # under 4 words: not a task
    sk = d.skill_breakdown([{**t, "id": f"T{i}"} for i, t in enumerate(tasks)])
    assert "graphics" in sk and "systems" in sk


def test_repo_survey_and_gaps(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("import numpy as np\nfrom PIL import Image\n")
    (tmp_path / "shaders").mkdir()
    for n in ("a.vert", "a.frag", "b.comp"):
        (tmp_path / "shaders" / n).write_text("void main(){}")
    (tmp_path / "CMakeLists.txt").write_text("project(x)")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("")
    c = d.analyze_repo(str(tmp_path))
    assert c["subjects"]["graphics"]["files"] == 3 and c["subjects"]["vision"]["files"] == 1
    assert c["build"] == ["CMakeLists.txt"] and c["files"] == 5         # node_modules skipped
    assert d.coverage_gaps([{"labels": ["code"]}], c) == ["graphics"]


def _plan(n=3):
    tasks = [{"title": f"task {i}", "detail": "write the GLSL shader for the mesh", "labels": ["graphics"],
              "proba": {"graphics": 0.9}} for i in range(n)]
    return d.new_plan("P", "goal", tasks)


def test_draft_edits_and_dependency_checks():
    p = _plan()
    d.edit(p, [{"op": "update", "id": "T2", "fields": {"depends_on": ["T1"]}}], by="director", reason="order")
    assert p["version"] == 2 and p["history"][-1]["by"] == "director"
    with pytest.raises(d.PlanError, match="cycle"):
        d.edit(p, [{"op": "update", "id": "T1", "fields": {"depends_on": ["T2"]}}])
    with pytest.raises(d.PlanError, match="unknown task"):
        d.edit(p, [{"op": "update", "id": "T1", "fields": {"depends_on": ["T9"]}}])
    with pytest.raises(d.PlanError, match="unknown subject"):
        d.edit(p, [{"op": "update", "id": "T1", "fields": {"labels": ["poetry"]}}])
    assert p["version"] == 2                                             # failed edits change nothing
    assert [t["id"] for t in d.ready_tasks(p)] == ["T1", "T3"]


def test_after_approval_the_director_can_only_propose():
    p = _plan(2)
    with pytest.raises(d.PlanError, match="no model"):
        d.approve(p)
    d.approve(p, assign_fn=lambda t: {"model": "m-small", "reason": "test"})
    v = p["version"]
    r = d.edit(p, [{"op": "add", "task": {"title": "extra", "detail": "x y z w"}}], by="director", reason="found")
    assert "proposal" in r and p["version"] == v and len(p["tasks"]) == 2
    d.decide(p, r["proposal"]["id"], accept=True)
    assert len(p["tasks"]) == 3 and p["version"] == v + 1
    with pytest.raises(d.PlanError, match="already"):
        d.decide(p, r["proposal"]["id"], accept=False)
    # A person's edits apply at once, and are logged as theirs.
    d.edit(p, [{"op": "drop", "id": "T3"}], by="human", reason="not needed")
    assert d.task(p, "T3")["status"] == "dropped" and p["history"][-1]["by"] == "human"
    with pytest.raises(d.PlanError, match="invalid|unknown"):
        d.propose(p, [{"op": "explode"}])


def report(status="done", followups=(), questions=()):
    return "Here is the work.\n```json\n" + json.dumps(
        {"status": status, "summary": "s", "followups": list(followups), "questions": list(questions)}) + "\n```"


def test_review_loop_feedback_and_reassignment_proposal():
    p = _plan(2)
    d.edit(p, [{"op": "update", "id": "T2", "fields": {"depends_on": ["T1"]}}])
    d.approve(p, assign_fn=lambda t: {"model": "m-small", "reason": "test"})
    d.set_running(p, True)
    p["policy"]["max_attempts"] = 2
    t1 = d.task(p, "T1")
    for n in (1, 2):
        d.start_attempt(p, t1)
        assert [t["id"] for t in d.ready_tasks(p)] == []                 # T1 running, T2 waits on it
        d.finish_attempt(p, t1, report())
        assert t1["status"] == "review"
        d.review(p, "T1", accept=False, feedback=f"missing normals {n}",
                 next_model_fn=lambda t: {"model": "m-big", "reason": "next best"})
    assert t1["status"] == "blocked"
    prop = [x for x in p["proposals"] if x["status"] == "pending"]
    assert prop and prop[0]["changes"][0] == {"op": "assign", "id": "T1", "model": "m-big", "reason": "next best"}
    d.decide(p, prop[0]["id"], accept=True)
    assert t1["status"] == "todo" and t1["assignee"]["model"] == "m-big"
    msgs = d.worker_messages(p, t1)
    assert "missing normals 2" in msgs[1]["content"] and "<- your task" in msgs[1]["content"]
    d.start_attempt(p, t1)
    d.finish_attempt(p, t1, "the shader source")
    assert t1["attempts"][-1]["report"]["status"] == "unreported" and t1["status"] == "review"
    d.review(p, "T1", accept=True)
    t2 = d.task(p, "T2")
    assert d.ready_tasks(p) == [t2]
    assert "the shader source" in d.worker_messages(p, t2)[1]["content"]   # accepted input flows on


def test_policy_followups_blocked_questions_and_finish():
    p = _plan(1)
    d.approve(p, assign_fn=lambda t: {"model": "m", "reason": "r"})
    d.set_running(p, True)
    d.set_policy(p, {"review": "flagged"})
    t = d.task(p, "T1")
    d.start_attempt(p, t)
    d.finish_attempt(p, t, report("blocked", questions=["Which GPU API?"]))
    assert t["status"] == "blocked" and t["questions"][0]["q"] == "Which GPU API?"
    d.answer(p, "T1", ["Vulkan"])
    assert t["status"] == "rework" and "Answer: Vulkan" in d.worker_messages(p, t)[1]["content"]
    d.start_attempt(p, t)
    d.finish_attempt(p, t, report(followups=["Set up the CMake build for the shader compiler."]),
                     check={"flagged": True})
    assert t["status"] == "review"                                       # flagged: a person looks
    d.review(p, "T1", accept=False, feedback="redo")
    d.start_attempt(p, t)
    d.finish_attempt(p, t, report(), check={"flagged": False})
    assert t["status"] == "done" and t["attempts"][-1]["review"]["by"] == "policy"
    assert p["status"] == "done"
    fu = [x for x in p["proposals"] if x["status"] == "pending"]
    assert fu and fu[0]["changes"][0]["task"]["depends_on"] == ["T1"]
    assert "systems" in fu[0]["changes"][0]["task"]["labels"]


def test_worker_errors_retry_then_block():
    p = _plan(1)
    d.approve(p, assign_fn=lambda t: {"model": "m", "reason": "r"})
    t = d.task(p, "T1")
    for _ in range(3):
        d.start_attempt(p, t)
        d.finish_attempt(p, t, None, error="connection refused")
    assert t["status"] == "blocked"


def test_assign_prefers_measured_then_default_then_largest():
    good = {"graded": {"n": 50, "accuracy": 0.8, "hallucination_rate": 0.1}, "subjects": {}}
    cands = [{"id": "small", "size": 1, "fits": True}, {"id": "big", "size": 9, "fits": True},
             {"id": "huge", "size": 99, "fits": False}]
    t = {"labels": ["graphics"], "proba": {"graphics": 0.9}}
    assert d.assign(t, cands, {"small": good}, {})["model"] == "small"
    assert d.assign(t, cands, {}, {"default_model": "small"})["model"] == "small"
    r = d.assign(t, cands, {}, {})
    assert r["model"] == "big" and "largest" in r["reason"]
    assert d.assign(t, cands, {}, {}, exclude=("big",))["model"] == "small"


def test_refined_plan_is_validated():
    p = d.new_plan("P", "", d.analyze_text(DESC))
    good = json.dumps({"tasks": [
        {"title": "Set up the build", "detail": "CMake toolchain", "acceptance": "builds on Linux"},
        {"title": "Write shaders", "detail": "GLSL for the mesh", "depends_on": ["Set up the build"]}]})
    d.replace_tasks(p, d.parse_refined("```json\n" + good + "\n```"), "director", "refined")
    live = [t for t in p["tasks"] if t["status"] != "dropped"]
    assert [t["title"] for t in live] == ["Set up the build", "Write shaders"]
    assert live[1]["depends_on"] == [live[0]["id"]] and live[0]["acceptance"] == "builds on Linux"
    with pytest.raises(d.PlanError, match="not an earlier task"):
        d.parse_refined(json.dumps({"tasks": [{"title": "a", "depends_on": ["b"]}]}))
    with pytest.raises(d.PlanError, match="not JSON"):
        d.parse_refined("sure! here is a plan")


def test_store_roundtrip_and_ids(tmp_path):
    st = d.ProjectStore(tmp_path)
    p = _plan(2)
    st.save(p)
    assert st.load(p["id"])["tasks"][0]["title"] == "task 0"
    assert st.list()[0]["tasks"] == {"todo": 2}
    with pytest.raises(d.PlanError):
        st.load("../etc")
