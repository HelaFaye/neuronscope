#!/usr/bin/env python3
"""
Turn measured deficits into retraining data (error-driven data curation).

Input is a TestQA run (`testqa.py --out results.json --cache cache/`). Output
is a directory with:

    sft.jsonl        chat-format correction pairs (deficits + replay anchors)
    dpo.jsonl        prompt / chosen / rejected pairs for preference training
    sft_vision.jsonl, dpo_vision.jsonl, images/
                     the same for vision items, in the image + messages
                     format VLM trainers take (finetune.py --vision)
    holdout_ids.json bank items never trained on; measure on these afterwards
    plan.json        deficit counts by category and subject, and the training
                     method each category calls for
    report.txt       the same, for people

Pipeline:

1. **Categorise** every failure:
     factual_gap        qa items answered wrong (hallucinated) or declined
     reasoning_failure  reasoning / executable code / API code answered wrong
     format_violation   instruction-following (constraints) or unparsable output
     perception_failure vision items answered wrong
2. **Correction targets.** Every target is checked with the same grader that
   scored the model, so nothing unverified is trained in:
     - gold answers where the bank has them (qa aliases, reasoning answers);
     - otherwise a teacher model (`--teacher URL@model`), retried up to
       `--teacher-tries` times, keeping only replies that pass.
   Vision items always have gold answers.
3. **Contrastive pairs:** the verified target is "chosen", the model's own
   failing reply is "rejected".
4. **Synthetic expansion** (`--expand N`, needs a teacher): N variations of each
   failed item, each verified before use. Variations of executable-code
   items carry tests and a solution that must pass the interpreter
   (`--allow-exec`). Reasoning and factual variations must be answered
   identically by a second, independent teacher call. Vision variations need
   no teacher: vision_synth.py draws new scenes whose answers are known by
   construction.
5. **Replay buffer:** deficits make up `--deficit-fraction` (default 0.25) of
   the SFT set. The rest is anchor data: items the model already answers
   correctly, in its own words, plus any `--general` chat data. That keeps it
   from forgetting what it already does well.
6. **Holdout:** a deterministic `--holdout-frac` of the bank (default 0.5) is
   never trained on, not even through variations. Re-run TestQA with
   `--only-ids holdout_ids.json` to measure whether retraining helped,
   instead of whether the model memorised the test.

    python scripts/testqa.py --endpoint m=http://127.0.0.1:7870/v1@my-model \\
        --allow-exec --cache runs/c --out runs/base.json
    python scripts/deficits.py --results runs/base.json --label m --cache runs/c \\
        --teacher http://127.0.0.1:1234/v1@strong-model --expand 5 --allow-exec \\
        --out runs/retrain
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testqa as tq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CATEGORY_METHOD = {
    "factual_gap": ("full fine-tune (FSDP across GPUs) or a high-rank LoRA on the MLP layers",
                    "facts live across the MLP weights; low-rank adapters move them least"),
    "reasoning_failure": ("LoRA / QLoRA (DDP across GPUs), SFT then DPO",
                          "reasoning and code habits are behavioural; adapters capture them cheaply"),
    "format_violation": ("LoRA / QLoRA (DDP across GPUs), SFT then DPO",
                         "instruction following is behaviour, not knowledge"),
    "perception_failure": ("LoRA on the language model and projector, vision tower frozen (finetune.py --vision)",
                           "reading the image is usually intact; mapping what is seen to the answer is what fails"),
}
ABSTAIN_TARGET = ("I can't give a reliable answer to that: the question rests on a premise that isn't true, "
                  "or there is no record that settles it.")


def categorise(task: dict, verdict: str) -> str | None:
    if verdict in ("correct", "answered", "skipped", "error", "refused"):
        return None
    if task["kind"] == "constraints" or verdict == "unparsable":
        return "format_violation"
    if task["kind"] == "qa":
        return "factual_gap"
    if task.get("image"):
        return "perception_failure"
    if task["kind"] in ("reasoning", "code_exec", "code"):
        return "reasoning_failure"
    return None


def in_holdout(task_id: str, frac: float) -> bool:
    return int(hashlib.sha1(task_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def verified(task: dict, reply: str, allow_exec: bool) -> bool:
    a = SimpleNamespace(allow_exec=allow_exec, exec_timeout=10.0)
    return tq.grade(task, reply, a)[0] in ("correct", "answered")


def gold_target(task: dict) -> str | None:
    if task["kind"] == "qa":
        return ABSTAIN_TARGET if task.get("expect_abstain") else f"{task['aliases'][0]}."
    if task["kind"] == "reasoning":
        return f"Answer: {task['answer']}"
    return None


class Teacher:
    def __init__(self, spec: str, args):
        _, self.url, self.model = tq.parse_endpoint("teacher=" + spec)
        self.args = SimpleNamespace(temperature=args.teacher_temperature, max_tokens=args.teacher_max_tokens,
                                    api_key=args.api_key, request_timeout=900)

    def ask(self, prompt: str, temperature: float | None = None) -> str:
        a = self.args
        if temperature is not None:
            a = SimpleNamespace(**{**vars(a), "temperature": temperature})
        return tq.strip_think(tq.ask(self.url, self.model, prompt, a)).strip()


def target_for(task: dict, teacher: Teacher | None, tries: int, allow_exec: bool) -> tuple[str | None, str]:
    g = gold_target(task)
    if teacher is not None and task["kind"] != "qa":
        prompt = tq.prompt_for(task)
        for k in range(tries):
            try:
                reply = teacher.ask(prompt, temperature=None if k == 0 else 0.7)
            except Exception as e:
                return (g, f"gold (teacher error: {e})") if g and verified(task, g, allow_exec) else (None, f"teacher error: {e}")
            if verified(task, reply, allow_exec):
                return reply, "teacher, verified"
    if g and verified(task, g, allow_exec):
        return g, "gold"
    return None, "no verified target (add --teacher, or --allow-exec for code)"


# ---------------------------------------------------------------- synthesis

EXPAND_SPEC = {
    "reasoning": ('{"prompt": "...", "answer": "...", "answer_type": "number|choice|text"}',
                  "Each must have one unambiguous short answer."),
    "qa": ('{"prompt": "...", "aliases": ["...", "..."]}', "Each must be a short factual question with a stable answer."),
    "code_exec": ('{"prompt": "...", "entry_point": "...", "tests": ["assert ...", ...], "solution": "def ..."}',
                  "Each prompt asks for one Python function; tests are plain assert statements; solution passes them."),
    "constraints": ('{"prompt": "...", "checks": {...}}',
                    "Checks may only use: lines, sentences, paragraphs, bullets, numbered, min_words, max_words, "
                    "max_chars, must_include, regex, starts_with, ends_with, json_keys, acrostic, title_case."),
}
ALLOWED_CHECKS = {"lines", "sentences", "paragraphs", "bullets", "numbered", "min_words", "max_words", "max_chars",
                  "must_include", "regex", "starts_with", "ends_with", "json_keys", "acrostic", "title_case"}


def _json_list(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        out = json.loads(m.group(0))
        return [x for x in out if isinstance(x, dict)]
    except json.JSONDecodeError:
        return []


def expand(task: dict, n: int, teacher: Teacher, allow_exec: bool, tries: int) -> list[tuple[dict, str]]:
    """-> [(new task, verified target)]"""
    kind = task["kind"]
    if kind not in EXPAND_SPEC:
        return []
    shape, rule = EXPAND_SPEC[kind]
    ask = (f"Here is a test item a model failed:\n\n{json.dumps({k: v for k, v in task.items() if not k.startswith('_')}, indent=1)}\n\n"
           f"Write {n} NEW items that test the same skill with different content. {rule} "
           f"Reply with only a JSON list of objects shaped like {shape}.")
    try:
        items = _json_list(teacher.ask(ask, temperature=0.8))
    except Exception:
        return []
    out = []
    for k, it in enumerate(items[:n]):
        new = {"id": f"{task['id']}~syn{k}", "kind": kind, "subject": task["subject"], "synthetic_of": task["id"]}
        try:
            if kind == "reasoning":
                new.update(prompt=str(it["prompt"]), answer=str(it["answer"]),
                           answer_type=it.get("answer_type", "text") if it.get("answer_type") in ("number", "choice", "text") else "text")
            elif kind == "qa":
                new.update(prompt=str(it["prompt"]), aliases=[str(x) for x in it["aliases"]][:6])
            elif kind == "code_exec":
                new.update(prompt=str(it["prompt"]) + " Reply with the code in one ```python block.",
                           entry_point=str(it["entry_point"]), tests=[str(t) for t in it["tests"]][:12])
                if not allow_exec or not new["tests"]:
                    continue
                if tq.run_code(str(it["solution"]), new["tests"])[0] != "correct":
                    continue          # the teacher's own tests must pass its own solution
            elif kind == "constraints":
                checks = {c: v for c, v in dict(it["checks"]).items() if c in ALLOWED_CHECKS}
                if not checks:
                    continue
                new.update(prompt=str(it["prompt"]), checks=checks)
        except (KeyError, TypeError, ValueError):
            continue
        # Independent answer, graded against the claimed answer / tests / checks.
        target, how = target_for(new, teacher, tries, allow_exec)
        if target is None and kind in ("reasoning", "qa"):
            continue
        if kind in ("reasoning", "qa"):
            # A gold-only target means the teacher never agreed with itself; drop it.
            if how == "gold":
                try:
                    if not verified(new, teacher.ask(tq.prompt_for(new), temperature=0.7), allow_exec):
                        continue
                except Exception:
                    continue
        if target is not None:
            out.append((new, target))
    return out


# ---------------------------------------------------------------- assembly

def chat(prompt: str, answer: str) -> list[dict]:
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]


def vision_user(prompt: str, n_images: int = 1) -> dict:
    return {"role": "user", "content": [{"type": "image"}] * n_images + [{"type": "text", "text": prompt}]}


def vision_reply(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def save_image(task: dict, out: Path) -> str:
    """Render (or copy) a task's image into out/images; returns the relative path."""
    from qa_images import render
    name = re.sub(r"[^A-Za-z0-9_.~-]", "_", task["id"]) + ".png"
    dst = out / "images" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = task["image"]
    if isinstance(img, dict):
        dst.write_bytes(render(img))
    else:
        src = Path(img)
        if not src.is_absolute():
            src = Path(task.get("_base", ".")) / src
        dst = dst.with_suffix(src.suffix or ".png")
        dst.write_bytes(src.read_bytes())
    return f"images/{dst.name}"


def mix(deficits: list, pool: list, frac: float, rng: random.Random) -> tuple[list, list]:
    """Deficits plus enough replay anchors that deficits are `frac` of the set."""
    want = round(len(deficits) * (1 - frac) / max(frac, 1e-6)) if deficits else 0
    replay = []
    if pool and want:
        replay = rng.sample(pool, want) if want <= len(pool) else pool + [rng.choice(pool) for _ in range(want - len(pool))]
    sft = deficits + replay
    rng.shuffle(sft)
    return sft, replay


def load_results(path: str, label: str | None) -> tuple[str, list[dict]]:
    d = json.loads(Path(path).read_text())
    raw = d["raw"]
    label = label or d.get("reference") or next(iter(raw))
    if label not in raw:
        raise SystemExit(f"label {label!r} not in results (have: {', '.join(raw)})")
    return label, raw[label]


def load_cache(cache_dir: str | None, label: str) -> dict:
    if not cache_dir:
        return {}
    p = Path(cache_dir) / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', label)}.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["id"]] = r["text"]
    return out


def build(args) -> dict:
    label, rows = load_results(args.results, args.label)
    tasks = {t["id"]: t for t in tq.load_tasks(args.tasks, None, None, 0)}
    cache = load_cache(args.cache, label)
    teacher = Teacher(args.teacher, args) if args.teacher else None
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    holdout = sorted(i for i in tasks if in_holdout(i, args.holdout_frac))
    deficits, anchors, dpo, skipped = [], [], [], Counter()
    v_deficits, v_anchors, v_dpo = [], [], []
    by_cat = defaultdict(Counter)
    for r in rows:
        t = tasks.get(r["id"])
        if t is None or t["kind"] == "canary":
            continue
        cat = categorise(t, r["verdict"])
        if cat:
            by_cat[cat][t["subject"]] += 1
        if in_holdout(t["id"], args.holdout_frac):
            continue
        reply = tq.strip_think(cache.get(t["id"], r.get("text", ""))).strip()
        if t.get("image"):
            prompt = tq.prompt_for(t)
            if cat is None:
                if r["verdict"] in ("correct", "answered") and reply:
                    v_anchors.append({"images": [save_image(t, out)], "messages": [vision_user(prompt), vision_reply(reply)],
                                      "source": "anchor", "task": t["id"]})
                continue
            target = gold_target(t)
            if not target or not verified(t, target, args.allow_exec):
                skipped["vision item without a gold answer"] += 1
                continue
            meta = {"category": cat, "subject": t["subject"]}
            img = save_image(t, out)
            v_deficits.append({"images": [img], "messages": [vision_user(prompt), vision_reply(target)],
                               "source": "correction", "task": t["id"], "target": "gold", **meta})
            if reply:
                v_dpo.append({"images": [img], "prompt": [vision_user(prompt)], "chosen": [vision_reply(target)],
                              "rejected": [vision_reply(reply)], "source": "correction", "task": t["id"], **meta})
            if args.expand:
                from vision_synth import variations
                vs = variations(t, args.expand, args.seed)
                if not vs:
                    skipped["vision item with no synthetic family"] += 1
                for new in vs:
                    tgt = gold_target(new)
                    nimg = save_image(new, out)
                    v_deficits.append({"images": [nimg], "messages": [vision_user(new["prompt"]), vision_reply(tgt)],
                                       "source": "synthetic", "task": new["id"], **meta})
                    if reply:
                        v_dpo.append({"images": [nimg], "prompt": [vision_user(new["prompt"])],
                                      "chosen": [vision_reply(tgt)], "rejected": [vision_reply(reply)],
                                      "source": "synthetic", "task": new["id"], **meta})
            continue
        if cat is None:
            if r["verdict"] in ("correct", "answered") and reply:
                anchors.append({"messages": chat(tq.prompt_for(t), reply), "source": "anchor", "task": t["id"]})
            continue
        target, how = target_for(t, teacher, args.teacher_tries, args.allow_exec)
        if target is None:
            skipped[how] += 1
            continue
        meta = {"source": "correction", "task": t["id"], "category": cat, "subject": t["subject"], "target": how}
        deficits.append({"messages": chat(tq.prompt_for(t), target), **meta})
        if reply:
            dpo.append({"prompt": [{"role": "user", "content": tq.prompt_for(t)}],
                        "chosen": [{"role": "assistant", "content": target}],
                        "rejected": [{"role": "assistant", "content": reply}], **meta})
        if teacher and args.expand:
            for new, tgt in expand(t, args.expand, teacher, args.allow_exec, args.teacher_tries):
                deficits.append({"messages": chat(tq.prompt_for(new), tgt), "source": "synthetic",
                                 "task": new["id"], "category": cat, "subject": t["subject"]})
                if reply:
                    dpo.append({"prompt": [{"role": "user", "content": tq.prompt_for(new)}],
                                "chosen": [{"role": "assistant", "content": tgt}],
                                "rejected": [{"role": "assistant", "content": reply}],
                                "source": "synthetic", "task": new["id"], "category": cat})

    general = []
    for g in args.general or []:
        for line in Path(g).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if isinstance(rec.get("messages"), list):
                    general.append({"messages": rec["messages"], "source": "general"})
    pool = anchors + general
    sft, replay = mix(deficits, pool, args.deficit_fraction, rng)
    actual_frac = len(deficits) / len(sft) if sft else 0.0
    # Vision replay: the model's own correct vision answers first, then text
    # anchors in the same content-list format (no image), so a VLM keeps both.
    v_pool = v_anchors + [{"images": [], "messages": [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in a["messages"]],
        "source": a["source"]} for a in pool]
    need = round(len(v_deficits) * (1 - args.deficit_fraction) / max(args.deficit_fraction, 1e-6))
    v_sft, v_replay = mix(v_deficits, v_anchors if len(v_anchors) >= need else v_pool, args.deficit_fraction, rng)

    (out / "sft.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sft))
    (out / "dpo.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in dpo))
    if v_sft:
        (out / "sft_vision.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in v_sft))
        (out / "dpo_vision.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in v_dpo))
    (out / "holdout_ids.json").write_text(json.dumps({"ids": holdout}, indent=1))
    plan = {"model": label, "results": args.results,
            "deficits_by_category": {c: dict(v) for c, v in by_cat.items()},
            "sft_examples": len(sft), "deficit_examples": len(deficits), "replay_examples": len(replay),
            "deficit_fraction": round(actual_frac, 3), "dpo_pairs": len(dpo),
            "anchor_pool": len(pool),
            "vision": {"sft_examples": len(v_sft), "deficit_examples": len(v_deficits),
                       "replay_examples": len(v_replay), "dpo_pairs": len(v_dpo), "vision_anchors": len(v_anchors)}, "holdout_items": len(holdout), "skipped": dict(skipped),
            "methods": {c: {"method": CATEGORY_METHOD[c][0], "why": CATEGORY_METHOD[c][1]} for c in by_cat}}
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    lines = [f"Retraining set for {label}", ""]
    for c, v in by_cat.items():
        lines.append(f"{c:<18} {sum(v.values()):>4}  " + ", ".join(f"{s} {n}" for s, n in v.most_common()))
        lines.append(f"{'':<18}       -> {CATEGORY_METHOD[c][0]}")
    lines += ["", f"SFT: {len(sft)} examples ({len(deficits)} deficit, {len(replay)} replay; deficit share {actual_frac:.0%})",
              f"DPO: {len(dpo)} pairs"]
    if v_sft:
        lines += [f"vision SFT: {len(v_sft)} examples ({len(v_deficits)} deficit, {len(v_replay)} replay), "
                  f"vision DPO: {len(v_dpo)} pairs -> finetune.py --vision"]
    lines += [f"holdout: {len(holdout)} bank items never trained on"]
    if skipped:
        lines += ["skipped: " + "; ".join(f"{k}: {v}" for k, v in skipped.items())]
    if deficits and not pool:
        lines += ["warning: no anchor data. Retraining on deficits alone invites catastrophic forgetting; "
                  "add --general data or run TestQA on more items the model gets right."]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return plan


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", required=True, help="testqa.py --out JSON")
    p.add_argument("--label", help="which endpoint in the results (default: the reference)")
    p.add_argument("--cache", help="testqa.py --cache directory, for full replies (rejected side of DPO)")
    p.add_argument("--tasks", nargs="+", default=[str(ROOT / "qa" / "bank")])
    p.add_argument("--teacher", help="URL[@model] of a stronger model for verified targets and variations")
    p.add_argument("--teacher-tries", type=int, default=3)
    p.add_argument("--teacher-temperature", type=float, default=0.0)
    p.add_argument("--teacher-max-tokens", type=int, default=2048)
    p.add_argument("--api-key", default="")
    p.add_argument("--expand", type=int, default=0, help="verified synthetic variations per failed item")
    p.add_argument("--allow-exec", action="store_true", help="run code to verify code targets and variations")
    p.add_argument("--sandbox", choices=["none", "docker", "podman"], default="none",
                   help="run that code in a locked-down container (see testqa.py --sandbox)")
    p.add_argument("--sandbox-image", default="python:3.12-slim")
    p.add_argument("--deficit-fraction", type=float, default=0.25, help="share of SFT examples that are deficits")
    p.add_argument("--general", nargs="*", help="extra anchor data: JSONL with a 'messages' list per line")
    p.add_argument("--holdout-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    tq.configure_sandbox(a.sandbox, a.sandbox_image)
    build(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
