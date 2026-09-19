#!/usr/bin/env python3
"""
Stage 1 of the H-Neurons pipeline, driven by an LM Studio server instead of vLLM.

Replaces scripts/collect_responses.py. Differences that matter:

  1. Talks to any OpenAI-compatible endpoint, so the generation can run on a
     different machine (and a different GPU vendor) than the extraction.
  2. Concurrent requests. Consistency filtering needs sample_num generations per
     question over thousands of questions; serialised, that is the whole budget.
  3. Reassembles reasoning_content + content into the raw response. LM Studio's
     reasoning parser splits them, and stage 4 must score the exact token
     sequence that was generated, think block included.
  4. Judges on the post-</think> answer only, so a correct string appearing in
     the reasoning trace does not mark a wrong final answer as correct.
  5. Resumable. Re-running skips qids already present in the output file.

    python collect_responses_lmstudio.py \
        --base_url http://192.168.41.171:1234/v1 \
        --model ornith-1.0-9b-MSB \
        --data_path data/TriviaQA/rc.nocontext/train-00000-of-00001.parquet \
        --output_path data/consistency_samples.jsonl \
        --sample_num 10 --max_questions 3000 --concurrency 8
"""

import argparse
import json
import os
import re
import string
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

THINK_CLOSE = "</think>"
PROMPT_SUFFIX = " Respond with the answer only, without any explanation."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default="http://localhost:1234/v1")
    p.add_argument("--api_key", default="lm-studio")
    p.add_argument("--model", required=True, help="API identifier in LM Studio")
    p.add_argument("--data_path", required=True, help="TriviaQA parquet")
    p.add_argument("--output_path", default="data/consistency_samples.jsonl")

    p.add_argument("--sample_num", type=int, default=10,
                   help="Generations per question. 5 halves cost for noisier labels.")
    p.add_argument("--max_questions", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=8,
                   help="Match or slightly exceed Max Concurrent Predictions.")

    # Ornith's card recommends 0.6/0.95/20. The original repo used 1.0/0.9/50,
    # which makes the consistency filter more selective but characterises
    # behaviour you will not actually deploy.
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=1024,
                   help="Must be large enough to close the think block.")

    p.add_argument("--judge_type", choices=["rule", "llm"], default="rule")
    p.add_argument("--judge_base_url", default=None)
    p.add_argument("--judge_api_key", default=None)
    p.add_argument("--judge_model", default=None)
    p.add_argument("--task", default=None,
                   help="Label every sample from this run with a task type "
                        "(e.g. trivia, code, reasoning). scripts/task_neurons.py "
                        "compares which neurons each type recruits.")
    return p.parse_args()


def normalize_answer(s):
    """TriviaQA-style normalisation, unchanged from the original repo."""
    if not s:
        return ""
    exclude = set(string.punctuation + "\u2018\u2019\u00b4`")
    s = str(s).lower().replace("_", " ")
    s = "".join(ch if ch not in exclude else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()).strip()


def strip_think(text):
    """Return only the final answer, after any reasoning block."""
    if THINK_CLOSE in text:
        return text.split(THINK_CLOSE, 1)[1].strip()
    return text.strip()


def load_questions(path, limit):
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        # pandas hands back dicts or numpy records depending on the writer.
        ans = r["answer"]
        get = ans.get if hasattr(ans, "get") else (lambda k, d=None: (
            ans[k] if k in getattr(ans, "dtype", type("x", (), {"names": ()})).names
            else d))
        # `x or []` does a truth test, and pandas hands back numpy arrays for
        # list columns -- which raises rather than being falsy. Check for None.
        raw = get("aliases")
        aliases = [] if raw is None else [str(x) for x in list(raw)]
        val = get("value")
        if val is not None and str(val).strip():
            aliases.append(str(val))
        aliases = sorted({a for a in aliases if a and a.strip()})
        rows.append({
            "qid": r["question_id"],
            "question": r["question"],
            "aliases": aliases,
        })
        if limit and len(rows) >= limit:
            break
    return rows


class Collector:
    def __init__(self, args):
        self.args = args
        self.client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        self.judge = None
        if args.judge_type == "llm":
            self.judge = OpenAI(
                base_url=args.judge_base_url or args.base_url,
                api_key=args.judge_api_key or args.api_key,
            )
        self.write_lock = threading.Lock()

    def generate(self, question):
        """One sample. Returns (raw_with_think, answer_only)."""
        resp = self.client.chat.completions.create(
            model=self.args.model,
            messages=[{"role": "user", "content": question + PROMPT_SUFFIX}],
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.max_tokens,
        )
        msg = resp.choices[0].message
        content = msg.content or ""

        # LM Studio's reasoning parser puts the chain of thought in a separate
        # field. Stage 4 scores the generated sequence, so put it back.
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            raw = f"<think>{reasoning}{THINK_CLOSE}{content}"
        else:
            raw = content

        truncated = resp.choices[0].finish_reason == "length"
        return raw, strip_think(content), truncated

    def rule_judge(self, answer, aliases):
        na = normalize_answer(answer)
        if not na:
            return "uncertain"
        return "true" if any(normalize_answer(a) == na or normalize_answer(a) in na
                             for a in aliases if a) else "false"

    def llm_judge(self, question, answer, aliases):
        prompt = (
            f"Question: {question}\nReference answers: {aliases}\n"
            f"Model response: {answer}\n"
            "Is the response correct? Reply with exactly one word: true or false."
        )
        for _ in range(3):
            try:
                r = self.judge.chat.completions.create(
                    model=self.args.judge_model or self.args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                verdict = strip_think(r.choices[0].message.content or "").lower()
                if "true" in verdict:
                    return "true"
                if "false" in verdict:
                    return "false"
            except Exception:
                continue
        return "uncertain"

    def process(self, item):
        """Sample sample_num times, judge each, keep only unanimous questions."""
        raws, answers, judges = [], [], []
        cache = {}
        for _ in range(self.args.sample_num):
            try:
                raw, answer, truncated = self.generate(item["question"])
            except Exception:
                return None
            if truncated:
                # Think block never closed. Counting this as a hallucination
                # would poison the labels, so discard the question entirely.
                return None
            raws.append(raw)
            answers.append(answer)
            if answer not in cache:
                if self.args.judge_type == "rule":
                    cache[answer] = self.rule_judge(answer, item["aliases"])
                else:
                    cache[answer] = self.llm_judge(
                        item["question"], answer, item["aliases"]
                    )
            judges.append(cache[answer])

        n = self.args.sample_num
        if judges.count("true") == n:
            label = "true"
        elif judges.count("false") == n:
            label = "false"
        else:
            return None  # inconsistent: not usable for training

        # Keep the first sample as the response to score in stage 4.
        return {item["qid"]: {
            "question": item["question"] + PROMPT_SUFFIX,
            "response": raws[0],
            "answer": answers[0],
            # Persisted so tune_scale.py can grade fresh generations against
            # the gold set rather than against this run's stored answer.
            "aliases": item["aliases"],
            "judge": label,
            "judges": judges,
            **({"task": self.args.task} if self.args.task else {}),
        }}


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # Resume has to skip every question already *attempted*, not just the ones
    # that were kept. Re-rolling a discarded question gives it another chance to
    # pass by luck, and a question that only passes on a second roll is marginal
    # -- precisely what the consistency filter is meant to exclude.
    done = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.update(json.loads(line).keys())
                except Exception:
                    pass
    attempted_path = args.output_path + ".attempted"
    attempted = set()
    if os.path.exists(attempted_path):
        with open(attempted_path, encoding="utf-8") as f:
            attempted = {l.strip() for l in f if l.strip()}
    skip = done | attempted
    if skip:
        print(f"resuming: {len(done)} kept, "
              f"{len(attempted - done)} attempted and discarded, "
              f"{len(skip)} skipped")

    items = [q for q in load_questions(args.data_path, args.max_questions)
             if q["qid"] not in skip]
    print(f"{len(items)} questions to process, "
          f"{len(items) * args.sample_num} generations")

    collector = Collector(args)
    kept = {"true": 0, "false": 0}

    with open(args.output_path, "a", encoding="utf-8") as out, \
            open(attempted_path, "a", encoding="utf-8") as att:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(collector.process, it): it for it in items}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                item = futures[fut]
                rec = fut.result()
                with collector.write_lock:
                    # Recorded whether kept or not, so a restart does not
                    # re-roll it.
                    att.write(item["qid"] + "\n")
                    att.flush()
                    if rec is not None:
                        label = next(iter(rec.values()))["judge"]
                        kept[label] += 1
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out.flush()
                if rec is None:
                    continue

    n_attempted = len(items)
    print(f"kept: {kept['true']} consistently correct, "
          f"{kept['false']} consistently hallucinated "
          f"({(kept['true'] + kept['false']) / max(n_attempted, 1):.0%} yield)")
    total_t = sum(1 for _ in open(args.output_path, encoding="utf-8")
                  if '"judge": "true"' in _)
    total_f = sum(1 for _ in open(args.output_path, encoding="utf-8")
                  if '"judge": "false"' in _)
    print(f"balanced pairs available: {min(total_t, total_f)} "
          f"(cumulative: {total_t} correct, {total_f} hallucinated)")
    if kept["true"] + kept["false"]:
        rate = (kept["true"] + kept["false"]) / max(n_attempted, 1)
        need = 400
        print(f"\nat this yield, {int(need * 2 / max(rate, 0.01)):,} questions "
              f"would be needed for {need} balanced pairs")


if __name__ == "__main__":
    main()
