# TestQA: graded evaluation, code interpreter, subject classifier

`scripts/testqa.py` asks one bank of graded prompts to any number of
OpenAI-compatible endpoints (LM Studio, llama-server, Studio, vLLM, hosted
APIs). It reports scores per task kind and per subject, and compares each
endpoint item by item against a reference.

```bash
python scripts/testqa.py \
    --endpoint base=http://127.0.0.1:1234/v1@my-model \
    --endpoint supp=http://127.0.0.1:1234/v1@my-model-supp050 \
    --allow-exec --cache runs/testqa-cache --out runs/testqa.json
```

Endpoints are `LABEL=URL[@model]`. `/v1` is appended if missing. `--cache`
stores every reply, so an interrupted run resumes and a grader change can be
re-applied without asking the models again.

## The bank (`qa/bank/*.jsonl`)

| file | kind | n | graded by |
|---|---|---|---|
| `reasoning.jsonl` | `reasoning` | 35 | final `Answer:` line; numbers with tolerance (fractions, `$`, `1,000`, `\frac{a}{b}` accepted), choices, or normalised text |
| `coding.jsonl` | `code_exec` | 18 | **the interpreter**: the reply's code runs against unit tests |
| `factual.jsonl` | `qa` | 17 | gold aliases; `expect_abstain` items reward "I don't know" |
| `canary.jsonl` | `canary` | 7 | ungraded: answered vs refused |

Every task has a `subject` (code, math, logic, science, factual, writing,
vision). Add your own files in the same format, or point `--tasks` at another
directory. Two kinds from `merge_eval.py` work as well: `code`
(`{"modules": [...]}`, every referenced symbol must exist) and plain `qa`.

Reasoning prompts get a suffix asking for a final `Answer: <value>` line.
Reasoning traces in `<think>` or `reasoning_content` are stripped before
grading.

Canaries catch an edit that improves the target metric by making the model
refuse more. Watch "canary newly refused" in the comparison.

## The interpreter

`code_exec` tasks give an entry point and assertions:

```json
{"id": "c-fib", "kind": "code_exec", "subject": "code", "entry_point": "fib",
 "prompt": "Write a Python function `fib(n)` ...",
 "tests": ["assert fib(10) == 55", "assert fib(300) % 1000 == 600"]}
```

The first fenced code block (or the whole reply, if it parses) runs in a fresh
`python -I` process with the tests appended. Verdicts: `correct`, `wrong`
(with the failing assertion), `timeout`, `unparsable`, `abstained`.

**It executes model-written code, so it is off unless `--allow-exec` is
given.** Without the flag, code is only compiled and checked for the entry
point (`skipped`). With the flag, each run gets an empty temp directory, a
stripped environment (no API keys), stdin closed, a wall-clock timeout and,
on POSIX, CPU, address-space and file-size rlimits. That contains accidents
but is not a security sandbox. For untrusted models, run TestQA inside a
container or VM with networking disabled.

Every bundled coding task is verified against a reference solution in
`tests/test_testqa.py`. Keep doing that for new tasks: that test found a wrong
expected value in this bank before it could mis-grade a model.

## Reading the output

```
score by kind (fraction correct/answered, graded n)
                        base          supp
  code_exec       0.83 (18)     0.78 (18)
  reasoning       0.71 (35)     0.69 (35)
  ...
supp vs base: gained 3, regressed 5, net -2, chi2=0.5 (not significant); canary newly refused 1
```

The comparison is a paired McNemar test on items whose verdict changed. Below
10 discordant items it says "too few changes to call". Read the regression
list, not just the net number.

## Subject classifier and routing

`scripts/subject_classifier.py` labels prompts by subject. It is a
dependency-free naive Bayes with keyword priors, trained in milliseconds on
the bank plus `qa/subject_seed.jsonl`.

```bash
python scripts/subject_classifier.py evaluate          # leave-one-out accuracy (~0.75 on 147 prompts)
python scripts/subject_classifier.py predict "Fix this segfault in my C++ loop"
python scripts/subject_classifier.py route --table qa/routing.example.json "What is in this photo?"
```

It has three uses:

1. TestQA labels any task without a `subject`, so per-subject tables work on
   your own task files.
2. Routing: a table maps model names to per-subject skill scores, and `route`
   picks the best fit for a prompt. Studio uses this for `"model": "auto"`
   (see [STUDIO.md](STUDIO.md)). Fill the skills from measurements: the
   `"skills"` block in `testqa.py --out` is exactly that table's shape.
3. It is the seed for choosing which model to load for a job based on its
   measured knowledge. Labelled lines added to `qa/subject_seed.jsonl` sharpen
   it, and `evaluate` tells you whether they did.

It is a router hint, not an oracle. At about 75% accuracy, a wrong route
costs you a weaker model, not a wrong answer, which is the right failure mode.
