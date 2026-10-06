# Studio

`viz/studio.py` is a model manager, chat window and OpenAI-compatible server
in front of `llama-server`. It covers the everyday LM Studio workflow and adds
the controls this project exists for.

```bash
python viz/studio.py --models-dir ~/models --server ~/llama.cpp/build/bin/llama-server
python viz/studio.py --models-dir A --models-dir B --idle-ttl 900 --min-graded 20
```

Open http://127.0.0.1:7870. Without `--models-dir` it scans LM Studio's own
model folders (`~/.lmstudio/models`, `~/.cache/lm-studio/models`), so both
apps share one library.

Studio does not implement inference, tokenization or sampling; llama.cpp does
that. Studio handles everything around it.

## Feature map

| | LM Studio | Studio |
|---|---|---|
| Model library with metadata (arch, quant, MoE, size, fit) | yes | yes; fit is estimated against this host before loading |
| Hugging Face search and resumable download | yes | yes; multi-part GGUFs offered as one set; `HF_TOKEN` for gated repos |
| Per-model load settings (GPU layers, context, batch, threads, flash-attn, KV cache type) | yes | yes; persisted per file |
| Chat with streaming, reasoning shown separately | yes | yes |
| Saved conversations, export | yes | yes; one JSON file per chat in `~/.neuronscope/chats`, Markdown export |
| Image input for vision models | yes | yes; an `mmproj-*.gguf` beside the model is loaded automatically |
| Presets (system prompt, sampling) | yes | yes; includes JSON-schema and GBNF constrained output |
| OpenAI-compatible local server | yes | yes, at `/v1` on the same port |
| Just-in-time model loading on API requests | yes | yes; a swap waits for in-flight requests to finish |
| Idle auto-unload (TTL) | yes | `--idle-ttl SECONDS` |
| Speculative decoding | yes | draft model, `--draft-max/min` |
| Tokens/s and time to first token per reply | yes | yes |
| **Live suppression α** (LoRA scale, no reload) | no | yes |
| **MoE active-expert override** | no | yes, key read from the model's architecture |
| **`model: "auto"`**: route each prompt to the best-suited local model | no | yes, on each model's measured per-subject accuracy, hallucination and abstention rates |
| **Rolling per-model stats** (graded answers, H-Neuron activations, live abstentions) | no | yes; models without stats are flagged ⚠ |
| Remote access | LM Link (proprietary) | `--host` + token + TLS |
| **Jobs page**: TestQA, deficit datasets, fine-tuning, merge/GGUF, SWE-bench, LiveBench, CLIP_benchmark from the browser | no | yes, at `/jobs` |

Not included: RAG and an MCP client. Both are project-sized. The transfer
system ships its own MCP *server* (see [TRANSFER.md](TRANSFER.md)).

## OpenAI-compatible API

```bash
curl http://127.0.0.1:7870/v1/models
curl http://127.0.0.1:7870/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-8b-gguf/qwen3-8b-q6_k", "messages": [{"role": "user", "content": "hi"}]}'
```

- Model ids are `<repo folder>/<file stem>`, lower-cased. A served name set in
  the load panel also works, as do the file name or full path.
- A request that names a model other than the loaded one loads it (JIT).
  `--no-jit` makes such requests fail instead.
- `"model": "auto"` picks a model from measured performance. See below.
  Naming a model is always a manual override, whether or not it has stats.
- Requests with `image_url` content go only to models with a vision projector;
  others get a 400 rather than a confusing failure.
- `stream: true` is passed straight through as SSE.

Point Cline, Continue, Open WebUI or any OpenAI SDK at `http://host:7870/v1`.

## How `auto` chooses

1. The subject classifier turns the last user message into a probability per
   subject (code, math, logic, science, factual, writing, vision). Images
   restrict the candidates to models with a vision projector.
2. **Only models with at least `--min-graded` (default 20) graded results
   compete.** A model with no stats is never auto-selected. You can still load
   it, pick it in the chat's model menu, or name it in an API request.
3. For each candidate and subject, the utility is
   `lower 95% bound of accuracy − cost × upper 95% bound of hallucination rate`.
   Abstaining scores zero, so a model that declines when unsure beats one that
   guesses wrong. `--hallucination-cost` sets the cost (default 1; 0 ranks on
   raw accuracy). Bounds rather than point estimates mean a model with more
   evidence wins over a lucky small sample.
4. A subject a model has fewer than 5 results for borrows its overall rates,
   with confidence bounds as wide as 5 observations. Measured evidence beats
   unverified transfer.
5. The probability-weighted utilities are compared and the best model is
   loaded (JIT). If none qualify, the request fails with 409 and says why.

The chat window's model menu offers **Loaded model**, **Auto (by performance
stats)** and every local model. Models without enough stats show an orange ⚠
there and in the model list; hovering explains why and how to add stats. Each
reply notes which model answered, and for auto picks, why (e.g.
`auto → coder-7b: subject code (97%) -> expected +0.56`).

## Rolling stats

Stats live in `~/.neuronscope/stats/<model id>.jsonl` (append-only) and are
tied to the model *file*: re-downloading or re-quantizing under the same name
starts from zero. Summaries count only the last `--stats-window` (default
500) observations of each kind, and nothing older than 180 days.

| kind | source | used by auto? |
|---|---|---|
| graded: right / hallucinated / abstained, per subject | `testqa.py --publish-stats` or `--record-stats` | **yes** |
| activation: H-Neuron classifier score per reply | Studio, for models with a classifier set in the load panel, when `--cett` (or `$NS_CETT`) is available | no, shown only |
| live: replies and how often the model declined | every chat and `/v1` reply | no, shown only |

Activation scoring runs one extra prefill per reply in the background, after
requests finish, on the CPU by default (`--score-ngl` to change). Use
`--score-every N` to sample. It is shown but not used for ranking, because
the classifier is a weak signal on whole replies (AUROC around 0.7); graded
answers are the evidence `auto` trusts.

```bash
python scripts/model_stats.py show                               # every model's rolling summary
python scripts/model_stats.py rank "Fix this segfault in my loop"  # what auto would pick, and why
curl http://127.0.0.1:7870/api/stats
```

## Runtime controls

**Suppression α.** Load a model with a NeuronScope LoRA (`export_lora.py`) in
the adapter field, then drag the slider. It sets the adapter scale through
llama-server's `/lora-adapters` with no reload, so you can compare behaviour
at several suppression strengths in one conversation.

**Active experts.** For MoE models the load panel exposes
`<arch>.expert_used_count`. See `scripts/sweep_experts.py` for measuring the
quality and speed curve.

## Jobs

`/jobs` (the **Jobs** link in the header) starts evaluation, retraining and
benchmark runs without a terminal: TestQA, deficit datasets, SWE-bench training
data, fine-tuning, merge and GGUF conversion, SWE-bench predict / evaluate /
import, LiveBench and CLIP_benchmark. Each job is a fixed script with a form of
typed fields; there is no free-form command line, and no value may start with
`-`, so a field cannot add flags. Output streams into the page and is kept in
`~/.neuronscope/jobs/<id>/` (`--jobs-dir`); jobs keep running when the browser
closes, and can be cancelled. `--max-jobs` (default 2) limits how many run at
once.

A typical retraining loop on one page: TestQA with a cache → Deficit dataset
→ Fine-tune → Merge and convert → load the new GGUF in Studio → TestQA again
on `holdout_ids.json`. See [RETRAINING.md](RETRAINING.md).

Jobs can train models and run model-written code, so they are available on a
loopback bind only, unless Studio is started with `--allow-remote-jobs`
(still behind the token). `--no-jobs` turns them off entirely.

## Security

Studio loads models, downloads files and runs inference, so binding beyond
loopback is gated:

```bash
python scripts/ns_security.py token --out ~/.config/neuronscope/studio.token
python scripts/ns_security.py selfsigned --host studio.lan --host 192.0.2.10
python viz/studio.py --host 0.0.0.0 --token-file ~/.config/neuronscope/studio.token \
    --tls-cert ns-cert.pem --tls-key ns-key.pem
```

- A non-loopback bind without a token refuses to start.
- A non-loopback bind without TLS needs `--allow-plaintext`, meaning the port
  is only reachable through a VPN or a TLS reverse proxy.
- The browser login sets an HttpOnly, SameSite=Strict cookie (Secure under TLS).
  API clients send `Authorization: Bearer <token>`.
- Failed logins are throttled per address. Request bodies are bounded.
- Jobs are off on a network bind unless `--allow-remote-jobs` is given, and
  only accept JSON requests, so a cross-site form cannot start one.

See [SECURITY.md](SECURITY.md).
