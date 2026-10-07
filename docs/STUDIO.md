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
| Several GPUs (split mode, tensor split, main GPU, which GPUs) | yes | yes; shown when NVIDIA GPUs are present, with a suggested split; fit estimate counts free VRAM on all of them ([HARDWARE.md](HARDWARE.md#nvidia--cuda)) |
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
| Remote access | LM Link (proprietary) | `--host` + token + TLS, device pairing with per-device tokens, and **linked hosts**: another machine's models served from this Studio |
| Chat with your documents (RAG) | yes | yes; BM25 with no model, or hybrid with any embedding GGUF / endpoint; replies cite passages |
| MCP client: chat can call tools from MCP servers | yes | yes; stdio or streamable HTTP, `mcp.json` in the same format; each call asks first unless auto-approved |
| **Jobs page**: TestQA, deficit datasets, fine-tuning, merge/GGUF, SWE-bench, LiveBench, CLIP_benchmark from the browser | no | yes, at `/jobs` |

The transfer system also ships an MCP *server* (see [TRANSFER.md](TRANSFER.md)).

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

1. The subject classifier turns the last user message into the subjects it
   involves, each weighted (code, math, logic, science, factual, writing,
   vision, graphics, systems, reverse-engineering; several at once when a
   request spans them). When the evidence is thin it says **unknown**, and
   models are ranked on their overall numbers instead. Images restrict the
   candidates to models with a vision projector.
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

**Checking a reply.** With `--cett` and a classifier set for the model, each
assistant reply has a *check* action (and the *check replies* toggle under the
chat runs it after every reply). It runs one extra pass over the reply on the
CPU, scores every token, and shows the result under the reply: how many tokens
crossed the classifier's 0.5 boundary, the peak and mean risk, and the reply
text shaded by risk with the flagged tokens underlined. The check is saved
with the chat and as a trace session in `--traces-dir`
(`~/.neuronscope/traces/<id>`); *open 3D view* shows it at `/viz/<id>/` on
the same Studio, with the same login. `POST /api/trace` with `{model,
messages, text}` does the same from a script; paired devices may call it.
See VISUALIZATION.md, "Which view when", for what to do with a flagged span.

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

## Documents (RAG)

The **Docs** tab holds collections of files (text, Markdown, code, HTML, JSON,
CSV; PDF with `pypdf`, DOCX with `python-docx`). Pick a collection in the
**docs** selector under the chat and each message retrieves the four best
passages, adds them as a numbered system message, and asks the model to cite
them; the passages appear under the reply, expandable, and are saved with the
chat.

Retrieval is BM25 by default, which needs no model and does well on names,
identifiers and error messages. For meaning-level matches add an embedding
model; search then fuses both rankings (reciprocal-rank fusion):

```bash
python viz/studio.py ... --rag-embed-gguf ~/models/nomic-embed-text-v1.5.Q8_0.gguf   # Studio runs it on demand
python viz/studio.py ... --rag-embed http://127.0.0.1:1234/v1@text-embedding-model  # or any /v1/embeddings
```

The embedding sidecar is a second llama-server (`--embedding`, port
`--backend-port + 1`, CPU unless `--rag-embed-ngl`), so it never displaces the
chat model. A collection remembers which embedding model built it and refuses
to mix in vectors from another. Collections live in `~/.neuronscope/rag`
(`--rag-dir`) and work from the command line too:

```bash
python scripts/rag.py add --collection notes docs/*.md
python scripts/rag.py search --collection notes "how does auto routing pick a model"
```

## Tools (MCP)

Studio connects to the MCP servers listed in `~/.neuronscope/mcp.json`
(`--mcp-config`), in the format LM Studio and Claude Desktop use:

```json
{"mcpServers": {
  "files":  {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]},
  "search": {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer ..."}},
  "clock":  {"command": "python", "args": ["clock_server.py"], "autoApprove": ["now"]},
  "old":    {"command": "...", "disabled": true}
}}
```

Tick **tools** under the chat and the model is offered every connected tool
(as `<server>__<tool>`; llama-server's Jinja templates, on by default, turn
them into the model's native tool format). When the model calls one, the call
and its arguments appear in the reply with **Allow** / **Deny**; nothing runs
until you choose, unless the server's `autoApprove` is `true` or lists that
tool. Results are shown under the call and fed back to the model, for up to 8
rounds per message. A declined call is reported to the model as declined.

The **Tools** tab shows each server's status and tools, and reconnects after
you edit the file. Stdio servers' logs go to `~/.neuronscope/mcp-logs/`.
`python scripts/mcp_client.py tools` checks a config from the command line.

MCP servers run with your permissions. Only configure servers you trust, and
keep `autoApprove` to read-only tools.

## Link: paired devices and other machines' models

The **Link** page (header) does two things, both built on one-time pairing
links instead of sharing the master token.

**Pair a device.** *Create pairing link* gives a link usable once, like
`https://studio.lan:7870/pair#c=K7QX-M2PA-9DTE&fp=…`, claimable for five
minutes by default.
Opened in a browser, it pairs that browser and logs it in; on the command line,
`python scripts/ns_pairing.py claim '<link>' --out laptop.token` writes a token
file for API clients. Each device gets its own token, stored on the host only
as a SHA-256 hash, listed with when it was last seen, and revocable at once.
Paired devices can chat (with documents), use `/v1` and keep saved chats.
Everything that administers the machine needs the master token: jobs (they
run code), load settings and llama-server flags, downloads, MCP tools (they
run with the host owner's permissions), pairing, revoking and links.

**Persistent or temporary access.** Each pairing link grants one of:

- **Persistent**: the device keeps access until you revoke it (a paired
  browser's cookie is renewed for a year at a time; revoking still ends it at
  once).
- **Temporary**: access ends a set time after pairing (1 hour, 8 hours, 1 day,
  7 days, 30 days, or a custom duration). The device's requests are refused
  from that moment, and a paired browser's cookie expires with it.

Choose under **Access** before creating the link; the page the device opens
says which it is getting. The device list shows each device's access and time
left, and **change…** extends or shortens temporary access, renews an expired
device (same token, no re-pairing) or makes it persistent. Expired devices stay
listed for 30 days, then drop off. A linked host with temporary access shows
"access expired" when it runs out, without calling the other machine.

The host's owner sets the policy at startup:

| flag | default | |
|---|---|---|
| `--pair-code-ttl` | `5m` | how long a pairing link can be claimed (up to `1d`) |
| `--pair-durations` | `1h,8h,1d,7d,30d` | the temporary choices offered |
| `--pair-max` | `90d` | the longest temporary access any pairing or extension may grant |
| `--pair-default` | `persistent` | what is preselected (`persistent` or a duration) |
| `--no-persistent-pairing` | off | every device expires; persistent access cannot be granted or restored |

```bash
# a shared machine: guests get a day at most, nothing permanent, 2-minute links
python viz/studio.py ... --no-persistent-pairing --pair-durations 1h,8h,1d --pair-max 1d --pair-code-ttl 2m
```

`POST /api/pair/start` takes `{"persistent": true}` or `{"persistent": false, "ttl": "8h"}`;
`POST /api/devices/update` takes the same plus the device `id`.

**Use another machine's models.** On the GPU box's Studio create a pairing
link, and paste it into this Studio's Link page with a name (say `gpu`). Its
models appear in the picker and in `/v1/models` as `gpu:<model id>`; requests
for them, from the chat or from any OpenAI client pointed at this Studio, are
forwarded to that machine with this Studio's device token, streamed back, and
can use this Studio's documents and tools. Stats and `auto` routing stay
per machine.

The link carries the host certificate's SHA-256 fingerprint (`fp=`). The
claiming side checks that the server presents exactly that certificate before
sending the code, and keeps checking it on every later request, so a
self-signed LAN certificate is pinned, not trusted blindly, and a machine in
the middle can neither take the code nor read the traffic. Pairing requires
the host to run with a token; without TLS the link has no fingerprint and
should only be used inside a VPN. Tokens for linked hosts are kept in
`~/.neuronscope/links.json` (0600); devices in `~/.neuronscope/devices.json`.

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

## Projects

`/projects` splits a project into tasks by skill, keeps a plan you approve,
starts worker models on whatever accelerators the machine has (AMD first), and
sends every result back for review. See [DIRECTOR.md](DIRECTOR.md).

## Connect

`/connect` has ready-made settings for Cline, Claude Desktop and other
OpenAI-compatible apps, and Studio serves MCP at `/mcp` so agents can use
NeuronScope's tools. See [CONNECT.md](CONNECT.md).
