# Studio

`viz/studio.py` is a model manager, chat window and OpenAI-compatible server
in front of `llama-server`. It covers the everyday LM Studio workflow and adds
the controls this project exists for.

```bash
python viz/studio.py --models-dir ~/models --server ~/llama.cpp/build/bin/llama-server
python viz/studio.py --models-dir A --models-dir B --idle-ttl 900 --routing qa/routing.example.json
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
| **`model: "auto"`**: route each prompt to the best-suited local model | no | yes, via the subject classifier and a routing table |
| Remote access | LM Link (proprietary) | `--host` + token + TLS |

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
- `"model": "auto"` (only with `--routing`) classifies the last user message
  (code, math, logic, science, factual, writing, vision) and picks the model
  whose skills match best. Fill the skills from real numbers:
  `testqa.py --out` writes per-subject scores under `"skills"`.
- Requests with `image_url` content go only to models with a vision projector;
  others get a 400 rather than a confusing failure.
- `stream: true` is passed straight through as SSE.

Point Cline, Continue, Open WebUI or any OpenAI SDK at `http://host:7870/v1`.

## Runtime controls

**Suppression α.** Load a model with a NeuronScope LoRA (`export_lora.py`) in
the adapter field, then drag the slider. It sets the adapter scale through
llama-server's `/lora-adapters` with no reload, so you can compare behaviour
at several suppression strengths in one conversation.

**Active experts.** For MoE models the load panel exposes
`<arch>.expert_used_count`. See `scripts/sweep_experts.py` for measuring the
quality and speed curve.

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

See [SECURITY.md](SECURITY.md).
