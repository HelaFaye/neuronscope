# Changelog

## Unreleased

### Security
- One bind policy for every network service (`scripts/ns_security.py`). A
  non-loopback bind requires a token of at least 32 characters, and TLS unless
  `--allow-plaintext` is passed. Tokens are compared in constant time, can come
  from `--token-file` or the environment, and argv use prints a warning.
  `ns_security.py token` and `selfsigned` generate a 256-bit token and a LAN cert.
- Transfer receiver and tuning worker: built-in TLS, bounded request bodies
  (JSON 64 KiB, chunks 8 MiB), socket timeouts, failed-auth throttling, 128-bit
  transfer ids, a cap on active transfers, a disk-space check, verified `pull`,
  and source/profile/candidate SHA-256 per tuning job.
- WebRTC signaling: a 128-bit invitation key per room, carried in the link
  fragment; knock/admit for code-only joins; join throttling; Origin
  allowlist; TLS/WSS; relay-only and LAN ICE modes; security headers.
- WebRTC page: a verification code over both DTLS fingerprints, streaming
  per-file SHA-256 verified by the receiver, and a `randomUUID` fallback.
- MCP: `mcp>=2.3,<3`, checked at startup. Tokens live in the OS keyring or
  0600 files, are passed via the environment instead of argv, and are
  redacted from output. Legacy inline tokens are migrated.
- Studio: same bind policy, login throttling, Secure cookie under TLS, bounded bodies.

### Added
- `benchmarks.py swebench train-data`: retraining data from SWE-bench's train
  split (gold patches, DPO against the model's own patches, failure modes
  prioritised by a test-run report, test-set exclusion, replay anchors).
- Vision retraining: deficits.py now writes `sft_vision.jsonl` /
  `dpo_vision.jsonl` with rendered images for failed vision items, and
  `vision_synth.py` adds variations whose answers are known by construction
  (no teacher). `finetune.py --vision [--freeze-projector]` trains VLMs with
  their processor (vision encoder frozen); `merge_export.py --vision` writes the
  language GGUF and the mmproj where llama.cpp can convert it.
- Live activations from llama-server, compiled and tested:
  `llama-tools/server-activations/apply_patch.py` (or
  `build_llama_tools.sh --server-activations`) patches current llama.cpp to
  serve `GET /activations` over SSE, one frame per generated token. Frames
  match PyTorch CETT on every layer in an integration test.
  `export_classifier_bin.py --gguf` folds the down_proj column norms into the
  classifier, so streamed scores are on the classifier's own scale. Fixes found
  by compiling: current llama.cpp names the down projection `ffn_out-N`, and the
  SSE subscription guard was destroyed as soon as it was built.
- External benchmarks (docs/BENCHMARKS.md): `clip_bench.py` runs CLIP/SigLIP
  and H-Neuron-edited variants through CLIP_benchmark (ImageNetV2,
  ImageNet-Sketch, VTAB via task_adaptation, ...) with selective metrics per
  scale, and exports ImageFolders for `clip_neurons.py`; `benchmarks.py`
  drives LiveBench and SWE-bench (prediction, Docker harness, import) into
  rolling stats, and imports published leaderboard scores (BenchLM) as
  reference-only stats shown in Studio.
- Retraining on deficits: `deficits.py` (categorised, grader-verified SFT and
  DPO data, verified synthetic expansion, replay buffer, holdout),
  `finetune.py` (QLoRA / LoRA / full, SFT then DPO, DDP or FSDP via
  `--launch`), `merge_export.py` (merge + GGUF + quantize, or GGUF LoRA);
  `testqa.py --only-ids`. See docs/RETRAINING.md.
- `scripts/build_llama_tools.sh` builds llama.cpp with cett-dump for any
  backend; `tests/test_llamacpp_integration.py` checks the compiled tool
  against PyTorch on a tiny converted model.
- CLIP/SigLIP H-Neurons (`clip_neurons.py`: collect, extract, evaluate,
  export) and `suppress_mmproj.py` for llama.cpp vision projectors.
- TestQA (`testqa.py`) with a 77-task bank: reasoning, executable coding through
  an opt-in resource-limited interpreter, factual QA, and canaries. Scores per
  kind and per subject, with paired comparison and a reply cache.
- Subject classifier and router (`subject_classifier.py`).
- Studio: OpenAI-compatible `/v1` API with JIT loading, `--idle-ttl`, and
  `model: "auto"` routing. Vision models are paired with their mmproj
  automatically and accept image attachments. Also adds saved chats with
  Markdown export, tokens/s and time to first token, stop/regenerate, and
  dark mode.
- Adaptive Tuning lab rebuilt: labelled form, resumable state path, CA file,
  score-by-scale chart with confidence intervals, baseline and next interval,
  results table, light/dark.
- `tests/test_ui_pages.py` syntax-checks every embedded page's JavaScript and
  the pages as actually served (needs `node`).
- `install.sh` detects NVIDIA (CUDA) and macOS (MPS) as well as AMD (ROCm) and CPU.

- Rolling per-model stats (`model_stats.py`): graded right/hallucinated/
  abstained per subject from TestQA, H-Neuron activation scores
  (`hscore.py`) and live abstentions; tied to the model file.
- Studio's `model: "auto"` ranks only models with enough graded results, on
  pessimistic per-subject utility; manual choices are always honoured. UI
  shows an orange ⚠ with an explanation for models without stats, stats lines
  and tables, and the reason behind each auto pick.
- TestQA bank reorganised by subject (210 graded items, 30 per subject):
  new logic, science, factual (including false-premise) and code
  (API-existence) items; writing graded by mechanical `constraints`
  checks; vision items with deterministically generated images
  (`qa_images.py`). `--per-subject`, `--list`, a per-subject
  right/hallucinated/abstained report with CIs, and `--record-stats` /
  `--publish-stats`.

### Fixed
- `extract_activations_gguf.py` measured the prompt with the transformers
  tokenizer on one path, and fell back to counting decoded characters when
  the prompt/response boundary token merged. It now always tokenizes the
  prompt with the same binary and uses the common token-id prefix; verified
  end to end against PyTorch on a converted model.
- `cett-dump` silently dropped the last decoder layer: llama.cpp computes the
  final layer's FFN only for tokens that produce logits, so that layer was
  never seen. Every token is now an output (`--last-layer-outputs-only`
  restores the old behaviour). Found by compiling it for the first time;
  it now matches the PyTorch reference to f16 precision on every layer.
- `hscore.py` located the response by decoded character length, which broke
  on byte-fallback tokens; it now compares prompt and full token ids.
- `delegate.py`'s H-Neuron gate called `/api/score` on the tracing proxy, which
  never implemented it, so every gate check failed as "unreachable". The proxy
  now implements it (`autotrace.py --classifier`).
- The WebRTC page sent 1 MiB DataChannel frames, above Chromium's 256 KiB
  SCTP limit, so the channel closed on the first chunk. Frames are now sized
  from `pc.sctp.maxMessageSize`.
- `adaptive_tuner.py tune` crashed because `--token` and `--auto-delete` were
  never defined. The tuning lab GUI dropped the token it collected.
- Studio listed `mmproj-*.gguf` projector files as loadable models.
- Quantization Lab page never initialised: the server injected the models
  directory into a `<script>` unquoted, and never filled the profiles
  placeholder (`__PROFILES__ is not defined`). Values are now JSON-encoded.
- Scale Sweep Lab page had a JavaScript syntax error (a stray `${''}`), so
  none of its controls worked.
- `mcp` 2.x renamed `FastMCP` to `MCPServer`; the MCP server imports either.

### Changed
- Documentation reorganised into `docs/`; machine-specific paths, IPs and
  model names replaced with generic examples. `env.sh` no longer embeds a
  volume UUID or model path (use `env.local.sh`).
- Removed stray empty files and an editor config with a personal path, and
  stopped tracking `delegation.json` (use `scripts/delegation.example.json`).
  `autotune.sh` became the parameterised `scripts/lmstudio_scale_compare.sh`.

## 2026-10-02
- Split torch-free helpers into `ns_constants.py`; single GGUF metadata path in
  `gguf_utils.py`; validated profile selections and geometry; classifier
  train/test overlap guard; fixed the GGUF extractor's per-sample prompt fallback;
  made tokenizer/extractor subprocess failures fatal.
