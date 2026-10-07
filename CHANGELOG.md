# Changelog

## Unreleased

### Added (connect)
- NeuronScope as an MCP server (`scripts/ns_mcp.py`, dependency-free): 26
  tools over Studio's API (models, chat, reply checks, routing, stats,
  devices, doctor, jobs, projects, traces), with read-only and destructive
  hints. Served by Studio at `/mcp` (streamable HTTP; refuses other web
  origins and non-JSON) and as a stdio process. Tested with the official
  `mcp` client over both transports. Every call carries the caller's token.
- Studio's Connect page (`/connect`): settings for Cline (OpenAI-compatible
  provider with a picker of this Studio's models, and the MCP entry), Claude
  Desktop and other OpenAI clients, with copy buttons, and a button that
  creates a revocable app token (a paired device). `scripts/ns_connect.py`
  prints the same and writes Cline's or Claude Desktop's config, keeping a
  backup and other servers. `POST /api/route` and `GET /api/doctor`.
- Fixed: a job log request for an unknown job id returned an empty log
  instead of 404.

### Fixed (from a newcomer dry run of the docs)
- TestQA crashed while printing per-subject results when a subject had no
  right answers.
- `install.sh --no-torch` (new) skips PyTorch for Studio, TestQA, Projects,
  transfer and the llama.cpp path; `requirements-core.txt` holds everything
  that does not need it (`accelerate` used to pull a multi-GB PyTorch in
  silently). A failed PyTorch install now gives advice for this machine
  (CPU, CUDA or ROCm) instead of always pointing at ROCm, and the installer
  ends by pointing at `doctor.py`.
- `doctor.py`: hardware advice from the detected devices (it assumed a Vega
  laptop), a devices section, build hints that use `build_llama_tools.sh` and
  `NS_LLAMA`, stage names instead of bare numbers, and a next step for Studio
  as well as for the pipeline.
- Studio prints the model ids it found at startup (what `"model"` and
  TestQA's `<model-id>` need).
- `cett-dump --help` lists its own flags; `ns_transfer_mcp.py --help` prints
  help instead of waiting on stdin; `--sandbox` without Docker says what to do.
- Docs: a glossary for enthusiasts (`docs/GLOSSARY.md`), how to get a first
  model, a way to verify the first extraction, a plain-language overview of
  the pipeline, the dataset link, the `--allow-exec` / `--sandbox` trade-off in
  the quick starts, and a broken README link.

### Security
- Studio refuses to start a backend on a port something else already holds
  (an orphaned server would otherwise answer the health check).
- Tuning worker results are signed (HMAC-SHA256, key derived from the worker
  token, per-job nonce); the controller refuses unsigned, altered or replayed
  results.
- WebRTC TURN credentials go only to peers inside a room, and with
  `--turn-secret-file` are minted per peer in coturn's REST format and expire.
- TestQA and deficits.py `--sandbox docker|podman`: model-written code runs in
  a container with no network, read-only filesystem, no capabilities, an
  unprivileged user and resource limits.
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

### Changed (visualization)
- Flagging is absolute: a token is flagged when the classifier's probability
  reaches `--flag-prob` (0.5, its own decision boundary), optionally smoothed
  (`--smooth`). Before, a token was flagged when its score was one SD above
  the reply's own mean, so every reply had flags, even clean ones; that is now
  only `--relative-z`, labelled as such. Cells are flagged only for H-neurons
  (an h_neurons.json, or the classifier's positive weights that
  `trace_sample.py --classifier` now stores in the trace); with neither, risky
  tokens are shown and no cell is.
- three.js and Godot show the reply text shaded by risk and a risk-over-time
  strip with the threshold (click either to jump), draw flagged H-neurons as
  rings on top of the field in a colour that keeps its hue under glow, order
  columns by classifier weight so H-neurons form a band, and state the
  flagging mode in the HUD. Godot: `N` jumps to the next flagged token.
- Themes: flagged colours chosen to stay distinct from active colours under
  common colour-vision deficiencies (ember: cyan on amber; cool: orange on teal).
- Fixed: Godot billboards ignored instance scale, so every cell drew at one size.
- Studio: *check* on a reply (or the *check replies* toggle) scores every token
  of it with the model's classifier, shades the reply text by risk, underlines
  flagged tokens, and links to a 3D view of that reply served by Studio at
  `/viz/<id>/`. Checks are saved with the chat and as trace sessions in
  `--traces-dir`. `POST /api/trace`, `GET /api/traces`; `HScorer.trace`.
- Fixed: the GGUF tokenizer decoded SentencePiece byte-fallback tokens
  (`<0xNN>`) to nothing, so characters outside the vocabulary vanished from
  decoded text and trace labels; multi-byte characters now land on the last
  of their byte tokens.
- Default theme is `dark`: a dark field with blue activations and red-orange
  flagged H-neurons (bloom.py, timeline.py, Godot via the API, Studio's 3D
  view, and the Live page). Studio, Jobs and the pairing pages open in dark
  mode unless light was chosen with the toggle.
- Subject classifier: returns every subject above a 30% share instead of one,
  and "unknown" when evidence is thin, which routing (`auto`, `route`,
  `model_stats.py rank`) treats as "use the best model overall". New task
  subjects graphics, systems (build) and reverse-engineering; 158 seed lines
  phrased as tasks rather than quiz questions; keywords match at word starts
  and count once per subject. `evaluate --tasks FILE` scores multi-label task
  files kept outside the repo.
- Director (`scripts/director.py`, Studio `/projects`): a description (and
  optionally a checkout) becomes skill-labelled tasks; a versioned plan you
  approve; after approval the director only proposes changes (follow-ups,
  a different model after repeated rejections) and you accept or reject them;
  each task gets the model whose graded results fit its skills (else the
  default, else the largest that fits, skipping models whose context is too
  short); workers end with a JSON report; results wait for review by policy
  (all, flagged, none); rejected work goes back with your feedback; blocked
  tasks ask questions; errors back off. Owner only.
- Worker models run as separate llama-servers placed by
  `scripts/accelerators.py` on ROCm, Vulkan, CUDA, Metal or the CPU, AMD
  first: per-device llama-server builds, settings, environment and pinned ROCm
  releases (`~/.neuronscope/hardware.json`), per-project device limits and
  settings, APU memory counted as VRAM plus GTT and shared with the CPU, Vega
  APUs (gfx90c) on Vulkan unless ROCm is opted into, automatic
  `HSA_OVERRIDE_GFX_VERSION` for RDNA2/3 small dies, multi-GPU layer splits
  before estimated-memory devices. `--max-workers`, `--worker-idle`, `--hardware`.
- Studio's llama-server command building is shared (`server_cmd`).
- Hub: Replay lists saved traces (Studio checks and `runs/`) to pick from.
- Docs: VISUALIZATION.md "Which view when" and how to read the 3D views.
- Removed tabs with nothing behind them: the planned Recipes tab, and the
  example Eval package is no longer enabled by default.

### Added
- Studio pairing: persistent (until revoked) or temporary access per pairing
  link, with host-configured expirations (`--pair-code-ttl`,
  `--pair-durations`, `--pair-max`, `--pair-default`,
  `--no-persistent-pairing`). Expiry is enforced on every request, browser
  cookies end with it, linked hosts report it, and the owner can extend,
  shorten, renew or make persistent from the Link page.
- `viz/bloom.py --demo`: a synthetic, clearly labelled trace for trying the
  three.js and Godot clients without a model. Godot usage (keys, `NS_FRAME`,
  `NS_PAUSED`, renderer fallback) in docs/VISUALIZATION.md.

### Added (CUDA)
- `scripts/cuda_info.py`: every NVIDIA GPU via nvidia-smi, and the decisions
  that follow: PyTorch wheel (CUDA 12.6 + `torch<2.15` for Maxwell/Pascal/Volta,
  e.g. Tesla M10), llama.cpp CUDA architectures and toolkit limit (CUDA 13
  cannot target sm<75), per-GPU training precision, QLoRA availability,
  multi-GPU split.
- `install.sh` installs the wheel that has kernels for every GPU, and its smoke
  test checks each GPU's kernels and compares an fp32 matmul against the CPU.
- `build_llama_tools.sh --backend cuda` builds for the detected GPUs
  (`--cuda-arch` to override) and refuses a CUDA 13 build for pre-Turing cards;
  `docker/llama-cuda.Dockerfile` builds with CUDA 12.9 (sm_50 by default),
  pinned to the llama.cpp commit the activation patch was tested on, with an
  optional CA secret for TLS-intercepting proxies. Built and tested here: the
  image's llama-server and cett-dump match PyTorch and the host build (CPU
  fallback; no GPU in this environment). The build script now clones shallowly.
- Studio: Visible GPUs, Split, Tensor split and Main GPU per model; the fit
  estimate uses free VRAM across all GPUs; `/api/gpus`.
- `finetune.py` picks precision from compute capability (fp32 on Maxwell and
  consumer Pascal; `is_bf16_supported()` also reports emulated bf16) and
  refuses QLoRA on Maxwell. `doctor.py`, `hostcheck.py`, `vram_budget.py` and
  host profiles see every GPU.

### Fixed
- Godot and three.js clients drew a narrow strip for most traces: neuron columns
  were spaced 0.06 units apart and the camera was fixed. Both now fit the field
  (640 units wide whatever the trace width) and frame it. three.js keeps the
  current token at a fixed depth, so long traces no longer drift past the
  camera.
- Godot HUD never showed FLAGGED (`Array.has` does not equate 146 with the
  JSON float 146.0). `NS_FRAME` / `NS_PAUSED` open the client on one token.
- The three.js viewer says when three.js cannot be loaded instead of showing a
  black page.
- Hub: the Live service can use a patched llama-server (`live source`) instead
  of always simulating; config values that would be read as flags are refused.
- Studio on phones: the chat log is no longer squeezed out by the sidebar.

### Added
- Studio Link: one-time pairing links (certificate fingerprint pinned) give
  browsers and API clients their own revocable device tokens; linked hosts
  serve another machine's models from this Studio as `name:model` in the
  picker and `/v1` (`scripts/ns_pairing.py`).
- Studio MCP client: tools from MCP servers (stdio or streamable HTTP,
  `mcp.json` in LM Studio / Claude Desktop format) offered to the chat model;
  every call is shown with Allow/Deny unless auto-approved; Tools tab.
- Studio RAG: document collections (Docs tab, `scripts/rag.py`), BM25 or hybrid
  retrieval with an embedding GGUF sidecar or any embeddings endpoint, cited
  passages under each reply.
- Studio `/jobs`: evaluation, retraining and benchmark runs from the browser
  (TestQA, deficits, SWE-bench training data, fine-tune, merge/GGUF, SWE-bench,
  LiveBench, CLIP_benchmark). Typed per-job forms with no free-form command
  line, persistent logs, cancel, a concurrency limit; loopback-only unless
  `--allow-remote-jobs`.
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
