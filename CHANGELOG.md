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
- `install.sh` detects NVIDIA (CUDA) and macOS (MPS) as well as AMD (ROCm) and CPU.

### Fixed
- The WebRTC page sent 1 MiB DataChannel frames, above Chromium's 256 KiB
  SCTP limit, so the channel closed on the first chunk. Frames are now sized
  from `pc.sctp.maxMessageSize`.
- `adaptive_tuner.py tune` crashed because `--token` and `--auto-delete` were
  never defined. The tuning lab GUI dropped the token it collected.
- Studio listed `mmproj-*.gguf` projector files as loadable models.

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
