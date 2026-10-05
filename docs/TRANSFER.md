# Moving models between machines

Model files are large and tuning often spans machines: a GPU box generates
candidates, an LM Studio host serves them, a laptop drives both. There are
four transfer paths. They solve different problems and can be combined.

| Path | Use when | Encryption | Authentication |
|---|---|---|---|
| WebRTC browser GUI | ad-hoc, two people, no install on the far side | always (DTLS) | 128-bit invite key; verification code compares DTLS fingerprints |
| Resumable CLI / HTTP(S) | a receiving host that should just accept files | TLS when started with a cert, otherwise a VPN | 256-bit bearer token |
| Tuning worker | the remote host can build and evaluate candidates itself | same as CLI | same as CLI |
| MCP control plane | an agent should drive transfers | local stdio | tokens held in the OS keyring / 0600 files |

Read [SECURITY.md](SECURITY.md) before exposing any of them beyond localhost.

```bash
pip install -r requirements-transfer-full.txt     # aiohttp, mcp>=2.3,<3, keyring, cryptography
python scripts/ns_transfer_router.py --help        # one entry point for all modes
```

## 1. WebRTC GUI (peer to peer)

```bash
python scripts/ns_security.py selfsigned --host transfer.lan --host 192.0.2.10   # once
python viz/transfer_lab.py --host 0.0.0.0 --tls-cert ns-cert.pem --tls-key ns-key.pem
```

Open the printed URL on both machines. The first browser creates a room and
shows a short **room code**. **Copy invite link** produces a URL whose `#r=…&k=…` fragment carries the room code and a
128-bit **invitation key**. Fragments are never sent to the server in HTTP
requests. Opening the link joins directly. Typing only the code sends a join
request that someone already in the room must **Admit**, which works for
reading the code aloud.

When peers connect, each side shows a six-digit **verification code**
derived from both DTLS certificate fingerprints. Compare it out of band. If
it matches, nothing (including the signaling server) is in the middle of the
encrypted channel.

Files stream through a DataChannel with backpressure, straight to a folder
you choose (File System Access API; Chromium-based browsers, HTTPS or
localhost). Each file is SHA-256 hashed while streaming. The receiver checks
the hash and discards the file on mismatch.

**Connectivity**:

| mode | flag / selector | trade-off |
|---|---|---|
| Direct | `--ice-mode direct` (default) | fastest; STUN reveals each peer's public IP to the other |
| Relay | `--ice-mode relay --turn-url turns:turn.example:5349 --turn-user … --turn-credential …` | all traffic via your TURN server; peers never learn each other's address |
| LAN / VPN | `--ice-mode lan` or `--no-stun` | host candidates only; for WireGuard / Tailscale / same network |

The server relays only signaling (SDP, ICE). It never receives file bytes. It
enforces an Origin allowlist (`--allowed-origin` for reverse-proxy hostnames),
throttles wrong keys and knocks per address, caps rooms at four peers, and
sends CSP / no-referrer / frame-deny headers.

## 2. Resumable CLI / HTTP(S)

Receiver, e.g. on the LM Studio host:

```bash
python scripts/ns_security.py token --out ~/.config/neuronscope/transfer.token
python scripts/ns_transfer.py serve --host 0.0.0.0 --port 8810 \
    --root ~/.lmstudio/models/neuronscope \
    --token-file ~/.config/neuronscope/transfer.token \
    --tls-cert ns-cert.pem --tls-key ns-key.pem
```

Sender:

```bash
export NS_TRANSFER_TOKEN="$(cat transfer.token)"     # or --token-file
python scripts/ns_transfer.py send  --url https://lmstudio-host:8810 --cafile ns-cert.pem --file model.gguf
python scripts/ns_transfer.py batch --url https://lmstudio-host:8810 --cafile ns-cert.pem --dir models/sweep --workers 2
python scripts/ns_transfer.py watch --url https://lmstudio-host:8810 --cafile ns-cert.pem --dir models/sweep
python scripts/ns_transfer.py status --url https://lmstudio-host:8810 --cafile ns-cert.pem
python scripts/ns_transfer.py pull  --url https://lmstudio-host:8810 --cafile ns-cert.pem --id <id> --out back/
```

How it works:
- Chunks of up to 8 MiB each carry their own SHA-256. The whole file is hashed
  again before it is published.
- Interrupted transfers resume from the receiver's offset. A dropped chunk is
  retried with backoff after asking the receiver what it already has.
- Files are published with an atomic rename into `--root[/subdir]`. LM Studio
  never sees a partial file.
- `watch` waits for a file's size and mtime to settle before sending it, and
  remembers what it has already sent across restarts.
- `pull` verifies the downloaded hash.

Without `--tls-cert`, a non-loopback bind refuses to start unless
`--allow-plaintext` is given. Use that flag only when the port is reachable
solely through WireGuard/Tailscale or a TLS reverse proxy.

## 3. Adaptive tuning worker

When the remote host can run NeuronScope itself, ship the source model once
and let it generate and evaluate candidates locally. Only small JSON job
messages cross the network:

```bash
# remote host
python scripts/tuning_worker.py --host 0.0.0.0 --port 8799 \
    --root /models/tuning --source /models/model.Q6_K.gguf --profile /models/h_neurons.json \
    --suppressor scripts/suppress_gguf.py \
    --evaluator 'python my_eval.py --model "{model}" --scale "{scale}"' \
    --token-file worker.token --tls-cert ns-cert.pem --tls-key ns-key.pem

# controller
python scripts/adaptive_tuner.py tune --worker https://remote:8799 --cafile ns-cert.pem \
    --token-file worker.token --state runs/adaptive-state.json \
    --min 0.2 --max 0.8 --batch-size 3 --margin-of-error 0.02 --baseline-score 0.91 --auto-delete
python viz/adaptive_tuning_lab.py          # same thing with a GUI
```

The worker records the SHA-256 of the source model, the profile and every
candidate. Deletion is limited to generated `.gguf` files inside its root.
The evaluator command comes only from the worker's own command line and is
never accepted over the API. See [LABS.md](LABS.md) for the search strategy.

## 4. MCP control plane

```bash
python scripts/ns_transfer_mcp.py        # stdio MCP server; requires mcp>=2.3,<3
```

Tools: `register_target`, `remove_target`, `list_targets`, `probe_target`,
`send_file`, `send_batch`, `start_watch`, `transfer_status`,
`pull_completed`.

- `register_target` moves the token into the OS keyring (with `keyring`
  installed) or `~/.config/neuronscope/secrets/<name>.token` (0600). The
  target state file is 0600 and holds no secrets. State from older versions
  with inline tokens is migrated on first load.
- The token reaches the CLI through the child's environment, never argv. It
  is redacted from any output and no tool returns it.
- The server speaks stdio only and opens no network listener.

## Recommended topology

```text
controller (laptop)
   ├── tuning worker over HTTPS ─────────► GPU host (builds + evaluates candidates)
   ├── ns_transfer watch over HTTPS/VPN ─► LM Studio host (receives finished GGUFs)
   ├── WebRTC GUI ───────────────────────► ad-hoc peer
   └── MCP (stdio) ──────────────────────► the agent driving all of the above
```
