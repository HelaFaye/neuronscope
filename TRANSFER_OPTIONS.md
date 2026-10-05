# NeuronScope transfer options

The transfer stack intentionally supports several modes. They solve different network and deployment cases and can coexist.

## 1. WebRTC browser GUI

Use `viz/transfer_lab.py`. Best for ad-hoc peer-to-peer transfers, browser/mobile endpoints, and no-install peers. The signaling server relays control metadata; model bytes stay on the WebRTC data channel.

## 2. Reliable CLI/HTTP mode

Use `scripts/ns_transfer.py`. Best for remote LM Studio machines that should simply receive files. It uses resumable chunked HTTP, per-chunk SHA-256, final-file SHA-256, durable `.part` files, and atomic publication. No Python packages beyond the standard library are required.

Receiver:

```bash
python scripts/ns_transfer.py serve --host 0.0.0.0 --port 8810 --root /models/LMStudio --token 'LONG-RANDOM-TOKEN'
```

Sender:

```bash
python scripts/ns_transfer.py batch --url http://REMOTE:8810 --token 'LONG-RANDOM-TOKEN' --dir models/ornith_sweep --glob '*.gguf' --workers 2
```

For security across untrusted networks, put the receiver behind WireGuard/Tailscale or HTTPS. The bearer token is authentication, not encryption.

## 3. Watch mode

Automatically transfer intermediate GGUFs as the tuning process creates them:

```bash
python scripts/ns_transfer.py watch --url http://REMOTE:8810 --token '...' --dir models/ornith_sweep --glob '*.gguf' --workers 1
```

A JSON state file prevents already transferred files from being resent after restart.

## 4. Adaptive worker mode

Use `scripts/tuning_worker.py` when the remote machine can generate and evaluate models itself. This avoids transferring intermediate GGUFs.

## 5. MCP control plane

Use `scripts/ns_transfer_mcp.py` when an MCP-capable agent should register destinations, probe them, send one file, send a batch, start a background watcher, or pull a completed transfer.

Install the optional MCP dependency:

```bash
python -m pip install -r requirements-transfer-full.txt
```

Run:

```bash
python scripts/ns_transfer_mcp.py
```

Available operations include `register_target`, `list_targets`, `probe_target`, `send_file`, `send_batch`, `start_watch`, `transfer_status`, and `pull_completed`.

## Recommended tuning topology

```text
Local controller
   |
   +-- adaptive worker ----------------> remote worker host
   |
   +-- CLI batch/watch ----------------> remote LM Studio host
   |
   +-- WebRTC GUI ---------------------> ad-hoc peer/mobile
   |
   +-- MCP ----------------------------> automation/control plane
```

Use worker mode when the remote machine can run NeuronScope. Use CLI/watch mode when the remote machine is only an inference/LM Studio appliance. Use WebRTC when you need a browser-based temporary peer. The MCP is transport-neutral and can start any CLI transfer automatically.

## Batch-tuning integration

The adaptive controller can produce candidates into a watched directory. The watcher transfers each completed `.gguf` to the LM Studio machine. You can therefore keep the LLM evaluation loop remote while keeping model generation local, even when the remote machine cannot run the NeuronScope worker.
