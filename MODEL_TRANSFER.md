# NeuronScope Model Transfer

NeuronScope Model Transfer is a cross-platform, browser-based P2P transfer surface intended for large GGUF/model files.

## Architecture

```text
Browser A ──WebSocket signaling──> NeuronScope transfer server <──WebSocket signaling── Browser B
    \                                                          /
     └──────── WebRTC DataChannel / DTLS / SCTP (file bytes) ─┘
```

The Python service relays room membership and WebRTC offer/answer/ICE metadata. File bytes are sent by the browsers over an encrypted WebRTC data channel; the Python service does not receive the file payload.

## Features in this version

- 8-character room codes and share links.
- Multi-peer room support (small mesh, capped at four peers).
- Browser-to-browser file transfer with 1 MiB chunks and data-channel backpressure.
- Direct-to-disk receiving through the File System Access API for large GGUF files.
- Pending-offer review before any file bytes are sent.
- LAN-accessible server mode with printed local URLs.
- WebRTC STUN configuration in the browser for internet/NAT traversal where possible.
- No TURN relay is bundled; networks that cannot establish direct WebRTC connectivity need a future TURN option or a LAN/VPN path.

## Launch

```bash
python -m pip install -r requirements-model-transfer.txt
python viz/transfer_lab.py --host 0.0.0.0 --port 8798
```

Open the printed LAN URL. On the receiving machine, open the same URL and use the room code.

For large model files, Chromium/Edge is recommended on the receiving side because `showDirectoryPicker()` permits direct streaming into a selected folder without buffering the complete GGUF in RAM.

## Security model

The room code controls signaling rendezvous; the receiver must explicitly accept an offer before file bytes are sent. WebRTC provides the encrypted data channel. This implementation does not claim peer identity/authentication, so room codes and any future remote signaling deployment should be treated as rendezvous credentials rather than identity proof.

Do not expose the signaling endpoint directly to the public internet without adding HTTPS/WSS, authentication/rate limiting, and a hardened deployment boundary.

## Relationship to Warp

This feature was implemented as an independent NeuronScope subsystem after reviewing the uploaded Warp source and its documented WebRTC/signaling architecture. No Warp source files are vendored into NeuronScope.
