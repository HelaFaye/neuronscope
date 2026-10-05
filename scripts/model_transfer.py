#!/usr/bin/env python3
"""NeuronScope Model Transfer: browser/WebRTC signaling and file-transfer UI server.

This server only handles signaling/control metadata. File bytes are transferred
peer-to-peer over WebRTC DataChannels in the browser.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web

CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
CODE_LEN = 8
ROOM_TTL = 15 * 60
MAX_PEERS = 4
MAX_SIGNAL_BYTES = 128 * 1024

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
INDEX = WEB_ROOT / "model_transfer.html"


def new_room_code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LEN))


def local_addresses() -> list[str]:
    found: list[str] = []
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        for info in infos:
            addr = info[4][0]
            if addr.startswith("127.") or addr in found:
                continue
            found.append(addr)
    except OSError:
        pass
    return found


@dataclass
class Peer:
    ws: web.WebSocketResponse
    peer_id: str
    room: str
    connected_at: float = field(default_factory=time.time)


@dataclass
class Room:
    code: str
    peers: dict[str, Peer] = field(default_factory=dict)
    last_empty_at: float | None = None

    def expired(self, now: float) -> bool:
        return not self.peers and self.last_empty_at is not None and now - self.last_empty_at > ROOM_TTL


class RoomManager:
    def __init__(self) -> None:
        self.rooms: dict[str, Room] = {}
        self.lock = asyncio.Lock()

    async def create_or_join(self, requested: str | None, ws: web.WebSocketResponse) -> tuple[Room, Peer]:
        async with self.lock:
            self._gc_locked()
            code = requested or self._unique_code_locked()
            room = self.rooms.get(code)
            if room is None:
                if requested:
                    # A client may join a room only if it currently exists.
                    raise ValueError("room-not-found")
                room = Room(code)
                self.rooms[code] = room
            if len(room.peers) >= MAX_PEERS:
                raise ValueError("room-full")
            peer = Peer(ws=ws, peer_id=secrets.token_hex(8), room=code)
            existing = list(room.peers.values())
            room.peers[peer.peer_id] = peer
            room.last_empty_at = None
            return room, peer

    async def leave(self, peer: Peer) -> list[Peer]:
        async with self.lock:
            room = self.rooms.get(peer.room)
            if room is None:
                return []
            room.peers.pop(peer.peer_id, None)
            others = list(room.peers.values())
            if not room.peers:
                room.last_empty_at = time.time()
            self._gc_locked()
            return others

    async def peers(self, room: str) -> list[Peer]:
        async with self.lock:
            rec = self.rooms.get(room)
            return list(rec.peers.values()) if rec else []

    def _unique_code_locked(self) -> str:
        while True:
            code = new_room_code()
            if code not in self.rooms:
                return code

    def _gc_locked(self) -> None:
        now = time.time()
        for code, room in list(self.rooms.items()):
            if room.expired(now):
                self.rooms.pop(code, None)


async def send_json(ws: web.WebSocketResponse, payload: dict[str, Any]) -> None:
    if not ws.closed:
        await ws.send_str(json.dumps(payload, separators=(",", ":")))


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "neuronscope-model-transfer", "p2p": True})


async def index(_: web.Request) -> web.StreamResponse:
    return web.FileResponse(INDEX)


async def static_file(request: web.Request) -> web.StreamResponse:
    rel = request.match_info["path"].lstrip("/")
    candidate = (WEB_ROOT / rel).resolve()
    if WEB_ROOT.resolve() not in candidate.parents:
        raise web.HTTPNotFound()
    if not candidate.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(candidate)


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=MAX_SIGNAL_BYTES)
    await ws.prepare(request)
    manager: RoomManager = request.app["rooms"]
    current: Peer | None = None

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                raw = msg.data
                if len(raw.encode("utf-8")) > MAX_SIGNAL_BYTES:
                    await send_json(ws, {"type": "error", "error": "message-too-large"})
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await send_json(ws, {"type": "error", "error": "bad-json"})
                    continue

                kind = data.get("type")
                if kind == "ping":
                    await send_json(ws, {"type": "pong"})
                    continue

                if kind == "join":
                    if current is not None:
                        await send_json(ws, {"type": "error", "error": "already-joined"})
                        continue
                    requested = data.get("room")
                    if requested is not None and not isinstance(requested, str):
                        await send_json(ws, {"type": "error", "error": "bad-room"})
                        continue
                    try:
                        room, current = await manager.create_or_join(requested, ws)
                    except ValueError as exc:
                        await send_json(ws, {"type": "error", "error": str(exc)})
                        continue
                    existing = [p for p in room.peers.values() if p.peer_id != current.peer_id]
                    await send_json(ws, {
                        "type": "joined",
                        "selfId": current.peer_id,
                        "room": room.code,
                        "peers": [p.peer_id for p in existing],
                    })
                    for p in existing:
                        await send_json(p.ws, {"type": "peer-joined", "peerId": current.peer_id})
                    continue

                if kind == "signal":
                    if current is None:
                        await send_json(ws, {"type": "error", "error": "not-joined"})
                        continue
                    target = data.get("to")
                    signal = data.get("data")
                    if not isinstance(target, str) or not isinstance(signal, dict):
                        await send_json(ws, {"type": "error", "error": "bad-signal"})
                        continue
                    peers = await manager.peers(current.room)
                    dest = next((p for p in peers if p.peer_id == target), None)
                    if dest is not None:
                        await send_json(dest.ws, {"type": "signal", "from": current.peer_id, "data": signal})
                    continue

                await send_json(ws, {"type": "error", "error": "unknown-type"})
            elif msg.type == web.WSMsgType.ERROR:
                break
    finally:
        if current is not None:
            others = await manager.leave(current)
            for p in others:
                await send_json(p.ws, {"type": "peer-left", "peerId": current.peer_id})

    return ws


def build_app() -> web.Application:
    app = web.Application()
    app["rooms"] = RoomManager()
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/{path:.*}", static_file)
    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuronScope cross-platform P2P model transfer server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8798)
    p.add_argument("--public-host", default=None, help="Hostname/IP to print in share URLs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    app = build_app()
    host_for_urls = args.public_host
    if not host_for_urls:
        addrs = local_addresses() if args.host in {"0.0.0.0", "::"} else []
        host_for_urls = addrs[0] if addrs else args.host

    scheme = "http"
    base = f"{scheme}://{host_for_urls}:{args.port}"
    print("NeuronScope Model Transfer")
    print(f"Listening: http://{args.host}:{args.port}/")
    print(f"Share URL: {base}/")
    if args.host in {"0.0.0.0", "::"}:
        for addr in local_addresses():
            print(f"LAN URL:   http://{addr}:{args.port}/")
    print("File bytes are sent peer-to-peer over WebRTC; the signaling server relays handshake metadata only.")
    web.run_app(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
