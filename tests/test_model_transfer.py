import asyncio
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_transfer import (CODE_ALPHABET, CODE_LEN, KEY_BYTES, Conn, JoinError,  # noqa: E402
                            RoomManager, new_room_code, new_room_key)


class DummyWS:
    closed = False


def test_room_code_shape():
    for _ in range(100):
        code = new_room_code()
        assert len(code) == CODE_LEN
        assert set(code) <= set(CODE_ALPHABET)


def test_invitation_key_is_128_bits():
    assert KEY_BYTES * 8 >= 128
    assert len({new_room_key() for _ in range(50)}) == 50
    # The display code is a label, not the secret; the key carries the entropy.
    assert CODE_LEN * math.log2(len(CODE_ALPHABET)) < KEY_BYTES * 8


def test_create_and_join_with_key():
    async def run():
        mgr = RoomManager()
        room, p1 = await mgr.create(DummyWS())
        room2, p2 = await mgr.join(room.code, room.key, DummyWS(), "10.0.0.2")
        assert room2 is room
        assert p1.peer_id != p2.peer_id
        assert len(await mgr.peers(room.code)) == 2

    asyncio.run(run())


def test_wrong_key_and_unknown_room_look_identical():
    async def run():
        mgr = RoomManager()
        room, _ = await mgr.create(DummyWS())
        errors = []
        for code, key in [(room.code, "wrong"), ("ABCDEFGHJK", room.key)]:
            try:
                await mgr.join(code, key, DummyWS(), f"10.0.0.{len(errors) + 3}")
            except JoinError as e:
                errors.append(str(e))
        assert errors == ["bad-room-or-key", "bad-room-or-key"]

    asyncio.run(run())


def test_join_bruteforce_is_throttled():
    async def run():
        mgr = RoomManager()
        room, _ = await mgr.create(DummyWS())
        for _ in range(mgr.throttle.max_failures):
            try:
                await mgr.join(room.code, "guess", DummyWS(), "10.9.9.9")
            except JoinError:
                pass
        try:
            await mgr.join(room.code, room.key, DummyWS(), "10.9.9.9")
        except JoinError as e:
            assert str(e) == "rate-limited"
        else:
            raise AssertionError("locked-out address still allowed to join")

    asyncio.run(run())


def test_knock_requires_admission():
    async def run():
        mgr = RoomManager()
        room, host = await mgr.create(DummyWS())
        guest = Conn(ws=DummyWS(), addr="10.0.0.5")
        knock, members = await mgr.knock(room.code, guest)
        assert [m.peer_id for m in members] == [host.peer_id]
        assert len(await mgr.peers(room.code)) == 1
        _, _, peer = await mgr.admit(host, knock.knock_id, True)
        assert peer is not None and guest.peer is peer
        assert len(await mgr.peers(room.code)) == 2

    asyncio.run(run())


def test_knock_denied():
    async def run():
        mgr = RoomManager()
        room, host = await mgr.create(DummyWS())
        guest = Conn(ws=DummyWS(), addr="10.0.0.6")
        knock, _ = await mgr.knock(room.code, guest)
        _, _, peer = await mgr.admit(host, knock.knock_id, False)
        assert peer is None and guest.peer is None

    asyncio.run(run())


def test_leave_notifies_remaining_peers():
    async def run():
        mgr = RoomManager()
        room, p1 = await mgr.create(DummyWS())
        _, p2 = await mgr.join(room.code, room.key, DummyWS())
        others = await mgr.leave(p1)
        assert [p.peer_id for p in others] == [p2.peer_id]

    asyncio.run(run())


def test_websocket_origin_check():
    from aiohttp.test_utils import TestClient, TestServer
    from model_transfer import build_app

    async def run():
        async with TestClient(TestServer(build_app())) as client:
            r = await client.get("/ws", headers={"Origin": "https://evil.example",
                                                 "Upgrade": "websocket", "Connection": "Upgrade",
                                                 "Sec-WebSocket-Version": "13",
                                                 "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="})
            assert r.status == 403
            ws = await client.ws_connect("/ws")
            await ws.send_json({"type": "create"})
            msg = await ws.receive_json()
            assert msg["type"] == "joined" and len(msg["key"]) >= 22
            ice = await (await client.get("/ice.json")).json()
            assert ice["mode"] == "direct"
            page = await client.get("/")
            assert "no-referrer" in page.headers["Referrer-Policy"]
            await ws.close()

    asyncio.run(run())


def test_static_gui_has_webrtc_transfer_surface():
    html = (ROOT / "web" / "model_transfer.html").read_text()
    for needle in ("RTCPeerConnection", "createDataChannel", "datachannel", "showDirectoryPicker",
                   "/ws", "iceTransportPolicy", "a=fingerprint:", "class Sha256", "sha256"):
        assert needle in html, needle
