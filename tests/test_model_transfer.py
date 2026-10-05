import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_transfer import CODE_ALPHABET, CODE_LEN, RoomManager, new_room_code  # noqa: E402


class DummyWS:
    closed = False


def test_room_code_shape():
    for _ in range(100):
        code = new_room_code()
        assert len(code) == CODE_LEN
        assert set(code) <= set(CODE_ALPHABET)


def test_create_and_join():
    async def run():
        mgr = RoomManager()
        room, p1 = await mgr.create_or_join(None, DummyWS())
        room2, p2 = await mgr.create_or_join(room.code, DummyWS())
        assert room2 is room
        assert p1.peer_id != p2.peer_id
        assert len(await mgr.peers(room.code)) == 2

    asyncio.run(run())


def test_unknown_room_rejected():
    async def run():
        mgr = RoomManager()
        try:
            await mgr.create_or_join("ABCDEFGH", DummyWS())
        except ValueError as e:
            assert str(e) == "room-not-found"
        else:
            raise AssertionError("unknown room accepted")

    asyncio.run(run())


def test_leave_notifies_remaining_peers():
    async def run():
        mgr = RoomManager()
        room, p1 = await mgr.create_or_join(None, DummyWS())
        _, p2 = await mgr.create_or_join(room.code, DummyWS())
        others = await mgr.leave(p1)
        assert [p.peer_id for p in others] == [p2.peer_id]

    asyncio.run(run())


def test_static_gui_has_webrtc_transfer_surface():
    html = (ROOT / "web" / "model_transfer.html").read_text()
    assert "RTCPeerConnection" in html
    assert "createDataChannel" in html
    assert "datachannel" in html
    assert "showDirectoryPicker" in html
    assert "/ws" in html
