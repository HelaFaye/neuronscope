"""bloom.py --demo: a synthetic trace the 3D clients can load, flagged on the invented claim."""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
sys.path.insert(0, str(ROOT / "scripts"))
import bloom  # noqa: E402


def test_demo_payload_layout_and_flags():
    blob, meta = bloom.build_payload(None, None, 97.0, 1.0, 120000, demo=True)
    T, N, L = struct.unpack("<iii", blob[:12])
    assert (T, N, L) == (meta["frames"], meta["cells"], meta["layers"])
    assert len(blob) == 12 + 8 * N + 4 * T * N + T * N        # the layout both clients decode
    assert "synthetic" in meta["model"]
    flagged = {meta["labels"][i].strip() for i in meta["flagged"]}
    assert flagged and flagged <= {"moved", "to", "Lyon", "1923", "1931", "in", "it", "was", ","}
    assert {"moved", "Lyon"} & flagged
