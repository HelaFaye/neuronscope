#!/usr/bin/env python3
"""Stand-in for llama-cett-dump: same CLI and file formats, deterministic data.

One token per character (id = code point). Activations for token t, layer l,
neuron j are 1 + (t + l + j) % 3, so a span mean is predictable."""
import json
import struct
import sys
from pathlib import Path

import numpy as np

args = sys.argv[1:]
opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("-") and not args[i + 1].startswith("--")}
manifest, outdir = opt["--manifest"], Path(opt["--outdir"])
N_FF = 4
for line in open(manifest, encoding="utf-8"):
    rec = json.loads(line)
    ids = np.array([ord(c) for c in rec["text"]], dtype="<i4")
    head = b"CETT" + struct.pack("<II", 1, len(ids)) + ids.tobytes()
    if "--tokenize-only" in args:
        (outdir / f"{rec['id']}.toks").write_bytes(head)
        continue
    n_layers = int(opt["--n-layers"])
    spans = rec["spans"]
    agg = np.zeros((len(spans), n_layers, N_FF), dtype=np.float32)
    for si, (a, b) in enumerate(spans):
        b = len(ids) if b < 0 else b
        t = np.arange(a, b)[:, None, None]
        l = np.arange(n_layers)[None, :, None]
        j = np.arange(N_FF)[None, None, :]
        agg[si] = (1 + (t + l + j) % 3).mean(axis=0)
    cells = len(spans) * n_layers
    body = struct.pack("<iiii", len(spans), n_layers, 1, N_FF)
    body += np.full(cells, 1, dtype="<i4").tobytes() + np.ones(cells, dtype=np.uint8).tobytes()
    body += agg.astype("<f2").tobytes()
    (outdir / f"{rec['id']}.bin").write_bytes(head + body)
