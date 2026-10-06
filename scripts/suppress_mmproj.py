#!/usr/bin/env python3
"""
Scale CLIP/SigLIP H-Neurons inside a llama.cpp multimodal projector GGUF
(the ``mmproj-*.gguf`` that LM Studio and llama-server load next to a
vision-language model).

The vision tower's MLP is fc1 -> GELU -> fc2, not gated, so the edit has to
be on fc2: neuron j's contribution is column j of the down projection.
Columns are not contiguous in GGML's row-major layout, which means a
quantized tensor cannot be edited exactly; mmproj files are almost always
F16/F32, and those are edited in place. Quantized projectors are refused.

Older converters stored fc1 as ``ffn_down`` and fc2 as ``ffn_up``, so the
down projection is identified by shape (its input width equals the profile's
neuron count), not by name.

    python scripts/suppress_mmproj.py --mmproj mmproj-model-f16.gguf \\
        --h_neurons models/clip/h_neurons.json --scale 0.5 \\
        --out mmproj-model-f16-supp050.gguf
"""
import argparse
import json
import os
import shutil

import numpy as np

FLOAT_TYPES = {"F32": np.float32, "F16": np.float16}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mmproj", required=True)
    p.add_argument("--h_neurons", required=True, help="profile from classifier.py on clip_neurons.py activations")
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", default=None, help="tensor prefix: v (vision, default) or t (text)")
    return p.parse_args(argv)


def find_down(tensors, prefix, layer, n_ff):
    cands = [tensors.get(f"{prefix}.blk.{layer}.ffn_{k}.weight") for k in ("down", "up")]
    for t in cands:
        if t is not None and int(t.shape[0]) == n_ff and int(t.shape[1]) != n_ff:
            return t
    raise SystemExit(f"layer {layer}: no {prefix}.blk.{layer}.ffn_* tensor with input width {n_ff}")


def main(argv=None):
    a = parse_args(argv)
    import gguf
    hn = json.load(open(a.h_neurons))
    n_ff = int(hn["n_neurons"])
    prefix = a.prefix or {"text": "t"}.get(hn.get("tower"), "v")
    by_layer = {int(k): sorted({int(i) for i in v}) for k, v in hn["by_layer"].items() if v}
    if not by_layer:
        raise SystemExit("no neurons in profile")

    reader = gguf.GGUFReader(a.mmproj)
    tensors = {t.name: t for t in reader.tensors}
    plan = []
    for layer, idx in sorted(by_layer.items()):
        t = find_down(tensors, prefix, layer, n_ff)
        tname = t.tensor_type.name
        if tname not in FLOAT_TYPES:
            raise SystemExit(f"{t.name} is {tname}; only F16/F32 projectors can be edited exactly")
        if max(idx) >= n_ff:
            raise SystemExit(f"neuron {max(idx)} out of range in layer {layer}")
        plan.append((t.name, int(t.data_offset), int(t.shape[0]), int(t.shape[1]), FLOAT_TYPES[tname], idx))
    del reader, tensors

    out = os.path.expanduser(a.out)
    if os.path.abspath(out) == os.path.abspath(a.mmproj):
        raise SystemExit("--out must differ from --mmproj")
    shutil.copyfile(a.mmproj, out)
    mm = np.memmap(out, dtype=np.uint8, mode="r+")
    for name, off, ne0, ne1, dt, idx in plan:
        nbytes = ne0 * ne1 * np.dtype(dt).itemsize
        w = mm[off:off + nbytes].view(dt).reshape(ne1, ne0)   # rows = outputs, cols = neurons
        w[:, idx] = (w[:, idx].astype(np.float32) * a.scale).astype(dt)
    mm.flush()
    del mm

    # Verify one edited and one untouched column against the source.
    src = {t.name: t for t in gguf.GGUFReader(a.mmproj).tensors}
    dst = {t.name: t for t in gguf.GGUFReader(out).tensors}
    name, _, ne0, ne1, dt, idx = plan[0]
    ws = np.asarray(src[name].data).reshape(ne1, ne0).astype(np.float32)
    wd = np.asarray(dst[name].data).reshape(ne1, ne0).astype(np.float32)
    err = np.abs(wd[:, idx[0]] - ws[:, idx[0]] * a.scale).max()
    other = next(j for j in range(ne0) if j not in idx)
    same = np.array_equal(ws[:, other], wd[:, other])
    print(f"check: {name} column {idx[0]} max abs error {err:.2e}; untouched column unchanged: {same}")
    if not same or err > 1e-2 * (np.abs(ws[:, idx[0]]).max() + 1e-6):
        raise SystemExit("verification failed; do not use this file")
    side = os.path.splitext(out)[0] + ".suppression.json"
    json.dump({"source": os.path.abspath(a.mmproj), "h_neurons": os.path.abspath(a.h_neurons),
               "scale": a.scale, "prefix": prefix,
               "n_neurons": sum(len(v) for v in by_layer.values()),
               "method": "fc2 column scale (F16/F32)"}, open(side, "w"), indent=2)
    print(f"wrote {out}\n      {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
