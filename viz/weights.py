#!/usr/bin/env python3
"""
Weight-space views: magnitude, quantization error, and where they land.

Every other view in this toolkit is activation-derived. This reads the weights
themselves, which answers questions activations cannot:

  magnitude     ||W[:, j]|| per neuron. A neuron that fires often but writes
                weakly is not the same as one that fires rarely and writes hard,
                and CETT folds the two together.

  quant error   ||Q(W)[:, j] - W[:, j]|| / ||W[:, j]|| between two quants of the
                same model. Which neurons a quantization actually damages.

  enrichment    whether the H-Neurons sit in the high-error tail more often than
                chance. If they do, that is a mechanistic link between
                quantization level and hallucination rather than a correlation
                between two summary numbers -- and it is the version of "low
                quants hallucinate more" that can be measured instead of
                repeated.

    python viz/weights.py --gguf model-Q6_K.gguf
    python viz/weights.py --gguf model-Q6_K.gguf --reference model-F16.gguf \\
        --h-neurons models/h_neurons.json
    python viz/weights.py --gguf ... --reference ... --dump w.npz   # headless

Only the ffn_down tensors are read, so this costs a fraction of a model load.
"""

import argparse
import json
import os
import sys

import numpy as np


def load_down_proj(path, n_layers=None):
    """-> (array, is_moe). Dense: [layers, neurons, hidden].
    MoE: [layers, experts, neurons, hidden]. Dequantized to float32."""
    try:
        import gguf
    except ImportError:
        raise SystemExit("needs the gguf package: pip install gguf")
    r = gguf.GGUFReader(path)
    by_name = {t.name: t for t in r.tensors}

    def kv(key):
        f = r.fields.get(key)
        if f is None:
            return None
        # contents() is the correct accessor. Reading parts[data[0]][0] treats a
        # string field as a numeric array and returns its first *byte*, which is
        # how chat_template came back as the integer 123 (the '{' character).
        try:
            v = f.contents()
            return v.decode("utf-8") if isinstance(v, bytes) else v
        except Exception:
            pass
        try:
            import gguf as _g
            if f.types and int(f.types[0]) == int(_g.GGUFValueType.STRING):
                return str(bytes(f.parts[f.data[0]]), "utf-8")
            return f.parts[f.data[0]][0].item()
        except Exception:
            return None
        f = r.fields.get(key)
        if f is None:
            return None
        try:
            return f.parts[f.data[0]][0].item()
        except Exception:
            try:
                return bytes(f.parts[f.data[0]]).decode()
            except Exception:
                return None

    arch = kv("general.architecture")
    n_layers = n_layers or (kv(f"{arch}.block_count") if arch else None)
    if not n_layers:
        raise SystemExit(f"could not read block_count from {path}")
    n_experts = kv(f"{arch}.expert_count") if arch else None

    out, moe = [], bool(n_experts)
    for l in range(n_layers):
        name = (f"blk.{l}.ffn_down_exps.weight" if moe
                else f"blk.{l}.ffn_down.weight")
        t = by_name.get(name)
        if t is None:
            raise SystemExit(f"{name} not in {path}")
        w = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        if moe:
            out.append(w.reshape(n_experts, -1, w.shape[-1]))
        else:
            # [neurons, hidden] once numpy reverses GGUF's dim order
            out.append(w.reshape(-1, w.shape[-1]))
    return np.stack(out), moe


def column_norms(w):
    """||W[:, j]|| per neuron, over the hidden axis."""
    return np.linalg.norm(w, axis=-1)


def quant_error(a, b):
    """Relative per-neuron error between two quants of the same weights.

    Normalised by the reference norm so a neuron with large weights and a
    neuron with small ones are on the same scale -- absolute error would just
    reproduce the magnitude map.
    """
    if a.shape != b.shape:
        raise SystemExit(f"shapes differ: {a.shape} vs {b.shape}. "
                         "These are not two quants of one model.")
    num = np.linalg.norm(a - b, axis=-1)
    den = np.linalg.norm(b, axis=-1)
    return num / (den + 1e-12)


def enrichment(values, mask, top_frac=0.05, n_perm=2000, seed=0):
    """Are masked neurons over-represented in the top tail of `values`?

    Reported as a ratio against a permutation null rather than a p-value from
    an assumed distribution: the values are neither independent nor normal, so
    a parametric test would be quietly wrong.
    """
    v, m = values.ravel(), mask.ravel()
    n_sel = int(m.sum())
    if n_sel == 0:
        return None
    k = max(1, int(len(v) * top_frac))
    top = np.zeros(len(v), dtype=bool)
    top[np.argsort(-v)[:k]] = True
    observed = int((top & m).sum())
    expected = k * n_sel / len(v)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.int32)
    idx = np.arange(len(v))
    for i in range(n_perm):
        null[i] = int(top[rng.choice(idx, n_sel, replace=False)].sum())
    p = float((null >= observed).sum() + 1) / (n_perm + 1)
    return {"top_frac": top_frac, "n_top": k, "n_selected": n_sel,
            "observed": observed, "expected": round(float(expected), 2),
            "ratio": round(observed / expected, 2) if expected else None,
            "p_permutation": round(p, 4),
            "null_mean": round(float(null.mean()), 2)}


def load_mask(path, shape):
    if not path:
        return None
    with open(path) as f:
        h = json.load(f)
    mask = np.zeros(shape, dtype=bool)
    flat = mask.reshape(shape[0], -1)
    for k, v in h["by_layer"].items():
        flat[int(k), np.asarray(v, dtype=int)] = True
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--reference", help="a second quant of the same model, for "
                                       "the error map (F16 is the usual choice)")
    p.add_argument("--h-neurons", help="models/h_neurons.json, to test overlap")
    p.add_argument("--top-frac", type=float, default=0.05)
    p.add_argument("--cmap", default="magma")
    p.add_argument("--dump", metavar="NPZ", help="write arrays and exit, no GPU")
    a = p.parse_args()

    print(f"reading {os.path.basename(a.gguf)}")
    w, moe = load_down_proj(a.gguf)
    norms = column_norms(w)
    print(f"  {'MoE ' if moe else ''}down_proj {w.shape} -> "
          f"norms {norms.shape}")

    err = None
    if a.reference:
        print(f"reading {os.path.basename(a.reference)}")
        ref, _ = load_down_proj(a.reference)
        err = quant_error(w, ref)
        print(f"  relative error: median {np.median(err):.5f}, "
              f"p99 {np.percentile(err, 99):.5f}, max {err.max():.5f}")

    mask = load_mask(a.h_neurons, norms.shape)
    if mask is not None:
        print(f"  H-Neurons: {int(mask.sum())} of {mask.size} "
              f"({mask.sum() / mask.size * 100:.3f}%)")
        for label, vals in (("weight magnitude", norms),
                            ("quantization error", err)):
            if vals is None:
                continue
            e = enrichment(vals, mask, a.top_frac)
            if not e:
                continue
            verdict = ("enriched" if e["p_permutation"] < 0.05 and e["ratio"] > 1
                       else "depleted" if e["p_permutation"] < 0.05
                       else "no different from chance")
            print(f"\n  H-Neurons in the top {int(a.top_frac * 100)}% by "
                  f"{label}:")
            print(f"    {e['observed']} observed vs {e['expected']} expected "
                  f"({e['ratio']}x), permutation p={e['p_permutation']}")
            print(f"    -> {verdict}")

    if a.dump:
        arrays = {"norms": norms}
        if err is not None:
            arrays["quant_error"] = err
        if mask is not None:
            arrays["h_mask"] = mask
        np.savez_compressed(a.dump, **arrays)
        print(f"\nwrote {a.dump}")
        return

    try:
        import fastplotlib as fpl
    except ImportError:
        raise SystemExit("needs fastplotlib to render; use --dump instead")
    if not fpl.enumerate_adapters():
        raise SystemExit("no WGPU adapter; use --dump, or install a Vulkan "
                         "driver (vulkan-radeon), or lavapipe for software "
                         "rendering")

    # MoE collapses to [layers, experts*neurons] for display: the expert axis is
    # categorical, so a 2D image of it is honest where a 3D volume would imply
    # continuity that is not there.
    def flat2d(x):
        return x.reshape(x.shape[0], -1)

    panels = [("magnitude", flat2d(norms))]
    if err is not None:
        panels.append(("quant error", flat2d(err)))
    if mask is not None and err is not None:
        overlay = flat2d(err).copy()
        overlay[~flat2d(mask)] = 0.0
        panels.append(("error at H-Neurons", overlay))

    shape = (len(panels), 1)
    fig = fpl.Figure(shape=shape, size=(1100, 260 * len(panels)),
                     names=[[n] for n, _ in panels])
    for i, (name, arr) in enumerate(panels):
        finite = arr[np.isfinite(arr)]
        lo, hi = np.percentile(finite, [1, 99.5]) if finite.size else (0, 1)
        fig[i, 0].add_image(arr, cmap=a.cmap, vmin=float(lo),
                            vmax=float(max(hi, lo + 1e-9)), name=name)
    print("\nrows are layers, columns are neurons; drag to pan, scroll to zoom")
    fig.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
