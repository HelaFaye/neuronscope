#!/usr/bin/env python3
"""
Play a token-resolved activation trace in 3D, with themes.

Reads a session written by scripts/trace_sample.py, where axis 0 is time, and
animates layers x neurons as the response is generated. Two neuron populations
are distinguished:

    active        high CETT at this token
    hallucinating active AND in the H-Neuron set AND scoring above threshold

That second definition matters. A neuron is not "hallucinating" because it is
in the H-Neuron set -- those fire constantly on grounded text too. It is
flagged when it is firing *and* the classifier score for that token is high, so
the marking tracks the moment rather than the membership.

    python viz/timeline.py runs/trace-abc
    python viz/timeline.py runs/trace-abc --theme ember --play
    python viz/timeline.py runs/trace-abc --dump frames.npz    # headless

On effects: pygfx has no bloom or glow post-processing, so themes encode state
through colormap, point size, opacity and background. At a few hundred thousand
points those read better than a glow would anyway.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from records import Session  # noqa: E402

# Themes live in themes.json so the pygfx, Godot and three.js frontends read
# one definition. A new theme is an entry in that file, not a code change in
# three places.
_THEME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "themes.json")
with open(_THEME_PATH) as _f:
    THEMES = json.load(_f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--theme", default="ember", choices=sorted(THEMES))
    p.add_argument("--h-neurons", help="models/h_neurons.json")
    p.add_argument("--active-pct", type=float, default=97.0,
                   help="percentile of CETT above which a neuron counts active")
    p.add_argument("--score-z", type=float, default=1.0,
                   help="per-token score z above which a frame counts as "
                        "hallucinating")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--play", action="store_true")
    p.add_argument("--dump", metavar="NPZ", help="write frames and exit, no GPU")
    p.add_argument("--volume", action="store_true",
                   help="4D: render every frame at once with time as the depth "
                        "axis, instead of one frame sliding along Z. The whole "
                        "trace is visible as a solid, so a burst is a shape "
                        "rather than a flash you have to catch.")
    p.add_argument("--decay", type=float, default=0.55,
                   help="--volume: how much older frames fade, 0 = none")
    p.add_argument("--list-themes", action="store_true")
    return p.parse_args()


def load_trace(path):
    """-> (frames [T, L, N], fields). Axis 0 is time."""
    s = Session(path)
    ids = s.ids
    if not ids:
        raise SystemExit(f"no samples in {path}")
    d = s.get(ids[0])
    f = d.get("fields") or {}
    if f.get("kind") != "trace":
        raise SystemExit(f"{path} is not a trace session; "
                         "produce one with scripts/trace_sample.py")
    n_frames, n_layers = int(f["n_frames"]), int(f["n_layers"])
    arr = d["agg"].astype(np.float32)
    frames = arr.reshape(n_frames, n_layers, -1)
    f["scores"] = d.get("scores")
    f["tokens"] = d.get("tokens")
    f["meta"] = s.meta
    return frames, f


def classify_frames(frames, fields, h_mask, active_pct, score_z):
    """-> (active [T,L,N] bool, halluc [T,L,N] bool, per-frame z scores).

    Threshold on the whole trace, not per frame: a per-frame percentile would
    mark the same fraction active at every token and erase exactly the
    variation the animation exists to show.
    """
    thresh = float(np.percentile(frames, active_pct))
    active = frames >= thresh

    scores = fields.get("scores")
    if scores is None or len(scores) != frames.shape[0]:
        z = np.zeros(frames.shape[0], dtype=np.float32)
    else:
        s = np.asarray(scores, dtype=np.float32)
        z = (s - s.mean()) / (s.std() + 1e-8)

    hot = (z >= score_z)[:, None, None]
    halluc = active & hot
    if h_mask is not None:
        halluc = halluc & h_mask[None, :, :]
    return active, halluc, z


def load_mask(path, shape):
    if not path:
        return None
    with open(path) as f:
        h = json.load(f)
    n_layers, n_neurons = shape
    mask = np.zeros(shape, dtype=bool)
    src_n = h["n_neurons"] * h.get("n_experts", 1)
    for k, v in h["by_layer"].items():
        idx = np.asarray(v, dtype=int)
        if src_n != n_neurons:
            # The trace is binned; map each neuron onto its bin.
            idx = np.minimum((idx * n_neurons) // src_n, n_neurons - 1)
        mask[int(k), np.unique(idx)] = True
    return mask


def main():
    args = parse_args()
    if args.list_themes:
        for name, t in THEMES.items():
            print(f"  {name:<10} {t['note']}")
        return

    theme = THEMES[args.theme]
    frames, fields = load_trace(args.session)
    T, L, N = frames.shape
    print(f"{T} frames x {L} layers x {N} neurons "
          f"({frames.nbytes / 1e6:.1f} MB), theme '{args.theme}'")

    h_mask = load_mask(args.h_neurons, (L, N))
    active, halluc, z = classify_frames(frames, fields, h_mask,
                                        args.active_pct, args.score_z)
    print(f"active: {active.mean() * 100:.2f}% of cells; "
          f"hallucinating: {halluc.mean() * 100:.3f}%")
    hot_frames = int((halluc.any(axis=(1, 2))).sum())
    print(f"{hot_frames} of {T} frames flagged")

    pieces = fields.get("pieces") or []
    if pieces and hot_frames:
        idx = np.where(halluc.any(axis=(1, 2)))[0][:8]
        stride = int(fields.get("stride", 1))
        shown = ["".join(pieces[i * stride:(i + 1) * stride]) for i in idx]
        print("  first flagged tokens: " +
              " | ".join(repr(s)[:16] for s in shown))

    if args.dump:
        np.savez_compressed(args.dump, frames=frames, active=active,
                            halluc=halluc, z=z,
                            theme=json.dumps(theme))
        print(f"\nwrote {args.dump}")
        return

    try:
        import fastplotlib as fpl
    except ImportError:
        raise SystemExit("needs fastplotlib to render; use --dump instead")
    if not fpl.enumerate_adapters():
        raise SystemExit("no WGPU adapter; use --dump, or install a Vulkan "
                         "driver, or lavapipe for software rendering")

    # Points in (layer, neuron, time) space. Only cells that are active at some
    # point are instantiated -- at 97% inactive, drawing every cell for every
    # frame would be almost entirely wasted geometry.
    ever = active.any(axis=0)
    ly, nx = np.nonzero(ever)
    n_pts = len(ly)
    print(f"{n_pts} points instantiated of {L * N} cells")

    lo, hi = theme["size"]
    a_lo, a_hi = theme["alpha"]

    if args.volume:
        # True 4D. Every (frame, cell) that was ever active becomes its own
        # point at depth z = frame, so the time axis is geometry rather than
        # animation state. The cost is one point per active cell per frame
        # instead of per cell, which is why only active ones are built.
        fi, li, ni = np.nonzero(active)
        n_pts = len(fi)
        print(f"volume: {n_pts:,} points ({T} frames x "
              f"{active[0].sum():,} avg active)")
        vmax = float(np.percentile(frames, 99.5)) or 1.0
        v = np.clip(frames[fi, li, ni] / vmax, 0, 1)
        hal = halluc[fi, li, ni]

        pos = np.column_stack([
            ni.astype(np.float32),
            li.astype(np.float32) * (N / max(L, 1)) * 0.05,
            fi.astype(np.float32) * 3.0,
        ])
        # Older frames recede rather than vanishing: depth does the ordering,
        # brightness only marks recency so the newest edge stays readable.
        age = 1.0 - (fi / max(T - 1, 1))
        fade = 1.0 - args.decay * age
        colors = np.tile(_rgba(theme["active"], 1.0), (n_pts, 1))
        colors[hal] = _rgba(theme["halluc"], 1.0)
        colors[:, 3] = np.clip((a_lo + (a_hi - a_lo) * v) * fade, 0.02, 1.0)
        sizes = (lo + (hi - lo) * v) * np.where(hal, 1.6, 1.0)

        fig = fpl.Figure(shape=(2, 1), size=(1100, 820), cameras=[["3d"], ["2d"]],
                         names=[["volume: neuron x layer x time"],
                                ["score over tokens"]])
        fig[0, 0].add_scatter(pos.astype(np.float32), colors=colors,
                              sizes=sizes, name="cells")
        fig[1, 0].add_line(
            np.column_stack([np.arange(T, dtype=np.float32), z]),
            colors=theme["halluc"], thickness=2.0, name="score z")
        print("\ndepth is time: the whole trace is one solid. Drag to orbit.")
        fig.show()
        fpl.loop.run()
        return

    fig = fpl.Figure(shape=(2, 1), size=(1100, 760), cameras=[["3d"], ["2d"]],
                     names=[["trace"], ["score over tokens"]])

    def frame_data(t):
        base = np.full(n_pts, float(t), dtype=np.float32)
        return np.column_stack([nx.astype(np.float32),
                                ly.astype(np.float32) * (N / max(L, 1)) * 0.05,
                                base])

    def frame_style(t):
        v = frames[t][ly, nx]
        vmax = float(np.percentile(frames, 99.5)) or 1.0
        norm = np.clip(v / vmax, 0, 1)
        sizes = lo + (hi - lo) * norm
        colors = np.tile(np.array([0.5, 0.5, 0.5, a_lo], dtype=np.float32),
                         (n_pts, 1))
        act = active[t][ly, nx]
        colors[act] = _rgba(theme["active"], a_hi)
        hal = halluc[t][ly, nx]
        colors[hal] = _rgba(theme["halluc"], 1.0)
        sizes[hal] *= 1.6
        return sizes, colors

    sizes0, colors0 = frame_style(0)
    pts = fig[0, 0].add_scatter(frame_data(0), colors=colors0, sizes=sizes0,
                                name="cells")
    fig[1, 0].add_line(np.column_stack([np.arange(T, dtype=np.float32), z]),
                       colors=theme["halluc"], thickness=2.0, name="score z")

    state = {"t": 0, "playing": args.play}

    def step():
        t = state["t"]
        pts.data[:, 2] = float(t)
        s, c = frame_style(t)
        pts.sizes = s
        pts.colors = c

    def on_render():
        if state["playing"]:
            state["t"] = (state["t"] + 1) % T
            step()

    fig.add_animations(lambda *a: on_render())
    print(f"\nplaying at ~{args.fps} fps; drag to orbit, scroll to zoom")
    fig.show()
    fpl.loop.run()


def _rgba(hexstr, alpha):
    h = hexstr.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [alpha],
                    dtype=np.float32)


if __name__ == "__main__":
    main()
