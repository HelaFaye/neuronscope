#!/usr/bin/env python3
"""
Budget VRAM across the model, its KV cache, and the visualizer.

The three compete for the same memory, and on a shared-memory iGPU they also
compete with the desktop. This computes each from real numbers rather than a
rule of thumb, and decides what to move to CPU when they do not all fit.

    python scripts/vram_budget.py --gguf model-Q6_K.gguf --ctx 8192
    python scripts/vram_budget.py --gguf ... --ctx 110592 --cache-type q8_0
    python scripts/vram_budget.py --gguf ... --vram 12 --viz godot --json

The KV cache is the term people get wrong. It scales linearly with context and
is independent of quantization: a Q4 model and a Q8 model of the same
architecture have identical cache costs. At long context it can exceed the
weights.

    bytes = 2 (K and V) * layers * kv_heads * head_dim * ctx * elem_size

kv_heads is the grouped-query count, not the attention head count -- using the
latter overestimates by the GQA ratio, often 4x or 8x.
"""

import argparse
import gguf_utils
import json
import os
import sys

GIB = 1024 ** 3

# Bytes per element for llama.cpp's --cache-type-k / --cache-type-v.
CACHE_TYPES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0,
               "q8_0": 1.0625, "q5_1": 0.75, "q5_0": 0.6875,
               "q4_1": 0.5625, "q4_0": 0.5}

# Rough working-set cost of each visualizer at a typical trace size. These are
# framebuffer plus geometry plus the effect chain, not precise allocations.
VIZ_COST = {
    "none": (0.0, "no visualizer"),
    "pygfx": (0.35, "points, no post-processing"),
    "threejs": (0.75, "bloom chain: several full-resolution render targets"),
    "godot": (0.90, "Forward+ with glow; half-resolution upscale assumed"),
}


def read_gguf_shape(path):
    """-> dict of the architecture numbers the KV formula needs."""
    try:
        import gguf
    except ImportError:
        raise SystemExit("needs the gguf package: pip install gguf")
    r = gguf.GGUFReader(path)

    def kv(key):
        return gguf_utils.read_kv(r, key)

    arch = kv("general.architecture")
    if not arch:
        raise SystemExit(f"no general.architecture in {path}")
    n_head = kv(f"{arch}.attention.head_count")
    n_kv = kv(f"{arch}.attention.head_count_kv")
    n_embd = kv(f"{arch}.embedding_length")
    k_len = kv(f"{arch}.attention.key_length")
    v_len = kv(f"{arch}.attention.value_length")
    if k_len is None and n_head and n_embd:
        k_len = n_embd // n_head
    return {
        "arch": arch,
        "n_layers": kv(f"{arch}.block_count"),
        "n_head": n_head,
        # Absent head_count_kv means no GQA, so it equals the head count.
        "n_kv_head": n_kv if n_kv is not None else n_head,
        "head_dim_k": k_len,
        "head_dim_v": v_len if v_len is not None else k_len,
        "n_embd": n_embd,
        "train_ctx": kv(f"{arch}.context_length"),
        "n_experts": kv(f"{arch}.expert_count"),
        "file_bytes": os.path.getsize(path),
    }


def kv_cache_bytes(shape, ctx, cache_type="f16", cache_type_v=None):
    """K and V are sized separately: they can use different quantizations and
    on some architectures different head dimensions."""
    kt = CACHE_TYPES[cache_type]
    vt = CACHE_TYPES[cache_type_v or cache_type]
    need = ("n_layers", "n_kv_head", "head_dim_k", "head_dim_v")
    if any(shape.get(k) in (None, 0) for k in need):
        return None
    per_tok_k = shape["n_layers"] * shape["n_kv_head"] * shape["head_dim_k"]
    per_tok_v = shape["n_layers"] * shape["n_kv_head"] * shape["head_dim_v"]
    return int(ctx * (per_tok_k * kt + per_tok_v * vt))


def plan(shape, ctx, vram_bytes, viz="pygfx", cache_type="f16",
         cache_type_v=None, ngl=None, reserve=0.75 * GIB):
    """Decide what fits on the GPU and what moves to CPU.

    Order of eviction: visualizer first, then model layers. The model is the
    thing whose speed the user notices; a visualizer on CPU is slower to draw
    but does not slow generation, and generation is the long pole.
    """
    model = shape["file_bytes"]
    cache = kv_cache_bytes(shape, ctx, cache_type, cache_type_v)
    viz_bytes = int(VIZ_COST.get(viz, VIZ_COST["pygfx"])[0] * GIB)
    n_layers = shape["n_layers"] or 1

    budget = max(0, vram_bytes - reserve)
    warnings, notes = [], []

    if cache is None:
        warnings.append("could not read the attention shape; the KV estimate "
                        "is missing and this plan is unreliable")
        cache = 0

    if cache > model:
        notes.append(f"the KV cache ({cache / GIB:.1f} GiB) is larger than the "
                     f"weights ({model / GIB:.1f} GiB) at this context")

    viz_device = "gpu"
    total = model + cache + viz_bytes
    if total > budget:
        viz_device = "cpu"
        notes.append("visualizer moved to CPU: model and cache take priority")
        total = model + cache

    # KV cache is placed before layers in llama.cpp when offloading, so it is
    # subtracted first here too.
    layers_on_gpu = n_layers
    if total > budget:
        for_layers = budget - cache
        per_layer = model / n_layers
        layers_on_gpu = max(0, int(for_layers // per_layer)) if per_layer else 0
        layers_on_gpu = min(layers_on_gpu, n_layers)
        if layers_on_gpu < n_layers:
            notes.append(f"only {layers_on_gpu} of {n_layers} layers fit; the "
                         "rest run on CPU")
        if layers_on_gpu == 0:
            warnings.append("nothing fits on the GPU. Lower --ctx, quantize "
                            "the cache, or use a smaller model.")
    if ngl is not None:
        layers_on_gpu = min(ngl, layers_on_gpu)

    if ctx > (shape.get("train_ctx") or ctx):
        warnings.append(f"--ctx {ctx} exceeds the model's trained context "
                        f"{shape['train_ctx']}; quality degrades past it")
    if cache_type in ("q4_0", "q4_1") or (cache_type_v or "") in ("q4_0", "q4_1"):
        warnings.append("4-bit KV cache measurably degrades long-context "
                        "recall; q8_0 is the usual safe choice")

    return {
        "model_bytes": model, "kv_bytes": cache, "viz_bytes": viz_bytes,
        "viz_device": viz_device, "layers_on_gpu": layers_on_gpu,
        "n_layers": n_layers, "vram_bytes": vram_bytes,
        "reserve_bytes": int(reserve), "budget_bytes": int(budget),
        "gpu_used": int(cache + (model / n_layers) * layers_on_gpu
                        + (viz_bytes if viz_device == "gpu" else 0)),
        "fits": layers_on_gpu == n_layers and viz_device == "gpu",
        "warnings": warnings, "notes": notes,
    }


def detect_vram():
    """Best effort. An iGPU's carve-out is a real limit but comes from RAM.
    NVIDIA: the sum over all GPUs, since llama-server splits layers across them."""
    try:
        from cuda_info import query_gpus
        gpus = query_gpus()
        if gpus:
            return sum(g["memory_total"] for g in gpus)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info()[1])
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # No discrete card detected; assume a shared carve-out of
                    # roughly a third of system RAM, which is a common default.
                    return int(int(line.split()[1]) * 1024 / 3)
    except OSError:
        pass
    return None


def fmt(b):
    return f"{b / GIB:6.2f} GiB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--vram", type=float, help="GiB; detected if omitted")
    p.add_argument("--viz", default="pygfx", choices=sorted(VIZ_COST))
    p.add_argument("--cache-type", default="f16", choices=sorted(CACHE_TYPES))
    p.add_argument("--cache-type-v", choices=sorted(CACHE_TYPES))
    p.add_argument("--ngl", type=int)
    p.add_argument("--reserve", type=float, default=0.75,
                   help="GiB held back for the desktop and driver")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    shape = read_gguf_shape(a.gguf)
    vram = int(a.vram * GIB) if a.vram else detect_vram()
    if not vram:
        raise SystemExit("could not detect VRAM; pass --vram")

    r = plan(shape, a.ctx, vram, a.viz, a.cache_type, a.cache_type_v,
             a.ngl, a.reserve * GIB)
    if a.json:
        print(json.dumps({**shape, **r}, indent=2))
        return 0 if r["fits"] else 1

    print(f"{os.path.basename(a.gguf)}")
    print(f"  {shape['arch']}, {shape['n_layers']} layers, "
          f"{shape['n_head']} heads / {shape['n_kv_head']} kv "
          f"(GQA {shape['n_head'] // max(shape['n_kv_head'], 1)}x), "
          f"head_dim {shape['head_dim_k']}")
    print(f"\ncontext {a.ctx}, cache {a.cache_type}"
          f"{'/' + a.cache_type_v if a.cache_type_v else ''}")
    print(f"  weights        {fmt(r['model_bytes'])}")
    print(f"  KV cache       {fmt(r['kv_bytes'])}")
    print(f"  visualizer     {fmt(r['viz_bytes'])}  ({a.viz}: "
          f"{VIZ_COST[a.viz][1]})")
    print(f"  {'-' * 34}")
    print(f"  total          {fmt(r['model_bytes'] + r['kv_bytes'] + r['viz_bytes'])}")
    print(f"  VRAM           {fmt(r['vram_bytes'])} "
          f"(reserving {fmt(r['reserve_bytes'])})")

    print(f"\nplan")
    print(f"  layers on GPU  {r['layers_on_gpu']}/{r['n_layers']}  "
          f"(-ngl {r['layers_on_gpu']})")
    print(f"  visualizer on  {r['viz_device'].upper()}")
    print(f"  GPU used       {fmt(r['gpu_used'])} of {fmt(r['budget_bytes'])}")
    for n in r["notes"]:
        print(f"  note: {n}")
    for w in r["warnings"]:
        print(f"  WARNING: {w}")

    if not r["fits"] and r["kv_bytes"]:
        alt = plan(shape, a.ctx, vram, a.viz, "q8_0", "q8_0", a.ngl,
                   a.reserve * GIB)
        if alt["fits"] or alt["layers_on_gpu"] > r["layers_on_gpu"]:
            print(f"\n  try --cache-type q8_0: KV drops to "
                  f"{fmt(alt['kv_bytes'])}, "
                  f"{alt['layers_on_gpu']}/{alt['n_layers']} layers fit")
    return 0 if r["fits"] else 1


if __name__ == "__main__":
    sys.exit(main())
