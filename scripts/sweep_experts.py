#!/usr/bin/env python3
"""
Sweep the active expert count on a MoE model and measure what it costs.

MoE models route each token to the top-k of N experts, and k is baked into the
GGUF as <arch>.expert_used_count. llama.cpp lets you override it at load time,
so you can trade quality against speed without re-quantizing anything:

    --override-kv qwen3moe.expert_used_count=int:6

This launches a server per value of k, evaluates the same task set against each,
and reports the paired comparison -- gains, regressions and tokens/sec -- so you
can see where raising k stops paying.

DENSE MODELS HAVE NO SUCH KNOB. Ornith-1.0-9B and 1.5-9B are dense; there is
nothing to sweep. This needs 35B-A3B, 397B, or another MoE.

    python scripts/sweep_experts.py \\
        --gguf ~/models/Ornith-1.5-35B-A3B-Q4_K_M.gguf \\
        --server ~/llama.cpp/build/bin/llama-server \\
        --tasks data/eval_tasks.jsonl \\
        --experts 2 4 6 8 --ngl 99

Raising k costs compute per token but touches more expert weights, so on a
CPU+GPU split the slowdown can be worse than linear once the extra experts fall
outside VRAM. The tokens/sec column is measured, not assumed.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_eval import GRADERS, compare, kind_of, mcnemar, run_endpoint  # noqa


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--server", required=True, help="path to llama-server")
    p.add_argument("--tasks", required=True)
    p.add_argument("--experts", nargs="+", type=int, required=True)
    p.add_argument("--arch", help="GGUF architecture key; read from the file "
                                  "if omitted")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--override-tensor", dest="ot",
                   help="e.g. '\\.ffn_.*_exps\\.=CPU' to keep routed experts in "
                        "RAM and always-active tensors on the GPU")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="everything after this is passed to llama-server")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--startup-timeout", type=int, default=600)
    p.add_argument("--out")
    return p.parse_args()


def gguf_moe_info(path):
    """-> (arch, n_expert, n_expert_used) from GGUF metadata."""
    try:
        import gguf
    except ImportError:
        raise SystemExit("needs the gguf package: pip install gguf")
    r = gguf.GGUFReader(path)

    def kv(key):
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
    if not arch:
        raise SystemExit(f"no general.architecture in {path}")
    return (arch, kv(f"{arch}.expert_count"),
            kv(f"{arch}.expert_used_count"))


def wait_healthy(port, proc, timeout):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                f"llama-server exited with code {proc.returncode} before "
                "becoming healthy; run the command by hand to see why")
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def measure_rate(port, max_tokens):
    """tokens/sec on a fixed prompt, from the server's own timings."""
    try:
        r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                          timeout=300, json={
                              "messages": [{"role": "user",
                                            "content": "Count from 1 to 40."}],
                              "temperature": 0.0, "max_tokens": max_tokens})
        r.raise_for_status()
        u = r.json().get("usage") or {}
        t = r.json().get("timings") or {}
        if t.get("predicted_per_second"):
            return float(t["predicted_per_second"])
        return u.get("completion_tokens")
    except Exception:
        return None


def main():
    args = parse_args()
    arch, n_expert, default_k = gguf_moe_info(args.gguf)
    print(f"architecture: {arch}")
    if not n_expert:
        raise SystemExit(
            f"{os.path.basename(args.gguf)} declares no {arch}.expert_count, so "
            "it is a dense model.\nThere is no active-parameter knob to sweep. "
            "Ornith 9B is dense; use 35B-A3B or 397B.")
    print(f"experts: {n_expert} total, {default_k} used by default")
    bad = [k for k in args.experts if not (1 <= k <= n_expert)]
    if bad:
        raise SystemExit(f"--experts {bad} outside 1..{n_expert}")

    tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8")
             if l.strip()]
    kinds = {}
    for t in tasks:
        kinds[kind_of(t)] = kinds.get(kind_of(t), 0) + 1
    print(f"{len(tasks)} tasks: " +
          ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))

    class A:  # run_endpoint reads these
        model = None
        concurrency = args.concurrency
        max_tokens = args.max_tokens

    results, rates = {}, {}
    for k in args.experts:
        label = f"k={k}"
        cmd = [args.server, "-m", args.gguf, "-ngl", str(args.ngl),
               "-c", str(args.ctx), "--port", str(args.port), "--host",
               "127.0.0.1", "--override-kv",
               f"{arch}.expert_used_count=int:{k}"]
        if args.ot:
            cmd += ["--override-tensor", args.ot]
        cmd += args.extra
        print(f"\n--- {label} ---")
        print(" ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            if not wait_healthy(args.port, proc, args.startup_timeout):
                print(f"  timed out after {args.startup_timeout}s, skipping")
                continue
            rates[label] = measure_rate(args.port, 128)
            if rates[label]:
                print(f"  {rates[label]:.1f} tok/s")
            results[label] = run_endpoint(
                label, f"http://127.0.0.1:{args.port}", tasks, A)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(2)   # let the port free up

    if not results:
        raise SystemExit("no successful runs")

    print(f"\n{'setting':<10} {'tok/s':>8} " +
          " ".join(f"{c:>10}" for c in ("correct", "wrong", "abstained",
                                        "refused")))
    for label in results:
        c = {}
        for x in results[label]:
            c[x["verdict"]] = c.get(x["verdict"], 0) + 1
        rate = f"{rates.get(label):.1f}" if rates.get(label) else "?"
        print(f"{label:<10} {rate:>8} " + " ".join(
            f"{c.get(k, 0):>10}" for k in ("correct", "wrong", "abstained",
                                           "refused")))

    ref = next(iter(results))
    out = {"gguf": args.gguf, "arch": arch, "n_expert": n_expert,
           "default_k": default_k, "reference": ref, "rates": rates,
           "comparisons": []}
    print(f"\nagainst {ref}:")
    for label in results:
        if label == ref:
            continue
        g, l, cg, cl = compare(results[ref], results[label], tasks)
        chi2, disc = mcnemar(len(g), len(l))
        verdict = ("significant" if chi2 > 3.84 else "not significant") \
            if disc >= 10 else "too few changes to call"
        print(f"  {label}: gained {len(g)}, regressed {len(l)}, "
              f"net {len(g) - len(l):+d}  chi2={chi2:.1f} ({verdict})")
        if cl or cg:
            print(f"    canary: {len(cl)} newly refused, "
                  f"{len(cg)} newly answered")
        out["comparisons"].append({"setting": label, "gained": len(g),
                                   "regressed": len(l), "chi2": chi2,
                                   "canary_newly_refused": len(cl)})

    if args.out:
        with open(args.out, "w") as f:
            json.dump({**out, "raw": results}, f, indent=2)
        print(f"\nwrote {args.out}")

    print("\nRead this as a curve, not a winner. If k=6 matches k=8 with no "
          "significant\ndifference but runs faster, k=6 is the better setting "
          "and the extra experts\nwere buying nothing.")


if __name__ == "__main__":
    main()
