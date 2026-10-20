#!/usr/bin/env python3
"""
Steering through low-dimensional random projections.

Suppressing single H-Neurons can damage what else those neurons do, such as
grammar or formatting. If the signal that separates confident wrong answers
from right ones is spread over many neurons and is strong, it survives a
random projection to a few dozen dimensions. In that small space it can be
estimated from a few hundred examples, and steered as a whole without
touching any single neuron.

The projection is not free. Against noise that is the same in every
direction, a projection to k of d dimensions keeps about sqrt(k/d) of one
direction's signal. A weak, narrow signal fades, and only a strong,
spread-out one survives. Real activations are far from isotropic, so
measure rather than assume: fit reports each layer's held-out AUC in the
projection next to the same classifier on the full space, and the spread
over several random projections. A large gap means raising --dim, or that
the signal is not redundant in this model.

  fit      for each chosen layer, project the MLP output (averaged over the
           reply's tokens) onto k random orthonormal directions R (d x k), and
           find the direction w in that space that best separates wrong from
           right answers (LDA with shrinkage). Held-out AUC per layer, and its
           spread over several random projections, say whether the signal is
           real and redundant. -> rp.pt
  steer    decode_lab.py variant "rp" (Hugging Face models): at every step the
           layer's MLP output h is projected, s = h.(R w); when s drifts past
           the right answers' typical value, it is pulled back:
             direction:  h -= alpha * max(0, s - c_right) * (R w)
             subspace:   h -= alpha * gate * R (R^T h - mu_right)
           where gate = sigmoid((s - c_mid) / tau). The steering acts only when
           the drift is there.
  export   the lifted direction as a llama.cpp control vector GGUF
           (--control-vector-scaled FILE SCALE). This is fixed, not gated:
           llama.cpp adds the same vector at every step.

Data: JSONL of {"prompt" or "messages", "reply", "label": "right"|"wrong"},
or TestQA results (--testqa runs/x.json --endpoint LABEL --tasks qa/bank),
whose correct and wrong verdicts are used.

    python scripts/rp_steer.py fit --model org/m --data pairs.jsonl --layers 10 14 18 --dim 48 --out runs/rp.pt
    python scripts/decode_lab.py --model org/m --variant plain='{}' \\
        --variant rp='{"rp": {"file": "runs/rp.pt", "alpha": 1.0, "mode": "direction"}}'
    python scripts/rp_steer.py export --file runs/rp.pt --out runs/rp-cvec.gguf
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RIGHT = {"right", "correct", "true", "t", "1", "yes"}
WRONG = {"wrong", "hallucinated", "false", "f", "0", "no"}


# ---------------------------------------------------------------- data

def read_pairs(path: str) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        lab = str(r.get("label")).lower()
        if lab not in RIGHT | WRONG:
            continue
        msgs = r.get("messages") or [{"role": "user", "content": r.get("prompt", "")}]
        out.append({"messages": msgs, "reply": r.get("reply", ""), "y": int(lab in WRONG)})
    return out


def from_testqa(results: str, endpoint: str | None, tasks: list[str]) -> list[dict]:
    """TestQA's correct (right) and wrong (wrong) answers, with the prompts from the task bank."""
    import testqa
    bank = {t["id"]: t for t in testqa.load_tasks(tasks, None, None, 0)}
    raw = json.loads(Path(results).read_text())["raw"]
    label = endpoint or next(iter(raw))
    out = []
    for r in raw[label]:
        if r.get("verdict") not in ("correct", "wrong") or r["id"] not in bank or not r.get("text"):
            continue
        out.append({"messages": [{"role": "user", "content": bank[r["id"]]["prompt"]}], "reply": r["text"],
                    "y": int(r["verdict"] == "wrong")})
    return out


# ---------------------------------------------------------------- linear algebra (torch)

def random_basis(d: int, k: int, seed: int):
    """d x k with orthonormal columns: a Gaussian matrix through QR."""
    import torch
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
    return q


def lda(z, y, shrink: float = 0.1):
    """Direction separating y=1 from y=0 in z [n, k]: (S + shrink*tr(S)/k I)^-1 (mu1 - mu0), unit length."""
    import torch
    z0, z1 = z[y == 0], z[y == 1]
    mu0, mu1 = z0.mean(0), z1.mean(0)
    c = torch.cat([z0 - mu0, z1 - mu1])
    s = c.T @ c / max(1, len(c) - 2)
    k = s.shape[0]
    s = s + shrink * (torch.trace(s) / k) * torch.eye(k)
    w = torch.linalg.solve(s, mu1 - mu0)
    return w / w.norm().clamp(min=1e-12), mu0, mu1


def auc(scores, y) -> float:
    """Probability that a wrong answer scores above a right one (ties count half)."""
    pos = [float(s) for s, t in zip(scores, y) if t == 1]
    neg = [float(s) for s, t in zip(scores, y) if t == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def split(n: int, frac: float, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = max(1, int(n * frac))
    return idx[k:], idx[:k]


def fit_layer(h, y, dim: int, seeds: list[int], shrink: float, test_frac: float, seed: int) -> dict:
    """h [n, d] mean MLP outputs for one layer. Fits on a training split and scores the held-out
    part, for each random projection; keeps the projection with the best held-out AUC."""
    import torch
    tr, te = split(len(y), test_frac, seed)
    tr_t, te_t = torch.tensor(tr), torch.tensor(te)
    ytr, yte = y[tr_t], y[te_t]
    if int(ytr.sum()) < 2 or int((1 - ytr).sum()) < 2:
        raise SystemExit("need at least two right and two wrong answers in the training split")
    best, aucs = None, []
    for s in seeds:
        R = random_basis(h.shape[1], dim, s)
        z = h @ R
        w, mu0, mu1 = lda(z[tr_t], ytr, shrink)
        a = auc((z[te_t] @ w).tolist(), yte.tolist())
        aucs.append(a)
        if best is None or a > best["auc"]:
            score_tr = z[tr_t] @ w
            c_right = float(score_tr[ytr == 0].mean())
            c_wrong = float(score_tr[ytr == 1].mean())
            best = {"seed": s, "R": R, "w": w, "mu_right": mu0, "mu_wrong": mu1, "auc": a,
                    "c_right": c_right, "c_wrong": c_wrong,
                    "tau": max(1e-3, float(score_tr.std()) / 2)}
    # the same classifier on the full space, for comparison: is the projection losing anything?
    wf, _, _ = lda(h[tr_t], ytr, shrink)
    full = auc((h[te_t] @ wf).tolist(), yte.tolist())
    finite = [a for a in aucs if not math.isnan(a)]
    best.update(aucs=[round(a, 4) for a in aucs], auc_full=round(full, 4),
                auc_spread=round(max(finite) - min(finite), 4) if finite else None)
    return best


# ---------------------------------------------------------------- model side

def mlp_of(layer):
    for name in ("mlp", "feed_forward", "ffn", "block_sparse_moe"):
        m = getattr(layer, name, None)
        if m is not None:
            return m
    raise ValueError("cannot find this layer's MLP")


def collect(model, tok, pairs: list[dict], layers: list[int]):
    """Mean MLP output over each reply's tokens, per layer -> {layer: [n, d]}."""
    import torch
    from concept_edit import decoder_layers
    from logit_lens import encode
    dec = decoder_layers(model)
    grab, hooks = {}, []
    for l in layers:
        def hook(_m, _i, out, l=l):
            grab[l] = (out[0] if isinstance(out, tuple) else out).detach()
        hooks.append(mlp_of(dec[l]).register_forward_hook(hook))
    dev = next(model.parameters()).device
    rows = {l: [] for l in layers}
    try:
        with torch.no_grad():
            for i, p in enumerate(pairs):
                ids, start = encode(tok, None, p["messages"], p["reply"])
                model(torch.tensor([ids], device=dev))
                for l in layers:
                    rows[l].append(grab[l][0, start:].float().mean(0).cpu())
                if (i + 1) % 25 == 0:
                    print(f"  {i + 1}/{len(pairs)}", flush=True)
    finally:
        for h in hooks:
            h.remove()
    return {l: torch.stack(v) for l, v in rows.items()}


class Steerer:
    """decode_lab.py's "rp" variant: hooks on the fitted layers' MLPs."""

    def __init__(self, path: str, alpha: float = 1.0, mode: str = "direction", layers=None, device=None):
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=False)
        self.alpha, self.mode = float(alpha), mode
        if mode not in ("direction", "subspace"):
            raise ValueError("rp mode: direction or subspace")
        keep = None if layers in (None, "all") else {int(x) for x in layers}
        self.parts = {}
        for l, p in obj["layers"].items():
            if keep is not None and int(l) not in keep:
                continue
            R = p["R"].float().to(device)
            u = R @ p["w"].float().to(device)
            self.parts[int(l)] = {"R": R, "u": u / u.norm().clamp(min=1e-12), "mu": p["mu_right"].float().to(device),
                                  "c_right": p["c_right"], "c_mid": (p["c_right"] + p["c_wrong"]) / 2, "tau": p["tau"]}
        self.last = {}
        self.hooks = []

    def attach(self, decoder_layers) -> None:
        for l, p in self.parts.items():
            self.hooks.append(mlp_of(decoder_layers[l]).register_forward_hook(self._hook(l, p)))

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _hook(self, l, p):
        import torch

        def f(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            s = hf @ p["u"]                                       # [B, T]
            self.last[l] = float(s[0, -1])
            if self.mode == "direction":
                push = torch.clamp(s - p["c_right"], min=0)
                hf = hf - self.alpha * push[..., None] * p["u"]
            else:
                gate = torch.sigmoid((s - p["c_mid"]) / p["tau"])
                z = hf @ p["R"] - p["mu"]
                hf = hf - self.alpha * gate[..., None] * (z @ p["R"].T)
            h2 = hf.to(h.dtype)
            return (h2, *out[1:]) if isinstance(out, tuple) else h2
        return f

    def flagged(self) -> float:
        """How far past the right answers' level the newest token is, averaged over layers (0 = not at all)."""
        if not self.last:
            return 0.0
        return sum(max(0.0, self.last[l] - self.parts[l]["c_right"]) for l in self.last) / len(self.last)


# ---------------------------------------------------------------- commands

def cmd_fit(a) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    pairs = from_testqa(a.testqa, a.endpoint, a.tasks) if a.testqa else read_pairs(a.data)
    y = torch.tensor([p["y"] for p in pairs])
    print(f"{len(pairs)} answers: {int((y == 0).sum())} right, {int(y.sum())} wrong")
    if int(y.sum()) < 4 or int((y == 0).sum()) < 4:
        raise SystemExit("need at least four right and four wrong answers")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype="auto" if a.dtype == "auto" else getattr(torch, a.dtype))
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    from concept_edit import decoder_layers
    n_layers = len(decoder_layers(model))
    layers = a.layers or [n_layers // 2, (3 * n_layers) // 4]
    feats = collect(model, tok, pairs, layers)
    seeds = [a.seed + i for i in range(a.projections)]
    out = {"model": a.model, "dim": a.dim, "shrink": a.shrink, "n": len(pairs), "layers": {}}
    lines = [f"{'layer':>5} {'AUC':>6} {'full-d':>7} {'spread':>7}  over {a.projections} random projections to {a.dim} dims"]
    for l in layers:
        r = fit_layer(feats[l], y, a.dim, seeds, a.shrink, a.test_frac, a.seed)
        out["layers"][l] = r
        lines.append(f"{l:>5} {r['auc']:>6.3f} {r['auc_full']:>7.3f} {r['auc_spread']:>7.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    summary = {k: v for k, v in out.items() if k != "layers"}
    summary["layers"] = {l: {k: v for k, v in r.items() if k not in ("R", "w", "mu_right", "mu_wrong")}
                         for l, r in out["layers"].items()}
    Path(a.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))
    print("AUC 0.5 is chance. A small spread across projections means the signal is redundant, as the method "
          "assumes; an AUC well below the full-d one means the projection is too small (raise --dim).")
    print(f"wrote {a.out}")
    return 0


def cmd_export(a) -> int:
    """The lifted direction of each layer as a llama.cpp control vector (adds -u: away from wrong answers)."""
    import torch
    try:
        import gguf
    except ImportError:
        raise SystemExit("pip install gguf (the package that ships with llama.cpp)")
    obj = torch.load(a.file, map_location="cpu", weights_only=False)
    w = gguf.GGUFWriter(a.out, "controlvector")
    w.add_string("controlvector.model_hint", str(obj.get("arch") or "llama"))
    w.add_uint32("controlvector.layer_count", len(obj["layers"]))
    for l, p in obj["layers"].items():
        if int(l) < 1:
            print(f"skipping layer {l}: llama.cpp control vectors start at layer 1")
            continue
        u = p["R"].float() @ p["w"].float()
        u = -(u / u.norm()) * float(p["c_wrong"] - p["c_right"])        # scale 1 = one class gap toward right
        w.add_tensor(f"direction.{int(l)}", u.numpy().astype("float32"))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}: llama-server --control-vector-scaled {a.out} 0.5 (fixed: not gated like decode_lab's rp)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="random projections and a separating direction per layer")
    f.add_argument("--model", required=True)
    f.add_argument("--data", help="JSONL of prompt/messages, reply, label")
    f.add_argument("--testqa", help="or TestQA results JSON")
    f.add_argument("--endpoint", help="which endpoint's answers in the TestQA results")
    f.add_argument("--tasks", nargs="+", default=[str(Path(__file__).resolve().parent.parent / "qa" / "bank")])
    f.add_argument("--layers", type=int, nargs="+", help="default: at one half and three quarters of the depth")
    f.add_argument("--dim", type=int, default=48, help="dimensions of the random projection")
    f.add_argument("--projections", type=int, default=5, help="random projections tried per layer")
    f.add_argument("--shrink", type=float, default=0.1)
    f.add_argument("--test-frac", type=float, default=0.25)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    f.add_argument("--out", required=True)
    x = sub.add_parser("export", help="a llama.cpp control vector GGUF (fixed, not gated)")
    x.add_argument("--file", required=True)
    x.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "fit" and not (a.data or a.testqa):
        raise SystemExit("give --data or --testqa")
    return cmd_fit(a) if a.cmd == "fit" else cmd_export(a)


if __name__ == "__main__":
    sys.exit(main())
