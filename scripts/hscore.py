#!/usr/bin/env python3
"""
Score one reply with an H-Neuron classifier: how strongly did the neurons
associated with confident wrong answers fire while the model wrote it?

    python scripts/hscore.py --gguf model.gguf --classifier models/classifier.npz \\
        --prompt "Who wrote Hamlet?" --response "Christopher Marlowe."

One prefill over prompt + response with llama-cett-dump, CETT averaged over the
response tokens, then the classifier's linear score. Uses the GGUF's own chat
template and tokenizer, so no transformers or torch.

Read the number as a weak, relative signal. The classifier was trained on
short factual answers (the answer-token span); here it sees a whole reply,
and reported AUROCs are in the 0.65-0.75 range. Track it over many replies
(model_stats.py does) rather than trusting any single score.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class HScorer:
    def __init__(self, binary: str, gguf: str, classifier: str, ngl: int = 99, batch: int = 4096,
                 device: str = "", timeout: float = 600, tokenizer=None):
        self.binary, self.gguf, self.ngl, self.batch, self.device = binary, gguf, ngl, batch, device
        self.timeout = timeout
        blob = np.load(classifier)
        self.coef = np.asarray(blob["coef"], dtype=np.float32)
        self.intercept = float(blob["intercept"])
        self.n_layers = int(blob["n_layers"]) if "n_layers" in blob else None
        self.n_neurons = int(blob["n_neurons"]) if "n_neurons" in blob else None
        if self.n_layers and self.n_neurons and self.coef.size != self.n_layers * self.n_neurons:
            raise ValueError("classifier coefficient size does not match its n_layers x n_neurons "
                             "(MoE classifiers are not supported for live scoring)")
        self._norms = None
        self._tok = tokenizer        # anything with render_chat/decode/has_template
        self.lock = threading.Lock()   # one prefill at a time per model

    def _tokenizer(self):
        if self._tok is None:
            import gguf_tokenizer
            self._tok = gguf_tokenizer.load(self.gguf)
            if not self._tok.has_template:
                raise ValueError("this GGUF has no chat template; cannot render the prompt")
        return self._tok

    def _col_norms(self, n_layers):
        if self._norms is None:
            from extract_activations_gguf import weight_col_norms
            norms = weight_col_norms(self.gguf, n_layers)
            self._norms = np.stack([norms[i] for i in range(n_layers)])
        return self._norms

    def _run(self, args):
        cmd = [self.binary, "-m", self.gguf, *args]
        if self.device:
            cmd += ["--device", self.device]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"cett-dump failed: {(r.stderr or r.stdout)[-400:]}")

    def score(self, messages: list[dict], response: str) -> dict:
        from extract_activations_gguf import read_aggregate, read_tokens
        tok = self._tokenizer()
        prompt = tok.render_chat(messages, True)
        text = prompt + response
        with self.lock, tempfile.TemporaryDirectory(prefix="hscore-") as work:
            m1 = os.path.join(work, "t.jsonl")
            with open(m1, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "p", "text": prompt}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"id": "r", "text": text}, ensure_ascii=False) + "\n")
            self._run(["--tokenize-only", "-ngl", "0", "-c", "4096", "--manifest", m1, "--outdir", work])
            ids = read_tokens(os.path.join(work, "r.toks"))
            p_ids = read_tokens(os.path.join(work, "p.toks"))
            n_tok = len(ids)
            if n_tok > self.batch:
                raise ValueError(f"{n_tok} tokens exceeds batch {self.batch}")
            # The response starts where the prompt's own tokenization ends. If
            # the boundary merged into one token, back off to the common prefix.
            start = 0
            while start < min(len(p_ids), n_tok) and p_ids[start] == ids[start]:
                start += 1
            if start >= n_tok:
                raise ValueError("empty response region")
            m2 = os.path.join(work, "s.jsonl")
            with open(m2, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "r", "text": text, "spans": [[start, -1]]}, ensure_ascii=False) + "\n")
            n_layers = self.n_layers or self._layers_from_gguf()
            self._run(["-ngl", str(self.ngl), "-b", str(self.batch), "-c", str(self.batch),
                       "--n-layers", str(n_layers), "--manifest", m2, "--outdir", work])
            _, agg, _seen, _counts, n_experts = read_aggregate(os.path.join(work, "r.bin"))
        if n_experts > 1:
            raise ValueError("MoE models are not supported for live scoring")
        feats = (agg[0] * self._col_norms(n_layers)).ravel().astype(np.float32)
        if feats.size != self.coef.size:
            raise ValueError(f"feature size {feats.size} != classifier {self.coef.size}: wrong classifier for this model?")
        s = float(feats @ self.coef + self.intercept)
        return {"score": s, "prob": 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, s)))),
                "n_tokens": int(n_tok - start)}

    def _layers_from_gguf(self) -> int:
        import gguf
        r = gguf.GGUFReader(self.gguf)
        arch = r.fields["general.architecture"].contents()
        arch = arch.decode() if isinstance(arch, bytes) else arch
        self.n_layers = int(r.fields[f"{arch}.block_count"].contents())
        return self.n_layers


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--binary", default=os.environ.get("NS_CETT"))
    p.add_argument("--gguf", default=os.environ.get("NS_GGUF"))
    p.add_argument("--classifier", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--response", required=True)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--batch", type=int, default=4096)
    a = p.parse_args(argv)
    if not a.binary or not a.gguf:
        raise SystemExit("need --binary and --gguf (or source env.sh)")
    s = HScorer(a.binary, a.gguf, a.classifier, a.ngl, a.batch)
    print(json.dumps(s.score([{"role": "user", "content": a.prompt}], a.response), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
