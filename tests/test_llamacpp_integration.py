"""Real llama.cpp integration: compiled cett-dump against PyTorch hooks.

Skipped unless NS_LLAMA points at a llama.cpp checkout built with
scripts/build_llama_tools.sh (needs sentencepiece). Builds a tiny random Llama,
converts it to GGUF with llama.cpp's own converter, and checks that
cett-dump's CETT matches ns_common.CETTManager on every layer, and that
hscore scores a reply end to end."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
LLAMA = Path(os.environ.get("NS_LLAMA", "")).expanduser()
CETT = LLAMA / "build" / "bin" / "llama-cett-dump"
pytestmark = pytest.mark.skipif(not CETT.exists(), reason="set NS_LLAMA to a llama.cpp build with cett-dump")


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory):
    spm = pytest.importorskip("sentencepiece")
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaForCausalLM
    d = tmp_path_factory.mktemp("tiny")
    words = "the cat dog sat on mat who wrote hamlet shakespeare paris capital france yes no know".split()
    rng = np.random.default_rng(0)
    (d / "corpus.txt").write_text("\n".join(" ".join(rng.choice(words, 10)) for _ in range(2000)))
    spm.SentencePieceTrainer.train(input=str(d / "corpus.txt"), model_prefix=str(d / "tok"), vocab_size=300,
                                   model_type="bpe", bos_id=1, eos_id=2, unk_id=0, pad_id=-1, byte_fallback=True)
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      bos_token_id=1, eos_token_id=2, tie_word_embeddings=False)
    hf = d / "hf"
    LlamaForCausalLM(cfg).save_pretrained(hf, safe_serialization=True)
    (hf / "tokenizer.model").write_bytes((d / "tok.model").read_bytes())
    (hf / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "LlamaTokenizer", "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>",
        "chat_template": "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}<|assistant|>{% endif %}"}))
    out = d / "tiny-f32.gguf"
    subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(hf), "--outfile", str(out),
                    "--outtype", "f32"], check=True, capture_output=True)
    return hf, out


def test_cett_dump_matches_pytorch(tiny_gguf, tmp_path):
    import torch
    from transformers import LlamaForCausalLM
    import ns_common
    from extract_activations_gguf import read_aggregate, read_tokens, weight_col_norms
    hf, gguf = tiny_gguf
    text = "the cat sat on the mat and the dog wrote hamlet"
    (tmp_path / "m.jsonl").write_text(json.dumps({"id": "x", "text": text}) + "\n")
    subprocess.run([str(CETT), "-m", str(gguf), "--tokenize-only", "-ngl", "0", "-c", "512",
                    "--manifest", str(tmp_path / "m.jsonl"), "--outdir", str(tmp_path)], check=True, capture_output=True)
    ids = read_tokens(tmp_path / "x.toks")
    (tmp_path / "s.jsonl").write_text(json.dumps({"id": "x", "text": text, "spans": [[0, -1], [2, 6]]}) + "\n")
    r = subprocess.run([str(CETT), "-m", str(gguf), "-ngl", "0", "-b", "512", "-c", "512", "--n-layers", "4",
                        "--manifest", str(tmp_path / "s.jsonl"), "--outdir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0 and "left unseen" not in r.stderr, r.stderr[-800:]
    _, agg, seen, _, _ = read_aggregate(tmp_path / "x.bin")
    assert seen.all(), "every layer, including the last, must be captured"
    wn = np.stack([weight_col_norms(str(gguf), 4)[l] for l in range(4)])
    m = LlamaForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
    mgr = ns_common.CETTManager(m)
    with torch.no_grad():
        m(torch.tensor([ids.tolist()]))
    ref = mgr.cett().numpy()
    for si, (a, b) in enumerate([(0, len(ids)), (2, 6)]):
        want = ref[:, a:b, :].mean(1)
        got = agg[si] * wn
        assert np.abs(got - want).max() / np.abs(want).max() < 5e-3


def test_hscore_end_to_end(tiny_gguf, tmp_path):
    import hscore
    _, gguf = tiny_gguf
    coef = np.random.default_rng(1).normal(size=4 * 128).astype(np.float32)
    np.savez(tmp_path / "clf.npz", coef=coef, intercept=0.0, n_layers=4, n_neurons=128)
    sc = hscore.HScorer(str(CETT), str(gguf), str(tmp_path / "clf.npz"), ngl=0, batch=512)
    res = sc.score([{"role": "user", "content": "who wrote hamlet"}], "shakespeare wrote hamlet")
    assert res["n_tokens"] > 0 and np.isfinite(res["score"]) and 0 < res["prob"] < 1
