"""Tiny, real, offline models for tests: a random Llama with a SentencePiece
BPE tokenizer that both transformers (fast tokenizer) and llama.cpp's
converter (tokenizer.model) understand, with identical token ids."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHAT_TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant|>{% endif %}")
WORDS = ("the a cat dog sat on mat paris france capital of is what who wrote hamlet shakespeare answer "
         "question yes no i don't know city river ocean blue red green one two three four five six seven "
         "eight nine ten hello world model think code python function return value number").split()


def fast_tokenizer_from_spm(model_file: str):
    """Llama-style fast tokenizer built from an SPM BPE model's pieces and scores."""
    import sentencepiece as spm
    from tokenizers import Tokenizer, decoders, models, normalizers
    from transformers import PreTrainedTokenizerFast
    sp = spm.SentencePieceProcessor(model_file=model_file)
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    # Merges: every multi-character piece, in score order, split into the two
    # existing pieces whose ranks are best (what SentencePiece BPE learned).
    merges = []
    for i in sorted(range(sp.get_piece_size()), key=lambda i: -sp.get_score(i)):
        p = sp.id_to_piece(i)
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_byte(i) or len(p) < 2:
            continue
        cands = [(vocab[p[:k]], vocab[p[k:]], k) for k in range(1, len(p)) if p[:k] in vocab and p[k:] in vocab]
        if cands:
            a, b, k = min(cands)
            merges.append((p[:k], p[k:]))
    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>", fuse_unk=True, byte_fallback=True))
    tk.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    tk.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(),
                                    decoders.Strip(" ", 1, 0)])
    t = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                pad_token="</s>")
    t.chat_template = CHAT_TEMPLATE
    return t, sp


def make_tiny_llama(out: Path, layers: int = 4, hidden: int = 64, ff: int = 128, seed: int = 0) -> Path:
    """Random Llama + tokenizer saved in HF format at out/hf. Returns that path."""
    import sentencepiece as spm
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    (out / "corpus.txt").write_text("\n".join(" ".join(rng.choice(WORDS, 12)) for _ in range(3000)))
    spm.SentencePieceTrainer.train(input=str(out / "corpus.txt"), model_prefix=str(out / "tok"), vocab_size=300,
                                   model_type="bpe", bos_id=1, eos_id=2, unk_id=0, pad_id=-1, byte_fallback=True,
                                   minloglevel=2)
    tok, sp = fast_tokenizer_from_spm(str(out / "tok.model"))
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=sp.get_piece_size(), hidden_size=hidden, intermediate_size=ff,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False)
    hf = out / "hf"
    LlamaForCausalLM(cfg).save_pretrained(hf, safe_serialization=True)
    tok.save_pretrained(hf)
    (hf / "tokenizer.model").write_bytes((out / "tok.model").read_bytes())
    cfgj = json.loads((hf / "tokenizer_config.json").read_text())
    cfgj.update(chat_template=CHAT_TEMPLATE, add_bos_token=True)
    (hf / "tokenizer_config.json").write_text(json.dumps(cfgj))
    return hf
