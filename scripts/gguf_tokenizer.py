#!/usr/bin/env python3
"""
Read the tokenizer out of a GGUF, so the pipeline does not need transformers.

transformers imports torch at module load, which drags a multi-hundred-MB
dependency into a pipeline whose only use for it is turning token ids into
strings and rendering a chat template. Both of those live in the GGUF already.

    python scripts/gguf_tokenizer.py --gguf model.gguf --show

Provides:
    load(path)                  -> GGufTokenizer
    .decode_id(i)               -> the piece for one token id
    .decode(ids)                -> the concatenated string
    .render_chat(messages)      -> the prompt text, via the embedded template
    .has_template

What it deliberately does NOT do is tokenize text. Turning a string into ids
needs the full BPE merge logic, and llama.cpp already does that correctly --
cett-dump's --tokenize-only pass is the tokenizer. This is the inverse
direction only, which is all the pipeline needs.
"""

import argparse
import json
import sys

# SentencePiece marks a word boundary with U+2581; byte-level BPE uses the
# GPT-2 byte map, where U+0120 is a leading space.
_SP_SPACE = "\u2581"
_BPE_SPACE = "\u0120"


def _decode_gpt2_bytes(s):
    """Undo the GPT-2 byte-to-unicode mapping used by byte-level BPE vocabs."""
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("\u00a1"), ord("\u00ac") + 1)) + \
         list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    rev = {chr(c): b for b, c in zip(bs, cs)}
    try:
        return bytes(rev[ch] for ch in s).decode("utf-8", errors="replace")
    except KeyError:
        return s


class GGufTokenizer:
    def __init__(self, tokens, model, chat_template, bos, eos):
        self.tokens = tokens
        self.model = model or ""
        self.chat_template = chat_template
        self.bos_id = bos
        self.eos_id = eos
        # Byte-level BPE vocabs contain the GPT-2 space marker; SentencePiece
        # ones contain U+2581. Detect once rather than guessing per token.
        sample = "".join(tokens[:2000])
        self._bpe = _BPE_SPACE in sample
        self._sp = _SP_SPACE in sample

    @property
    def has_template(self):
        return bool(self.chat_template)

    def decode_id(self, i):
        i = int(i)
        if i < 0 or i >= len(self.tokens):
            return ""
        t = self.tokens[i]
        if t.startswith("<") and t.endswith(">") and len(t) > 2:
            return ""            # control token, contributes no text
        if self._bpe:
            return _decode_gpt2_bytes(t)
        if self._sp:
            return t.replace(_SP_SPACE, " ")
        return t

    def decode(self, ids):
        return "".join(self.decode_id(i) for i in ids)

    def pieces(self, ids):
        return [self.decode_id(i) for i in ids]

    def render_chat(self, messages, add_generation_prompt=True):
        if not self.chat_template:
            raise SystemExit(
                "this GGUF carries no chat template; pass --tokenizer to use "
                "a transformers tokenizer instead")
        try:
            from jinja2 import Template
        except ImportError:
            raise SystemExit("rendering the chat template needs jinja2: "
                             "pip install jinja2")
        # llama.cpp templates use raise_exception and sometimes strftime_now.
        import datetime

        def _raise(msg):
            raise RuntimeError(msg)

        tpl = Template(self.chat_template, trim_blocks=True,
                       lstrip_blocks=True)
        return tpl.render(messages=messages,
                          add_generation_prompt=add_generation_prompt,
                          bos_token=self.tokens[self.bos_id]
                          if self.bos_id is not None else "",
                          eos_token=self.tokens[self.eos_id]
                          if self.eos_id is not None else "",
                          raise_exception=_raise,
                          strftime_now=lambda f: datetime.datetime.now()
                          .strftime(f))


def load(path):
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
                return bytes(f.parts[f.data[0]]).decode("utf-8")
            except Exception:
                return None

    f = r.fields.get("tokenizer.ggml.tokens")
    if f is None:
        raise SystemExit(f"{path} has no tokenizer.ggml.tokens")
    tokens = [bytes(f.parts[i]).decode("utf-8", errors="replace")
              for i in f.data]

    return GGufTokenizer(
        tokens=tokens,
        model=kv("tokenizer.ggml.model"),
        chat_template=kv("tokenizer.chat_template"),
        bos=kv("tokenizer.ggml.bos_token_id"),
        eos=kv("tokenizer.ggml.eos_token_id"),
    )


def find_answer_span(pieces, answer, start=0):
    """Locate `answer` in the decoded pieces, returning a token range.

    Works on the concatenated string rather than a pre-tokenized answer, so
    stage 2 is not needed: the answer text as generated is matched against the
    text the tokens decode to, and the covering token range is returned.
    """
    if not answer or not answer.strip():
        return None
    # Character offset of each token, from `start` onwards.
    offs, acc = [], 0
    for i in range(start, len(pieces)):
        offs.append((acc, acc + len(pieces[i]), i))
        acc += len(pieces[i])
    hay = "".join(pieces[start:])

    target = answer.strip()
    pos = hay.find(target)
    if pos < 0:
        # Fall back to a whitespace-insensitive search, since a tokenizer may
        # split differently than the stored answer string was spaced.
        squash = lambda s: "".join(s.split())
        sq_hay, sq_t = squash(hay), squash(target)
        j = sq_hay.find(sq_t)
        if j < 0:
            return None
        # Map the squashed offset back by counting non-space characters.
        seen = 0
        pos = None
        for k, ch in enumerate(hay):
            if not ch.isspace():
                if seen == j:
                    pos = k
                    break
                seen += 1
        if pos is None:
            return None
        end = pos + len(target)
    else:
        end = pos + len(target)

    lo = hi = None
    for a, b, i in offs:
        if lo is None and b > pos:
            lo = i
        if a < end:
            hi = i + 1
    return (lo, hi) if lo is not None and hi is not None and hi > lo else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    t = load(a.gguf)
    print(f"vocab       : {len(t.tokens):,} tokens ({t.model})")
    print(f"bos / eos   : {t.bos_id} / {t.eos_id}")
    print(f"encoding    : {'byte-level BPE' if t._bpe else 'sentencepiece' if t._sp else 'plain'}")
    print(f"chat template: {'yes' if t.has_template else 'NO'}")
    if a.show and t.has_template:
        print("\n" + t.chat_template[:600])
        print("\nrendered sample:")
        print(repr(t.render_chat([{"role": "user", "content": "Hi?"}])[:300]))


if __name__ == "__main__":
    main()
