"""gguf_tokenizer without a model file."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))



def test_gguf_tokenizer_decodes_byte_fallback():
    """SentencePiece vocabs spell characters they lack as <0xNN> byte tokens;
    those used to decode to nothing, dropping them from text and trace labels."""
    import gguf_tokenizer
    toks = ["<unk>", "<s>", "</s>", "▁the", "▁cat"] + [f"<0x{b:02X}>" for b in range(256)]
    t = gguf_tokenizer.GGufTokenizer(toks, "llama", None, 1, 2)
    byte = lambda b: 5 + b
    ids = [3, byte(ord("T")), 4] + [byte(b) for b in "🐈".encode()] + [byte(0xE2)]   # last: truncated
    p = t.pieces(ids)
    assert p[:3] == [" the", "T", " cat"]
    assert p[3:7] == ["", "", "", "🐈"]
    assert p[7] == "�"
    assert t.decode(ids) == "".join(p) == " theT cat🐈�"
    assert t.pieces([1, 2]) == ["", ""]                # control tokens stay hidden
