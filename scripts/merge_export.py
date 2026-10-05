#!/usr/bin/env python3
"""
Merge a fine-tuned LoRA adapter into its base model and convert to GGUF, or
convert the adapter alone to a GGUF LoRA that llama-server/Studio can scale
at runtime.

    # merged model -> GGUF -> quantized GGUF
    python scripts/merge_export.py --base org/base-model --adapter runs/adapter \\
        --out runs/merged --gguf Q4_K_M
    # adapter only (Studio's α slider then blends base and fine-tune live)
    python scripts/merge_export.py --base org/base-model --adapter runs/adapter \\
        --out runs/adapter-gguf --lora-gguf

Uses llama.cpp's own converters from $NS_LLAMA (scripts/build_llama_tools.sh)
and its llama-quantize. Writes a neuronscope-export.json next to the result.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def llama_dir(arg: str | None) -> Path:
    d = Path(arg or os.environ.get("NS_LLAMA", "~/llama.cpp")).expanduser()
    if not (d / "convert_hf_to_gguf.py").exists():
        raise SystemExit(f"no llama.cpp checkout at {d}; run scripts/build_llama_tools.sh or pass --llama")
    return d


def merge(base: str, adapter: str, out: Path, dtype: str) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base, dtype=getattr(torch, dtype))
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok_src = adapter if (Path(adapter) / "tokenizer_config.json").exists() else base
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(out)
    # SentencePiece models: llama.cpp's converter prefers the original file.
    for name in ("tokenizer.model",):
        for src in (Path(base), Path(adapter)):
            if (src / name).exists() and not (out / name).exists():
                shutil.copy(src / name, out / name)
    return out


def run(cmd: list[str]) -> None:
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit((r.stderr or r.stdout)[-2000:])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", required=True, help="HF id or path of the base model the adapter was trained on")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--llama", help="llama.cpp checkout (default $NS_LLAMA)")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--gguf", nargs="?", const="F16", metavar="QUANT",
                   help="also convert to GGUF; optional quant type such as Q4_K_M or Q8_0 (default F16)")
    p.add_argument("--lora-gguf", action="store_true", help="convert only the adapter to a GGUF LoRA")
    a = p.parse_args(argv)
    out = Path(a.out)
    meta = {"base": a.base, "adapter": a.adapter}
    if a.lora_gguf:
        ll = llama_dir(a.llama)
        out.mkdir(parents=True, exist_ok=True)
        dst = out / (Path(a.adapter).name + "-lora.gguf")
        run([sys.executable, str(ll / "convert_lora_to_gguf.py"), a.adapter, "--base", a.base,
             "--outfile", str(dst), "--outtype", "f16"])
        meta["lora_gguf"] = str(dst)
    else:
        if (Path(a.adapter) / "adapter_config.json").exists():
            merged = merge(a.base, a.adapter, out / "hf", a.dtype)
        else:
            # finetune.py --method full: already a complete model.
            merged = Path(a.adapter)
            if not (merged / "tokenizer.model").exists() and (Path(a.base) / "tokenizer.model").exists():
                shutil.copy(Path(a.base) / "tokenizer.model", merged / "tokenizer.model")
        meta["merged"] = str(merged)
        if a.gguf:
            ll = llama_dir(a.llama)
            f16 = out / "model-f16.gguf"
            run([sys.executable, str(ll / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(f16), "--outtype", "f16"])
            meta["gguf"] = str(f16)
            if a.gguf.upper() not in ("F16", "FP16"):
                q = out / f"model-{a.gguf.upper()}.gguf"
                run([str(ll / "build" / "bin" / "llama-quantize"), str(f16), str(q), a.gguf.upper()])
                meta["gguf"] = str(q)
    (out / "neuronscope-export.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
