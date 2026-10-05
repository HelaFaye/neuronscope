#!/usr/bin/env python3
"""
Local fine-tuning on measured deficits: SFT, then optional DPO, with QLoRA,
LoRA or full-parameter training, on one GPU or several.

    # 1. data (see deficits.py)
    python scripts/deficits.py --results runs/base.json --cache runs/c --out runs/retrain ...
    # 2-3. train (QLoRA: 4-bit base + low-rank adapters; only the adapters learn)
    python scripts/finetune.py --model org/base-model --data runs/retrain --method qlora --out runs/adapter
    # several GPUs (LoRA/QLoRA -> data parallel; --method full --fsdp -> sharded)
    python scripts/finetune.py --launch 4 --model ... --data runs/retrain --method lora --out runs/adapter
    # 4. merge and convert (see merge_export.py)
    python scripts/merge_export.py --base org/base-model --adapter runs/adapter --out runs/merged --gguf Q4_K_M

Data: `sft.jsonl` (chat `messages`) and optionally `dpo.jsonl` (prompt /
chosen / rejected) in --data. SFT loss is computed on the assistant reply
only. DPO runs after SFT, on top of the SFT adapter, when --dpo is given.

Which method:
  qlora   base weights in 4-bit NF4, adapters in 16-bit. Least memory; needs a
          CUDA GPU that bitsandbytes supports.
  lora    base weights in 16-bit (or 32 on CPU). Works on ROCm, CPU and older
          GPUs where 4-bit kernels are unavailable.
  full    every weight trains. For knowledge (factual) deficits. Needs
          several times the model size in memory; use --fsdp across GPUs.

Hardware notes: bf16 is used where the GPU supports it, fp16 otherwise (e.g.
Maxwell/Pascal cards such as the Tesla M10 have no bf16 and slow fp16; check
that `torch.cuda.get_arch_list()` still includes your card's sm_XX, since
recent PyTorch wheels drop old architectures). `--check` prints all of this
without training.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def hardware_report() -> dict:
    import torch
    rep = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "devices": []}
    if torch.cuda.is_available():
        arch = torch.cuda.get_arch_list()
        rep["arch_list"] = arch
        rep["hip"] = getattr(torch.version, "hip", None)
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            cap = f"sm_{p.major}{p.minor}"
            rep["devices"].append({"index": i, "name": p.name, "capability": cap,
                                   "memory_gib": round(p.total_memory / 2**30, 1),
                                   "bf16": torch.cuda.is_bf16_supported() if i == 0 else None,
                                   "supported_by_this_torch": rep["hip"] is not None or cap in arch
                                   or any(a.startswith("compute_") for a in arch)})
    try:
        import bitsandbytes  # noqa: F401
        rep["bitsandbytes"] = True
    except Exception:
        rep["bitsandbytes"] = False
    return rep


def precision(rep: dict) -> dict:
    if not rep["cuda"]:
        return {"bf16": False, "fp16": False}
    if rep["devices"] and rep["devices"][0]["bf16"]:
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": True}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def sft_dataset(path: Path):
    """Chat records -> conversational prompt/completion, so loss covers only the reply."""
    from datasets import Dataset
    rows = []
    for r in load_jsonl(path):
        msgs = r["messages"]
        if not msgs or msgs[-1]["role"] != "assistant":
            continue
        rows.append({"prompt": msgs[:-1], "completion": [msgs[-1]]})
    if not rows:
        raise SystemExit(f"no usable SFT records in {path}")
    return Dataset.from_list(rows)


def dpo_dataset(path: Path):
    from datasets import Dataset
    rows = [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in load_jsonl(path)]
    return Dataset.from_list(rows)


def load_model(a, rep):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prec = precision(rep)
    dtype = torch.bfloat16 if prec["bf16"] else torch.float16 if prec["fp16"] else torch.float32
    kw = {"dtype": dtype}
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if a.method == "qlora":
        if not rep["cuda"] or not rep["bitsandbytes"]:
            raise SystemExit("qlora needs a CUDA GPU and bitsandbytes; use --method lora on ROCm or CPU")
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kw["device_map"] = {"": local_rank if local_rank >= 0 else 0}
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    if a.method == "qlora":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=a.gradient_checkpointing)
    return model, tok, prec


def lora_config(a):
    if a.method == "full":
        return None
    from peft import LoraConfig
    targets = "all-linear" if a.lora_targets == "all-linear" else a.lora_targets.split(",")
    return LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                      target_modules=targets, task_type="CAUSAL_LM")


def common_args(a, prec, out: Path, model) -> dict:
    kw = dict(output_dir=str(out), per_device_train_batch_size=a.batch_size,
              gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, num_train_epochs=a.epochs,
              max_steps=a.max_steps, logging_steps=a.logging_steps, save_strategy="no", report_to=[],
              gradient_checkpointing=a.gradient_checkpointing, seed=a.seed, max_length=a.max_length,
              bf16=prec["bf16"], fp16=prec["fp16"], lr_scheduler_type="cosine", warmup_ratio=0.03,
              warmup_steps=max(1, int(0.03 * a.max_steps)) if a.max_steps > 0 else 10)
    if a.fsdp:
        layers = list(getattr(model, "_no_split_modules", None) or [])
        kw["fsdp"] = "full_shard auto_wrap"
        kw["fsdp_config"] = {"transformer_layer_cls_to_wrap": layers} if layers else {}
    return kw


def config(cls, **kw):
    """Build a trl/transformers config, dropping arguments this version does
    not know (e.g. warmup_ratio was removed in transformers 5)."""
    import inspect
    known = set(inspect.signature(cls.__init__).parameters)
    if "warmup_ratio" in known:
        kw.pop("warmup_steps", None)
    return cls(**{k: v for k, v in kw.items() if k in known})


def train(a) -> int:
    rep = hardware_report()
    if a.fsdp and a.method != "full":
        print("note: --fsdp is meant for --method full; adapters train fine with plain data parallelism")
    model, tok, prec = load_model(a, rep)
    data = Path(a.data)
    out = Path(a.out)
    from trl import SFTConfig, SFTTrainer
    peft_cfg = lora_config(a)
    sft_args = config(SFTConfig, **common_args(a, prec, out / "sft-run", model), completion_only_loss=True)
    trainer = SFTTrainer(model=model, args=sft_args, train_dataset=sft_dataset(data / "sft.jsonl"),
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    model = trainer.model
    sft_dir = out if not a.dpo else out / "sft"
    model.save_pretrained(sft_dir)
    tok.save_pretrained(sft_dir)

    if a.dpo:
        if not (data / "dpo.jsonl").exists() or not (data / "dpo.jsonl").read_text().strip():
            raise SystemExit("--dpo given but dpo.jsonl is empty")
        from trl import DPOConfig, DPOTrainer
        dkw = common_args(a, prec, out / "dpo-run", model)
        dkw.update(learning_rate=a.dpo_lr, num_train_epochs=a.dpo_epochs, max_steps=a.dpo_max_steps)
        # With adapters, the reference policy is the same model with adapters
        # disabled, so no second copy of the weights is loaded.
        dtrainer = DPOTrainer(model=model, ref_model=None, args=config(DPOConfig, **dkw, beta=a.dpo_beta),
                              train_dataset=dpo_dataset(data / "dpo.jsonl"), processing_class=tok)
        dtrainer.train()
        dtrainer.model.save_pretrained(out)
        tok.save_pretrained(out)
    meta = {"base": a.model, "method": a.method, "data": str(data), "dpo": bool(a.dpo),
            "lora": None if a.method == "full" else {"r": a.lora_r, "alpha": a.lora_alpha, "targets": a.lora_targets},
            "precision": prec, "hardware": rep}
    (out / "neuronscope-finetune.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", help="HF id or local path of the base model (bf16/fp16 weights, not GGUF)")
    p.add_argument("--data", help="deficits.py output directory")
    p.add_argument("--out", help="adapter (or full model) output directory")
    p.add_argument("--method", choices=["qlora", "lora", "full"], default="qlora")
    p.add_argument("--dpo", action="store_true", help="run DPO on dpo.jsonl after SFT")
    p.add_argument("--fsdp", action="store_true", help="shard weights across GPUs (for --method full)")
    p.add_argument("--launch", type=int, default=0, metavar="N_GPUS", help="re-run this command under accelerate on N GPUs")
    p.add_argument("--check", action="store_true", help="print the hardware report and exit")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="all-linear", help="all-linear or a comma list like q_proj,v_proj,down_proj")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--dpo-beta", type=float, default=0.1)
    p.add_argument("--dpo-lr", type=float, default=5e-6)
    p.add_argument("--dpo-epochs", type=float, default=1)
    p.add_argument("--dpo-max-steps", type=int, default=-1)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.check:
        print(json.dumps(hardware_report(), indent=2))
        return 0
    if not (a.model and a.data and a.out):
        p.error("--model, --data and --out are required")
    if a.launch:
        if not shutil.which("accelerate"):
            raise SystemExit("pip install accelerate")
        argv_in = list(argv if argv is not None else sys.argv[1:])
        args, skip = [], False
        for x in argv_in:
            if skip:
                skip = False
                continue
            if x == "--launch":
                skip = True
                continue
            if not x.startswith("--launch="):
                args.append(x)
        cmd = ["accelerate", "launch", "--num_processes", str(a.launch), "--multi_gpu", __file__, *args]
        print("running:", " ".join(cmd))
        return subprocess.call(cmd)
    return train(a)


if __name__ == "__main__":
    raise SystemExit(main())
