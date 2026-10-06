# Retraining on measured deficits

Suppression (the H-Neuron edit) makes a model decline more. Retraining can
make it *know* more or *reason* better on what it currently gets wrong. This
is targeted retraining, also called error-driven data curation or hard-negative
mining, built on what TestQA (and the benchmark importers) already measure.

```text
testqa.py ──► deficits.py ──► finetune.py ──► merge_export.py ──► testqa.py --only-ids holdout
 measure       curate          QLoRA/LoRA/      merge + GGUF        measure again, on items
               (verified)      full, SFT+DPO    (or GGUF LoRA)      never trained on
```

## 1. Measure

```bash
python scripts/testqa.py --endpoint m=http://127.0.0.1:7870/v1@my-model --allow-exec \
    --cache runs/cache --out runs/base.json
```

`--cache` matters: it keeps the model's full replies, which become the
"rejected" side of preference pairs.

## 2. Curate the deficits

```bash
python scripts/deficits.py --results runs/base.json --label m --cache runs/cache \
    --teacher http://gpu-host:1234/v1@strong-model --expand 5 --allow-exec \
    --general data/general-chat.jsonl --deficit-fraction 0.25 --out runs/retrain
```

What it does:

| step | detail |
|---|---|
| **Categorise** | `factual_gap` (factual answers wrong or declined), `reasoning_failure` (reasoning, executable code, API code wrong), `format_violation` (instruction following, unparsable output), `perception_failure` (vision items wrong) |
| **Correction pairs (SFT)** | gold answers where the bank has them; otherwise a teacher model's reply. **Every target must pass the same grader that failed the model**, so nothing wrong is trained in |
| **Contrastive pairs (DPO)** | `prompt / chosen (verified target) / rejected (the model's own failing reply)` |
| **Synthetic expansion** | `--expand N`: the teacher writes N variations of each failure. Code variations must ship tests that their own solution passes in the interpreter. Reasoning and factual variations must be answered identically by a second, independent teacher call. Anything unverified is dropped |
| **Replay buffer** | deficits are `--deficit-fraction` (default 25%) of the SFT set; the rest is anchor data: what the model already answers correctly, in its own words, plus `--general` chat data. This is the guard against catastrophic forgetting |
| **Holdout** | half the bank (deterministic by id) is never trained on, not even through variations. `holdout_ids.json` lists it |

`plan.json` and `report.txt` give counts by category and subject, and the
method each category calls for. The teacher can be any OpenAI-compatible
endpoint: a larger local model, or a hosted one if its terms allow training
on its outputs.

### Vision deficits

Vision items go to `sft_vision.jsonl` and `dpo_vision.jsonl`, with their
images rendered into `images/`, in the image + messages format VLM trainers
take. Their targets are the bank's gold answers. `--expand N` needs no teacher
for them: `scripts/vision_synth.py` draws N new scenes in the failed item's
skill family (counting, colour, shape, position, size, reading text and
numbers, arithmetic), and the answer is known because the program drew it.
Every bank vision item maps to a family. Replay anchors are the model's own
correct vision answers, topped up with text anchors when there are too few.

```bash
python scripts/vision_synth.py --family count_shape -n 4 --out /tmp/preview   # look at what it draws
```

### SWE-bench deficits

Repository-level coding failures need longer, real-world data than the bank
has. `benchmarks.py swebench train-data` builds `sft.jsonl` and `dpo.jsonl` in
the same layout from SWE-bench's train split: gold patches as targets, the
model's own failed patches as DPO rejections, failure modes from a test run
setting priorities, and this directory's `sft.jsonl` as replay. See
[BENCHMARKS.md](BENCHMARKS.md#retraining-on-swe-bench-deficits).

## 3. Train

```bash
python scripts/finetune.py --check                                    # GPUs, precision, bitsandbytes
python scripts/finetune.py --model org/base-model --data runs/retrain --method qlora --dpo --out runs/adapter
python scripts/finetune.py --launch 4 --model ... --method lora --dpo --out runs/adapter      # 4 GPUs, data parallel
python scripts/finetune.py --launch 4 --model ... --method full --fsdp --out runs/full        # 4 GPUs, sharded
```

| deficit | method | why | multi-GPU |
|---|---|---|---|
| behaviour, logic, format | `qlora` / `lora`, SFT then `--dpo` | habits, not knowledge; adapters capture them cheaply | data parallel (DDP): each GPU holds the model plus a tiny adapter |
| factual / knowledge | `full` (or high-rank LoRA on the MLPs) | facts are spread across the MLP weights | `--fsdp` shards weights, gradients and optimizer state across GPUs |

- **QLoRA** loads the base in 4-bit NF4 and trains 16-bit adapters on all
  linear layers (`--lora-targets` to narrow). It needs CUDA and bitsandbytes.
- **LoRA** is the fallback for ROCm, CPU and GPUs without 4-bit kernels.
- Train from the bf16/fp16 Hugging Face checkpoint, not a GGUF.
- SFT loss is on the reply only. DPO uses the adapter-disabled model as its
  reference, so no second copy of the weights is loaded.
- Precision is chosen per GPU: bf16 where supported, otherwise fp16, and fp32
  on CPU. Older cards (Maxwell/Pascal, e.g. a Tesla M10) have no bf16, and
  recent PyTorch wheels may not include their `sm_XX`; `--check` shows
  whether yours is supported before you start.

### Vision-language models

```bash
python scripts/finetune.py --vision --model org/vlm --data runs/retrain --method lora --dpo --out runs/vlm-adapter
```

`--vision` loads the model with its processor and trains on the vision files.
Adapters go on the language model and the projector; the vision encoder is
always frozen. `--freeze-projector` also leaves the projector alone, which
means the base model's existing `mmproj` GGUF still matches the result: use it
when llama.cpp cannot convert your model's projector (below).

## 4. Merge and convert

```bash
python scripts/merge_export.py --base org/base-model --adapter runs/adapter --out runs/merged --gguf Q4_K_M
python scripts/merge_export.py --base org/base-model --adapter runs/adapter --out runs/lora --lora-gguf
```

The first merges the adapter into the base, converts it with llama.cpp's
converter and quantizes it. The second converts only the adapter to a GGUF
LoRA: load it in Studio's adapter field and the α slider blends base and
retrained behaviour live, the same way it does for suppression.

With `--vision`, merging writes the language GGUF and also tries to convert the
projector to `mmproj-model-f16.gguf`. llama.cpp converts projectors for the VLM
families it serves (Gemma 3, Qwen-VL, SmolVLM, Pixtral and others) but not
every Hugging Face layout, HF-format LLaVA included. When it cannot, the
command exits with status 2, records `mmproj_error` in
`neuronscope-export.json`, and points you at `--freeze-projector`.

## 5. Measure again, honestly

```bash
python scripts/testqa.py --endpoint before=...@my-model --endpoint after=...@my-model-retrained \
    --only-ids runs/retrain/holdout_ids.json --allow-exec
```

Only the holdout says whether retraining generalised. The trained half will
look better whether or not anything was learned. Check the canaries and the
per-subject table too: a model that improved on code but regressed on writing
has traded one deficit for another. Re-run with `--publish-stats` so Studio's
`auto` routing sees the new numbers.

## Tested

`tests/test_retrain.py` runs the whole chain on a tiny Llama: verified
deficit dataset with holdout and replay share, LoRA SFT plus DPO,
merge, a Q8_0 GGUF that llama.cpp loads, and a GGUF LoRA adapter. The vision
path runs on a tiny LLaVA: vision dataset with verified variations, LoRA SFT
plus DPO with and without a frozen projector, merge and language GGUF. QLoRA,
FSDP and real VLM projector conversion need GPUs or downloads and have not
been run here.
