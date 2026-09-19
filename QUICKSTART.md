# NeuronScope — Arch / EndeavourOS quickstart

Written for a Ryzen 7 7730U (Vega 8 / `gfx90c`, RADV) with 32 GB RAM and a
12 GB GPU carve-out. ROCm does not support `gfx90c`, so everything here uses
**Vulkan for the GPU and CPU for PyTorch**. That is expected, not a fallback.

---

## 1. System packages

```bash
sudo pacman -S --needed base-devel cmake git python python-pip \
    vulkan-radeon vulkan-tools vulkan-headers shaderc
```

Check the GPU is visible:

```bash
vulkaninfo --summary | grep deviceName
```

You should see `AMD Radeon Graphics (RADV RENOIR)`. If you also see
`Skipping this driver` for `amdvlkpro64.so`, that is the AMDVLK Pro ICD failing;
`env.sh` pins RADV so it cannot be picked up.

## 2. Python environment

```bash
cd neuronscope
./install.sh                 # creates venv/ and installs CPU torch
source venv/bin/activate
```

If `source venv/bin/activate` says *No such file or directory*, `install.sh`
has not run or did not finish — that step creates the venv.

If pacman gives 404s on every mirror, your package database is stale relative
to the mirrors. Arch does not support partial upgrades:

```bash
sudo pacman -Syu             # sync AND upgrade, not -Sy
```

`install.sh` detects `gfx90c`, installs **CPU** PyTorch (correct — ROCm has no
support for this GPU), and runs a smoke test. Ignore any suggestion to install
ROCm.

### Is my GPU being used?

Yes, for everything that matters — and if the ROCm override works on your
machine, PyTorch can use it too. llama.cpp's Vulkan backend runs both serving
and activation extraction, and the visualizers use Vulkan through WGPU or the
browser. PyTorch runs on CPU because ROCm has no `gfx90c` support — but the
bf16 PyTorch extraction path was superseded by `cett-dump` on Vulkan, which is
faster here *and* fits in 12 GB where bf16 never would. Nothing is waiting on
ROCm.

If you want to test the override anyway:

```bash
./scripts/try_rocm.sh            # check only
./scripts/try_rocm.sh --install  # install ROCm torch first
```

It checks correctness against CPU, not just that a matmul ran — a masqueraded
target can produce plausible garbage rather than failing outright.

### What ROCm do I already have?

```bash
./scripts/rocm_inventory.sh
```

Reports the real gfx target (unsetting any override), your system ROCm version
and its rocBLAS kernels, the PyTorch build in the active python and *its*
bundled kernels, and whether any of them cover your GPU.

Two things it knows that are easy to miss. ROCm 7.x ships **generic** targets
like `gfx9-generic` that cover a whole family — if your system rocBLAS has one,
`gfx90c` is supported directly and `HSA_OVERRIDE_GFX_VERSION` is unnecessary.
And the pip torch wheel **bundles its own ROCm**, so `./install.sh --rocm 6.0`
changes only the venv and never touches `/opt/rocm`. Do not uninstall system
ROCm to "downgrade" — rocsolver, rccl, magma-hip and others depend on it, and
the wheel would not have noticed either way.

**If the inventory says your system rocBLAS has the kernels but the wheel does
not**, prefer the distro build over borrowing a Tensile library across ROCm
major versions:

```bash
sudo pacman -S python-pytorch-rocm     # if not already installed
rm -rf venv
./install.sh --system-torch            # venv with --system-site-packages
```

Arch's `python-pytorch-rocm` is built against the same ROCm as `/opt/rocm`, so
its rocBLAS and Tensile kernels match. The `ROCBLAS_TENSILE_LIBPATH` route
works sometimes and is worth 30 seconds, but it pairs a 6.4 wheel with 7.2.4
kernels and AMD does not support that combination.

**If it aborts with a rocBLAS error** naming `TensileLibrary.dat` for your
target, the override is fine — HIP saw the device — but the wheel ships no
matrix-multiply kernels for it. ROCm 6.4 wheels carry gfx1030, gfx1100-1102,
gfx1200-1201, gfx908, gfx90a and gfx942; **no gfx900**, which is what a Vega
APU masquerades as. Try a wheel that still has it:

```bash
./install.sh --rocm 6.0      # or 5.7
```

`try_rocm.sh` now lists the kernels present in your wheel and says whether your
target is among them, rather than reporting a generic failure.

**If it works**, the PyTorch scripts (`merge_selective`, `export_lora`,
`intervene_model`) can use the GPU, though they are one-off weight edits where
it hardly matters, and small models can use the bf16 hook path. It does **not**
make Ornith's bf16 extraction viable: 9B at bf16 is ~18 GB against a 12 GB
carve-out. `cett-dump` on Vulkan stays the extraction path.

Two Arch specifics. `pip install` outside a virtualenv is blocked by PEP 668,
so either `source venv/bin/activate` first or use Arch's own build:

```bash
sudo pacman -Syu
sudo pacman -S python-pytorch-rocm rocminfo
```

Or into the venv, once `try_rocm.sh` passes:

```bash
./install.sh --rocm 6.4      # explicit --rocm overrides the APU default
```

And if you already have `HSA_OVERRIDE_GFX_VERSION` exported, `rocminfo` reports
the masquerade rather than the hardware. The script unsets it for detection and
prints both.

## 3. Build llama.cpp with Vulkan

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cp -r llama-tools/cett-dump ~/llama.cpp/tools/
echo 'add_subdirectory(cett-dump)' >> ~/llama.cpp/tools/CMakeLists.txt

cd ~/llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target llama-server llama-cett-dump llama-quantize
```

## 4. Point at your model

```bash
cd ~/neuronscope
$EDITOR env.sh          # NS_VOLUME_UUID and NS_MODEL_REL are pre-filled
source env.sh
```

It resolves the model by filesystem UUID, so the `/run/media` path can change.
If the drive is not mounted it prints the `udisksctl` command.

## 5. Check everything

```bash
python scripts/doctor.py
```

Walks every prerequisite in order and ends with one next action. Then:

```bash
python scripts/vram_budget.py --gguf "$NS_GGUF" --ctx 47104 --vram 12
```

This reads your model's real attention shape and tells you what fits. Worth
running before anything else — it costs seconds and settles the context
question with numbers instead of guesses.

---

## The first real test

Everything above is setup. This is the gate:

```bash
printf 'The capital of France is' > /tmp/seq.txt
time ~/llama.cpp/build/bin/llama-cett-dump -m "$NS_GGUF" \
    -ngl 99 -b 4096 --n-layers 32 --prompt-file /tmp/seq.txt --out /tmp/d.bin
```

**Expect one record per decoder layer.** If you get that, the whole local half
of the project works.

If you get **zero**, it is almost certainly tensor naming. llama.cpp PR #20785
documents that Qwen3.5 and qwen3next use `final_output-{layer}` where older
architectures use `l_out-{layer}` — and Ornith is Qwen3.5-based, so its
`ffn_down` node may be named differently too. `cett-dump` now tries
`ffn_down-`, `ffn_out-` and `ffn_down_out-`. To find the real name:

```bash
~/llama.cpp/build/bin/llama-eval-callback -m "$NS_GGUF" -p "hi" -n 1 2>&1 \
    | grep -oE '^[a-z_]+-[0-9]+' | sort -u | head -30
```

Then add the prefix to `FFN_DOWN_PREFIXES` in
`llama-tools/cett-dump/cett-dump.cpp` and rebuild.

The `time` matters too: it tells you the model load cost off that external
drive. Minutes rather than seconds means copy the GGUF to internal storage.

---

## Things you can do today

**Chat UI with the runtime dials LM Studio lacks:**

```bash
python viz/studio.py --models-dir "$(dirname "$(dirname "$NS_GGUF")")" \
    --server ~/llama.cpp/build/bin/llama-server --token "$(openssl rand -hex 16)"
```

**Near-realtime activation view, no fork needed:**

```bash
~/llama.cpp/build/bin/llama-server -m "$NS_GGUF" -ngl 99 -c 8192 --port 8080 &
python viz/stream.py --token secret --host 0.0.0.0 &
python scripts/autotrace.py --upstream http://127.0.0.1:8080 \
    --binary "$NS_CETT" --gguf "$NS_GGUF" --tokenizer "$NS_TOKENIZER" \
    --n-layers 32 --publish http://127.0.0.1:7890 --port 8088
```

Point Cline at `:8088`. Open `http://<this-machine>:7890` on your phone.

**Weight analysis — needs no collection run, no classifier:**

```bash
python viz/weights.py --gguf "$NS_GGUF" --dump w.npz
```

---

## The pipeline, in order

Stages 1 and 7 want a fast GPU; run them on your friend's 9060 XT. Stage 4
suits this laptop — it is prefill only, so the ~4 tok/s generation ceiling does
not apply.

```bash
# 1. collect (on the fast box, via its LM Studio server)
python scripts/collect_responses_lmstudio.py \
    --base_url http://192.168.41.171:1234/v1 --model ornith-1.0-9b \
    --data_path data/TriviaQA/rc.nocontext/train-00000-of-00001.parquet \
    --output_path data/consistency_samples.jsonl \
    --sample_num 5 --max_questions 200 --concurrency 4

# 2-3. label and split
python scripts/make_answer_tokens.py --input_path data/consistency_samples.jsonl \
    --output_path data/answer_tokens.jsonl --model_path "$NS_TOKENIZER"
python scripts/sample_balanced_ids.py --input_path data/answer_tokens.jsonl \
    --output_path data/train_qids.json --num_samples 400

# 4. extract (here, Vulkan)
python scripts/extract_activations_gguf.py --binary "$NS_CETT" \
    --gguf "$NS_GGUF" --tokenizer "$NS_TOKENIZER" \
    --input_path data/answer_tokens.jsonl --ids_path data/train_qids.json \
    --output_root data/activations --ngl 99 --batch 4096 \
    --locations answer_tokens all_except_answer_tokens

# 5. classify
python scripts/classifier.py --acts_root data/activations \
    --train_ids data/train_qids.json --train_mode 3-vs-1 --C 1.0 \
    --out_dir models
```

After stage 5 you have `models/h_neurons.json` and the visualizers,
`export_lora.py` and `merge_eval.py` all become useful.

---

## Known state

Every Python component is tested against synthetic data; `tests/test_audit.py`
runs with no model and no GPU. The C++ (`cett-dump`, and the optional
`server-activations` fork) has **never been compiled** — no compiler was
available where it was written. Treat the first build of each as a debugging
session.

No real activations have been extracted yet. Step 5 above is where that
changes.
