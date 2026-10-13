# Engine options: llama.cpp and vLLM

Studio serves models with one of two engines, chosen per model in the model
panel (**Engine**):

| engine | serves | best for |
|---|---|---|
| **llama.cpp** (`llama-server`) | GGUF files | one machine, CPU + any GPU (Vulkan, CUDA, ROCm, Metal), mixed vendors, small memory; the only engine NeuronScope's activation tools read (cett-dump) |
| **vLLM** (`vllm serve`) | Hugging Face checkpoint folders (also AWQ / GPTQ / fp8), GGUF experimentally | many parallel requests on NVIDIA or AMD Instinct/RDNA GPUs; prefix caching, chunked prefill, tensor and pipeline parallelism |

Hugging Face folders under your models folders (a `config.json` with
`*.safetensors` next to it) show up in the model list and default to vLLM.
Install vLLM separately (`pip install vllm`, or AMD's ROCm build) and tell
Studio its command with `--vllm` or `$NS_VLLM` if it is not on the PATH.

## The options

Open a model → **Engine options**. Each option shows:
- its flag;
- what it does and when it helps;
- whether the installed engine has it. Studio reads `llama-server --help`;
  options the build lacks are folded away.

They are grouped:
- **Speed**: flash attention (auto/on/off), prompt threads, physical
  batch, parallel slots, busy-wait, priority.
- **Memory and KV cache**:
  - K and V cache types (`q8_0` halves the cache; a quantized V needs
    flash attention);
  - the KV cache in system RAM;
  - loading mode: mmap, mlock or read in.
- **Long context**:
  - context shift with *tokens to keep*: StreamingLLM-style. The first
    tokens (the attention sinks) stay, old middle tokens drop, and memory
    stays flat in long chats;
  - prompt cache reuse, so a repeated system prompt or document is not
    computed again;
  - vLLM: prefix caching and chunked prefill.
- **Speculative decoding**:
  - a draft model, its limits and its own device. For example, the draft
    on the iGPU and the big model on the dGPU;
  - **draft-free n-gram methods** (`ngram-mod`, `ngram-map-k`…), which
    propose from text already in the context. They need no extra memory
    and work well on code and edits;
  - EAGLE-3 / MTP heads where the model ships them;
  - vLLM: a JSON speculative config.
- **Devices**:
  - which devices hold layers (`Vulkan0,Vulkan1`);
  - RPC servers on other machines;
  - vLLM tensor / pipeline parallel.
- **CPU placement**: core range or mask, strict placement, NUMA mode.
- **Environment and launch**: a closed list of documented variables
  (ROCm, Vulkan, CUDA, vLLM), and an optional `numactl` or `taskset`
  wrapper.

Older Studio settings keep working:
- *flash attention on* becomes `-fa on`;
- a config's `cache_type` becomes both cache types.

The flag spellings follow the build: llama.cpp renamed `--draft-max` to
`--spec-draft-n-max`, and Studio passes whichever the binary accepts. After
**Load**, the panel shows the exact command and anything it left out.

From a terminal, `python scripts/engine_options.py --server <llama-server>`
lists the options this build has. With `--args '{...}'` it prints the flags
a setting produces.

## Recipes

Each of these is a starting point, not a promise. Hardware differs, so
measure the result.

### A small GPU and long chats
- K and V cache types `q8_0`, flash attention `on`;
- context shift on, *tokens to keep* `-1`;
- prompt cache reuse `256`.

### Faster generation without a second model
- Speculative decoding → method `ngram-mod`.

For code and repetitive text this often gives 1.5–2× and costs nothing.

### Faster generation with a draft model
- Draft model: the smallest model of the same family, which shares the
  tokenizer.
- Draft tokens at most: 8–16.

If acceptance is low, lower the limit.

### An NVIDIA dGPU and an AMD iGPU in one laptop
CUDA and ROCm cannot share a model, so:
1. Build llama.cpp for **Vulkan** (Setup → Build llama.cpp → vulkan). Both
   chips are then Vulkan devices.
2. Devices: `Vulkan0,Vulkan1`.
3. Split mode: `layer`, which hands the work over once per token.
   Not `row`, which syncs inside every layer over a slow bus.
4. Tensor split weighted to the fast chip, e.g. `85,15`.

Or keep the whole model on the dGPU, and put only the draft model on the
iGPU (draft model device `Vulkan1`).

### AMD with vLLM (ROCm)
- Environment `VLLM_USE_TRITON_FLASH_ATTN=0`: Composable Kernel flash
  attention instead of Triton, often faster prefill.
- `VLLM_ROCM_USE_AITER=1` where the build has AITER.
- Multi-GPU: `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` (e.g. `INT4`) for
  quick all-reduce.
- On transfer hangs, `HSA_ENABLE_SDMA=0`.
- For a GPU the runtime does not list, `HSA_OVERRIDE_GFX_VERSION`.

### A Ryzen laptop (one CCX, SMT)
- CPU cores: the physical cores, e.g. CPU mask `0x5555` on an 8-core / 16-
  thread part where siblings are adjacent. Check `lscpu -e` for your
  numbering.
- Strict placement on, threads = physical cores.

This keeps threads on one L3 cache instead of migrating.

### Several NUMA nodes (multi-socket, Threadripper/EPYC)
- Launch wrapper `numactl`, node 0, and NUMA `numactl`.

The OS's automatic NUMA balancing (`kernel.numa_balancing`) moves pages
around during inference. Turning it off is a system setting that needs
root (`sysctl -w kernel.numa_balancing=0`), so Studio does not change it.

## Let the machine find its own settings

```bash
python scripts/engine_optimizer.py --model models/m-Q4_K_M.gguf --trials 30 --out runs/opt-m --kv --spec \
    --apply ~/.neuronscope/studio.json
```

It is also in Studio: Jobs → Setup → *Find the fastest engine settings*.
Each trial:
1. starts llama-server with one combination of settings (threads, batch,
   ubatch, flash attention, and with `--kv`, `--spec`, `--placement`: KV
   types, n-gram speculation, CPU placement);
2. sends the same request a few times and reads the server's own timings;
3. stops it.

The search:
- **Bayesian (Optuna TPE)** when optuna is installed (`pip install
  optuna`); otherwise random search, then one option at a time around the
  best;
- the engine's defaults are measured first, so the report says how much
  faster the winner is;
- trials are cached, so a run resumes and a longer run builds on a shorter
  one;
- `--apply` writes the winner into Studio's settings for that model;
- `--fix device=Vulkan0` holds an option at a value.

**Objectives:**
- `gen`: generation tokens/s, the default;
- `prompt`: prompt tokens/s;
- `request`: the time of a whole request shaped by `--prompt-tokens` and
  `--gen-tokens`.

Only speed is measured. A quantized KV cache can change answers slightly,
so check one with TestQA or the gate. The suppression α is a quality
setting: tune it with `tune_scale_server.py`, not for tokens per second.

## What Studio does not do for you

These are real gains in some setups, but outside what an application should
change, or they come with the engine already:

- **BIOS memory profiles, HugePages, kernel settings:** system
  administration. Measure before and after with the engine optimizer
  (below).
- **CUDA/HIP graphs:** vLLM captures them by default (*No CUDA/HIP graphs*
  turns that off). llama.cpp uses CUDA graphs on its own where they apply.
- **Tree speculative decoding, lookahead/Jacobi decoding:** not in either
  engine's server as a switch. The n-gram methods are the draft-free
  option that exists today.
- **H2O / heavy-hitter KV eviction:** not in llama.cpp or vLLM. Context
  shift with *tokens to keep* is the attention-sink half of StreamingLLM,
  and is available.
- **Mamba / SSM hybrids, Mixture-of-Depths, ternary (BitNet b1.58)
  models:** these are model choices. llama.cpp loads the architectures it
  supports (Mamba, Mamba-2, Jamba, several hybrids, and BitNet through its
  TQ1/TQ2 types). Load such a GGUF like any other. TestQA and the gate tell
  you whether it is better for your work, and NeuronScope's neuron tools
  read activations the same way.
