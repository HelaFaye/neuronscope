# Getting started

NeuronScope is a set of Python tools plus two small llama.cpp additions. Most
of it runs anywhere Python 3.10+ runs; the parts that touch model internals
want either PyTorch (any backend) or a llama.cpp build.

| Component | Linux | macOS | Windows | Needs |
|---|---|---|---|---|
| Studio (model manager, chat, `/v1` API) | yes | yes | yes | Python stdlib, a `llama-server` binary |
| Transfer (WebRTC GUI, resumable CLI, MCP) | yes | yes | yes | stdlib; `aiohttp` for the GUI |
| TestQA, subject classifier | yes | yes | yes¹ | `requests` |
| CLIP / SigLIP H-Neurons | yes | yes | yes | `torch`, `transformers`, `pillow` |
| Text H-Neurons, PyTorch path | yes | yes | yes | `torch`, `transformers` |
| Text H-Neurons, GGUF path (`cett-dump`) | yes | yes | build with CMake | llama.cpp checkout |

¹ The code interpreter's CPU/memory limits use POSIX `resource`; on Windows
only the wall-clock timeout applies.

## 1. Install

```bash
git clone <this repo> neuronscope && cd neuronscope
./install.sh              # creates venv/, picks a PyTorch build for your GPU, smoke-tests it
source venv/bin/activate
```

`install.sh` detects AMD (ROCm), NVIDIA (CUDA) or neither (CPU) and installs a
matching PyTorch wheel. On macOS or Windows, or if you prefer to manage it
yourself:

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install torch                                    # pick the right build from pytorch.org
pip install -r requirements.txt
pip install -r requirements-transfer-full.txt        # optional: WebRTC GUI, MCP, keyring
```

Then check what you have:

```bash
python scripts/doctor.py
```

It walks every prerequisite in order and ends with one next action.

## 2. Build llama.cpp (for serving and GGUF extraction)

```bash
scripts/build_llama_tools.sh --backend vulkan     # or cpu, cuda, metal, hip; --dir to choose the checkout
```

It clones llama.cpp, adds the `cett-dump` extractor, and builds
`llama-cett-dump`, `llama-server`, `llama-quantize` and `llama-eval-callback`.
Vulkan works on AMD, Intel and NVIDIA and is the most portable choice; see
[HARDWARE.md](HARDWARE.md) for backend notes.

`cett-dump` is tested against current llama.cpp: on a model converted with
llama.cpp's own converter its CETT values match the PyTorch hook path on every
layer (`tests/test_llamacpp_integration.py`, run with `NS_LLAMA` set).

## 3. Point NeuronScope at a model

```bash
export NS_GGUF=~/models/Qwen3-8B-Q6_K.gguf
export NS_LLAMA=~/llama.cpp
source env.sh            # derives NS_CETT, NS_LLAMA_SERVER, NS_LAYERS; checks the file
```

Put those exports in `env.local.sh` (gitignored) to make them stick.

## 4. Pick a starting point

**Run models like LM Studio, with the NeuronScope dials:**

```bash
python viz/studio.py --models-dir ~/models --server "$NS_LLAMA_SERVER"
```

Open http://127.0.0.1:7870. Any OpenAI client can use http://127.0.0.1:7870/v1.
See [STUDIO.md](STUDIO.md).

**Benchmark one or more endpoints:**

```bash
python scripts/testqa.py --endpoint mine=http://127.0.0.1:7870/v1@<model-id> --allow-exec
```

See [TESTQA.md](TESTQA.md).

**Find and suppress H-Neurons in a text model:** [PIPELINE.md](PIPELINE.md).

**Do the same for CLIP or SigLIP:** [VISION.md](VISION.md).

**Move models between machines:** [TRANSFER.md](TRANSFER.md), and read
[SECURITY.md](SECURITY.md) before exposing anything beyond localhost.

## The first real test for the GGUF path

```bash
printf 'The capital of France is' > /tmp/seq.txt
"$NS_CETT" -m "$NS_GGUF" -ngl 99 -b 4096 --n-layers "$NS_LAYERS" \
    --prompt-file /tmp/seq.txt --out /tmp/d.bin
```

Expect one record per decoder layer. If you get zero, the architecture names
its FFN output node differently. List the node names:

```bash
~/llama.cpp/build/bin/llama-eval-callback -m "$NS_GGUF" -p "hi" -n 1 2>&1 \
    | grep -oE '^[a-z_]+-[0-9]+' | sort -u | head -30
```

then add the prefix to `FFN_DOWN_PREFIXES` in
`llama-tools/cett-dump/cett-dump.cpp` and rebuild.

## Tests

```bash
pip install pytest
python -m pytest -q
```

The suite needs no model and no GPU. It covers the security policy, both
transfer paths (including TLS), the signaling server, the CLIP pipeline on a
tiny random CLIP, TestQA against fake endpoints, and Studio's API against a
fake `llama-server`. The C++ tools have their own build notes in
`llama-tools/*/`.
