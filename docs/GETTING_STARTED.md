# Getting started

New to the vocabulary? [GLOSSARY.md](GLOSSARY.md) explains GGUF, CETT,
H-Neurons, abstention, MCP and the rest, and why each matters here.

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

## The short way

```bash
git clone <this repo> neuronscope && cd neuronscope
./install.sh            # add --no-torch for a lighter install without PyTorch
./neuronscope           # or start "NeuronScope" from the app menu
```

Studio opens in your browser on its **Setup** page: build llama.cpp with one
button (or point it at an existing build), add your models folder, run the
health check. Everything after that is in the browser; see
[UI_AND_CLI.md](UI_AND_CLI.md) for where each feature lives, and the steps
below for doing the same from a terminal.

## 1. Install

```bash
git clone <this repo> neuronscope && cd neuronscope
./install.sh              # creates venv/, picks a PyTorch build for your GPU, smoke-tests it
source venv/bin/activate
```

`install.sh` detects AMD (ROCm), NVIDIA (CUDA) or neither (CPU) and installs a
matching PyTorch wheel. PyTorch is only needed for the PyTorch extraction path,
CLIP/SigLIP and fine-tuning; for Studio, TestQA, Projects, transfer and the
llama.cpp path, `./install.sh --no-torch` is a much smaller install. On NVIDIA it checks every GPU's architecture: older
cards (Maxwell, Pascal, Volta, e.g. a Tesla M10) need the CUDA 12.6 wheels and
`torch<2.15`, which `python scripts/cuda_info.py` reports and the installer
picks. On macOS or Windows, or if you prefer to manage it yourself:

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

It checks, in order, Python and packages, your GPUs and the devices Studio's
worker models can use, your llama.cpp build, the model in `NS_GGUF`, and which
pipeline stages already have output. Each problem comes with what it blocks
and the fix, and it ends with what to do next: one line for using models in
Studio, one for the H-Neuron pipeline.

## 2. Build llama.cpp (for serving and GGUF extraction)

```bash
scripts/build_llama_tools.sh --backend vulkan     # or cpu, cuda, metal, hip; --dir to choose the checkout
```

It clones llama.cpp, adds the `cett-dump` extractor, and builds
`llama-cett-dump`, `llama-server`, `llama-quantize` and `llama-eval-callback`
(about 5 minutes on a laptop CPU), then prints the lines to put in
`env.local.sh`. A warning that llama-server's built-in web UI could not be
downloaded is harmless: Studio is the UI.
Vulkan works on AMD, Intel and NVIDIA and is the most portable choice. With
`--backend cuda` the CUDA architectures come from your GPUs (`--cuda-arch` to
build for others), and pre-Turing cards need a CUDA 12.x toolkit, or
`docker/llama-cuda.Dockerfile`; see [HARDWARE.md](HARDWARE.md#nvidia--cuda).

`cett-dump` is tested against current llama.cpp: on a model converted with
llama.cpp's own converter its CETT values match the PyTorch hook path on every
layer (`tests/test_llamacpp_integration.py`, run with `NS_LLAMA` set).

## 3. Get a model and point NeuronScope at it

No GGUF yet? Start Studio (step 4) and use its **Models** tab to search Hugging
Face and download one; a 7-8B model at Q4_K_M (about 5 GB) is a good first
choice. Studio, TestQA and Projects need nothing more.

For the H-Neuron pipeline, tell the scripts which model to study:

```bash
export NS_GGUF=~/models/Qwen3-8B-Q6_K.gguf
export NS_LLAMA=~/llama.cpp
source env.sh            # derives NS_CETT, NS_LLAMA_SERVER, NS_LAYERS; checks the file
```

Put those exports in `env.local.sh` (gitignored) to make them stick. Use a
Q8_0 or F16 file here if you can: quantization rounds the very activation
sizes the pipeline measures (see [GLOSSARY.md](GLOSSARY.md#models-and-files)).

## 4. Pick a starting point

**Run models like LM Studio, with the NeuronScope dials:**

```bash
python viz/studio.py --models-dir ~/models --server "$NS_LLAMA_SERVER"
```

Open http://127.0.0.1:7870. Any OpenAI client can use http://127.0.0.1:7870/v1.
Studio prints the ids of the models it found; that id is what goes in an API
request's `"model"` and in `<model-id>` below. See [STUDIO.md](STUDIO.md).

**Benchmark one or more endpoints:**

```bash
python scripts/testqa.py --endpoint mine=http://127.0.0.1:7870/v1@<model-id> --per-subject 5   # a quick first pass
python scripts/testqa.py --endpoint mine=http://127.0.0.1:7870/v1@<model-id> --allow-exec --sandbox docker
```

`--allow-exec` grades code questions by running the model's code; with
`--sandbox docker` (or `podman`) that happens in a locked-down container,
without it only under local CPU and memory limits. See [TESTQA.md](TESTQA.md).

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

It should end with `cett-dump: finished, 1 ok, 0 failed`. To see what it
captured:

```bash
python -c "import sys; sys.path.insert(0, 'scripts')
from extract_activations_gguf import read_aggregate
spans, agg = read_aggregate('/tmp/d.bin')[:2]
print('spans x layers x neurons:', agg.shape)"
```

The middle number must equal `$NS_LAYERS`: one averaged row of neuron
activations per decoder layer. If the run fails with "capture failed (0
records)", the architecture names its FFN output node differently. List the
node names:

```bash
"$NS_LLAMA"/build/bin/llama-eval-callback -m "$NS_GGUF" -p "hi" -n 1 2>&1 \
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
