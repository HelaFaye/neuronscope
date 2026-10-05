# NeuronScope

A cross-platform workbench for finding, visualizing and editing the neurons
behind confident mistakes in local models. It also covers serving, routing,
benchmarking and moving those models between machines.

NeuronScope began as a reworking of [thunlp/H-Neurons](https://github.com/thunlp/H-Neurons)
(Gao et al., [arXiv:2512.01797](https://arxiv.org/abs/2512.01797)). That paper
shows a sparse set of MLP neurons ("H-Neurons") is causally linked to
over-confident answers, and scaling them down makes a model more willing to
abstain. NeuronScope extends the method in several directions:

- **any hardware**: llama.cpp (Vulkan, CUDA, Metal, CPU) or PyTorch (CUDA, ROCm, MPS, CPU);
- **any serving stack**: LM Studio, llama-server, or any OpenAI-compatible endpoint;
- **text and image-text models**: decoder LLMs, plus CLIP and SigLIP;
- **ships the result**: edited GGUFs, LoRA adapters, `mmproj` projectors and HF checkpoints that load where you already run models.

## What's in the box

| Area | What it does | Guide |
|---|---|---|
| **H-Neuron pipeline** | collect → label → extract CETT → sparse classifier → tune scale → export (GGUF in place, LoRA, HF) | [docs/PIPELINE.md](docs/PIPELINE.md) |
| **Vision H-Neurons** | the same method for CLIP/SigLIP: confident image-caption mismatches, `mmproj` editing | [docs/VISION.md](docs/VISION.md) |
| **Studio** | LM Studio-style model library, chat with images, saved chats, Hugging Face downloads, and an OpenAI-compatible `/v1` server with JIT loading, idle unload and `model: "auto"` routing; plus a live suppression slider and MoE expert control | [docs/STUDIO.md](docs/STUDIO.md) |
| **TestQA** | 210 graded items, 30 in each of seven subjects (reasoning, executable code, API existence, false-premise facts, instruction-following writing, generated-image vision) against any endpoints; per-subject right/hallucinated/abstained rates; paired comparison; feeds routing stats | [docs/TESTQA.md](docs/TESTQA.md) |
| **Subject classifier + stats routing** | labels prompts (code, math, logic, science, factual, writing, vision); `model: "auto"` picks the model with the best measured record for that subject, from rolling per-model stats | [docs/TESTQA.md](docs/TESTQA.md#subject-classifier-and-routing) |
| **Transfer** | encrypted P2P browser transfer, resumable HTTPS CLI with watch mode, remote tuning worker, MCP control plane | [docs/TRANSFER.md](docs/TRANSFER.md) |
| **External benchmarks** | CLIP_benchmark (ImageNetV2, ImageNet-Sketch, VTAB, ...) for CLIP and its edits; LiveBench and SWE-bench for chat/coding models, feeding stats; leaderboard (BenchLM) imports as reference | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| **Retraining on deficits** | turns measured failures into verified SFT and DPO data with replay and a holdout, trains with QLoRA, LoRA or full fine-tuning (DDP/FSDP), merges and converts to GGUF | [docs/RETRAINING.md](docs/RETRAINING.md) |
| **Labs** | capability-preserving quantization, scale sweeps, adaptive remote tuning | [docs/LABS.md](docs/LABS.md) |
| **Visualization** | 2D/3D activation maps, token-resolved timelines, live tracing proxy, weight views | [docs/VISUALIZATION.md](docs/VISUALIZATION.md) |
| **Merging** | neuron-granular selective merges, mergekit plugin, MoE expert-count sweeps | [docs/MERGING.md](docs/MERGING.md) |

## Quick start

```bash
./install.sh && source venv/bin/activate     # picks CUDA / ROCm / MPS / CPU PyTorch
python scripts/doctor.py                     # what is ready, and what to do next
```

Full setup, including building llama.cpp with the `cett-dump` extractor:
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md). Hardware notes:
[docs/HARDWARE.md](docs/HARDWARE.md).

```bash
# Serve and chat with local GGUFs; OpenAI clients use http://127.0.0.1:7870/v1
python viz/studio.py --models-dir ~/models --server ~/llama.cpp/build/bin/llama-server

# Benchmark endpoints on reasoning, coding (executed), facts and canaries
python scripts/testqa.py --endpoint base=http://127.0.0.1:7870/v1@<model-id> --allow-exec

# Find H-Neurons in a CLIP model
python scripts/clip_neurons.py collect --model openai/clip-vit-base-patch32 --images <imagefolder> --out runs/clip
```

## Read before you trust a result

- **Suppression is not correction.** Scaling H-Neurons down trades confident
  errors for abstentions. It does not add knowledge. Measure both sides; on
  coding workloads the trade may not be worth it.
- **Scope is narrower than "hallucination".** Text labels come from short-form
  factual QA by default. Long-form confabulation, RAG groundedness and
  invented APIs may involve different neurons; collect from the domain you
  care about.
- **Do not extract from a quantized model.** CETT measures activation
  magnitudes, which quantization perturbs. Generate from a quant if you like.
- **Labels are the leverage.** Validate the judge (`judge_agreement.py`) before
  tuning anything downstream.

More in [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md).

## Security

Every network service follows one policy. Loopback binds work without
credentials. Any other bind requires a 256-bit token, and TLS unless you
explicitly state the port is behind a VPN or TLS proxy. WebRTC transfers are
always DTLS-encrypted and can be verified with an out-of-band code. See
[docs/SECURITY.md](docs/SECURITY.md) for the threat model and checklist.

## Repository layout

```
install.sh, env.sh           environment setup (env.local.sh for your paths, gitignored)
scripts/                     pipeline stages, exporters, evaluators, transfer, security
  ns_common.py, profiles.py    shared hooks, sequence building, portable profiles
  clip_neurons.py              vision H-Neurons; suppress_mmproj.py edits mmproj GGUFs
  testqa.py                    evaluation harness; subject_classifier.py routing
  ns_transfer*.py, model_transfer.py, tuning_worker.py, ns_security.py
viz/                         browser and native frontends (studio, hub, labs, explorers)
web/                         WebRTC transfer page
qa/                          TestQA bank (one file per subject), subject seed data
llama-tools/                 cett-dump extractor and the optional activations server patch
tests/                       pytest suite; no model or GPU needed
docs/                        guides
```

## Tests

```bash
pip install pytest && python -m pytest -q
```

## Licence

MIT, following upstream H-Neurons.
