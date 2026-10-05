# Hardware notes

NeuronScope does two very different kinds of work, and they want different
hardware:

| Work | Bound by | Runs well on |
|---|---|---|
| Generation (collect responses, evaluate, chat) | memory bandwidth per token | a discrete GPU, via llama.cpp or LM Studio |
| Activation extraction (`cett-dump`) | one prefill per sample | any llama.cpp backend, including integrated GPUs |
| Classifier training | RAM | CPU; ~3 GB at 400 pairs, ~7 GB at 1000 |
| PyTorch hook extraction | VRAM at bf16 | a GPU that fits the full bf16 model, or CPU |

The pipeline is split so each stage can run on the machine that suits it.
Collection and scale tuning talk to any OpenAI-compatible server over HTTP, so
a laptop can drive a GPU box elsewhere on the network; see
[TRANSFER.md](TRANSFER.md) for moving the resulting files.

## Low-power hosts

On an integrated GPU or a CPU-only machine:

| stage | where | why |
|---|---|---|
| 1 collect | a GPU host, over HTTP | generation is bandwidth-bound; an iGPU manages a few tokens/s |
| 4 extract | locally, `cett-dump` on Vulkan/CPU | prefill only, no generation |
| 5 classify | locally, CPU | sklearn |
| 7 tune | a GPU host, `tune_scale_server.py` | many generations per scale |

`cett-dump` loads the model once for the whole manifest and resumes from
existing dumps, so it is the stage that runs well on modest hardware.

## Backends

**Vulkan** (llama.cpp) is the most portable GPU path: AMD, Intel and NVIDIA,
Linux and Windows. PyTorch has no usable Vulkan backend, so Vulkan exists only
inside llama.cpp here. Systems with more than one Vulkan driver installed
(e.g. AMDVLK and RADV) sometimes pick one that fails to initialise; set
`NS_VULKAN_ICD` to the driver JSON you want before `source env.sh`.

**CUDA / Metal**: build llama.cpp with `-DGGML_CUDA=ON` or `-DGGML_METAL=ON`
and install the matching PyTorch. Nothing in NeuronScope is vendor-specific.

**ROCm (AMD)**: `install.sh` detects the gfx target and picks a PyTorch ROCm
wheel index, then runs a bf16 matmul against CPU to prove the result is
correct, not merely that a kernel ran.

- RDNA4 (gfx1200/1201) needs ROCm 7.x wheels.
- RDNA2 dies below gfx1030 need `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
- RDNA1 and Vega APUs are unsupported by the wheels; `install.sh` installs CPU
  PyTorch and the Vulkan path does the heavy lifting.
- Distros AMD does not officially support usually work anyway, because the pip
  wheels bundle the ROCm userspace and only need the mainline `amdgpu` driver.
  `./install.sh --docker` prints a container recipe if they don't.

Diagnostics:

```bash
./scripts/rocm_inventory.sh      # real gfx target, system vs wheel rocBLAS kernels
./scripts/try_rocm.sh            # correctness check of an override against CPU
./scripts/try_rocm.sh --install  # install a ROCm torch first
```

If system rocBLAS has kernels for your GPU but the wheel does not, prefer a
distro-built PyTorch (`./install.sh --system-torch` creates a venv with system
site packages) over mixing a wheel with kernels from another ROCm major
version.

## Sizing

```bash
python scripts/vram_budget.py --gguf "$NS_GGUF" --ctx 32768 --vram 12
python scripts/hostcheck.py
```

`vram_budget.py` reads the model's real attention shape and reports what fits
at a given context. `hostcheck.py` estimates whether a stage will fit this
machine before it starts, rather than after it fails; the classifier and
Studio both call it.

For MoE models on consumer hardware, keep routed experts in system RAM and
everything else on the GPU:

```bash
--override-tensor '\.ffn_.*_exps\.=CPU'
```

Generation is bound by *active* parameters, so a Q4 30B-A3B can be faster than
a dense Q6 9B despite being three times larger on disk.
