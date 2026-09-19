# NeuronScope

A reworking of [thunlp/H-Neurons](https://github.com/thunlp/H-Neurons)
(Gao et al., [arXiv:2512.01797](https://arxiv.org/abs/2512.01797)) for a setup
the original does not target: AMD hardware, a reasoning model, generation
offloaded to LM Studio, and a final artifact that runs back inside LM Studio.

---

## Read this before you start

**LM Studio cannot do the analysis.** It is a llama.cpp/MLX front-end. Its API
is chat, completions, embeddings, tokenization and plugins — there is no hidden
state endpoint and no way to register a forward hook. Stages 2 through 5 need
PyTorch. LM Studio covers stage 1 (generation) and stage 7 (serving the edited
model), and nothing in between.

**Vulkan is not a compute path here.** PyTorch has no usable Vulkan backend.
Vulkan exists for you only inside llama.cpp. Everything hooked runs on ROCm or
CPU.

**Do not quantize the model you extract from.** CETT measures activation
magnitude. Quantization perturbs exactly the quantity being measured. Generate
from a quant if you like; extract from bf16.

**Suppression is not correction.** The paper's finding is that H-Neurons are
causally linked to *over-compliance*. Scaling them down makes the model more
willing to decline, not more likely to be right. You are trading hallucinations
for abstentions. On short factual QA that is usually worth it. On a coding
workload it may just make the model less useful. Measure both sides.

**Scope is narrower than "hallucination" suggests.** The labels come from
TriviaQA: a short factual answer that does not match a gold string. This is not
RAG groundedness, not fabricated citations, not long-form confabulation. If your
real failure mode is one of those, the neurons you find may not transfer.

---

## Paths

`env.sh` resolves the model by filesystem UUID, so an auto-mounted volume that
lands on a different path still works. Source it before any local stage:

```bash
source env.sh          # sets NS_GGUF, NS_LAYERS, NS_BATCH, NS_CETT
```

It fails loudly with the `udisksctl mount` command if the volume is absent,
rather than letting a stage start against a missing file.

## Install

```bash
./install.sh
source venv/bin/activate
```

The installer detects your gfx target, picks a matching PyTorch ROCm wheel
index, and runs a bf16 matmul to prove it works. It prints a suggested
`--gpu_mem` value at the end — keep that number.

Arch and EndeavourOS are not AMD-supported ROCm distros. The pip wheels bundle
the ROCm userspace and usually work anyway, since all they need from the host is
the mainline `amdgpu`/`amdkfd` driver. If they don't, `./install.sh --docker`
prints the container recipe, which sidesteps the distro question entirely.

RDNA4 (gfx1200/1201) needs ROCm 7.x wheels. RDNA2 dies below gfx1030 need
`HSA_OVERRIDE_GFX_VERSION=10.3.0`. RDNA1 is unsupported; CPU extraction is
likely faster than fighting it.

---

## PIPELINE

### 0. Preflight

```bash
python scripts/preflight.py --model_path ornith-ai/Ornith-1.0-9B --n_pairs 400
```

Loads only the config and a meta-device skeleton — no VRAM, no weights. Tells
you the layer count, the intermediate size, whether a vision tower pollutes the
`down_proj` match, and how much RAM the classifier will want. That last number
is what usually ends these runs. Writes `preflight.json`.

### 1. Collect responses (LM Studio)

Point at any OpenAI-compatible server, local or over LAN.

```bash
python scripts/collect_responses_lmstudio.py \
  --base_url http://192.168.41.171:1234/v1 \
  --model ornith-1.0-9b \
  --data_path data/TriviaQA/rc.nocontext/train-00000-of-00001.parquet \
  --output_path data/consistency_samples.jsonl \
  --sample_num 5 --max_questions 200 --concurrency 4
```

Start small and read the `balanced pairs available` line. That yield rate tells
you how many questions you actually need to screen; scale from a measured number
rather than a guess.

For LM Studio, set **context length to 8192** (not 47k or 110k — the KV cache
reservation is what forces layers onto CPU) and raise **Max Concurrent
Predictions** to 4–8. Leave **Seed at -1**: a fixed seed makes all `sample_num`
generations identical and the consistency filter meaningless.

Resumable. Two servers can be driven in parallel with different
`--max_questions` slices and output paths; `cat` the results together.

### 2. Tag answer tokens

```bash
python scripts/make_answer_tokens.py \
  --input_path data/consistency_samples.jsonl \
  --output_path data/answer_tokens.jsonl \
  --model_path ornith-ai/Ornith-1.0-9B
```

Upstream pays GPT-4o to find the factual span. The collector already stored the
post-`</think>` answer, so this just tokenizes it. Free.

### 3. Balanced splits

```bash
python scripts/sample_balanced_ids.py --input_path data/answer_tokens.jsonl \
  --output_path data/train_qids.json --num_samples 400
python scripts/sample_balanced_ids.py --input_path data/answer_tokens.jsonl \
  --output_path data/test_qids.json --num_samples 150 --exclude data/train_qids.json
```

### 4. Extract activations

```bash
python scripts/extract_activations.py \
  --model_path ornith-ai/Ornith-1.0-9B \
  --input_path data/answer_tokens.jsonl \
  --ids_path data/train_qids.json \
  --output_root data/activations \
  --gpu_mem 14GiB --cpu_mem 40GiB \
  --locations answer_tokens all_except_answer_tokens
```

The slow stage. With `--gpu_mem` set, layers split across GPU and CPU; omit it
for pure CPU. Writes fp16 `.npy` per sample plus `neuron_index.json`.

Repeat with `--ids_path data/test_qids.json --output_root data/activations_test`.

### 5. Train and identify

```bash
python scripts/classifier.py \
  --acts_root data/activations --train_ids data/train_qids.json \
  --train_mode 3-vs-1 --penalty l1 --C 1.0 \
  --test_ids data/test_qids.json --test_acts_root data/activations_test \
  --out_dir models
```

`--C` is the sparsity knob and needs a sweep. Too high and you sweep in neurons
the model needs for general language, which wrecks it at intervention time. The
paper lands under 0.1% of all neurons; the script warns if you are far off.

Writes `models/classifier.npz` and `models/h_neurons.json`.

### 6. Visualise

```bash
python scripts/score_tokens.py \
  --model_path ornith-ai/Ornith-1.0-9B \
  --classifier models/classifier.npz \
  --input_path data/consistency_samples.jsonl \
  --n 8 --gpu_mem 14GiB --out report.html
```

Standalone HTML: every token shaded by its H-Neuron score, reasoning block
separated from the final answer, plus a per-layer histogram. The question worth
asking of it — does the signal spike *inside* the reasoning trace, before the
model commits? If so you have an early warning signal rather than a post-hoc
flag, which is the most interesting thing this whole exercise could produce.

### 7. Intervene and ship

```bash
python scripts/intervene_model.py \
  --model_path ornith-ai/Ornith-1.0-9B \
  --h_neurons models/h_neurons.json --scale 0.1   # dry run

python scripts/intervene_model.py \
  --model_path ornith-ai/Ornith-1.0-9B \
  --h_neurons models/h_neurons.json --scale 0.1 \
  --output_path models/ornith-9b-suppressed
```

Then convert and quantize for LM Studio:

```bash
python llama.cpp/convert_hf_to_gguf.py models/ornith-9b-suppressed \
  --outfile ornith-9b-suppressed-f16.gguf --outtype f16
./llama.cpp/build/bin/llama-quantize \
  ornith-9b-suppressed-f16.gguf ornith-9b-suppressed-Q8_0.gguf Q8_0
```

Drop the GGUF into `~/.lmstudio/models/local/ornith-9b-suppressed/`.

Because the edit is a static weight change it survives conversion — llama.cpp
control vectors could not express this, since they only add to the residual
stream and cannot scale individual MLP neurons. Verify it survived quantization
by comparing abstention rates on held-out questions before and after.

---

---

## Profiles: tune once, reuse everywhere

A suppression profile is the neuron set plus a tuned scale, keyed to a model
fingerprint. It is a few kilobytes of JSON, so you can keep one per model per
deployment config and check them into version control.

### 7. Tune the scale

```bash
python scripts/tune_scale.py \
  --model_path ornith-ai/Ornith-1.0-9B \
  --h_neurons models/h_neurons.json \
  --eval_path data/consistency_samples.jsonl \
  --scales 1.0 0.5 0.25 0.1 0.0 --n_eval 50 \
  --gpu_mem 14GiB --config_name trivia-q8 --save
```

Sweeps the scale and reports four numbers at each point: correct, abstained,
wrong, and perplexity on a canary text. The trade you are looking at is `wrong`
falling while `canary ppl` stays flat. If perplexity climbs, the classifier
selected too many neurons — go back and lower `--C`. Reaching for a gentler
scale instead just hides the problem.

The tuner will not pick a scale that damages the canary by more than 5%, and it
says so rather than silently choosing one.

### 8. Apply at runtime

```bash
python scripts/apply_profile.py --model_path <m> --list
python scripts/apply_profile.py --model_path <m> --config_name trivia-q8 \
  --prompt "Who wrote the novel Stoner?" --compare 1.0 0.25
```

Profiles apply as forward pre-hooks, not weight edits. Scaling column *j* of
`down_proj` is identical to scaling input element *j* before the matmul, so the
hook gives the same result with no weight mutation and a scale you can change
between generations. As a library:

```python
from profiles import Profile, SuppressionHandle
with SuppressionHandle(model, Profile.load(path)) as h:
    ...                  # suppressed
    h.set_scale(1.0)     # off, no reload
```

Profiles are keyed on architecture geometry (model type, layers, hidden,
intermediate, heads, vocab). Applying one to a mismatched model raises rather
than silently scaling arbitrary neurons. A fine-tune with identical geometry
gets a warning, because its neurons are different even though the shapes match.

### 9. Export

**As a LoRA adapter (hot-swappable):** the intervention is exactly a rank-*k*
update, `dW = W[:, S] @ diag(s - 1) @ E_S^T`, with *k* the H-Neuron count in
that layer. So it converts to a standard LoRA:

```bash
python scripts/export_lora.py --model_path <m> \
  --profile profiles/<fp>/trivia-q8.json --output_dir adapters/suppress
python llama.cpp/convert_lora_to_gguf.py adapters/suppress \
  --base <m> --outfile suppress-lora.gguf
./llama.cpp/build/bin/llama-server -m base-Q8_0.gguf \
  --lora-scaled suppress-lora.gguf 1.0
```

Because `dW` is linear in `(s - 1)`, the adapter's runtime scale a gives an
effective neuron scale of `1 + a*(s - 1)`. a=0 is the untouched model, a=1 is
the profile as tuned. `llama-server` exposes `/lora-adapters` to change it on a
running server, and `/completion` accepts per-request overrides — so suppression
strength becomes a dial rather than a rebuild.

**If you serve a fixed quant, pass `--gguf`.** llama.cpp applies the delta
against the *quantized* base. With `B` built from bf16 weights you get
`Q(W)[:,S] + W_bf16[:,S]*(s-1)`, so the base's quantization error rides along
unscaled. Point `--gguf` at the exact file you will serve and `B` is built from
`Q(W)` instead:

```
Q(W)[:,S] + Q(W)[:,S]*(s-1) = s*Q(W)[:,S]
```

Exactly the intended edit on the weights actually in use. Only the `ffn_down`
tensors are read, so it costs a fraction of a model load.

```bash
python scripts/export_lora.py --model_path <m> \
  --profile profiles/<fp>/trivia-q6.json \
  --gguf ~/.lmstudio/models/.../ornith-1.0-9b-Q6_K.gguf \
  --output_dir adapters/suppress
```

**As a baked model (maximum compatibility):**

```bash
python scripts/intervene_model.py --model_path <m> \
  --profile profiles/<fp>/trivia-q8.json --output_path models/suppressed
```

Or merge the adapter into a standalone GGUF with `llama-export-lora`. LM
Studio's LoRA support is less clearly documented than llama.cpp's; if it will
not load the adapter, merge and ship one file.

### What varies per config, and what doesn't

Neuron indices are a property of the model and do not transfer across models,
not even same-family ones. The *scale* is a deployment choice and is worth
re-tuning per config, because a Q4 base and a bf16 base do not respond
identically to the same suppression. Keep one profile per `(model, config)`
pair — `trivia-q8`, `trivia-q4`, `strict`, `lenient` — and record what each was
tuned against. `tune_scale.py` stores the full sweep in the profile's
`evaluations` list, so the numbers behind a choice stay with it.


---

## Vulkan and fixed quants

**Vulkan serves, it cannot analyse.** PyTorch has no usable Vulkan compute
backend, so stages 4 through 7 still need ROCm or CPU. Everything downstream of
the profile — LoRA adapter, baked GGUF, llama.cpp, LM Studio — runs on Vulkan
fine. LoRA is applied in the ggml graph rather than in backend code, so it is
backend-agnostic in principle; adapters have had backend-specific bugs though,
so load yours at scale 0 and confirm the output matches the plain base before
trusting it.

**A fixed quant makes the LoRA path strictly better than baking.** The base
GGUF is never touched or re-quantized, so there is no "did the edit survive
quantization" question at all. With `--gguf` the adapter is exact rather than
approximate. And because suppression strength is a runtime dial, you can do the
coarse sweep in PyTorch with `tune_scale.py` and then re-tune live against the
real Q6_K Vulkan server without re-exporting anything:

```bash
curl -X POST http://127.0.0.1:8080/lora-adapters \
  -H 'Content-Type: application/json' -d '[{"id":0,"scale":0.6}]'
```

That second pass matters. The scale `tune_scale.py` picks is measured on
dequantized or bf16 weights; the quant you deploy will not respond identically.
Treat the PyTorch number as a starting point and the live sweep as the answer.

**Quantizing does not shrink the extraction step.** Dequantized Q6_K weights
occupy the same memory as bf16 once they are PyTorch tensors, so a 9B model is
still ~18 GB and still needs `--gpu_mem` offload on a 16 GB card. Choosing a
6-bit deployment target does not make stage 4 fit.


## Layout

```
install.sh                          ROCm detection, venv, smoke test
requirements.txt                    torch excluded; install.sh picks the index
scripts/
  ns_common.py                      hooks, model loading, sequence construction
  profiles.py                       profile format, fingerprinting, pre-hooks
  preflight.py                      config inspection and resource estimates
  collect_responses_lmstudio.py     stage 1
  make_answer_tokens.py             stage 2
  sample_balanced_ids.py            stage 3
  extract_activations.py            stage 4
  classifier.py                     stage 5
  extract_activations_gguf.py       stage 4, llama.cpp path
  score_tokens.py                   stage 6
  tune_scale.py                     stage 7, PyTorch
  tune_scale_server.py              stage 7, over HTTP (no PyTorch)
  apply_profile.py                  stage 8
  export_lora.py                    stage 9a
  intervene_model.py                stage 9b
llama-tools/cett-dump/              llama.cpp eval-callback extractor
profiles/<fingerprint>/<config>.json
```

## What was changed from upstream, and why

| Change | Reason |
|---|---|
| Hooks filtered to text decoder layers | `'down_proj' in name` also catches a multimodal model's vision tower, corrupting the flat-index → (layer, neuron) map |
| Hook captures to CPU float32 | Upstream stacks hooked tensors and calls `.to(model.device)`, which raises the moment `device_map` splits layers |
| Layer index read from module path | Firing order is not guaranteed to be layer order |
| Sequence built as prompt + response ids | Chat templates for reasoning models strip `<think>` from prior assistant turns, so the upstream round-trip does not reproduce what was generated |
| Answer search starts after `</think>` | Otherwise the span matches inside the reasoning trace |
| `neuron_index.json` written explicitly | Intervention no longer has to assume uniform `intermediate_size` |
| Preallocated fp32 matrix, saga solver | `np.array(list_of_arrays)` copies, then liblinear promotes to fp64; ~4× the peak RAM |
| vLLM replaced with any OpenAI endpoint | Generation runs on a different machine, OS and GPU vendor than extraction |
| Truncated generations discarded | Counting an unclosed think block as a hallucination poisons the labels |
| `--max_layer_fraction` guard | Aborts intervention if a bad `C` selected enough neurons to damage general capability |
| Pre-hook instead of weight edit | Same maths, no weight mutation, scale changeable at runtime |
| Portable profiles keyed to a fingerprint | Reuse a tuned intervention; refuse to apply it to the wrong model |
| LoRA export | The edit is rank-k, so llama.cpp can hot-swap and rescale it |
| `--gguf` weight source | Building B from the served quant makes the adapter exact, not approximate |

## Status

Written against the papers and the upstream source, but **not executed against
real weights** — treat the first run of each stage as a debugging session. The
likeliest thing to need adjusting is `TEXT_LAYER_RE` in `ns_common.py` if your
model's remote code names its modules unconventionally. `preflight.py` will
show you the names.

## Licence

Upstream H-Neurons is MIT. These scripts follow suit.

---

## Extraction without PyTorch (llama.cpp, Vulkan)

`llama-tools/cett-dump/` is a llama.cpp tool that captures the same quantity the
PyTorch hooks do, via `cb_eval` -- the callback `llama-cvector-generator` uses.
For the node named `ffn_down-<il>`, `t->src[1]` is the input to down_proj, which
is exactly what `CETTManager` hooks.

| | weights | runtime | footprint | 16GB card |
|---|---|---|---|---|
| `extract_activations.py` | bf16 safetensors | PyTorch + ROCm/CPU | ~18 GB | needs offload |
| `extract_activations_gguf.py` | Q6_K GGUF | llama.cpp + Vulkan | ~8 GB | fits |

That is the whole point: the quantized path fits where the bf16 path does not,
and it removes ROCm from the extraction step entirely.

Extraction runs in two phases. Spans must be expressed in the tokenization that
produces the activations, so phase 1 gets the token ids with no forward passes,
the driver computes the regions, and phase 2 does the real work.

Reduction happens inside the tool. Writing raw `[n_tok, n_ff]` per layer would
be **918 MB per sample** at 500 tokens x 14336 x 32 layers; reducing over the
spans first makes it **1.8 MB**. The tool accumulates `|a| / ||layer output||`
and the driver applies the weight column norm afterwards -- that constant is
per-neuron and non-negative, so both mean and max commute with it and the split
is exact, not an approximation. It also means the tool never touches weights.

```bash
# build (see llama-tools/cett-dump/BUILD.md)
cp -r llama-tools/cett-dump ~/llama.cpp/tools/
echo 'add_subdirectory(cett-dump)' >> ~/llama.cpp/tools/CMakeLists.txt
cd ~/llama.cpp && cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-cett-dump -j

# extract
python scripts/extract_activations_gguf.py \
  --binary ~/llama.cpp/build/bin/llama-cett-dump \
  --gguf ~/models/ornith-1.0-9b-Q6_K.gguf \
  --tokenizer ornith-ai/Ornith-1.0-9B \
  --input_path data/answer_tokens.jsonl --ids_path data/train_qids.json \
  --output_root data/activations --ngl 99 --batch 4096 \
  --locations answer_tokens all_except_answer_tokens
```

Output is byte-compatible with the PyTorch path, so stages 5-9 are unchanged.

The tool emits its own token ids and the driver decodes those, so answer spans
are located against the tokenization that actually produced the activations
rather than a re-tokenization that might disagree at the boundaries.

**Do not mix the two paths in one training set.** This measures the quantized
model, the other measures bf16. For a fixed-quant deployment measuring the quant
is arguably more honest, but the neuron sets are calibrated to different weights.

### MoE

Supported, via the `ffn_moe_down-<il>` node. `ggml_mul_mat_id` puts the expert
ids in `src[2]`, so each (token, slot) already knows which expert produced it.
The index becomes three-axis:

```
flat = (layer * n_experts + expert) * n_neurons + neuron
```

Dense models report `n_experts = 1` and the axis collapses, so nothing
downstream changes. Means are divided by each expert's own routed-token count
rather than the span length -- a token only reaches the experts it was routed
to, so the span length is the wrong divisor.

**The feature count is the constraint, not the extraction.** A 48x128x768 MoE
is 4.7M features: 9.4 MB per sample, and a 400-pair 3-vs-1 classifier matrix of
**30.2 GB**. That does not fit 32 GB of RAM. Use `--top-experts`:

| kept | features | classifier |
|---|---|---|
| 128 (all) | 4,718,592 | 30.2 GB |
| 32 | 1,179,648 | 7.5 GB |
| 16 | 589,824 | 3.8 GB |
| 8 | 294,912 | 1.9 GB |

It ranks experts by mean activation over a sample of the training set and keeps
the busiest per layer. Rarely-routed experts are the right thing to drop: few
tokens reach them, so their per-cell estimates are the noisiest in the matrix
anyway.

There is a second, subtler cost. With top-8-of-128 routing each expert sees
roughly 6% of tokens, so per-expert estimates need far more data than the dense
case for the same reliability. Budget more collection, not just more RAM.

Selected neurons are written to `h_neurons_moe.json` keyed by `"layer:expert"`,
alongside the flat form in `h_neurons.json`.


---

## Low-power AMD (Vega iGPU, no ROCm)

On a Vega-class APU such as the Ryzen 7 7730U (`gfx90c`), ROCm is unsupported
and the `HSA_OVERRIDE_GFX_VERSION=9.0.0` masquerade is unreliable for PyTorch.
`install.sh` detects this and installs CPU torch. The pipeline still runs, but
the work has to be placed correctly:

| stage | where | why |
|---|---|---|
| 1 collect | fast GPU box | generation is memory-bound; ~4 tok/s on DDR4-3200 |
| 4 extract | the APU, Vulkan | prefill only, one pass per sample, no generation |
| 5 classify | the APU, CPU | sklearn; ~3 GB at 400 pairs, ~7 GB at 1000 |
| 7 tune | fast GPU box | `tune_scale_server.py` over HTTP |

Stage 4 is the one that runs well locally: it never generates, so the
bandwidth ceiling that makes stage 1 impractical does not apply. `cett-dump`
loads the model once for the whole manifest and resumes from existing dumps.

Stage 7 uses `tune_scale_server.py` against `llama-server` with the adapter
loaded, sweeping the adapter scale a rather than re-exporting. Effective neuron
scale is `1 + a*(s - 1)`, so one adapter covers the range. It reports a paired
McNemar comparison, since the same questions are asked at every scale with
greedy decoding and only changed verdicts carry information.

---

## Keeping LM Studio (and LM Link)

Nothing here touches LM Studio's install, config or model directory. The one
capability it lacks is adapter loading, so the GUI needs a pre-merged GGUF.

**Merge late.** Tune with base + adapter against `llama-server` -- one 8.3GB
file plus a ~30MB adapter, with the scale as a runtime dial -- and merge only
the scale you settle on. Merging every candidate costs a full model copy each.

```bash
source env.sh
./scripts/install_to_lmstudio.sh --lora adapters/suppress-lora.gguf \
    --alpha 1.0 --tag s010 --dry-run     # check the destination first
./scripts/install_to_lmstudio.sh --lora adapters/suppress-lora.gguf \
    --alpha 1.0 --tag s010
```

It infers LM Studio's model root from the base model's location, lays the file
out as `<publisher>/<repo>/<file>.gguf` so the GUI picks it up on rescan, refuses
to overwrite, checks free space first, and drops an `neuronscope.json` beside the
model recording which adapter and scale produced it. The tag goes in the name
because several suppression strengths will coexist in the model list and are
otherwise indistinguishable.

**LM Link** shares models that are *loaded on a peer device*, keyed by
`deviceIdentifier`. So the merged file has to exist on whichever machine
actually runs it. If you tune on the fast box and want the result reachable from
the laptop over LM Link, install it there, not locally.

---

## Improving results: where the leverage actually is

There is no training loop in this pipeline. The only free parameters are which
neurons and how hard, so a competitive signal against a stronger model has
nothing to optimise against. What does move the numbers, in order:

**1. Label quality.** Every mislabelled pair is noise the L1 classifier fits,
and it puts the wrong neurons in the profile -- which no amount of scale tuning
undoes. `rule_judge` is substring matching on gold aliases; with a verbose
reasoning model it errs in both directions. Measure before you trust it:

```bash
python scripts/judge_agreement.py --input_path data/consistency_samples.jsonl \
    --base_url http://host:8080/v1 --model <judge> --n 150
```

Cohen's kappa above 0.8 means keep the rule judge. Below 0.6 means your labels
are noise and nothing downstream is meaningful yet. Read the disagreements
rather than assuming the model judge is right -- it has its own failure modes,
and disagreement marks where labels are ambiguous, not which judge is correct.

**2. Domain match.** Neurons from TriviaQA short-form recall may do nothing for
hallucinated APIs in agentic coding. If `eval_code_hallucination.py` comes back
null, the fix is collecting stage 1 from code tasks, not a different alpha.

**3. Sample size.** More balanced pairs sharpens the neuron set, and the cost
lands on whichever machine runs stage 1.

A stronger model is useful here as a judge and as a source of adversarial eval
items. Using its outputs to fine-tune this model is a different project, and
one that most providers' terms prohibit.

---

## Training, and why this repo mostly is not it

There is no training loop here. NeuronScope suppresses; it cannot add capability.
If the goal is a better local model over time, the order of leverage is:

**1. Upgrade the checkpoint.** Ornith-1.5-9B is MIT-licensed and scores 47.0 on
Terminal-Bench 2.1 and 70.6 on SWE-Bench Verified. Zero GPU-hours, and it will
beat anything you can train at home. Do this before anything else.

**2. Train an abstention LoRA.** `export_sft_dataset.py` turns the collected
pairs into it directly: consistently-correct questions teach "answer",
consistently-hallucinated teach "decline". Same behavioural target as NeuronScope,
reached by gradient descent instead of a scalar on a neuron column, so it can
shape *when* to decline in a way clamping cannot. Needs a rented 24GB card for
an afternoon -- it will not run on a Vega iGPU or a 16GB card.

**3. Compare the two** on the same held-out set with
`eval_code_hallucination.py`. If the LoRA wins clearly, NeuronScope was scaffolding.
If it does not, that is a real result about how localised this behaviour is.

Replicating Ornith's own GRPO self-improvement loop is a different project at a
different scale -- it wants H100s, and the vendor already runs it and ships you
the checkpoints. What is worth keeping from this repo either way is the data and
verifier layer: consistency-filtered labels, a validated judge, and an
executable hallucination metric that is already shaped like a reward function.
The GPU is the cheap, rentable part; that layer is not.

---

## NeuronScope

Two frontends over one record format, because no single toolkit does both jobs.

**Web dashboard** (`viz/server.py`) -- stdlib only, no install, works
over the LAN. Pipeline state read from the filesystem, layer map, and a live
alpha sweep driver. This is the remote half.

```bash
python viz/server.py --root . --port 7860
python viz/server.py --root . --host 0.0.0.0   # LAN, no auth
```

**Fastplotlib explorer** -- for the 2D/3D activation map. Fastplotlib sits on
Pygfx/WGPU and is built for exactly this shape of data; a 32x14336 map is 458k
points, comfortably inside its envelope, and Vega 8's Vulkan support is
sufficient. Of the four candidates: Open3D is for point clouds and meshes, not
array-shaped scientific data; Piviz-3d I could not verify exists. Pygfx is the
right rendering layer and Fastplotlib is the right API on top of it.

```bash
python viz/explore.py runs/ornith-q6 --h-neurons models/h_neurons.json
python viz/explore.py runs/ornith-q6 --dump prepared.npz   # headless
```

Four panels: mean map, contrast (mean of hallucinated minus mean of correct),
a MIP volume over samples x layers x neurons, and per-layer profiles with
H-Neuron counts overlaid. Click a 2D panel to print the layer, neuron and value.

The contrast panel is the one worth looking at. Your H-Neurons should appear
there as bright columns; if the classifier found neurons the contrast map does
not corroborate, that is a warning about the neuron set, not about the plot.

All array preparation is in module-level functions that never import
fastplotlib, so `--dump` works headless with no GPU and any other frontend can
reuse them. The neuron axis is **max**-pooled for the volume, not mean-pooled:
H-Neurons are under 0.1% of neurons and averaging 14336 columns erases them.

The catch is that Fastplotlib renders locally and has no native remote mode --
its Jupyter backend is the only remote path and it is clunky. Hence the split:
dashboard for remote monitoring, Fastplotlib for local exploration, both
reading the same records.

**Record format** (`viz/records.py`) -- a directory, not a single file,
so it is append-only, resumable and survives a crash mid-run. Deliberately not
HDF5 or a database: a directory of `.npz` opens in numpy on any platform with no
driver and no version pinning, and you can rsync half of one off a remote box
while it is still being written.

```python
from viz.records import Recorder, Session
with Recorder("runs/ornith-q6", meta) as r:
    r.add(qid, agg=cett_aggregate, tokens=ids, scores=per_token, verdict="wrong")
    r.event("sweep", alpha=0.5, correct=182, wrong=11)
```

Sessions are keyed by model fingerprint **and quantization**. Two sessions from
the same model at different quants have identical shapes and different meanings,
so a shape check alone would not catch the mistake; `merge_check` compares both
and refuses.

---

## Comparing models

`viz/compare.py` sorts every pair into a comparability tier and refuses
index-level arithmetic outside the ones where indices mean the same thing.

| tier | condition | index-level? |
|---|---|---|
| IDENTICAL | same fingerprint and quant | yes -- a repeat measurement |
| REQUANTIZED | same fingerprint, different quant | yes -- same neurons, perturbed values |
| LINEAGE | same geometry, asserted post-training relationship | yes, but unverified |
| UNRELATED | different geometry, or different weights unasserted | no |

Layer 5 neuron 1234 in Ornith has no relationship to the same index in
Qwen3.5. Differencing two unrelated maps produces noise that looks convincingly
like structure, so the tool declines rather than drawing it.

Distribution comparisons (depth profile, concentration, p99) and behavioural
ones (which items each model got wrong, Jaccard over failure sets) are valid at
every tier, because neither depends on indices lining up. Depth profiles are
resampled onto ten relative-depth bins and normalized afterwards, so a 32-layer
and a 48-layer model are directly comparable.

```bash
python viz/compare.py runs/ornith-q6 runs/ornith-q4
python viz/compare.py runs/qwen35 runs/ornith10 --assert-lineage
```

The REQUANTIZED tier is where the interesting experiment lives: does Q4 recruit
different neurons than Q6? That is the mechanistic version of "low quants
hallucinate more", and unlike the folklore it is measurable.

`--assert-lineage` is a hypothesis you are asserting, not a fact the tool can
check -- post-training does not permute neurons, so indices *plausibly* still
correspond. The reported correlation is how you test it: near zero means the
assertion was wrong and the shifts are meaningless.

---

## Selective weight merging

`scripts/merge_selective.py` transplants or blends weights at neuron
granularity, guided by the H-Neuron map or a layer range.

```bash
python scripts/merge_selective.py --base A --donor B \
    --h-neurons models/h_neurons.json --dry-run
python scripts/merge_selective.py --base A --donor B --layers 8:24 \
    --alpha 0.5 --assert-lineage --output_path models/merged
```

**A neuron is three weight vectors, not one.** In a gated MLP, neuron *j* is
defined jointly by `gate_proj[j,:]`, `up_proj[j,:]` and `down_proj[:,j]`.
Splicing only the `down_proj` column pairs the base's activation pattern with
the donor's output projection, producing a neuron present in *neither* model.
Verified: a full triple transplant reproduces the donor neuron exactly, while a
`down_proj`-only splice matches neither source. The output stays fluent either
way, so nothing but an evaluation would catch it. This tool always moves all
three.

**Merging needs lineage, not just matching shapes.** Same tiering as
`compare.py`: identical fingerprints merge freely, different fingerprints with
matching geometry require `--assert-lineage`, and differing geometry is refused.
Two independently trained models have no neuron correspondence, so a merge
between them is noise -- fluent noise, which is worse.

`--dry-run` reports per-layer divergence on the selected neurons before writing
anything. Near-zero divergence means the merge is a no-op and not worth an
evaluation run.

## Which neurons for which task

`scripts/task_neurons.py` trains one classifier per task type and asks whether
they select the same neurons.

```bash
# label each collection run
python scripts/collect_responses_lmstudio.py ... --task trivia
python scripts/collect_responses_lmstudio.py ... --task code

python scripts/task_neurons.py --acts_root data/activations \
    --ids data/train_qids.json --samples data/consistency_samples.jsonl \
    --out_dir models/by_task
```

**Overlap is reported against chance, not raw**, because raw Jaccard is not
comparable across set sizes. Two independent sets of 400 drawn from 458,752
neurons share 0.35 cells by accident; two sets of 40,000 share 3,488. Measured
on planted data: two random 4,000-neuron sets score 0.77x -- chance -- while
two sets sharing a 1,200-neuron core score **35x** despite a Jaccard of only
0.18. Jaccard alone would have called that "mostly different".

Three outcomes, all worth having:

- **mostly shared** — hallucination is a general mechanism in this model; one
  profile transfers, and a task specialist buys nothing on this axis
- **mostly disjoint** — task-specific failure modes; a profile per domain is
  warranted, and delegating to a specialist has something underneath it
- **partly shared** — the intersection is the general mechanism, the remainder
  is where a domain profile earns its keep

Each task gets a normal `h_neurons_<task>.json`, so you tune, export and
evaluate it exactly like any other profile.

## Merge methods, ranked

`scripts/merge_methods.py` lists all 18 methods in the installed mergekit,
ordered by how likely each is to work on a first attempt, and generates a valid
config for any of them.

```bash
python scripts/merge_methods.py                    # ranked list with rationale
python scripts/merge_methods.py --short            # names and summaries only
python scripts/merge_methods.py --verify           # check against your mergekit
python scripts/merge_methods.py --method slerp --models A B
python scripts/merge_methods.py --method ties --base BASE --models A B C
```

**Reach for first:** `slerp` (exactly two models -- averaging flattens weight
magnitudes, arc interpolation preserves them), `ties` (three or more, with a
base -- handles the sign conflicts and magnitude swamping that break naive
averaging), `dare_ties` (usually a small win over TIES for the same cost).

**Narrower use:** `task_arithmetic` -- still the right tool for *removing* a
behaviour, since a negative weight subtracts one. `linear` when the models are
already very close. `model_stock` when you genuinely have three or more
fine-tunes of one base.

**With a reason:** `dare_linear`, `della`, `della_linear`, `breadcrumbs`,
`breadcrumbs_ties`, `nuslerp`, `multislerp`, `karcher`, `sce`, `nearswap`,
`arcee_fusion`.

**Special cases:** `passthrough` is not a merge -- it stacks layer ranges and
changes the parameter count, so nothing else in this toolkit applies to the
result. `neuronscope_select` is ours and needs a trained classifier first.

The generator enforces each method's arity and base requirement, so
`slerp` with three models or `ties` without a base fails immediately rather
than at merge time.

Whatever you pick, run the result through `merge_eval.py`. A merge that
retained nothing still produces fluent text; a paired comparison against the
sources on untuned tasks is the only evidence you get.

## mergekit integration

mergekit already does general merging well -- TIES, DARE, SLERP, task
arithmetic, model stock, SCE -- and reimplementing any of it would be waste.
What NeuronScope adds is a different **selection criterion**.

TIES and DARE decide what to keep by *magnitude*: large deltas survive, small
ones are pruned or randomly dropped. That is a statistical heuristic with no
reference to what the parameters do. NeuronScope selects by a classifier trained on
measured behaviour. Semantic selection instead of magnitude selection, and to
my knowledge nobody has tried it as a merge criterion.

`scripts/mergekit_neuronscope.py` registers `neuronscope_select` via mergekit's
`@merge_method` decorator:

```bash
pip install -e /path/to/mergekit
export NEURONSCOPE_NEURONS=models/h_neurons.json
python scripts/mergekit_neuronscope.py            # prints a config to start from
mergekit-yaml merge.yaml ./out
```

```yaml
merge_method: neuronscope_select
base_model: ornith-ai/Ornith-1.0-9B
models:
  - model: ornith-ai/Ornith-1.5-9B
    parameters: {weight: 1.0}
parameters:
  invert: false      # true merges everything EXCEPT the H-Neurons
```

`invert: true` is the more interesting run: keep the base's H-Neurons and take
the donor everywhere else. If the H-Neurons matter, the two directions should
behave differently, and that difference is the experiment.

Attention, norm and embedding tensors pass through as the base unchanged -- the
neuron index describes the MLP intermediate dimension and means nothing
elsewhere. gate/up are indexed by row, down_proj by column, so the complete
triple moves together (see `merge_selective.py` for why a partial one is
silently wrong).

**On mergekit-moe.** frankenMoE routers are randomly initialised or derived from
prompt hidden states, not trained. Ornith already ships genuinely trained MoE at
35B-A3B; assembling one from 9B dense copies is very likely worse than just
using it.

**Lineage is unchecked.** mergekit will merge any two models with matching
shapes. Neuron indices correspond only if both were post-trained from the same
base -- true for Ornith-1.0 and 1.5 (both from Qwen 3.5), false in general.
Neither mergekit nor this plugin can verify it.

## Merging and measuring the result

`scripts/merge_eval.py` compares any set of served endpoints on the same tasks
and reports **what improved and what regressed, per item**. It does not care how
the variants were produced -- mergekit, `merge_selective.py`, a fine-tune, or
just two checkpoints. Serve each and point at it.

```bash
python scripts/merge_eval.py \
    --endpoint base=http://127.0.0.1:8080 \
    --endpoint merged=http://127.0.0.1:8081 \
    --tasks data/eval_tasks.jsonl --reference base --out cmp.json
```

Aggregate accuracy hides the answer. Going from 60% to 64% might be 12 fixed
and 8 broken, or 4 fixed and 0 broken -- identical headline, completely
different models. Verified: those two cases give chi2 0.80 and 4.00
respectively, one not significant and one significant. The tool prints the
decomposition and names the regressions.

Three task kinds, mixable in one file:

```
{"id": "qa-1",  "prompt": "...", "aliases": ["Paris"]}      graded on aliases
{"id": "cod-1", "prompt": "...", "modules": ["requests"]}   graded by symbol resolution
{"id": "can-1", "prompt": "..."}                            canary: answered or refused
```

**Include canaries.** They are ungraded general-capability prompts, and they
catch the failure mode that a target metric cannot: a merge whose hallucination
rate improves because the model now refuses more. In the test above a variant
scored +1 net while newly refusing 3 of 4 canaries -- narrower, not better, and
invisible without them. `data/eval_tasks.example.jsonl` shows the format.

## MoE active-parameter sweeps

The top-k expert count is baked into the GGUF as `<arch>.expert_used_count`,
and llama.cpp lets you override it at load time -- so you can trade quality
against speed with no re-quantization:

```bash
python scripts/sweep_experts.py \
    --gguf ~/models/Ornith-1.5-35B-A3B-Q4_K_M.gguf \
    --server ~/llama.cpp/build/bin/llama-server \
    --tasks data/eval_tasks.jsonl --experts 2 4 6 8 --ngl 99 \
    --override-tensor '\.ffn_.*_exps\.=CPU'
```

It reads the architecture and expert count from the GGUF itself (the override
key is `qwen3moe.` for one model and `glm4moe.` for another, so hardcoding it
breaks), launches a server per value, measures tokens/sec, and runs the same
paired comparison as `merge_eval.py`.

Read it as a curve, not a winner: if k=6 matches k=8 with no significant
difference and runs faster, the extra experts were buying nothing.

**Dense models have no such knob** and the script says so rather than failing
obscurely. Ornith 9B is dense; this needs 35B-A3B or 397B.

The `--override-tensor` regex above is the standard consumer-hardware MoE
recipe: routed experts stay in RAM, always-active tensors (attention, shared
experts, embeddings) go to the GPU. Generation is bandwidth-bound on *active*
parameters, so a Q4 35B-A3B reads roughly 2-3GB per token against 8.28GB for a
dense Q6 9B -- potentially faster despite being four times the size.

## Visualization modes

| mode | shows | where |
|---|---|---|
| token trace | per-token classifier score, reasoning vs answer | `score_tokens.py` -> HTML |
| dashboard | pipeline state, per-layer H-Neuron bars, live alpha sweep | `viz/server.py`, remote |
| explorer | mean map, contrast map, 3D MIP volume, click-inspect | `viz/explore.py`, local GPU |
| comparison | depth profiles, concentration, failure overlap, tiered index diff | `viz/compare.py` |

| weights | magnitude, quantization error, H-Neuron enrichment | `viz/weights.py` |

### Time-resolved 3D

`scripts/trace_sample.py` captures CETT **per token** rather than aggregated
over a region. No change to cett-dump was needed: spans are arbitrary
`[start, end)` ranges, so one span per token gives `[tokens, layers, neurons]`.

```bash
python scripts/trace_sample.py --binary ... --gguf ... --tokenizer ... \
    --input_path data/consistency_samples.jsonl --qid <id> \
    --out runs/trace-<id> --bin-neurons 512 --classifier models/classifier.npz
python viz/timeline.py runs/trace-<id> --theme ember --play
```

Size is why this is per-sample: one 500-token sequence on a 32x14336 model is
459 MB unbinned, about 16 MB at 512 bins. Binning max-pools, never means --
H-Neurons are under 0.1% of neurons and averaging erases them.

**What counts as hallucinating.** A neuron is not flagged for being in the
H-Neuron set; those fire constantly on grounded text too. It is flagged when it
is firing *and* the classifier score for that token is high, so the marking
tracks the moment rather than the membership. Verified: with a burst planted on
two cells across five frames, the player flags exactly those frames and exactly
those cells, and flags nothing when the score signal is removed.

The active threshold is global across the trace, not per frame. A per-frame
percentile would mark the same fraction active at every token and erase the
variation the animation exists to show.

### Three frontends, one backend

| frontend | renderer | for |
|---|---|---|
| `viz/timeline.py` | pygfx / WGPU | analysis. Exact sizes, no effects. |
| `viz/godot/` | Godot 4.7 Forward+ | standalone desktop, real glow |
| `viz/bloom.py` | three.js in a browser | remote, zero install |

Parity survives because the clients are thin. `viz/bloom.py` serves the API all
three consume:

    GET /api/meta     frames, cells, layers, per-frame z, labels, flagged
    GET /api/theme    the selected theme, from themes.json
    GET /api/trace    i32 T,N,L | i32 layer[N] | i32 neuron[N]
                      | f32 intensity[T][N] | u8 state[T][N]

The Godot client is verified against the Godot 4.7 source, not from memory --
which caught three bugs: `background_mode = 3` is `BG_CANVAS` (wanted
`BG_COLOR = 1`), `billboard_mode = 3` is `BILLBOARD_PARTICLES` (wanted
`BILLBOARD_ENABLED = 1`), and the HUD `Label` was parented to a `Node3D`
instead of a `CanvasLayer`, so it inherited no canvas transform.

State (0 idle, 1 active, 2 flagged) is decided **once, in Python**. The clients
only draw. One implementation of the thresholding, the H-Neuron mapping and the
scoring; three renderers. `tests/` asserts the byte offsets, state semantics,
point sizes and frame rate match across the Godot and three.js clients, so a
change to one that is not mirrored in the other fails.

It also keeps the GPU free: classification is CPU work done once at load, not
per frame on the device that is also running the model. Godot uses a MultiMesh
so the whole field is one draw call with half-resolution glow; three.js sends
positions once and streams only intensities.

```bash
python viz/bloom.py runs/trace-abc --theme ember --host 0.0.0.0   # backend + web
NS_API=http://127.0.0.1:7880 godot --path viz/godot                # desktop
python viz/timeline.py runs/trace-abc --theme ember                # analysis
```

Bloom is deliberately absent from `timeline.py`. It blurs neighbours together,
so a bright cell reads as a smear and you lose the spatial precision the view
exists for. Use the pretty ones to show people; use pygfx to decide anything.

**Themes** (`--list-themes`): `clinical`, `ember`, `cool`, `mono`. Each sets
colormap, background, the two accent colours, and how state maps to point size
and opacity. Pygfx has no bloom or glow post-processing, so there are no shader
effects -- at a few hundred thousand points, size and opacity read better than
a glow would anyway.

### Weight views

`viz/weights.py` reads the `ffn_down` tensors directly -- a fraction of a model
load -- and answers what activations cannot.

```bash
python viz/weights.py --gguf model-Q6_K.gguf --reference model-F16.gguf \
    --h-neurons models/h_neurons.json
python viz/weights.py --gguf ... --reference ... --dump w.npz   # headless
```

**Magnitude** is `||W[:, j]||` per neuron. A neuron that fires often but writes
weakly is not the same as one that fires rarely and writes hard; CETT folds the
two together and this separates them.

**Quantization error** is the relative per-neuron difference between two quants
of one model. Normalising by the reference norm is load-bearing: absolute error
correlates with magnitude at r=0.97 and would just redraw the magnitude map,
while the relative version sits at r=0.06.

**Enrichment** is the one worth running. It tests whether the H-Neurons fall in
the high-error tail more often than chance, against a permutation null rather
than an assumed distribution -- the values are neither independent nor normal,
so a parametric test would be quietly wrong. A positive result is a mechanistic
link between quantization level and hallucination, not a correlation between two
summary numbers. It is the version of "low quants hallucinate more" that can be
measured instead of repeated.

---

## VRAM budget

`scripts/vram_budget.py` computes the three competing costs from the model's
own architecture rather than a rule of thumb.

```bash
python scripts/vram_budget.py --gguf model-Q6_K.gguf --ctx 8192
python scripts/vram_budget.py --gguf ... --ctx 110592 --cache-type q8_0 --viz godot
```

The KV cache is the term people get wrong. It scales linearly with context and
is **independent of model quantization** -- a Q4 and a Q8 of one architecture
have identical cache costs.

```
bytes = 2 * layers * kv_heads * head_dim * ctx * elem_size
```

`kv_heads` is the grouped-query count, not the attention head count. Using the
latter overestimates by the GQA ratio, commonly 4x or 8x. Verified against
Llama-3-8B: 32 layers, 32/8 heads, head_dim 128, f16 at 8192 ctx = exactly
1.00 GiB.

On a 9B Q6_K with a 12 GiB carve-out:

| context | KV cache | layers on GPU | visualizer |
|---|---|---|---|
| 8192 | 1.00 GiB | 32/32 | GPU |
| 47104 | 5.75 GiB | 21/32 | CPU |
| 110592 | 13.50 GiB | 0/32 | CPU |

At 110K the cache alone exceeds the whole carve-out before a single weight.
When things do not fit, the visualizer is evicted to CPU **first** -- it being
slower to draw does not slow generation, and generation is the long pole.

## Hardware profiles

A config says what you want; a host says what a machine can do. Keeping them
apart is what lets one config be checked against several machines.

```bash
python scripts/hostprofiles.py detect --save laptop --vram 12
python scripts/hostprofiles.py add rig --vram 16 --ram 32 --cores 16 \
    --gpu "RX 9060 XT" --backend vulkan --dedicated
python scripts/hostprofiles.py plan --gguf model.gguf --ctx 47104
```

`plan` runs one config against every saved machine and prints a verdict per
host. `detect` fingerprints the current box so the right profile is selected
automatically, and configs can name a target host -- so you can check what will
fit on the rig from the laptop.

**Dedicated versus shared matters more than the number.** A discrete card's
VRAM is exclusively yours; an iGPU carve-out comes out of system RAM and the
desktop compositor draws from the same pool, so shared profiles reserve 1.6x
more headroom. `detect` will not guess an iGPU's carve-out -- it is a BIOS
setting and is not reported -- so pass `--vram` for those.

## Named configs

A config is a complete reusable setup under a name: model, load settings,
preset, visualizer, cache type and hardware limits. The name is passed to
llama-server as `--alias`, so **Cline sees the config name** rather than a
filename.

Every setting is audited against the model's real architecture, and warnings
attach to the field that caused them so the UI marks the setting rather than
listing complaints at the bottom. `turboquant` is flagged as an **error**: it
has no implementation in llama.cpp, being a research method for KV-cache
compression rather than a switch. `cache_type q8_0` is the same lever at lower
sophistication and does exist.

## Realtime, without a fork

Four ways to get activations while you work, ranked by what is available today:

| | how | available | latency |
|---|---|---|---|
| **A** | trace the finished response (`scripts/autotrace.py`) | **now** | one prefill behind |
| **B** | PyTorch hooks during `generate()` | now, needs bf16 in torch | per token |
| **C** | `llama-cpp-python` + `cb_eval` via ctypes | no compile, fiddly | per token |
| **D** | fork `llama-server --activations` (`llama-tools/server-activations/`) | **written, uncompiled** | per token |

The assumption worth dropping is that activations must arrive *with* the
tokens. They do not.

### A. The tracing proxy

`scripts/autotrace.py` sits in front of llama-server, forwards requests
untouched, streams the response back with no added latency, then traces the
completed text in the background and pushes frames to the viewer.

```bash
llama-server -m model.gguf --port 8080 &
python viz/stream.py --token secret --host 0.0.0.0 &
python scripts/autotrace.py --upstream http://127.0.0.1:8080 \
    --binary ~/llama.cpp/build/bin/llama-cett-dump --gguf model.gguf \
    --tokenizer ornith-ai/Ornith-1.5-9B --n-layers 36 \
    --publish http://127.0.0.1:7890 --port 8088
```

Point Cline or Studio at `:8088` instead of `:8080`; nothing downstream knows.
The request is forwarded unaltered, so **the traced run is the served run** --
a proxy that changed sampling would be visualising a different generation than
the one you read.

Cost is one extra prefill per traced response on whichever machine holds the
GGUF, which competes with the next generation for the GPU. So tracing is
skipped while a trace is already running, and `--trace-every N` samples instead
of tracing everything.

`viz/stream.py` gained `/api/push`, so a trace produced anywhere on the network
can drive the viewer.

## D. The fork

`llama-tools/server-activations/` adds `--activations` to llama-server. One
self-contained header and three insertion points in `server.cpp`, documented in
`PATCH.md`. Configuration comes from the environment rather than the argument
parser, so `common/arg.cpp` is untouched -- one fewer file to re-merge when
llama.cpp moves.

```bash
cp ns_activations.h llama.cpp/tools/server/     # then apply PATCH.md
NS_ACTIVATIONS=sparse NS_CLASSIFIER=models/classifier.bin \
  ./build/bin/llama-server -m model.gguf --parallel 1 --port 8080
python viz/stream.py --source http://127.0.0.1:8080 --token secret
```

**It carries the classifier.** Without one the server can report activity but
cannot flag anything -- "this neuron is busy" and "this token is likely
fabricated" are different claims, and only the second needs trained weights.
`export_classifier_bin.py` writes a flat float32 blob the server reads with no
numpy and no parser, and the server refuses it if the length does not match the
model. Every frame carries `scored`, and the client shows "peak (no
classifier)" rather than a flag when it is false.

**Idle costs nothing.** The callback returns before any device copy when no one
is subscribed, so the flag can stay on.

**Single slot only.** With `--parallel > 1` a batch interleaves tokens from
several sequences and the node carries no recoverable sequence id, so it
refuses to enable rather than emitting frames that mix conversations.

**Live scores are not replay scores.** The server emits `|a| / ||layer output||`
without the `||W[:, j]||` factor, since holding dequantised down_proj weights in
RAM is a large cost for a live view. Same shape, same story, different scale.

Untested: there is no compiler here. The wire format is verified against the
Python client, and the classifier blob round-trips byte-identically, but the
first build will need real eyes.

## Live streaming (protocol built, source not)

`viz/stream.py` implements near-realtime activation streaming for a phone or
SBC. **The source does not exist yet**: it needs `llama-server --activations`,
a fork that reduces in-process and emits alongside the token stream. The eval
callback in `llama-tools/cett-dump` fires during generation as well as prefill,
so the C++ side is a known quantity; the protocol is the part that needed
designing and testing, and that is what this is.

```bash
python viz/stream.py --simulate --token secret --host 0.0.0.0
```

Bandwidth drives the design. Measured on a 36x14336 field:

| tier | per token | at 4 tok/s | for |
|---|---|---|---|
| raw | 2.5 MB | 10 MB/s | nothing; it is here as the baseline |
| binned | 92 KB | 360 KB/s | a desktop on the same switch |
| sparse | 0.7 KB | 2.7 KB/s | a phone, over LTE |

Reduction happens server-side and the client picks a tier, so a phone asking
for `sparse` and a desktop asking for `binned` share one generation. Binning
max-pools: mean pooling over 28 neurons would turn a 1.8 peak into 0.08.

Subscribers are independently backpressured with a bounded queue and drop
frames rather than blocking. A stalled phone must not stall the desktop, and
more importantly must not stall generation behind it.

Auth is the same bearer token or cookie as Studio, with a query-string fallback
because `EventSource` cannot set headers. TLS via `--tls-cert`, though on an
untrusted network a WireGuard or Tailscale tunnel beats a self-signed cert that
users learn to click through.

## Doctor

`scripts/doctor.py` walks every prerequisite in dependency order and reports
what works, what is missing, what that blocks, and the fix.

```bash
python scripts/doctor.py
python scripts/doctor.py --gguf ~/models/model-Q6_K.gguf
python scripts/doctor.py --json
```

It checks python, packages, torch and Vulkan devices (including the ICD-conflict
warning), the llama.cpp binaries this project needs, the model's architecture
and whether its `ffn_down` tensors are dense or MoE, which pipeline stages have
output, and whether the ports are free. It ends with a single next action rather
than a wall of status.

## Host capability checks

`scripts/hostcheck.py` estimates a mode's peak cost before it starts. Several
stages here fail in ways that are obvious in hindsight and invisible in advance
-- a MoE classifier matrix is 30GB on a model whose weights are 20GB, and you
find out an hour in.

```bash
python scripts/hostcheck.py                      # what this machine has
python scripts/hostcheck.py --mode classifier \
    --layers 48 --experts 128 --neurons 768 --pairs 400
```

It is wired into `classifier.py` and `merge_selective.py`, which warn by
default and refuse with `--strict-host`. A warning you can override beats a
crash four steps in, and beats a hard limit that is wrong about your machine.

It reads `MemAvailable` rather than `MemFree`, since page cache is reclaimable
and counting it as used would understate what a large allocation can get.

---

## Studio

`viz/studio.py` is a model manager, server supervisor and chat window.

```bash
python viz/studio.py --models-dir ~/.models \
    --server ~/llama.cpp/build/bin/llama-server
python viz/studio.py --models-dir A --models-dir B --host 0.0.0.0
```

It does not implement inference, tokenization, sampling or GGUF loading --
llama.cpp does all of that. What it adds is the layer around it: scanning for
models and reading their metadata, per-model settings that persist, starting
and stopping the server, streaming chat, and the two runtime controls LM Studio
does not expose:

- **suppression α slider**, live via `/lora-adapters`, no reload
- **active expert count** for MoE, with the override key read from the model's
  own architecture rather than hardcoded

It also estimates fit before loading, reusing `hostcheck.py`, so a model that
will not fit says so in the list rather than after a failed load.

**Hub search and download.** A second tab searches Hugging Face for GGUF repos,
lists their files grouped so multi-part models are offered as a set rather than
as pieces, and downloads with resume. A dropped connection picks up where it
stopped; cancelling keeps the `.part` file and never publishes a truncated
model. Files land as `publisher/repo/file.gguf`, which is what the scanner and
LM Studio both expect. Set `HF_TOKEN` for gated repos.

**Presets** are chat-time, separate from load settings, so switching a system
prompt or temperature does not restart the server. Four built in (`default`,
`deterministic`, `ornith recommended`, `terse`); saving your own persists to the
settings file.

Streaming is SSE passthrough, including `reasoning_content` rendered separately
from the answer. That matters more than it sounds: at a few tokens per second
on an iGPU, a non-streaming chat window is indistinguishable from a hang.

Multi-part GGUFs are listed once by their `-00001-of-` part, which is what
llama.cpp wants passed to `-m`.

Stdlib only, apart from the optional `gguf` package for metadata. No build
step, no bundled browser, and it works over the LAN because the GPU is often on
another machine. **No authentication** -- `--host 0.0.0.0` puts model loading
and chat on your network.

**Authentication.** `--token` (or `NS_STUDIO_TOKEN`) gates every route. Without
it the server prints a warning when bound to `0.0.0.0`, because anyone reaching
the port could otherwise load models, download files and run inference.
Comparison is timing-safe; the cookie is HttpOnly and SameSite=Strict.

**Speculative decoding.** A draft model field on the load panel wires `-md`,
`--draft-max` and `--draft-min`. This is the biggest speed lever on a
bandwidth-bound host: a small model proposes tokens the large one verifies in a
batch, so per-token weight reads drop.

**Structured output.** Presets carry a JSON schema or a GBNF grammar, passed to
llama-server as constrained decoding -- a guarantee about output shape, not a
request. A malformed schema is rejected with a 400 rather than confusing the
server.

Still missing versus LM Studio: RAG, MCP and LM Link. The first two are each
project-sized and neither serves what NeuronScope is for; LM Link is
proprietary device pairing, and `--host` with `--token` covers the practical
need.

## Tests

```bash
python tests/test_audit.py
```

Exercises the cross-file contracts with a torch stub, so it runs anywhere with
no model and no GPU: profile dims match what `intervene_model.py` reads, the
geometry gate rejects mismatched models, hooks attach only to text decoder
layers and never the vision tower, out-of-range neuron indices and nonexistent
layers are refused, and `TEXT_LAYER_RE` is defined exactly once.
