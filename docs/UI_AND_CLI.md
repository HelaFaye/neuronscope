# Every feature: in the UI, on the command line, and for agents

After `./install.sh`, nothing needs a terminal: start **NeuronScope** from the
app menu (or `./neuronscope`, or `NeuronScope.bat` on Windows) and Studio opens
in your browser. Everything below is also scriptable, for people who prefer a
shell and for automation; and the third column is what an AI agent (Cline,
Claude Desktop) can do through NeuronScope's MCP server ([CONNECT.md](CONNECT.md)).

Jobs on the Jobs page are the same scripts as the command-line column, built
from typed forms: no free-form command line, values cannot smuggle in flags,
each run keeps its log, and the form's fields are tested against the script's
own options.

## Setting up

| feature | in the UI | command line | MCP tool |
|---|---|---|---|
| Start Studio and open it | app menu → NeuronScope | `./neuronscope`, `python scripts/launch.py` | — |
| Models folder, llama-server, cett-dump, unloading, routing thresholds, worker limits | **Setup** | `~/.neuronscope/config.json`, or Studio's flags | — |
| Build llama.cpp with cett-dump | **Setup → Build**, or Jobs → Setup | `scripts/build_llama_tools.sh --backend vulkan` | `start_job build_llama` |
| Health check (packages, GPUs, build, model, pipeline) | **Setup → Health check** | `python scripts/doctor.py` | `doctor` |
| Hardware for worker models (ROCm, Vulkan, CUDA, Metal, CPU) | **Setup → Hardware** | `~/.neuronscope/hardware.json`, `python scripts/accelerators.py` | `devices` |
| Restart Studio | **Setup → Restart Studio** | stop and start it | — |
| What each feature needs (packages, programs, llama.cpp backends, drivers, devices, memory), with fixes for this OS | **Setup → Requirements** | `python scripts/ns_requirements.py` | `requirements` |
| Install a missing Python package | **Setup → Requirements → Install**, or Jobs → Setup | `python scripts/ns_requirements.py install <pkg>` | `start_job install_package` |
| Environment snapshots and what changed | **Setup → Requirements → snapshots** | `ns_requirements.py snapshot`, `diff`, `freeze` | `start_job env_snapshot` |

Network exposure (binding beyond this computer, tokens, TLS, remote jobs) is
deliberately not on the Setup page: set it in `config.json` or with flags
([SECURITY.md](SECURITY.md)).

## Using models

| feature | in the UI | command line | MCP tool |
|---|---|---|---|
| Find and download models (Hugging Face) | Studio → **Models** | `hf download …` | — |
| Load, unload, per-model settings | Studio → **Models** | `POST /api/load`, `/api/unload` | `load_model`, `unload_model` |
| Chat (images, saved chats, presets) | Studio → **Chats** | `/v1/chat/completions` (any OpenAI client) | `chat` |
| `auto` routing by measured skill | model picker → auto | `"model": "auto"`; `python scripts/model_stats.py rank "…"` | `route_prompt` |
| Check a reply for hallucination risk, 3D view of it | chat → **check** | `POST /api/trace` | `check_reply` |
| Chat with documents (RAG) | Studio → **Docs** | `python scripts/rag.py` | — |
| MCP tools inside chat | Studio → **Tools** | `~/.neuronscope/mcp.json`, `python scripts/mcp_client.py` | — |
| Other devices and linked machines | Studio → **Link** | `/api/pair/*`, `/api/links/*` | — |
| Cline, Claude Desktop, OpenAI apps | **Connect** | `python scripts/ns_connect.py` | — |

## Measuring and improving models

| feature | in the UI | command line | MCP tool |
|---|---|---|---|
| TestQA (graded, per subject) | **Jobs → Evaluate** | `python scripts/testqa.py` | `start_job testqa` |
| Per-model, per-subject results | Studio → Models (stats) | `python scripts/model_stats.py show` | `model_stats` |
| SWE-bench, LiveBench, CLIP_benchmark | **Jobs → Benchmarks** | `python scripts/benchmarks.py`, `clip_bench.py` | `start_job …` |
| Retraining on failures (data, fine-tune, merge) | **Jobs → Retrain** | `deficits.py`, `finetune.py`, `merge_export.py` | `start_job …` |
| Compare models, item by item; does suppression help | **Jobs → Analysis** | `merge_eval.py`, `compare_models.py` | `start_job …` |
| Neuron review: which neurons go with wrong answers, by source, subject and time; 3D over time | **Review** | `python scripts/neuron_review.py` | `review_sources`, `review_summary`, `review_label` |
| Add TestQA runs and other benchmarks to the review | **Jobs → Review** | `neuron_review.py ingest-testqa`, `ingest-items` | `start_job review_testqa`, `review_items` |
| Mark a checked chat reply right or wrong | chat → check → *right / wrong* | `POST /api/review/label` | `review_label` |
| Projects split by skill, worker models, review | **Projects** | `python scripts/director.py analyze`, `/api/projects` | `create_project`, `get_project`, `review_task`, … |

## Finding and editing H-Neurons

| feature | in the UI | command line | MCP tool |
|---|---|---|---|
| Pipeline stages 0–9 | **Jobs → H-Neurons** (in stage order) | the scripts in [PIPELINE.md](PIPELINE.md) | `start_job preflight`, `collect`, … |
| What each stage has produced | **Lab → Pipeline dashboard**; Setup → Health check | `python scripts/doctor.py` | `doctor` |
| Edit a GGUF, export a LoRA | **Jobs → H-Neurons** (stage 9) | `suppress_gguf.py`, `export_lora.py` | `start_job suppress_gguf` |
| Vision (CLIP/SigLIP), mmproj edits | **Jobs → Vision** | `clip_neurons.py`, `suppress_mmproj.py` | `start_job clip_*` |
| Detector checks (cross-validation, per task, where it spikes) | **Jobs → Analysis** | `cv.py`, `task_neurons.py`, `when.py`, `score_tokens.py` | `start_job …` |
| Selective merge, MoE expert sweep | **Jobs → Merge** | `merge_selective.py`, `sweep_experts.py` | `start_job …` |
| Quantization lab, scale sweep, adaptive tuning | **Lab** | `viz/quant_lab.py`, `viz/scale_sweep_lab.py`, `viz/adaptive_tuning_lab.py` | — |

## Looking inside

| feature | in the UI | command line | MCP tool |
|---|---|---|---|
| Live activations while generating | **Lab → Live** | `python viz/stream.py` | — |
| 3D replay of a trace or checked reply | **Lab → Replay**; chat → check → open 3D view | `python viz/bloom.py <trace>`; Godot client | `list_traces` |
| Timeline, explorer, weights (desktop windows) | **Jobs → Views** | `viz/timeline.py`, `viz/explore.py`, `viz/weights.py` | `start_job view_*` |
| Compare activation sessions | **Jobs → Views → Compare** | `python viz/compare.py` | `start_job compare` |
| Trace one sample for the 3D views | **Jobs → H-Neurons → Trace one sample** | `python scripts/trace_sample.py` | `start_job trace` |

## Moving models

| feature | in the UI | command line | MCP tool |
|---|---|---|---|
| Browser-to-browser encrypted transfer | **Lab → Model transfer** | `python viz/transfer_lab.py` | — |
| Resumable HTTPS transfer, watch mode | — | `python scripts/ns_transfer.py` | (`ns_transfer_mcp.py`) |
| Remote tuning worker | Lab → Adaptive tuning | `python scripts/tuning_worker.py` on the worker | — |

The worker side of a remote transfer or tuning run is started on the other
machine, by design: it opens a port there.
