# Credits

NeuronScope is built on other people's research and software. This page
lists every paper, method, dataset and project it uses, and where each one
is used. When a feature adapts someone's work, it gets a line here and a
mention in the code or doc that uses it.

Something missing or credited wrongly? Please open an issue.

## Where it started

| Work | Used for |
|---|---|
| **H-Neurons**: Gao et al., [arXiv:2512.01797](https://arxiv.org/abs/2512.01797); code at [thunlp/H-Neurons](https://github.com/thunlp/H-Neurons) | The method NeuronScope grew from: CETT, sparse classifiers over MLP neurons, and suppressing the neurons tied to over-confident answers. What changed from upstream is listed in [RESEARCH_NOTES.md](RESEARCH_NOTES.md). |

## Reading what a model computes

| Work | Used for |
|---|---|
| nostalgebraist (2020), [*interpreting GPT: the logit lens*](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) | `logit_lens.py`; the per-layer logits SLED compares |
| Belrose et al. (2023), *Eliciting Latent Predictions from Transformers with the Tuned Lens*, [arXiv:2303.08112](https://arxiv.org/abs/2303.08112) | Named in `logit_lens.py` and [CONFIDENCE.md](CONFIDENCE.md) as the more accurate lens for early layers |
| Xiao et al. (2024), *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM), ICLR, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) | `when.py` ignores the first position (the attention sink); the engine's "tokens to keep" with context shift |

## Editing and steering

| Work | Used for |
|---|---|
| Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*, NeurIPS, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) | `concept_edit.py`: a difference-of-means direction, removed from the weights (abliteration) |
| Wollschläger et al. (2025), *The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence*, ICML, [arXiv:2502.17420](https://arxiv.org/abs/2502.17420) | The idea of a multi-dimensional concept cone. `concept_edit.py --method pca` builds a simpler cone, from PCA |
| Jim Lai (grimjim) (2025), *projected abliteration* and *norm-preserving biprojected abliteration*, Hugging Face community blog | Biprojection and norm preservation in `concept_edit.py` |
| [Heretic](https://github.com/p-e-w/heretic) (Philipp Emanuel Weidmann) | `concept_edit.py --search` follows its parameterization of ablation strength per layer |
| Turner et al. (2023), *Steering Language Models With Activation Engineering* (ActAdd), [arXiv:2308.10248](https://arxiv.org/abs/2308.10248); Panickssery (Rimsky) et al. (2024), *Steering Llama 2 via Contrastive Activation Addition*, ACL, [arXiv:2312.06681](https://arxiv.org/abs/2312.06681) | Adding a direction to the residual stream: `decode_lab.py`'s `steer` variant, control-vector exports |
| Nonta & Rambhatla (2026), *Refusal-Orthogonal Gated Editing for Safer Localized LLM Adaptation* (SAGE), ICLR 2026 TTU workshop | `sage_edit.py` |
| Louizos, Welling & Kingma (2018), *Learning Sparse Neural Networks through L0 Regularization*, ICLR, [arXiv:1712.01312](https://arxiv.org/abs/1712.01312) | The Hard Concrete gates in `sage_edit.py` |
| Johnson & Lindenstrauss (1984), *Extensions of Lipschitz mappings into a Hilbert space*, Contemporary Mathematics 26 | Why a random low-dimensional projection keeps a spread-out signal: `rp_steer.py` |
| Friedman (1989), *Regularized Discriminant Analysis*, JASA 84(405) | The shrinkage LDA in `rp_steer.py` |

## Decoding and serving

| Work | Used for |
|---|---|
| Zhang et al. (2024), *SLED: Self Logits Evolution Decoding for Improving Factuality in Large Language Models*, NeurIPS, [arXiv:2411.02433](https://arxiv.org/abs/2411.02433) | `decode_lab.py`'s `sled` variant ([DECODING.md](DECODING.md)) |
| Leviathan, Kalman & Matias (2023), *Fast Inference from Transformers via Speculative Decoding*, ICML, [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) | Engine options for speculative decoding with a draft model |
| Li et al. (2025), *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*, [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) | The `draft-eagle3` engine option, where llama.cpp supports it |
| Saxena (2023), [prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) | The n-gram speculative decoding options |
| Ma et al. (2024), *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (BitNet b1.58), [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) | The ternary TQ1_0 / TQ2_0 types `quant_select.py` can try |

## Training, merging and data

| Work | Used for |
|---|---|
| Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | LoRA exports of an edit; `finetune.py` |
| Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) | `finetune.py --method qlora` |
| Rafailov et al. (2023), *Direct Preference Optimization*, [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) | `deficits.py` writes `dpo.jsonl`; `finetune.py --dpo` |
| Gekhman et al. (2024), *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*, EMNLP, [arXiv:2405.05904](https://arxiv.org/abs/2405.05904) | Why `deficits.py` teaches a decline, not an answer, for facts the model does not hold ([RETRAINING.md](RETRAINING.md)) |
| Goddard et al. (2024), *Arcee's MergeKit*, [arXiv:2403.13257](https://arxiv.org/abs/2403.13257); [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) | `mergekit_neuronscope.py` registers a merge method in it |
| Ilharco et al. (2023), *Editing Models with Task Arithmetic*, [arXiv:2212.04089](https://arxiv.org/abs/2212.04089); Yadav et al. (2023), *TIES-Merging*, [arXiv:2306.01708](https://arxiv.org/abs/2306.01708); Yu et al. (2024), *Language Models are Super Mario* (DARE), [arXiv:2311.03099](https://arxiv.org/abs/2311.03099) | The merge methods NeuronScope's neuron-selected merge is compared with ([MERGING.md](MERGING.md)); mergekit implements them |

## Retrieval, search and statistics

| Work | Used for |
|---|---|
| Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond* | `rag.py`, `context_compress.py` |
| Cormack, Clarke & Büttcher (2009), *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR | `rag.py` combines BM25 and embeddings |
| Bergstra et al. (2011), *Algorithms for Hyper-Parameter Optimization* (TPE), NeurIPS; Akiba et al. (2019), *Optuna*, KDD, [arXiv:1907.10902](https://arxiv.org/abs/1907.10902) | `engine_optimizer.py`'s search |
| Wilson (1927), *Probable inference, the law of succession, and statistical inference*, JASA 22(158) | Confidence intervals on right / wrong / declined rates (TestQA, model stats) |
| McNemar (1947), *Note on the sampling error of the difference between correlated proportions or percentages*, Psychometrika 12(2) | Paired comparisons: the retrain gate, model comparisons, merges, expert sweeps |
| Efron (1979), *Bootstrap Methods: Another Look at the Jackknife*, Annals of Statistics 7(1) | Bootstrap intervals in `cv.py` |

## Vision

| Work | Used for |
|---|---|
| Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | `clip_neurons.py` |
| Zhai et al. (2023), *Sigmoid Loss for Language Image Pre-Training* (SigLIP), [arXiv:2303.15343](https://arxiv.org/abs/2303.15343) | `clip_neurons.py` |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | `clip_bench.py` presents a transformers model through OpenCLIP's interface, which CLIP_benchmark expects |
| [LAION-AI/CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) | Zero-shot image benchmarks ([VISION.md](VISION.md)) |

## Datasets and benchmarks

| Work | Used for |
|---|---|
| Joshi et al. (2017), *TriviaQA*, ACL, [arXiv:1705.03551](https://arxiv.org/abs/1705.03551) | The default labels for text H-Neurons |
| Jimenez et al. (2024), *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, ICLR, [arXiv:2310.06770](https://arxiv.org/abs/2310.06770); [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench) | `benchmarks.py swebench`, SWE-bench retraining data |
| White et al. (2025), *LiveBench*, ICLR, [arXiv:2406.19314](https://arxiv.org/abs/2406.19314); [LiveBench/LiveBench](https://github.com/LiveBench/LiveBench) | `benchmarks.py livebench` |
| Deng et al. (2009), *ImageNet*, CVPR; Recht et al. (2019), *Do ImageNet Classifiers Generalize to ImageNet?* ([ImageNetV2](https://github.com/modestyachts/ImageNetV2)), ICML; Wang et al. (2019), *Learning Robust Global Representations by Penalizing Local Predictive Power* ([ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)), NeurIPS; Hendrycks et al. (2021), *Natural Adversarial Examples* (ImageNet-A), CVPR, and *The Many Faces of Robustness* (ImageNet-R), ICCV | Vision benchmarks through CLIP_benchmark |
| Zhai et al. (2019), *A Large-scale Study of Representation Learning with the Visual Task Adaptation Benchmark* (VTAB), [google-research/task_adaptation](https://github.com/google-research/task_adaptation) | Vision benchmarks through CLIP_benchmark |
| Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10/100) | Vision benchmarks through CLIP_benchmark |
| [BenchLM](https://benchlm.ai/benchmarks) and other published leaderboards | `benchmarks.py leaderboard import`. Imported scores keep their benchmark's name (HumanEval, MMLU, GSM8K, SimpleQA…) only to sort them by subject; NeuronScope does not use those datasets itself |

## Software it builds on

| Project | Used for |
|---|---|
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | Serving (`llama-server`), quantizing (`llama-quantize`, `llama-imatrix`), control vectors, GGUF and `gguf-py`, converters. `llama-cett-dump` is NeuronScope's addition built on it ([ENGINE.md](ENGINE.md)) |
| [PyTorch](https://pytorch.org) | Hooks, extraction, training, decoding variants |
| Hugging Face [transformers](https://github.com/huggingface/transformers), [datasets](https://github.com/huggingface/datasets), [huggingface_hub](https://github.com/huggingface/huggingface_hub), [safetensors](https://github.com/huggingface/safetensors), [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [accelerate](https://github.com/huggingface/accelerate) | Loading models and data, LoRA, SFT and DPO training; the Hub's parquet export when `datasets` is missing |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | QLoRA |
| [scikit-learn](https://scikit-learn.org), [NumPy](https://numpy.org), [pandas](https://pandas.pydata.org), [Apache Arrow](https://arrow.apache.org) | Classifiers, arrays, tables, parquet |
| [Optuna](https://optuna.org) | Engine optimization |
| [three.js](https://threejs.org) and [Godot Engine](https://godotengine.org) | The 3D views and the Godot client |
| [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.com), [vLLM](https://github.com/vllm-project/vllm) | Serving stacks NeuronScope connects to |
| [Model Context Protocol](https://modelcontextprotocol.io) and [FastMCP](https://github.com/jlowin/fastmcp); [Cline](https://github.com/cline/cline) | `ns_mcp.py` and the MCP client; the Cline configuration |
| [Tailscale](https://tailscale.com), [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) | Reaching hosts and Studio from elsewhere ([REMOTE.md](REMOTE.md), [SECURITY.md](SECURITY.md)) |
| WebRTC | Peer-to-peer model transfer ([TRANSFER.md](TRANSFER.md)) |
| [Home Assistant](https://www.home-assistant.io) REST API | Power rules on the System page ([SYSTEM.md](SYSTEM.md)) |
| [psutil](https://github.com/giampaolo/psutil) | System monitoring, where installed |

Each project keeps its own licence. Using its models, datasets or outputs
is subject to that licence, and the same goes for the model weights you
load into NeuronScope.
