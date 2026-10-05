# NeuronScope Quantization Lab

This adds a **capability-preserving quantization layer** without changing NeuronScope's H-Neuron intervention semantics and without adding an SFT/training system.

## What it does

1. Builds a deterministic calibration corpus from one or more weighted text sets.
2. Computes a llama.cpp importance matrix (`llama-imatrix`).
3. Reads NeuronScope H-Neuron profiles and converts their layer-level concentration into protection scores.
4. Selects the highest-scoring decoder layers under a protection budget.
5. Uses `llama-quantize --imatrix` plus `--tensor-type` to quantize normal tensors at the base level and selected FFN down-projection tensors at a higher precision.
6. Builds in RAM (`/dev/shm`) when enough tmpfs space exists, otherwise falls back to disk scratch unless RAM-only mode was requested.
7. Copies the completed GGUF to the destination atomically and writes a `.neuronscope.json` build manifest containing source/output hashes, profiles, calibration, plan and commands.

The first implementation deliberately protects **whole FFN down tensors per selected layer**, because the normal llama.cpp quantizer exposes tensor-level type overrides rather than arbitrary per-neuron bit allocation. Future work can replace this planner with finer-grained block/channel allocation without changing the GUI or build manifest format.

## CLI

```bash
python scripts/quantization_lab.py inspect /path/to/model-F16.gguf

python scripts/quantization_lab.py plan \
  --model /path/to/model-F16.gguf \
  --profile profiles/<fingerprint>/math.json::1.0 \
  --profile profiles/<fingerprint>/coding.json::1.25 \
  --base-quant Q4_K_M \
  --protected-quant Q6_K \
  --budget-percent 10 \
  --json

python scripts/quantization_lab.py build \
  --model /path/to/model-F16.gguf \
  --output /path/to/model-Q4-preserved.gguf \
  --calibration /data/math.txt::1.0 \
  --calibration /data/general.txt::0.5 \
  --profile profiles/<fingerprint>/math.json::1.0 \
  --profile profiles/<fingerprint>/coding.json::1.25 \
  --base-quant Q4_K_M \
  --protected-quant Q6_K \
  --budget-percent 10 \
  --scratch auto \
  --imatrix-bin ~/llama.cpp/build/bin/llama-imatrix \
  --quantize-bin ~/llama.cpp/build/bin/llama-quantize
```

For production use, start from an F16/BF16/F32 source. The harness refuses to treat a filename that looks already quantized as canonical unless `--allow-requantize-source` is explicitly supplied.

## GUI

```bash
python viz/quant_lab.py \
  --models-dir ~/.models \
  --profiles-dir ./profiles \
  --host 127.0.0.1 \
  --port 8796
```

Then open `http://127.0.0.1:8796/`.

The GUI exposes source/target quants, weighted capability profiles, weighted calibration sources, scratch policy, tool paths, protection budget, a protection-plan preview, and a live build log.
