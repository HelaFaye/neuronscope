# NeuronScope Scale Sweep Lab

The Scale Sweep Lab builds static GGUF variants from a NeuronScope H-Neuron profile.
It treats the same scale operation as a symmetric control:

- `0 < scale < 1`: suppression
- `scale == 1`: baseline/no-op
- `scale > 1`: amplification

It does not fine-tune or retrain the model.

## GUI

```bash
python viz/scale_sweep_lab.py --host 127.0.0.1 --port 8797
```

Open `http://127.0.0.1:8797/`.

The GUI visualizes the H-Neuron activation/selection footprint by layer and the requested scale grid. It also controls batch size, scratch policy, evaluation tolerance, and automatic deletion of underperforming candidates.

## Evaluation contract

The optional evaluator command receives these environment variables and formatting fields:

- `NS_MODEL` / `{model}` — candidate GGUF path
- `NS_SCALE` / `{scale}` — scale used
- `NS_BASELINE` / `{baseline}` — baseline score if supplied
- `NS_PROFILE` / `{profile}` — H-Neuron profile path

It must print JSON containing either `score` in `[0,1]` or `correct` + `total`.

Examples:

```json
{"correct": 82, "total": 100, "wrong": 12, "abstained": 6}
```

When `correct`/`total` are present, a 95% Wilson interval is calculated. With automatic pruning enabled, a candidate is deleted when its upper confidence bound is below `baseline - margin_of_error`.

## CLI

```bash
python scripts/scale_sweep_lab.py sweep \
  --model /path/to/base.Q6_K.gguf \
  --profile /path/to/h_neurons.json \
  --output-dir /path/to/sweep \
  --min-scale 0.20 \
  --max-scale 0.40 \
  --step 0.05 \
  --batch-size 2 \
  --scratch auto
```

The state file is JSON and can be reused to inspect or resume a run. Existing same-sized candidates are skipped.

## Safety boundaries

The lab delegates GGUF modification to the repository's existing `scripts/suppress_gguf.py` backend. It does not reimplement GGUF internals. Keep the canonical unmodified source GGUF and build every candidate from it; do not chain requantized candidates or repeatedly scale a previously scaled model.
