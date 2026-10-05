# NeuronScope Adaptive Remote Tuning

This subsystem separates tuning control from model execution.

The **controller** may run on your workstation. The **worker** runs on the machine that has the LLM/GGUF and evaluator. The worker keeps the source model, H-Neuron profile, generated candidates, and evaluation artifacts local. Only small JSON job/control messages cross the control connection.

Use the Model Transfer service separately for initial source/profile/dataset movement and for retrieving only the final candidates you want to keep.

## Adaptive search behavior

The search is intentionally coarse-to-fine:

```text
wide range
   ↓
batch of N scales
   ↓
evaluate
   ↓
find best result
   ↓
contract interval toward best
   ↓
fresh interior batch
   ↓
repeat
```

A run stops when one of these conditions is met:

- the current interval is at or below the selected autotuning resolution;
- the best result has a 95% Wilson score interval whose width is within the selected margin of error;
- the configured maximum number of batches is reached;
- no new scale can be generated.

The state JSON persists after every job and after every narrowing decision, so reconnecting to the same state file resumes from the saved `next_interval` rather than restarting the broad search.

## Automatic deletion

Enable `--auto-delete` only when a baseline score is supplied. After a batch completes, candidates whose upper confidence bound is below `baseline_score - margin_of_error` are eligible for deletion, except for the best candidate from that batch.

Deletion is constrained to the worker's configured output root.

## Remote worker

Start this on the machine that actually has the LLM/GGUF:

```bash
python scripts/tuning_worker.py \\
  --host 0.0.0.0 \\
  --port 8799 \\
  --root /models/neuronscope-tuning \\
  --source /models/Huihui-Ornith-1.5-9B-abliterated.Q6_K.gguf \\
  --profile /models/h_neurons.json \\
  --suppressor /opt/neuronscope/scripts/suppress_gguf.py \\
  --evaluator 'python /opt/evaluators/ornith_eval.py --model "{model}" --scale "{scale}"' \\
  --workers 1 \\
  --token 'REPLACE-WITH-A-LONG-RANDOM-TOKEN'
```

For a machine exposed outside a trusted LAN, place the worker behind HTTPS/WSS or a VPN such as WireGuard/Tailscale. The bearer token is authentication, not transport encryption.

## Controller CLI

```bash
python scripts/adaptive_tuner.py tune \\
  --state /path/to/ornith-adaptive-state.json \\
  --worker http://REMOTE-HOST:8799 \\
  --min 0.20 \\
  --max 0.80 \\
  --initial-step 0.20 \\
  --resolution 0.01 \\
  --batch-size 3 \\
  --margin-of-error 0.02 \\
  --baseline-score 0.91 \\
  --token 'REPLACE-WITH-THE-SAME-TOKEN' \\
  --auto-delete
```

## Controller GUI

```bash
python viz/adaptive_tuning_lab.py \\
  --host 127.0.0.1 \\
  --port 8800
```

Open:

```text
http://127.0.0.1:8800/
```

Set:

- Worker URL
- Worker token
- minimum/maximum scale
- initial step
- autotuning resolution
- models per batch
- margin of error
- optional baseline score
- optional automatic deletion

The chart displays the observed score by scale, while the JSON panel exposes the exact persisted state and selected interval.

## Recommended workflow for a remote LLM machine

1. Use **Model Transfer** to move the canonical source GGUF, matching H-Neuron profile, and calibration/evaluator data to the remote machine.
2. Start `tuning_worker.py` on that machine.
3. Run the controller GUI locally.
4. Let the controller narrow the search remotely; intermediate GGUFs stay on the remote disk.
5. Retrieve only retained/final GGUFs with Model Transfer.
6. Keep the controller state JSON as the experiment manifest.

This avoids moving multi-GB candidates after every scale test.
