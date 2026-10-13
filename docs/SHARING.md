# Sharing results with a scoreboard

**Off unless you turn it on.** NeuronScope can send TestQA results to a
scoreboard, so models can be compared across machines and setups. It does
not run a scoreboard of its own: `scripts/scoreboard_server.py` is a
reference server that anyone can host. Nothing is sent until you turn
sharing on and give an address.

Studio: Settings → **Sharing results**. Terminal:
`python scripts/score_cards.py preview|submit|check|history`.

## What is sent

**A card, which may be public:**
- **The model.**
  - The GGUF file's SHA-256, size and name (never its path), and the
    quantization.
  - Its Hugging Face repo, only when the file is verified to be in it.
- **The setup (`variant`)**, so results of different setups are never mixed:
  - the hash of the H-Neuron set and the suppression scale;
  - the LoRA's hash and scale;
  - fine-tune, merge, concept-edit and SAGE lineage recorded next to the
    model;
  - the engine (llama.cpp, vLLM, remote) and its version;
  - the engine options that can change answers: context, KV cache types,
    speculative decoding, flash attention, expert count;
  - sampling.

  Options that only change speed (threads, batch, CPU placement) are left
  out. The `variant_id` is a hash of all of this.
- **Hardware, coarse**: OS, CPU model, cores, RAM and GPU memory in whole
  GB, GPU names.
- **The test**: the TestQA bank's fingerprint (a hash of the items asked),
  and the right / wrong / declined counts per subject.

**A verification log, private.** Per item: the id, the verdict and the
model's reply. A scoreboard uses it only to check the card. The reference
server:
- grades every reply again with the same grader;
- rejects the card if the replies do not grade to its numbers;
- keeps the log in a table no public route reads. Only its operator's
  admin token can fetch a log, to look into a disputed result.

**Never sent:** chats, prompts you typed, file paths, user or computer
names, tokens, anything about other models. Every submission is saved on
your computer exactly as sent (`~/.neuronscope/submissions/`) and listed in
Settings.

## Public or private

| the model | a card for it is |
|---|---|
| found by model search: its file's SHA-256 is in the Hugging Face repo it came from | public |
| unlisted, and you listed it by hand (Settings → Sharing results → *list this unlisted model*) | public, marked "unlisted, published by its submitter" |
| unlisted | private: stored and verified, seen only by the scoreboard's operator |

**Automatic submission:**
- *Submit automatically when a TestQA run finishes* covers models found on
  Hugging Face.
- **Advanced → Also submit results for unlisted models** adds the rest:
  privately, unless listed by hand.
- Manual submission (*Preview exactly what is sent*, then *Submit*) works
  for any model once sharing is on.

## Hosting a scoreboard

```bash
python scripts/scoreboard_server.py --db board.sqlite --port 8765 --admin-token-file admin.token --verify-hf
```

Put it behind TLS (`--tls-cert/--tls-key`, or a reverse proxy) on the
internet. It verifies against its own checkout's TestQA bank: cards from a
different bank version are stored as *unverified* and not shown.

Each public row is one model file in one setup, aggregated over its
submissions:
- accuracy and hallucination rate;
- per-subject counts;
- the hardware the runs came from.
