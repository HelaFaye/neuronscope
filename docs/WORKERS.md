# Workers: where project tasks run, and how work is spread

Studio's director splits a project into tasks (Projects page). The **Workers**
page (`/workers`, *Workers* in the header) shows where those tasks run and
lets you steer them.

## Who does what

- **This Studio is the controller.** It plans, assigns, and sends each task
  to a worker, and it keeps the results.
- **A worker is where a model runs and the work happens:**

| worker | what it is | how it starts |
|---|---|---|
| **this computer** | a model in Studio's model folders | Studio starts it on a free device when a task needs it, and stops it when idle |
| **endpoint** | an OpenAI-compatible server you run yourself (llama-server, vLLM, LM Studio, Ollama), here or on another computer | you start it; Studio only sends it tasks |
| **remote host** | a model on a computer running NeuronScope Remote, or on a linked Studio | through that host's agent ([REMOTE.md](REMOTE.md)) |

An endpoint on another computer is reached directly over the network, so
put it on a VPN or a LAN you trust. llama-server can require a key
(`--api-key`), which you give Studio when you add the endpoint. A remote host
through NeuronScope Remote adds pairing, TLS, the host owner's permissions
and its thermal guard.

All the roles and ports in one place: [ROLES.md](ROLES.md).

## What the page shows

- **Tiles**: tasks running and waiting, done and failed since Studio
  started, workers ready, and workers paused or draining.
- **Workers**: state, where each runs, tasks running against its limit,
  results (done, failed, time per task, tokens per second, last error).
  This computer's models appear once they run, have results or settings;
  *show every local model* lists all of them.
- **GPUs**: temperatures (edge, junction, memory), VRAM, load and power, for
  this computer and for each remote host. The sources are nvidia-smi,
  rocm-smi, or Linux's amdgpu sensors. A value that cannot be read shows as
  ?; nothing is estimated.
- **Queue**: every unfinished task, its worker, why it was chosen, and why it
  waits: dependencies, project paused, worker paused, draining, busy, down
  or too hot, or a retry after a failed attempt.
- **Activity**: assignments, starts, finishes, failures, tests and changes.

## Controls

- **Pause**: the worker takes no new tasks, and tasks assigned to it wait.
- **Drain**: it finishes what it has, then pauses.
- **Tasks at once**: a limit per worker. An endpoint defaults to the value
  you gave it; this computer's models follow the project's parallel limit
  unless you set one.
- **Subjects**: the subjects a worker takes. None ticked means any.
- **Test** (endpoints, remote hosts): a one-line chat, timed.
- **Stop** (this computer): stops the model now. A task running on it fails
  and is retried.
- **Move a task**: pick another worker in the queue. It takes effect the
  next time the task starts.

## Routing by subject

For each subject (code, math, writing…), choose up to three workers in
order. A task goes to the first one that is not paused, draining or too hot.
Subjects without a route use the director's choice: measured TestQA
results, then the project's default model, then the largest model that
fits. Routes apply when a plan is approved; approve again to re-route a
plan.

For example, with a heavy GPU elsewhere and a small one here:
- route code, math and logic to the heavy endpoint;
- route writing to the small local model;
- leave the rest to measured results.

To know which worker is better at a subject rather than guessing, measure
both with TestQA (Jobs → *TestQA run*) and let the measured route decide.

## Overheating

NeuronScope Remote can pause a host before its GPUs get too hot (host
console → *GPUs and thermal guard*):
- above the limit you set, the host refuses new chats and traces, and
  unloads what LM Studio and Ollama can unload;
- it resumes once it is a few degrees cooler;
- Studio sees a paused host as *too hot*, and its tasks wait.

GPUs also throttle and protect themselves near their own maximum. The guard
is an earlier, gentler stop that you choose.

## API

| route | what |
|---|---|
| `GET /api/workers/overview` | workers, routes, subjects, queue, telemetry, activity |
| `POST /api/workers/endpoint` | add or update `{name, url, model, token, max_parallel, note}`, or `{remove: name}` |
| `POST /api/workers/control` | `{id, paused, draining, max_parallel, takes, stop}` |
| `POST /api/workers/routes` | `{routes: {subject: [worker ids]}}` |
| `POST /api/workers/test` | `{id}`: a timed one-line chat (endpoints and remote hosts) |

Worker ids: a local model's id; `ep:<name>` for an endpoint;
`<host>:<model>` for a remote one. Endpoints, controls and routes are kept
in `workers.json` next to Studio's `config.json` (owner-readable only, since
it holds endpoint keys).
