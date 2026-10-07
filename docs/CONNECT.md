# Connecting Cline, Claude Desktop and other apps

Studio offers other apps two separate things. Set up either or both:

1. **Your local models as the app's AI.** Studio's OpenAI-compatible API at
   `http://127.0.0.1:7870/v1`. Any app with an "OpenAI compatible" provider
   can chat with the models in your Studio, including `auto`, which picks the
   model with the best measured record for each prompt.
2. **NeuronScope as tools (MCP).** An agent can list and route models, check
   a reply for hallucination risk, run TestQA, retraining and benchmark jobs,
   and plan and review projects. These are the same operations as the UI, under
   the same rules.

The quickest way in is Studio's **Connect** page (`/connect`, link in the
header). It shows every setting below for the address you opened Studio
with, with copy buttons.

## Cline

**Models.** In Cline's settings choose *API Provider → OpenAI Compatible*:

| field | value |
|---|---|
| Base URL | `http://127.0.0.1:7870/v1` |
| API Key | Studio's token if it has one; otherwise any text (Cline wants something) |
| Model ID | a model id from Studio's list (printed at startup, shown under Models), or `auto` |

Coding agents work best with models that follow long instructions: a 14B+
coder model if your hardware allows. TestQA's code subject is a quick way to
compare yours (`model_stats` shows the results to an agent).

**Tools.** In Cline open *MCP Servers → Configure* (it opens
`cline_mcp_settings.json`) and add the entry from the Connect page:

```json
{
  "mcpServers": {
    "neuronscope": {
      "type": "streamableHttp",
      "url": "http://127.0.0.1:7870/mcp",
      "disabled": false,
      "autoApprove": ["studio_status", "list_models", "route_prompt", "model_stats", "devices", "doctor",
                      "job_kinds", "job_status", "job_log", "list_projects", "get_project", "list_traces",
                      "requirements", "project_requirements", "review_sources",
                      "review_summary"]
    }
  }
}
```

Or let NeuronScope write it: `python scripts/ns_connect.py cline --write`
(`--editor cursor`, `codium`, `code-insiders` or `windsurf` for other
editors). It adds only the `neuronscope` entry and keeps a backup of the file.

`autoApprove` lists the tools that only read; Cline will ask before
anything that chats, starts a job or changes a project. Studio must be
running for this entry. If your Cline cannot reach Studio over HTTP, use the
stdio entry from the Connect page (`--transport stdio` for the CLI). Cline
then starts `scripts/ns_mcp.py` itself, which talks to Studio for it.

## Claude Desktop

*Settings → Developer → Edit Config*, add the stdio entry from the Connect
page (or `python scripts/ns_connect.py claude-desktop --write`), and restart
Claude Desktop:

```json
{"mcpServers": {"neuronscope": {"command": "/path/to/neuronscope/venv/bin/python",
                                "args": ["/path/to/neuronscope/scripts/ns_mcp.py", "--studio", "http://127.0.0.1:7870"]}}}
```

## Other apps

Open WebUI, Continue, LibreChat or your own scripts: base URL
`http://127.0.0.1:7870/v1`, any API key unless Studio has a token, and a
model id. Any MCP client can use `http://127.0.0.1:7870/mcp` (streamable
HTTP) or start `scripts/ns_mcp.py` (stdio).

## The tools

| tool | kind | what it does |
|---|---|---|
| `studio_status` | reads | What Studio is serving right now: the loaded model, its uptime, whether auto routing has stats to work with. |
| `list_models` | reads | Local GGUF models Studio can serve, with their measured accuracy and hallucination rate when TestQA has graded them. The 'id' is what every other tool's 'model' takes. |
| `load_model` | changes state | Load a model into Studio's chat server (unloads the current one). Chat requests also load models on demand, so this is only needed to preload. |
| `unload_model` | changes state | Unload Studio's chat model to free memory. |
| `chat` | changes state | Ask one of the user's local models. 'model' may be 'auto' to let Studio pick the model with the best measured record for the prompt's subject. |
| `check_reply` | changes state | Score each token of a reply for hallucination risk from the model's own neuron activations (needs a classifier for that model). Returns the flagged text and a link to a 3D view. |
| `route_prompt` | reads | Which subjects a prompt or task involves, and which model 'auto' would pick for it and why. |
| `model_stats` | reads | Per-model, per-subject results from graded runs: accuracy, hallucination and abstention rates with confidence intervals. |
| `devices` | reads | Accelerators on this machine (ROCm, Vulkan, CUDA, Metal, CPU) with free memory, and the worker models running on them. |
| `doctor` | reads | Check the installation: packages, GPUs, llama.cpp build, model, pipeline progress, with a fix for each problem. |
| `job_kinds` | reads | Jobs Studio can run (TestQA, retraining, benchmarks, pipeline stages, builds) and the typed fields each takes. |
| `start_job` | **starts or stops work** | Start a job of one of the kinds from job_kinds, with its fields as 'values'. Runs on the user's machine; long jobs can take hours. |
| `job_status` | reads | One job's state, or the most recent jobs. |
| `job_log` | reads | The end of a job's output. |
| `cancel_job` | **starts or stops work** | Stop a running job. |
| `list_projects` | reads | Projects the director is planning or running. |
| `create_project` | changes state | Turn a description (notes, a task list, a README) into a draft plan of skill-labelled tasks. Nothing runs until a person approves it. |
| `get_project` | reads | A project's plan: tasks, their status, assigned models, results awaiting review, pending proposals. |
| `edit_project` | changes state | Change a plan: changes is a list of {op: add|update|drop|reopen|assign, ...} (see docs/DIRECTOR.md). Logged as the person's edit. |
| `approve_project` | changes state | Approve a draft plan: assigns a model to every task and freezes it. |
| `start_project` | **starts or stops work** | Start (or resume) an approved project: worker models begin on ready tasks. Refused while the machine lacks a required item, unless `force`. |
| `pause_project` | changes state | Pause a running project. |
| `review_task` | changes state | Accept a task's result, or send it back with feedback for the next attempt. |
| `decide_proposal` | changes state | Accept or reject a change the director proposed. |
| `answer_task` | changes state | Answer a blocked task's questions; it goes back in the queue. |
| `list_traces` | reads | Saved per-reply checks and traces, viewable in 3D at <studio>/viz/<id>/. |
| `requirements` | reads | What each feature needs (packages, programs, llama.cpp backends, drivers, devices, memory), what is missing and the fix for this OS. |
| `project_requirements` | reads | Check a project's own requirements (toolchain, libraries, packages, hardware, features, notes) on this machine. |
| `set_project_requirements` | changes state | Replace a project's requirement list. |
| `scan_project_requirements` | changes state | Infer a project's requirements again from its checkout's build files; a person's items are kept. |
| `review_sources` | reads | Models with recorded replies, and the sources, subjects and verdicts to filter the neuron review by. |
| `review_summary` | reads | Neurons that fire more on wrong answers (or with risk, or most often) for a filter, and the error rate per day, week or month. |
| `review_label` | changes state | Record whether a checked reply was right or wrong, so it counts in the review. |

`python scripts/ns_mcp.py --list` prints the same list.

## Tokens and safety

- **On this computer** (Studio on `127.0.0.1`, the default) no token is
  needed.
- **Studio with a token** (any network bind): the owner token gives an agent
  everything the owner can do, including jobs and projects. For an app,
  prefer the **Create a token for an app** button on the Connect page. That
  token is a paired device: it can chat, use `/v1`, check replies and read
  status, but cannot start jobs or change projects, and you can revoke it under
  Link. Cline sends it as an `Authorization: Bearer` header; `ns_connect.py
  --token-file` writes it into the settings file and makes the file
  private (0600).
- Every tool call goes back through Studio's API with the caller's own token,
  so an agent can never do more than the person or device it acts for.
- `/mcp` refuses requests from other web origins and anything that is not
  JSON, so a web page you visit cannot drive it through your browser.
- Jobs run on your machine and can take hours, and TestQA with `--allow-exec`
  runs model-written code. Keep `start_job` and `start_project` out of
  `autoApprove`.

## Without the UI

```bash
python scripts/ns_connect.py show                     # every snippet for this Studio
python scripts/ns_connect.py cline --write             # Cline's MCP settings
python scripts/ns_mcp.py --list                        # the tools
python scripts/ns_mcp.py --studio http://HOST:7870 --token-file studio.token   # stdio server, for any client
```
