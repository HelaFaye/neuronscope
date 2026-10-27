# Add-ons

**Studio → Add-ons** (`/addons`) manages NeuronScope's optional parts:

- **Built into Studio.** Parts not everyone needs. **Remove** one and its pages, API routes, jobs and sidebar entries go; **Reinstall** brings them back. The code stays in the checkout, so nothing is downloaded, and your data (dictionaries, graphs, projects…) is kept either way.
- **Packages.** Plugins ([PLUGINS.md](PLUGINS.md)) installed from a `.zip` file or a repository, updated, turned on or off, and uninstalled. A package runs code on this computer, so it stays off until you turn it on, and goes off again whenever its files change (a SHA-256 over every file is checked).

## Built-in add-ons

| add-on | what removing it takes away |
|---|---|
| Features and emotions | `/features`, the Interpret jobs, the decoding lab job |
| Graphs | `/graphs` and its API |
| Neuron review | `/review`, the Review jobs |
| Projects and workers | `/projects`, `/workers`, the director |
| Remote hosts | `/remote`, controlling other computers |
| Device pairing | `/link`, pairing links, linked Studios |
| Labs and 3D views | `/lab`, the view jobs |
| Documents (RAG) | document collections |
| System monitor and automation | `/system`, power and activity rules |
| Retraining | the Retrain and Merge jobs |
| Benchmarks | the Benchmarks jobs |
| H-Neuron pipeline | the H-Neurons, Vision, Analysis and Tuning jobs |
| Sharing results | publishing to the scoreboard |

Chat, models, jobs, settings, setup, permissions and help are the core and stay.

## Packages

- **From a file.** Drop a `.zip` on the Add-ons page. `plugin.json` sits at the top of the zip or in one top folder. Absolute paths, `..`, links and device files are refused; at most 200 MB and 5000 files.
- **From a repository.** Add the address of an `index.json` (https, or a path on this computer). Every package it lists can be installed or updated, and each download must match the `sha256` the index gives.
- **Uninstall** deletes an installed package's folder. A package that came with NeuronScope (in the checkout's `plugins/`) can only be turned off.

Make a package and a repository:

```bash
python viz/addons.py pack plugins/comfyui -o dist/       # dist/comfyui-1.0.zip
python viz/addons.py index dist/                         # dist/index.json, with each package's sha256
```

Serve `dist/` over https (for example a GitHub release or Pages site) and add
`https://…/index.json` as a repository.

A package can add jobs, reply checks, a page, MCP tools and graph nodes
([PLUGINS.md](PLUGINS.md)). With `"tab": {"label": "…"}` beside its `page`, its
page also gets a sidebar tab (under *Add-ons*) while it is on.

## ComfyUI

`plugins/comfyui` comes with NeuronScope. Turn it on, and a **ComfyUI** tab shows ComfyUI inside Studio. Its jobs (Jobs → Plugins: comfyui):

- **install** clones [ComfyUI](https://github.com/comfy-org/ComfyUI) into its own folder (default `~/.neuronscope/comfyui`) and gives it its own Python environment (`.venv`). It installs the PyTorch build for this machine: CUDA, ROCm, XPU, Apple (MPS), DirectML or CPU, picked from the GPUs found. Optionally it also adds:
  - [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager);
  - the LLM node packs [comfyui_LLM_party](https://github.com/heshengtao/comfyui_LLM_party), [ComfyUI-LLMs-Toolkit](https://github.com/ComfyUI-Kelin/ComfyUI-LLMs-Toolkit) and [ComfyUI-LLMs](https://github.com/leoleelxh/ComfyUI-LLMs);
  - NeuronScope's own nodes (below).
  
  Run it again to repair an install.
- **run** starts ComfyUI on a port (8188) and keeps it running until you cancel the job. It listens on this computer only unless you change that; ComfyUI has no login of its own.
- **update** runs `git pull` for ComfyUI and its custom nodes, then updates their requirements.
- **add a custom node** clones any GitHub, GitLab or Codeberg repository into `custom_nodes`.

**NeuronScope's nodes in ComfyUI** (category *NeuronScope*):

- **chat** asks your models through Studio. `auto` picks a model by subject. Reasoning comes out separately.
- **hallucination check** gives each token's risk from the model's neurons, the flagged text, and a link to the 3D view.
- **run a graph** runs a Studio graph ([GRAPHS.md](GRAPHS.md)), such as `template:screen-and-verify`, and returns its output.

They reach Studio at `http://127.0.0.1:7870`, set per node, using `NS_STUDIO_TOKEN` or `~/.neuronscope/studio.token` if Studio has a token. ComfyUI is GPL-3.0 and is downloaded from its own repository; none of it is shipped here.

## What else should be an add-on

These parts of NeuronScope could become packages next. Each is used by some
people and not others, and each has its own heavy dependencies or its own
attack surface:

| candidate | why |
|---|---|
| **The decoding lab** (`decode_lab.py`, `feature_lab.py`) | pulls in PyTorch and transformers; only needed for features, emotions, SLED and steering |
| **Retraining** (finetune, SFT/DPO, SAGE, merges) | the largest dependencies (PEFT, TRL, bitsandbytes) and GPU time; most users only evaluate |
| **SWE-bench and LiveBench** | Docker and big datasets |
| **Vision** (CLIP neurons, mmproj suppression) | only for multimodal models |
| **Remote hosts and the transfer lab** | network-facing; better absent when unused |
| **Device pairing** | the same; a single-user desktop does not need it |
| **The Godot client** | a separate program and download; could be a package that installs Godot and the client |
| **Themes and widgets beyond the defaults** | the jewel styles, extra 3D themes: easy to share as small packages |
| **Connectors** (Home Assistant automation, scoreboard sharing) | each talks to one outside service |
| **Graph node packs** | domain screens (legal, medical, code security) as plugin nodes |
| **Language packs** | translations of the UI and docs |

The built-in add-ons above already let you remove the first several today; turning
them into separately downloadable packages would also keep their dependencies off
machines that never use them.
