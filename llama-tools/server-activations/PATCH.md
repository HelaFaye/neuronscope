# Adding `--activations` to llama-server

Three insertion points, all in `tools/server/server.cpp`. Configuration is read
from the environment rather than the argument parser, so `common/arg.cpp` is
left alone — one fewer file to re-merge when llama.cpp moves.

```bash
cp ns_activations.h /path/to/llama.cpp/tools/server/
```

## 1. Include and globals

Near the other includes at the top of `server.cpp`:

```cpp
#include "ns_activations.h"

static ns::streamer g_ns_stream;
static ns::ctx      g_ns_ctx;
```

## 2. Register the callback before the context is created

Find where the server builds its context — search for `common_init_from_params`
inside `server_context::load_model`. `common_params` already carries `cb_eval`
and `cb_eval_user_data`, so nothing in llama.cpp's context creation changes:

```cpp
// --- ns: activation streaming -------------------------------------------
if (const char * mode = getenv("NS_ACTIVATIONS")) {
    if (params.n_parallel > 1) {
        // A batch with several slots interleaves tokens from different
        // sequences and the node carries no sequence id we can recover, so
        // frames would silently mix two conversations.
        LOG_WRN("%s: NS_ACTIVATIONS ignored, needs --parallel 1\n", __func__);
    } else {
        g_ns_stream.cfg.enabled = true;
        g_ns_stream.cfg.reduction =
            strcmp(mode, "binned") == 0 ? ns::tier::binned :
            strcmp(mode, "raw")    == 0 ? ns::tier::raw    : ns::tier::sparse;
        if (const char * v = getenv("NS_TOPK")) g_ns_stream.cfg.top_k = atoi(v);
        if (const char * v = getenv("NS_BINS")) g_ns_stream.cfg.bins  = atoi(v);

        g_ns_ctx.str      = &g_ns_stream;
        params.cb_eval           = ns::eval_callback;
        params.cb_eval_user_data = &g_ns_ctx;
    }
}
```

Immediately **after** `common_init_from_params` returns, the layer count is
known, so finish the setup there:

```cpp
if (g_ns_stream.cfg.enabled) {
    g_ns_ctx.n_layers = llama_model_n_layer(model);
    if (const char * p = getenv("NS_CLASSIFIER")) {
        const int n_ff = llama_model_n_ff(model);
        if (g_ns_stream.clf.load(p, g_ns_ctx.n_layers * n_ff)) {
            LOG_INF("%s: ns activations scored by %s\n", __func__, p);
        }
    }
    LOG_INF("%s: ns activations on, %d layers\n", __func__, g_ns_ctx.n_layers);
}
```

If `llama_model_n_ff` is absent in your revision, read it off the first
`ffn_down` tensor instead, or pass it via `NS_NFF`.

## 3. REQUIRED: keep the callback alive across graph reuse

**Without this the callback fires once and then silently stops.** llama.cpp
reuses the compute graph between decodes, and the eval callback is only
installed on the rebuild path. Found by hrhdegenetrix in llama.cpp PR #20785;
I had missed it entirely.

In `src/llama-context.cpp`, inside `llama_context::process_ubatch`, move the
call so it runs before the reuse check rather than only after it:

```diff
     const auto gparams = graph_params(res, ubatch, mctx, gtype);

+    // Always set the eval callback, including on graph reuse.
+    ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval,
+                                         cparams.cb_eval_user_data);
+
     if (!graph_reuse_disable && res->can_reuse(gparams)) {
@@
         res->reset();

         ggml_backend_sched_reset(sched.get());
-        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval,
-                                             cparams.cb_eval_user_data);
```

This is also why `llama-tools/cett-dump` works: it evaluates one sequence per
process and never hits the reuse path. A long-running server does.

## 4. The SSE route

Beside the other `svr->Get(...)` registrations:

```cpp
svr->Get("/activations", [](const httplib::Request & req,
                            httplib::Response & res) {
    if (!g_ns_stream.cfg.enabled) {
        res.status = 503;
        res.set_content("{\"error\":\"start with NS_ACTIVATIONS set\"}",
                        "application/json");
        return;
    }
    auto s = std::make_shared<ns::sink>(g_ns_stream.cfg.max_queue);
    g_ns_stream.subscribe(s);
    res.set_chunked_content_provider("text/event-stream",
        [s](size_t, httplib::DataSink & sink) {
            // Long poll: hand over whatever has accumulated, then yield.
            // Returning true keeps the connection; false ends it.
            for (int i = 0; i < 64; ++i) {
                auto f = s->pop();
                if (!f) break;
                const std::string line = "data: " + ns::to_json(*f, "") + "\n\n";
                if (!sink.write(line.data(), line.size())) return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return true;
        },
        [s](bool) { g_ns_stream.unsubscribe(s); });
});
```

## 5. Optional: token text alongside frames

The callback cannot see the token — it only sees graph nodes. If you want the
text in the frame rather than correlating by index on the client, set a global
in the completion loop where the sampled token is detokenised, and pass it to
`ns::to_json`. This is the only change that touches generation, which is why it
is optional.

## Prior art

llama.cpp PR #20785 (hrhdegenetrix, closed as out of scope for the server) does
the same thing at a different granularity: it captures `l_out`, the residual
stream after each layer, which is what sparse autoencoders want. This captures
`ffn_down` inputs, which is per-neuron and what H-Neurons needs. They are
complementary, not competing, and the branch is worth reading --
`hrhdegenetrix:activation-capture`, with a companion repo at
`llama-sae-feature-interpretability`.

Their measured overhead on an RTX 3090 with Qwen3-8B: ~140.7 tok/s disabled,
~131.8 enabled, about 6%. That matches the design assumption here that an
unsubscribed callback costs nothing.

## Running it

```bash
NS_ACTIVATIONS=sparse NS_TOPK=48 \
NS_CLASSIFIER=models/classifier.bin \
  ./build/bin/llama-server -m model.gguf --parallel 1 --port 8080

python viz/stream.py --source http://127.0.0.1:8080 --token-file viewer.token --host 0.0.0.0 --allow-plaintext
```

## What to expect

**Cost when idle is nothing.** The callback returns early if no one is
subscribed, before any device copy. Leaving the flag on is free.

**Cost when watching** is one `ggml_backend_tensor_get` per layer per token
plus a pass over `n_layers * n_ff` floats. On a 36x14336 model that is ~2 MB of
device-to-host copy per token. At 4 tok/s it is not the bottleneck; at 60 tok/s
on a fast card it might be, in which case use `NS_ACTIVATIONS=sparse` and
accept the top-k selection cost instead.

**Prefill is skipped.** Frames are only emitted when the batch is a single
token, which is generation. A multi-token batch is prompt processing.

**Live scores are not replay scores.** The server emits `|a| / ||layer output||`
without the `||W[:, j]||` factor, because holding dequantised down_proj weights
in RAM is a large cost for a live view. `scored` in each frame says whether a
classifier was loaded; when it was not, `s` is peak activation, which shows
activity but flags nothing.
