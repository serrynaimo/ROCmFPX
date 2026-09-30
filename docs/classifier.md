# Jev-style decision classifier in llama-server

llama-server can answer typed questions about a text with calibrated probabilities instead of generated text:
the TypeSafe *System One* API made popular by Jev, served by an open "pointer" decision model such as
[TinyJev](https://github.com/ankit-aglawe/tinyjev) or [Kev](https://github.com/jaredpalmer/kev).

```
POST /v1/classifier            (alias: /v1/systemone)
{"state": "Arr, me parrot ate the treasure map again.",
 "questions": {
   "author": {"type": "choice", "instructions": "Who most likely wrote this?",
              "criteria": {"pirate": "A pirate", "grandma": "A grandmother", "robot": "A robot"}},
   "sea":    {"type": "noul",   "instructions": "Does the text mention the sea, ships or water?"},
   "chaos":  {"type": "score",  "instructions": "How chaotic is this?", "criteria": ["calm", "mild", "wild"]}}}

-> {"model": "tinyjev-4b", "latency_ms": 974,
    "answers": {"author": {"type": "choice", "choice": "pirate", "confidence": 0.6,
                           "probabilities": {"pirate": 0.735, "grandma": 0.118, "robot": 0.147}},
                "sea":    {"type": "noul", "noul": 0.0374},
                "chaos":  {"type": "score", "score": 1.9, "legend": {"0": "calm", "1": "mild", "2": "wild"},
                           "probabilities": {...}, "confidence": 0.95}}}
```

The routes sit behind the normal API key check. There are three ways to serve them:

| Mode | Flag | What answers |
|---|---|---|
| native | `--classifier-head FILE` (+ `--classifier-config FILE`) | this server's own model |
| router | (router mode, no model) | the child named by the request's `"model"`, e.g. a child started with `--classifier-head` |
| forward | `--classifier-upstream URL` | an external service; `--classifier-models` limits which `"model"` values pass (default `tinyjev-4b,auto`) |

## Native mode

The backbone is the model loaded with `-m` (a pointer model's backbone is a plain decoder, e.g. Qwen3), on any backend:
`-ngl 99` for a GPU, `-ngl 0 -dev none` for pure CPU (without `-dev none` llama.cpp may still offload large matmuls
to a GPU). Two more files, loaded like `--mmproj` / `--chat-template-file` are:

- `--classifier-head FILE`: the decision head, safetensors with `q.weight`, `q.bias`, `k.weight`, `k.bias`
  (`[head_dim, n_embd]` / `[head_dim]`, F32/F16/BF16). TinyJev ships it as `head.safetensors`.
- `--classifier-config FILE`: everything that is prompt layout rather than weights. A `tinyjev.json` manifest works
  as-is; every key is optional and defaults to the pointer family:

```json
{
  "name": "tinyjev-4b",
  "family": "pointer",
  "head": { "head_dim": 256, "temperature": 1.0 },
  "max_state": 8192, "max_branch": 8192, "max_options": 255,
  "layout": {
    "tokens": { "state": "<|fim_prefix|>", "question": "<|fim_middle|>", "option": "<|box_start|>",
                "close": "<|box_end|>", "decide": "<|fim_suffix|>" },
    "escape_special": true,
    "option_format": "{name}: {desc}",
    "boolean_labels": { "false": "no", "true": "yes" }
  }
}
```

Per question the prompt is `<state> text <question> instructions (<option> option <close>)* <decide>`; state,
instructions and option descriptions may be strings, numbers, lists or objects (rendered like the Python reference).
With `escape_special`, `<|name|>` in user text becomes `<¦name¦>` so it can never forge a delimiter. The head scores
the final normed hidden state at `<decide>` (query) against each `<close>` (keys):
`z_i = (W_k h_i + b_k) . (W_q h_decide + b_q) / sqrt(head_dim) / temperature`, softmax over the options.
Several questions about one state share the state's computation.

Implementation notes (`tools/server/server-classifier.cpp`): a private `llama_context` on the loaded weights (own
small KV cache, `-c` sized, 4096 by default), embeddings mode OFF with outputs requested only for `<decide>` and the
`<close>` positions, and the post-norm hidden state captured from the graph (`result_norm`) through the eval
callback. Embeddings mode would force every token through the last layer and the vocabulary projection.

Measured with TinyJev-4B BF16 on a Ryzen 9 7900 / RX 7900 XT (OpenDecision, 100 cases): identical answers to the
Python reference on 100/100 cases, probabilities within 0.01; median 56 ms per question on the GPU, 452-463 ms on
12-18 CPU threads (`-dev none`).

Example (CPU, pinned to one CCD):

```
llama-server -m TinyJev-4B-BF16.gguf -dev none -ngl 0 -c 4096 -np 1 -fa on -t 12 -tb 12 -C 0xFFF000 --cpu-strict 1 \
  --classifier-head head.safetensors --classifier-config classifier.json --host 127.0.0.1 --port 8077
```

## Related: `--gpu-keepalive-ms`

Some Windows drivers evict an idle GPU's whole VRAM to system RAM seconds after it has no display work.
`--gpu-keepalive-ms N` clears a 64 KiB buffer on each used GPU every N ms (generic backend API) so the device stays up.
