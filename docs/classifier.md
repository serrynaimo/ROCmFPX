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

The routes sit behind the normal API key check. There are two ways to serve them:

| Mode | Flag | What answers |
|---|---|---|
| native | `--classifier-head FILE` (+ `--classifier-config FILE`) | this server's own model |
| router | (router mode, no model) | the child named by the request's `"model"`, e.g. a child started with `--classifier-head` |

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

`--classifier-cache DIR` adds a persistent answer cache. The model is deterministic, so an answer depends only on
the model, head and config files and on the tokens of the state and the question. Each answer is appended to
`DIR/<name>-<identity>.tsv` (one line per answer, loaded into memory at start) and a repeated question about the same
state is answered without a decode. Caching is per question, so a request may mix cached and fresh answers; the
response carries `"cached": true` when nothing had to be decoded. Changing the model, head or config starts a new
file; a file over 64 MiB loses its older half at the next start.

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

### Letter family (no head)

Many newer decision models (Wald, StartLux-Decision, decider, TinyJev v2) have no head: the options are lettered in
a text prompt and the answer is the model's own next-token logits of those letters. `"family": "letter"` in the
config serves them; `--classifier-config` alone enables it, there is no `--classifier-head`.

```
llama-server -m Wald-4B-v1.2-Q8_0.gguf --classifier-config classifier.json -dev none -ngl 0 -c 8192
```

```json
{"name": "wald-4b", "family": "letter",
 "letter": {
   "prompt": "State:\n{state}\n\nQuestion: {instructions}{options}\nAnswer: (",
   "prompt_no_state": "Question: {instructions}{options}\nAnswer: (",
   "option_line": "\n({letter}) {text}",
   "variants": ["{letter}", " {letter}"],
   "bare_keys": "positional",
   "temperature": {"choice|2": 1.19, "choice|3-4": 1.74, "noul|2": 1.32, "default": 1.37}}}
```

| Key | Meaning |
|---|---|
| `prompt` | the whole prompt of one question; `{state}` may appear twice. Special tokens written here (a chat template) are parsed, user text is escaped |
| `prompt_no_state`, `empty_state` | for a blank state: another prompt, or the text that stands in for the state |
| `option_line` | one option, `{letter}` is A, B, .. |
| `variants` | token spellings of a letter whose probabilities are added |
| `state_format` | `text` (labelled lines, as the pointer family) or `json` (compact JSON; `json_index_min` adds `_index` to the items of long arrays) |
| `score_format`, `true_first`, `noul_instructions`, `same_as_name`, `strip` | how levels and yes/no are written |
| `bare_keys` | choice keys that only number the options: `positional` drops a leading `a: `, `bare` shows the description alone |
| `temperature` | a number, or a map looked up as `<type>\|<bucket>` (option count 2, 3-4, 5-8, 9+), `<type>`, `default` |

`"style": "ollama"` (TinyJev v2) replaces the option lines by the user message of Ollama's decision API: compact
JSON with the state and the schema of every question, then `Requested field: "<name>"`. `prompt` wraps it as
`{user}`, normally in the model's chat template with the Modelfile's system text.

Each question is one pass with a single output row. The questions of a request share their common token prefix
(the state; with `"style": "ollama"` the whole schema as well): it is decoded once, the sequence state is saved, and
each question continues from a restored copy. More than 26 options are read in chunks whose winners meet in a final
read.

### Recommended model

For a classifier that runs next to a chat model, on the CPU, we recommend the
[StartLux-Decision](https://huggingface.co/startlux-models) models, and on AMD Ryzen CPUs their **BF16** files rather
than Q8_0: on a AMD Ryzen 9 7900 12-Core Processor (12 threads) the BF16 file of StartLux-Decision-2B answers in about half the time of its
Q8_0 file (a 21-option choice 1.05 s against 2.0 s, a yes/no question 0.39 s against 0.75 s), at 3.8 GB against 2.0 GB.

Measured here on CPU, BF16, 12 threads: OpenDecision 500 and our own 115 requests (theme choice for a document, table
and chat yes/no questions), with the latency weighted by how often we ask each kind.

| Model | Family | OpenDecision 500 | Own requests | Latency |
|---|---|---|---|---|
| TinyJev-4B v1 | pointer | 476 | 105 | 0.86 - 0.89 s |
| **StartLux-Decision-2B** | letter | 486 | 110 | 0.57 - 0.72 s |
| Wald-4B v1.2 (plain prompt) | letter | 488 | 111 | 1.1 - 1.25 s |
| Wald-4B v1.2 (its repeated-state prompt) | letter | 490 | 110 | 1.6 - 1.75 s |
| TinyJev-4B v2 | letter, `ollama` | 490 | 110 | 1.9 s |

StartLux-Decision-2B is the only one that is both faster and more accurate than the 4B pointer model; the larger
ones gain a few OpenDecision cases for 1.5 to 3 times the latency. Its config:

```json
{"name": "startlux-decision-2b", "family": "letter",
 "letter": {
   "prompt": "<|im_start|>system\nApply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only.<|im_end|>\n<|im_start|>user\nEvidence:\n{state}\n\nQuestion: {instructions}\nOptions:{options}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
   "empty_state": "(none)", "option_line": "\n{letter}) {text}", "variants": ["{letter}"],
   "state_format": "json", "json_index_min": 8, "score_format": "{name}: {desc}", "true_first": true,
   "noul_instructions": "Which answer fits the evidence?", "bare_keys": "bare", "same_as_name": true, "strip": true,
   "temperature": {"noul": 2.2907, "choice": 1.5319, "score": 1.6163}}}
```

Two things to know: its probabilities are cautious (the shipped temperatures flatten them, so thresholds near 0 or 1
are reached less often than with TinyJev v1), and the weights are CC BY-NC 4.0, so commercial use needs a licence
from StartLux Labs. Wald-4B is the Apache-2.0 alternative.

## Related: `--gpu-keepalive-ms`

Some Windows drivers evict an idle GPU's whole VRAM to system RAM seconds after it has no display work.
`--gpu-keepalive-ms N` clears a 64 KiB buffer on each used GPU every N ms (generic backend API) so the device stays up.

## Router mode

One public port for chat and classifier models: the router starts each model as a child from a `--models-preset`
INI and routes by the request's `"model"` (name or alias). Classifier routes whose `"model"` is missing or unknown go
to the first model with `classifier-head` in its preset; `--models-default NAME` sends other requests without a
known `"model"` (and GET routes such as `/slots` without `?model=`) to NAME, like a single-model server.

```ini
version = 1

[qwen/qwen3.8-27b]
model = E:\Models\...\Swift.gguf
alias = swift
load-on-startup = true
gpu-keepalive-ms = 2000

[tinyjev-4b]
model = E:\Models\AnkitAI\TinyJev-4B-GGUF\TinyJev-4B-BF16.gguf
alias = auto
load-on-startup = true
dev = none
ngl = 0
classifier-head = E:\Models\AnkitAI\TinyJev-4B-GGUF\head.safetensors
classifier-config = E:\Models\AnkitAI\TinyJev-4B-GGUF\classifier.json
```

```
llama-server --models-preset models.ini --models-max 0 --models-default qwen/qwen3.8-27b --port 1234
```

The API key may be shared with outside callers (a gateway, a web page); `--models-api-loopback-only` keeps model
management (load, unload, download via `POST /models`, delete, `GET /models?reload`) to loopback clients.
