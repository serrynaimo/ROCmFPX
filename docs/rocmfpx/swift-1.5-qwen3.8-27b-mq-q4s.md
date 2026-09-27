# Swift-1.5-Qwen3.8-27B ROCMFPX-MQ-Q4S

A 4-bit ROCmFP4 quantization of
[Swift 1.5 Qwen3.8-27B](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)
(UkisAI's Qwen3.8-27B fine-tune) that promotes the 61 most sensitive tensors
to `Q6_0_ROCMFPX`. Built for the 20 GB RX 7900 XT: the promotions cost no decode
speed, prefill speed or VRAM.

Why this model, and why at `reasoning_effort: low`: on our 50-task set it
solved everything at `low`, in the least wall time of any setting we measured
(see [Evidence](#evidence)). Lower-effort thinking text is also more
predictable, so the MTP draft is accepted more often and decode runs about 20%
faster than at `xhigh`.

## Lane

- Public name: `Swift-1.5-Qwen3.8-27B-ROCMFPX-MQ-Q4S`
- Policy: `Q4_0_ROCMFP4` bulk, 61 whole-tensor promotions to `Q6_0_ROCMFPX`
- Tensor policy file: [`qwen3.8-27b-mq-q4s.tensor-types.txt`](qwen3.8-27b-mq-q4s.tensor-types.txt)
- Tensor inventory: 360 F32, 445 `Q4_0_ROCMFP4`, 61 `Q6_0_ROCMFPX` (866 total)
- Size: 16,075,544,864 bytes (14.97 GiB), 4.70 BPW over the quantized tensors
- Loads only in ROCmFPX builds: the two tensor types do not exist in stock
  `llama.cpp`.

Promoted tensors:

| tensors | count | where |
| --- | ---: | --- |
| `output.weight` | 1 | |
| `blk.64.nextn.eh_proj.weight` | 1 | MTP head projection |
| `attn_k`, `attn_v`, `attn_output` | 3 x 17 | the 16 full-attention layers (3, 7, 11, ... 63) and the MTP block 64 |
| `ffn_down` | 8 | boundary layers 0, 1, 2, 3, 61, 62, 63, 64 |

Why this shape: on RDNA3 the K-quants decode about 60% slower than the ROCmFP4
kernel path, so a mixture that promotes to `Q5_K`/`Q6_K` (what the automatic
planner picks) pays for its quality in speed. `Q6_0_ROCMFPX` stays on the fast
path. The promoted set is the planner's own sensitivity ranking, re-expressed
in that type. The attention promotions are nearly free (GQA, 4 KV heads); the
eight `ffn_down` tensors carry most of the added size (0.33 GiB in total).

The policy was derived on a different Qwen3.8-27B fine-tune and transfers
unchanged: the planner's no-pin picks on the Swift 1.5 BF16 are identical
(182 of 182), as are all tensor names and shapes.

## Reproduction

Source: the BF16 GGUF (two shards, 50.9 GiB) and the importance matrix
(`ukisai_Swift-1.5-Qwen3.8-27b-imatrix.gguf`, calibration-v6 corpus) from
[`bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF`](https://huggingface.co/bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF).
Pass the first shard; the second is found automatically. Never requantize from
a low-bit file.

```bash
llama-quantize \
  --imatrix ukisai_Swift-1.5-Qwen3.8-27b-imatrix.gguf \
  --tensor-type-file docs/rocmfpx/qwen3.8-27b-mq-q4s.tensor-types.txt \
  ukisai_Swift-1.5-Qwen3.8-27b-bf16-00001-of-00002.gguf \
  Swift-1.5-Qwen3.8-27B-ROCMFPX-MQ-Q4S.gguf \
  Q4_0_ROCMFP4 24
```

About 4 minutes on a 12-core CPU; the GPU is not used. Add `--dry-run` first:
it must report `quant size = 15320.35 MiB (4.70 BPW)` and convert nothing to
`q5_K`.

The policy file pins **every** quantized tensor, not only the promoted ones.
That is deliberate: left to itself the quantizer escalates a varying set of
`attn_qkv` and `ffn_gate` tensors to `Q5_K`, which silently puts them on the
slow kernel path.

The tensor names are those of the `qwen35` 27B architecture, so the same file
applies to Qwen3.8-27B itself and to other fine-tunes of it. Use an importance
matrix computed for the model you quantize.

For vision, pair the file with the f16 `mmproj` of the base Qwen3.8-27B model.
Swift 1.5 ships its own projector; it is bit-identical to the base one
(334/334 tensors), so either works.

## Evidence

RX 7900 XT 20 GB, ROCm 7.2 HIP, Windows 11.

Perplexity, wikitext-2 test, 580 chunks, `-c 512 -b 512 -fa on`, f16 KV:

| file | PPL | size GiB |
| --- | ---: | ---: |
| `MQ-Q4S` (61 tensors) | 7.1516 ± 0.048 | 14.97 |

Speed and memory, served with `-c 92160`, q4_0 KV, MTP draft `n-max 4`,
`-ub 256`, with the vision `mmproj` loaded:

| measure | value |
| --- | ---: |
| prose decode, MTP, tok/s | 42.7 |
| prefill at 20k / 50k, tok/s | 561 / 453 |
| VRAM after a 50k prefill, GB | 17.95 dedicated + 1.32 shared |

Reasoning effort, 50 runs per setting (4 tool-calling agentic tasks x 3,
10 reasoning tasks x 2, 9 harder python-verified tasks x 2, temperature 1,
exact-match grading), same server, effort set per request:

| effort | output tokens | wall s | correct | decode tok/s, median | MTP draft acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| `low` | 24,760 | 341 | 50/50 | 78.1 | 0.90 |
| `medium` | 25,793 | 359 | 48/50 | 76.3 | 0.88 |
| `xhigh` | 28,892 | 474 | 48/50 | 65.3 | 0.80 |

By task family at `low`: agentic A-D 4,509 tokens, 12/12; reasoning R1-R10
5,508 tokens, 20/20; hard H1-H9 14,743 tokens, 18/18.

Repeating a setting moves the token total by a few percent and the score by
one or two answers, so `low` and `medium` are not separable on this set; the
step to `xhigh` (39% more wall time for no more correct answers) is. Thinking
is never switched off: `low` is the floor we run.

## Notes

- If a variant of this recipe spills VRAM on your card, drop the eight
  `ffn_down` promotions first.
- A 24 GB RX 7900 XTX has room for more promotions or more context. Not
  measured.
- Swift 1.5 Qwen3.8-27B is distributed under the Swift Open License v1.0
  (free for individuals and organisations under US$1M gross annual revenue);
  check its terms before redistributing a quantized file.
