# Results

## Release 0.1.0

Measured 2026-10-18 on 4× CMP 170HX (74 SMs, 65,536 MiB and 180 W per card,
PCIe Gen2 x16 on one PLX switch, peer-to-peer on, driver 610.57.04), host
2× Xeon E5-2697A v4 with 204 GiB of RAM, engine `18305948a`, recipe defaults:
TP4, NVFP4 weights, bf16 KV with a 10 GiB budget per card, DFlash with 7 drafts,
`FULL_AND_PIECEWISE` CUDA graphs.

### Quality

`HumanEval` (all 164 problems, the model writes the function, run against the
dataset's tests in a no-network sandbox) and `GSM8K` (the first 200 test
problems, exact numeric match). Thinking on, `reasoning_effort` medium,
temperature 0.6 / top_p 0.95 / top_k 20, 16,000-token cap, 8 concurrent, one
sampled run per problem.

| Suite | Passed | Empty answers | Median tokens per answer | Median wall per request |
|---|---:|---:|---:|---:|
| HumanEval | 161/164 (98.2 %) | 1 | 233 | 2.5 s |
| GSM8K (200) | 195/200 (97.5 %) | 1 | 182 | 2.2 s |

The two empty answers finished without reaching the token cap; not chased.
The whole 364-request run took six minutes.

### Throughput

The protocol Morrowmake's recipes use (from MiaAI-Lab's DGX Spark kits):
temperature 0, thinking off, every reply pinned to exactly 400 tokens
(`min_tokens` + `ignore_eos`), a fresh salt in each prompt so the prefix cache
is out of it, and the decode rate measured after the first content token. One
user is the median of five streaming decode rates on one boot; two, four and
eight users are the median of three runs of the aggregate rate: actual
completion tokens divided by the batch's wall time, prefill and client
overhead included. Cold prefill is the median of nine rates over three
real-text prompts of 24K, 34K and 38K tokens, with no prefix-cache hits.
Harness: [`bench/cmp_recipe_bench.py`](../bench/cmp_recipe_bench.py).

| | TP4, peer-to-peer on |
|---|---:|
| Streaming decode, 1 user, structured / code / prose | **308.6 / 286.1 / 178.8 tok/s** |
| Decode, 2 users, aggregate, structured / code / prose | **426.5 / 372.0 / 232.5 tok/s** |
| Decode, 4 users, aggregate, structured / code / prose | **582.1 / 525.0 / 329.3 tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **696.8 / 659.0 / 413.3 tok/s** |
| Cold prefill | **2,473 tok/s** |
| KV pool at 1,048,576 context | 1,643,784 tokens (1.57 full-length requests) |

The decode step is 14.8 ms for one user whatever the prompt (15 ms at two,
28 ms at four, 48 ms at eight). What moves between the columns is the drafter:
of 7 drafts per step, 99 % are accepted on a counting task, about half on code
and a fifth on prose. On an easier structured task (counting) one user reaches
530 tok/s; the structured prompt in the table is a 40-row markdown table.

### The MTP head instead of DFlash

`SPEC_MODE=mtp` (2 drafts), same protocol, same boot:

| | TP4, peer-to-peer on |
|---|---:|
| Streaming decode, 1 user, structured / code / prose | 128.9 / 119.6 / 114.5 tok/s |
| Decode, 2 users, aggregate | 193.7 / 180.6 / 173.7 tok/s |
| Decode, 4 users, aggregate | 292.5 / 267.5 / 252.4 tok/s |
| Decode, 8 users, aggregate | 417.2 / 379.8 / 362.3 tok/s |
| KV pool at 1,048,576 context | 1,823,860 tokens (1.74 full-length requests) |

The boot check accepted 248 of 1,040 MTP drafts (24 %), against 671 of 693 for
DFlash on the same two prompts. DFlash is the better drafter for every column.

### Context and sanity

| Check | Result |
|---|---|
| `(45*12)+(78/6)` | 553 |
| Context | 1,048,576-token requests accepted (the KV pool holds 1.57 of them) |
| Image | a 448×448 image, red left half and blue right half: "The left half of the image is red, and the right half is blue" (227 prompt tokens), with `patches/mimo_v2_omni.py` mounted. Without it: "white and red" / "entirely red" |

### What was not measured

Video and audio input; sixteen or more users; sustained load; long-context
retrieval. Image input beyond the one test above.

## Notes on the method

- Every number here was measured through the OpenAI-compatible API from the
  same machine, with nothing else talking to the server.
- The step time comes from the server's own `spec_decode_num_drafts_total`
  counter (one draft per scheduler step), not from counting SSE chunks, which
  under-reports when a chunk carries a whole accepted block.
- Raw runs for this release: [`bench/results/release-0.1.0/`](../bench/results/release-0.1.0/),
  measured through the recipe's own `./start.sh`. A run of the same configuration
  started by hand earlier the same day came out within a few percent.
