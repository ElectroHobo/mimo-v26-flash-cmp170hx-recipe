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

### Fewer drafts for larger batches

`SPEC_SCHEDULE=[[1,2,7],[3,32,4]]` (7 drafts for one or two users, 4 from
three users up), same protocol:

| | default (7 drafts always) | schedule |
|---|---:|---:|
| 1 user, structured / code / prose | 308.6 / 286.1 / 178.8 | 328.2 / 289.1 / 177.4 |
| 2 users | 426.5 / 372.0 / 232.5 | 442.3 / 387.8 / 247.7 |
| 4 users | 582.1 / 525.0 / 329.3 | 564.9 / 544.6 / **389.2** |
| 8 users | **696.8 / 659.0** / 413.3 | 599.6 / 583.4 / 420.5 |

One and two users run the same depth, so those rows show the run-to-run
spread (a few percent). Shallower drafting at four users helps prose (+18 %)
and at eight users costs structured output and code (−14 % / −11 %), so the
default stays at 7 drafts throughout; set a schedule if your traffic is
prose-heavy at moderate concurrency.

### Adaptive draft depth (the engine's `adaptive_k`)

The fork can pick the draft count per step from the request count and from
recent acceptance (`SPEC_ADAPTIVE_K`, the engine's `adaptive_k`; what the GLM
recipe calls calibrated adaptive depth). Two settings, same protocol:

| | default (7 drafts) | by load 7,7,6,5,4 + acceptance | acceptance only, 4–7 |
|---|---:|---:|---:|
| 1 user, structured / code / prose | 308.6 / 286.1 / 178.8 | **323.8 / 300.3 / 192.7** | 303.3 / 285.3 / 184.8 |
| 2 users | 426.5 / 372.0 / 232.5 | 377.1 / 394.7 / 262.2 | 438.8 / **413.9 / 273.2** |
| 4 users | 582.1 / 525.0 / 329.3 | **614.3 / 564.4 / 361.4** | 508.1 / 544.2 / **370.2** |
| 8 users | **696.8 / 659.0** / 413.3 | 609.6 / 580.1 / 416.6 | 696.8 / 625.7 / 398.8 |

Prose gains 8–17 % wherever the depth can drop; structured output loses
13–14 % wherever it does. The step costs given to the chooser (`costs`) were
estimates, not measured. The default stays at a fixed 7; set
`SPEC_ADAPTIVE_K={"by_load":[7,7,6,5,4],"accept":{"costs":[1.0,1.04,1.08,1.12]}}`
for traffic that is mostly prose at low concurrency.

### Expert parallelism

`EXPERT_PARALLEL=1` (whole routed experts per card, vLLM `--enable-expert-parallel`)
instead of slicing every expert four ways, same protocol:

| | default | expert parallel |
|---|---:|---:|
| 1 user, structured / code / prose | **308.6 / 286.1 / 178.8** | 297.2 / 260.4 / 170.1 |
| 2 users | **426.5 / 372.0 / 232.5** | 388.2 / 364.9 / 234.6 |
| 4 users | **582.1 / 525.0 / 329.3** | 559.7 / 500.7 / 313.9 |
| 8 users | **696.8 / 659.0 / 413.3** | 658.2 / 631.4 / 392.0 |

Slower by 4–9 % on every row, so it is off by default.

### Peer-to-peer off

`P2P=off` (NCCL through host memory, no custom all-reduce over peer access),
same protocol:

| | default (P2P on) | P2P off |
|---|---:|---:|
| 1 user, structured / code / prose | **308.6 / 286.1 / 178.8** | 181.3 / 164.4 / 96.8 |
| 2 users | **426.5 / 372.0 / 232.5** | 228.4 / 210.5 / 139.5 |
| 4 users | **582.1 / 525.0 / 329.3** | 304.9 / 296.7 / 185.3 |
| 8 users | **696.8 / 659.0 / 413.3** | 426.1 / 388.9 / 254.3 |

About 40 % slower on every row. On this host (one PLX switch, Broadwell CPUs)
peer access is essential for TP4; a report of P2P hurting TP4 on a similar
host (PixelML/club-170hx) does not hold for this model and engine.

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

### Soak

`bench/soak.py`, 20 minutes, 8 concurrent streams of mixed prompts (six kinds,
200–800 tokens, thinking on for half, temperature 0.7) plus one image request
every 30 s: **1,646 requests, 0 failures, 532,160 completion tokens at a steady
443 tok/s; 40 of 40 image requests answered correctly.** No Xid, no errors in
the server log, cards at 41–43 °C afterwards. Log: `bench/results/release-0.1.0/soak-20min-8streams.txt`.

### What was not measured

Video and audio input; sixteen or more users; runs longer than the 20-minute
soak; long-context retrieval. Image input beyond the one test above.

## Notes on the method

- Every number here was measured through the OpenAI-compatible API from the
  same machine, with nothing else talking to the server.
- The step time comes from the server's own `spec_decode_num_drafts_total`
  counter (one draft per scheduler step), not from counting SSE chunks, which
  under-reports when a chunk carries a whole accepted block.
- Raw runs for this release: [`bench/results/release-0.1.0/`](../bench/results/release-0.1.0/),
  measured through the recipe's own `./start.sh`. A run of the same configuration
  started by hand earlier the same day came out within a few percent.
