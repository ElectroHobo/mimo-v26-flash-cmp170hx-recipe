# Results

## Release 0.2.0

Measured 2026-10-18 on the same machine, engine, image and checkpoint as 0.1.0
(below), recipe defaults: TP4, NVFP4 weights, bf16 KV with a 10 GiB budget per
card, DFlash with 7 drafts for one request, 5 for two and 3 from three up,
a 2,048-token prefill chunk read in 384-token slices while anyone is decoding,
`FULL_AND_PIECEWISE` CUDA graphs. Same protocol as 0.1.0 (described under
"Throughput" below); the quality suites were not re-run, since the draft depth
changes how many tokens are proposed per step, not which tokens are produced.

| | TP4, peer-to-peer on |
|---|---:|
| Streaming decode, 1 user, structured / code / prose | **322.0 / 274.9 / 170.1 tok/s** |
| Decode, 2 users, aggregate, structured / code / prose | **427.0 / 397.7 / 255.1 tok/s** |
| Decode, 4 users, aggregate, structured / code / prose | **572.7 / 538.3 / 412.1 tok/s** |
| Decode, 8 users, aggregate, structured / code / prose | **753.8 / 738.0 / 555.5 tok/s** |
| Cold prefill | **2,722 tok/s** (23,887 / 34,186 / 38,341-token prompts; 2,471 at a 1,024 chunk) |
| KV pool at 1,048,576 context | 1,615,953 tokens (1.54 full-length requests; 1,643,784 at a 1,024 chunk) |

Against 0.1.0's fixed 7 drafts, re-measured the same day (685.6 / 645.6 /
398.4 at eight users), that is +10 % / +14 % / +39 % at eight users and the
same within spread at one (one-user prose ran 170–182 across today's boots
at the same depth). A run of the same defaults earlier in the day, before
the prefill chunk and fair prefill changed, gave 322.2 / 284.2 / 182.3,
408.1 / 400.7 / 254.3, 513.6 / 535.6 / 396.2 and 756.6 / 754.7 / 546.5:
neither setting touches decode. That earlier run's file was overwritten by
the final one (same tag) and is not kept; its figures survive only here and
in the all-reduce table below. Raw runs: [`bench/results/release-0.2.0/`](../bench/results/release-0.2.0/).

### Draft depth by number of requests (the new default)

Why the step grows with users: every request puts drafts + 1 tokens through
the model per step, and in a mixture of experts the number of distinct experts
whose weights are read per step grows with tokens in flight, so the headroom
that makes a second user nearly free in a dense model is already spent on the
drafts. Measured from the engine's step counter with 7 drafts: 15.3 ms per
step for one user, 20.8 for two, 27–32 for four, while accepted tokens per
request per step stay the same (3.6 structured, 4.4 code, 2.6 prose). Fewer
drafts under load make the step cheaper at little cost in accepted tokens:
on prose the drafter's per-position acceptance is 0.73, 0.48, 0.28, 0.18,
0.09, 0.06, 0.03, so positions five to seven are nearly dead weight.

Four schedules on 2026-10-18, same protocol, each on its own boot, the
default re-measured the same day (aggregate tok/s, structured / code / prose):

| users | fixed 7 (0.1.0 default) | 7 / 5 / 3 by requests (**0.2.0 default**) | 7 / 5 / 4 | 7, then 3 from two |
|---|---:|---:|---:|---:|
| 1 | 306.4 / 281.0 / 180.0 | 343.1 / 290.1 / 173.2 | 308.5 / 274.8 / 179.6 | 332.3 / 286.2 / 184.7 |
| 2 | 414.4 / 399.1 / 246.1 | 426.1 / 394.7 / 266.3 | 429.5 / 395.4 / 262.0 | 400.0 / 386.6 / 286.8 |
| 4 | 570.5 / 493.1 / 323.9 | 577.5 / 531.6 / 401.9 | 570.1 / 543.4 / 388.8 | 570.9 / 541.1 / 405.4 |
| 8 | 685.6 / 645.6 / 398.4 | **760.4 / 725.0 / 554.1** | 624.3 / 586.2 / 419.4 | 754.2 / 723.8 / 554.6 |

Three drafts under load wins every column from four users up (+11 % / +12 %
/ +39 % at eight) and is repeatable: the release run, on another boot, gave
754 / 738 / 556, and "7, then 3 from two" (the same depth at eight) 754 / 724 /
555. One
user is the same depth in every column, so that row shows the run-to-run
spread. Four drafts under load is worse than both seven and three, on two
boots; a guess is kernel padding (five tokens verified per request padded to
eight), not proved. At two users only prose moves, whichever depth; the
two-user step is the expert fan-out above. `SPEC_SCHEDULE=none` restores the
fixed depth.

### KV offload to host RAM and disk

Several agents starting from the same long context will, by the time a later
one arrives, find other traffic has pushed that context's KV off the cards.
Without an offload tier the engine re-reads the prompt; with one
(`KV_OFFLOAD_GB`, and `KV_OFFLOAD_DIR` for a disk tier behind it) it copies the
blocks back. `bench/kv_offload_bench.py`: a 100K-token prefix with three codes
hidden in it, read cold, then warm (GPU prefix hit), then a flood of 18 ×
100K tokens of unique filler (more than the 1.64 M-token pool, so the prefix
is evicted; 22 preemptions in every run), then the prefix again. Time to first
token, 2026-10-18, one run each on its own boot, at the 1,024-token prefill
chunk (the 0.1.0 default; the pool figures below are that chunk's):

| | offload off | 64 GiB RAM tier | 16 GiB RAM + disk tier on a SATA SSD |
|---|---:|---:|---:|
| cold | 49.8 s | 49.7 s | 49.7 s |
| warm (GPU hit) | 0.51 s | 0.36 s | 0.38 s |
| **after eviction** | **49.8 s** (recompute) | **0.62 s** (from RAM) | **4.86 s** (from disk) |
| codes correct | 3/3 | 3/3 | 3/3 |
| tier chunk hits | — | 798 of 817 queries | 798 (2.6 GB read back) |

The whole flood was stored (44.7 GB for about 1.9 M tokens, 24 KB per token),
and the KV pool on the cards is unchanged at 1,643,784 tokens. The drafter
needs `disable_eagle_block_drop` in the speculative config for the tier to
hit (vLLM #52735; the recipe sets it with offload on); the engine warns that
it is experimental and may lower acceptance, and the boot check and the codes
above were unaffected. The disk tier is write-through, wrote 49.6 GB at about
150 MB/s during the flood, and never deletes a file, so size and prune the
directory yourself (47 GB after one run here). With the RAM tier cut to 16 GiB
the flood pushed the prefix out of RAM and the reload came from disk. Both
tiers are off by default: the RAM tier wants host memory (64 GiB here, in
`/dev/shm`), and whether your traffic returns to evicted prefixes is for you
to know. Not measured: reloads during traffic rather than after a flood, long
answers after a reload, several users.

### Fair prefill, and the prefill chunk

A long prompt that arrives while someone is mid-answer shares every scheduler
step with that answer, so the decoder waits for each prefill chunk.
`bench/prefill_fairness.py`: one request streams a 1,500-token answer; three
seconds in, a 30K-token prompt arrives. Decoder rate before, during and after
the prompt is read; the longest pause between two streamed chunks; and how
long the prompt took to read. 2026-10-18, two runs each (one for the
2,048-chunk column with 384-token slices), the first run shown:

| | 1,024 chunk, no fair prefill | `FAIR_CHUNK=384` | `FAIR_CHUNK=384`, `MAX_BATCHED=2048` | `FAIR_CHUNK=128`, `MAX_BATCHED=2048` |
|---|---:|---:|---:|---:|
| decoder before / after, tok/s | 198 / 186 | 189 / 194 | 195 / 182 | 188 / 179 |
| **decoder while the prompt is read** | **9.9** | **15.1** | **16.1** | **35.6** |
| longest pause between chunks | 0.43 s | 0.20 s | 0.21 s | 0.10 s |
| the 30K prompt's read | 11.6 s | 14.0 s | 14.3 s | 19.2 s |

On the finished 0.2.0 defaults (384-token slices, 2,048 chunk) the same test
read 16.9 tok/s for the decoder during the prompt, a 0.21 s longest pause and
14.1 s for the prompt.

Without fair prefill a mid-answer user runs at a twentieth of their speed for
as long as the other prompt takes. GLM's recipe uses 384-token slices; here
that halves the pause and lifts the mid-read decoder by half for a 20 % slower
read of the long prompt. 128-token slices give the decoder 3.6× for a 65 %
slower read. 384 is the 0.2.0 default, as in GLM's recipe: one user alone
loses nothing, and several agents sharing the box is what the recipe is for.
`FAIR_CHUNK=128` favours the decoders further; `0` turns it off.

The prefill chunk itself (`MAX_BATCHED`) trades KV pool for prompt speed, the
pool because each sliding-window layer reserves in-flight tokens per request:

| `MAX_BATCHED` | KV pool, tokens | cold prefill, tok/s |
|---|---:|---:|
| 1,024 (0.1.0 default) | 1,643,784 | 2,471 |
| 2,048 (**0.2.0 default**) | 1,615,953 (−1.7 %) | 2,728 (+10 %) |
| 4,096 | 1,563,025 (−4.9 %) | 2,859 (+16 %) |

### The all-reduce kernel

`VLLM_CUSTOM_ALLREDUCE_ALGO` picks the custom all-reduce kernel (`1stage` or
`2stage`); unset, the kernel's own size heuristic chooses. Decode bench, same
day as the 0.2.0 numbers above (aggregate tok/s, structured / code / prose):

| users | unset (default; the earlier defaults run above, file not kept) | `1stage` | `2stage` |
|---|---:|---:|---:|
| 1 | 322.2 / 284.2 / 182.3 | 310.5 / 292.3 / 172.9 | 319.3 / 279.8 / 168.7 |
| 2 | 408.1 / 400.7 / 254.3 | 379.8 / 382.9 / 272.3 | 433.3 / 393.9 / 264.0 |
| 4 | 513.6 / 535.6 / 396.2 | 551.0 / 514.3 / 397.9 | 581.7 / 538.6 / 416.5 |
| 8 | 756.6 / 754.7 / 546.5 | 725.1 / 679.4 / 519.0 | 761.8 / 731.0 / 546.1 |

One-shot is slower at eight users; two-stage matches the default within
spread, so the default stays unset.

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

The decode step is 15 ms for one user whatever the prompt, 21 ms at two users,
27–32 at four and about 48 at eight (engine steps from
`vllm:iteration_tokens_total_count`; an earlier figure of 15 ms at two users
counted drafts, which are per request, not steps). The step grows with users
because every request puts 8 tokens through the model per step and the number
of distinct experts read per step grows with tokens in flight; accepted tokens
per request per step do not change with users. What moves between the columns
is the drafter: of 7 drafts per step, 99 % are accepted on a counting task,
about half on code and a fifth on prose. On an easier structured task
(counting) one user reaches 530 tok/s; the structured prompt in the table is a
40-row markdown table.

### Fewer drafts for larger batches (0.1.0 test)

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
