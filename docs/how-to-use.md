# How to use this repo

## What you need

Four CMP 170HX cards exposing 64 GiB each on x16 links, an NVIDIA driver of 580
or newer, Docker with the NVIDIA Container Toolkit usable by your user, about
190 GiB of free disk for the checkpoint, and `git`, `curl`, `flock`, `setsid`,
Python 3 and `jq` on the host. The README's [one-time host setup](../README.md#one-time-host-setup)
has the checks.

## Step by step

1. `git clone https://github.com/ElectroHobo/mimo-v26-flash-cmp170hx-recipe && cd mimo-v26-flash-cmp170hx-recipe`
2. Optional: `cp .env.example .env` and set the cards (`CUDA_VISIBLE_DEVICES`),
   where the checkpoint should go (`MODELS_DIR`) and the port. `./start.sh`
   writes `.env` from the example on its first run if you skip this.
3. `./start.sh` — preflight, pull the image (20 GB), download the checkpoint
   (183 GiB, resumable, rate-limited without `HF_TOKEN`), launch, wait for
   `/health`, run the boot check. The first start compiles kernels and takes
   10–15 minutes after the download; later starts take about 8 minutes (the
   weights alone take ~4 minutes to load).
4. `./start.sh smoke` — one chat request and one tool call.
5. `./start.sh status` at any time; `./start.sh logs` to follow the server.

`DRY=1 ./start.sh` prints everything a real start would do, including the
server's environment and command, and changes nothing.

## Talk to it

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-v2.6-flash",
  "messages": [{"role": "user", "content": "What is (45*12)+(78/6)? Reply with just the number."}],
  "max_tokens": 200, "temperature": 0
}'
```

MiMo thinks before answering: the reasoning is returned in the message's
`reasoning` field and the answer in `content`. Size `max_tokens` for both. To
turn thinking off for a request, add `"chat_template_kwargs": {"enable_thinking": false}`.
Tool calls use the `mimo` parser (`tools` and `tool_choice` as in the OpenAI
API); see [`examples/`](../examples/).

## Choose the drafter

| | `SPEC_MODE=dflash` (default) | `SPEC_MODE=mtp` | `SPEC_MODE=none` |
|---|---|---|---|
| What drafts | MiMo's DFlash model, 5 layers, in the checkpoint's `dflash/` folder | the one-layer MTP head inside the checkpoint | nothing |
| Drafts per step | 7 for one request, 5 for two, 3 from three up (`SPEC_SCHEDULE`; `SPEC_N`, 1–8, caps it) | 2 (`SPEC_N`) | — |
| Best for | everything measured: 322 / 275 / 170 tok/s for one user (structured / code / prose) | fp8 KV, or the largest KV pool (1.82 M tokens): 129 / 120 / 115 tok/s for one user | measuring the base model; set `BOOT_CHECK=0` |
| KV precision | bf16 only | bf16 or `KV_DTYPE=fp8_e4m3` | bf16 or fp8 |

Changing any of these needs `./start.sh restart`.

## Keep long contexts around (KV offload)

Off by default. When several requests share a long prefix (agents working
from the same documents, a long chat that keeps going), other traffic can push
that prefix's KV off the cards and the next request re-reads it: about 50 s
for 100K tokens here. `KV_OFFLOAD_GB=64` in `.env` keeps evicted blocks in
host RAM and copies them back in under a second (`docs/results.md`); the
container's `/dev/shm` grows to hold the tier, so the host needs that much RAM
free. `KV_OFFLOAD_DIR=/path/on/a/fast/disk` adds a disk tier behind the RAM
tier: a reload from a SATA SSD took about 5 s, and the directory is
write-through and never pruned by the engine, so give it a disk of its own
and prune it yourself. Both need `./start.sh restart`. The KV pool on the
cards does not change.

## Serving other machines

Set `HOST=0.0.0.0` and an `API_KEY` in `.env`, then restart. The key is passed
to the server through its environment, never on the command line, and every
`/v1` request must send it as `Authorization: Bearer <key>`. `/health` never
needs it.

## Day to day

- `./start.sh restart` after any change to `.env`.
- `./start.sh update` pulls this repository and the image if the pin moved,
  then restarts; your `.env` is kept.
- `./start.sh stop` stops only the container this checkout started; the image,
  weights and compile caches stay, so the next start is quick.
- The server has no restart policy under `./start.sh`; Docker Compose brings
  it back after a crash up to three times.

## Advanced

### PCIe peer-to-peer

Every start runs a content-verified peer-to-peer check between the four cards
(`p2p_check.py`). On a pass, NCCL and the custom all-reduce use peer access;
on a failure the server runs with P2P off and says so in the log
(`grep '\[p2p\]' logs/serve.log`). `P2P=off` skips it; `P2P=force` enables
peer access without the check — only if you know your driver and switch are
right.

### Memory

`KV_MEM` is the KV cache budget per card in bytes (default 10 GiB, about
1.62 M tokens at bf16 with the 2,048 chunk). `MAX_LEN` caps a request and does not change memory.
`MAX_BATCHED` is the prefill chunk and trades KV pool for prefill speed, see
`.env.advanced.example`. `GPU_UTIL` is fixed at 0.94.

### Kill switches

`VLLM_CUSTOM_ALLREDUCE_FLAGS=0` turns off the fork's flags-in-data all-reduce;
`P2P=off` turns off peer access. Both are environment variables: set them in
`.env` or as a prefix, then restart.

### Run the container by hand

The server is a plain `vllm serve` on Morrowmake's image; `DRY=1 ./start.sh`
prints the exact command and environment, which can be run with `docker run`
and the mounts shown there.
