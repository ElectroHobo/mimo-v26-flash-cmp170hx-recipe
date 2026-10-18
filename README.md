# MiMo-V2.6-Flash-RL on 4× CMP 170HX

[![engine](https://img.shields.io/badge/engine-vllm--cmp170hx%2018305948ac-blue)](https://github.com/Morrowmake/vllm-cmp170hx/tree/18305948acba3f4c4d912cdd56f2202d2ebd94e8)
![release](https://img.shields.io/badge/release-0.1.0-2ea44f)

**XiaomiMiMo's MiMo-V2.6-Flash-RL (309B MoE, 15B active, text / image / video /
audio input) on four CMP 170HX cards with the full 1,048,576-token context:
309 / 286 / 179 tok/s for one user (structured / code / prose), 697 / 659 / 413
tok/s across eight, 2,470 tok/s cold prefill and a 1,643,784-token KV pool.
NVFP4 weights, bf16 KV, MiMo's own DFlash drafter, every weight on the GPUs.**

Serve [ProCreations/MiMo-V2.6-Flash-RL-NVFP4](https://huggingface.co/ProCreations/MiMo-V2.6-Flash-RL-NVFP4)
on four NVIDIA CMP 170HX cards with [Morrowmake's vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx)
fork of vLLM: an OpenAI-compatible API with the model's full context,
speculative decoding, multimodal input, reasoning and tool calls. The engine
runs only in a container pinned by digest. One command from a fresh clone,
after the [one-time host setup](#one-time-host-setup):

```bash
git clone https://github.com/ElectroHobo/mimo-v26-flash-cmp170hx-recipe
cd mimo-v26-flash-cmp170hx-recipe
./start.sh          # pull the engine image, fetch the model, start, wait for /health
```

`./start.sh` skips anything already done, so running it twice is safe; an
interrupted download resumes. The checkpoint is downloaded at the exact
Hugging Face revision this release was validated with.

**Prefer Docker Compose?** The same container, also one command from a fresh
clone: `docker compose up` (or `docker compose up -d`). Use one or the other,
not both.

> **This is a community recipe.** It is made and maintained by its author, not
> by Morrowmake, who has not tested, endorsed or supported it. It runs on their
> engine and their image; what is ours is the configuration, the scripts
> around it, one bind-mounted engine file ([`patches/`](patches/), upstream
> vLLM #58235's vision fix) and the measurements. Report problems in
> [this repository's issues](https://github.com/ElectroHobo/mimo-v26-flash-cmp170hx-recipe/issues).

> **Which setups this release is for.** Four cards in tensor-parallel 4,
> measured on **PCIe x16 links** (cards with the x16 capacitor modification)
> behind one PCIe switch, with peer-to-peer on. Performance on x4 links has not
> been measured. `./start.sh` reports each selected card's link width and warns
> if one is narrower than x16.

## One-time host setup

- **4× NVIDIA CMP 170HX, each exposing 64 GiB**; PCIe x16 links
- Linux with NVIDIA driver **580 or newer** (measured on 610.57.04)
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  usable by your user: `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi`
  lists the cards
- optional: a P2P-capable driver build for verified peer-to-peer; without it
  the server runs with P2P disabled (`P2P=auto` checks at every start)
- for `./start.sh`: `git`, `curl`, `flock`, `setsid`, Python 3 for the boot
  check, and `jq` for the smoke test
- about 190 GiB of free disk for the checkpoint, plus the engine image (20 GB)
  and the kernel compile caches

## Hardware

| | |
|---|---|
| GPUs | 4× NVIDIA CMP 170HX (sm_80), 65,536 MiB each, power limit 180 W |
| Links | PCIe Gen2 x16 per card, all four on one PLX switch; peer-to-peer verified at start |
| Host | 2× Xeon E5-2697A v4, 204 GiB RAM — the step runs as full CUDA graphs, so a slow CPU does not hold it back |
| Driver | 580 or newer (measured on 610.57.04) |
| Disk | ~190 GiB free for the checkpoint (183 GiB; the Hub repo's `gguf/` folder is not downloaded) |
| Software | Docker with the NVIDIA Container Toolkit |

Every weight sits on the four GPUs (45.5 GiB per card), with 10 GiB of KV cache
per card. Tensor-parallel 4; the four cards are chosen with
`CUDA_VISIBLE_DEVICES`.

## Engine

| | |
|---|---|
| Engine | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx/tree/18305948acba3f4c4d912cdd56f2202d2ebd94e8) at `18305948acba3f4c4d912cdd56f2202d2ebd94e8` (tag `qwen38-recipe-1.0.0`) |
| Image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:5b45c4d6feafefe2eba962375b0a9a0b64fd22b9e05abb4d1966f14a32a2db79`. The first start compiles kernels, so it is slower; later starts reuse `./cache` |
| Checkpoint | `ProCreations/MiMo-V2.6-Flash-RL-NVFP4`, revision `34cebc5d75743a93ab76145ec609cfef9da8c417` |
| Precision | NVFP4 weights (ModelOpt mixed precision), run through Marlin as 4-bit weights with bf16 activations; bf16 KV cache |
| Speculation | MiMo's DFlash drafter (the checkpoint's `dflash/` folder), 7 draft tokens; the MTP head is the alternative |
| Graphs | the whole decode step as one CUDA graph (`FULL_AND_PIECEWISE`) |
| Patch | one engine file bind-mounted over the image's copy: `mimo_v2_omni.py` with upstream vLLM [#58235](https://github.com/vllm-project/vllm/pull/58235)'s vision-sink fix ([`patches/`](patches/)); `VISION_PATCH=0` runs without it, text unchanged, images misread |

The engine's `MiMoV2OmniForCausalLM` is selected with `--hf-overrides`, so the
checkpoint files are used as downloaded.

## Headline numbers

Measured 2026-10-18 on 4× CMP 170HX, 74 SMs per card, P2P on, recipe defaults,
180 W per card and PCIe x16 links, under the protocol Morrowmake's recipes use:
temperature 0, thinking off, every reply pinned to 400 tokens, a salt per
request; one user is the median of five streaming decode rates, two to eight
users the median of three aggregate rates (completion tokens over batch wall
time, prefill and client overhead included); cold prefill the median over three
real-text prompts, three repeats each, no cache hits. The harness is
[`bench/cmp_recipe_bench.py`](bench/cmp_recipe_bench.py).

| | Tensor-parallel 4 + peer-to-peer |
|---|---:|
| **1 user** · structured / code / prose | **308.6 / 286.1 / 178.8** tok/s |
| **2 users, total** · structured / code / prose | 426.5 / 372.0 / 232.5 |
| **4 users, total** · structured / code / prose | 582.1 / 525.0 / 329.3 |
| **8 users, total** · structured / code / prose | 696.8 / 659.0 / 413.3 |
| **Prompt reading** (cold prefill) | 2,473 tok/s (24K / 34K / 38K-token prompts) |
| **KV pool** (tokens) | 1,643,784 (1.57 full-length requests) |

Decode is 14.8 ms per step for one user whatever the prompt; the drafter is
accepted 99 % of the time on counting, ~50 % on code and ~20 % on prose, which
is where the spread between the columns comes from.

| Other measurements | Value |
|---|---|
| Quality | HumanEval 161/164; GSM8K 195/200 (first 200 problems), thinking on |
| Arithmetic sanity | (45×12)+(78/6) = 553 |
| Context | 1,048,576 tokens accepted |
| Image input | a red/blue split image is described correctly with the patch; without it, misread (see `patches/`) |

Not yet measured: video and audio input, sustained load, sixteen users,
long-context retrieval. See [docs/known-limits.md](docs/known-limits.md).

## Using it

`./start.sh` runs preflight (read-only GPU, link, disk and port checks),
installs the engine, downloads the checkpoint, launches the server detached
(output in `logs/serve.log`) and waits for `/health`, then runs a two-request
startup check. Commands:

| Command | What it does |
|---|---|
| `./start.sh` | preflight, install, download, launch (default) |
| `./start.sh install` | the engine image only |
| `./start.sh download` | the checkpoint only |
| `./start.sh stop` / `restart` | stop or restart the server this checkout started |
| `./start.sh status` | server, `/health`, KV cache line |
| `./start.sh logs` | follow `logs/serve.log` |
| `./start.sh update` | `git pull`, pull the image if the pin moved, restart |
| `./start.sh smoke` | one chat request and one tool call |
| `DRY=1 ./start.sh` | print what would happen, change nothing |

The API listens on `127.0.0.1:8000` as model `mimo-v2.6-flash`, with no API
key. Reasoning comes back in the message's `reasoning` field and the answer in
`content`; thinking can be turned off per request with
`"chat_template_kwargs": {"enable_thinking": false}`. Examples: [`examples/`](examples/).

## Options

Settings go in `.env` (copied from [`.env.example`](.env.example) on the first
run); a prefix assignment beats `.env` for one run
(`CUDA_VISIBLE_DEVICES=0,2,3,4 ./start.sh restart`). The common ones:

| Setting | Default | Meaning |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | the four cards (exactly four, distinct) |
| `HOST` | `127.0.0.1` | listen address; `0.0.0.0` for every interface (set `API_KEY` first) |
| `PORT` | `8000` | API port |
| `API_KEY` | unset | require `Authorization: Bearer <key>` |
| `SERVED_MODEL_NAME` | `mimo-v2.6-flash` | model id clients send |
| `MODELS_DIR` | `./models` | checkpoint location |
| `SPEC_MODE` | `dflash` | `dflash` (MiMo's drafter, fastest everywhere measured), `mtp` (the head in the checkpoint; allows fp8 KV) or `none` (set `BOOT_CHECK=0`, the startup check expects drafting) |
| `SPEC_N` | `7` / `2` | draft tokens per step for dflash / mtp |
| `MAX_LEN` | `1048576` | context; lowering it frees nothing, the KV cache is a byte budget |
| `KV_MEM` | `10737418240` | KV cache bytes per card (10 GiB ≈ 1.64 M tokens) |
| `KV_DTYPE` | `bfloat16` | or `fp8_e4m3` with `mtp` / `none` |
| `MAX_SEQS` | `32` | concurrent sequences |
| `MAX_BATCHED` | `1024` | tokens per scheduler step; bigger costs KV pool (see `.env.advanced.example`) |
| `P2P` | `auto` | `auto` (content-checked), `off`, `force` |
| `SPEC_SCHEDULE` | empty | drafts by batch size, e.g. `[[1,2,7],[3,32,4]]`; helps prose at moderate concurrency, costs structured output at eight users (`docs/results.md`) |
| `SPEC_ADAPTIVE_K` | empty | the engine's load- and acceptance-adaptive draft depth, a JSON object; helps prose, costs structured output (`docs/results.md`) |
| `EXPERT_PARALLEL` | `0` | `1` keeps whole experts per card; measured 4–9 % slower here |
| `EXTRA_ARGS` | empty | extra vLLM arguments; options that change the deployment are refused |

GPU memory fraction (0.94) and the tensor-parallel layout are fixed: `serve.sh`
refuses other values. `FIRST_LIGHT=1` is a 32K/eager/no-speculation diagnostic,
not a serving mode. Everything else is in [`.env.advanced.example`](.env.advanced.example).

### Choosing a drafter

DFlash is the default and the faster choice for every kind of output measured:
with the MTP head instead (`SPEC_MODE=mtp`, 2 drafts) one user gets
128.9 / 119.6 / 114.5 tok/s (structured / code / prose) and eight users
417.2 / 379.8 / 362.3, so even on prose, where DFlash accepts only a fifth of
its drafts, DFlash is ahead. MTP remains for two reasons: it allows
`KV_DTYPE=fp8_e4m3`, and without the drafter's weights the KV pool grows to
1,823,860 tokens. `SPEC_MODE=none` is for measuring the base model.

## Update

- `./start.sh update` pulls the new release and engine image and restarts; your
  `.env` is preserved.
- With Compose: `git pull`, then `docker compose up -d`.

## Licence

The recipe is MIT ([`LICENSE`](LICENSE)); its scripts descend from
Morrowmake's MIT-licensed Qwen3.8-Flash-Next recipe. The engine is Apache-2.0;
the checkpoint and the model are MIT (see [docs/model-licences.md](docs/model-licences.md)).

## Credits

- **Morrowmake** — the [vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx)
  engine and image this runs on, and the
  [Qwen3.8-Flash-Next recipe](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe)
  whose scripts (`start.sh`, `serve.sh`, the P2P and boot checks, Compose) this
  recipe adapts. Not involved in this recipe.
- **kaka86mm** — [vllm-mimo-170hx](https://github.com/kaka86mm/vllm-mimo-170hx)
  (MIT), the first MiMo-V2.6 deployment on these cards; the launch
  configuration here (block size, the 1,024-token prefill chunk and its KV-pool
  finding, the KV budget, the parsers) descends from theirs.
- **ProCreations** — the [NVFP4 checkpoint](https://huggingface.co/ProCreations/MiMo-V2.6-Flash-RL-NVFP4)
  (MIT) and the DFlash drafter packaged with it.
- **XiaomiMiMo** — [MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)
  (MIT) and its DFlash drafter.
- **MiaAI-Lab** — the benchmark protocol and category prompts (from their
  [DGX Spark MiMo recipe](https://github.com/MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks),
  AGPL-3.0; the prompts are used here as data, the harness is our own), and the
  observation that MiMo's two drafters win different workloads.

If this recipe helps you, credit the people above and link their work.
Citation metadata for this repository: [`CITATION.cff`](CITATION.cff).
