# Changelog

## 0.2.0 — 2026-10-18

- Draft depth follows the number of requests: 7 drafts for one, 5 for two, 3
  from three up (`SPEC_SCHEDULE` default; `none` restores a fixed `SPEC_N`).
  Every request puts drafts + 1 tokens through the model per step, and in a
  mixture of experts the step cost grows with tokens in flight, so fewer
  drafts under load are faster: 753.8 / 738.0 / 555.5 tok/s across eight users
  (structured / code / prose) against a fixed 7's 685.6 / 645.6 / 398.4
  re-measured the same day (0.1.0 published 696.8 / 659.0 / 413.3); one user
  322.0 / 274.9 / 170.1, unchanged within spread.
- Optional KV offload: `KV_OFFLOAD_GB` keeps evicted KV blocks in host RAM and
  `KV_OFFLOAD_DIR` adds a disk tier behind it (vLLM's OffloadingConnector), so
  a long shared prefix pushed off the cards is copied back instead of
  re-prefilled; `bench/kv_offload_bench.py` measures it. Numbers in
  `docs/results.md`.
- The prefill chunk (`MAX_BATCHED`) is 2,048: cold prefill 2,722 tok/s against
  2,471, for a KV pool of 1,615,953 tokens against 1,643,784 (the 0.1.0 note
  that 8,192 would leave 0.45 M was wrong; 4,096 measured leaves 1,563,025).
- Fair prefill on by default: `FAIR_CHUNK=384` reads a prompt in 384-token
  slices while anyone is decoding (`FAIR_PARTIAL` prompts at once), as GLM's
  recipe does; a user mid-answer keeps 15 tok/s instead of 10 while a 30K
  prompt lands, and that prompt reads 20 % slower. `0` turns it off, `128`
  favours the decoders more; measured with `bench/prefill_fairness.py`.
- Measured and left alone: the all-reduce kernel switch
  (`VLLM_CUSTOM_ALLREDUCE_ALGO`), default kept.
- results.md: the step time at two users is 21 ms, not 15 (the earlier figure
  counted drafts, which are per request, not engine steps).
- Engine, image and checkpoint pins unchanged.

## 0.1.0 — 2026-10-18

First release: MiMo-V2.6-Flash-RL (NVFP4) on four CMP 170HX cards.

- Engine: Morrowmake/vllm-cmp170hx at `18305948acba3f4c4d912cdd56f2202d2ebd94e8`
  (tag `qwen38-recipe-1.0.0`); image
  `ghcr.io/morrowmake/vllm-cmp170hx@sha256:5b45c4d6feafefe2eba962375b0a9a0b64fd22b9e05abb4d1966f14a32a2db79`.
  The engine is used as released, plus one bind-mounted file: `patches/mimo_v2_omni.py`,
  upstream vLLM #58235's vision-sink fix, without which images are misread
  (`VISION_PATCH=0` leaves it out).
- Checkpoint: `ProCreations/MiMo-V2.6-Flash-RL-NVFP4` at revision
  `34cebc5d75743a93ab76145ec609cfef9da8c417`, without its `gguf/` folder; the
  DFlash drafter is its `dflash/` folder.
- Tensor-parallel 4, 1,048,576-token context, bf16 KV with a 10 GiB budget per
  card (1,643,784 tokens), MiMo's DFlash drafter with 7 draft tokens, the MTP
  head as an option, the whole decode step as one CUDA graph.
- One command from a fresh clone: `./start.sh` or `docker compose up`, adapted
  from Morrowmake's Qwen3.8-Flash-Next recipe (MIT): preflight, pinned image,
  resumable download, content-verified peer-to-peer, two-request boot check.
- Measured: 308.6 / 286.1 / 178.8 tok/s for one user (structured / code /
  prose), 696.8 / 659.0 / 413.3 across eight, 2,473 tok/s cold prefill;
  HumanEval 161/164, GSM8K 195/200; with the MTP head instead of DFlash
  128.9 / 119.6 / 114.5 for one user — `docs/results.md`.
