# Changelog

## 0.1.0 — 2026-10-18

First release: MiMo-V2.6-Flash-RL (NVFP4) on four CMP 170HX cards.

- Engine: Morrowmake/vllm-cmp170hx at `18305948acba3f4c4d912cdd56f2202d2ebd94e8`
  (tag `qwen38-recipe-1.0.0`); image
  `ghcr.io/morrowmake/vllm-cmp170hx@sha256:5b45c4d6feafefe2eba962375b0a9a0b64fd22b9e05abb4d1966f14a32a2db79`.
  The engine is used as released: no patches, no bind-mounted files.
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
  HumanEval 161/164, GSM8K 195/200 — `docs/results.md`.
