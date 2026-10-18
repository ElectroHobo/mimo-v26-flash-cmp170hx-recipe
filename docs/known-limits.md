# Status and known limits

## Status

Release 0.1.0 is a first release. The server loads, passes the boot check,
answers correctly on two code and maths suites and has published decode and
prefill numbers ([results.md](results.md)). It has run for a few hours at a
time on one machine. It is not yet soaked.

## Known limits

- **Measured on one machine only**: four CMP 170HX with the x16 modification
  on one PLX switch, peer-to-peer on, driver 610.57.04. Other link widths,
  switches or drivers are untested.
- **Multimodal input is not measured.** The engine's omni model class is
  selected and loads the vision and audio towers, but no image, video or audio
  request has been tested on this engine build. Text is what the numbers cover.
- **DFlash is weak on prose.** The drafter accepts about one draft in five on
  essay-style output, against most drafts on code and structured output. For
  prose-heavy work try `SPEC_MODE=mtp`; its numbers on this engine are not yet
  published.
- **fp8 KV is only available without DFlash.** The drafter runs on the Triton
  attention backend, which has no fp8 KV path on sm_80, so `KV_DTYPE=fp8_e4m3`
  needs `SPEC_MODE=mtp` or `none`.
- **The prefill chunk is small on purpose.** `MAX_BATCHED=1024` keeps the KV
  pool at 1.64 M tokens; each sliding-window layer reserves window plus
  in-flight tokens per request, so a larger chunk costs pool (8,192 leaves
  about 0.45 M). Prefill is correspondingly modest: ~2,500 tok/s cold.
- **NVFP4 runs on Marlin as 4-bit weights with bf16 activations**, the same
  path the fork uses for int4 checkpoints; there is no FP4 hardware on these
  cards. The quant has not been compared against the unquantised model beyond
  the two suites in results.md.
- **The KV budget is fixed bytes, not a fraction.** `KV_MEM` sizes the cache;
  `MAX_LEN` only caps a request. A card with less than 64 GiB will not fit the
  45.5 GiB of weights plus 10 GiB of KV.
- **Not measured:** sixteen or more users, sustained load, long-context
  retrieval at the 1 M limit, the `v3.3`-era official fp8 checkpoint.
