# Status and known limits

## Status

Release 0.1.0 is a first release. The server loads, passes the boot check,
answers correctly on two code and maths suites, has published decode and
prefill numbers and passed a 20-minute, 8-stream soak with image requests mixed
in ([results.md](results.md)). Longer runs have not been measured.

## Known limits

- **Measured on one machine only**: four CMP 170HX with the x16 modification
  on one PLX switch, peer-to-peer on, driver 610.57.04. Other link widths,
  switches or drivers are untested.
- **Image input needs the bind-mounted patch.** The fork's `mimo_v2_omni.py`
  applies the vision encoder's attention sink in a form that is a no-op on
  uniform inputs, so flat-colour images are misread (upstream vLLM #58235, not
  yet merged there or in the fork). The recipe mounts a patched copy
  (`patches/`); with it a red/blue test image is described correctly. Only that
  one image test has been run; video and audio input are untested.
- **DFlash accepts few drafts on prose** (about one in five, against most on
  code and structured output), so prose decodes at 179 tok/s for one user
  against 309 on structured output. The MTP head is not the answer: it is
  slower on every column measured (`docs/results.md`).
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
