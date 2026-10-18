# Status and known limits

## Status

Release 0.2.0 changes the draft depth under load and adds an optional KV
offload tier; the 0.1.0 server loaded, passed the boot check, answered
correctly on two code and maths suites, and passed a 20-minute, 8-stream soak
with image requests mixed in ([results.md](results.md)). Longer runs have not
been measured, and the quality suites have not been re-run on 0.2.0 (the
drafter changes nothing about which tokens are produced, only how many are
proposed per step).

## Known limits

- **Peer-to-peer must work**: with `P2P=off` decode is about 40 % slower on every
  measurement (results.md). Without a P2P-capable driver build the server runs,
  but not at the published speed.
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
  code and structured output), so prose decodes at 170 tok/s for one user
  against 322 on structured output. The MTP head is not the answer: it is
  slower on every column measured (`docs/results.md`).
- **fp8 KV is only available without DFlash.** The drafter runs on the Triton
  attention backend, which has no fp8 KV path on sm_80, so `KV_DTYPE=fp8_e4m3`
  needs `SPEC_MODE=mtp` or `none`.
- **The prefill chunk trades KV pool for prompt speed.** `MAX_BATCHED=2048`
  (0.2.0) gives 1.62 M tokens of pool and 2,720 tok/s cold prefill; each
  sliding-window layer reserves in-flight tokens per request, so 1,024 has
  1.7 % more pool and reads prompts 10 % slower, 4,096 has 3 % less pool and
  reads 5 % faster (results.md). Prefill is modest either way.
- **A long prompt still slows whoever is mid-answer.** With the default
  `FAIR_CHUNK=384` a 30K prompt arriving during an answer holds that answer
  to about 15 tok/s until the prompt is read (10 without fair prefill, 36 at
  `128`), and the prompt itself reads 20 % slower than it would alone
  (results.md). Decode and prefill share one engine; a disaggregated prefill
  would be the real fix.
- **NVFP4 runs on Marlin as 4-bit weights with bf16 activations**, the same
  path the fork uses for int4 checkpoints; there is no FP4 hardware on these
  cards. The quant has not been compared against the unquantised model beyond
  the two suites in results.md.
- **The KV budget is fixed bytes, not a fraction.** `KV_MEM` sizes the cache;
  `MAX_LEN` only caps a request. A card with less than 64 GiB will not fit the
  45.5 GiB of weights plus 10 GiB of KV.
- **Not measured:** sixteen or more users, sustained load, long-context
  retrieval at the 1 M limit, the `v3.3`-era official fp8 checkpoint.
