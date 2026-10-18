# patches/ — one file bind-mounted over the engine

`mimo_v2_omni.py` is the engine's `vllm/model_executor/models/mimo_v2_omni.py`
from `18305948acba3f4c4d912cdd56f2202d2ebd94e8` with the two changes of upstream
vLLM pull request [#58235](https://github.com/vllm-project/vllm/pull/58235)
(not merged upstream at the time of this release, and not in Morrowmake's fork):

1. the vision encoder's learned attention sink is applied as an extra softmax
   logit (`sinks_bias_key0=False`) instead of a bias on each sequence's first
   key. The key-0 form is a no-op on uniform inputs, so every patch token of a
   flat-colour image comes out identical and the image is misread. Measured
   here before the change: a red/blue split image was described as "white and
   red" or "entirely red"; after it: "the left half is red, and the right half
   is blue". Text output is unaffected either way (same answers, same speed).
2. `packed_modules_mapping` on the omni wrapper class, copied from the text
   model, so fused projections map the same way under quantisation.

`mimo_v2_omni.vision-sink.diff` is the exact difference. `./start.sh` and
`docker-compose.yml` mount the file read-only at the engine's path. Set
`VISION_PATCH=0` to run the engine's own copy (text is identical; images are
misread). Remove this folder once the fork carries the fix: the mount is the
only place the recipe touches the engine.
