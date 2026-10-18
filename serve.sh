#!/bin/bash
# Launch MiMo-V2.6-Flash-RL (NVFP4) on four CMP 170HX cards, tensor-parallel 4.
#
# Usage: serve.sh [dflash|mtp|none|--help]
#   (no argument)  same as "dflash" -- the default.
#   dflash         MiMo's own DFlash drafter (the dflash/ folder of the
#                  checkpoint), num_speculative_tokens = SPEC_N (default 7).
#   mtp            the MTP head inside the checkpoint, SPEC_N drafts (default 2).
#   none           no speculative decoding.
# Set DRY=1 to print the environment and the command that would run instead
# of running them.
#
# Paths (all relative to this repo unless overridden):
#   VENV            the engine's venv (default ./venv; /opt/venv in the image)
#   MODEL           checkpoint directory (default ./models/MiMo-V2.6-Flash-RL-NVFP4);
#                   the DFlash drafter is its dflash/ subfolder
#   CUDA_HOME       CUDA toolkit root               (default /usr/local/cuda-13.3)
# API:
#   HOST            address to listen on (default 127.0.0.1: this machine only).
#                   0.0.0.0 listens on every interface -- set API_KEY first.
#   PORT            default below
#   API_KEY         if set, every /v1 request must send "Authorization: Bearer
#                   <key>" (passed to vLLM as VLLM_API_KEY, never on the command line).
#   SERVED_MODEL_NAME  model id clients send (default below)
# Cards: CUDA_VISIBLE_DEVICES selects exactly four distinct physical cards
#   (nvidia-smi numbering; default below).
# Env overrides: MAX_LEN, MAX_SEQS, MAX_BATCHED, KV_MEM, KV_DTYPE, SPEC_N, SPEC_SCHEDULE,
#   EXPERT_PARALLEL, GRAPH_MODE, REASONING_PARSER, TOOL_PARSER, EXTRA_ARGS, P2P, BOOT_CHECK.
#   GPU_UTIL must stay 0.94 (refused otherwise); the KV cache is sized by KV_MEM
#   bytes per card, not by the context, so MAX_LEN can be lowered freely.
#   FIRST_LIGHT=1 is the 32K/eager/no-speculation diagnostic, not a serving mode.
#
# ===== sm_80 switches =======================================================
# Each has a one-variable kill switch: set it to 0 in the environment and
# restart, no rebuild.
#   VLLM_CUSTOM_ALLREDUCE_FLAGS=0    flags-in-data two-shot TP4 all-reduce
#   P2P=auto                         verify peer content before enabling P2P
#   P2P=off                          disable all peer transports
#   P2P=force                        bypass the content check
# ============================================================================
set -euo pipefail
# generated:serving-defaults:begin
# Serving defaults for MiMo-V2.6-Flash-RL. Constants are deliberately not
# exported. The engine switches are listed only where the serving value differs
# from the engine's own default.
MIMO_DEFAULT_FIRST_LIGHT=0
MIMO_DEFAULT_MODE=dflash
MIMO_DEFAULT_DIAGNOSTIC_MODE=none
MIMO_DEFAULT_LAYOUT=tp4
MIMO_DEFAULT_CUDA_VISIBLE_DEVICES=0,1,2,3
MIMO_DEFAULT_VLLM_CUSTOM_ALLREDUCE_FLAGS=1
MIMO_DEFAULT_VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1
MIMO_DEFAULT_NCCL_P2P_LEVEL=SYS
MIMO_DEFAULT_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
MIMO_DEFAULT_VLLM_WORKER_MULTIPROC_METHOD=spawn
MIMO_DEFAULT_MAX_LEN=1048576
MIMO_DEFAULT_DIAGNOSTIC_MAX_LEN=32768
MIMO_DEFAULT_GPU_UTIL=0.94
MIMO_DEFAULT_KV_MEM=10737418240
MIMO_DEFAULT_KV_DTYPE=bfloat16
MIMO_DEFAULT_BLOCK_SIZE=128
MIMO_DEFAULT_DFLASH_SPEC_N=7
MIMO_DEFAULT_MTP_SPEC_N=2
MIMO_DEFAULT_MAX_SEQS=32
MIMO_DEFAULT_MAX_BATCHED=2048
MIMO_DEFAULT_FAIR_CHUNK=384
MIMO_DEFAULT_FAIR_PARTIAL=2
MIMO_DEFAULT_REASONING_PARSER=mimo
MIMO_DEFAULT_TOOL_PARSER=mimo
MIMO_DEFAULT_PORT=8000
MIMO_DEFAULT_SERVED_MODEL_NAME=mimo-v2.6-flash
MIMO_DEFAULT_TP=4
MIMO_DEFAULT_PP=1
MIMO_DEFAULT_HF_OVERRIDES='{"architectures":["MiMoV2OmniForCausalLM"]}'
MIMO_DEFAULT_GRAPH_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
MIMO_DEFAULT_DIAGNOSTIC_GRAPH_CONFIG='{"cudagraph_mode":"NONE"}'
# generated:serving-defaults:end
FIRST_LIGHT=${FIRST_LIGHT:-$MIMO_DEFAULT_FIRST_LIGHT}
case "$FIRST_LIGHT" in 0|1) ;; *) echo 'FIRST_LIGHT must be 0 or 1' >&2; exit 2;; esac
if [ "$FIRST_LIGHT" = 1 ]; then MODE=${1:-$MIMO_DEFAULT_DIAGNOSTIC_MODE}; else MODE=${1:-$MIMO_DEFAULT_MODE}; fi
case "$MODE" in
  -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
  dflash|mtp|none) ;;
  *) echo "serve.sh: unknown argument '$MODE' (expected dflash, mtp, none or --help)" >&2; exit 2 ;;
esac
# Container compilers use a private temporary directory on the cache mount.
# This runs for start.sh, Compose and direct launches before Python imports.
if [ -n "${RECIPE_CACHE_ROOT:-}" ] && [ "${DRY:-0}" != 1 ]; then
    [ "$RECIPE_CACHE_ROOT" = /cache ] || { echo '[boot-check] FAIL: container cache root must be /cache' >&2; exit 1; }
    temp_base="${TMPDIR:-/cache/tmp}"
    case "$temp_base/" in /cache/*) ;; *) echo '[boot-check] FAIL: container TMPDIR must be under /cache' >&2; exit 1 ;; esac
    mkdir -p "$temp_base"
    temp_base="$(realpath "$temp_base")"
    case "$temp_base/" in /cache/*) ;; *) echo '[boot-check] FAIL: container TMPDIR resolves outside /cache' >&2; exit 1 ;; esac
    TMPDIR="$(mktemp -d "$temp_base/run.XXXXXXXX")"
    export TMPDIR
    export TMP="$TMPDIR" TEMP="$TMPDIR"
    unset temp_base
fi
# Cache seeds are opt-in until the image's cold/warm acceptance passes.
if [ "${VLLM_IMAGE_CACHE_SEED:-0}" = 1 ] && [ "${DRY:-0}" = 1 ]; then
    echo "[image-cache] enabled (dry run)"
elif [ "${VLLM_IMAGE_CACHE_SEED:-0}" = 1 ]; then
    [ -f /opt/image-tools/cache_seed.py ] || { echo "[image-cache] enabled but image helper missing" >&2; exit 1; }
    cache_exports="$(/opt/venv/bin/python /opt/image-tools/cache_seed.py seed /opt/cache-seed)"
    eval "$cache_exports"
    unset cache_exports
else
    echo "[image-cache] disabled"
fi
# Explicit compiler destinations also cover launches without a seed.
if [ -n "${RECIPE_CACHE_ROOT:-}" ] && [ "${DRY:-0}" != 1 ]; then
    export HOME="${HOME:-/cache}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
    export HF_HOME="${HF_HOME:-/cache/huggingface}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"
    export VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR="${VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR:-$HOME/.cache/vllm/custom_allreduce_flags}"
    export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
    export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$HOME/.tilelang/cache}"
    export TILELANG_TMP_DIR="${TILELANG_TMP_DIR:-$HOME/.tilelang/cache/tmp}"
    export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$HOME/.nv/ComputeCache}"
    export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$HOME}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/torchinductor}"
    for cache_path in "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
        "$VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR" "$VLLM_CACHE_ROOT" "$TILELANG_CACHE_DIR" "$TILELANG_TMP_DIR" \
        "$CUDA_CACHE_PATH" "$FLASHINFER_WORKSPACE_BASE" "$TORCHINDUCTOR_CACHE_DIR"; do
        case "$cache_path/" in /cache/*) ;; *) echo "[boot-check] FAIL: compiler cache outside /cache: $cache_path" >&2; exit 1 ;; esac
        mkdir -p "$cache_path"
        cache_path="$(realpath "$cache_path")"
        case "$cache_path/" in /cache/*) ;; *) echo "[boot-check] FAIL: cache symlink outside /cache: $cache_path" >&2; exit 1 ;; esac
        [ -w "$cache_path" ] || { echo "[boot-check] FAIL: cache is not writable: $cache_path" >&2; exit 1; }
    done
    unset cache_path
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$REPO_ROOT/venv}"
MODEL="${MODEL:-$REPO_ROOT/models/MiMo-V2.6-Flash-RL-NVFP4}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export CUDA_HOME
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
LAYOUT=${LAYOUT:-$MIMO_DEFAULT_LAYOUT}
if [ "$LAYOUT" != tp4 ]; then echo 'This recipe serves one layout: LAYOUT=tp4' >&2; exit 2; fi
TP=$MIMO_DEFAULT_TP PP=$MIMO_DEFAULT_PP
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$MIMO_DEFAULT_CUDA_VISIBLE_DEVICES}
# Card numbers are physical (nvidia-smi) numbers.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*)){3}$ ]] ||
   [ "$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | sort -u | wc -l)" -ne 4 ]; then
  echo 'CUDA_VISIBLE_DEVICES must select exactly four distinct cards' >&2; exit 2
fi
# Flags-in-data two-shot custom all-reduce kernel for the TP4 all-reduces it covers.
export VLLM_CUSTOM_ALLREDUCE_FLAGS=${VLLM_CUSTOM_ALLREDUCE_FLAGS:-$MIMO_DEFAULT_VLLM_CUSTOM_ALLREDUCE_FLAGS}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-$MIMO_DEFAULT_VLLM_WORKER_MULTIPROC_METHOD}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-$MIMO_DEFAULT_PYTORCH_CUDA_ALLOC_CONF}
# Content-verified peer access: sets VLLM_CUSTOM_ALLREDUCE_PCIE_P2P and
# NCCL_P2P_LEVEL only after a pass.
source "$REPO_ROOT/p2p_check.sh"
FL_ARGS=()
# The graph mode defaults come from the serving defaults; GRAPH_MODE overrides them.
if [ "$FIRST_LIGHT" = 1 ]; then DEFAULT_GRAPH_CONFIG=$MIMO_DEFAULT_DIAGNOSTIC_GRAPH_CONFIG; else DEFAULT_GRAPH_CONFIG=$MIMO_DEFAULT_GRAPH_CONFIG; fi
DEFAULT_GRAPH_MODE=${DEFAULT_GRAPH_CONFIG#*\"cudagraph_mode\":\"}
DEFAULT_GRAPH_MODE=${DEFAULT_GRAPH_MODE%%\"*}
GRAPH_MODE=${GRAPH_MODE:-$DEFAULT_GRAPH_MODE}
case "$GRAPH_MODE" in FULL_AND_PIECEWISE|PIECEWISE|NONE) ;; *) echo 'GRAPH_MODE must be FULL_AND_PIECEWISE or PIECEWISE (NONE with FIRST_LIGHT=1)' >&2; exit 2;; esac
if [ "$GRAPH_MODE" = "$DEFAULT_GRAPH_MODE" ]; then
  GRAPH_ARGS=(--compilation-config "$DEFAULT_GRAPH_CONFIG")
else
  GRAPH_ARGS=(--compilation-config "{\"cudagraph_mode\":\"$GRAPH_MODE\"}")
fi
if [ "$FIRST_LIGHT" = 1 ]; then
  if [ "$MODE" != none ] || [ "${MAX_LEN-$MIMO_DEFAULT_DIAGNOSTIC_MAX_LEN}" != 32768 ] || [ "$GRAPH_MODE" != NONE ]; then
    echo 'FIRST_LIGHT=1 requires no speculation, MAX_LEN=32768 and GRAPH_MODE=NONE' >&2; exit 2
  fi
  FL_ARGS=(--enforce-eager)
  MAX_LEN=$MIMO_DEFAULT_DIAGNOSTIC_MAX_LEN
else
  if [ "$GRAPH_MODE" = NONE ]; then
    echo 'GRAPH_MODE=NONE requires FIRST_LIGHT=1' >&2; exit 2
  fi
  MAX_LEN=${MAX_LEN-$MIMO_DEFAULT_MAX_LEN}
  if [[ ! "$MAX_LEN" =~ ^[1-9][0-9]*$ ]] || [ "$MAX_LEN" -gt 1048576 ]; then
    echo 'MAX_LEN must be a positive number of tokens, at most 1048576 (the model limit)' >&2; exit 2
  fi
fi
GPU_UTIL=${GPU_UTIL-$MIMO_DEFAULT_GPU_UTIL}
if [ "$GPU_UTIL" != 0.94 ]; then
  echo 'This recipe requires GPU_UTIL=0.94; size the KV cache with KV_MEM instead' >&2; exit 2
fi
# The KV cache is a fixed byte budget per card, so the context length does not
# change what the weights leave free. 10 GiB per card holds ~1.64 M tokens.
KV_MEM=${KV_MEM:-$MIMO_DEFAULT_KV_MEM}
if [[ ! "$KV_MEM" =~ ^[1-9][0-9]*$ ]]; then echo 'KV_MEM must be a positive number of bytes' >&2; exit 2; fi
# bf16 KV is the default; fp8_e4m3 halves KV bytes per token, but the DFlash
# drafter's attention refuses fp8 KV on sm_80, so it is allowed with mtp or none only.
KV_DTYPE=${KV_DTYPE:-$MIMO_DEFAULT_KV_DTYPE}
case "$KV_DTYPE" in bfloat16|fp8_e4m3) ;; *) echo 'KV_DTYPE must be bfloat16 or fp8_e4m3' >&2; exit 2;; esac
if [ "$MODE" = dflash ] && [ "$KV_DTYPE" != bfloat16 ]; then
  echo 'dflash requires KV_DTYPE=bfloat16 (the drafter runs on the Triton attention backend, which has no fp8 KV on sm_80)' >&2; exit 2
fi
SPEC=()
if [ "$MODE" = dflash ]; then
  DRAFT_DIR="$MODEL/dflash"
  # A dry run on the host sees container paths, so the folder is checked only for a real start.
  if [ "${DRY:-0}" != 1 ] && [ ! -f "$DRAFT_DIR/config.json" ]; then
    echo "no DFlash drafter at $DRAFT_DIR (the checkpoint's dflash/ folder)" >&2; exit 2
  fi
  SPEC_N=${SPEC_N:-$MIMO_DEFAULT_DFLASH_SPEC_N}
  if [[ ! "$SPEC_N" =~ ^[1-9][0-9]*$ ]] || [ "$SPEC_N" -gt 8 ]; then echo 'SPEC_N must be 1..8 for dflash (the drafter blocks are 8 wide)' >&2; exit 2; fi
  SPEC_JSON="{\"method\":\"dflash\",\"model\":\"$DRAFT_DIR\",\"num_speculative_tokens\":$SPEC_N"
elif [ "$MODE" = mtp ]; then
  SPEC_N=${SPEC_N:-$MIMO_DEFAULT_MTP_SPEC_N}
  if [[ ! "$SPEC_N" =~ ^[1-9][0-9]*$ ]]; then echo 'SPEC_N must be positive' >&2; exit 2; fi
  SPEC_JSON="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_N"
fi
MAX_SEQS=${MAX_SEQS:-$MIMO_DEFAULT_MAX_SEQS}
# KV offload (KV_OFFLOAD_GB, default 0: off): a tier of host RAM, in GiB, that keeps KV
# blocks evicted from the cards and copies them back instead of re-prefilling (vLLM's
# OffloadingConnector with TieringOffloadingSpec; a shared mmap in /dev/shm, so start.sh
# sizes shm to fit). KV_OFFLOAD_DIR (empty: none) adds a disk tier behind the RAM one;
# start.sh mounts the directory named in .env at /kvtier. A drafter's cache group would
# otherwise zero every lookup (vLLM #52735), so offload also sets disable_eagle_block_drop
# in the speculative config. docs/results.md has the numbers.
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-0}
KV_OFFLOAD_DIR=${KV_OFFLOAD_DIR:-}
[[ "$KV_OFFLOAD_GB" =~ ^[0-9]+$ ]] || { echo 'KV_OFFLOAD_GB must be a whole number of GiB' >&2; exit 2; }
if [ -n "$KV_OFFLOAD_DIR" ] && [ "$KV_OFFLOAD_GB" = 0 ]; then
  echo 'KV_OFFLOAD_DIR needs a RAM tier in front of it: set KV_OFFLOAD_GB' >&2; exit 2
fi
KV_OFFLOAD=()
if [ "$KV_OFFLOAD_GB" != 0 ]; then
  OFFLOAD_JSON="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$((KV_OFFLOAD_GB * 1073741824))"
  [ -n "$KV_OFFLOAD_DIR" ] && OFFLOAD_JSON+=",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$KV_OFFLOAD_DIR\"}]"
  KV_OFFLOAD=(--kv-transfer-config "$OFFLOAD_JSON}}")
fi
if [ "$MODE" != none ]; then
  # SPEC_SCHEDULE: drafts by batch size, vLLM's num_speculative_tokens_per_batch_size, a JSON
  # list of [first_batch,last_batch,drafts] ranges without spaces starting at batch 1, e.g.
  # [[1,2,7],[3,32,4]] (drafts above SPEC_N are clamped to it). Empty: dflash gets the default
  # schedule, 7 drafts for one request, 5 for two, 3 from three up to MAX_SEQS (every request
  # puts drafts+1 tokens through the model per step, and in a mixture of experts the step
  # cost grows with tokens in flight; docs/results.md); mtp gets none. none: SPEC_N at every
  # batch size.
  SPEC_SCHEDULE=${SPEC_SCHEDULE:-}
  if [ -z "$SPEC_SCHEDULE" ] && [ "$MODE" = dflash ]; then
    SPEC_SCHEDULE="[[1,1,7]"
    [ "$MAX_SEQS" -ge 2 ] && SPEC_SCHEDULE+=",[2,2,5]"
    [ "$MAX_SEQS" -ge 3 ] && SPEC_SCHEDULE+=",[3,$MAX_SEQS,3]"
    SPEC_SCHEDULE+="]"
  fi
  [ "$SPEC_SCHEDULE" = none ] && SPEC_SCHEDULE=
  if [ -n "$SPEC_SCHEDULE" ]; then
    SCHEDULE_RANGE='\[[1-9][0-9]*,[1-9][0-9]*,[0-9]+\]'
    if [[ ! "$SPEC_SCHEDULE" =~ ^\[$SCHEDULE_RANGE(,$SCHEDULE_RANGE)*\]$ ]]; then
      echo 'SPEC_SCHEDULE must be [[first,last,drafts],...] without spaces' >&2; exit 2
    fi
    SPEC_JSON+=",\"num_speculative_tokens_per_batch_size\":$SPEC_SCHEDULE"
  fi
  # SPEC_ADAPTIVE_K (empty: off): the engine's load- and acceptance-adaptive draft count, a JSON
  # object without spaces for its adaptive_k field, e.g. {"by_load":[7,7,6,5,4]}. SPEC_N must be
  # at least the deepest count. See the engine's SpeculativeConfig.adaptive_k for the keys.
  SPEC_ADAPTIVE_K=${SPEC_ADAPTIVE_K:-}
  if [ -n "$SPEC_ADAPTIVE_K" ]; then
    [[ "$SPEC_ADAPTIVE_K" =~ ^\{.*\}$ ]] || { echo 'SPEC_ADAPTIVE_K must be a JSON object' >&2; exit 2; }
    SPEC_JSON+=",\"adaptive_k\":$SPEC_ADAPTIVE_K"
  fi
  [ "$KV_OFFLOAD_GB" != 0 ] && SPEC_JSON+=",\"disable_eagle_block_drop\":true"
  SPEC=(--speculative-config "$SPEC_JSON}")
fi
# EXPERT_PARALLEL=1 keeps whole routed experts on each card instead of slicing every
# expert four ways (vLLM --enable-expert-parallel). Off by default: see docs/results.md.
# FAIR_CHUNK (default 384; 0: off): while any request is decoding, prefill goes in slices of this
# many tokens (vLLM --prefill-chunk-with-decodes) so the decoders keep getting steps; up to
# FAIR_PARTIAL prompts may be mid-prefill at once (--max-num-partial-prefills). Measured in
# docs/results.md: a 30K prompt landing mid-answer drops the decoder from ~195 to 10 tok/s
# without it, 15 with 384-token slices, 36 with 128, while the prompt's own read takes
# 20 % / 65 % longer.
FAIR_CHUNK=${FAIR_CHUNK:-$MIMO_DEFAULT_FAIR_CHUNK}
FAIR_PARTIAL=${FAIR_PARTIAL:-$MIMO_DEFAULT_FAIR_PARTIAL}
[[ "$FAIR_CHUNK" =~ ^[0-9]+$ ]] || { echo 'FAIR_CHUNK must be a whole number of tokens (0: off)' >&2; exit 2; }
[[ "$FAIR_PARTIAL" =~ ^[1-9][0-9]*$ ]] || { echo 'FAIR_PARTIAL must be a positive whole number' >&2; exit 2; }
FAIR_ARGS=()
[ "$FAIR_CHUNK" != 0 ] && FAIR_ARGS=(--prefill-chunk-with-decodes "$FAIR_CHUNK" --max-num-partial-prefills "$FAIR_PARTIAL")
EXPERT_PARALLEL=${EXPERT_PARALLEL:-0}
case "$EXPERT_PARALLEL" in 0|1) ;; *) echo 'EXPERT_PARALLEL must be 0 or 1' >&2; exit 2;; esac
EP_ARGS=()
[ "$EXPERT_PARALLEL" = 1 ] && EP_ARGS=(--enable-expert-parallel)
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MIMO_DEFAULT_SERVED_MODEL_NAME}
PORT=${PORT:-$MIMO_DEFAULT_PORT}
# The key goes to vLLM through the environment rather than --api-key, so it
# does not show up in the process list.
if [ -n "${API_KEY:-}" ]; then export VLLM_API_KEY="$API_KEY"; fi
# --hf-overrides selects the omni (text, image, video, audio) model class; the
# checkpoint's config.json names the text-only one. --trust-remote-code is for
# the DFlash drafter's own model code in dflash/.
CMD=("$VENV/bin/vllm" serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "${HOST:-127.0.0.1}"
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" ${EP_ARGS[@]+"${EP_ARGS[@]}"}
  --hf-overrides "$MIMO_DEFAULT_HF_OVERRIDES" --trust-remote-code
  --kv-cache-dtype "$KV_DTYPE" --block-size "$MIMO_DEFAULT_BLOCK_SIZE"
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" --kv-cache-memory "$KV_MEM"
  --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "${MAX_BATCHED:-$MIMO_DEFAULT_MAX_BATCHED}"
  --reasoning-parser "${REASONING_PARSER:-$MIMO_DEFAULT_REASONING_PARSER}"
  --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-$MIMO_DEFAULT_TOOL_PARSER}"
  --enable-prefix-caching --async-scheduling
  --port "$PORT"
  ${FL_ARGS[@]+"${FL_ARGS[@]}"} "${GRAPH_ARGS[@]}" ${SPEC[@]+"${SPEC[@]}"} ${KV_OFFLOAD[@]+"${KV_OFFLOAD[@]}"}
  ${FAIR_ARGS[@]+"${FAIR_ARGS[@]}"})
# Additional arguments are whitespace-separated.
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
# FlexibleArgumentParser accepts abbreviations, underscores and dotted JSON keys.
# Config files and model overrides can change several controlled fields at once.
CONTROLLED_ARGS=(
  --config --model --hf-config-path --hf-overrides --model-class-overrides
  --model-impl --runner --convert --load-format --model-loader-extra-config
  --device-ids --distributed-executor-backend --nnodes --node-rank
  --tensor-parallel-size --pipeline-parallel-size
  --decode-context-parallel-size --prefill-context-parallel-size
  --data-parallel-size --data-parallel-size-local --data-parallel-rank
  --data-parallel-start-rank --data-parallel-address --data-parallel-rpc-port
  --data-parallel-backend --data-parallel-hybrid-lb --data-parallel-external-lb
  --data-parallel-multi-port-external-lb --enable-expert-parallel
  --enable-elastic-ep --elastic-ep-max-dp-size
  --dtype --quantization --quantization-config --kv-cache-dtype
  --kv-cache-dtype-skip-layers --mamba-cache-dtype --mamba-ssm-cache-dtype
  --attention-config --compilation-config --optimization-level --performance-mode
  --cudagraph-capture-sizes --max-cudagraph-capture-size
  --enforce-eager --speculative-config
  --spec-method --spec-model --spec-tokens --diffusion-config
  --language-model-only --limit-mm-per-prompt --mm-encoder-only
  --mm-encoder-tp-mode --mm-encoder-attn-dtype
  --max-model-len --gpu-memory-utilization --device-memory-utilization --kv-cache-memory-bytes --kv-cache-memory
  --num-gpu-blocks-override --block-size --trust-remote-code
  --prefill-chunk-with-decodes --max-num-partial-prefills
  # No KV or weight offload.
  --cpu-offload-gb --cpu-offload-params --offload-backend --offload-group-size --offload-num-in-group
  --offload-prefetch-step --offload-params --kv-offloading-size --kv-offloading-backend --kv-transfer-config
)
for argument in ${EXTRA[@]+"${EXTRA[@]}"}; do
  blocked=0
  case "$argument" in
    --) blocked=1 ;;
    --*)
      option=${argument%%=*}
      option=${option%%.*}
      option=${option//_/-}
      for controlled in "${CONTROLLED_ARGS[@]}"; do
        if [[ "$controlled" == "$option"* || "--no-${controlled#--}" == "$option"* ]]; then
          blocked=1; break
        fi
      done ;;
    -*)
      option=${argument%%=*}
      option=${option%%.*}
      for controlled in -tp -pp -dp -dcp -pcp -ep -n -r -q -sc -cc -ac -dc -O; do
        if [[ "$option" == "$controlled"* || "$controlled" == "$option"* ]]; then
          blocked=1; break
        fi
      done ;;
  esac
  if [ "$blocked" = 1 ]; then
    echo "EXTRA_ARGS cannot override the deployment contract: $argument" >&2; exit 2
  fi
done
CMD+=(${EXTRA[@]+"${EXTRA[@]}"})
echo "serve.sh: MiMo-V2.6-Flash-RL NVFP4 TP=$TP PP=$PP GPUs=$CUDA_VISIBLE_DEVICES KV=$KV_DTYPE KV_MEM=$KV_MEM ctx=$MAX_LEN speculation=$MODE${SPEC_N:+ k=$SPEC_N}${SPEC_SCHEDULE:+ schedule=$SPEC_SCHEDULE}${SPEC_ADAPTIVE_K:+ adaptive_k=$SPEC_ADAPTIVE_K} EP=$EXPERT_PARALLEL KV_OFFLOAD=${KV_OFFLOAD_GB}G${KV_OFFLOAD_DIR:++disk} FAIR_CHUNK=$FAIR_CHUNK FIRST_LIGHT=$FIRST_LIGHT"
echo "serve.sh: CUDA graph mode=$GRAPH_MODE (GRAPH_MODE=$GRAPH_MODE)"
BOOT_CHECK=${BOOT_CHECK:-1}
case "$BOOT_CHECK" in 0|1) ;; *) echo "[boot-check] FAIL: BOOT_CHECK must be 0 or 1" >&2; exit 2 ;; esac
if [ "${DRY:-0}" = 1 ]; then
  echo "[boot-check] BOOT_CHECK=$BOOT_CHECK (dry run)"
  echo "serve.sh: DRY=1, layout $LAYOUT (PP=$PP TP=$TP), environment the server would get:"
  env | LC_ALL=C sort | grep -E '^(VLLM_|NCCL_|PYTORCH_CUDA_ALLOC_CONF=|CUDA_VISIBLE_DEVICES=|CUDA_DEVICE_ORDER=)' \
    | grep -v '^VLLM_API_KEY=' | sed 's/^/  /' || true
  if [ -n "${VLLM_API_KEY:-}" ]; then echo "  VLLM_API_KEY=(set, not shown)"; else echo "  (no API key: /v1 is open to anyone who can reach HOST:PORT)"; fi
  echo "serve.sh: DRY=1, command:"
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi
if [ "$BOOT_CHECK" = 0 ]; then
  echo "[boot-check] disabled (BOOT_CHECK=0)"
elif [ "${RECIPE_BOOT_CHECK_OWNER:-}" = start ]; then
  echo "[boot-check] delegated to start.sh after health"
else
  CLIENT_HOST=${HOST:-127.0.0.1}
  case "$CLIENT_HOST" in 0.0.0.0|::|'[::]'|'') CLIENT_HOST=127.0.0.1 ;; esac
  boot_args=()
  [ -z "${RECIPE_CACHE_ROOT:-}" ] || boot_args+=(--container)
  exec "$VENV/bin/python" "$REPO_ROOT/boot_check.py" ${boot_args[@]+"${boot_args[@]}"} \
    --base "http://$CLIENT_HOST:$PORT" \
    --model "$SERVED_MODEL_NAME" \
    --ready-timeout "${READY_TIMEOUT:-1800}" --serve "${CMD[@]}"
fi
exec "${CMD[@]}"
