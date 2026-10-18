#!/usr/bin/env bash
# download.sh — fetch the checkpoint into ./models (or MODELS_DIR).
#
#   ProCreations/MiMo-V2.6-Flash-RL-NVFP4 at the pinned revision: ~183 GiB,
#   without its gguf/ folder (a second copy of the weights for other engines,
#   which is excluded). The DFlash drafter is its dflash/ folder.
#
# Budget ~190 GiB of free disk. Uses the hf client from the engine image, so
# run ./install.sh first. An interrupted download resumes. Runs the preflight
# first, so it stops before downloading if a card reports under 60 GiB or the
# disk is too small.
#
# Already present? Exits 0. Re-fetch:  REFRESH_WEIGHTS=1 ./download.sh
#
# Equivalent to: ./start.sh download
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" download
