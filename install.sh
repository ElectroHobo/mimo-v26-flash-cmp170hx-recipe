#!/usr/bin/env bash
# install.sh — get the pinned engine: pulls the engine image pinned in
# start.sh by digest: Morrowmake's vllm-cmp170hx release image.
#
# Already there? Exits 0 without touching it.
# Pull anyway:  FORCE_INSTALL=1 ./install.sh
#
# Equivalent to: ./start.sh install
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" install
