#!/usr/bin/env bash
set -euo pipefail
ROBOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="$ROBOT_DIR/.cache"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
mkdir -p "$XDG_CACHE_HOME"
exec "$ROBOT_DIR/.venv/bin/python" "$ROBOT_DIR/scripts/launch.py" "$@"
