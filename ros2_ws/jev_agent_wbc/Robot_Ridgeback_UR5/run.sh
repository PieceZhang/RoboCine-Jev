#!/usr/bin/env bash
set -euo pipefail
ROBOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/w84384651/miniconda3/envs/env_isaaclab/bin/python}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
exec "$ISAAC_PYTHON" "$ROBOT_DIR/scripts/run_isaac.py" "$@"
