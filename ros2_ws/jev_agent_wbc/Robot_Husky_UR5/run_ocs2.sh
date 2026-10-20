#!/usr/bin/env bash
set -euo pipefail
robot_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
isaac_python="${ISAAC_PYTHON:-/home/w84384651/miniconda3/envs/env_isaaclab/bin/python}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
exec "$isaac_python" "$robot_dir/../scripts/ocs2_wheeled_runtime.py" --robot-dir "$robot_dir" "$@"
