#!/usr/bin/env bash
set -euo pipefail
ROBOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/w84384651/miniconda3/envs/env_isaaclab/bin/python}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
if [[ "${1:-}" == "--coverage" ]]; then
    shift
    exec "$ISAAC_PYTHON" "$ROBOT_DIR/../scripts/validate_robot_sensor_coverage.py" --robot-dir "$ROBOT_DIR" "$@"
fi
exec "$ISAAC_PYTHON" "$ROBOT_DIR/../scripts/mobile_manipulator/run.py" --robot-dir "$ROBOT_DIR" "$@"
