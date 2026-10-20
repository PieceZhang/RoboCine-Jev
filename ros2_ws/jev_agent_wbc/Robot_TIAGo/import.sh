#!/usr/bin/env bash
set -euo pipefail
ROBOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/w84384651/miniconda3/envs/env_isaaclab/bin/python}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
"$ISAAC_PYTHON" "$ROBOT_DIR/../scripts/mobile_manipulator/import_urdf.py" --urdf "$ROBOT_DIR/urdf/tiago_physics.urdf" --output "$ROBOT_DIR/assets/usd/tiago.usd" "$@"
"$ISAAC_PYTHON" "$ROBOT_DIR/scripts/finalize_usd.py"
exec "$ISAAC_PYTHON" "$ROBOT_DIR/scripts/build_sensor_asset.py"
