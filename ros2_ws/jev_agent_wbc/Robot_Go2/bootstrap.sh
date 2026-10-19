#!/usr/bin/env bash
set -euo pipefail
ROBOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/w84384651/miniconda3/envs/env_isaaclab/bin/python}"
mkdir -p "$ROBOT_DIR/vendor" "$ROBOT_DIR/results"
clone_pin() {
  local remote="$1" directory="$2" revision="$3"
  if [[ ! -d "$directory/.git" ]]; then
    git clone "$remote" "$directory"
  fi
  if [[ "$(git -C "$directory" rev-parse HEAD)" != "$revision" ]]; then
    git -C "$directory" checkout --detach "$revision"
  fi
}
clone_pin https://github.com/zzzJie-Robot/LeggedManip_Lab.git "$ROBOT_DIR/vendor/LeggedManip_Lab" 4e12109d1eccb1d9da2e4a35f189fabea3f13677
clone_pin https://github.com/nayon007/Isaac-sim-go2-with-arm.git "$ROBOT_DIR/vendor/Isaac-sim-go2-with-arm" 1a0607b1b571302df72ceab35dafcd6c3a1103ef
clone_pin https://github.com/isaac-sim/IsaacLab.git "$ROBOT_DIR/vendor/IsaacLab" b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8
if git -C "$ROBOT_DIR/vendor/IsaacLab" apply --check "$ROBOT_DIR/isaaclab_sim51_metadata.patch" 2>/dev/null; then
  git -C "$ROBOT_DIR/vendor/IsaacLab" apply "$ROBOT_DIR/isaaclab_sim51_metadata.patch"
elif ! git -C "$ROBOT_DIR/vendor/IsaacLab" apply --reverse --check "$ROBOT_DIR/isaaclab_sim51_metadata.patch" 2>/dev/null; then
  echo "IsaacLab dependency metadata differs from the recorded compatibility patch." >&2
  exit 1
fi
if [[ ! -x "$ROBOT_DIR/.venv/bin/python" ]]; then
  "$ISAAC_PYTHON" -m venv --system-site-packages "$ROBOT_DIR/.venv"
fi
"$ROBOT_DIR/.venv/bin/python" -m pip install -c "$ROBOT_DIR/constraints.txt" \
  fastapi==0.115.7 wheel==0.45.1 \
  -e "$ROBOT_DIR/vendor/IsaacLab/source/isaaclab" \
  -e "$ROBOT_DIR/vendor/IsaacLab/source/isaaclab_assets" \
  -e "$ROBOT_DIR/vendor/IsaacLab/source/isaaclab_rl[rsl-rl]" \
  -e "$ROBOT_DIR/vendor/IsaacLab/source/isaaclab_tasks" \
  -e "$ROBOT_DIR/vendor/LeggedManip_Lab/source/LeggedManip_Lab" \
  -r "$ROBOT_DIR/vendor/LeggedManip_Lab/requirements.txt"
"$ROBOT_DIR/.venv/bin/python" "$ROBOT_DIR/scripts/prepare_d1.py"
