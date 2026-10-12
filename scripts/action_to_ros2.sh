#!/usr/bin/env bash
# action_to_ros2.sh — one camera action end to end:
#   step 1 (any machine, Python):  recommend → build → optimize → export → retime [--view]
#   step 2 (Linux + ROS 2 Humble): serve exports/traj_samples_retimed.json on /cam_desired_traj
#
# Usage:
#   scripts/action_to_ros2.sh --raw datasets/CameraMovementsRecommand_output_json/result0.json
#   scripts/action_to_ros2.sh --raw ... [--scene cinema_traj/scenes/frl_apartment_0.json] [--view]
# Env overrides: FRAME_ID=map  WITH_SCENE_MARKERS=1 (subject capsule markers in RViz)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SCENE="cinema_traj/scenes/frl_apartment_0.json"
RAW=""
DO_VIEW=0
while [ $# -gt 0 ]; do
  case "$1" in
    --raw)   RAW="$2"; shift 2;;
    --scene) SCENE="$2"; shift 2;;
    --view)  DO_VIEW=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$RAW" ] || { echo "need --raw <recommendation json>" >&2; exit 2; }
[ -f "$RAW" ] || { echo "not found: $RAW" >&2; exit 2; }
[ -f "$SCENE" ] || { echo "not found: $SCENE" >&2; exit 2; }
RAW_ABS="$(cd "$(dirname "$RAW")" && pwd)/$(basename "$RAW")"

# ── planning env (torch needed locally; conda cinema_traj is the known-good one)
if ! python3 -c "import torch" >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cinema_traj
  else
    echo "python env lacks torch and conda not found — activate your env first" >&2; exit 1
  fi
fi

# ── step 1: plan the action ────────────────────────────────────────────────
python run_traj.py --stage recommend --raw "$RAW_ABS" --scene "$SCENE" --force
python run_traj.py --stage build     --scene "$SCENE" --force
python run_traj.py --stage optimize  --scene "$SCENE" --force
python run_traj.py --stage export    --scene "$SCENE"
python run_traj.py --stage retime    --scene "$SCENE" --force
if [ "$DO_VIEW" -eq 1 ]; then
  python run_traj.py --stage view    --scene "$SCENE"
fi

SCENE_STEM="$(basename "$SCENE" .json)"
OUT_DIR="$REPO_ROOT/cinema_traj_out/$SCENE_STEM"
RETIMED_JSON="$OUT_DIR/exports/traj_samples_retimed.json"
echo
echo "▶ flyable trajectory: $RETIMED_JSON"

# ── step 2: serve over ROS 2 ──────────────────────────────────────────────
FRAME_ID="${FRAME_ID:-map}"
BRIDGE_ARGS=(--ros-args -p "traj_file:=$RETIMED_JSON" -p "frame_id:=$FRAME_ID")
if [ "${WITH_SCENE_MARKERS:-0}" = "1" ]; then
  BRIDGE_ARGS+=(-p "scene_file:=$REPO_ROOT/$SCENE")
fi

if command -v ros2 >/dev/null 2>&1; then
  if [ -f "$REPO_ROOT/ros2_ws/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/ros2_ws/install/setup.bash"
  fi
  echo "▶ serving on /cam_desired_traj (latched nav_msgs/Path)…"
  exec ros2 run cinema_traj_bridge traj_server "${BRIDGE_ARGS[@]}"
else
  cat <<EOF
▶ ros2 CLI not found on this machine — on the Linux box run:
    cd ${REPO_ROOT}/ros2_ws
    colcon build --packages-select cinema_traj_bridge && source install/setup.bash
    ros2 run cinema_traj_bridge traj_server \\
        --ros-args -p traj_file:=$RETIMED_JSON -p frame_id:=$FRAME_ID
EOF
fi
