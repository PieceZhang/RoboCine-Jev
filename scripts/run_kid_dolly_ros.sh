#!/usr/bin/env bash
# run_kid_dolly_ros.sh — full kid-dolly loop in one command:
#
#   jvs_mock_simview.launch.py   Isaac GUI: kujiale stage + spawned JVS robot
#                                (command adapter OFF — the whole-body QP
#                                publishes /isaac_joint_commands directly)
#   whole_body_controller        jvs_mock WBC: relative 6D task-space JSON →
#                                50 Hz velocity-level QP on [wheels, joint1..6]
#
# The planned trajectory comes from the cinema_traj pipeline (no hand-authored
# coordinates):
#   python tools/scenario_gen.py \
#       cinema_traj/scenes/config/kujiale_0021_kid_dolly.yml
#   python run_traj.py --stage export \
#       --scene cinema_traj/scenes/kujiale_0021_kid_dolly.json
#   → cinema_traj_out/kujiale_0021_kid_dolly/exports/traj_samples*.json
#     (this script converts it to the WBC's jvs_relative v1 stream)
#
# The WBC has no trigger topic: it starts tracking settle_s (2 s) after
# synchronized /joint_states + /base_controller/odom arrive, and prints
# WBC_RESULT {...} when the pass completes (telemetry in the metrics dir).
#
# Skip the Isaac stage (WBC only):
#   bash scripts/run_kid_dolly_ros.sh --no-isaac
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO/cinema_traj_out/kujiale_0021_kid_dolly"
TRAJ_JSON="$OUT_DIR/exports/traj_samples_retimed.json"     # preferred: retimed
[[ -f "$TRAJ_JSON" ]] || TRAJ_JSON="$OUT_DIR/exports/traj_samples.json"
WBC_JSON="$OUT_DIR/exports/jarvis_task_space.json"         # jvs_relative v1
WBC_OUT="$OUT_DIR/wbc_metrics"
STAGE_USDA="$OUT_DIR/simview/simview.usda"
WBC_LOG=/tmp/robocine_kid_dolly_wbc.log
NO_ISAAC=0
for a in "$@"; do [[ "$a" == "--no-isaac" ]] && NO_ISAAC=1; done

# ── ROS 2 Humble + workspace ──────────────────────────────────────────────── #
# ROS/conda setup scripts reference unbound vars — relax `set -u` around them
set +u
source /opt/ros/humble/setup.bash
source "$REPO/ros2_ws/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# ── Isaac Sim python env (for the jvs_mock runner) ────────────────────────── #
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
# conda activate reorders PATH and drops /opt/ros/humble/bin (xacro lives
# there) — restore the ROS layer in front of the conda bin
export PATH="/opt/ros/humble/bin:$PATH"
command -v xacro >/dev/null || { echo "❌ xacro still not on PATH"; exit 1; }
set -u
export OMNI_KIT_ACCEPT_EULA=yes
export DISPLAY="${DISPLAY:-:1}"
export ROBOCINE_REPO="$REPO"     # tracking node's cinema_traj import path

PIDS=()
trap 'kill ${PIDS[@]} 2>/dev/null' EXIT INT TERM

# ── 1. Isaac backend (jvs_mock simview stage) ─────────────────────────────── #
# Spawn pose = plan start park point, matched to the CURRENT (10:51 regen)
# simview stage + retimed plan: floor is at Isaac z=0 (NOT −0.3 — that was
# the old stage), and the dolly start waypoint (4.725, 0.808) looks +X at
# the Kid, so the base parks 0.32 m behind it with plan yaw 90° (arm IK
# converges off the max-extension singularity there; 0.40 m stalls at
# ~2 cm residual). The runner converts (x, z, yaw) → Isaac (x, −z, yaw−90°).
# After Isaac prints "JVS simview ready", pre-pose the arm to the start-pose
# IK solution — otherwise the arm settles drooping into joint limits and
# ALIGN's fallback keeps it there (44° pitch error at the dolly start).
if [[ "$NO_ISAAC" != "1" ]]; then
    echo "══ 1/2 jvs_mock simview backend (Isaac GUI, third-person viewport) ══"
    # spawn = plan start park pose; activate_camera:=false keeps the default
    # Perspective viewport (third person — robot + subject both visible);
    # start_trajectory_follower:=false + start_command_adapter:=false — the
    # canned demo follower and the adapter would fight the whole-body QP
    # (/isaac_joint_commands comes from the WBC alone)
    ros2 launch jvs_mock jvs_mock_simview.launch.py \
        simview_stage:="$STAGE_USDA" \
        x:=4.725 z:=0.408 yaw:=0.0 spawn_height:=0.002 \
        activate_camera:=false start_trajectory_follower:=false \
        start_command_adapter:=false \
        > /tmp/robocine_kid_dolly_isaac.log 2>&1 &
    PIDS+=($!)
    # wait for the backend, then pre-pose the arm at the start IK solution
    # (must be AFTER the position drives take over; ~75 s Isaac cold start)
    for _ in $(seq 1 60); do
        sleep 2
        grep -q "JVS simview ready" /tmp/robocine_kid_dolly_isaac.log 2>/dev/null && break
    done
    # two-step pre-pose to the planner's start config (jarvis_traj.json
    # configs[0]) — joint5 travels 2.4 rad and the position drive stalls
    # mid-way if handed the full span at once
    (ros2 topic pub --rate 20 /arm_position_controller/commands \
        std_msgs/msg/Float64MultiArray \
        "data: [0.005, -0.08, 1.2, 0.0, -1.2, 0.0]" & PP=$!
     sleep 8; kill $PP) >/dev/null 2>&1
    (ros2 topic pub --rate 20 /arm_position_controller/commands \
        std_msgs/msg/Float64MultiArray \
        "data: [0.005, -0.155, 2.432, -0.001, -2.406, -0.006]" & PP=$!
     sleep 22; kill $PP) >/dev/null 2>&1
    sleep 3
else
    echo "══ 1/2 skipped (--no-isaac) ══"
fi

# ── 2. WBC task-space export + whole-body QP tracker ──────────────────────── #
echo "══ 2/2 WBC task-space export + whole-body QP tracker ══"
[[ -f "$TRAJ_JSON" ]] || { echo "❌ no traj samples under $OUT_DIR/exports (run export first)"; exit 1; }
# jvs_relative v1: absolute world-Y-up camera stream → relative 6D in the
# start optical frame, 10 Hz — the whole-body QP's trajectory_file format
python scripts/export_jarvis_trajectory.py --traj "$TRAJ_JSON" --out "$WBC_JSON"
# The old dolly demo's tighter base caps (0.4 m/s / 1.5 rad/s) map onto the
# WBC's per-input velocity_limits vector [v, ω, joint1..6]; arm joints keep
# the URDF-aligned defaults. No trigger topic — tracking starts after settle.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ros2 run jvs_mock whole_body_controller \
    --ros-args \
    --params-file "$REPO/ros2_ws/jvs_mock/config/whole_body_controller.yaml" \
    -p trajectory_file:="$WBC_JSON" \
    -p controller_mode:="${WBC_MODE:-stage1}" \
    -p output_directory:="$WBC_OUT" \
    -p "velocity_limits:=[0.4, 1.5, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5]" \
    > "$WBC_LOG" 2>&1 &
PIDS+=($!)

echo ""
echo "✅ loop up (pids: ${PIDS[*]}):"
echo "   whole-body QP ← $WBC_JSON (auto-starts after settle)"
if [[ "$NO_ISAAC" != "1" ]]; then
    echo "   Isaac GUI: third-person viewport (robot + Kid both visible)"
fi
echo "   WBC log: $WBC_LOG   telemetry: $WBC_OUT/telemetry.csv"
# wait for the completion summary (the pass runs on its own)
for _ in $(seq 1 300); do
    sleep 2
    grep -q "WBC_RESULT" "$WBC_LOG" 2>/dev/null && break
done
if grep -q "WBC_RESULT" "$WBC_LOG"; then
    echo "🎬 WBC_RESULT:"
    grep "WBC_RESULT" "$WBC_LOG" | tail -1
else
    echo "⚠️  no WBC_RESULT (timeout) — log tail:"
    tail -20 "$WBC_LOG"
fi
