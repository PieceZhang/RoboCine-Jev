#!/usr/bin/env bash
# Basic-control profile: validated fresh measured-state gravity, original Ki1.
# Strict historical acceptance is still emitted unchanged by the native runner.
set -euo pipefail
ROBOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROBOT_DIR/run_adaptive_wbc.sh" \
  --cases forward,backward,left,right,diagonal_left,diagonal_right,yaw_left,yaw_right,camera_shot \
  --move-seconds 5 --stop-seconds 5 \
  --upper-mode bounded_integral --park-damping-scale 3 \
  --parking-profile brake_settle_v1 --park-posture-mode measured \
  --waist-posture-cost .15 --fine-collision \
  --g1-gravity-profile actual_physx_v1 --ground-material source_multiply --rgb \
  --upper-damping-scale 3 --g1-arm-stiffness-scale 2 \
  --g1-collision-gain .4 --g1-velocity-profile yaw_phase_symmetric2_v1 \
  --g1-park-yaw-assist --g1-refresh-posture-on-park --terminal-damping-cost .03 \
  --g1-gravity-hold-profile fresh_current_state_v1 "$@"
