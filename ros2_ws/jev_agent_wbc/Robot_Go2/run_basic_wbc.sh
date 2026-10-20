#!/usr/bin/env bash
# Final measured basic-control profile; original strict runner status is retained.
set -euo pipefail
ROBOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 1 && "$1" == "--preflight" ]]; then
  exec "$ROBOT_DIR/.venv/bin/python" "$ROBOT_DIR/scripts/preflight_basic_wbc.py"
fi
if [[ $# -ne 2 || "$1" != "--output" ]]; then
  echo "Usage: $0 --output NEW_DIRECTORY | --preflight" >&2
  exit 64
fi
"$ROBOT_DIR/.venv/bin/python" "$ROBOT_DIR/scripts/preflight_basic_wbc.py"
exec "$ROBOT_DIR/run_adaptive_wbc.sh" \
  --cases forward,backward,left,right,diagonal_left,diagonal_right,yaw_left,yaw_right,camera_shot \
  --move-seconds 5 --stop-seconds 5 --fine-collision \
  --go2-response-profile measured_backward_v1 --go2-native-posture x58_all_v1 \
  --go2-prediction-profile bounded_ramp_history_v2 \
  --go2-world-z-feedback bounded_integral_v1 \
  --go2-world-z-seed fresh_actual_output_continuity_v2 \
  --go2-terminal-profile endpoint_xy58_15_yaw0_v1 \
  --go2-active-stand-brake vxy_p05_v1 --camera-excursion diagnostic \
  --ground-material source_multiply --parking-anchor-reset --park-damping-scale 3 \
  --rgb --output "$2"
