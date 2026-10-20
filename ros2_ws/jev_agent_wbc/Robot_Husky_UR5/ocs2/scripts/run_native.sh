#!/usr/bin/env bash
set -eo pipefail
ocs2_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache_dir="${JEV_OCS2_CACHE:-$ocs2_dir/.cache}"
binary="${JEV_OCS2_BINARY:-$cache_dir/backend/camera_mpc_probe}"
[[ -x "$binary" ]] || { echo 'First run ocs2/scripts/build_native.sh --mpc.' >&2; exit 2; }
# Optional args are a validated internal request and an output policy path.
# This command never connects to PhysX or issues physical commands.
exec "$binary" "$ocs2_dir/backend.json" "$ocs2_dir/model.json" "$@"
