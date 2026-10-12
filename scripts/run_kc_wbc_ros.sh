#!/usr/bin/env bash
# Start the two independent Stage 3 KC-WBC launch groups in the same ROS domain.
# Usage: scripts/run_kc_wbc_ros.sh [--trajectory-file FILE] [--stage FILE]
#        [--domain ID] [--headless] [--output-directory DIR]
#        [--mesh-source groundtruth|lidar] [--mesh-cache FILE] [--force-mesh-rebuild]
#        [--x M] [--z M] [--yaw RAD] [--spawn-height M]
# Defaults: shot_combo1.json trajectory, groundtruth mesh source
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trajectory="$repo/ros2_ws/jvs_mock/trajectories/shot_combo1.json"
stage="$repo/cinema_traj_out/kujiale_0021_kid_combo/simview/simview.usda"
domain=176
headless=false
output_directory=""
mesh_cache=""
mesh_source=groundtruth
force_mesh_rebuild=false
x=5.0 z=1.0 yaw=0.0 spawn_height=0.002

usage() {
    sed -n '2,7p' "$0" | sed 's/^# *//'
    echo 'Example: scripts/run_kc_wbc_ros.sh --trajectory-file ros2_ws/jvs_mock/trajectories/shot_combo1.json'
    echo 'LiDAR:   scripts/run_kc_wbc_ros.sh --mesh-source lidar --headless'
}

while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --headless) headless=true; shift ;;
        --force-mesh-rebuild) force_mesh_rebuild=true; shift ;;
        --trajectory-file|--trajectory|--stage|--domain|--output-directory|--mesh-source|--mesh-cache|--x|--z|--yaw|--spawn-height)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --trajectory-file|--trajectory) trajectory=$2 ;;
                --stage) stage=$2 ;;
                --domain) domain=$2 ;;
                --output-directory) output_directory=$2 ;;
                --mesh-cache) mesh_cache=$2 ;;
                --mesh-source) mesh_source=$2 ;;
                --x) x=$2 ;;
                --z) z=$2 ;;
                --yaw) yaw=$2 ;;
                --spawn-height) spawn_height=$2 ;;
            esac
            shift 2 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$domain" =~ ^[0-9]+$ ]] && ((10#$domain <= 232)) || {
    echo "ROS domain must be an integer from 0 to 232" >&2; exit 2;
}
domain=$((10#$domain))
[[ "$mesh_source" == groundtruth || "$mesh_source" == lidar ]] || {
    echo "mesh source must be groundtruth or lidar" >&2; exit 2;
}
[[ -f "$trajectory" ]] || { echo "Trajectory not found: $trajectory" >&2; exit 1; }
[[ -f "$stage" ]] || { echo "Stage not found: $stage" >&2; exit 1; }
[[ -f /opt/ros/humble/setup.bash && -f "$repo/ros2_ws/install/setup.bash" ]] || {
    echo "ROS Humble or ros2_ws/install is missing; build the workspace first" >&2; exit 1;
}
trajectory="$(realpath "$trajectory")"
stage="$(realpath "$stage")"
echo "Trajectory: $trajectory"
echo "Mesh source: $mesh_source"

# ROS setup scripts may read variables that are not defined yet.
set +u
source /opt/ros/humble/setup.bash
source "$repo/ros2_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID="$domain" OMNI_KIT_ACCEPT_EULA=yes
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

common=(ros_domain_id:="$domain" stage_view:="$stage" trajectory_file:="$trajectory"
        x:="$x" z:="$z" yaw:="$yaw" spawn_height:="$spawn_height")
isaac=("${common[@]}" headless:="$headless")
if [[ "$mesh_source" == lidar ]]; then
    isaac+=(enable_lidar:=true)
fi
follow=("${common[@]}" mesh_source:="$mesh_source" force_mesh_rebuild:="$force_mesh_rebuild")
[[ -z "$output_directory" ]] || follow+=(output_directory:="$output_directory")
[[ -z "$mesh_cache" ]] || follow+=(mesh_cache_file:="$mesh_cache")

log_dir="$(mktemp -d /tmp/run_kc_wbc_ros.XXXXXX)"
isaac_pid=""
follow_pid=""
cleanup() {
    trap - EXIT INT TERM
    # Stop the command publisher first, then the simulator.
    if [[ -n "$follow_pid" ]] && kill -0 "$follow_pid" 2>/dev/null; then
        kill -TERM "$follow_pid" 2>/dev/null || true
        wait "$follow_pid" 2>/dev/null || true
    fi
    if [[ -n "$isaac_pid" ]] && kill -0 "$isaac_pid" 2>/dev/null; then
        kill -TERM "$isaac_pid" 2>/dev/null || true
        wait "$isaac_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Isaac log: $log_dir/isaac.log"
ros2 launch jvs_mock jvs_kc_wbc_isaac.launch.py "${isaac[@]}" \
    >"$log_dir/isaac.log" 2>&1 &
isaac_pid=$!

# A cold Isaac start can take longer than trajfollow's reset timeout.
# Check this launch's own log so an older node in the domain cannot satisfy readiness.
deadline=$((SECONDS + 180))
until grep -Fq 'JVS simview ready:' "$log_dir/isaac.log"; do
    if ! kill -0 "$isaac_pid" 2>/dev/null; then
        echo "Isaac exited before its reset service appeared:" >&2
        tail -40 "$log_dir/isaac.log" >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "Timed out waiting for Isaac readiness; see $log_dir/isaac.log" >&2
        exit 1
    fi
    sleep 2
done

echo "Isaac ready; starting KC-WBC trajectory follow"
ros2 launch jvs_mock jvs_kc_wbc_trajfollow.launch.py "${follow[@]}" &
follow_pid=$!
echo "Press Ctrl-C to stop trajfollow, then Isaac."
finished_pid=""
if wait -n -p finished_pid "$follow_pid" "$isaac_pid"; then
    status=0
else
    status=$?
fi
if [[ "$finished_pid" == "$isaac_pid" ]]; then
    echo "Isaac exited before trajectory follow finished; see $log_dir/isaac.log" >&2
    exit 1
fi
exit "$status"
