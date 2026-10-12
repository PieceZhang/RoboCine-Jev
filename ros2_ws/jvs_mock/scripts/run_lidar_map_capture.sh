#!/usr/bin/env bash
# Invoke from any directory; requires this worktree's colcon install.
set -eo pipefail
package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(dirname "$package_dir")"
source /opt/ros/humble/setup.bash
source "$workspace_dir/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:?Set an unused nonzero ROS_DOMAIN_ID before capture}"
export ROS_LOCALHOST_ONLY=1
export FASTRTPS_DEFAULT_PROFILES_FILE="$package_dir/config/lidar_test_loopback.xml"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR="$workspace_dir/log/lidar_ros"
export GZ_PARTITION="lidar_map_${ROS_DOMAIN_ID}"
export XDG_CACHE_HOME="$workspace_dir/log/lidar_xdg/cache"
export XDG_CONFIG_HOME="$workspace_dir/log/lidar_xdg/config"
export XDG_DATA_HOME="$workspace_dir/log/lidar_xdg/data"
export CUDA_CACHE_PATH="$workspace_dir/log/lidar_cuda"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
exec timeout --signal=INT --kill-after=10s 600s /usr/bin/python3 \
  "$package_dir/scripts/jvs_isaac_bootstrap.py" \
  --isaac-python "${ISAAC_PYTHON:?Set ISAAC_PYTHON to the Isaac Sim Python executable}" \
  --runner "$package_dir/scripts/capture_lidar_map.py" -- \
  --xacro "$package_dir/urdf/jvs_mock.urdf.xacro" "$@"
