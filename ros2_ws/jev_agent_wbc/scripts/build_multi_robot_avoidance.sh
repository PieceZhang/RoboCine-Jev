#!/usr/bin/env bash
# Reuse the deployed pinned OCS2 dependencies; build only our native adapters.
# No downloads or modifications to the historical backend cache.
set -euo pipefail
package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${OCS2_SOURCE_ROOT:-/tmp/jev-wbc-ocs2-official}"
prefix_dir="${JEV_OCS2_PREFIX:-/tmp/jev-ocs2-wheel-runtime/prefix}"
ros_setup="${JEV_ROS_SETUP:-/opt/ros/humble/setup.bash}"
[[ -f "$ros_setup" ]] || { echo 'Set JEV_ROS_SETUP to the deployed ROS2 setup.' >&2; exit 2; }
[[ -f "$prefix_dir/share/ocs2_ddp/cmake/ocs2_ddpConfig.cmake" ]] || {
  echo 'Set JEV_OCS2_PREFIX to the deployed pinned OCS2 installation.' >&2; exit 2;
}
# The upstream ROS setup references unset shell variables.
set +u
source "$ros_setup"
set -u
export CMAKE_PREFIX_PATH="$prefix_dir${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
for robot in Robot_Husky_UR5 Robot_Ridgeback_UR5; do
  native="$package/$robot/ocs2"
  build_cache="$native/.cache/avoidance-build"
  python3 - "$source_dir" "$native/UPSTREAM_MANIFEST.json" <<'VERIFY'
import hashlib,json,pathlib,subprocess,sys
source=pathlib.Path(sys.argv[1]);manifest=json.loads(pathlib.Path(sys.argv[2]).read_text())
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==manifest['revision'],'Unpinned OCS2 source'
subprocess.run(['git','-C',str(source),'diff','--exit-code','HEAD','--'],check=True,stdout=subprocess.DEVNULL)
for item in manifest['files']:
    assert hashlib.sha256((source/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
VERIFY
  cmake -S "$native" -B "$build_cache/backend" \
    -DOCS2_SOURCE_ROOT="$source_dir" -DCMAKE_BUILD_TYPE=Release \
    -DJEV_BUILD_MPC=ON -DCMAKE_CXX_FLAGS='-include cstddef' \
    -DPython3_EXECUTABLE=/usr/bin/python3
  cmake --build "$build_cache/backend" -j "${JEV_BUILD_JOBS:-2}"
  ctest --test-dir "$build_cache/backend" --output-on-failure
  mkdir -p "$native/.cache/avoidance"
  # Rename complete files atomically; a running process retains its old inode.
  for binary in camera_mpc_worker libmeasured_collision_guard.so; do
    cp "$build_cache/backend/$binary" "$native/.cache/avoidance/$binary.new"
    mv "$native/.cache/avoidance/$binary.new" "$native/.cache/avoidance/$binary"
  done
done
