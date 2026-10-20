#!/usr/bin/env bash
# Reproducible native OCS2 build; no changes to the Isaac Python environment.
set -eo pipefail
ocs2_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache_dir="${JEV_OCS2_CACHE:-$ocs2_dir/.cache}"
source_dir="${OCS2_SOURCE_ROOT:-$cache_dir/source}"
prefix_dir="${JEV_OCS2_PREFIX:-$cache_dir/prefix}"
revision=88ce706c17fcc4591b6c24c87a4a85ff6ed9e92d
mode="${1:---mpc}"
if [[ "$mode" != --mpc && "$mode" != --dynamics-only ]]; then
  echo 'usage: build_native.sh [--mpc|--dynamics-only]' >&2; exit 2
fi
mkdir -p "$cache_dir/logs"
if [[ ! -d "$source_dir/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/leggedrobotics/ocs2.git "$source_dir"
  git -C "$source_dir" checkout --detach "$revision"
fi
python3 - "$source_dir" "$ocs2_dir/UPSTREAM_MANIFEST.json" <<'VERIFY'
import hashlib,json,pathlib,subprocess,sys
source=pathlib.Path(sys.argv[1]);manifest=json.loads(pathlib.Path(sys.argv[2]).read_text())
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==manifest['revision'], 'Unpinned OCS2 source'
subprocess.run(['git','-C',str(source),'diff','--exit-code','HEAD','--'],check=True,stdout=subprocess.DEVNULL)
for item in manifest['files']:
    assert hashlib.sha256((source/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
VERIFY
if [[ "$mode" == --mpc ]]; then
  ros_setup="${JEV_ROS_SETUP:-/opt/ros/humble/setup.bash}"
  [[ -f "$ros_setup" ]] || { echo 'Set JEV_ROS_SETUP to a compatible installed ROS2 setup.' >&2; exit 2; }
  source "$ros_setup"
  export CMAKE_PREFIX_PATH="$prefix_dir${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  packages=(ocs2_thirdparty ocs2_core ocs2_oc ocs2_mpc ocs2_test_tools/ocs2_qp_solver ocs2_ddp)
  for package in "${packages[@]}"; do
    package_name="${package##*/}"
    {
      cmake -S "$source_dir/$package" -B "$cache_dir/build/$package_name" \
        -DCMAKE_INSTALL_PREFIX="$prefix_dir" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF -DCMAKE_CXX_FLAGS='-include cstddef' \
        -DPython3_EXECUTABLE=/usr/bin/python3
      cmake --build "$cache_dir/build/$package_name" -j "${JEV_BUILD_JOBS:-2}"
      cmake --install "$cache_dir/build/$package_name"
    } 2>&1 | tee "$cache_dir/logs/$package_name.log"
  done
fi
{
  cmake -S "$ocs2_dir" -B "$cache_dir/backend" \
    -DOCS2_SOURCE_ROOT="$source_dir" -DCMAKE_BUILD_TYPE=Release \
    -DJEV_BUILD_MPC="$([[ "$mode" == --mpc ]] && echo ON || echo OFF)" \
    -DCMAKE_CXX_FLAGS='-include cstddef' -DPython3_EXECUTABLE=/usr/bin/python3
  cmake --build "$cache_dir/backend" -j "${JEV_BUILD_JOBS:-2}"
  ctest --test-dir "$cache_dir/backend" --output-on-failure
} 2>&1 | tee "$cache_dir/logs/backend.log"
echo "Native executables: $cache_dir/backend"
