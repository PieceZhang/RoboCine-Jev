#!/usr/bin/env bash
set -euo pipefail
ROBOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROBOT_DIR"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/w84384651/miniconda3/envs/env_isaaclab/bin/python}"
mkdir -p upstream assets policies config results
if [[ ! -d upstream/WBC-AGILE/.git ]]; then git clone --filter=blob:none https://github.com/nvidia-isaac/WBC-AGILE.git upstream/WBC-AGILE; fi
git -C upstream/WBC-AGILE checkout --detach ad4ff6ee6b38da207592b3e71dcee4c118049f6d
if [[ ! -d upstream/IsaacLab/.git ]]; then git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git upstream/IsaacLab; fi
[[ "$(git -C upstream/IsaacLab rev-parse HEAD)" == 37ddf626871758333d6ed89cf64ad702aef127d0 ]]
apply_metadata_patch() {
    local directory="$1" patchfile="$2"
    if git -C "$directory" apply --check "$patchfile" 2>/dev/null; then
        git -C "$directory" apply "$patchfile"
    elif ! git -C "$directory" apply --reverse --check "$patchfile" 2>/dev/null; then
        echo "Local metadata differs from recorded compatibility patch: $patchfile" >&2; exit 1
    fi
}
apply_metadata_patch upstream/IsaacLab "$ROBOT_DIR/patches/isaaclab_sim51_metadata.patch"
apply_metadata_patch upstream/WBC-AGILE "$ROBOT_DIR/patches/agile_sim51_metadata.patch"
if [[ ! -x .venv/bin/python ]]; then "$ISAAC_PYTHON" -m venv --system-site-packages .venv; fi
.venv/bin/python -m pip install --cache-dir .cache/pip --find-links .cache/wheels -c config/constraints.txt wheel==0.45.1 \
  -e upstream/IsaacLab/source/isaaclab -e upstream/IsaacLab/source/isaaclab_tasks \
  -e upstream/IsaacLab/source/isaaclab_assets -e upstream/IsaacLab/source/isaaclab_rl \
  -e upstream/WBC-AGILE
.venv/bin/python -m pip install --no-deps --no-build-isolation -e upstream/WBC-AGILE/agile/algorithms/rsl_rl
.venv/bin/python scripts/download_assets.py
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > config/environment.freeze.txt
