#!/usr/bin/env bash
# Build SAM 2.1 using only the active sam2 environment's CUDA tools and headers.
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "sam2" ]]; then
  echo "Activate the sam2 Conda environment before running this script." >&2
  exit 1
fi

sam2_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="${CONDA_PREFIX:?Activate the sam2 Conda environment}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export SAM2_BUILD_ALLOW_ERRORS=0
export MAX_JOBS="${MAX_JOBS:-2}"

# The Conda CUDA package selects GCC 14, but PyTorch's CUDA 12.8 build requires GCC <14.
# Use Ubuntu's supported compiler for this build, without changing either toolchain.
export CC="${SAM2_CC:-/usr/bin/gcc}"
export CXX="${SAM2_CXX:-/usr/bin/g++}"
export NVCC_PREPEND_FLAGS="-ccbin=${CXX}"

# Conda keeps SDK headers under targets/; PyTorch's NVIDIA wheels supply math headers.
sam2_header_paths="$(python - <<'PY'
from pathlib import Path
import os
import sys
import sysconfig
paths = [Path(sys.prefix) / 'targets/x86_64-linux/include']
paths.extend(sorted((Path(sysconfig.get_path('purelib')) / 'nvidia').glob('*/include')))
print(os.pathsep.join(str(path) for path in paths if path.is_dir()))
PY
)"
export CPATH="${sam2_header_paths}${CPATH:+:${CPATH}}"

python -m pip install --no-build-isolation -e "${sam2_repo_root}/sam21" \
  'matplotlib>=3.9.1' 'opencv-python>=4.7.0' 'eva-decord>=0.6.1' "$@"
