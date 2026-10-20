#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# != 1 ]]; then
  echo 'Usage: bootstrap.sh /absolute/path/to/python3.11' >&2
  exit 2
fi
"$1" -c 'import sys; assert sys.version_info >= (3,11), "Current pinned upstream uses enum.StrEnum (Python3.11+)"'
if [[ -e "$task_dir/.venv" ]]; then
  echo 'Isolated .venv already exists; refusing to modify it. Use a separate fresh copy for a rebuild.' >&2
  exit 2
fi
"$1" "$task_dir/fetch_sources.py"
"$1" -m venv "$task_dir/.venv"
env -u PIP_EXTRA_INDEX_URL -u PIP_INDEX_URL PIP_CONFIG_FILE=/dev/null \
  "$task_dir/.venv/bin/python" -m pip install --no-cache-dir --index-url https://pypi.org/simple \
  -r "$task_dir/requirements.lock.txt"
"$task_dir/.venv/bin/python" -m pip check
echo 'Core-component environment prepared. Full World/Executor and physical execution are separate readiness gates.'
