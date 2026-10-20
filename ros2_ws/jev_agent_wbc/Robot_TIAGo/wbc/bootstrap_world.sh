#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# != 2 ]]; then
  echo 'Usage: bootstrap_world.sh /absolute/path/to/python3.11 /absolute/path/to/NEW-isolated-venv' >&2
  exit 2
fi
"$1" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use Python3.11 for this frozen dependency closure"'
if [[ -e "$2" ]]; then
  echo 'Refusing to modify an existing environment. Supply a new isolated directory.' >&2
  exit 2
fi
"$1" "$task_dir/fetch_sources.py"
"$1" -m venv "$2"
env -u PIP_EXTRA_INDEX_URL -u PIP_INDEX_URL PIP_CONFIG_FILE=/dev/null \
  "$2/bin/python" -m pip install --no-cache-dir --index-url https://pypi.org/simple \
  -r "$task_dir/requirements.world.lock.txt"
"$2/bin/python" -m pip check
echo 'Isolated full-framework dependencies installed. Physical motion/feedback/avoidance gates require measured acceptance.'
