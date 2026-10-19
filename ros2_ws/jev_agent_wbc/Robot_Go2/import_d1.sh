#!/usr/bin/env bash
set -euo pipefail
ROBOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$ROBOT_DIR/.cache"
unset PYTHONPATH
mkdir -p "$ROBOT_DIR/results"
IMPORT_STAMP="$(mktemp "$ROBOT_DIR/results/d1_import.XXXXXX")"
trap 'rm -f "$IMPORT_STAMP"' EXIT
"$ROBOT_DIR/.venv/bin/python" "$ROBOT_DIR/scripts/import_d1.py" "$@"
REPORT_PATH="$ROBOT_DIR/assets/generated/go2_d1/import_report.json"
if [[ ! -f "$REPORT_PATH" || ! "$REPORT_PATH" -nt "$IMPORT_STAMP" ]]; then
  echo "D1 import did not create a fresh conversion report." >&2
  exit 1
fi
"$ROBOT_DIR/.venv/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("conversion_pass") else 2)' "$REPORT_PATH"
