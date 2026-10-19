#!/usr/bin/env bash
set -euo pipefail
ROBOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROBOT_DIR"
unset PYTHONPATH
export MPLCONFIGDIR="$ROBOT_DIR/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y WANDB_MODE=disabled
ROBOT_RESULT_DIR="$ROBOT_DIR/results/$(date +%Y%m%d_%H%M%S)_$$"
ROBOT_ARGS=("$@")
for ((ROBOT_ARG_INDEX=0; ROBOT_ARG_INDEX<${#ROBOT_ARGS[@]}; ROBOT_ARG_INDEX++)); do
  case "${ROBOT_ARGS[ROBOT_ARG_INDEX]}" in
    --output)
      if ((ROBOT_ARG_INDEX+1>=${#ROBOT_ARGS[@]})) || [[ -z "${ROBOT_ARGS[ROBOT_ARG_INDEX+1]}" || "${ROBOT_ARGS[ROBOT_ARG_INDEX+1]}" == --* ]]; then
        echo "--output requires a new result directory path" >&2; exit 2
      fi
      ROBOT_RESULT_DIR="${ROBOT_ARGS[ROBOT_ARG_INDEX+1]}";;
    --output=*)
      ROBOT_RESULT_DIR="${ROBOT_ARGS[ROBOT_ARG_INDEX]#--output=}"
      if [[ -z "$ROBOT_RESULT_DIR" ]]; then echo "--output requires a new result directory path" >&2; exit 2; fi;;
    --help|-h) exec "$ROBOT_DIR/.venv/bin/python" "$ROBOT_DIR/scripts/validate_locomotion.py" --help;;
  esac
done
exec "$ROBOT_DIR/.venv/bin/python" "$ROBOT_DIR/scripts/supervise.py" --result "$ROBOT_RESULT_DIR" -- "$ROBOT_DIR/.venv/bin/python" -u "$ROBOT_DIR/scripts/validate_locomotion.py" "$@" --output "$ROBOT_RESULT_DIR"
