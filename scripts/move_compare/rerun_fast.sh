#!/bin/bash
# move_compare: re-run the planner on every per-move yml via tools/plan_fast.py
# (single-process fast path, PR #34) and write per-variant logs with an
# E2E_WALL= marker so build_timing_table.py can consume them.
#
# Usage:  scripts/move_compare/rerun_fast.sh            (from anywhere)
# Env:    ROBOCINE_PY  python to use (default: robocine_plan conda env)
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1
D=cinema_traj_out/move_compare
PY=${ROBOCINE_PY:-$HOME/miniconda3/envs/robocine_plan/bin/python}

echo "=== FAST RE-RUN ($(date +%F\ %H:%M:%S)) ==="
for Y in cinema_traj/scenes/config/kujiale_0021_kid_*.yml; do
  b=$(basename "$Y" .yml)                       # kujiale_0021_kid_<move>
  v=${b#kujiale_0021_kid_}                      # <move>
  [ "$v" = "combo" ] && continue                # multi-move, not in matrix
  [ "$v" = "dolly" ] && continue                # superseded by dolly_in/out
  /usr/bin/time -f "E2E_WALL=%es" "$PY" tools/plan_fast.py "$Y" \
      > "$D/${v}.fast.plan.log" 2>&1
  e2e=$(grep -oE "E2E_WALL=[0-9.]+s" "$D/${v}.fast.plan.log" | grep -oE "[0-9.]+")
  marks=$(grep -cE "⏱ \[fast" "$D/${v}.fast.plan.log")
  ok=$(grep -c "check_trackable ✅" "$D/${v}.fast.plan.log")
  echo "$v: E2E=${e2e}s marks=$marks trackable=$ok"
done
echo "=== done; run scripts/move_compare/build_timing_table.py to refresh the table ==="
