#!/bin/bash
# Driver v3: 12 moves; fm renders via build-with-cast (fixes yellow capsule)
# + render without --rebuild; truck rows fm=N/A.
SIYU=/home/w84384651/Documents/RoboCine-Siyu
FM=/home/w84384651/Documents/RoboCine-fm
CFG=$SIYU/cinema_traj/scenes/config
BOUT=$SIYU/cinema_traj_out/kujiale_0021_kid_combo
FMOUT=$FM/cinema_traj_out/kujiale_0021_kid_combo
CMP=$SIYU/cinema_traj_out/move_compare
PY_ISAAC=/home/w84384651/miniconda3/envs/env_isaaclab/bin/python3.11
PY_PLAN=/home/w84384651/miniconda3/envs/robocine_plan/bin/python3.11
MOVES="${MOVES:-dolly_in dolly_out pan_left pan_right orbit_swing crane_up crane_down static_shot zoom_in zoom_out truck_left truck_right}"
FM_SKIP="truck_left truck_right"
mkdir -p "$CMP"

baseline_side() {
  local m=$1 D=$CMP/$1 ok=1 TS got
  rm -f "$D/baseline/FAILED"
  ( cd "$SIYU" && SCENE_YML="$CFG/kujiale_0021_kid_$m.yml" \
      bash scripts/run_kid_combo_ros.sh --plan-only > "$D.baseline.plan.log" 2>&1 ) || ok=0
  grep -q "plan-only done" "$D.baseline.plan.log" 2>/dev/null || ok=0
  mkdir -p "$D/baseline"
  if [ "$ok" != 1 ]; then echo "  BASELINE FAIL"; echo FAIL > "$D/baseline/FAILED"; return; fi
  cp "$BOUT/viewer/kujiale_0021_kid_combo.html" "$D/baseline/viewer.html"
  TS=$(date +%s)
  ( cd "$SIYU" && ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y "$PY_ISAAC" \
      tools/simview_isaac.py render "$BOUT/traj_final.json" \
      --fps 10 --every 0.2 --video-fps 10 --mp4 --width 1280 --height 720 \
      --cameras shot,overview > "$D.baseline.render.log" 2>&1 )
  got=0
  for f in $(find "$BOUT/simview/video" -name "ShotCam_*.mp4" -newermt "@$TS"); do cp "$f" "$D/baseline/shot.mp4"; got=1; break; done
  for f in $(find "$BOUT/simview/video" -name "OverviewCam_*.mp4" -newermt "@$TS"); do cp "$f" "$D/baseline/overview.mp4"; break; done
  [ "$got" = 1 ] && echo "  baseline ok" || { echo "  BASELINE RENDER FAIL"; echo FAIL > "$D/baseline/FAILED"; }
}

fm_side() {
  local m=$1 D=$CMP/$1 TS got
  rm -rf "$D/fm_run"; mkdir -p "$D/fm"; rm -f "$D/fm/FAILED"
  case " $FM_SKIP " in *" $m "*) echo "  fm N/A (vocab)"; echo "N/A" > "$D/fm/UNSUPPORTED"; return;; esac
  ( cd "$FM" && "$PY_PLAN" -m cinema_traj.fm.plan \
      --scene cinema_traj/scenes/kujiale_0021_kid_combo.json \
      --intent "/tmp/fm_intents/$m.json" \
      --out "$D/fm_run" > "$D.fm.plan.log" 2>&1 ) || { echo "  FM PLAN FAIL"; echo FAIL > "$D/fm/FAILED"; return; }
  [ -s "$D/fm_run/exports/traj_samples_retimed.json" ] || { echo "  FM NO RETIMED"; echo FAIL > "$D/fm/FAILED"; return; }
  python3 /tmp/fm_to_frames.py "$D/fm_run/exports/traj_samples_retimed.json" \
      "$D/fm_run/traj_exec_frames.json" > /dev/null || { echo "  FM CONVERT FAIL"; echo FAIL > "$D/fm/FAILED"; return; }
  cp "$D/fm_run/traj_exec_frames.json" "$FMOUT/traj_final.json"
  ( cd "$FM" && "$PY_PLAN" run_traj.py --stage view \
      --scene cinema_traj/scenes/kujiale_0021_kid_combo.json >> "$D.fm.plan.log" 2>&1 )
  cp "$FMOUT/viewer/kujiale_0021_kid_combo.html" "$D/fm/viewer.html" 2>/dev/null || echo "  FM HTML MISSING"
  # build the stage WITH the cast (fixes the yellow-capsule subject) then render
  ( cd "$FM" && "$PY_ISAAC" tools/simview_isaac.py build \
      --traj "$FMOUT/traj_final.json" \
      --scene cinema_traj/scenes/kujiale_0021_kid_combo.json \
      --cast "$FMOUT/cast.json" --subject-glb none >> "$D.fm.plan.log" 2>&1 ) || echo "  FM BUILD WARN"
  TS=$(date +%s)
  ( cd "$FM" && ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y "$PY_ISAAC" \
      tools/simview_isaac.py render "$FMOUT/traj_final.json" \
      --fps 10 --every 0.2 --video-fps 10 --mp4 --width 1280 --height 720 \
      --cameras shot,overview > "$D.fm.render.log" 2>&1 )
  got=0
  for f in $(find "$FMOUT/simview/video" -name "ShotCam_*.mp4" -newermt "@$TS"); do cp "$f" "$D/fm/shot.mp4"; got=1; break; done
  for f in $(find "$FMOUT/simview/video" -name "OverviewCam_*.mp4" -newermt "@$TS"); do cp "$f" "$D/fm/overview.mp4"; break; done
  [ "$got" = 1 ] && echo "  fm ok" || { echo "  FM RENDER FAIL"; echo FAIL > "$D/fm/FAILED"; }
}

for m in $MOVES; do
  echo "========== $m ========== $(date +%T)"
  rm -rf "$CMP/$m"
  baseline_side "$m"
  fm_side "$m"
done
echo "========== DRIVER v3 DONE ========== $(date +%T)"
