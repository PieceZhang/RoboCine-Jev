#!/usr/bin/env bash
# drive_e2e.sh — PR #28 E2E move-compare matrix (controller + actual video)
#
# For every baseline move yml: full run_kid_combo_ros.sh chain (plan side +
# Isaac real-physics + whole-body QP controller) → the robot end-effector
# camera's ACTUAL video (ros_capture/fp_cam.mp4) + per-run step timing.
# For every fm-green move: same execution stack fed by the fm trajectory
# (FM_TRAJ entry — export/retime/live identical, only the path source
# differs). fm crane×2 (planner FAIL) and truck×2 (vocab N/A) are recorded
# as non-runnable markers.
#
# The Isaac cv2 writer leaves fp_cam.mp4 headerless when the kit is SIGINTed
# mid-write, so the canonical frame dump is the PNG sequence — actual_video
# is remuxed from fp_cam_%05d.png @30fps whenever the mp4 won't probe.
#
# Output: cinema_traj_out/move_compare_e2e/<move>/<side>/
#   actual_video.mp4  run.log  step_timing.log  status.txt
# + E2E_INDEX.md (matrix + wall-clock table).
set -uo pipefail

REPO=/home/w84384651/Documents/RoboCine-Siyu
cd "$REPO"
E2E="$REPO/cinema_traj_out/move_compare_e2e"
MOVES="dolly_in dolly_out pan_left pan_right orbit_swing crane_up crane_down static_shot zoom_in zoom_out truck_left truck_right"
FM_GREEN="dolly_in dolly_out pan_left pan_right orbit_swing static_shot zoom_in zoom_out"
mkdir -p "$E2E"

wait_isaac_gone() {   # make sure the previous sim fully exited before spawning the next
    for _ in $(seq 1 12); do
        pgrep -f "[j]vs_simview_isaac" >/dev/null || return 0
        sleep 5
    done
    pkill -9 -f "[j]vs_simview_isaac" 2>/dev/null
    sleep 5
}

collect_video() {   # collect_video <tagged_ros_capture_dir> <dest_actual_mp4>
    local cap=$1 dest=$2
    if [[ -f "$cap/fp_cam.mp4" ]] && ffprobe -v error -show_entries format=duration \
            -of csv=p=0 "$cap/fp_cam.mp4" >/dev/null 2>&1; then
        cp -f "$cap/fp_cam.mp4" "$dest"
    elif ls "$cap"/fp_cam_*.png >/dev/null 2>&1; then
        ffmpeg -y -loglevel error -framerate 30 -i "$cap/fp_cam_%05d.png" \
            -c:v libx264 -pix_fmt yuv420p "$dest" || true
    fi
}

run_one() {   # run_one <move> <side>
    local m=$1 side=$2 rc=0 wall src
    local out="$E2E/$m/$side"
    local tag="${m}__${side}"
    local tagged="$REPO/cinema_traj_out/kujiale_0021_kid_combo__${tag}"
    mkdir -p "$out"
    local t0=$EPOCHREALTIME
    echo "──── E2E $m / $side  $(date '+%T') ────"
    rm -rf "$tagged"          # clean slate: fresh ros_capture + telemetry
    if [[ "$side" == baseline ]]; then
        SCENE_YML="$REPO/cinema_traj/scenes/config/kujiale_0021_kid_${m}.yml" \
        RUN_TAG="$tag" \
            bash scripts/run_kid_combo_ros.sh >"$out/run.log" 2>&1 || rc=$?
    else
        src="$REPO/cinema_traj_out/move_compare/$m/fm_run/traj_exec_frames.json"
        [[ -f "$src" ]] || { echo "MISSING fm traj: $src" | tee "$out/status.txt"; return; }
        SCENE_YML="$REPO/cinema_traj/scenes/config/kujiale_0021_kid_${m}.yml" \
        FM_TRAJ="$src" RUN_TAG="$tag" \
            bash scripts/run_kid_combo_ros.sh >"$out/run.log" 2>&1 || rc=$?
    fi
    wall=$(awk -v a="$EPOCHREALTIME" -v b="$t0" 'BEGIN{printf "%.1f", a-b}')
    collect_video "$tagged/ros_capture" "$out/actual_video.mp4"
    rm -f "$tagged"/ros_capture/fp_cam_*.png   # PNGs remuxed (or run failed) — reclaim GBs
    cp -f "$tagged/step_timing.log" "$out/step_timing.log" 2>/dev/null || true
    {
        echo "move=$m side=$side rc=$rc wall_s=$wall"
        echo "video=$([[ -f $out/actual_video.mp4 ]] && echo YES || echo NO)"
        echo "wbc_pass=$(grep -c 'PASS — every shooting segment' "$out/run.log" 2>/dev/null || true)"
    } > "$out/status.txt"
    echo "──── done $m / $side rc=$rc wall=${wall}s video=$([[ -f $out/actual_video.mp4 ]] && echo YES || echo NO)"
    wait_isaac_gone
}

# fm non-runnable markers (round-2 planner state, documented not retried)
for m in crane_up crane_down; do
    mkdir -p "$E2E/$m/fm"
    echo "fm planner FAIL (height-window dead zone; round-2) — no executable trajectory" \
        > "$E2E/$m/fm/status.txt"
done
for m in truck_left truck_right; do
    mkdir -p "$E2E/$m/fm"
    echo "fm vocab N/A (no lateral_truck type; round-2) — no executable trajectory" \
        > "$E2E/$m/fm/status.txt"
done

for m in $MOVES;  do run_one "$m" baseline; done
for m in $FM_GREEN; do run_one "$m" fm; done

python3 "$REPO/scripts/move_compare/build_e2e_index.py"
echo "══ E2E sweep complete → $E2E/E2E_INDEX.md ══"
