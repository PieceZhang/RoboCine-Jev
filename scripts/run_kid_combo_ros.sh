#!/usr/bin/env bash
# run_kid_combo_ros.sh — GENERIC END-TO-END one-command driver:
#
#   planner stages → simview stage → Isaac real-physics (PR #24 passive
#   casters — real wheels is the ONLY mode) + whole-body QP tracker
#   (jvs_mock whole_body_controller) → per-segment tracking-error stats
#   (WBC telemetry) + the robot's END-EFFECTOR CAMERA actual video
#   (ros_capture/fp_cam.mp4). Every step is timed; a summary table prints
#   at the end and appends to <OUT_DIR>/step_timing.log.
#
# Generic entry (PR #28 move-compare matrix / any single-move yml):
#   SCENE_YML=<yml>   which scenario beats to plan (default: kid combo)
#   RUN_TAG=<id>      isolates this run's outputs in
#                     cinema_traj_out/kujiale_0021_kid_combo__<id>/
#                     (ros_capture actual video, wbc telemetry, timing).
#                     Unset → the canonical shared dir (original behavior).
#   FM_TRAJ=<json>    fm/planner (PR #28) trajectory in traj_final schema:
#                     skips scene/subject/plan stages and runs export →
#                     retime → shotlist → view → simview → LIVE side on the
#                     SAME execution stack (apples-to-apples vs baseline).
#   WBC_MODE / WBC_OBSTACLE / PLANNER_ENV — pass-through (see below).
#
# Sequence (see cinema_traj/scenes/config/kujiale_0021_kid_combo.yml):
#   MOVE TO START (spawn @ plan start, ALIGN/ARMED)        [not shooting]
#   → orbit_quarter  90° around the STANDING Kid            [SHOOT]
#   → arc_1 transition                                      [not shooting]
#   → pan_left       ENE anchor, 45° pan                    [SHOOT]
#   → arc_2 dwell                                           [not shooting]
#   → move_out       NE pull-back                           [SHOOT]
#   → HOLD                                                  [not shooting]
#
# Steps [1..8] = planner side. They mirror the frl_apartment demo chain
# (extract → add_subject_track → recommend → build/optimize/export/retime/
# view) mapped onto this scenario's inputs:
#   ① scene      kujiale_0021_kid_combo.json is a COMMITTED refined scene —
#                re-extracting would only produce a draft that overwrites
#                manual curation, so this step verifies presence instead
#                (same idea as frl's FileExistsError guard; there is no
#                --map ply source for the kujiale copy).
#   ② subject    the kid subject_track (standing at 6.0, 0.5) lives in the
#                scene json — verified, not re-injected. To rebuild a
#                walking subject instead:
#                  python scripts/add_subject_track.py \
#                      --scene cinema_traj/scenes/kujiale_0021_kid_combo.json \
#                      --waypoints "..." --speed 0.6 --height 1.68
#   ③ plan       scenario yml beats are this scenario's "recommend":
#                  python tools/scenario_gen.py <yml>
#                (plan.json → build → optimize → cast.json → clearance
#                audit; the frl chain's LLM `--stage recommend` is the
#                other entry into the same build/optimize stages)
#   ④⑤⑥⑦        run_traj export → retime (jarvis) → shotlist → view
#                (viewer html to eyeball /cam_desired_traj offline)
#   ⑧            simview stage build (Isaac USD, standing Kid, no Xbot)
# Steps [9..12] = live side: spawn from the FRESH configs[0] (never
# hardcoded), export the WBC task-space stream (jvs_relative v1), single
# auto-started pass — the whole-body QP tracks after its 2 s settle (it has
# no trigger topic), HOLD on WBC_RESULT, telemetry error table, actual
# first-person video in ros_capture/fp_cam.mp4.
#
# NOTE on paths: run_traj's export/retime/view stages key their I/O off the
# SCENE stem (cinema_traj_out/kujiale_0021_kid_combo/) — the plan side
# therefore always runs in that SHARED dir and the artifacts are snapshotted
# into OUT_DIR before the live side, so successive tagged runs never
# clobber each other.
#
# Usage:
#   bash scripts/run_kid_combo_ros.sh --gui        # full chain, Isaac GUI
#   bash scripts/run_kid_combo_ros.sh              # full chain, headless+FP record
#   bash scripts/run_kid_combo_ros.sh --plan-only  # steps 1-8 only (no robot)
#   bash scripts/run_kid_combo_ros.sh --skip-plan  # reuse exports, live side only
#   bash scripts/run_kid_combo_ros.sh --auto --gui # same auto-started pass (flag kept for compat)
#   # PR #28 matrix single run (baseline):
#   SCENE_YML=cinema_traj/scenes/config/kujiale_0021_kid_dolly_in.yml \
#       RUN_TAG=dolly_in__baseline bash scripts/run_kid_combo_ros.sh
#   # PR #28 matrix single run (fm side — external trajectory, same stack):
#   FM_TRAJ=<traj_final-schema.json> RUN_TAG=dolly_in__fm \
#       bash scripts/run_kid_combo_ros.sh
#   flags: --gui --auto --no-record --no-isaac --plan-only --skip-plan
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_JSON="$REPO/cinema_traj/scenes/kujiale_0021_kid_combo.json"
# per-move comparison runs override SCENE_YML (PR #28 test matrix);
# default stays the canonical combo yml
SCENE_YML="${SCENE_YML:-$REPO/cinema_traj/scenes/config/kujiale_0021_kid_combo.yml}"
# generic-run isolation (see header): FM_TRAJ replaces the plan stages with
# an externally-produced trajectory (fm/planner); RUN_TAG snapshots plan
# artifacts + keeps every live output in its own directory.
RUN_TAG="${RUN_TAG:-}"
FM_TRAJ="${FM_TRAJ:-}"
SCENARIO="kujiale_0021_kid_combo"                 # run_traj I/O is scene-keyed
SHARED_DIR="$REPO/cinema_traj_out/$SCENARIO"      # plan side runs here
OUT_DIR="${OUT_DIR:-$SHARED_DIR${RUN_TAG:+__$RUN_TAG}}"
mkdir -p "$OUT_DIR"
# plan-side artifact paths (always the SHARED scene dir — see header NOTE)
PLAN_TRAJ="$SHARED_DIR/exports/traj_samples_retimed.json"
PLAN_FF="$SHARED_DIR/exports/jarvis_traj.json"
PLAN_SHOTLIST="$SHARED_DIR/exports/shotlist.json"
# live-side artifact paths (isolated per run when RUN_TAG is set)
TRAJ_JSON="$OUT_DIR/exports/traj_samples_retimed.json"
FF_JSON="$OUT_DIR/exports/jarvis_traj.json"
WBC_JSON="$OUT_DIR/exports/jarvis_task_space.json"   # jvs_relative v1 → WBC
WBC_OUT="$OUT_DIR/wbc_metrics"                       # WBC telemetry dir
SHOTLIST="$OUT_DIR/exports/shotlist.json"
STAGE_USDA="$OUT_DIR/simview/simview.usda"
RECORD_DIR="$OUT_DIR/ros_capture"                    # robot FP camera video
ERR_CSV="$WBC_OUT/telemetry.csv"
WBC_LOG=/tmp/robocine_${RUN_TAG:-combo}_wbc.log
ISAAC_LOG=/tmp/robocine_${RUN_TAG:-combo}_isaac.log
NO_ISAAC=0 AUTO=0 NO_RECORD=0 GUI=0 PLAN_ONLY=0 SKIP_PLAN=0
for a in "$@"; do
    case "$a" in
        --no-isaac) NO_ISAAC=1 ;;
        --auto) AUTO=1 ;;
        --no-record) NO_RECORD=1 ;;
        --gui) GUI=1 ;;
        --plan-only) PLAN_ONLY=1 ;;
        --skip-plan) SKIP_PLAN=1 ;;
        --real-wheels) echo "note: real wheels is the default now (kinematic bypass removed)" ;;
        *) echo "unknown flag: $a (see header)"; exit 2 ;;
    esac
done

# ── timing helper ─────────────────────────────────────────────────────────── #
RUN_T0=$EPOCHREALTIME
STEP_TOTAL=12
declare -a STEP_NAMES=() STEP_TIMES=()
_step_t0=$EPOCHREALTIME
_step_done() {  # _step_done <slot "n" or ""> <name>
    local t
    t=$(awk -v a="$EPOCHREALTIME" -v b="$_step_t0" 'BEGIN{printf "%.1f", a-b}')
    STEP_NAMES+=("$2${1:+ }"); STEP_TIMES+=("$t")
    echo "⏱️  ${1:+[$1/$STEP_TOTAL] }$2: ${t}s"
}
step() {  # step <n> <name> <cmd...> — run, time, log
    local n="$1" name="$2"; shift 2
    echo ""
    echo "══ [$n/$STEP_TOTAL] $name ══"
    _step_t0=$EPOCHREALTIME
    "$@"
    _step_done "$n" "$name"
}
step_skip() {  # step_skip <n> <name> <reason>
    echo ""
    echo "══ [$1/$STEP_TOTAL] $2 ══"
    echo "   skipped — $3"
    STEP_NAMES+=("$2 (skip)"); STEP_TIMES+=("0.0")
}
TIMING_LOG="$OUT_DIR/step_timing.log"
timing_summary() {
    echo ""
    echo "══ step timing summary ══"
    local i total
    for i in "${!STEP_NAMES[@]}"; do
        printf "   ⏱️  %-36s %8ss\n" "${STEP_NAMES[$i]}" "${STEP_TIMES[$i]}"
    done
    total=$(awk -v a="$EPOCHREALTIME" -v b="$RUN_T0" 'BEGIN{printf "%.1f", a-b}')
    # persist: one timestamped block per run (args + every step) so
    # successive runs can be compared without scraping stdout
    {
        echo "── $(date '+%F %T') tag: ${RUN_TAG:-<shared>} args: $* ──"
        for i in "${!STEP_NAMES[@]}"; do
            printf "%-38s %8ss\n" "${STEP_NAMES[$i]}" "${STEP_TIMES[$i]}"
        done
        printf "%-38s %8ss\n" "E2E TOTAL WALL" "$total"
    } >> "$TIMING_LOG"
    echo "   📝 appended → $TIMING_LOG"
    echo ""
    echo "══ E2E total wall: ${total}s → $OUT_DIR ══"
    if [[ -f "$RECORD_DIR/fp_cam.mp4" ]]; then
        echo "🎬 actual video (robot end-effector camera): $RECORD_DIR/fp_cam.mp4"
    fi
}

# ── ROS 2 Humble + workspace ──────────────────────────────────────────────── #
set +u
source /opt/ros/humble/setup.bash
source "$REPO/ros2_ws/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# ── Isaac Sim python env (planner stages need torch; the sim needs the kit) ─ #
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PATH="/opt/ros/humble/bin:$PATH"
command -v xacro >/dev/null || { echo "❌ xacro still not on PATH"; exit 1; }
set -u
export OMNI_KIT_ACCEPT_EULA=yes
export DISPLAY="${DISPLAY:-:1}"
export ROBOCINE_REPO="$REPO"     # tracking node's cinema_traj import path
export JVS_TRACK_DEBUG="${JVS_TRACK_DEBUG:-0}"  # 1 = TRACK u/v_ff debug lines

# ── planner env override (steps ③-⑦ only: torch+numpy+yaml, no Isaac deps) ── #
# The batched IK gate runs fine in env_isaaclab (CPU batching is the win), but
# the dedicated robocine_plan env carries a cu128 torch that actually executes
# on Blackwell (sm_120) — env_isaaclab's cu126 build cannot launch kernels
# there. Set PLANNER_ENV to override/skip; steps ⑧+ always stay env_isaaclab.
PLAN_PY=~/miniconda3/envs/env_isaaclab/bin/python
if [[ -z "${PLANNER_ENV:-}" && -x ~/miniconda3/envs/robocine_plan/bin/python ]]; then
    PLANNER_ENV=robocine_plan
fi
if [[ -n "${PLANNER_ENV:-}" && "$PLANNER_ENV" != "env_isaaclab" ]]; then
    if ~/miniconda3/envs/"$PLANNER_ENV"/bin/python -c "import torch, numpy, yaml" 2>/dev/null; then
        PLAN_PY=~/miniconda3/envs/"$PLANNER_ENV"/bin/python
        echo "🧠 planner env: $PLANNER_ENV ($(PLAN_PY="$PLAN_PY" "$PLAN_PY" -c \
            'import cinema_traj.core.jarvis_batched as jb; print(jb.pick_device())' 2>/dev/null || echo cpu))"
    else
        echo "⚠️  PLANNER_ENV=$PLANNER_ENV missing torch/numpy/yaml — falling back to env_isaaclab"
    fi
fi

# ════════════════════════════════════════════════════════════════════════════
# PLANNER SIDE (steps 1-8)
# ════════════════════════════════════════════════════════════════════════════
if [[ "$SKIP_PLAN" != "1" ]]; then

    if [[ -n "$FM_TRAJ" ]]; then
        # fm/planner entry (PR #28): skip ①②③, seed the SHARED traj_final
        # with the external trajectory, run the SAME export→retime→…→live
        # stack. cast.json is scene-level and shared (same committed scene).
        [[ -f "$FM_TRAJ" ]] || { echo "❌ FM_TRAJ not found: $FM_TRAJ"; exit 1; }
        [[ -f "$SHARED_DIR/cast.json" ]] || { echo "❌ $SHARED_DIR/cast.json missing (run one baseline first)"; exit 1; }
        step_skip 1 "scene (committed)" "fm entry — external trajectory ($FM_TRAJ)"
        step_skip 2 "subject track" "fm entry"
        step_skip 3 "plan+build+optimize+audit" "fm entry — copying FM trajectory into the shared plan dir"
        cp -f "$FM_TRAJ" "$SHARED_DIR/traj_final.json"
        echo "   fm trajectory → $SHARED_DIR/traj_final.json"
    else

    # ① scene — committed refined scene; presence check only (re-extract
    #    would overwrite manual curation with a draft — frl chain's
    #    FileExistsError guard, same reasoning)
    if [[ -f "$SCENE_JSON" ]]; then
        step_skip 1 "scene (committed)" \
            "kujiale_0021_kid_combo.json exists (refined, committed). To redraft a scene from raw data use the extract stage manually — not wired here on purpose."
    else
        echo "❌ $SCENE_JSON missing — this scenario requires the committed scene copy"; exit 1
    fi

    # ② subject track — the kid (standing, static) is baked into the scene
    _step_t0=$EPOCHREALTIME
    echo ""
    echo "══ [2/$STEP_TOTAL] subject track ══"
    python - "$SCENE_JSON" <<'PY' || { echo "❌ scene subject_track check failed"; exit 1; }
import json, sys
d = json.load(open(sys.argv[1]))
st = d.get("subject_track")
assert st and st.get("frames"), "no subject_track in scene"
f0 = st["frames"][0]["pos"]
print(f"   subject '{st['subject_id']}' @ ({f0[0]:.2f}, {f0[1]:.2f}, {f0[2]:.2f}), "
      f"{len(st['frames'])} frame(s), heading {st['frames'][0].get('heading', 0):.2f} rad")
PY
    _step_done 2 "subject track (in-scene)"

    # ③ plan → build → optimize → cast → audit (scenario yml beats == the
    #    "recommend" of the frl chain, semantic entry)
    step 3 "plan+build+optimize+audit" \
        "$PLAN_PY" tools/scenario_gen.py "$SCENE_YML"

    fi   # FM_TRAJ entry joins here with traj_final.json in place

    # ④ export (camera stream) — auto-triggers jarvis retime on overspeed
    step 4 "export" \
        "$PLAN_PY" run_traj.py --stage export --scene "$SCENE_JSON"

    # ⑤ retime (jarvis platform: base/arm config stream + trackability proof)
    step 5 "retime (jarvis)" \
        "$PLAN_PY" run_traj.py --stage retime --scene "$SCENE_JSON" --force

    # ⑥ shotlist + dwell-ff repair — the shotlist feeds the stats step; the
    #    dwell repair splices base interpolation across allow_unreachable
    #    arc holds (frozen base = zero feedforward = the robot never
    #    relocates during the transition; measured 100cm+ without, R29)
    _step_t0=$EPOCHREALTIME
    echo ""
    echo "══ [6/$STEP_TOTAL] shotlist + dwell-ff fix ══"
    "$PLAN_PY" scripts/fix_dwell_arcs.py "$PLAN_FF"
    "$PLAN_PY" - "$PLAN_TRAJ" "$PLAN_SHOTLIST" <<'PY'
import json, sys
samples = json.load(open(sys.argv[1]))["samples"]
runs = []
for s in samples:
    lab = s.get("segment") or "?"
    if runs and runs[-1]["segment"] == lab:
        runs[-1]["t_end"] = s["t"]
    else:
        runs.append({"segment": lab, "t_start": s["t"], "t_end": s["t"]})
shots = []
for i, r in enumerate(runs):
    seg = [s for s in samples if r["t_start"] <= s["t"] <= r["t_end"]]
    shots.append({"index": i + 1, "segment": r["segment"],
                  "shooting": not r["segment"].startswith("arc"),
                  "t_start_s": round(r["t_start"], 2), "t_end_s": round(r["t_end"], 2),
                  "duration_s": round(r["t_end"] - r["t_start"], 2),
                  "cam_start_m": [round(seg[0]["x"], 2), round(seg[0]["y"], 2), round(seg[0]["z"], 2)],
                  "cam_end_m": [round(seg[-1]["x"], 2), round(seg[-1]["y"], 2), round(seg[-1]["z"], 2)]})
doc = {"_comment": "regenerated by run_kid_combo_ros.sh step 6 — shooting vs "
                   "non-shooting arcs; times = retimed executable stream",
       "duration_s": shots[-1]["t_end_s"] if shots else 0.0,
       "total_shooting_s": round(sum(s["duration_s"] for s in shots if s["shooting"]), 2),
       "shots": shots}
json.dump(doc, open(sys.argv[2], "w"), indent=1)
for s in shots:
    print(f'   {"SHOOT" if s["shooting"] else "  arc"} {s["segment"]:14s} '
          f't {s["t_start_s"]:6.2f} → {s["t_end_s"]:6.2f}  ({s["duration_s"]:5.2f}s)')
PY
    _step_done 6 "shotlist + dwell-ff fix"

    # ⑦ view — offline self-contained HTML viewer of the trajectory
    step 7 "view (html)" \
        "$PLAN_PY" run_traj.py --stage view --scene "$SCENE_JSON"

    # snapshot plan artifacts into OUT_DIR (no-op for the untagged shared
    # run) so tagged runs keep their own copies and the live side below is
    # fully isolated from the next run's plan side
    if [[ "$OUT_DIR" != "$SHARED_DIR" ]]; then
        mkdir -p "$OUT_DIR/exports" "$OUT_DIR/viewer"
        cp -f "$SHARED_DIR/traj_final.json" "$OUT_DIR/"
        cp -f "$SHARED_DIR/cast.json" "$OUT_DIR/"
        cp -f "$PLAN_TRAJ" "$PLAN_FF" "$PLAN_SHOTLIST" \
              "$SHARED_DIR/exports/traj_samples.json" "$OUT_DIR/exports/"
        cp -f "$SHARED_DIR"/viewer/*.html "$OUT_DIR/viewer/" 2>/dev/null || true
    fi

    # ⑧ simview stage build — Isaac USD with the standing Kid cast
    step 8 "simview build" \
        python tools/simview_isaac.py build \
            --traj "$OUT_DIR/traj_final.json" --scene "$SCENE_JSON" \
            --cast "$OUT_DIR/cast.json" --subject-glb none --fps 10
fi

if [[ "$PLAN_ONLY" == "1" ]]; then
    timing_summary
    echo ""
    echo "✅ --plan-only done → $OUT_DIR"
    exit 0
fi

# ════════════════════════════════════════════════════════════════════════════
# LIVE SIDE (steps 9-12) — real wheels, no kinematic bypass (PR #24)
# ════════════════════════════════════════════════════════════════════════════
# ── pre-flight cleanup: stray WBC/tracker/traj_server/Isaac from prior runs ── #
# (a surviving command source keeps publishing /isaac_joint_commands or
#  /cmd_vel and poisons the fresh sim: measured — it dragged a just-spawned
#  robot 1.3 m off its park and baked a 1.19 m anchor into the next run's
#  ALIGN; two command sources also interleave telemetry writes)
for _pat in "[w]hole_body_controller" "[t]racking_cam_desired" "[c]inema_traj_bridge" "[j]vs_simview_isaac" "[k]in_base_integrator"; do
    pkill -9 -f "$_pat" 2>/dev/null || true
done
sleep 3

PIDS=()
trap 'kill ${PIDS[@]:-} 2>/dev/null' EXIT INT TERM

# spawn pose + arm pre-pose come from the FRESH configs[0] (never hardcoded)
read -r SPAWN_X SPAWN_Z SPAWN_YAW Q1 Q2 Q3 Q4 Q5 Q6 < <(python - "$FF_JSON" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["configs"][0]
print(c["base"][0], c["base"][1], c["base"][2], *c["q"])
PY
)
Q0_CSV="$(printf '%s, %s, %s, %s, %s, %s' "$Q1" "$Q2" "$Q3" "$Q4" "$Q5" "$Q6")"
echo "spawn from configs[0]: x=$SPAWN_X z=$SPAWN_Z yaw=$SPAWN_YAW  q=[$Q0_CSV]"

# keep the previous measurement (mv, never overwrite live data)
if [[ -f "$ERR_CSV" ]]; then
    mv "$ERR_CSV" "$ERR_CSV.prev"
    echo "(previous err csv → $ERR_CSV.prev)"
fi

if [[ "$NO_ISAAC" != "1" ]]; then
    _step_t0=$EPOCHREALTIME
    echo ""
    echo "══ [9/$STEP_TOTAL] isaac backend (real wheels) ══"
    if [[ "$GUI" == "1" ]]; then
        echo "    ⚠️  GUI runs the sim at a fraction of real-time — tracking stays"
        echo "    correct (sim-clock playhead) but the pass takes minutes of wall"
        echo "    clock. First-person recording is forced OFF (drags sim below 5%)."
        NO_RECORD=1
    fi
    # add_ground_plane:=false is LOAD-BEARING in BOTH modes: the Isaac
    # groundPlane (z=0) z-fights the simview scene's own floor mesh — the
    # contact pair makes the wheels slip (5-14% commanded speed executed;
    # 92-105% without it)
    RECORD_ARGS=()
    if [[ "$NO_RECORD" != "1" ]]; then
        mkdir -p "$RECORD_DIR"
        RECORD_ARGS=(record_dir:="$RECORD_DIR" record_fps:=30.0)
    fi
    HEADLESS=true; [[ "$GUI" == "1" ]] && HEADLESS=false
    # third person in GUI: keep Isaac's free roaming viewport camera —
    # activate_camera:=true would switch the viewport to the robot's
    # ONBOARD first-person camera instead. headless ignores this flag.
    GUI_ACTIVE=false; [[ "$GUI" != "1" ]] && GUI_ACTIVE=true
    ros2 launch jvs_mock jvs_mock_simview.launch.py \
        simview_stage:="$STAGE_USDA" \
        x:="$SPAWN_X" z:="$SPAWN_Z" yaw:="$SPAWN_YAW" spawn_height:=0.002 \
        activate_camera:="$GUI_ACTIVE" start_trajectory_follower:=false \
        start_command_adapter:=false \
        headless:="$HEADLESS" add_ground_plane:=false \
        "${RECORD_ARGS[@]}" \
        > "$ISAAC_LOG" 2>&1 &
    PIDS+=($!)
    for _ in $(seq 1 60); do
        sleep 2
        grep -q "JVS simview ready" "$ISAAC_LOG" 2>/dev/null && break
    done
    grep -q "JVS simview ready" "$ISAAC_LOG" || {
        echo "❌ Isaac backend never became ready:"; tail -5 "$ISAAC_LOG"; exit 1; }
    # two-step arm pre-pose (joint5 travels far; the position drive stalls
    # if handed the full span at once): generic halfway, then configs[0].q
    (ros2 topic pub --rate 20 /arm_position_controller/commands \
        std_msgs/msg/Float64MultiArray \
        "data: [0.005, -0.09, 1.2, 0.0, -1.2, 0.0]" & PP=$!
     sleep 8; kill $PP) >/dev/null 2>&1
    (ros2 topic pub --rate 20 /arm_position_controller/commands \
        std_msgs/msg/Float64MultiArray \
        "data: [$Q0_CSV]" & PP=$!
     sleep 22; kill $PP) >/dev/null 2>&1
    sleep 3
    _step_done 9 "isaac backend (ready)"
else
    step_skip 9 "isaac backend" "--no-isaac"
fi

# ⑩ WBC task-space stream + whole-body QP tracker
_step_t0=$EPOCHREALTIME
echo ""
echo "══ [10/$STEP_TOTAL] WBC task-space export + whole-body QP tracker ══"
[[ -f "$TRAJ_JSON" ]] || { echo "❌ $TRAJ_JSON missing (run the plan side first)"; exit 1; }
# jvs_relative v1: absolute world-Y-up camera stream → relative 6D in the
# start optical frame, 10 Hz — the whole-body QP's trajectory_file format
"$PLAN_PY" scripts/export_jarvis_trajectory.py \
    --traj "$TRAJ_JSON" --out "$WBC_JSON"
# no trigger topic: the WBC starts tracking settle_s (2 s) after synchronized
# /joint_states + /base_controller/odom arrive. jarvis_traj.json stays the
# spawn/pre-pose + fix_dwell_arcs source (planner config stream).
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ros2 run jvs_mock whole_body_controller \
    --ros-args \
    --params-file "$REPO/ros2_ws/jvs_mock/config/whole_body_controller.yaml" \
    -p trajectory_file:="$WBC_JSON" \
    -p controller_mode:="${WBC_MODE:-stage1}" \
    -p output_directory:="$WBC_OUT" \
    ${WBC_OBSTACLE:+-p obstacle_file:="$WBC_OBSTACLE"} \
    > "$WBC_LOG" 2>&1 &
PIDS+=($!)
_step_done 10 "WBC task-space export + tracker"

echo ""
echo "✅ loop up (pids: ${PIDS[*]}):"
echo "   whole-body QP ← $WBC_JSON (auto-starts after settle)"
python - "$SHOTLIST" <<'PY'
import json, sys
sl = json.load(open(sys.argv[1]))
print("   shots:", " → ".join(
    f'{s["segment"]}{"*" if s["shooting"] else ""}' for s in sl["shots"]))
PY
[[ "$NO_ISAAC" != "1" && "$NO_RECORD" != "1" ]] && echo "   FP recording → $RECORD_DIR"
echo "   WBC log: $WBC_LOG   telemetry: $ERR_CSV"

# ⑪ the single tracking pass — the WBC has no trigger topic; the pass runs
#    as soon as the node settles (step 10), so both modes just watch for
#    the WBC_RESULT completion summary.
_step_t0=$EPOCHREALTIME
echo ""
echo "══ [11/$STEP_TOTAL] tracking pass (auto-started by the WBC) ══"
STARTED=0
for _ in $(seq 1 60); do
    sleep 2
    grep -q "whole-body trajectory" "$WBC_LOG" 2>/dev/null && { STARTED=1; break; }
done
[[ "$STARTED" == "1" ]] || { echo "❌ WBC never started tracking (see $WBC_LOG)"; tail -20 "$WBC_LOG"; exit 1; }
echo "── WBC tracking (GUI / FP recording may slow the sim below real-time) ──"
for _ in $(seq 1 300); do
    sleep 2
    grep -q "WBC_RESULT" "$WBC_LOG" 2>/dev/null && break
done
if grep -q "WBC_RESULT" "$WBC_LOG"; then
    echo "🎬 WBC_RESULT — HOLDing 15s of tail footage, then closing"
else
    echo "⚠️  no WBC_RESULT in the WBC log — closing anyway"
fi
sleep 15   # HOLD tail in the recording; SIGINT then lets Isaac mux the mp4
kill -INT ${PIDS[@]:-} 2>/dev/null
sleep 20   # Isaac shutdown + cv2 mux
echo "WBC tail:"
tail -8 "$WBC_LOG"
_step_done 11 "tracking pass (wall)"

# ⑫ per-segment error stats: WBC telemetry × shotlist → mean/max cm per
#    segment; pass bar = <3cm mean on every SHOOTING segment
echo ""
echo "══ [12/$STEP_TOTAL] per-segment tracking error ══"
PASS_BAR=0.03 python - "$ERR_CSV" "$SHOTLIST" <<'PY' || true
import csv, json, os, sys
import numpy as np
rows = list(csv.DictReader(open(sys.argv[1])))
shots = json.load(open(sys.argv[2]))["shots"]
bar = float(os.environ.get("PASS_BAR", "0.03"))
if not rows:
    print("   ⚠️ no telemetry rows recorded"); sys.exit(0)
ts = np.array([float(r["time_s"]) for r in rows])
ep = np.array([float(r["position_error"]) for r in rows])
print(f"   {'seg':16s} {'kind':6s} {'n':>5s} {'mean':>8s} {'max':>8s}  verdict")
ok = True
for s in shots:
    m = (ts >= s["t_start_s"]) & (ts <= s["t_end_s"])
    if not m.any():
        print(f"   {s['segment']:16s} {('SHOOT' if s['shooting'] else 'arc'):6s}"
              f"    —       —        —      (no data)")
        continue
    mean, mx = ep[m].mean() * 100, ep[m].max() * 100
    if s["shooting"]:
        good = mean < bar * 100
        ok &= good
        verdict = f"{'✅' if good else '❌'} <3cm mean" + ("" if good else f" (max {mx:.1f}cm)")
    else:
        verdict = "  (not shooting)"
    print(f"   {s['segment']:16s} {('SHOOT' if s['shooting'] else 'arc'):6s} {m.sum():5d} "
          f"{mean:7.1f}cm {mx:7.1f}cm  {verdict}")
print("")
print("   ✅ PASS — every shooting segment <3cm mean" if ok
      else "   ❌ FAIL — a shooting segment ≥3cm mean (iterate: tracker params / plan durations)")
PY

timing_summary
