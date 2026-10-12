#!/usr/bin/env bash
# run_scenario.sh — generic scenario demo runner: pipeline camera trajectory +
# Isaac Sim render. ALL camera geometry is computed by the cinema_traj pipeline
# (plan → build → optimize); nothing here is hand-authored.
#
# Steps:
#   1. tools/scenario_gen.py     scenario yml → plan.json → build → optimize
#                                → cast.json → clearance audit (hard assert)
#   2. tools/simview_isaac.py build    author the USDA stage (pure python)
#   3. tools/simview_isaac.py render   render passes → PNG frames (old frames
#                                      purged first) → timestamped mp4s in
#                                      <out>/simview/video/
#
# Usage:
#   bash scripts/run_scenario.sh                 # uses SCENARIO env (set by a
#                                                # per-scenario wrapper script)
#   SCENARIO=<name> bash scripts/run_scenario.sh # direct
#   DO_RENDER=0  → trajectory only (fast intent iteration, skips Isaac)
#   CAMERAS="shot,overview" | "shot" | "overview"
#
# A scenario = two files:
#   cinema_traj/scenes/<scenario>.json         scene (staged people MUST also
#                                              exist as person_* obstacle OBBs,
#                                              else the optimiser flies through
#                                              them)
#   cinema_traj/scenes/config/<scenario>.yml   staging (cast/props) + camera
#                                              intent (beats) + audit bounds
# See scripts/run_scenario_kujiale_0021_sofa_trio.sh for a preset example.
set -euo pipefail

# ── configuration comes from environment (set by a wrapper script or you) ──
SCENARIO="${SCENARIO:?set SCENARIO (see scripts/run_scenario_*.sh)}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
FPS="${FPS:-10}"                    # cinema_traj DEFAULT_FPS (traj & video fps)
DO_RENDER="${DO_RENDER:-1}"         # 1 = full render; 0 = trajectory only
CAMERAS="${CAMERAS:-shot,overview}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
# ── derived paths ───────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_JSON="$REPO/cinema_traj/scenes/${SCENARIO}.json"
SCENARIO_YML="$REPO/cinema_traj/scenes/config/${SCENARIO}.yml"
OUT_DIR="$REPO/cinema_traj_out/${SCENARIO}"
TRAJ_JSON="$OUT_DIR/traj_final.json"
CAST_JSON="$OUT_DIR/cast.json"

cd "$REPO"
set +u  # conda's activate.d scripts reference unbound vars
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
set -u
export OMNI_KIT_ACCEPT_EULA=yes

echo "══ 1/3 scenario: plan → build → optimize → cast → audit ══"
python tools/scenario_gen.py "$SCENARIO_YML"

if [[ "$DO_RENDER" != "1" ]]; then
    echo "DO_RENDER=0 → done. trajectory: $TRAJ_JSON"
    exit 0
fi

echo "══ 2/3 simview: author USDA stage ══"
CAST_ARG=()
[[ -f "$CAST_JSON" ]] && CAST_ARG=(--cast "$CAST_JSON")   # subject scenarios
python tools/simview_isaac.py build --traj "$TRAJ_JSON" --scene "$SCENE_JSON" \
    "${CAST_ARG[@]}" --fps "$FPS"

echo "══ 3/3 simview: headless render ($CAMERAS) → mp4 ══"
python tools/simview_isaac.py render "$TRAJ_JSON" --fps "$FPS" \
    --every "$(python -c "print(1.0/$FPS)")" --video-fps "$FPS" --mp4 \
    --width "$WIDTH" --height "$HEIGHT" --cameras "$CAMERAS"

echo "✅ $SCENARIO done → $OUT_DIR  (videos in $OUT_DIR/simview/video/)"
