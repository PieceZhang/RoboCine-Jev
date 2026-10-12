#!/usr/bin/env bash
# run_scenario_kujiale_0021_kid_dolly.sh — single-beat demo: level frontal
# close-up on the SEATED Kid cast figure, then a straight move_out dolly
# (camera pulls back along the Kid's facing axis toward the TV).
#
# The Kid IS the subject: the scene copy kujiale_0021_kid_dolly.json carries
# a static subject_track pinned to her seat, and simview is built with
# --subject-glb none so no Xbot stand-in is baked over the cast figure.
#
# After the pipeline finishes, open the Isaac Sim GUI (needs DISPLAY):
#   DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=yes python tools/simview_isaac.py gui \
#       cinema_traj_out/kujiale_0021_kid_dolly/simview/simview.usda
#   (viewport camera dropdown → /World/ShotCam for POV; timeline: bottom bar)
SCENARIO="kujiale_0021_kid_dolly" \
DO_RENDER=0 \
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_scenario.sh"
