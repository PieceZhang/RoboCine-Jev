#!/usr/bin/env bash
# run_scenario_kujiale_0021_sofa_trio.sh — the sofa-trio demo with all knobs
# preset. This is an EXAMPLE of how to launch scripts/run_scenario.sh for a
# concrete scenario; copy it for your own scenario.
SCENARIO="kujiale_0021_sofa_trio" \
DO_RENDER=1 \
CAMERAS="shot,overview" \
WIDTH=1280 \
HEIGHT=720 \
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_scenario.sh"
