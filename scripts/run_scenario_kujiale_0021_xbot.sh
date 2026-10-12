#!/usr/bin/env bash
# run_scenario_kujiale_0021_xbot.sh — Xbot walking-subject 3-shot demo
# (crane follow → pull-out → Hitchcock zoom), preset knobs for the generic
# runner. Example of scripts/run_scenario.sh for a subject scenario.
SCENARIO="kujiale_0021_xbot" \
DO_RENDER=1 \
CAMERAS="shot,overview" \
WIDTH=1280 \
HEIGHT=720 \
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_scenario.sh"
