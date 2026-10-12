#!/usr/bin/env python3
"""Round-2 fm intents: 12 moves; truck -> fm unsupported (no type in fm vocab);
orbit -> single pan_left beat (fm has no buildable single-segment orbit;
user wants ONE continuous smooth segment)."""
import json
from pathlib import Path

OUT = Path("/tmp/fm_intents")
OUT.mkdir(exist_ok=True)

MOVES = {
    "dolly_in":    ("move_in",     None, 6.0, "dolly in push"),
    "dolly_out":   ("move_out",    None, 6.0, "dolly out pull back"),
    "pan_left":    ("pan_left",    30.0, 6.0, "pan left"),
    "pan_right":   ("pan_right",   30.0, 6.0, "pan right"),
    "orbit_swing": ("pan_left",    20.0, 6.0, "orbit (fm single-segment pan form)"),
    "crane_up":    ("crane",       None, 6.0, "crane up"),
    "crane_down":  ("crane",       None, 6.0, "crane down"),
    "static_shot": ("static",      None, 6.0, "static shot"),
    "zoom_in":     ("zoom_in_out", None, 6.0, "zoom in"),
    "zoom_out":    ("zoom_out_in", None, 6.0, "zoom out"),
}

for name, (ttype, sweep, dur, prompt) in MOVES.items():
    traj = {"type": ttype, "duration_s": dur}
    if sweep is not None:
        traj["sweep_deg"] = sweep
    intent = {
        "target": "kid",
        "prompt": f"single-move camera test: {prompt}",
        "beats": [{
            "anchor": {"elevation": "medium", "distance": "far",
                       "start_angle_deg": 0, "azimuth_deg": -15},
            "traj": traj,
        }],
    }
    (OUT / f"{name}.json").write_text(json.dumps(intent, indent=1))
    print(f"intent {name}: {ttype}")
print("truck_left/truck_right: fm vocab has NO truck type -> N/A rows")
