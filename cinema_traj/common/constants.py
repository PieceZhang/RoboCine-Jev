"""Cinematography toolset constants (paper §3.2, §4.1.1, appendix D/E)."""

from __future__ import annotations

import math

# 13 paper movement types + 3 subject-relative extension primitives (ours)
OBJECT_TRAJ_TYPES = [
    "orbit_full", "orbit_half", "orbit_quarter",
    "pan_left", "pan_right", "tilt_up", "tilt_down",
    "move_in", "move_out", "crane", "zoom_in_out", "zoom_out_in",
    "static", "arc_swing", "lateral_truck",
]
SUBJECT_TRAJ_TYPES = ["track_follow", "track_orbit", "track_lead"]
TRANSITION_TRAJ_TYPES = ["arc"]
ALL_TRAJ_TYPES = OBJECT_TRAJ_TYPES + SUBJECT_TRAJ_TYPES + TRANSITION_TRAJ_TYPES

ORBIT_SWEEP_DEG = {"orbit_full": 360.0, "orbit_half": 180.0, "orbit_quarter": 90.0}

# Placement → forbidden object-level movement types (paper appendix E.2)
PLACEMENT_FORBIDDEN = {
    "wall": {"orbit_full", "orbit_half"},
    "ceiling": {"crane"},
    "floor_wall": {"orbit_full"},
    "freestanding": set(),
}

# LLM viewing preferences → numeric ranges (paper appendix D.3)
ELEVATION_BANDS_DEG = {
    "low": (10.0, 20.0),
    "medium": (25.0, 35.0),
    "high": (40.0, 55.0),
    "overhead": (60.0, 80.0),
}
DISTANCE_MULTIPLIERS = {"close": 0.7, "medium": 1.0, "far": 1.4}

# Anchor scoring weights (paper §4.1.1): wall-adjacent vs freestanding
ANCHOR_WEIGHTS = {
    "wall":       {"vis": 0.35, "dist": 0.20, "elev": 0.10, "open": 0.10, "perp": 0.25},
    "freestanding": {"vis": 0.45, "dist": 0.25, "elev": 0.15, "open": 0.15, "perp": 0.0},
}

# Anchor candidate sampling (paper appendix D.3)
DISTANCE_FACTORS = (0.5, 0.7, 1.0, 1.3)
AZIMUTH_OFFSETS_COUNT = 10          # uniformly spaced within ±45° of face normal
AZIMUTH_OFFSET_SPAN_DEG = 45.0
ELEVATIONS_PER_BAND = 3
ANCHOR_COLLISION_MARGIN = 0.10      # m, SDF clearance required for a candidate
SCENE_BOUNDS_MARGIN = 0.30          # m, candidates must stay inside room bounds
VIEWING_DIST_SCALE = 1.2            # viewing distance = median extent × 1.2
VIEWING_DIST_CLAMP = (0.3, 3.0)     # m
VIEWING_DIST_ROOM_FRAC = 0.40       # ≤ 40% of smallest horizontal room dim
TOP_K_ANCHORS = 3                   # restart candidates kept for the optimizer

# Door passage preference (paper appendix F.2): doorframe/archway open < door
DOOR_PREF = {"doorframe": 0.0, "archway": 0.0, "doorway": 0.0, "door": 2.0}
TRANSITION_FRAMES = 30              # frames per cross-room sub-arc

# Default durations (s) per movement — LLM may override
DEFAULT_DURATION = 6.0
ZOOM_M_PEAK = 2.0                   # peak focal multiplier for zoom shots
DEFAULT_FPS = 10.0

DEG = 180.0 / math.pi
RAD = math.pi / 180.0
