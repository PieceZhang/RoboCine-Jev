"""Phase-1 whole-body reachability checker (layered plan, step 1).

The optimizer plans the CAMERA as a point; this checker asks whether the
physical robot — wheeled base + mast + 6-axis arm + tablet camera — can
actually place the camera at every sampled pose:

  1. camera height ∈ the reachable band [mast_top − arm_reach … mast_top +
     arm_reach] intersected with "above the base footprint";
  2. SOME base placement exists: a candidate chassis center b (sampled on a
     ring around the camera's ground projection at geometrically feasible
     standoffs) such that
       a. |cam − (b + arm_base_offset)| ∈ [min_reach, max_reach]  (arm solves),
       b. the base+mast cylinder at b clears the scene SDF by ≥ base radius
     (base placement is chosen per frame — continuity/joint feasibility is
     Phase 2, the KC-SVSDF-style coupled-gradient whole-body optimizer);
  3. look direction is assumed achievable (spherical-wrist assumption —
     ``camera.look_any_direction``), reported as an assumption, not checked.

Failure reporting mirrors ``stages.optimize.verify_collision_free``: per
segment, worst frame, human-readable reason (height band / arm reach / base
placement with nearest obstacle). Robot dimensions come from
``cinema_traj.robot`` in config.yaml (see config.example.yaml) with the
defaults below as fallback — MEASURE THE REAL ROBOT and override.

Phase 2 roadmap (documented so it survives context loss): add the whole-body
layer — variables base SE(2) + 6 joints, objective ‖FK(q(t)) − x_ref(t)‖² +
collision + smoothness, gradients propagated along the kinematic chain
(Jacobian-transpose, KC-SVSDF §back-end); this checker then becomes the
front-end feasibility oracle for that optimizer.
"""

from __future__ import annotations

import numpy as np

from ..common import constants as C
from ..common.scene import Scene
from ..common.sdf import SceneSDF
from .trajectories import Segment

# --------------------------------------------------------------------------- #
# Robot model
# --------------------------------------------------------------------------- #
DEFAULT_ROBOT: dict = {
    "base": {"radius": 0.30, "height": 0.40},
    "mast": {"height": 0.55, "half_xy": 0.10},
    "arm": {"max_reach": 0.65, "min_reach": 0.15,
            "base_offset": [0.0, 0.95, 0.0]},
    "camera": {"height_range": None, "look_any_direction": True},
}


class RobotModel:
    """Geometric embodiment summary (Phase 1: placement-level, no joints)."""

    def __init__(self, d: dict | None = None):
        cfg = _deep_merge(DEFAULT_ROBOT, d or {})

        self.base_radius = float(cfg["base"]["radius"])
        self.base_height = float(cfg["base"]["height"])
        self.mast_height = float(cfg["mast"]["height"])
        self.arm_base = np.asarray(cfg["arm"]["base_offset"], dtype=np.float64)
        self.max_reach = float(cfg["arm"]["max_reach"])
        self.min_reach = float(cfg["arm"]["min_reach"])
        hr = cfg["camera"].get("height_range")
        if hr:
            self.height_range = (float(hr[0]), float(hr[1]))
        else:
            arm_top = float(self.arm_base[1]) + self.max_reach
            arm_bottom = max(float(self.arm_base[1]) - self.max_reach,
                             self.base_height + 0.15)
            self.height_range = (arm_bottom, arm_top)
        self.look_any_direction = bool(cfg["camera"]["look_any_direction"])
        # body probe heights for placement checks: chassis TOP (probing lower
        # would false-positive against the floor the robot stands on), mast
        # mid, arm base
        self.body_heights = np.array(sorted({
            self.base_height,
            self.base_height + 0.5 * self.mast_height,
            float(self.arm_base[1]),
        }))

    # -- geometry ---------------------------------------------------------- #
    def reach_ring(self, cam_y: float) -> tuple[float, float] | None:
        """Horizontal standoff interval [r_lo, r_hi] from the camera's ground
        projection to a chassis center whose arm can hold the camera at height
        ``cam_y``. None when the vertical gap alone exceeds arm reach."""
        dy = abs(cam_y - float(self.arm_base[1]))
        if dy > self.max_reach:
            return None
        r_hi = float(np.sqrt(max(self.max_reach ** 2 - dy ** 2, 0.0)))
        r_lo = 0.0 if self.min_reach <= dy else float(np.sqrt(
            max(self.min_reach ** 2 - dy ** 2, 0.0)))
        if r_hi <= r_lo:
            return None
        return r_lo, r_hi

    def to_dict(self) -> dict:
        return {
            "base_radius_m": self.base_radius,
            "base_height_m": self.base_height,
            "mast_height_m": self.mast_height,
            "arm_base_offset_m": self.arm_base.tolist(),
            "arm_reach_m": [self.min_reach, self.max_reach],
            "camera_height_range_m": list(self.height_range),
            "look_any_direction_assumed": self.look_any_direction,
        }


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Frame-level check (vectorized over base-placement candidates)
# --------------------------------------------------------------------------- #
_N_ANGLES = 16
_N_RADII = 5


def _candidates(cam_pos: np.ndarray, ring: tuple[float, float]) -> np.ndarray:
    """Chassis-center candidates (M,3): ring around the camera's ground
    projection, radii spanning the feasible standoff interval."""
    radii = np.linspace(ring[0] + 0.05 * (ring[1] - ring[0]),
                        ring[1] - 0.05 * (ring[1] - ring[0]), _N_RADII)
    angles = np.linspace(0.0, 2.0 * np.pi, _N_ANGLES, endpoint=False)
    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    b = np.empty((rr.size, 3))
    b[:, 0] = cam_pos[0] + rr.ravel() * np.cos(aa.ravel())
    b[:, 1] = 0.0                                  # chassis center on the floor
    b[:, 2] = cam_pos[2] + rr.ravel() * np.sin(aa.ravel())
    return b


def check_frame(cam_pos: np.ndarray, sdf: SceneSDF, robot: RobotModel,
                labels: list[str] | None = None) -> dict:
    """Reachability of ONE camera position.

    Returns {"ok": bool, "reason": str} — the reason names the binding
    constraint when infeasible (height band / arm reach / base placement)."""
    y = float(cam_pos[1])
    if not (robot.height_range[0] - 1e-6 <= y <= robot.height_range[1] + 1e-6):
        return {"ok": False, "kind": "height",
                "reason": f"相机高度 {y:.2f}m 超出可达带 "
                          f"[{robot.height_range[0]:.2f}, {robot.height_range[1]:.2f}]m"}
    ring = robot.reach_ring(y)
    if ring is None:
        return {"ok": False, "kind": "reach",
                "reason": f"垂直距离 |{y:.2f}−{float(robot.arm_base[1]):.2f}| "
                          f"> 臂展 {robot.max_reach:.2f}m"}

    cand = _candidates(cam_pos, ring)              # (M,3) chassis centers
    # body clearance: worst over base/mast query heights per candidate
    body_pts = np.repeat(cand, robot.body_heights.size, axis=0)
    body_pts[:, 1] = np.tile(robot.body_heights, cand.shape[0])
    clearance = sdf.numpy_query(body_pts).reshape(cand.shape[0], -1).min(axis=1)
    # room containment (geometric, any room)
    ok_room = np.array([sdf.room_bounds_of_point(c) is not None for c in cand])
    feasible = (clearance >= robot.base_radius) & ok_room

    if feasible.any():
        best = int(np.argmax(np.where(feasible, clearance, -np.inf)))
        return {"ok": True, "kind": "ok", "base_pos": cand[best].tolist(),
                "body_clearance_m": round(float(clearance[best]), 3)}

    # diagnose the binding constraint on the best (least bad) candidate
    best = int(np.argmax(np.where(ok_room, clearance, -np.inf)))
    pos = cand[best].copy()
    pos[1] = robot.body_heights[-1]                # attribute at mast top
    if not ok_room[best]:
        return {"ok": False, "kind": "placement",
                "reason": "底盘候选位置越出房间边界"}
    near = ""
    if labels:
        vals, idx = sdf.numpy_query_nearest(pos[None, :])
        if idx[0] < len(labels):
            near = f"，最近障碍 {labels[int(idx[0])]}"
    return {"ok": False, "kind": "placement",
            "reason": f"无可放置底盘：最佳位置身体净空 {clearance[best]:.2f}m "
                      f"< 底盘半径 {robot.base_radius:.2f}m{near}"}


# --------------------------------------------------------------------------- #
# Trajectory-level report
# --------------------------------------------------------------------------- #
def check_reachability(segments: list, scene: Scene, sdf: SceneSDF,
                       robot_cfg: dict | None = None, fps: float | None = None) -> dict:
    """Per-segment reachability audit of a finished trajectory — the Phase-1
    whole-body feasibility gate. Same report shape as
    ``stages.optimize.verify_collision_free``: status + violations + reasons."""
    robot = RobotModel(robot_cfg)
    labels = ([f"structure {s.kind} '{s.id}'" for s in scene.structures]
              + [f"object '{o.label}' ({o.id})" for o in scene.objects])
    violations = []
    for seg in segments:
        th, t_abs = seg.sample_times(fps or C.DEFAULT_FPS)
        pos = seg.positions(th, None, scene.subject).numpy()
        results = [check_frame(p, sdf, robot, labels) for p in pos]
        bad = [i for i, r in enumerate(results) if not r["ok"]]
        if not bad:
            continue
        kinds = {}
        for i in bad:
            kinds[results[i]["kind"]] = kinds.get(results[i]["kind"], 0) + 1
        worst = bad[0]
        violations.append({
            "segment": seg.name,
            "type": seg.type,
            "frames_unreachable": len(bad),
            "frames_total": len(results),
            "kinds": kinds,
            "worst_t": round(float(t_abs[worst]), 3),
            "worst_pos": [round(float(x), 3) for x in pos[worst]],
            "reason": results[worst]["reason"],
        })
    ok = not violations
    reasons = [
        f"segment {v['segment']} ({v['type']}): {v['frames_unreachable']}/"
        f"{v['frames_total']} 帧不可达 @ t={v['worst_t']}s pos={v['worst_pos']} — {v['reason']}"
        for v in violations
    ]
    return {"status": "ok" if ok else "failed",
            "fps_used": float(fps or C.DEFAULT_FPS),
            "robot": robot.to_dict(),
            "assumptions": ["look direction assumed achievable (spherical wrist)"
                            if robot.look_any_direction else None],
            "violations": violations, "reasons": reasons}
