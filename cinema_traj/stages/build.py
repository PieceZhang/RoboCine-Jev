"""Parametric Atomic Trajectory Builder (paper §3.3): instantiates the plan's
atomic commands as parametric segments with anchors from the anchor selector.

plan.json + scene → traj_init.json (segments with θ_fix / θ_free + anchors).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cinema_traj.common import constants as C
from cinema_traj.common.scene import ObjectNode, Scene, load_scene
from cinema_traj.common.se3 import cart_to_sph
from cinema_traj.common.cloud_sdf import cloud_sdf_from_scene
from cinema_traj.common.sdf import SceneSDF
from cinema_traj.common.subject import SubjectTrack
from cinema_traj.core import anchors as A
from cinema_traj.core.plan_schema import validate_plan
from cinema_traj.core.trajectories import (ArcTransition, CraneSegment, DollySegment,
                                           OrbitSegment, PanSegment, Segment, StaticSegment,
                                           SwingSegment, TiltSegment, TrackFollowSegment,
                                           TrackLeadSegment, TrackOrbitSegment, ZoomSegment)
from vlm_intention.common.atomic_io import atomic_write_json, load_json
from vlm_intention.common.config import load_config


# --------------------------------------------------------------------------- #
# Plan grouping: [(anchor_step, transition_step|None, traj_step)]
# --------------------------------------------------------------------------- #
def group_steps(steps: list) -> list:
    groups, i = [], 0
    while i < len(steps):
        s = steps[i]
        if s.get("kind") != "anchor":
            raise ValueError(f"step {i + 1}: expected anchor, got {s.get('kind')}")
        trans = None
        if i + 1 < len(steps) and steps[i + 1].get("kind") == "transition":
            trans = steps[i + 1]
            i += 1
        if i + 1 >= len(steps) or steps[i + 1].get("kind") not in ("object_traj", "subject_traj"):
            raise ValueError(f"step {i + 2}: expected trajectory after anchor")
        groups.append((s, trans, steps[i + 1]))
        i += 2
    return groups


# --------------------------------------------------------------------------- #
# Segment factory
# --------------------------------------------------------------------------- #


def _spiral_close_ratio(chord: float, r_s: float, margin_deg: float = 84.0):
    """Close ratio rho so the worst look-vs-travel angle over the spiral
    chord equals ``margin_deg`` (6 deg inside the 90 deg joint6 branch
    flip). Returns (rho, feasible).

    Derived: worst angle is at the far endpoint B of the chord —
    cos(angle_B) = (cos s - rho)/q with q = chord/r_s; requiring
    angle_B = m and eliminating s via the chord identity
    q^2 = 1 + rho^2 - 2 rho cos s gives
        rho = sqrt(1 - (sin m * q)^2) - cos m * q.
    """
    q = min(chord / max(r_s, 1e-6), 1.9)
    cm, sm = math.cos(math.radians(margin_deg)), math.sin(math.radians(margin_deg))
    disc = 1.0 - (sm * q) ** 2
    if disc <= 0.0 or q <= 1e-6:
        return 0.55, False
    rho = math.sqrt(disc) - cm * q
    return max(min(rho, 0.98), 0.50), rho >= 0.60


def _spiral_chord(c, az_s, r_s, direction, chord, target_id, name,
                  duration, t_start, room, target_mode, a_y,
                  arc_bow: bool = False):
    """SwingSegment on the subject polar grid with dead-center look."""
    rho, ok = _spiral_close_ratio(chord, r_s)
    sweep = math.acos(max(-1.0, min(1.0, rho + math.cos(math.radians(84.0))
                                    * (chord / max(r_s, 1e-6)))))
    r_e = max(r_s * rho, 0.5)
    az_e = az_s + direction * sweep
    p_s = np.array([c[0] + r_s * math.cos(az_s), a_y, c[2] + r_s * math.sin(az_s)])
    p_e = np.array([c[0] + r_e * math.cos(az_e), a_y, c[2] + r_e * math.sin(az_e)])
    chord_v = p_e - p_s
    L = float(np.linalg.norm(chord_v))
    nh = np.cross(chord_v, [0.0, 1.0, 0.0]) if L > 1e-6 else np.array([1.0, 0.0, 0.0])
    nn = np.linalg.norm(nh)
    n_hat = (nh / nn).tolist() if nn > 1e-6 else [1.0, 0.0, 0.0]
    alpha = 0.0
    if arc_bow:
        # bow the chord into a true ARC: pass through the mid-azimuth
        # point on the (r_s+r_e)/2 circle — the viewer line reads as a
        # curve around the subject instead of a straight chord. The bow
        # is locked (deterministic geometry, monotone-ish radius close).
        r_m = 0.5 * (r_s + r_e)
        az_m = az_s + direction * sweep / 2.0
        p_m = np.array([c[0] + r_m * math.cos(az_m), a_y,
                        c[2] + r_m * math.sin(az_m)])
        bow = float(np.dot(p_m - 0.5 * (p_s + p_e),
                           np.asarray(n_hat, dtype=np.float64)))
        alpha = math.degrees(math.atan2(2.0 * bow, max(L, 1e-6)))
    seg = SwingSegment(name, duration, t_start, room, target_mode, target_id,
                       fix={"center": c.tolist(),
                            "p_s": p_s.tolist(), "p_e": p_e.tolist(),
                            "chord": max(L, 1e-6), "n_hat": n_hat},
                       free={"alpha_deg": _P(alpha, alpha, alpha)})
    return seg, ok


def make_segment(traj_step: dict, anchor, obj: ObjectNode | None, subject: SubjectTrack | None,
                 t_start: float, scene: Scene) -> Segment:
    ttype = traj_step["type"]
    duration = float(traj_step.get("duration_s", C.DEFAULT_DURATION))
    room, target_mode, target_id, center = _target_info(traj_step, obj, subject, t_start, scene)
    if ttype.startswith("orbit"):
        sweep = C.ORBIT_SWEEP_DEG[_orbit_variant(ttype)]
        # plan-level sweep override (same contract as pan/tilt): a hand-authored
        # beat may demand a tighter arc than the variant's nominal sweep when
        # the room cannot host it (e.g. a subject seated against a wall — the
        # flyable quadrant is narrower than a quarter)
        if traj_step.get("sweep_deg") is not None:
            sweep = abs(float(traj_step["sweep_deg"]))
        c = center
        a = np.asarray(anchor.pos, dtype=np.float64)
        az0, el0, r0 = cart_to_sph(a - c)
        r0 = max(math.hypot(a[0] - c[0], a[2] - c[2]), 0.3)     # horizontal radius
        direction = float(traj_step.get("direction", 1.0))
        room = room or ""
        top = _room_top(scene, room)
        # SPIRAL ORBIT (dead-center look): a constant-radius orbit with the
        # look on the subject is exactly 90° arm yaw everywhere (the joint6
        # branch flip). The old workaround was a 20° tangent look lead
        # (subject ~17° off optical axis). Instead the camera swings the
        # planned sweep's chord around the subject while the radius closes
        # analytically (worst look-vs-travel = 84°, 6° inside the wall) and
        # the subject stays dead-center the whole move.
        chord_orb = 2.0 * r0 * math.sin(math.radians(sweep) / 2.0)
        seg, _ = _spiral_chord(c, az0, r0, direction, chord_orb,
                               target_id, f"orbit_{target_id}",
                               duration, t_start, room, target_mode,
                               float(a[1]), arc_bow=True)
    elif ttype in ("move_in", "move_out"):
        c = center
        a = np.asarray(anchor.pos, dtype=np.float64)
        d = a - c
        dh = d.copy()
        dh[1] = 0.0                          # level dolly: floor-plane axis, constant height
        nn = float(np.linalg.norm(dh))
        d_hat = (dh / nn).tolist() if nn > 1e-6 else [1.0, 0.0, 0.0]
        f = 0.55 if ttype == "move_in" else 1.45
        # a close-up on the SUBJECT must still keep the body capsule clear:
        # the chest-center r_e floor of 0.3 would let the camera end inside
        # her 0.3m capsule (the hard verify gate correctly rejects that)
        r_e_lo = 0.55 if subject is not None else 0.3
        if ttype == "move_out":
            # a pull-back must out-run its start: r_s ≤ anchor dist < 1.15·anchor ≤ r_e.
            # A loose r_e floor lets the optimizer flatten r_e below r_s, shipping
            # 后拉 as a push-in (r_e free to 0.3 while collision cost prefers near).
            # r_s is PINNED to the anchor distance: the anchor IS the shot's
            # start (the previous beat parks exactly there) — freeing it let
            # the optimizer drag the start 1.75 m off the join (2.91 → 1.16),
            # breaking arc continuity AND parking the opening frames in the
            # steep up-look IK dead zone (move_out: 20 unreachable waypoints).
            seg = DollySegment(f"{ttype}_{target_id}", duration, t_start, room,
                               target_mode, target_id,
                               fix={"center": c.tolist(), "dir": d_hat},
                               free={"r_s": _P(nn, nn, nn),
                                     # r_e hi = the minimum legal pull
                                     # (1.15·anchor): the old 4.0 m cap let
                                     # the optimizer park the tail in the
                                     # far low-pitch IK fringe where the
                                     # config stream goes unreachable; a
                                     # long climbing pull-back needs its own
                                     # beat variant
                                     "r_e": _P(max(nn * f, r_e_lo),
                                               max(r_e_lo, 1.15 * nn),
                                               max(r_e_lo, 1.15 * nn))})
        else:
            seg = DollySegment(f"{ttype}_{target_id}", duration, t_start, room,
                               target_mode, target_id,
                               fix={"center": c.tolist(), "dir": d_hat},
                               free={"r_s": _P(nn, 0.4 * nn, 1.5 * nn),
                                     "r_e": _P(max(nn * f, r_e_lo), r_e_lo, 4.0)})
    elif ttype == "lateral_truck":
        # LATERAL TRUCK (user semantic for 横摇): the camera translates
        # along the sofa from the LEFT cast person to the RIGHT one at the
        # corridor distance, look locked on the subject (centered frame).
        # Geometry = a straight chord (ArcTransition, alpha free for the
        # optimizer); endpoints from the scene's person_<id> OBBs so the
        # shot adapts to any staging. Falls back to the subject-anchored
        # corridor line when the cast is absent.
        c = np.asarray(center, dtype=np.float64)
        subj = scene.subject.state_at(t_start).pos if subject is not None else c
        direction = float(traj_step.get("direction", 1.0))
        truck_z = subj[2] - 2.55          # corridor depth north of the sofa
        xs = []
        for pid in ("person_Dad", "person_Mom"):
            po = scene.object_by_id(pid)
            if po is not None:
                xs.append(float(po.obb.center[0]))
        if len(xs) < 2:
            xs = [subj[0] - 0.8, subj[0] + 0.8]
        x_l, x_r = min(xs), max(xs)
        a_y = float(np.asarray(anchor.pos, dtype=np.float64)[1])
        # SPIRAL TRUCK (dead-center look): a pure lateral chord looking at
        # the subject is exactly 90° arm yaw at its midpoint (d|r|/dt = 0
        # at closest approach for ANY path) — unplannable, and the old
        # 0.85 m look lead drifted the subject 38° -> 2° across the shot.
        # Instead swing the person-span chord around the subject with the
        # analytic close ratio (worst look-vs-travel = 84°): subject
        # dead-center, Dad -> Mom coverage, mild approach by design.
        x_a = x_l if direction > 0 else x_r
        az_s = math.atan2(truck_z - c[2], x_a - c[0])
        r_s = max(math.hypot(x_a - c[0], truck_z - c[2]), 0.3)
        # corridor cap: the spiral's radial close pushes the chord's far
        # end north, toward the sofa-side floor-clutter band where the
        # base cannot park (disc search included). Bisect the chord so
        # the far endpoint stays at z <= truck_z + 0.15 — the radius
        # close then runs mostly along the clear corridor, and the base
        # keeps the tangent-rear / south-disc parking row it verifiably
        # fits on.
        chord_full = max(abs(x_r - x_l), 0.2)
        z_cap = truck_z + 0.15
        chord = chord_full

        def _end_z(ch_):
            rho, _ = _spiral_close_ratio(ch_, r_s)
            sw = math.acos(max(-1.0, min(1.0, rho + math.cos(math.radians(84.0))
                                         * (ch_ / max(r_s, 1e-6)))))
            r_e = r_s * rho
            az_e = az_s + direction * sw
            return c[2] + r_e * math.sin(az_e)

        if _end_z(chord_full) > z_cap:
            lo, hi = 0.4, chord_full
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                if _end_z(mid) <= z_cap:
                    lo = mid
                else:
                    hi = mid
            chord = lo
        seg, _ = _spiral_chord(c, az_s, r_s, direction, chord,
                               target_id, f"truck_{target_id}",
                               duration, t_start, room, target_mode, a_y)
    elif ttype == "arc_swing":
        # SHOOTING spiral swing around the target: start at the anchor,
        # end at a NEARER position swung `swing_deg` around the target —
        # the radial close keeps look-vs-travel inside the arm's window
        # (a constant-radius orbit is exactly 90° and gated out). Look
        # stays on the target the whole move (ArcTransition look_target
        # with the subject/object center).
        c = np.asarray(center, dtype=np.float64)
        a = np.asarray(anchor.pos, dtype=np.float64)
        r_s = max(math.hypot(a[0] - c[0], a[2] - c[2]), 0.3)
        az_s = math.atan2(a[2] - c[2], a[0] - c[0])
        direction = float(traj_step.get("direction", 1.0))
        swing = direction * math.radians(float(traj_step.get("swing_deg", 40.0)))
        r_e = max(r_s * float(traj_step.get("close_ratio", 0.70)), 0.5)
        p_s = a.copy()
        p_e = np.array([c[0] + r_e * math.cos(az_s + swing),
                        a[1],
                        c[2] + r_e * math.sin(az_s + swing)])
        chord_v = p_e - p_s
        L = float(np.linalg.norm(chord_v))
        nh = np.cross(chord_v, [0.0, 1.0, 0.0]) if L > 1e-6 else np.array([1.0, 0.0, 0.0])
        nn = np.linalg.norm(nh)
        n_hat = (nh / nn).tolist() if nn > 1e-6 else [1.0, 0.0, 0.0]
        seg = SwingSegment(f"swing_{target_id}", duration, t_start, room,
                           target_mode, target_id,
                           fix={"center": c.tolist(),
                                "p_s": p_s.tolist(), "p_e": p_e.tolist(),
                                "chord": max(L, 1e-6), "n_hat": n_hat},
                           # α bounds: southward bulges (negative α) swing the
                           # chord ends' tangents toward the TV corridor where
                           # the base cannot park (measured: α=-6.6 twisted the
                           # start tangent to 109° = 1 unreachable waypoint).
                           # Positive-only α bulges toward the sofa arm, away
                           # from every gate-relevant obstacle.
                           free={"alpha_deg": _P(0.0, 0.0, 20.0)})
    elif ttype == "crane":
        c = center
        a = np.asarray(anchor.pos, dtype=np.float64)
        az0, el0, r0 = cart_to_sph(a - c)
        h_s = traj_step.get("height_s")
        h_e = traj_step.get("height_e")
        if h_s is not None and h_e is not None:
            # EXPLICIT small-amplitude crane (user semantic): the camera
            # rides a locked vertical arc at the ANCHOR distance whose
            # endpoints are the requested heights — both inside the jarvis
            # camera band [0.25, 0.63] BY CONSTRUCTION. Far-range is the
            # workable regime: a close-range crane (small r) looks steeply
            # UP at the head (the up-look IK dead zone) and parks its base
            # in the sofa/table gap; at the anchor distance the same
            # height window is a gentle ±10 deg eps arc with proven
            # corridor parking.
            r_c = max(r0, 1.0)
            c1 = float(c[1])
            eps_s = math.asin(max(-0.999, min(0.999, (float(h_s) - c1) / r_c)))
            eps_e = math.asin(max(-0.999, min(0.999, (float(h_e) - c1) / r_c)))
            seg = CraneSegment(f"crane_{target_id}", duration, t_start, room,
                               target_mode, target_id,
                               fix={"center": c.tolist(), "az": float(az0),
                                    "eps_s": float(eps_s)},
                               free={"radius": _P(r_c, r_c, r_c),
                                     "eps_e": _P(eps_e, eps_e, eps_e)})
        else:
            seg = CraneSegment(f"crane_{target_id}", duration, t_start, room,
                               target_mode, target_id,
                               fix={"center": c.tolist(), "az": float(az0), "eps_s": float(el0)},
                               free={"radius": _P(r0, max(0.3, 0.6 * r0), 2.0 * r0),
                                     "eps_e": _P(el0 + math.radians(25),
                                                 el0 + math.radians(8), el0 + math.radians(60))})
    elif ttype in ("pan_left", "pan_right", "tilt_up", "tilt_down"):
        sweep = float(traj_step.get("sweep_deg",
                                    45.0 if ttype.startswith("pan") else 30.0))
        if ttype in ("pan_right", "tilt_down"):
            sweep = -abs(sweep)
        cls = PanSegment if ttype.startswith("pan") else TiltSegment
        seg = cls.__new__(cls)
        Segment.__init__(seg, f"{ttype}_{target_id}", duration, t_start, room,
                         target_mode, target_id,
                         fix={"anchor": np.asarray(anchor.pos).tolist(),
                              "center": center.tolist(), "sweep_deg": sweep,
                              "sweep_return": bool(traj_step.get("sweep_return", False))})
    elif ttype in ("zoom_in_out", "zoom_out_in"):
        seg = ZoomSegment.__new__(ZoomSegment)
        m = C.ZOOM_M_PEAK if ttype == "zoom_in_out" else 1.0 / C.ZOOM_M_PEAK
        Segment.__init__(seg, f"{ttype}_{target_id}", duration, t_start, room,
                         target_mode, target_id,
                         fix={"anchor": np.asarray(anchor.pos).tolist(),
                              "center": center.tolist(),
                              "m_peak": m, "fov0_deg": 55.0})
    elif ttype == "static":
        seg = StaticSegment(f"static_{target_id}", duration, t_start, room,
                            target_mode, target_id,
                            fix={"anchor": np.asarray(anchor.pos).tolist(),
                                 "center": center.tolist()})
    elif ttype in ("track_follow", "track_orbit", "track_lead"):
        detail = anchor.detail or {}
        elev0 = float(detail.get("elev", math.radians(20)))
        dist0 = float(detail.get("dist", 1.8))
        az0 = float(detail.get("az_rel", 0.0))
        lead0 = math.radians(165.0) if az0 >= 0 else math.radians(-165.0)
        el_lo, el_hi = 0.05, 1.2
        if ttype == "track_follow":
            # az pinned by the anchor step (azimuth_deg) → keep the optimizer
            # within ±10° of the demanded side; otherwise full circle as before
            az_p = _P(az0, az0 - 0.17, az0 + 0.17) if detail.get("az_pinned") \
                else _P(az0, -math.pi, math.pi)
            fix = {"key_point": "head"}
            if traj_step.get("elev_end_deg") is not None:
                fix["elev_end"] = math.radians(float(traj_step["elev_end_deg"]))
            seg = TrackFollowSegment(f"follow_{target_id}", duration, t_start, room,
                                     "subject_head", target_id,
                                     fix=fix,
                                     free={"az_rel": az_p,
                                           "elev": _P(elev0, el_lo, el_hi),
                                           "dist": _P(dist0, 0.8, 4.0)})
        elif ttype == "track_orbit":
            seg = TrackOrbitSegment(f"torbit_{target_id}", duration, t_start, room,
                                    "subject_head", target_id,
                                    fix={"key_point": "head"},
                                    free={"az_s": _P(az0, -math.pi, math.pi),
                                          "az_e": _P(az0 + math.radians(90),
                                                     az0 - math.radians(270),
                                                     az0 + math.radians(270)),
                                          "elev": _P(elev0, el_lo, el_hi),
                                          "dist": _P(dist0, 0.8, 4.0)})
        else:
            seg = TrackLeadSegment(f"lead_{target_id}", duration, t_start, room,
                                   "subject_head", target_id,
                                   fix={"key_point": "head"},
                                   free={"lead_angle": _P(lead0, -math.pi, math.pi),
                                         "elev": _P(elev0, el_lo, el_hi),
                                         "dist": _P(dist0, 0.8, 4.0)})
    else:
        raise ValueError(f"unsupported trajectory type '{ttype}'")
    return seg


def _orbit_variant(ttype: str) -> str:
    for k in C.ORBIT_SWEEP_DEG:
        if ttype == k:
            return k
    return "orbit_quarter"


def _P(value, lo, hi):
    from cinema_traj.core.trajectories import Param
    return Param(float(value), float(min(lo, hi)), float(max(lo, hi)))


# --------------------------------------------------------------------------- #
# Orbit feasibility: radius bounds vs actual clearance around the center
# --------------------------------------------------------------------------- #
# must match costs.cost_sdf's d_safe — the optimizer penalizes Φ below it, so
# bounds clamped at a smaller margin hand the optimizer circles it then
# fights (it shortens the sweep to escape dirty sectors — a 360° that
# silently collapses into a ~108° arc)
ORBIT_FEAS_MARGIN = 0.2
ORBIT_MIN_FEASIBLE_M = 0.35                       # below this a 360° shot is absurd


def _orbit_arc_feasible(sdf: SceneSDF, center: np.ndarray, height: float, r: float,
                        azs: np.ndarray, direction: float, margin: float,
                        base_d: float | None = None,
                        base_margin: float = 0.15) -> bool:
    """One candidate radius: the swept arc at ``height`` is SDF-clear AND
    inside the room bounds (the OBB union reads points beyond an exterior
    wall as free space — the room constraint is separate). Only the arc the
    orbit actually flies is sampled — a half/quarter orbit must not pay for
    the closed side behind its target.

    ``base_d`` (jarvis): also require SDF clearance at the companion BASE
    points — the floor robot trails the camera by an arm extension along the
    tangent, so its path is a slightly larger circle that must be drivable
    (on the floor, y=0 — a sofa may clear camera height but block the base)."""
    n_az = azs.size
    pts = np.stack([center[0] + r * np.cos(azs),
                    np.full(n_az, height),
                    center[2] + r * np.sin(azs)], axis=1)
    if float(sdf.numpy_query(pts).min()) < margin:
        return False
    base_pts = None
    if base_d is not None:
        tangent = np.stack([-np.sin(azs), np.zeros(n_az), np.cos(azs)], axis=1)
        base_pts = pts - direction * base_d * tangent
        base_pts[:, 1] = 0.0                        # the base rides the floor
        if float(sdf.numpy_query(base_pts).min()) < base_margin:
            return False
    bounds = sdf.room_bounds_of_point(center)
    if bounds is not None:
        lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
        if not bool(((pts >= lo + C.SCENE_BOUNDS_MARGIN) &
                     (pts <= hi - C.SCENE_BOUNDS_MARGIN)).all()):
            return False
        if base_pts is not None and not bool(
                ((base_pts >= lo + C.SCENE_BOUNDS_MARGIN) &
                 (base_pts <= hi - C.SCENE_BOUNDS_MARGIN)).all()):
            return False
    return True


def _orbit_feasible_radius(sdf: SceneSDF, center: np.ndarray, height: float,
                           r_start: float, r_min: float, az_lo: float = 0.0,
                           az_span: float = 2.0 * math.pi, n_az: int = 24,
                           step: float = 0.05,
                           margin: float = None,
                           base_d: float | None = None,
                           base_margin: float = 0.15) -> float | None:
    """Largest r ∈ [r_min, r_start] whose swept arc is feasible (see
    :func:`_orbit_arc_feasible`). Descending linear sweep (clearance is not
    monotone in r — furniture gaps)."""
    margin = ORBIT_FEAS_MARGIN if margin is None else margin
    azs = az_lo + np.linspace(0.0, az_span, n_az)
    direction = 1.0 if az_span >= 0 else -1.0
    r = r_start
    while r >= r_min - 1e-9:
        if _orbit_arc_feasible(sdf, center, height, r, azs, direction, margin,
                               base_d, base_margin):
            return float(r)
        r -= step
    return None


def _snap_orbit_radius_value(sdf: SceneSDF, center: np.ndarray, height: float,
                             value: float, r_cap: float, azs: np.ndarray,
                             direction: float, margin: float,
                             base_d: float | None = None,
                             base_margin: float = 0.15,
                             step: float = 0.05) -> float:
    """Snap an orbit's init radius into the feasible band, scanning UP.

    The band clamp returns the LARGEST feasible radius, but feasibility in r
    is not monotone (furniture holes): the init value inherited from the
    anchor distance can sit inside a hole within [lo, r] — the reachability
    hard gate then rejects a shot the clamp believed it had fixed. ``r_cap``
    is itself feasible (the descending scan found it), so the scan always
    terminates there at the latest."""
    v = float(value)
    if _orbit_arc_feasible(sdf, center, height, v, azs, direction, margin,
                           base_d, base_margin):
        return v
    while v < r_cap - 1e-9:
        v = min(v + step, r_cap)
        if _orbit_arc_feasible(sdf, center, height, v, azs, direction, margin,
                               base_d, base_margin):
            return v
    return float(r_cap)


def _clamp_orbit_radius_to_clearance(seg, sdf: SceneSDF) -> None:
    """Tighten an orbit's radius bounds to what the room actually allows.

    The paper sizes radius bounds as multiples of the anchor distance; with a
    subject near walls/furniture (or a 远景/全景 preference that outgrows the
    room) every radius in those bounds collides, and the optimizer cannot fix
    what the bounds forbid — the result ships a colliding trajectory (PR #2
    known issue: 21% collision / 52% occlusion on 全周环绕). Sample the real
    circle: clamp hi to the largest feasible radius, lower lo if the room
    only allows a tighter orbit than the style asked for, and refuse the shot
    when even ORBIT_MIN_FEASIBLE_M doesn't fit."""
    from cinema_traj.core.trajectories import Param

    p = seg.free.get("radius")
    if p is None:
        return
    center = seg._f("center")
    height = float(seg.free["height"].value)
    az_lo = float(seg.fix.get("phi_s", 0.0))
    az_span = float(seg.fix.get("direction", 1.0)) * \
        math.radians(float(seg.fix.get("sweep_deg", 360.0)))
    # two-tier: prefer the full d_safe band; cramped rooms (demo apartment,
    # furnished Replica scenes) fall back to half — the optimizer's softplus
    # still pushes toward d_safe, but the bounds no longer reject the shot
    margin = ORBIT_FEAS_MARGIN
    r = _orbit_feasible_radius(sdf, center, height, p.hi, ORBIT_MIN_FEASIBLE_M,
                               az_lo=az_lo, az_span=az_span)
    if r is None:
        margin = ORBIT_FEAS_MARGIN / 2.0
        r = _orbit_feasible_radius(sdf, center, height, p.hi, ORBIT_MIN_FEASIBLE_M,
                                   az_lo=az_lo, az_span=az_span, margin=margin)
    if r is None:
        near = float(sdf.numpy_query(
            np.array([[center[0], height, center[2]]])).min())
        raise ValueError(
            f"orbit {seg.name}: 目标周围无 ≥{ORBIT_MIN_FEASIBLE_M}m 的可行环绕半径"
            f"（净空 {near:.2f}m）— 请把主体放在更开阔的位置，或改用 track_orbit")
    lo = min(p.lo, r) if r >= p.lo else max(ORBIT_MIN_FEASIBLE_M, r - 0.15)
    # value can sit in a furniture hole inside [lo, r] — snap it feasible
    # (the anchor distance presets have no knowledge of the room's radial
    # holes; without the snap the reachability gate rejects the shot)
    azs = az_lo + np.linspace(0.0, az_span, 24)
    direction = 1.0 if az_span >= 0 else -1.0
    value = _snap_orbit_radius_value(sdf, center, height, float(p.value), r,
                                     azs, direction, margin)
    seg.free["radius"] = Param(float(min(max(value, lo), r)), lo, r)


def _clamp_dolly_run_to_clearance(seg, sdf: SceneSDF,
                                  margin: float = None,
                                  step: float = 0.02) -> None:
    """Drone-platform dolly feasibility: the camera traverses the WHOLE
    [r_s, r_e] run on the dolly line, so every radius in between must be
    SDF-clear — the subject close-up floor (r_e ≥ 0.55) can otherwise park
    the camera inside furniture the line grazes (frl_apartment_0 console:
    19% collision at the close-up end). Shift/shrink [r_s, r_e] into the
    largest contiguous feasible run containing the approach radius; refuse
    the shot when no feasible radius exists at all."""
    from cinema_traj.core.trajectories import Param

    margin = ORBIT_FEAS_MARGIN if margin is None else margin
    p_s, p_e = seg.free["r_s"], seg.free["r_e"]
    c = np.asarray(seg._f("center"), dtype=np.float64)
    d = np.asarray(seg._f("dir"), dtype=np.float64)

    def clear_at(r_val: float) -> bool:
        p = c + r_val * d
        if sdf.cloud is not None and sdf.cloud.numpy_query(p) < margin:
            return False
        if float(sdf.numpy_query(p)) < margin:
            return False
        bounds = sdf.room_bounds_of_point(c)         # outside-wall ≠ free space
        if bounds is not None:
            lo_b, hi_b = np.asarray(bounds[0]), np.asarray(bounds[1])
            if not np.all((p >= lo_b + C.SCENE_BOUNDS_MARGIN)
                          & (p <= hi_b - C.SCENE_BOUNDS_MARGIN)):
                return False
        return True

    r_hi = max(float(p_s.hi), float(p_e.hi))
    r_min = max(ORBIT_MIN_FEASIBLE_M, 0.3)
    feasible = [float(r) for r in np.arange(r_min, r_hi, step) if clear_at(float(r))]
    if not feasible:
        center_clear = float(sdf.numpy_query(c))
        raise ValueError(
            f"{seg.name}: 推拉线上无任何 ≥{margin}m 净空的可行半径"
            f"（中心净空 {center_clear:.2f}m）— 请换推拉方向或机位")
    # largest contiguous feasible run (grid gaps > step break a run)
    runs, start, prev = [], feasible[0], feasible[0]
    for r in feasible[1:]:
        if r - prev > step * 1.5:
            runs.append((start, prev))
            start = r
        prev = r
    runs.append((start, prev))
    largest = max(runs, key=lambda ab: ab[1] - ab[0])
    # prefer the run containing r_s (keep the approach leg); else the largest
    chosen = next(((a, b) for a, b in runs if a <= float(p_s.value) <= b),
                  largest)
    new_lo = max(r_min, chosen[0])
    new_hi = min(r_hi, chosen[1])
    if new_hi < new_lo:
        new_lo = new_hi = max(r_min, chosen[0])
    for p in (p_s, p_e):
        nv = float(min(max(float(p.value), new_lo), new_hi))
        p.value, p.lo, p.hi = nv, max(float(p.lo), new_lo), min(float(p.hi), new_hi)


# --------------------------------------------------------------------------- #
# Jarvis reachability clamps (floor-robot workspace — geometry layer)
# --------------------------------------------------------------------------- #
def _orbit_reachable_radius_cap(seg, base_sdf, caps, subject,
                                n_wp: int = 5, step: float = 0.05):
    """Largest orbit radius whose INIT-sweep arc passes the real
    base+arm reachability check at ``n_wp`` sparse waypoints (or None when
    even ``lo`` fails — the build verify stage then raises the proper
    error). Descends from the current bound so the optimizer's clearance
    cost can never drift the radius past what the robot can actually hold.

    Uses the STRICT 3 mm IK tolerance regardless of the configured export
    tolerance: this scan produces BOUNDS the optimizer is free to wander
    inside, so it must be conservative — the looser live-tracking
    accommodation (IK tol 3 cm) is for the export gate, not a planning
    guarantee (with 3 cm the scan blessed radius 2.3 m whose far end the
    arm cannot actually hold, and the optimizer shipped a 25%-unreachable
    arc)."""
    import dataclasses

    from cinema_traj.core.jarvis import camera_reachable
    from cinema_traj.common.se3 import look_at_R, rotation_to_yaw_pitch

    strict_caps = dataclasses.replace(caps, ik_tol_pos=3e-3, ik_tol_rot=1.2e-3)

    center = seg._f("center")
    height = float(seg.free["height"].value)
    phi_s = float(seg.fix["phi_s"])
    direction = float(seg.fix.get("direction", 1.0))
    sweep = direction * math.radians(float(seg.fix.get("sweep_deg", 360.0)))
    if seg.target_mode.startswith("subject") and subject is not None:
        look = subject.key_point(seg.t_start, "head")
    else:
        look = center
    look = np.asarray(look, dtype=np.float64)
    us = np.linspace(0.0, 1.0, n_wp)
    phis = phi_s + sweep * us
    p = seg.free["radius"]
    # descend from at most value+1 m: the init value is what the verify gate
    # proves next, the optimizer only needs headroom ABOVE it, and a full
    # descent from a room-wide hi costs thousands of IK solves on scenes
    # where the reachable band sits far below hi
    r = min(float(p.hi), float(p.value) + 1.0)
    r_min = max(float(p.lo), ORBIT_MIN_FEASIBLE_M)
    while r >= r_min - 1e-9:
        ok = True
        for phi, u in zip(phis, us):
            pos = np.array([center[0] + r * math.cos(phi), height,
                            center[2] + r * math.sin(phi)])
            to_l = look - pos
            n = float(np.linalg.norm(to_l))
            if n < 1e-6:
                ok = False
                break
            R = look_at_R(pos, look)
            yaw, pitch = rotation_to_yaw_pitch(R)
            tangent = math.atan2(-math.sin(phi) * direction,
                                 math.cos(phi) * direction)
            if camera_reachable(pos, math.radians(yaw), math.radians(pitch),
                                base_sdf, tangent_yaw=tangent,
                                caps=strict_caps) is None:
                ok = False
                break
        if ok:
            return r
        r -= step
    return None


def _clamp_segment_to_jarvis_workspace(seg, sdf: SceneSDF, caps,
                                       base_sdf: SceneSDF | None = None,
                                       subject=None) -> None:
    """Tighten a segment's geometry into what the Jarvis floor robot can
    reach — the shape-level mirror of ``_clamp_orbit_radius_to_clearance``.

    Retiming stretches the time axis but never the shape: an anchor floating
    1.6 m up (drone eye level) is unreachable no matter how slow the move.
    Measured workspace (FK sampling + IK): the arm's camera band is
    ≈ [0.13, 0.96] m physically, but LEVEL looks need z ≥ ~0.55 and steep
    down-looks need z ≤ ~0.85 — the cinematic band below keeps every shot in
    the jointly-feasible (height, look-pitch) region. The post-optimize
    audit (stages.optimize, jarvis platform) re-verifies each waypoint with
    full base+arm IK. ``sdf`` is the full camera collision field (camera-arc
    clearance); ``base_sdf`` the floor-free field for base parking (defaults
    to ``sdf``)."""
    base_sdf = base_sdf if base_sdf is not None else sdf
    from cinema_traj.core.trajectories import Param

    z_lo, z_hi = caps.cinematic_z_band()
    t = seg.type

    if t.startswith("orbit"):
        p = seg.free["height"]
        lo, hi = max(float(p.lo), z_lo), min(float(p.hi), z_hi)
        if lo > hi:
            raise ValueError(f"orbit {seg.name}: 高度带 [{p.lo:.2f}, {p.hi:.2f}] 与 "
                             f"Jarvis 可达带 [{z_lo:.2f}, {z_hi:.2f}] 无交集")
        h_val = float(min(max(float(p.value), lo), hi))
        if seg.target_mode.startswith("subject"):
            # subject-head orbit: PIN the camera at the clamped anchor height.
            # A free height lets the optimizer chase cost gradients DOWN the
            # band (measured: 0.305 m → 54° up-look → base+IK unreachable at
            # 25% of waypoints), and a floor-level camera framing a standing
            # subject's face is a garbage shot anyway. Object orbits keep
            # the height DOF (their target sits inside the band).
            seg.free["height"] = Param(h_val, h_val, h_val)
        else:
            seg.free["height"] = Param(h_val, lo, hi)
        # the base trails the camera along the tangent — its circle must be
        # drivable too (radius ≈ √(r² + d²) points, checked directly)
        center = seg._f("center")
        height = float(seg.free["height"].value)
        az_lo = float(seg.fix.get("phi_s", 0.0))
        az_span = float(seg.fix.get("direction", 1.0)) * \
            math.radians(float(seg.fix.get("sweep_deg", 360.0)))
        p = seg.free["radius"]
        r = _orbit_feasible_radius(base_sdf, center, height, float(p.hi),
                                   max(float(p.lo), ORBIT_MIN_FEASIBLE_M),
                                   az_lo=az_lo, az_span=az_span,
                                   base_d=float(np.median(caps.arm_extension)),
                                   base_margin=caps.base_clearance_margin)
        if r is None:
            raise ValueError(
                f"orbit {seg.name}: 底盘伴飞圆（外扩 {float(np.median(caps.arm_extension)):.2f}m）"
                f"无行驶净空 — 请改用 track_orbit 或拉远机位")
        lo = min(float(p.lo), r) if r >= float(p.lo) else max(ORBIT_MIN_FEASIBLE_M, r - 0.15)
        # re-snap under the STRICTER base-companion feasibility (the camera-
        # clear snap above can land on a radius whose base circle is not
        # drivable)
        azs = az_lo + np.linspace(0.0, az_span, 24)
        direction = 1.0 if az_span >= 0 else -1.0
        value = _snap_orbit_radius_value(
            base_sdf, center, height, float(p.value), r, azs, direction,
            ORBIT_FEAS_MARGIN, base_d=float(np.median(caps.arm_extension)),
            base_margin=caps.base_clearance_margin)
        seg.free["radius"] = Param(float(min(max(value, lo), r)), lo, r)
        # finally cap hi by REAL base+arm reachability along the arc: the
        # companion-circle check above proves floor clearance for the
        # tangent-trailing base only — it cannot see the IK envelope (a
        # corridor can park the camera fine at the init radius yet refuse
        # every base pose one metre further out). Without the cap the
        # optimizer's clearance cost drifts the radius into territory the
        # post-optimize reachability audit then rejects.
        r_cap = _orbit_reachable_radius_cap(seg, base_sdf, caps, subject)
        if r_cap is not None and r_cap < float(seg.free["radius"].hi):
            pr = seg.free["radius"]
            seg.free["radius"] = Param(
                float(min(max(float(pr.value), pr.lo), r_cap)),
                float(pr.lo), r_cap)

    elif t == "crane":
        # y(t) = c1 + r·sin(eps), eps: eps_s → eps_e — sweep radius down until
        # the whole vertical arc stays inside the band
        c1 = float(seg._f("center")[1])
        eps_s = float(seg.fix["eps_s"])
        eps_e = seg.free["eps_e"]
        eps_span = np.linspace(min(eps_s, float(eps_e.lo)), max(eps_s, float(eps_e.hi)), 16)
        r = seg.free["radius"]
        r_try = float(r.hi)
        while r_try >= max(float(r.lo), 0.2) - 1e-9:
            ys = c1 + r_try * np.sin(eps_span)
            if float(ys.min()) >= z_lo - 1e-9 and float(ys.max()) <= z_hi + 1e-9:
                break
            r_try -= 0.05
        else:
            raise ValueError(
                f"crane {seg.name}: 竖直弧无法压入 Jarvis 可达带 [{z_lo:.2f}, {z_hi:.2f}] "
                f"— 请降低起止仰角")
        seg.free["radius"] = Param(float(min(float(r.value), r_try)),
                                   float(r.lo), r_try)

    elif t in ("track_follow", "track_orbit", "track_lead"):
        # camera height ≈ dist·sin(elev) above the subject's feet; heads sit
        # at ~1.6m so these shots look UP — keep the camera inside the band
        # and near the anchor-chosen geometry (box bounds cannot express the
        # dist×elev product, so tighten around init; the audit catches the rest)
        d0 = float(seg.free["dist"].value)
        el0 = float(seg.free["elev"].value)
        d_lo, d_hi = max(0.8, 0.8 * d0), 1.25 * d0
        el_lo = max(float(seg.free["elev"].lo),
                    math.asin(min(0.99, z_lo / d_hi)) - 0.15,
                    el0 - 0.15)
        el_hi = min(float(seg.free["elev"].hi),
                    math.asin(min(0.99, z_hi / d_lo)) + 0.15,
                    el0 + 0.15)
        if el_lo > el_hi:
            raise ValueError(
                f"{seg.name}: 跟拍仰角无法同时满足主体取景与 Jarvis 可达带")
        seg.free["dist"] = Param(d0, d_lo, d_hi)
        seg.free["elev"] = Param(float(min(max(el0, el_lo), el_hi)), el_lo, el_hi)

    elif t in ("static", "pan_left", "pan_right", "tilt_up", "tilt_down",
               "zoom_in_out", "zoom_out_in"):
        anchor = np.asarray(seg.fix["anchor"], dtype=np.float64)
        y0 = float(min(max(anchor[1], z_lo), z_hi))
        if abs(y0 - anchor[1]) > 1e-9:
            anchor[1] = y0
            seg.fix["anchor"] = anchor.tolist()

    elif t in ("move_in", "move_out"):
        # dolly height = center height (target) — pin a clamped ride height
        c1 = float(seg._f("center")[1])
        seg.fix["cam_height"] = float(min(max(c1, z_lo), z_hi))
        # the camera rides the dolly line at cam_height looking at the center;
        # the feasible r-set along the line is NOT monotone (furniture holes:
        # kujiale move_in_sofa — both ends clear, the middle parked the base
        # inside sofa_02). Sample the FULL interval with the same
        # base+arm check the export uses (camera_reachable) and clamp both
        # bounds into the largest contiguous feasible run — the camera
        # traverses [r_s, r_e], so every radius in between must be feasible.
        d = np.asarray(seg._f("dir"), dtype=np.float64)
        c = seg._f("center")
        p_s, p_e = seg.free["r_s"], seg.free["r_e"]
        r_lo = min(float(p_s.lo), float(p_e.lo))
        r_hi = max(float(p_s.hi), float(p_e.hi))
        lo, hi = _dolly_feasible_run(c, d, float(seg.fix["cam_height"]),
                                     r_lo, r_hi, base_sdf, caps)
        if lo is None:
            raise ValueError(
                f"{seg.name}: 推拉线上无任何底盘+臂可行半径（净空/IK 全败，"
                f"臂展 {float(np.median(caps.arm_extension)):.2f}m）— "
                f"请换推拉方向或机位")
        for p in (p_s, p_e):
            nv = float(min(max(float(p.value), lo), hi))
            p.value, p.lo, p.hi = nv, max(float(p.lo), lo), min(float(p.hi), hi)


def _dolly_feasible_run(c, d, cam_height, r_lo, r_hi, base_sdf,
                        caps) -> tuple:
    """Largest feasible radius run on the dolly line — two-tier scan.

    Tier 1 (cheap): SDF + room-bounds check of the look-relative base parking
    spot at the median arm extension, 0.05 m grid → contiguous feasible runs.
    Tier 2 (exact): probe 5 points of the largest run with the FULL
    canonical-resolution check (``camera_reachable``, base parking + arm IK,
    all arm extensions), shrinking the ends inward until they pass; an
    interior probe failure truncates the run at the last passing probe.

    The look aims at the segment's real center (pitch matches the shot).
    Returns (lo, hi) of the verified run, or (None, None)."""
    from cinema_traj.core.jarvis import camera_reachable

    target = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    dm = float(np.median(caps.arm_extension))

    def _cam_pose(r_val):
        cam = target + float(r_val) * d
        cam[1] = cam_height
        to_t = target - cam
        n = float(np.linalg.norm(to_t))
        if n < 1e-6:
            return None, None, None
        yaw = math.atan2(to_t[0], to_t[2])
        pitch = math.asin(max(-1.0, min(1.0, to_t[1] / n)))
        return cam, yaw, pitch

    def _sdf_ok(r_val):
        cam, yaw, pitch = _cam_pose(r_val)
        if cam is None:
            return False
        look = (target - cam) / max(float(np.linalg.norm(target - cam)), 1e-9)
        bpt = np.array([cam[0] - look[0] * dm, 0.0, cam[2] - look[2] * dm])
        if float(base_sdf.numpy_query(bpt)) < caps.base_clearance_margin:
            return False
        bounds = base_sdf.room_bounds_of_point(bpt)
        if bounds is None:
            return True
        lo_b, hi_b = np.asarray(bounds[0]), np.asarray(bounds[1])
        return (bpt[0] >= lo_b[0] + C.SCENE_BOUNDS_MARGIN and
                bpt[0] <= hi_b[0] - C.SCENE_BOUNDS_MARGIN and
                bpt[2] >= lo_b[2] + C.SCENE_BOUNDS_MARGIN and
                bpt[2] <= hi_b[2] - C.SCENE_BOUNDS_MARGIN)

    def _full_ok(r_val):
        cam, yaw, pitch = _cam_pose(r_val)
        if cam is None:
            return False
        return camera_reachable(cam, yaw, pitch, base_sdf, tangent_yaw=None,
                                caps=caps) is not None

    # tier 1: SDF scan → contiguous runs
    feasible = [float(r) for r in np.arange(r_lo, r_hi + 1e-9, 0.05) if _sdf_ok(r)]
    if not feasible:
        return None, None
    runs: list = []
    run_lo = run_hi = feasible[0]
    for r_val in feasible[1:]:
        if r_val - run_hi <= 0.075:
            run_hi = r_val
        else:
            runs.append((run_lo, run_hi))
            run_lo = run_hi = r_val
    runs.append((run_lo, run_hi))
    runs.sort(key=lambda rh: rh[1] - rh[0], reverse=True)

    # tier 2: exact verification of the largest run, ends shrunk inward
    for run_lo, run_hi in runs[:2]:
        probes = np.linspace(run_lo, run_hi, 5)
        passing = [p for p in probes if _full_ok(float(p))]
        if len(passing) < 2:
            continue
        return float(min(passing)), float(max(passing))
    return None, None


# --------------------------------------------------------------------------- #
# 拍摄段硬门：anchor 候选 × (钳制 + 密采样可达性预审计)
# --------------------------------------------------------------------------- #
def _build_shooting_segment_verified(traj_step, anchor_list, obj, scene,
                                     sdf, base_sdf, caps, t_start) -> tuple:
    """Build the segment per anchor candidate, clamp it into the Jarvis
    workspace, then densely verify EVERY waypoint with the same base+arm IK
    check the export uses. First fully-reachable candidate wins; exhausting
    the candidates raises (the shot is not plannable for this robot — say so,
    never ship an unreachable shooting segment).

    Two-stage verify per candidate: a coarse sieve (fps/3) rejects badly
    placed anchors cheaply (an infeasible waypoint burns ~45 IK candidates,
    so full-density proof is reserved for candidates that pass the sieve)."""
    from cinema_traj.core.jarvis import verify_reachable_trajectory
    from cinema_traj.common import constants as C

    last_report = None
    for anchor in anchor_list:
        seg = make_segment(traj_step, anchor, obj, scene.subject, t_start, scene)
        if seg.type.startswith("orbit"):
            _clamp_orbit_radius_to_clearance(seg, sdf)
        _clamp_segment_to_jarvis_workspace(seg, sdf, caps, base_sdf=base_sdf,
                                           subject=scene.subject)
        if len(anchor_list) == 1:
            # single candidate: the sieve exists to reject bad anchors
            # cheaply across the candidate LOOP — with one candidate the
            # full-density verify below subsumes it (saves the fps/3 pass)
            sieve = {"status": "ok"}
        else:
            sieve = verify_reachable_trajectory([seg], scene.subject, base_sdf,
                                                caps, fps=C.DEFAULT_FPS / 3.0)
        if sieve["status"] != "ok":
            last_report = sieve
            continue
        report = verify_reachable_trajectory([seg], scene.subject, base_sdf, caps)
        last_report = report
        if report["status"] == "ok":
            return seg, anchor
    bad = (last_report or {}).get("violations") or [{}]
    v0 = bad[0]
    raise ValueError(
        f"{traj_step.get('type', '?')} → {traj_step.get('target', '?')}: "
        f"{len(anchor_list)} 个机位候选下均存在不可达采样点"
        f"（如 t={v0.get('t')}s pos={v0.get('pos')}）— "
        f"拍摄段不可达不允许规划，请换运镜类型或机位")


def _target_info(traj_step, obj, subject, t_start, scene):
    if traj_step["kind"] == "subject_traj":
        assert subject is not None
        sid = subject.subject_id
        pos = subject.state_at(t_start).pos
        room = next((r.id for r in scene.rooms if r.contains(pos)),
                    scene.rooms[0].id if scene.rooms else "")
        return room, "subject_head", sid, pos
    if obj is not None:
        return obj.room, "object", obj.id, obj.obb.center
    # object-level movement (e.g. 前推 move_in) aimed at the dynamic subject:
    # shot center = chest key point at t_start (or head — head-level rays
    # clear low console decor that chest-level rays graze); target_mode
    # keeps look_target tracking the (possibly moving) head frame by frame
    assert subject is not None, f"target '{traj_step['target']}' is neither object nor subject"
    sid = subject.subject_id
    pos = subject.state_at(t_start).pos
    room = next((r.id for r in scene.rooms if r.contains(pos)),
                scene.rooms[0].id if scene.rooms else "")
    key = "head" if traj_step.get("look_point") == "head" else "chest"
    return room, "subject_head", sid, subject.key_point(t_start, key)


def _room_top(scene: Scene, room_id: str) -> float:
    for r in scene.rooms:
        if r.id == room_id:
            return float(r.bounds_max[1])
    return 3.0


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build_trajectory(scene: Scene, plan: dict, jarvis_caps=None) -> tuple:
    """Returns (segments, anchors_debug). Raises if the plan is invalid.

    ``jarvis_caps`` (optional JarvisCapabilities) switches on the floor-robot
    reachability layer: anchors are filtered into the reachable camera band
    and every segment's geometry is clamped (heights, radius/base-circle,
    dolly ride height) before the optimizer ever sees it."""
    errs = validate_plan(plan, scene)
    if errs:
        raise ValueError("plan invalid:\n- " + "\n- ".join(errs))
    from cinema_traj.common.sdf import scene_sdf_cached
    sdf = scene_sdf_cached(scene)
    # jarvis base-placeability field: the base rides the floor, so floor/
    # ceiling structures must not zero its parking-spot SDF
    base_sdf = None
    if jarvis_caps is not None:
        from cinema_traj.core.jarvis import build_base_sdf
        base_sdf = build_base_sdf(scene)
    groups = group_steps(plan["steps"])
    subject_id = scene.subject.subject_id if scene.subject else None

    # -- pass A: times + anchors -------------------------------------------- #
    t = 0.0
    infos = []
    prev_end = None                     # previous segment's camera end pose —
                                        # the CONTINUITY anchor for the next
                                        # static shot (see below)
    for gi, (anchor_step, trans_step, traj_step) in enumerate(groups):
        trans_dur = float(trans_step.get("duration_s", 3.0)) if trans_step else 0.0
        traj_t_start = t + trans_dur
        traj_dur = float(traj_step.get("duration_s", C.DEFAULT_DURATION))
        tgt = traj_step["target"]
        if tgt == subject_id:
            obj = None
            if gi > 0 and prev_end is not None and                     anchor_step.get("azimuth_deg") is None:
                # CONTINUITY BY CONSTRUCTION: a static shot with no explicit
                # azimuth anchors AT the previous segment's end pose. Any
                # other choice puts the next shot a hop away from where the
                # robot actually stands — the relocation then needs a base
                # re-park under a static camera (arm folding against a
                # rotating base), which this plant tracks 30-70 cm off
                # (R36-R54), while zero-motion dwells track at ~2.6 cm
                # (arc_2, R52) and the swing's own deceleration re-park at
                # 1.3 cm (R40). The look (subject chest/head) is identical
                # on both sides, so the whole camera POSE is continuous and
                # the transition arc degenerates into the at-rest dwell the
                # sequence design intends.
                from cinema_traj.core.anchors import Anchor
                anchor = Anchor(np.asarray(prev_end, dtype=np.float64).copy(),
                                0.0, {"az_rel": 0.0, "elev": 0.0, "dist": 0.0})
                print(f"   🔗 beat {gi + 1}: anchor ← 前段终点 "
                      f"({prev_end[0]:.2f}, {prev_end[2]:.2f})（连续性）")
            else:
                anchor = A.select_subject_anchor(
                    scene.subject, traj_t_start, scene, sdf,
                    anchor_step.get("elevation", "medium"), anchor_step.get("distance", "medium"),
                    start_angle_deg=anchor_step.get("start_angle_deg"),
                    azimuth_deg=anchor_step.get("azimuth_deg"),
                    jarvis_caps=jarvis_caps)
            anchor_list = [anchor]
        else:
            obj = scene.object_by_id(tgt)
            anchor_list = A.select_anchor(obj, scene, sdf,
                                          anchor_step.get("elevation", "medium"),
                                          anchor_step.get("distance", "medium"),
                                          jarvis_caps=jarvis_caps)
            if not anchor_list:
                anchor_list = [A.Anchor(obj.obb.center + np.array([1.5, 0.3, 0.0]))]
        if jarvis_caps is not None:
            # 拍摄段硬门: anchor 候选逐个尝试（重建段 → 钳制 → 密采样可达性
            # 预审计），保证规划出的拍摄段不含不可达/不可跟踪的几何。全部
            # 候选失败 → 报错（不静默改运镜语义）。
            seg, anchor = _build_shooting_segment_verified(
                traj_step, anchor_list, obj, scene, sdf, base_sdf,
                jarvis_caps, traj_t_start)
        else:
            anchor = anchor_list[0]
            seg = make_segment(traj_step, anchor, obj, scene.subject,
                               traj_t_start, scene)
            if seg.type.startswith("orbit"):
                _clamp_orbit_radius_to_clearance(seg, sdf)
            elif seg.type in ("move_in", "move_out"):
                _clamp_dolly_run_to_clearance(seg, sdf)
        infos.append({"trans_step": trans_step, "trans_dur": trans_dur, "seg": seg,
                      "anchor": anchor, "target": tgt})
        prev_end = seg.endpoints(scene.subject)[1]
        t = traj_t_start + traj_dur

    # -- pass B: arcs between consecutive segments --------------------------- #
    segments: list = []
    for gi, info in enumerate(infos):
        if gi > 0 and info["trans_step"] is not None:
            seg = info["seg"]
            arc = ArcTransition(f"arc_{gi}", info["trans_dur"],
                                info["seg"].t_start - info["trans_dur"], seg.room,
                                seg.target_mode, info["target"])
            # look toward the destination while transitioning
            obj = scene.object_by_id(info["target"])
            if obj is not None:
                arc.fix["center"] = obj.obb.center.tolist()
            elif scene.subject is not None:
                arc.fix["center"] = scene.subject.key_point(seg.t_start, "head").tolist()
            segments.append(arc)
        segments.append(info["seg"])

    # rebuild arcs with concrete endpoints (paper: chord + horizontal normal)
    for i, seg in enumerate(segments):
        if seg.type != "arc":
            continue
        p_s = _end_pos(segments[i - 1], scene.subject) if i > 0 else np.zeros(3)
        p_e = _start_pos(segments[i + 1], scene.subject) if i + 1 < len(segments) else p_s + np.array([1.0, 0, 0])
        seg.rebuild(p_s, p_e)

    anchors_debug = {info["target"]: {"pos": np.asarray(info["anchor"].pos).tolist(),
                                      "score": round(float(info["anchor"].score), 4)}
                     for info in infos}
    return segments, anchors_debug


def _start_pos(seg: Segment, subject) -> np.ndarray:
    import torch
    th = torch.tensor([0.0], dtype=torch.float64)
    return seg.positions(th, None, subject).numpy()[0]


def _end_pos(seg: Segment, subject) -> np.ndarray:
    import torch
    th = torch.tensor([1.0], dtype=torch.float64)
    return seg.positions(th, None, subject).numpy()[0]


# --------------------------------------------------------------------------- #
# Stage entry
# --------------------------------------------------------------------------- #
def run(cfg=None, scene_path=None, plan_path=None, force=False) -> dict:
    cfg = cfg or load_config()
    ct = cfg.get("cinema_traj", {})
    scene = load_scene(scene_path or ct.get("scene", ""))
    out_dir = _out_dir(cfg, scene)
    plan_path = Path(plan_path) if plan_path else out_dir / "plan.json"
    plan = load_json(plan_path)
    out_path = out_dir / "traj_init.json"

    print(f"🔧 轨迹实例化: {len(plan['steps'])} 步计划")
    jarvis_caps = None
    if ct.get("platform", "drone") == "jarvis":
        from cinema_traj.core.jarvis import jarvis_caps_from_config
        jarvis_caps = jarvis_caps_from_config(ct.get("jarvis", {}))
        print(f"🤖 Jarvis 可达性层: 相机高度带 "
              f"[{jarvis_caps.cinematic_z_lo:.2f}, {jarvis_caps.cinematic_z_hi:.2f}]m，"
              f"臂展 {max(jarvis_caps.arm_extension):.2f}m")
    segments, anchors = build_trajectory(scene, plan, jarvis_caps)

    out = {
        "scene_id": scene.scene_id,
        "plan_id": plan.get("plan_id"),
        "anchors": anchors,
        "segments": [s.to_dict() for s in segments],
        "total_duration_s": round(sum(s.duration for s in segments), 2),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, out)
    print(f"✅ traj_init → {out_path}（{len(segments)} 段，总时长 "
          f"{out['total_duration_s']}s）")
    return out


def _out_dir(cfg, scene) -> Path:
    ct = cfg.get("cinema_traj", {})
    base = ct.get("outputs_dir", "cinema_traj_out")
    p = Path(base)
    if not p.is_absolute():
        from vlm_intention.common.config import repo_root
        p = repo_root() / p
    return p / scene.scene_id


if __name__ == "__main__":                                # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="参数化轨迹实例化（anchor + θ_init）")
    ap.add_argument("--scene")
    ap.add_argument("--plan", help="plan.json 路径（默认 <out>/<scene>/plan.json）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else None
    run(cfg, scene_path=a.scene, plan_path=a.plan, force=a.force)
