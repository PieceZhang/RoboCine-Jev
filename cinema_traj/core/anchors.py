"""Anchor selector (paper §3.3.1 + appendix D.3): face-normal-biased candidate
sampling on OBB faces, SDF collision filtering, five-term scoring

    S(pos) = w_vis·S_vis + w_dist·S_dist + w_elev·S_elev
           + w_open·S_open + w_perp·S_perp

S_vis measures OBB corner visibility via exact ray-OBB slab tests (engineering
substitute for the paper's ray casting — same semantics, exact, cheaper).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..common import constants as C
from ..common.obb import OBB, ray_obb_intersect
from ..common.scene import ObjectNode, Room, Scene
from ..common.se3 import rot_y, sph_offset
from ..common.sdf import SceneSDF
from ..common.subject import SubjectTrack


@dataclass
class Anchor:
    pos: np.ndarray
    score: float = 0.0
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Viewing distance
# --------------------------------------------------------------------------- #
def viewing_distance(obj: ObjectNode, room: Room, pref: str = "medium") -> float:
    """median extent × 1.2, clamped [0.3,3.0] m, ≤40% smallest horizontal room
    dim (paper D.3); LLM distance preference multiplies it."""
    if obj.viewing_distance:
        base = obj.viewing_distance
    else:
        base = float(np.median(obj.obb.extents)) * C.VIEWING_DIST_SCALE
    base = float(np.clip(base, *C.VIEWING_DIST_CLAMP))
    if room is not None:
        horiz = float(np.min(room.size[[0, 2]]))
        base = min(base, C.VIEWING_DIST_ROOM_FRAC * horiz)
    return max(base * C.DISTANCE_MULTIPLIERS.get(pref, 1.0), 0.25)


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #
def _accessible_face_axes(obb: OBB, placement: str,
                          struct_center: np.ndarray | None) -> list[tuple[int, float]]:
    """(axis_index, sign) face list. Faces pointing toward the attached
    structure are unreachable (inside the wall/ceiling) and are skipped; the
    bottom face is always skipped (below the floor)."""
    faces = []
    for ax in range(3):
        for sgn in (1.0, -1.0):
            if ax == 1 and sgn < 0:
                continue                      # never approach from below the floor
            if struct_center is not None:
                n_hat = obb.rotation[:, ax] * sgn
                to_struct = np.asarray(struct_center, dtype=np.float64) - obb.center
                if float(n_hat @ to_struct) > 0.5 * np.linalg.norm(to_struct) and \
                        np.linalg.norm(to_struct) > 1e-6:
                    continue                  # face flush against its structure
            faces.append((ax, sgn))
    return faces


def anchor_candidates(obj: ObjectNode, room: Room, placement: str,
                      sdf: SceneSDF, pref_dist: float,
                      elevation_band: tuple[float, float],
                      struct_center: np.ndarray | None = None) -> list[np.ndarray]:
    """Face-normal-biased candidates: 4 distance factors × 10 azimuth offsets
    (±45° around the face normal) × 3 elevations per face."""
    obb = obj.obb
    cands: list[np.ndarray] = []
    az_offsets = np.linspace(-C.AZIMUTH_OFFSET_SPAN_DEG, C.AZIMUTH_OFFSET_SPAN_DEG,
                             C.AZIMUTH_OFFSETS_COUNT)
    elevs = np.radians(np.linspace(elevation_band[0], elevation_band[1],
                                   C.ELEVATIONS_PER_BAND))
    for ax, sgn in _accessible_face_axes(obb, placement, struct_center):
        n_hat = obb.rotation[:, ax] * sgn
        # in-plane basis of the face
        u = obb.rotation[:, (ax + 1) % 3]
        v = obb.rotation[:, (ax + 2) % 3]
        face_center = obb.center + n_hat * obb.half[ax]
        for factor in C.DISTANCE_FACTORS:
            d = pref_dist * factor
            for az in az_offsets:
                azr = math.radians(az)
                for el in elevs:
                    # direction: face normal tilted by elevation, rotated by azimuth around it
                    dirv = n_hat * math.cos(el) + (u * math.cos(azr) + v * math.sin(azr)) * math.sin(el)
                    pos = face_center + dirv * d
                    if room is not None and not _inside_bounds(pos, room, C.SCENE_BOUNDS_MARGIN):
                        continue
                    if sdf.numpy_query(pos) < C.ANCHOR_COLLISION_MARGIN:
                        continue
                    cands.append(pos)
    return cands


def _inside_bounds(p: np.ndarray, room: Room, margin: float) -> bool:
    return bool(np.all(p >= room.bounds_min + margin) and np.all(p <= room.bounds_max - margin))


# --------------------------------------------------------------------------- #
# Scoring terms
# --------------------------------------------------------------------------- #
def visibility_score(pos: np.ndarray, target: OBB, blockers: list[OBB]) -> float:
    """Fraction of the 8 target corners with an unobstructed line of sight."""
    corners = target.corners()
    vis = 0
    for corner in corners:
        d = corner - pos
        dist = np.linalg.norm(d)
        if dist < 1e-9:
            vis += 1
            continue
        blocked = False
        for obb in blockers:
            t_e, t_x = ray_obb_intersect(pos[None], d[None], obb)
            if float(t_x[0]) > float(t_e[0]) and float(t_e[0]) > 1e-6 and float(t_e[0]) < 1.0:
                blocked = True
                break
        vis += 0 if blocked else 1
    return vis / 8.0


def _dist_score(pos, obj_center, pref_dist):
    d = np.linalg.norm(pos - obj_center)
    sigma = 0.35 * pref_dist
    return float(np.exp(-((d - pref_dist) ** 2) / (2.0 * sigma * sigma)))


def _elev_score(pos, obj_center, band):
    el = math.degrees(math.atan2(pos[1] - obj_center[1],
                                 np.linalg.norm((pos - obj_center)[[0, 2]])))
    lo, hi = band
    center = (lo + hi) / 2.0
    half = max((hi - lo) / 2.0, 1e-6)
    return float(max(0.0, 1.0 - abs(el - center) / (1.5 * half)))


def _open_score(pos, room):
    if room is None:
        return 0.5
    c = room.center
    half_diag = np.linalg.norm(room.size[[0, 2]]) / 2.0 + 1e-6
    return float(max(0.0, 1.0 - np.linalg.norm((pos - c)[[0, 2]]) / half_diag))


def _perp_score(pos, obj: ObjectNode, structures) -> float:
    """Wall-adjacent objects: reward viewing directions from the room interior,
    i.e. horizontal view dir aligned with the wall's inward normal."""
    n_wall = None
    for s in structures:
        if s.id == obj.attached_structure:
            n_wall = s.obb.largest_face_normal()
            break
    if n_wall is None:
        return 0.0
    horiz = np.array([n_wall[0], n_wall[2]])
    nh = np.linalg.norm(horiz)
    if nh < 1e-6:
        return 0.0
    view = np.array([(pos - obj.obb.center)[0], (pos - obj.obb.center)[2]])
    vv = np.linalg.norm(view)
    if vv < 1e-6:
        return 0.0
    # inward = away from the wall (whichever side the object center sits on)
    inward = horiz / nh * (1.0 if obj.obb.center[0] * n_wall[0] + obj.obb.center[2] * n_wall[2] >= 0 else -1.0)
    return float(max(0.0, (view / vv) @ inward))


def score_anchor(pos: np.ndarray, obj: ObjectNode, scene: Scene, sdf: SceneSDF,
                 pref_dist: float, band: tuple[float, float]) -> float:
    w = C.ANCHOR_WEIGHTS["wall"] if obj.placement in ("wall", "floor_wall") else \
        C.ANCHOR_WEIGHTS["freestanding"]
    blockers = [o for o in scene.collision_obbs() if o is not obj.obb]
    s_vis = visibility_score(pos, obj.obb, blockers)
    s_dist = _dist_score(pos, obj.obb.center, pref_dist)
    s_elev = _elev_score(pos, obj.obb.center, band)
    room = scene.room_of(obj)
    s_open = _open_score(pos, room)
    s_perp = _perp_score(pos, obj, scene.structures) if obj.placement in ("wall", "floor_wall") else 0.0
    return (w["vis"] * s_vis + w["dist"] * s_dist + w["elev"] * s_elev
            + w["open"] * s_open + w["perp"] * s_perp)


def select_anchor(obj: ObjectNode, scene: Scene, sdf: SceneSDF,
                  elevation_pref: str = "medium", distance_pref: str = "medium",
                  top_k: int = C.TOP_K_ANCHORS, jarvis_caps=None) -> list[Anchor]:
    """Score all candidates; return the top-k (optimizer restart options).

    ``jarvis_caps`` (optional) filters candidates into the floor robot's
    reachable camera height band before scoring — a drone eye-level anchor
    would be unreachable geometry no matter how it scores."""
    room = scene.room_of(obj)
    band = C.ELEVATION_BANDS_DEG.get(elevation_pref, C.ELEVATION_BANDS_DEG["medium"])
    pref_dist = viewing_distance(obj, room, distance_pref)
    struct_center = None
    if obj.attached_structure:
        for s in scene.structures:
            if s.id == obj.attached_structure:
                struct_center = s.obb.center
                break
    cands = anchor_candidates(obj, room, obj.placement, sdf, pref_dist, band, struct_center)
    if not cands:                             # fall back to a coarse ring around the object
        cands = _fallback_ring(obj, room, pref_dist)
    if jarvis_caps is not None:
        z_lo, z_hi = jarvis_caps.cinematic_z_band()
        banded = [p for p in cands if z_lo <= float(p[1]) <= z_hi]
        cands = banded or sorted(cands, key=lambda p: abs(float(p[1]) - (z_lo + z_hi) / 2.0))[:6]
    scored = [Anchor(p, score_anchor(p, obj, scene, sdf, pref_dist, band)) for p in cands]
    scored.sort(key=lambda a: a.score, reverse=True)
    return scored[:top_k] if top_k and scored else scored


def _fallback_ring(obj: ObjectNode, room: Room | None, pref_dist: float) -> list[np.ndarray]:
    """Uniform sparse ring around the object center (paper mentions a
    supplementary uniform set)."""
    out = []
    for az in range(0, 360, 30):
        for el in (10.0, 30.0):
            p = obj.obb.center + sph_offset(math.radians(az), math.radians(el), pref_dist)
            if room is None or _inside_bounds(p, room, 0.2):
                out.append(p)
    return out


# --------------------------------------------------------------------------- #
# Subject-relative anchor (extension)
# --------------------------------------------------------------------------- #
def select_subject_anchor(subject: SubjectTrack, t: float, scene: Scene, sdf: SceneSDF,
                          elevation_pref: str = "medium",
                          distance_pref: str = "medium",
                          start_angle_deg: float | None = None,
                          azimuth_deg: float | None = None,
                          jarvis_caps=None) -> Anchor:
    """Best spherical offset relative to the subject's heading at time t.

    Samples azimuths around the heading (front/sides/over-shoulder), the LLM
    elevation band, and preference-scaled distances; scores by SDF clearance,
    visibility of the head key point, and room openness.

    ``jarvis_caps`` (optional): candidates outside the floor robot's reachable
    camera height band are skipped, and the last-resort fallback is clamped
    into the band.

    Frame of reference: by default the candidate sphere is centered on the
    subject's FEET (ground position, y≈0) with elevations from the fuzzy
    label band. When ``start_angle_deg`` is given (recommend contract: initial
    gimbal pitch, degrees, +up), the sphere is instead centered on the HEAD
    key point at exactly elevation = −start_angle_deg — a level (水平, 0°)
    recommendation then yields candidates at face height.

    ``azimuth_deg`` pins the azimuth to ONE heading-relative value (0° = in
    front of the subject, 180° = behind) instead of sampling the 12-way
    fan — hand-authored plans (and tests) use it to demand 后方/前方 shots
    deterministically; scoring still picks elevation/distance/obstruction.
    """
    subj = subject.state_at(t)
    subject_feet = subj.pos                     # ground point, y≈0
    subject_heading = subj.heading
    head_point = subject.key_point(t, "head")   # loop-invariant look target

    # sphere center + elevation axis: exact contract angle OR fuzzy label band
    if start_angle_deg is not None:
        # a camera pitched DOWN p° sits p° above the target (and conversely):
        # elevation above the head = −start_angle; 水平 0° → face height
        sphere_center = head_point
        elev_candidates = [math.radians(-float(start_angle_deg))]
    else:
        sphere_center = subject_feet            # legacy label path, unchanged
        band = C.ELEVATION_BANDS_DEG.get(elevation_pref, C.ELEVATION_BANDS_DEG["medium"])
        elev_candidates = [math.radians(e) for e in np.linspace(band[0], band[1], 3)]
    if distance_pref in C.DISTANCE_MULTIPLIERS:
        dist_candidates = [max(0.25, 1.6 * C.DISTANCE_MULTIPLIERS[distance_pref] * m)
                           for m in (0.8, 1.0, 1.3)]
    else:
        dist_candidates = [max(0.25, 1.2 * m) for m in (0.7, 1.0, 1.4)]
    if azimuth_deg is not None:
        azimuth_candidates = [math.radians(float(azimuth_deg))]
    else:
        azimuth_candidates = [0.0, math.radians(30), math.radians(-30), math.radians(60),
                              math.radians(-60), math.radians(90), math.radians(-90),
                              math.radians(120), math.radians(-120), math.radians(165),
                              math.radians(-165), math.radians(180)]

    best_anchor, best_score = None, -1e9
    subject_room = next((r for r in scene.rooms if r.contains(subject_feet)), None)
    jband = jarvis_caps.cinematic_z_band() if jarvis_caps is not None else None
    for az_rel in azimuth_candidates:
        for elev_rad in elev_candidates:
            offset_dir = rot_y(-subject_heading) @ sph_offset(az_rel, elev_rad, 1.0)
            for dist in dist_candidates:
                cam_pos = sphere_center + offset_dir * dist
                if jband is not None:
                    # start_angle_deg candidates sit at the HEAD key point
                    # (world y up to ~1.0 on raised floors) — clamp the height
                    # into the reachable band instead of skipping the whole
                    # candidate; the pitch contract is preserved in the az/
                    # elev detail, only the shot height yields to the robot
                    cam_pos = cam_pos.copy()
                    cam_pos[1] = min(max(cam_pos[1], jband[0]), jband[1])
                if subject_room is not None and \
                        not _inside_bounds(cam_pos, subject_room, 0.2):
                    continue
                # clearance of the anchor AND the approach line to the
                # subject — dollies/orbits fly that line, and with the real
                # cloud in the field an anchor can sit clear while the line
                # grazes geometry between it and her
                clearance = _line_clearance(sdf, cam_pos, head_point)
                if clearance < C.ANCHOR_COLLISION_MARGIN:
                    continue
                head_visible = _point_visible(cam_pos, head_point, scene.collision_obbs())
                score = 0.5 * head_visible + 0.3 * min(clearance / 0.5, 1.0) \
                    + 0.2 * _open_score(cam_pos, subject_room)
                if score > best_score:
                    best_score = score
                    if start_angle_deg is not None:
                        # detail stays FEET-relative: build.py inits the track-
                        # segment spherical offsets against the ground track
                        # (d·sin(el) from feet), so convert the head-relative win
                        elev_detail = math.asin(max(-1.0, min(1.0,
                                            (cam_pos[1] - subject_feet[1]) / dist)))
                    else:
                        elev_detail = elev_rad   # legacy branch: byte-identical
                    best_anchor = Anchor(cam_pos, score,
                                         {"az_rel": az_rel, "elev": elev_detail,
                                          "dist": dist, "az_pinned": azimuth_deg is not None})
    if best_anchor is None:                      # last resort: 1.5 m in front
        if start_angle_deg is not None:
            cam_pos = head_point + rot_y(-subject_heading) @ np.array([1.5, 0.0, 0.0])
            if jband is not None:                # clamp into the reachable band
                cam_pos[1] = min(max(cam_pos[1], jband[0]), jband[1])
            elev_feet = math.asin(max(-1.0, min(1.0,
                                  (cam_pos[1] - subject_feet[1]) / 1.5)))
            best_anchor = Anchor(cam_pos, 0.0,
                                 {"az_rel": 0.0, "elev": elev_feet, "dist": 1.5})
        else:                                    # legacy fallback, unchanged
            cam_pos = subject_feet + rot_y(-subject_heading) @ np.array([1.5, 0.3, 0.0])
            if jband is not None:
                cam_pos[1] = min(max(cam_pos[1], jband[0]), jband[1])
            best_anchor = Anchor(cam_pos, 0.0,
                                 {"az_rel": 0.0, "elev": math.radians(10), "dist": 1.5})
    return best_anchor



def _line_clearance(sdf: SceneSDF, cam_pos: np.ndarray,
                    target: np.ndarray, n: int = 6) -> float:
    """Min SDF clearance along cam_pos→target (target end excluded — the
    subject sits there). Samples the anchor point plus n interior points."""
    d = np.asarray(target, float) - np.asarray(cam_pos, float)
    ts = np.linspace(0.0, 0.8, n)                     # stop short of her
    pts = np.asarray(cam_pos, float)[None, :] + ts[:, None] * d[None, :]
    return float(sdf.numpy_query(pts).min())

def _point_visible(pos: np.ndarray, target: np.ndarray, blockers: list[OBB]) -> float:
    d = target - pos
    dist = np.linalg.norm(d)
    if dist < 1e-9:
        return 1.0
    for obb in blockers:
        t_e, t_x = ray_obb_intersect(pos[None], d[None], obb)
        if float(t_x[0]) > float(t_e[0]) and float(t_e[0]) > 1e-6 and float(t_e[0]) < 1.0:
            return 0.0
    return 1.0


def points_visible_batch(pos: np.ndarray, ray_targets: np.ndarray,
                         excl_points: np.ndarray, blockers: list[OBB]) -> np.ndarray:
    """Vectorized _point_visible over N frames (bool: True = visible).

    ``ray_targets``: where each frame's visibility ray ends (the look point,
    or the subject head for LED-look segments). ``excl_points``: the
    per-frame containment-exclusion point (the look point — an OBB that
    CONTAINS it is being framed, not blocking). One batched slab test per
    BLOCKER over all frames instead of one python call per frame×blocker.
    """
    d = np.asarray(ray_targets, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    dist = np.linalg.norm(d, axis=1)
    vis = np.ones(len(pos), dtype=bool)
    for obb in blockers:
        t_e, t_x = ray_obb_intersect(pos, d, obb)
        hit = ((t_x > t_e) & (t_e > 1e-6) & (t_e < 1.0) & (dist > 1e-9))
        if not hit.any():
            continue
        inside = obb.contains(excl_points)
        vis[hit & ~inside] = False
    return vis


# --------------------------------------------------------------------------- #
# Visitation order (paper §3.3.3)
# --------------------------------------------------------------------------- #
def nearest_neighbor_order(start_pos: np.ndarray, objects: list[ObjectNode]) -> list[str]:
    """Deterministic nearest-neighbor visitation when the user gives no order."""
    remaining = list(objects)
    order = []
    cur = np.asarray(start_pos, dtype=np.float64)
    while remaining:
        nxt = min(remaining, key=lambda o: np.linalg.norm(o.obb.center - cur))
        order.append(nxt.id)
        remaining.remove(nxt)
        cur = nxt.obb.center
    return order
