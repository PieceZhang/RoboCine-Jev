"""Parametric atomic trajectory library (paper §3.3 + subject-relative extension).

Every segment is τ(t; θ_fix, θ_free): fixed params come from anchors/geometry,
free params are optimizable with per-parameter (value, lo, hi) bounds. All
positions are evaluated in torch so the SDF optimizer gets gradients; passing
``params=None`` evaluates a detached snapshot (used by build/export/viewer).

Segment types:
  paper §D.1 — orbit_full/half/quarter, pan_left/right, tilt_up/down,
               move_in/out, crane, zoom_in_out/out_in, static
  paper §3.3.2 — arc (transition)
  ours       — track_follow / track_orbit / track_lead (dynamic subject)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

from ..common import constants as C
from ..common.se3 import ease_in_out, sph_offset
from ..common.subject import SubjectTrack


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
@dataclass
class Param:
    value: float
    lo: float
    hi: float


class Segment:
    type = "base"
    # free params optimized in pass 1 (subset preserves shot style, paper §D.5)
    optimize_names: tuple = ()

    def __init__(self, name: str, duration: float, t_start: float, room: str,
                 target_mode: str, target_id: str,
                 fix: dict | None = None, free: dict[str, Param] | None = None):
        self.name = name
        self.duration = float(duration)
        self.t_start = float(t_start)
        self.room = room
        self.target_mode = target_mode        # "object" | "subject_head" | "subject_chest"
        self.target_id = target_id
        self.fix: dict = fix or {}
        self.free: dict[str, Param] = free or {}

    # -- free-parameter plumbing ----------------------------------------- #
    @property
    def free_names(self) -> list:
        return list(self.free.keys())

    def free_flat(self):
        return torch.tensor([self.free[n].value for n in self.free_names])

    def params_from_flat(self, theta):
        """theta: tensor of ALL free params → {name: tensor} (optimizer works on
        a subset; this maps flat vectors back)."""
        return {n: theta[i] for i, n in enumerate(self.free_names)}

    def write_back(self, params: dict) -> None:
        for n, v in params.items():
            p = self.free[n]
            p.value = float(v.clamp(p.lo, p.hi))

    def _p(self, name: str, params: dict | None):
        """Free param as a float64 torch tensor (graph-connected when the
        optimizer passes its leaf; detached snapshot otherwise)."""
        if params is not None and name in params:
            return params[name].double()
        p = self.free[name]
        return torch.tensor(p.value, dtype=torch.float64)

    def _f(self, key: str):
        return np.asarray(self.fix[key], dtype=np.float64)

    # -- evaluation ------------------------------------------------------- #
    def positions(self, th, params: dict | None = None, subject=None):
        raise NotImplementedError

    def look_target(self, th, t_abs, subject: SubjectTrack | None = None):
        """Look-at points (F,3). Object segments: fixed center. Subject
        segments: key point at absolute time. Pan/tilt override this."""
        if self.target_mode == "object":
            c = torch.tensor(self._f("center"), dtype=torch.float64)
            return c.expand(th.shape[0], 3)
        name = "head" if self.target_mode == "subject_head" else "chest"
        pts = [subject.key_point(float(t), name) for t in t_abs.tolist()]
        return torch.tensor(np.asarray(pts), dtype=torch.float64)

    def fov_deg(self, th):                    # default: constant FOV
        return None

    def sample_times(self, fps: float | None = None):
        fps = fps or C.DEFAULT_FPS
        n = max(int(round(self.duration * fps)) + 1, 2)
        th = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        t_abs = self.t_start + th * self.duration
        return th, t_abs

    def endpoints(self, subject=None) -> tuple[np.ndarray, np.ndarray]:
        th = torch.tensor([0.0, 1.0], dtype=torch.float64)
        p = self.positions(th, None, subject).numpy()
        return p[0].copy(), p[1].copy()

    def reachable_aabb(self, subject: "SubjectTrack | None" = None):
        """Conservative axis-aligned bound on ALL positions attainable under
        ANY free-parameter value within its [lo, hi] bounds (None = unknown →
        the optimizer must use the full scene SDF). Used for per-segment box
        pruning; overriding types MUST keep the bound conservative."""
        return None

    # -- serialization ---------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "name": self.name, "type": self.type, "duration_s": self.duration,
            "t_start": self.t_start, "room": self.room,
            "target_mode": self.target_mode, "target_id": self.target_id,
            "fix": _jsonify(self.fix),
            "free": {n: [p.value, p.lo, p.hi] for n, p in self.free.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        sub = REGISTRY.get(d["type"])
        if sub is None:
            raise ValueError(f"unknown segment type '{d['type']}'")
        seg = sub.__new__(sub)
        Segment.__init__(seg, d["name"], d["duration_s"], d.get("t_start", 0.0),
                         d.get("room", ""), d.get("target_mode", "object"),
                         d.get("target_id", ""), _unjsonify(d.get("fix", {})),
                         {n: Param(*v) for n, v in d.get("free", {}).items()})
        return seg


# --------------------------------------------------------------------------- #
# Stationary / orientation-only shots
# --------------------------------------------------------------------------- #
class StaticSegment(Segment):
    type = "static"
    optimize_names = ()

    def positions(self, th, params=None, subject=None):
        p = torch.tensor(self._f("anchor"), dtype=torch.float64)
        return p.expand(th.shape[0], 3).clone()


class _StationaryLookSegment(Segment):
    """Shared base for pan/tilt: camera parked at the anchor, look direction
    rotated from center by a yaw/pitch sweep. positions() is constant."""
    optimize_names = ()

    def __init__(self, *a, sweep_deg: float = 45.0, **kw):
        super().__init__(*a, **kw)
        self.fix["sweep_deg"] = float(sweep_deg)

    def positions(self, th, params=None, subject=None):
        p = torch.tensor(self._f("anchor"), dtype=torch.float64)
        return p.expand(th.shape[0], 3).clone()


class PanSegment(_StationaryLookSegment):
    type = "pan_left"          # pan_right uses sweep_deg < 0

    def look_target(self, th, t_abs, subject=None):
        a = self._f("anchor"); c = self._f("center")
        v0 = torch.tensor(c - a, dtype=torch.float64)
        if self.fix.get("sweep_return"):
            # there-and-back sweep (0→sweep→0, rest-to-rest): the shot ends
            # at its start pose, so the following transition arc has NO look
            # swing — the base park never chases the gimbal and the next
            # segment starts clean (measured: a one-way 30° pan made the
            # return arc's base yaw teleport at 29 rad/s ≈ 37× the cap and
            # bled 30-40 cm of error into the next two segments)
            u = ease_in_out(2.0 * th)
            u = torch.where(th < 0.5, u, ease_in_out(2.0 * (1.0 - th)))
        else:
            u = ease_in_out(th)
        ang = math.radians(self.fix["sweep_deg"]) * u
        ca, sa = torch.cos(ang), torch.sin(ang)
        x = ca * v0[0] + sa * v0[2]
        z = -sa * v0[0] + ca * v0[2]
        v = torch.stack([x, v0[1].expand_as(x), z], dim=-1)
        return torch.tensor(a, dtype=torch.float64) + v


class TiltSegment(_StationaryLookSegment):
    type = "tilt_up"           # tilt_down uses sweep_deg < 0

    def look_target(self, th, t_abs, subject=None):
        a = self._f("anchor"); c = self._f("center")
        v0 = c - a
        n = np.linalg.norm(v0)
        v0u = v0 / max(n, 1e-9)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(v0u, up)
        if np.linalg.norm(right) < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        # spherical sweep: keep |v| = n, elevation offset by sweep
        el0 = math.asin(max(-1.0, min(1.0, v0u[1])))
        ang = math.radians(self.fix["sweep_deg"]) * ease_in_out(th).numpy()
        pts = []
        for dth in ang:
            el = el0 + float(dth)
            az = math.atan2(v0u[2], v0u[0])
            pts.append(a + sph_offset(az, el, n))
        return torch.tensor(np.asarray(pts), dtype=torch.float64)


class ZoomSegment(Segment):
    """Stationary camera, focal-length envelope f(t)=f0[1+(m-1)sin(πt̂)] (paper
    eq. §3.3.2). Expressed as FOV for export; zoom_out_in is m<1 first half."""
    type = "zoom_in_out"
    optimize_names = ()

    def __init__(self, *a, m_peak: float = C.ZOOM_M_PEAK, fov0_deg: float = 55.0, **kw):
        super().__init__(*a, **kw)
        self.fix["m_peak"] = float(m_peak)
        self.fix["fov0_deg"] = float(fov0_deg)

    def positions(self, th, params=None, subject=None):
        p = torch.tensor(self._f("anchor"), dtype=torch.float64)
        return p.expand(th.shape[0], 3).clone()

    def fov_deg(self, th):
        m = self.fix["m_peak"]; f0 = self.fix["fov0_deg"]
        ratio = 1.0 + (m - 1.0) * torch.sin(math.pi * th)
        half0 = torch.tensor(math.radians(f0) / 2.0, dtype=th.dtype)
        half = torch.atan(torch.tan(half0) / ratio)
        return torch.rad2deg(2.0 * half)


# --------------------------------------------------------------------------- #
# Object-level moving shots
# --------------------------------------------------------------------------- #
class OrbitLeadMixin:
    """Look target for a constant-radius orbit: exact centering needs
    look⊥travel = 90° of arm yaw — measured DEAD at the seated camera
    band (arm window ±75°). Leading the look 15-25° along the tangent
    keeps the angle inside the window with the subject consistently
    framed slightly off optical axis (well inside a 55° FOV)."""
    def _lead_look(self, th, subject, c):
        import torch as _t
        head = np.asarray(subject.key_point(0.0, "head"), dtype=np.float64)
        lead = math.radians(float(self.fix.get("lead_deg", 0.0)))
        if abs(lead) < 1e-6 or subject is None:
            return None
        phi_s = float(self.fix["phi_s"])
        direction = float(self.fix.get("direction", 1.0))
        sweep = direction * math.radians(float(self.fix.get("sweep_deg", 360.0)))
        r = float(self.free["radius"].value) if "radius" in self.free else 1.0
        u = ease_in_out(th).numpy() if hasattr(th, "numpy") else np.asarray(th)
        phis = phi_s + sweep * u
        tang = np.stack([-np.sin(phis) * direction, np.zeros_like(phis),
                         np.cos(phis) * direction], axis=1)
        pts = head[None, :] + (r * math.sin(lead)) * tang
        return _t.tensor(pts, dtype=_t.float64)


class OrbitSegment(OrbitLeadMixin, Segment):
    """p(t) = (c.x + r·cosφ, h, c.z + r·sinφ), φ: φ_s→φ_e (direction encoded in
    φ_e sign of sweep). Free: radius, height, end_angle (paper: r,h,φ_e)."""
    type = "orbit_quarter"
    optimize_names = ("radius", "height", "end_angle")

    def __init__(self, *a, direction: float = 1.0, sweep_deg: float = 90.0,
                 lead_deg: float = 0.0, **kw):
        super().__init__(*a, **kw)
        self.fix["direction"] = float(direction)
        self.fix["sweep_deg"] = float(sweep_deg)
        if lead_deg:
            self.fix["lead_deg"] = float(lead_deg)

    def reachable_aabb(self, subject=None):
        # |p − c| in xz is exactly r; y is exactly h — the cylinder's AABB
        r = max(abs(self.free["radius"].hi), abs(self.free["radius"].lo))
        c = self._f("center")
        lo = np.array([c[0] - r, self.free["height"].lo, c[2] - r])
        hi = np.array([c[0] + r, self.free["height"].hi, c[2] + r])
        return lo, hi

    def look_target(self, th, t_abs, subject=None):
        if self.fix.get("lead_deg") and subject is not None:
            lead = self._lead_look(th, subject, self._f("center"))
            if lead is not None:
                return lead
        return super().look_target(th, t_abs, subject)

    def positions(self, th, params=None, subject=None):
        c = self._f("center")
        r = self._p("radius", params)
        h = self._p("height", params)
        phi_s = self.fix["phi_s"]
        phi_e = self._p("end_angle", params)
        # Directed sweep WITHOUT angle wrapping: end_angle's bounds
        # (az0 + dir·[0.3, 1.6]·sweep) keep it on one side of phi_s, so the
        # raw difference is continuous and differentiable. Wrapping here
        # would fold ±360° orbits to a zero span (and float() used to detach
        # the graph, freezing end_angle in the optimizer).
        span = phi_e - phi_s
        u = ease_in_out(th)
        phi = phi_s + span * u
        ct = torch.as_tensor(c, dtype=torch.float64)
        x = ct[0] + r * torch.cos(phi)
        z = ct[2] + r * torch.sin(phi)
        y = h.expand(x.shape)
        return torch.stack([x, y, z], dim=-1)


class DollySegment(Segment):
    """p(t) = c + d̂·lerp(r_s, r_e, t̂), d̂ = anchor approach direction."""
    type = "move_in"
    optimize_names = ("r_s", "r_e")

    def reachable_aabb(self, subject=None):
        # the reachable set is the segment between the extreme radii on the
        # FIXED direction line — exact
        c = self._f("center")
        d = self._f("dir")
        rs, re = self.free["r_s"], self.free["r_e"]
        r_lo = min(rs.lo, rs.hi, re.lo, re.hi)
        r_hi = max(rs.lo, rs.hi, re.lo, re.hi)
        pts = np.stack([c + r_lo * d, c + r_hi * d])
        if "cam_height" in self.fix:
            # jarvis height clamp: y is pinned for every radius
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            lo[1] = hi[1] = float(self.fix["cam_height"])
            return lo, hi
        return pts.min(axis=0), pts.max(axis=0)

    def positions(self, th, params=None, subject=None):
        c = torch.tensor(self._f("center"), dtype=torch.float64)
        d = torch.tensor(self._f("dir"), dtype=torch.float64)
        rs = self._p("r_s", params)
        re = self._p("r_e", params)
        u = ease_in_out(th)
        r = rs + (re - rs) * u
        p = c + r.unsqueeze(-1) * d
        if "cam_height" in self.fix:
            # jarvis reachability clamp: camera rides at the clamped height
            p = torch.stack([p[..., 0],
                             torch.full_like(p[..., 1], float(self.fix["cam_height"])),
                             p[..., 2]], dim=-1)
        return p


class CraneSegment(Segment):
    """Vertical arc: elevation ε_s→ε_e at fixed azimuth & radius around c."""
    type = "crane"
    optimize_names = ("radius", "eps_e")

    def reachable_aabb(self, subject=None):
        # spherical-shell radius bound in all axes — conservative
        r = max(abs(self.free["radius"].hi), abs(self.free["radius"].lo))
        c = self._f("center")
        return c - r, c + r

    def positions(self, th, params=None, subject=None):
        c = torch.tensor(self._f("center"), dtype=torch.float64)
        az = self.fix["az"]
        r = self._p("radius", params)
        eps_s = self.fix["eps_s"]
        eps_e = self._p("eps_e", params)
        u = ease_in_out(th)
        eps = eps_s + (eps_e - eps_s) * u
        ct, st = torch.cos(eps), torch.sin(eps)
        ca, sa = math.cos(az), math.sin(az)
        x = c[0] + r * ct * ca
        y = c[1] + r * st
        z = c[2] + r * ct * sa
        return torch.stack([x, y, z], dim=-1)


class ArcTransition(Segment):
    """Transition arc (paper eq. 4): p(t)=(1-t̂)p_s+t̂p_e+(L/2)tan(α)sin(πt̂)n̂."""
    type = "arc"
    optimize_names = ("alpha_deg",)

    def positions(self, th, params=None, subject=None):
        ps = torch.tensor(self._f("p_s"), dtype=torch.float64)
        pe = torch.tensor(self._f("p_e"), dtype=torch.float64)
        n = torch.tensor(self._f("n_hat"), dtype=torch.float64)
        L = self.fix["chord"]
        alpha = torch.deg2rad(self._p("alpha_deg", params))
        u = ease_in_out(th)
        bulge = (L / 2.0) * torch.tan(alpha) * torch.sin(math.pi * th)
        return (1 - u).unsqueeze(-1) * ps + u.unsqueeze(-1) * pe + bulge.unsqueeze(-1) * n

    def rebuild(self, p_s: np.ndarray, p_e: np.ndarray) -> "ArcTransition":
        """Pass 2: fresh arc between updated endpoints (alpha reset to 0)."""
        chord_v = p_e - p_s
        L = float(np.linalg.norm(chord_v))
        nh = np.cross(chord_v, [0.0, 1.0, 0.0]) if L > 1e-6 else np.array([1.0, 0.0, 0.0])
        nn = np.linalg.norm(nh)
        n_hat = nh / nn if nn > 1e-6 else np.array([1.0, 0.0, 0.0])
        self.fix["p_s"], self.fix["p_e"], self.fix["chord"], self.fix["n_hat"] = \
            p_s.tolist(), p_e.tolist(), max(L, 1e-6), n_hat.tolist()
        self.free["alpha_deg"] = Param(0.0, -89.0, 89.0)
        return self

    def look_target(self, th, t_abs, subject=None):
        """Look toward the destination during transitions: fixed center for
        object targets, tracked head point for subject targets."""
        if self.target_mode.startswith("subject"):
            return super().look_target(th, t_abs, subject)
        c = self.fix.get("center", self.fix.get("p_e", [0.0, 0.0, 0.0]))
        ct = torch.tensor(np.asarray(c, dtype=np.float64), dtype=torch.float64)
        return ct.expand(th.shape[0], 3)


class SwingSegment(ArcTransition):
    """SHOOTING spiral swing — the differential-base "orbit".

    A constant-radius orbit needs look⊥travel = 90° of arm yaw at every
    sample; under the physical gate (driving base heading = path tangent)
    that is unreachable for this arm at ANY radius (verified by a full
    radius×azimuth reachability matrix). A swing instead travels a chord
    that closes distance while crossing azimuth, so the look-vs-travel
    angle peaks well inside the arm's window — cinematically an
    "orbit-in": the camera arcs around the subject while easing closer,
    looking at her the whole time. Geometry = the arc primitive
    (chord + α bulge); semantics = a shooting segment (hard gates, no
    dwell exemption — the name must NOT start with 'arc_' so the export's
    hold-detection keeps treating it as shooting)."""
    type = "arc_swing"

    def look_target(self, th, t_abs, subject=None):
        # lead_look_offset: a LATERAL TRUCK looking straight at the subject
        # needs 90° of arm yaw (dead window, same physics as the orbit) —
        # offsetting the look point along the travel direction tilts the
        # look-vs-travel angle inside the ±75° window with the subject
        # consistently framed slightly off optical axis.
        off = self.fix.get("lead_look_offset")
        if off and subject is not None:
            head = np.asarray(subject.key_point(0.0, "head"), dtype=np.float64)
            pts = (head + np.asarray(off, dtype=np.float64))[None, :]                 .repeat(th.shape[0], axis=0)
            return torch.tensor(pts, dtype=torch.float64)
        return super().look_target(th, t_abs, subject)


# --------------------------------------------------------------------------- #
# Subject-relative tracking shots (our extension)
# --------------------------------------------------------------------------- #
def _heading_offset(heading_t, az_rel, el, dist):
    """World offset from the subject: rotate the local spherical offset
    (az_rel=0 → in front of the subject) into the heading frame."""
    ce = torch.cos(el)
    x = dist * ce * torch.cos(az_rel)
    y = dist * torch.sin(el)
    z = dist * ce * torch.sin(az_rel)
    # rotate by -heading around Y (matches numpy rot_y(-h))
    c, s = torch.cos(-heading_t), torch.sin(-heading_t)
    xr = c * x + s * z
    zr = -s * x + c * z
    y = torch.broadcast_to(y, xr.shape).clone() if y.dim() < xr.dim() else y
    return torch.stack([xr, y, zr], dim=-1)


def _subject_pos_t(subject: SubjectTrack, t_abs):
    return torch.tensor(np.asarray([subject.state_at(float(t)).pos for t in t_abs.tolist()]),
                        dtype=torch.float64)


def _subject_heading_t(subject: SubjectTrack, t_abs):
    return torch.tensor([subject.state_at(float(t)).heading for t in t_abs.tolist()],
                        dtype=torch.float64)


class TrackFollowSegment(Segment):
    """Constant spherical offset (azimuth relative to heading, elevation,
    distance) held while the subject moves. Optional ``elev_end`` (fix,
    radians) ramps the elevation to that value across the shot with the
    usual ease — 后方跟随上移 without leaving the follow primitive."""
    type = "track_follow"
    optimize_names = ("az_rel", "elev", "dist")
    target_mode = "subject_head"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.target_mode = self.target_mode if self.target_mode != "object" else "subject_head"
        self.fix["key_point"] = self.fix.get("key_point", "head")

    def reachable_aabb(self, subject=None):
        # the heading-frame offset has norm exactly `dist` for any az/elev —
        # bound = subject path AABB inflated by the max |dist|
        if subject is None:
            return None
        d = max(abs(self.free["dist"].hi), abs(self.free["dist"].lo))
        pts = np.asarray([np.asarray(f["pos"], float) for f in subject.frames])
        return pts.min(axis=0) - d, pts.max(axis=0) + d

    def positions(self, th, params=None, subject=None):
        if subject is None:
            raise ValueError("track segments require the subject track")
        t_abs = self.t_start + th * self.duration
        pos = _subject_pos_t(subject, t_abs)
        heading = _subject_heading_t(subject, t_abs)
        az = self._p("az_rel", params)
        el = self._p("elev", params)
        d = self._p("dist", params)
        if "elev_end" in self.fix:
            el = el + (float(self.fix["elev_end"]) - el) * ease_in_out(th)
        return pos + _heading_offset(heading, az, el, d)


class TrackOrbitSegment(Segment):
    """Azimuth sweeps az_s→az_e (relative to heading) while following."""
    type = "track_orbit"
    optimize_names = ("az_e", "elev", "dist")
    target_mode = "subject_head"

    def reachable_aabb(self, subject=None):
        # same norm-=`dist` bound as track_follow (az sweep never lengthens it)
        if subject is None:
            return None
        d = max(abs(self.free["dist"].hi), abs(self.free["dist"].lo))
        pts = np.asarray([np.asarray(f["pos"], float) for f in subject.frames])
        return pts.min(axis=0) - d, pts.max(axis=0) + d

    def positions(self, th, params=None, subject=None):
        if subject is None:
            raise ValueError("track segments require the subject track")
        t_abs = self.t_start + th * self.duration
        pos = _subject_pos_t(subject, t_abs)
        heading = _subject_heading_t(subject, t_abs)
        az_s = self._p("az_s", params)
        az_e = self._p("az_e", params)
        el = self._p("elev", params)
        d = self._p("dist", params)
        u = ease_in_out(th)
        az = az_s + (az_e - az_s) * u
        return pos + _heading_offset(heading, az, el, d)


class TrackLeadSegment(TrackFollowSegment):
    """Offset held at a fixed lead angle ahead of the heading — lead ≈ ±165°
    gives over-the-shoulder."""
    type = "track_lead"
    optimize_names = ("lead_angle", "elev", "dist")

    def positions(self, th, params=None, subject=None):
        lead = self._p("lead_angle", params)
        return super().positions(th, {"az_rel": lead, **{k: v for k, v in (params or {}).items()
                              if k != "lead_angle"}}, subject)


# --------------------------------------------------------------------------- #
# Registry + pass-1 free subsets
# --------------------------------------------------------------------------- #
REGISTRY: dict[str, type[Segment]] = {c.type: c for c in [
    StaticSegment, PanSegment, TiltSegment, ZoomSegment,
    OrbitSegment, DollySegment, CraneSegment, ArcTransition, SwingSegment,
    TrackFollowSegment, TrackOrbitSegment, TrackLeadSegment,
]}
REGISTRY["pan_right"] = PanSegment
REGISTRY["tilt_down"] = TiltSegment
REGISTRY["zoom_out_in"] = ZoomSegment
REGISTRY["orbit_full"] = OrbitSegment
REGISTRY["orbit_half"] = OrbitSegment
REGISTRY["move_out"] = DollySegment

FREE_SUBSET: dict[str, tuple] = {
    "orbit_full": ("radius", "height", "end_angle"),
    "orbit_half": ("radius", "height", "end_angle"),
    "orbit_quarter": ("radius", "height", "end_angle"),
    "move_in": ("r_s", "r_e"),
    "move_out": ("r_s", "r_e"),
    "crane": ("radius", "eps_e"),
    "arc": ("alpha_deg",),
    "arc_swing": ("alpha_deg",),
    "track_follow": ("az_rel", "elev", "dist"),
    "track_orbit": ("az_e", "elev", "dist"),
    "track_lead": ("lead_angle", "elev", "dist"),
    "static": (), "pan_left": (), "pan_right": (), "tilt_up": (), "tilt_down": (),
    "zoom_in_out": (), "zoom_out_in": (),
}


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def _unjsonify(obj):
    return obj          # JSON round-trip already gives lists/floats; loaders use np.asarray
