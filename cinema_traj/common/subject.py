"""Dynamic subject track: time-parameterized position/heading + capsule obstacle.

This is our extension beyond CinemaTraj (the paper assumes a static scene and
explicitly lists moving subjects as future work — §5, §K.2). The subject is

  * an occlusion guard target (head / chest key point queried at time t),
  * a look_at target for tracking shots,
  * a moving capsule obstacle in the collision field.

Design note: ``t`` is DATA, not a differentiable variable. At each sampled
frame the capsule is a static differentiable obstacle; the free parameters
being optimized are the camera's spherical offsets, so this is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

from .se3 import angle_lerp

DEFAULT_KEY_POINTS = {"head": 0.92, "chest": 0.72, "feet": 0.0}   # fraction of height


@dataclass
class SubjectState:
    t: float
    pos: np.ndarray            # feet position on the floor
    heading: float             # yaw (rad), direction the subject faces


@dataclass
class SubjectTrack:
    subject_id: str = "subject"
    height_m: float = 1.75
    frames: list = field(default_factory=list)   # sorted [{t, pos[3], heading}]
    key_points: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: dict) -> "SubjectTrack":
        kp = {k: v if isinstance(v, (int, float)) else v[1]
              for k, v in (d.get("key_points_local") or {}).items()} or dict(DEFAULT_KEY_POINTS)
        frames = sorted(d.get("frames", []), key=lambda f: f["t"])
        if not frames:
            raise ValueError("subject_track has no frames")
        return cls(subject_id=d.get("subject_id", "subject"),
                   height_m=float(d.get("height_m", 1.75)), frames=frames, key_points=kp)

    def to_dict(self) -> dict:
        return {"subject_id": self.subject_id, "height_m": self.height_m,
                "frames": self.frames, "key_points_local": self.key_points}

    # ------------------------------------------------------------------ #
    # Interpolation
    # ------------------------------------------------------------------ #
    @property
    def t0(self) -> float:
        return self.frames[0]["t"]

    @property
    def t1(self) -> float:
        return self.frames[-1]["t"]

    def state_at(self, t: float) -> SubjectState:
        """Clamped linear interpolation; heading uses shortest-arc wrap."""
        fr = self.frames
        if t <= fr[0]["t"]:
            f = fr[0]
        elif t >= fr[-1]["t"]:
            f = fr[-1]
        else:
            lo, hi = 0, len(fr) - 1
            while hi - lo > 1:                       # bisect on sorted frames
                mid = (lo + hi) // 2
                if fr[mid]["t"] <= t:
                    lo = mid
                else:
                    hi = mid
            a, b = fr[lo], fr[hi]
            u = (t - a["t"]) / max(b["t"] - a["t"], 1e-9)
            pos = (1 - u) * np.asarray(a["pos"], float) + u * np.asarray(b["pos"], float)
            return SubjectState(t, pos, angle_lerp(a["heading"], b["heading"], u))
        return SubjectState(t, np.asarray(f["pos"], float), float(f["heading"]))

    def key_point(self, t: float, name: str = "head") -> np.ndarray:
        s = self.state_at(t)
        frac = self.key_points.get(name, DEFAULT_KEY_POINTS.get(name, 0.9))
        return s.pos + np.array([0.0, frac * self.height_m, 0.0])

    def heading_at(self, t: float) -> float:
        return self.state_at(t).heading

    # ------------------------------------------------------------------ #
    # Capsule obstacle
    # ------------------------------------------------------------------ #
    def capsule_axis(self, t: float, shrink: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Capsule axis (feet+shrink·h → feet+0.9h), world."""
        s = self.state_at(t)
        a = s.pos + np.array([0.0, shrink * self.height_m, 0.0])
        b = s.pos + np.array([0.0, 0.9 * self.height_m, 0.0])
        return a, b

    def capsule_sdf(self, p: np.ndarray, t: float, radius: float = 0.30) -> np.ndarray:
        """Exact capsule SDF (numpy) around the subject at time t."""
        a, b = self.capsule_axis(t)
        return _capsule_sdf_np(p, a, b, radius)

    def capsule_sdf_torch(self, p, t: float, radius: float = 0.30):
        """Differentiable capsule SDF at absolute time t."""
        if not _HAS_TORCH:
            raise RuntimeError("torch not available")
        a, b = self.capsule_axis(t)
        at = torch.as_tensor(a, dtype=p.dtype, device=p.device)
        bt = torch.as_tensor(b, dtype=p.dtype, device=p.device)
        return _capsule_sdf_torch(p, at, bt, radius)

    def capsule_axes(self, t_arr) -> tuple[np.ndarray, np.ndarray]:
        """Per-time capsule axis endpoints for an array of times: two (F,3)
        arrays (feet+0.1h → feet+0.9h). The axis depends only on t, so callers
        can vectorize per-frame distance evaluation in one batch."""
        ts = np.atleast_1d(np.asarray(t_arr, dtype=np.float64))
        a = np.empty((ts.shape[0], 3))
        b = np.empty((ts.shape[0], 3))
        for i, t in enumerate(ts):
            s = self.state_at(float(t))
            a[i] = s.pos + np.array([0.0, 0.1 * self.height_m, 0.0])
            b[i] = s.pos + np.array([0.0, 0.9 * self.height_m, 0.0])
        return a, b

    def capsule_sdf_rows(self, p: np.ndarray, t_arr, radius: float = 0.30) -> np.ndarray:
        """Numpy capsule SDF where each row of p (F,3) is evaluated against the
        capsule pose at the matching time in t_arr (F,)."""
        a, b = self.capsule_axes(t_arr)
        p2 = np.atleast_2d(np.asarray(p, dtype=np.float64))
        ab = b - a                                            # (F,3)
        ab2 = np.maximum((ab * ab).sum(axis=-1), 1e-12)       # (F,)
        u = np.clip(((p2 - a) * ab).sum(axis=-1) / ab2, 0.0, 1.0)
        proj = a + u[:, None] * ab
        return np.linalg.norm(p2 - proj, axis=-1) - radius

    def summary(self) -> dict:
        """Compact description for the LLM planner prompt."""
        d0, d1 = self.state_at(self.t0), self.state_at(self.t1)
        return {
            "subject_id": self.subject_id,
            "height_m": self.height_m,
            "duration_s": round(self.t1 - self.t0, 2),
            "start_pos": [round(float(x), 2) for x in d0.pos],
            "end_pos": [round(float(x), 2) for x in d1.pos],
            "start_heading_deg": round(float(np.degrees(d0.heading)), 1),
            "end_heading_deg": round(float(np.degrees(d1.heading)), 1),
        }


# --------------------------------------------------------------------------- #
# Capsule SDF primitives
# --------------------------------------------------------------------------- #
def _point_segment_dist_np(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    ab = b - a
    ab2 = float(ab @ ab)
    u = np.clip(((p - a) @ ab) / max(ab2, 1e-12), 0.0, 1.0)
    proj = a + u[:, None] * ab
    return np.linalg.norm(p - proj, axis=1)


def _capsule_sdf_np(p: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    p2 = np.atleast_2d(np.asarray(p, dtype=np.float64))
    out = _point_segment_dist_np(p2, a, b) - radius
    return out if np.ndim(p) == 2 else out[0]


def _capsule_sdf_torch(p, a, b, radius: float):
    ab = b - a
    ab2 = (ab * ab).sum().clamp_min(1e-12)
    u = torch.clamp(((p - a) * ab).sum(dim=-1) / ab2, 0.0, 1.0)
    proj = a + u.unsqueeze(-1) * ab
    return (p - proj).norm(dim=-1) - radius
