"""Causal reference-aware SE(3) camera stabilization; no robot/control writes.

All poses are world Z-up, camera_link (+X forward, +Y left, +Z up).
Twists are [translation, rotation] in the reference camera frame.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def left_jacobian(phi):
    theta = np.linalg.norm(phi)
    w = skew(phi)
    if theta < 1e-5:
        return np.eye(3)+(.5-theta**2/24)*w+(1/6-theta**2/120)*(w@w)
    return np.eye(3)+(1-np.cos(theta))/theta**2*w+(theta-np.sin(theta))/theta**3*(w@w)


def se3_log(position, rotation):
    phi = Rotation.from_matrix(rotation).as_rotvec()
    return np.r_[np.linalg.solve(left_jacobian(phi), position), phi]


def se3_exp(twist):
    return left_jacobian(twist[3:])@twist[:3], Rotation.from_rotvec(twist[3:]).as_matrix()


@dataclass(frozen=True)
class StabilizerConfig:
    translation_cutoff_hz: float = .5
    rotation_cutoff_hz: float = .5
    stages: int = 2
    max_translation_correction_m: float = .12
    max_rotation_correction_deg: float = 8.
    reset_gap_s: float = .15


class VirtualStabilizer:
    def __init__(self, config=None):
        self.config = config or StabilizerConfig()
        c = self.config
        if (not np.isfinite([c.translation_cutoff_hz, c.rotation_cutoff_hz,
                c.max_translation_correction_m, c.max_rotation_correction_deg, c.reset_gap_s]).all()
                or min(c.translation_cutoff_hz, c.rotation_cutoff_hz,
                    c.max_translation_correction_m, c.max_rotation_correction_deg, c.reset_gap_s) <= 0
                or c.stages not in (1, 2, 3)):
            raise ValueError('invalid stabilization configuration')
        self.reset()

    def reset(self):
        self.time = None
        self.state = None
        self.resets = 0

    def update(self, t, reference_position, reference_rotation, actual_position, actual_rotation):
        rp, ap = np.asarray(reference_position, float), np.asarray(actual_position, float)
        rr, ar = np.asarray(reference_rotation, float), np.asarray(actual_rotation, float)
        if (rp.shape != (3,) or ap.shape != (3,) or rr.shape != (3, 3) or ar.shape != (3, 3)
                or not np.isfinite(np.r_[t, rp, ap, rr.ravel(), ar.ravel()]).all()):
            raise ValueError('finite timestamp, positions and rotation matrices required')
        error = se3_log(rr.T@(ap-rp), rr.T@ar)
        # Avoid principal-log jumps: reset safely for near-antipodal tracking errors.
        reset = (self.time is None or t < self.time or t-self.time > self.config.reset_gap_s
                 or np.linalg.norm(error[3:]) > np.deg2rad(170))
        if reset:
            self.state = np.tile(error, (self.config.stages, 1))
            self.resets += 1
        elif t > self.time:
            cutoff = np.r_[np.full(3, self.config.translation_cutoff_hz),
                           np.full(3, self.config.rotation_cutoff_hz)]
            alpha = -np.expm1(-2*np.pi*cutoff*(t-self.time))
            value = error
            for stage in self.state:
                stage += alpha*(value-stage)
                value = stage
        self.time = float(t)
        delta, rotation = se3_exp(self.state[-1])
        p, r = rp+rr@delta, rr@rotation
        # Bound only the virtual/physical difference; preserve the physical pose.
        correction = p-ap
        scale = min(1., self.config.max_translation_correction_m/max(np.linalg.norm(correction), 1e-15))
        p = ap+scale*correction
        correction_r = Rotation.from_matrix(ar.T@r).as_rotvec()
        scale_r = min(1., np.deg2rad(self.config.max_rotation_correction_deg)/max(np.linalg.norm(correction_r), 1e-15))
        r = ar@Rotation.from_rotvec(scale_r*correction_r).as_matrix()
        return p, r, dict(translation_correction_m=float(np.linalg.norm(p-ap)),
                         rotation_correction_deg=float(np.rad2deg(np.linalg.norm(correction_r))*scale_r),
                         translation_limited=bool(scale < 1.), rotation_limited=bool(scale_r < 1.),
                         reset=bool(reset), raw_twist=error.tolist(), filtered_twist=self.state[-1].tolist())
