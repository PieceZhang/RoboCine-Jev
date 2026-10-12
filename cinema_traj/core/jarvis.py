"""Jarvis mobile-manipulator layer: kinematics, workspace, reachability,
trackability.

Robot definition vendored from ``ros2_ws/jvs_mock`` on branch
``yuelin/isaac-robot-controller`` (commit 2b9256a "create mock robot"):

  * ``urdf/jvs_mock.urdf.xacro`` — differential-drive base (wheel separation
    0.21 m, wheel radius 0.07 m), neck-mounted 6-DOF arm with ASYMMETRIC joint
    boxes (joint2 [-1.187, 0.698], joint5 [-2.618, 0.035] rad) and a head
    camera fixed to link6 — look = camera_link +x, roll 0;
  * ``config/jarvis_controllers.yaml`` — diff-drive hardware caps
    (linear 1.0 m/s / 1.0 m/s², angular 3.0 rad/s / 7.0 rad/s²).

The consumer-side whole-body QP controller (``ros2_ws/jvs_mock``
``whole_body_controller`` + ``whole_body_qp.Limits``) commands the same 8-dim
input ``u = [v_base, ω_base, q̇1..6]`` through one velocity-level QP with hard
box bounds: velocity_limits [0.5, 1.2, 1, 1, 1, 1.2, 1.2, 1.5] — the joint
entries are exactly the per-joint URDF velocity limits used here (so a
trajectory inside :class:`JarvisCapabilities` is provably inside the QP's
joint bounds), while the base entries (0.5 m/s, 1.2 rad/s) are tighter than
this module's 0.8/2.5 planner caps, in which case the QP saturates before
the caps bite; the QP additionally enforces the wheel-level row
``|v| + |ω|·b/2 ≤ wheel_speed`` (15 rad/s × 0.07 m = 1.05 m/s) that
``check_trackable`` mirrors on the planner side.

Reachability vs trackability (the two planner concerns this module serves):

  * REACHABILITY is a geometry question — "can the base park close enough and
    the arm stretch to put the camera at this pose at all" — answered by
    :func:`camera_reachable` and enforced in the trajectory GEOMETRY layer
    (build-stage clamps); retiming cannot fix an unreachable shape.
  * TRACKABILITY is an execution question — "at what speed can the camera
    follow this path direction before some joint/base saturates" — answered by
    the direction-dependent cap ``v_max(û) = min_i cap_i / |(J⁺[û;0])_i|`` and
    enforced by retiming (stretch the time axis, never clip the path).

Conventions (pinned by tests):

  * cinema_traj world frame is Y-up: yaw 0 → +Z, positive toward +X, pitch up.
  * The base rides the floor plane (world y = 0); heading θ maps the URDF +x
    (forward) axis to the world yaw direction (sin θ, 0, cos θ).
  * Camera orientation is completed upright from the look vector (roll 0) —
    identical to ``ros2_ws/cinema_traj_bridge``'s ``quaternion_from_yaw_pitch``.
  * Task twist τ = [v_cam(3); ω_cam(3)] in world axes; input u = [v, ω, q̇1..6]
    (8-dim); τ = J·u with J the 6×8 geometric Jacobian.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from .retime import PlatformCaps, wrap_deg

_URDF_SOURCE = ("ros2_ws/jvs_mock @ origin/yuelin/isaac-robot-controller "
                "(2b9256a create mock robot)")

# --------------------------------------------------------------------------- #
# Robot definition (URDF values — do not edit without syncing the xacro)
# --------------------------------------------------------------------------- #
NECK_ORIGIN = (0.098, 0.0, 0.266)          # base_link → neck_base_link (fixed)
CAMERA_ORIGIN = (0.038, 0.0, 0.1165)       # link6 → camera_link (fixed)

ARM_JOINTS = [
    # name, origin xyz (parent→child), axis, lower, upper, max_velocity (rad/s)
    ("joint1", (0.0, 0.0, 0.0), "z", -1.3962634016, 1.3962634016, 1.0),   # ±80° green
    ("joint2", (0.0, 0.0, 0.10), "y", -1.18682389136, 0.698131700798, 1.0),
    ("joint3", (0.0, 0.0195, 0.20429), "y", -0.523598775598, 2.61799387799, 1.0),
    ("joint4", (-0.082, -0.020, 0.168), "z", -math.pi / 2, math.pi / 2, 1.2),
    ("joint5", (0.0, 0.034, 0.088), "y", -2.61799387799, 0.03490658504, 1.2),
    ("joint6", (0.068, -0.035, 0.036), "z", -0.698131700798, 0.698131700798, 1.5),  # ±40° green
]
Q_LO = np.array([j[3] for j in ARM_JOINTS])
Q_HI = np.array([j[4] for j in ARM_JOINTS])
JOINT_MAX_VEL = np.array([j[5] for j in ARM_JOINTS])
JOINT_NAMES = [j[0] for j in ARM_JOINTS]

WHEEL_SEPARATION = 0.21                    # jarvis_controllers.yaml
WHEEL_RADIUS = 0.07
BASE_BODY_RADIUS = 0.13                    # base_link cylinder (URDF)

# jarvis_controllers.yaml diff_drive_controller limits (hardware)
HW_BASE_LINEAR_VEL = 1.0
HW_BASE_LINEAR_ACC = 1.0
HW_BASE_ANGULAR_VEL = 3.0
HW_BASE_ANGULAR_ACC = 7.0


# --------------------------------------------------------------------------- #
# Small SE(3) helpers (numpy, Y-up world)
# --------------------------------------------------------------------------- #
def _axis_rot(axis: str, q: float) -> np.ndarray:
    """3×3 rotation about a URDF local axis."""
    c, s = math.cos(q), math.sin(q)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _h(R: np.ndarray, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def base_yaw_rot(yaw: float) -> np.ndarray:
    """World rotation of the base for heading ``yaw``: URDF x̂ (forward) →
    (sin yaw, 0, cos yaw), URDF ẑ (up) → world +Y."""
    c, s = math.cos(yaw), math.sin(yaw)
    # columns: forward, left, up
    return np.array([[s, c, 0.0], [0.0, 0.0, 1.0], [c, -s, 0.0]], dtype=np.float64)


def base_transform(bx: float, bz: float, yaw: float) -> np.ndarray:
    """base_link pose on the floor (world y=0) as a 4×4."""
    return _h(base_yaw_rot(yaw), (bx, 0.0, bz))


def camera_from_look(yaw: float, pitch: float) -> np.ndarray:
    """Upright camera rotation (world ← camera): x = look vector, roll 0.
    Matches traj_loader.quaternion_from_yaw_pitch exactly."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    x = np.array([cp * sy, sp, cp * cy])
    z = np.array([0.0, 1.0, 0.0]) - sp * x        # ŷ orthogonalized against x
    zn = np.linalg.norm(z)
    z = z / zn if zn > 1e-9 else np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _rot_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis × angle) of R."""
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    if ang > math.pi - 1e-6:                       # near π: axis from the diagonal
        a = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        k = int(np.argmax(a))
        axis = np.zeros(3)
        axis[k] = a[k] if a[k] > 1e-9 else 1.0
        j, l = (k + 1) % 3, (k + 2) % 3
        axis[j] = R[j, l] / (2.0 * axis[k]) if axis[k] > 1e-9 else 0.0
        axis[l] = R[l, j] / (2.0 * axis[k]) if axis[k] > 1e-9 else 0.0
        n = np.linalg.norm(axis)
        return axis / n * ang if n > 1e-9 else np.array([0.0, 0.0, ang])
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * math.sin(ang)) * ang


# --------------------------------------------------------------------------- #
# Forward kinematics
# --------------------------------------------------------------------------- #
def fk_frames(bx: float, bz: float, yaw: float, q) -> list[np.ndarray]:
    """World 4×4 frame of every arm joint origin (after its rotation) plus the
    camera, in chain order — joint1..joint6 then camera_link."""
    T = base_transform(bx, bz, yaw) @ _h(np.eye(3), NECK_ORIGIN)
    frames = []
    for (name, origin, axis, lo, hi, vel), qi in zip(ARM_JOINTS, q):
        T = T @ _h(np.eye(3), origin) @ _h(_axis_rot(axis, float(qi)), (0, 0, 0))
        frames.append(T)
    frames.append(T @ _h(np.eye(3), CAMERA_ORIGIN))
    return frames


def fk_camera(bx: float, bz: float, yaw: float, q) -> np.ndarray:
    """World 4×4 pose of camera_link for the full robot state."""
    return fk_frames(bx, bz, yaw, q)[-1]


def look_of(T_cam: np.ndarray) -> np.ndarray:
    """Unit look vector (camera +x axis in world)."""
    return T_cam[:3, 0]


def yaw_pitch_of(T_cam: np.ndarray) -> tuple[float, float]:
    """(yaw, pitch) in radians of the camera look vector (Y-up convention)."""
    look = look_of(T_cam)
    return math.atan2(look[0], look[2]), math.asin(min(1.0, max(-1.0, look[1])))


# --------------------------------------------------------------------------- #
# Geometric Jacobian — τ = J·u, u = [v, ω, q̇1..6]
# --------------------------------------------------------------------------- #
def jacobian(bx: float, bz: float, yaw: float, q) -> np.ndarray:
    """6×8 geometric Jacobian at the full robot state.

    Columns: base forward drive v (world direction of URDF +x), base yaw rate
    ω (about world +Y through the base origin), then the six revolute joints.
    Rows: camera-origin linear velocity (world xyz), camera angular velocity
    (world xyz)."""
    frames = fk_frames(bx, bz, yaw, q)
    p_cam = frames[-1][:3, 3]
    R0 = base_yaw_rot(yaw)
    fwd = R0[:, 0]

    J = np.zeros((6, 8))
    J[:3, 0] = fwd                                   # base forward drive
    # base yaw: rotation about world +Y through the base origin
    omega_y = np.array([0.0, 1.0, 0.0])
    J[:3, 1] = np.cross(omega_y, p_cam - np.array([bx, 0.0, bz]))
    J[3:, 1] = omega_y
    for i, (name, origin, axis, lo, hi, vel) in enumerate(ARM_JOINTS):
        T_i = frames[i]
        a = T_i[:3, :3] @ _AXIS_UNIT[axis]
        J[:3, 2 + i] = np.cross(a, p_cam - T_i[:3, 3])
        J[3:, 2 + i] = a
    return J


_AXIS_UNIT = {"x": np.array([1.0, 0.0, 0.0]),
              "y": np.array([0.0, 1.0, 0.0]),
              "z": np.array([0.0, 0.0, 1.0])}


# --------------------------------------------------------------------------- #
# Workspace envelope — FK sampling (deterministic, cached)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorkspaceEnvelope:
    """Camera position/orientation envelope over the joint boxes (base at the
    origin, heading 0; base-frame z = height above the floor)."""
    z_min: float
    z_max: float
    reach_min: float                     # min horizontal dist base→camera
    reach_max: float
    pitch_min: float                     # rad, look elevation
    pitch_max: float
    n_samples: int

    def z_band(self, margin: float = 0.0) -> tuple[float, float]:
        return (self.z_min + margin, self.z_max - margin)

    def pitch_band(self, margin: float = 0.0) -> tuple[float, float]:
        return (self.pitch_min + margin, self.pitch_max - margin)


def _batched_fk_positions(qs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera positions (n,3) and look vectors (n,3) in the base frame for a
    batch of joint vectors (n,6)."""
    n = qs.shape[0]
    T = np.broadcast_to(_h(np.eye(3), NECK_ORIGIN), (n, 4, 4)).copy()
    for j, (name, origin, axis, lo, hi, vel) in enumerate(ARM_JOINTS):
        Rots = np.stack([_axis_rot(axis, q) for q in qs[:, j]])      # (n,3,3)
        Mj = np.zeros((n, 4, 4))
        Mj[:, 3, 3] = 1.0
        Mj[:, :3, :3] = Rots
        Mj[:, :3, 3] = np.asarray(origin)
        T = T @ Mj
    T = T @ np.broadcast_to(_h(np.eye(3), CAMERA_ORIGIN), (n, 4, 4))
    return T[:, :3, 3], T[:, :3, 0]


_ENVELOPE: WorkspaceEnvelope | None = None


def workspace_envelope(n: int = 60000) -> WorkspaceEnvelope:
    """Sample the joint boxes (deterministic seed) → camera envelope."""
    global _ENVELOPE
    if _ENVELOPE is not None:
        return _ENVELOPE
    rng = np.random.default_rng(20260910)
    qs = rng.uniform(Q_LO, Q_HI, size=(n, 6))
    pos, look = _batched_fk_positions(qs)
    z = pos[:, 2]
    reach = np.hypot(pos[:, 0], pos[:, 1])
    pitch = np.arcsin(np.clip(look[:, 2], -1.0, 1.0))
    _ENVELOPE = WorkspaceEnvelope(
        z_min=float(z.min()), z_max=float(z.max()),
        reach_min=float(reach.min()), reach_max=float(reach.max()),
        pitch_min=float(pitch.min()), pitch_max=float(pitch.max()),
        n_samples=n)
    return _ENVELOPE


# --------------------------------------------------------------------------- #
# Inverse kinematics — 6-DOF arm for a camera pose, base pose given
# --------------------------------------------------------------------------- #
_IK_TOL_POS = 3e-2          # m — the live tracker's feedback loop closes
                             # residuals well below this; the old 3 mm gate
                             # rejected far-pullback (low-pitch) configs the
                             # arm can actually hold within ~1.7 cm
_IK_TOL_ROT = 1.2e-3 * 10   # rad ≈ 0.69° — invisible in the look direction


def _arm_jacobian_base(T_base: np.ndarray, q) -> tuple[np.ndarray, np.ndarray]:
    """Joint-space Jacobian (6×6) and camera pose in the BASE frame."""
    T = _h(np.eye(3), NECK_ORIGIN)
    frames = []
    for (name, origin, axis, lo, hi, vel), qi in zip(ARM_JOINTS, q):
        T = T @ _h(np.eye(3), origin) @ _h(_axis_rot(axis, float(qi)), (0, 0, 0))
        frames.append(T)
    T_cam = T @ _h(np.eye(3), CAMERA_ORIGIN)
    p_cam = T_cam[:3, 3]
    J = np.zeros((6, 6))
    for i, (name, origin, axis, lo, hi, vel) in enumerate(ARM_JOINTS):
        a = frames[i][:3, :3] @ _AXIS_UNIT[axis]
        J[:3, i] = np.cross(a, p_cam - frames[i][:3, 3])
        J[3:, i] = a
    return J, T_cam


def arm_ik(T_base: np.ndarray, T_target: np.ndarray, q0=None,
           tol_pos: float = _IK_TOL_POS, tol_rot: float = _IK_TOL_ROT,
           iters_per_seed: int = 120, max_seeds: int = 6,
           damp: float = 2e-3) -> tuple[np.ndarray | None, float]:
    """Damped-least-squares IK for the 6 arm joints putting camera_link at
    ``T_target`` (world) with the base at ``T_base``.

    Two robustness mechanisms (the arm has wrist/elbow singularities and
    asymmetric joint boxes, where naive clamped DLS stalls):

      * active-set limits — a joint sitting at a box face whose step would
        push it further out is pinned and the step re-solved on the free
        joints (plain hard clamping stalls the Newton flow);
      * deterministic multistart with stall restarts — seeds: the warm start
        ``q0``, mid-range, then draws from a per-target seeded RNG (hash of
        the target pose → reproducible across runs); a seed that stops
        improving for 20 iterations is abandoned for the next.

    Returns (q, final_error) — q is None when unconverged."""
    T_rel = _inv(T_base) @ T_target
    p_t, R_t = T_rel[:3, 3], T_rel[:3, :3]
    digest = int.from_bytes(
        np.asarray(T_rel[:3, :], dtype=np.float64).tobytes()[:16], "little")
    rng = np.random.default_rng(digest)
    seeds = [] if q0 is None else [np.asarray(q0, dtype=np.float64)]
    seeds.append((Q_LO + Q_HI) / 2.0)
    while len(seeds) < max_seeds:
        seeds.append(rng.uniform(Q_LO, Q_HI))
    best_q, best_err = None, np.inf
    for seed in seeds:
        q = np.clip(seed, Q_LO, Q_HI)
        seed_best, stall = np.inf, 0
        for _ in range(iters_per_seed):
            J, T_cam = _arm_jacobian_base(T_base, q)
            e_p = p_t - T_cam[:3, 3]
            e_o = _rot_log(R_t @ T_cam[:3, :3].T)
            n_p, n_o = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_o))
            err = math.hypot(n_p, n_o)
            if err < seed_best - 1e-5:
                seed_best, stall = err, 0
            else:
                stall += 1
            if n_p <= tol_pos and n_o <= tol_rot:
                return np.clip(q, Q_LO, Q_HI), err
            if stall > 20:                          # singularity-stuck → next seed
                break
            e = np.concatenate([e_p, e_o])
            at_lo = q <= Q_LO + 1e-9
            at_hi = q >= Q_HI - 1e-9
            free = np.ones(6, dtype=bool)
            dq = np.zeros(6)
            for _active in range(6):                # active-set refinement
                pinned = ~free
                if pinned.all():
                    break
                Jf = J[:, free]
                dq_f = Jf.T @ np.linalg.solve(Jf @ Jf.T + (damp ** 2) * np.eye(6), e)
                dq = np.zeros(6)
                dq[free] = dq_f
                new_pin = (at_lo & (dq < 0)) | (at_hi & (dq > 0))
                if bool((new_pin == pinned).all()):
                    break
                free = ~new_pin
            step = float(np.linalg.norm(dq))
            if step > 0.35:                         # trust region
                dq *= 0.35 / step
            q = np.clip(q + dq, Q_LO, Q_HI)
        best_err = min(best_err, seed_best)
    return None, best_err


# --------------------------------------------------------------------------- #
# Reachability — can base + arm jointly produce this camera pose?
# --------------------------------------------------------------------------- #
def camera_reachable(pos, yaw: float, pitch: float, sdf,
                     tangent_yaw: float | None = None, seed_q=None,
                     caps: "JarvisCapabilities | None" = None,
                     prefer: tuple | None = None) -> dict | None:
    """Canonical-resolution reachability of one camera pose.

    Strategy: the base parks at arm-extension distance ``d`` from the camera's
    ground projection along −(heading), heading = path tangent when the camera
    is moving (the base must DRIVE its path — a differential base cannot
    translate sideways) else the look yaw; the look-vs-heading yaw offset is
    absorbed by the arm. Every candidate base point must be SDF-clear and
    inside the room; the arm IK must converge inside the joint boxes.

    ``prefer`` = (yaw_offset, d) tried first — the export stage passes the
    previous sample's chosen branch so the configuration stream stays
    continuous (no base teleport between neighbour samples).

    Returns ``{"base": (x, z, yaw), "q": (6,), "d": float, "yaw_offset": float}``
    or ``None``. ``sdf`` is a SceneSDF (collision + room bounds)."""
    pos = np.asarray(pos, dtype=np.float64)
    caps = caps or JARVIS_DEMO_CAPS
    env = workspace_envelope()
    z_lo, z_hi = env.z_band(caps.z_margin)
    if not (z_lo - 0.02 <= pos[1] <= z_hi + 0.02):
        return None

    # base heading candidates: the path tangent — and its reverse, because a
    # differential base drives backwards as happily as forwards (the shot may
    # pull back while looking at the target, putting the look ~180° off the
    # motion tangent; only the reverse heading fits the yaw wrist)
    if tangent_yaw is None:
        theta0s = [yaw]
    else:
        theta0s = [tangent_yaw, _wrap(tangent_yaw + math.pi)]
    if prefer is not None:
        # try the theta0 that hosts the previous branch heading FIRST — the
        # tangent branch can otherwise succeed with a ±90° look offset before
        # the continuity-preserving branch is ever reached (base teleport)
        prefer_yaw = float(prefer[0])
        theta0s.sort(key=lambda th: abs(_wrap(prefer_yaw - th)))
    prefer_off = None
    if prefer is not None:
        # prefer = (previous ABSOLUTE base yaw, d): keep the configuration
        # stream continuous — same parked/driving heading across samples
        prefer_off = float(prefer[0])
    seen_theta0 = set()
    for theta0 in theta0s:
        key = round(theta0, 6)
        if key in seen_theta0:
            continue
        seen_theta0.add(key)
        r = _camera_reachable_theta0(pos, yaw, pitch, sdf, caps, theta0,
                                     yaw_off_rel=yaw - theta0,
                                     seed_q=seed_q, prefer_off=prefer_off,
                                     prefer_d=None if prefer is None else float(prefer[1]),
                                     driving=tangent_yaw is not None)
        if r is not None:
            return r
    return None


def _camera_reachable_theta0(pos, yaw, pitch, sdf, caps, theta0, yaw_off_rel,
                             seed_q, prefer_off, prefer_d,
                             driving: bool = False) -> dict | None:
    yaw_off = _wrap(yaw_off_rel)
    if driving:
        # DIFFERENTIAL DRIVE: while the base moves, its heading must equal
        # the path tangent (or the exact reverse — driving backwards). The
        # old free offset grid let the prefer-branch lock a constant
        # heading while the position slid sideways — kinematically
        # impossible for a diff base; the whole look offset rides the arm.
        offsets = [0.0, math.pi]
    else:
        offsets = sorted({0.0, round(yaw_off, 9), min(yaw_off, math.pi / 2),
                          max(yaw_off, -math.pi / 2), math.pi / 2, -math.pi / 2},
                         key=lambda o: abs(o - yaw_off))
    d_grid = list(caps.arm_extension)
    if prefer_d is not None:
        d_grid = [prefer_d] + [d for d in d_grid if abs(d - prefer_d) > 1e-9]
    for off in offsets:
        if abs(off) > 2.4:                          # ≈ ±137° — beyond the yaw wrist
            continue
        theta = theta0 + off
        sy, cy = math.sin(theta), math.cos(theta)
        # base candidates: the tangent-rear line FIRST (legacy behaviour,
        # continuity-friendly), then the arm-reach disc — the intersection
        # of a sphere centred on the camera (radius = max arm reach) with
        # the ground plane. The base may park anywhere on that disc while
        # holding heading = tangent (it drives its own path); the pose is
        # unreachable only when the base fits NOWHERE on the disc.
        disc = [(d, 0.0) for d in d_grid]
        r_reach = max(caps.arm_extension)
        for ang in (45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0):
            disc.extend(((0.75 * r_reach, ang), (r_reach, ang)))
        for d, ang in disc:
            a = math.radians(ang)
            ca, sa = math.cos(a), math.sin(a)
            bx = pos[0] - (sy * ca - cy * sa) * d
            bz = pos[2] - (sy * sa + cy * ca) * d
            p_base = np.array([bx, 0.0, bz])
            if float(sdf.numpy_query(p_base)) < caps.base_clearance_margin:
                continue
            bounds = sdf.room_bounds_of_point(p_base)
            if bounds is not None:
                lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
                if not (bx >= lo[0] + caps.base_clearance_margin and
                        bx <= hi[0] - caps.base_clearance_margin and
                        bz >= lo[2] + caps.base_clearance_margin and
                        bz <= hi[2] - caps.base_clearance_margin):
                    continue
            T_base = base_transform(bx, bz, theta)
            T_target = _h(camera_from_look(yaw, pitch), pos)
            q, _err = arm_ik(T_base, T_target, q0=seed_q,
                             tol_pos=caps.ik_tol_pos, tol_rot=caps.ik_tol_rot)
            if q is not None:
                return {"base": (bx, bz, theta), "q": q, "d": d,
                        "yaw_offset": off}
    return None


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------- #
# JarvisCapabilities — the retime loop's platform contract, Jarvis side
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JarvisCapabilities(PlatformCaps):
    """Execution caps of the Jarvis base+arm, aligned with the whole-body QP
    controller's hard bounds (``whole_body_qp.Limits``).

    Defaults: base 0.8 m/s / 2.5 rad/s (planner-side caps, legacy
    tracking-yaml values — the WBC QP's 0.5/1.2 is tighter and saturates on
    its side, never clipping the planned shape), per-joint URDF velocity
    limits (identical to the WBC QP's joint bounds → provably safe), plus the
    wheel-level row |v| + |ω|·b/2 ≤ wheel_speed_limit checked by
    ``check_trackable`` and enforced by the QP. Yaw rate cap is the
    arm yaw joint (1.0 rad/s): the base may yaw faster, but a stationary pan
    can only be served by the arm, and one cap for both keeps the retimed
    stream provable in every configuration."""

    max_base_linear_vel: float = 0.8                  # m/s (planner cap; WBC QP 0.5)
    max_base_angular_vel: float = 2.5                 # rad/s (planner cap; WBC QP 1.2)
    max_base_linear_acc: float = 1.0                  # m/s² (controllers yaml)
    max_base_angular_acc: float = 7.0                 # rad/s² (controllers yaml)
    max_base_linear_acc: float = 1.0                  # m/s² (controllers yaml)
    max_base_angular_acc: float = 7.0                 # rad/s² (controllers yaml)
    max_joint_vel: tuple = tuple(JOINT_MAX_VEL)       # URDF per-joint (rad/s)
    min_gimbal_pitch: float = field(default=None)     # → envelope pitch band
    max_gimbal_pitch: float = field(default=None)
    max_gimbal_pitch_rate: float = 1.0                # rad/s, arm pitch chain
    max_yaw_rate: float = 1.0                         # rad/s, min(ω_base, q̇1)
    wheel_speed_limit: float = 15.0 * 0.07            # m/s, URDF wheel ω × r
    half_track: float = 0.105                         # m, wheel y-offset (URDF)
    command_period: float = 0.05
    arm_extension: tuple = (0.30, 0.40, 0.50, 0.25, 0.55, 0.20, 0.60, 0.45, 0.35)
    base_clearance_margin: float = 0.20               # m; 0.18 front extent + margin
    z_margin: float = 0.06                            # m, envelope safety margin
    cinematic_z_lo: float = 0.55                      # m, shot height band (measured
    cinematic_z_hi: float = 0.93                      #  jointly-feasible height/pitch)
    ik_tol_pos: float = _IK_TOL_POS
    ik_tol_rot: float = _IK_TOL_ROT

    def __post_init__(self):
        pos = [self.max_base_linear_vel, self.max_base_angular_vel,
               self.max_base_linear_acc, self.max_base_angular_acc,
               self.max_gimbal_pitch_rate, self.max_yaw_rate,
               self.command_period, *self.max_joint_vel, *self.arm_extension]
        if any(v <= 0.0 for v in pos):
            raise ValueError("all Jarvis caps must be > 0")
        # resolve envelope-derived pitch limits (frozen dataclass → object.__setattr__)
        lo, hi = self.pitch_band()
        if lo >= hi:
            raise ValueError("gimbal pitch band empty")
        object.__setattr__(self, "min_gimbal_pitch",
                           lo if self.min_gimbal_pitch is None
                           else float(self.min_gimbal_pitch))
        object.__setattr__(self, "max_gimbal_pitch",
                           hi if self.max_gimbal_pitch is None
                           else float(self.max_gimbal_pitch))

    # -- envelope views --------------------------------------------------- #
    def z_band(self) -> tuple[float, float]:
        return workspace_envelope().z_band(self.z_margin)

    def cinematic_z_band(self) -> tuple[float, float]:
        """Shot height band the geometry layer clamps into. The physical arm
        envelope is wider ([≈0.13, ≈0.96] m), but level looks need z ≥ ~0.55
        and steep down-looks z ≤ ~0.85 (FK/IK-measured coupling) — this band
        keeps every shot jointly feasible."""
        return (self.cinematic_z_lo, self.cinematic_z_hi)

    def pitch_band(self, margin: float = 0.03) -> tuple[float, float]:
        env = workspace_envelope()
        return (env.pitch_min + margin, env.pitch_max - margin)

    # -- PlatformCaps contract -------------------------------------------- #
    def tangential_accel(self) -> float:
        return self.max_base_linear_acc

    def vertex_speed_limits(self, dirn: np.ndarray, dt_out: float) -> np.ndarray:
        """Per-vertex speed cap: corners slow by the tangential-accel × period
        rule (full 3-D direction change — the arm sees pitch corners too), the
        straightaway cap is the Jacobian-projected ``v_max(û)``."""
        N = dirn.shape[0] + 1
        v_lim = np.full(N, np.inf)
        for i in range(1, N - 1):
            d = float(np.linalg.norm(dirn[i - 1] - dirn[i]))
            if d > 1e-9:
                v_lim[i] = min(v_lim[i], self.tangential_accel() * dt_out / d)
        for i in range(N - 1):
            v_lim[i] = min(v_lim[i], v_max(dirn[i], self))
        v_lim[0] = v_lim[-1] = 0.0                  # rest at both ends
        return v_lim

    def interval_factor(self, dp: np.ndarray, dts: np.ndarray,
                        yaw_l: np.ndarray, pitch_l: np.ndarray,
                        dt_out: float) -> float:
        """Worst stretch factor over intervals: the canonical-resolution input
        ``u = J⁺·[Δp/dt; ω]`` must satisfy every cap."""
        if len(dp) == 0:
            return 1.0
        vel = dp / dts[:, None]
        yaw_rate = np.diff(np.concatenate([yaw_l[:1], yaw_l])) / \
            np.concatenate([dts[:1], dts])
        pitch_rate = np.diff(np.concatenate([pitch_l[:1], pitch_l])) / \
            np.concatenate([dts[:1], dts])
        factor = 1.0
        base_cap = np.concatenate(
            [[self.max_base_linear_vel, self.max_base_angular_vel],
             np.asarray(self.max_joint_vel)])
        for k in range(len(dp)):
            look = camera_from_look(float(yaw_l[k]), float(pitch_l[k]))[:, 0]
            r_hat = np.cross(look, np.array([0.0, 1.0, 0.0]))
            n = np.linalg.norm(r_hat)
            r_hat = r_hat / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
            omega = yaw_rate[k] * np.array([0.0, 1.0, 0.0]) + pitch_rate[k] * r_hat
            # PHASE-SPLIT yaw preview: while DRIVING, the look-yaw is
            # served primarily by the base — its own omega cap, further
            # bound by the wheel-level constraint |v|+|omega|*b/2 <= wheel
            # speed. The arm yaw joint's 1.0 rad/s is the STATIONARY pan
            # cap and must not slow driving corners 2.5x. PREVIEW only:
            # the authoritative gate stays check_trackable on the IK'd
            # configuration stream — if the real allocation leans on
            # joint1 past its cap, the export stretch round slows the
            # stream there.
            v_h = float(np.hypot(vel[k][0], vel[k][2]))
            cap_k = base_cap.copy()
            cap_k[1] = min(self.max_base_angular_vel,
                           max(0.0, self.wheel_speed_limit - v_h) / self.half_track)
            cap_k[2] = (self.max_joint_vel[0] if v_h <= 0.05   # stationary: arm pans
                        else cap_k[1])                          # driving: base yaws
            factor = max(factor, twist_factor(vel[k], omega, self, cap_k))
        return factor

    def build_commands(self, ts, dp, dts, yaw_l, pitch_l) -> list:
        """Jarvis commands (base v/ω + q̇) are built from the IK'd
        configuration stream in the jarvis export stage — the generic retime
        loop's drone-schema ZOH commands are not used."""
        return []


JARVIS_DEMO_CAPS = JarvisCapabilities()


def _aligned_jacobian(dirn3: np.ndarray) -> np.ndarray:
    """Reference-configuration Jacobian re-aligned so the base heading follows
    the horizontal projection of ``dirn3`` (canonical moving resolution)."""
    ref = _reference_state()
    bx, bz, yaw, q = ref
    J = jacobian(bx, bz, yaw, q)
    h = math.hypot(dirn3[0], dirn3[2])
    if h < 1e-9:
        return J
    alpha = math.atan2(dirn3[0], dirn3[2]) - yaw    # world-yaw rotation to apply
    c, s = math.cos(alpha), math.sin(alpha)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    J = J.copy()
    J[:3, :] = R @ J[:3, :]
    return J


_REF_STATE: tuple | None = None


def _reference_state() -> tuple[float, float, float, np.ndarray]:
    """Deterministic mid-workspace configuration (cached): camera 0.45 m ahead
    of the base at 0.70 m height looking forward — the J used for loop-time
    v_max(û) / twist factors. The exact post-IK check (check_trackable) does
    not rely on it."""
    global _REF_STATE
    if _REF_STATE is not None:
        return _REF_STATE
    target = _h(camera_from_look(0.0, 0.0), (0.45, 0.70, 0.0))
    T_base = base_transform(0.0, 0.0, 0.0)
    q, _ = arm_ik(T_base, target)
    if q is None:                                   # pragma: no cover
        q = (Q_LO + Q_HI) / 2.0
    _REF_STATE = (0.0, 0.0, 0.0, q)
    return _REF_STATE


def v_max(dirn3, caps: JarvisCapabilities) -> float:
    """Direction-dependent trackable speed: the camera may move along unit
    direction ``dirn3`` at most ``min_i cap_i / |(J⁺[û;0])_i|`` before some
    input (base drive / base yaw / a joint) saturates."""
    J = _aligned_jacobian(np.asarray(dirn3, dtype=np.float64))
    u = np.linalg.pinv(J) @ np.concatenate([np.asarray(dirn3, dtype=np.float64),
                                            np.zeros(3)])
    return _input_scale_cap(u, caps)


def twist_factor(vel3, omega3, caps: JarvisCapabilities,
                 caps_u: np.ndarray | None = None) -> float:
    """Stretch factor for one camera twist: how much the interval must slow so
    ``u = J⁺·τ`` respects every cap (≥1 when already inside).  ``caps_u``
    overrides the per-input cap vector — used by the phase-split yaw
    preview in JarvisCapabilities.interval_factor."""
    J = _aligned_jacobian(np.asarray(vel3, dtype=np.float64))
    u = np.linalg.pinv(J) @ np.concatenate([np.asarray(vel3, dtype=np.float64),
                                            np.asarray(omega3, dtype=np.float64)])
    return max(1.0, _required_slowdown(u, caps, caps_u))


def _input_scale_cap(u: np.ndarray, caps: JarvisCapabilities,
                     caps_u_override: np.ndarray | None = None) -> float:
    cap_vec = caps_u_override if caps_u_override is not None else \
        np.concatenate([[caps.max_base_linear_vel, caps.max_base_angular_vel],
                        np.asarray(caps.max_joint_vel)])
    denom = np.abs(u)
    mask = denom > 1e-9
    if not mask.any():
        return np.inf
    return float((cap_vec[mask] / denom[mask]).min())


def _required_slowdown(u: np.ndarray, caps: JarvisCapabilities,
                       caps_u: np.ndarray | None = None) -> float:
    cap = _input_scale_cap(u, caps, caps_u)
    return 1.0 / cap if np.isfinite(cap) and cap > 0 else 1.0


# --------------------------------------------------------------------------- #
# Trackability verification — exact, on the IK'd configuration stream
# --------------------------------------------------------------------------- #
def trackability_factor(configs: list, caps: JarvisCapabilities) -> float:
    """Worst cap ratio over the configuration stream (≥1 ⇒ the stream must
    slow by this factor; the export stage's stretch round loop uses it)."""
    worst = 1.0
    for c in config_commands(configs)[:-1]:         # drop the terminal zero row
        worst = max(worst, float(c["v_base"]) / caps.max_base_linear_vel,
                    abs(float(c["omega_base"])) / caps.max_base_angular_vel)
        for rate, cap in zip(np.abs(c["qdot"]), caps.max_joint_vel):
            worst = max(worst, float(rate) / cap)
    return worst


def check_trackable(samples: list, configs: list, caps: JarvisCapabilities,
                    tol: float = 1e-6) -> list:
    """Exact post-IK verification: finite-difference the configuration stream
    (base x/z/yaw + 6 joints) and check every cap — the per-input caps plus
    the differential wheel-level row ``|v| + |ω|·b/2 ≤ wheel_speed_limit``
    (the same constraint the WBC QP enforces on its wheel pair). Returns
    violation strings (empty ⇒ provably trackable). ``configs`` rows:
    ``{"t", "base": (x, z, yaw), "q": (6,)}`` aligned with ``samples``."""
    warns = []
    for a, b, sa, sb in zip(configs, configs[1:], samples, samples[1:]):
        dt = max(float(sb["t"]) - float(sa["t"]), 1e-9)
        pa, pb = np.asarray(a["base"]), np.asarray(b["base"])
        v = math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / dt
        if v > caps.max_base_linear_vel + tol:
            warns.append(f"t={sb['t']:.2f}: base speed {v:.3f} > "
                         f"{caps.max_base_linear_vel}")
        w = abs(_wrap(float(pb[2]) - float(pa[2]))) / dt
        if w > caps.max_base_angular_vel + tol:
            warns.append(f"t={sb['t']:.2f}: base yaw rate {w:.2f} rad/s > "
                         f"{caps.max_base_angular_vel}")
        # The phase-split retime preview bounds driving ω by
        # (wheel_speed_limit − v)/half_track; this closes the proof on the
        # realized configuration stream (wheel circumference factored out).
        wheel = v + w * caps.half_track
        if wheel > caps.wheel_speed_limit + tol:
            warns.append(f"t={sb['t']:.2f}: wheel-level speed {wheel:.3f} m/s "
                         f"> {caps.wheel_speed_limit:.2f} (|v| + |ω|·b/2)")
        qa, qb = np.asarray(a["q"]), np.asarray(b["q"])
        qd = np.abs(qb - qa) / dt
        for i, (rate, cap) in enumerate(zip(qd, caps.max_joint_vel)):
            if rate > cap + tol:
                warns.append(f"t={sb['t']:.2f}: {JOINT_NAMES[i]} rate "
                             f"{rate:.3f} > {cap:.2f} rad/s")
    for k, c in enumerate(configs):
        q = np.asarray(c["q"])
        for i in range(6):
            if q[i] < Q_LO[i] - tol or q[i] > Q_HI[i] + tol:
                warns.append(f"t={c['t']:.2f}: {JOINT_NAMES[i]}={q[i]:.3f} "
                             f"outside [{Q_LO[i]:.3f}, {Q_HI[i]:.3f}]")
                break
    return warns


def config_commands(configs: list) -> list:
    """Per-interval feedforward commands from the configuration stream:
    ``{t, v_base, omega_base, qdot: (6,)}`` (ZOH over the interval)."""
    cmds = []
    for a, b in zip(configs, configs[1:]):
        dt = max(float(b["t"]) - float(a["t"]), 1e-9)
        pa, pb = np.asarray(a["base"]), np.asarray(b["base"])
        cmds.append({
            "t": float(a["t"]),
            "v_base": math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / dt,
            "omega_base": _wrap(float(pb[2]) - float(pa[2])) / dt,
            "qdot": (np.asarray(b["q"]) - np.asarray(a["q"])) / dt})
    if configs:
        last = configs[-1]
        cmds.append({"t": float(last["t"]), "v_base": 0.0, "omega_base": 0.0,
                     "qdot": np.zeros(6)})
    return cmds


def verify_reachable_trajectory(segments, subject, sdf, caps: JarvisCapabilities,
                                fps: float | None = None) -> dict:
    """Dense post-optimization reachability audit — the jarvis hard gate
    (mirror of the collision gate ``verify_collision_free``).

    Samples every segment at ``fps`` and runs the base-parking + arm-IK
    reachability check per waypoint (base heading from the path tangent,
    warm-started arm IK), reporting the reachable fraction plus one violation
    entry per unreachable waypoint. Retiming can slow a trajectory but never
    fix an unreachable SHAPE, so this gate runs on traj_final before any
    export.

    Dispatches to the batched tensor oracle (:mod:`.jarvis_batched`) —
    verdict-identical, orders of magnitude faster on long trajectories; set
    ``ROBOCINE_IK_SEQUENTIAL=1`` to force this per-waypoint reference path."""
    if os.environ.get("ROBOCINE_IK_SEQUENTIAL") != "1":
        from .jarvis_batched import verify_reachable_batched
        return verify_reachable_batched(segments, subject, sdf, caps, fps=fps)
    return _verify_reachable_trajectory_seq(segments, subject, sdf, caps, fps)


def _verify_reachable_trajectory_seq(segments, subject, sdf,
                                     caps: JarvisCapabilities,
                                     fps: float | None = None) -> dict:
    """Sequential reference audit — one ``camera_reachable`` per waypoint,
    prev-q warm-start threading (the batched oracle's parity reference and
    fallback decider)."""
    from ..common import constants as C
    from ..common.se3 import look_at_R, rotation_to_yaw_pitch
    fps = fps or C.DEFAULT_FPS
    violations: list = []
    per_segment: list = []
    total = okc = 0
    prev_q = None
    for seg in segments:
        th, t_abs = seg.sample_times(fps=fps)
        pos = seg.positions(th, None, subject).numpy()
        look = seg.look_target(th, t_abs, subject).numpy()
        yawp = [rotation_to_yaw_pitch(look_at_R(p, l)) for p, l in zip(pos, look)]
        yaw = np.array([math.radians(a) for a, _ in yawp])
        pitch = np.array([math.radians(b) for _, b in yawp])
        # base heading candidates: path tangent (the base must DRIVE its path);
        # stationary waypoints fall back to the look yaw
        tang = yaw.copy()
        for i in range(len(pos)):
            a = pos[max(i - 1, 0), [0, 2]]
            b = pos[min(i + 1, len(pos) - 1), [0, 2]]
            v = b - a
            if math.hypot(v[0], v[1]) > 1e-4:
                tang[i] = math.atan2(v[0], v[1])
        seg_ok = 0
        for i in range(len(pos)):
            total += 1
            r = camera_reachable(pos[i], float(yaw[i]), float(pitch[i]), sdf,
                                 tangent_yaw=float(tang[i]), seed_q=prev_q, caps=caps)
            if r is None:
                violations.append({
                    "segment": seg.name, "type": seg.type,
                    "t": round(float(t_abs[i]), 3),
                    "pos": [round(float(v), 3) for v in pos[i]]})
            else:
                okc += 1
                seg_ok += 1
                prev_q = r["q"]
        per_segment.append({"segment": seg.name, "type": seg.type,
                            "reachable": seg_ok, "total": int(len(pos))})
    reasons = [f"{v['segment']} ({v['type']}) @ t={v['t']}s pos={v['pos']} "
               f"不可达（底盘落点+臂 IK 无解）" for v in violations[:10]]
    return {"status": "ok" if not violations else "failed",
            "reachable_frac": round(okc / max(total, 1), 4),
            "per_segment": per_segment,
            "violations": violations, "reasons": reasons}


def _build_base_sdf_impl(scene):
    """Floor/ceiling-free collision field for BASE-placeability checks.

    The base RIDES the floor — the scene's floor OBB (a collision structure
    for the flying camera) would zero the SDF at every y=0 candidate point
    and reject every parking spot. Walls and furniture stay; the real-cloud
    field is dropped too (floor points are the driving surface, not
    obstacles)."""
    from ..common.sdf import SceneSDF
    obbs = [s.obb for s in scene.structures if s.kind not in ("floor", "ceiling")]
    obbs += [o.obb for o in scene.objects]
    return SceneSDF(obbs, scene.room_bounds())


def jarvis_caps_from_config(d: dict | None) -> JarvisCapabilities:
    """Build JarvisCapabilities from the ``cinema_traj.jarvis`` config block
    (all keys optional — defaults are the URDF/tracking-yaml aligned values)."""
    d = d or {}

    def _f(key, default):
        v = d.get(key)
        return default if v is None else float(v)

    joint_vel = d.get("max_joint_vel")
    if joint_vel is not None:
        joint_vel = tuple(float(v) for v in joint_vel)
        if len(joint_vel) != 6:
            raise ValueError("jarvis.max_joint_vel needs 6 values (joint1..6)")
    return JarvisCapabilities(
        max_base_linear_vel=_f("max_base_linear_vel", 0.8),
        max_base_angular_vel=_f("max_base_angular_vel", 2.5),
        max_base_linear_acc=_f("max_base_linear_acc", 1.0),
        max_base_angular_acc=_f("max_base_angular_acc", 7.0),
        max_joint_vel=joint_vel or tuple(JOINT_MAX_VEL),
        max_gimbal_pitch_rate=_f("max_gimbal_pitch_rate", 1.0),
        max_yaw_rate=_f("max_yaw_rate", 1.0),
        command_period=_f("command_period", 0.05),
        arm_extension=tuple(float(v) for v in d.get("arm_extension",
                                                    (0.30, 0.40, 0.50, 0.25, 0.55, 0.20, 0.60, 0.45, 0.35))),
        base_clearance_margin=_f("base_clearance_margin", 0.20),
        z_margin=_f("z_margin", 0.06),
        cinematic_z_lo=_f("cinematic_z_lo", 0.55),
        cinematic_z_hi=_f("cinematic_z_hi", 0.93))


_BASE_SDF_CACHE: dict = {}


def build_base_sdf(scene):
    """Memoizing wrapper: the floor/ceiling-pruned base-placeability field
    is deterministic per scene; build.run and optimize.run both build it."""
    key = getattr(scene, "scene_id", id(scene))
    if key not in _BASE_SDF_CACHE:
        _BASE_SDF_CACHE[key] = _build_base_sdf_impl(scene)
    return _BASE_SDF_CACHE[key]
