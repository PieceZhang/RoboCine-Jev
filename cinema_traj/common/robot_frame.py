"""Robot camera frame — Z+ FORWARD, X+ RIGHT, Y+ DOWN (the robot's definition).

The pipeline's internal frame is +Y up with look-at rotations stored as
columns (right, up, forward) — see ``cinema_traj.common.se3``. The robot
consumes a DIFFERENT body-fixed camera convention:

    X+ = image right
    Y+ = image DOWN
    Z+ = optical FORWARD

This is exactly the OpenCV/gsplat camera convention already used by
``tools/gsplat_render.viewmat_from_pos_look``, so this module reuses that
same math, expressed as a body→world rotation (columns = the robot camera's
axes in world space):

    R_robot = [ x_right | y_down | z_forward ]      (3×3, det = +1)

Consumers:
  * ``rotation_robot``      — 3×3 body→world rotation for a look direction
  * ``quaternion_robot``    — (w, x, y, z) body→world quaternion
  * ``pose_robot``          — position + quaternion in one call (export step)

Down-stream frame notes (unchanged, for reference):
  * USD/Isaac cameras look down local −Z with +X image right — a robot-frame
    camera equals a USD camera rotated 180° about its X axis.
  * ROS/gz demo uses x-forward/y-left/z-up (FLU) — see
    ``ros2_ws/cinema_traj_bridge/traj_loader.py``.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

WORLD_UP = np.array([0.0, 1.0, 0.0])          # pipeline world is +Y up


def rotation_robot(pos, look, world_up=WORLD_UP) -> np.ndarray:
    """Body→world rotation (3×3) with columns (X+ right, Y+ down, Z+ forward).

    Same look-at math as ``tools/gsplat_render.viewmat_from_pos_look``;
    returns the camera-to-world rotation instead of the world-to-camera
    view matrix. Handles the degenerate forward ∥ up case by nudging up.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    look = np.asarray(look, dtype=np.float64).reshape(3)
    f = look - pos
    n = np.linalg.norm(f)
    if n < 1e-9:
        return np.eye(3)
    z_fwd = f / n                                    # Z+ forward
    up = np.asarray(world_up, dtype=np.float64)
    if np.linalg.norm(np.cross(z_fwd, up)) < 1e-6:   # looking straight up/down
        up = up + np.array([1e-3, 0.0, 0.0])
        up /= np.linalg.norm(up)
    x_right = np.cross(z_fwd, up)                    # f×up = image right
    x_right /= np.linalg.norm(x_right)
    y_up_world = np.cross(x_right, z_fwd)            # world-ish "up" on the image plane
    y_down = -y_up_world                             # Y+ DOWN
    return np.stack([x_right, y_down, z_fwd], axis=1)


def rotation_robot_torch(pos, look, world_up=None):
    """Differentiable batch version of ``rotation_robot``.

    pos/look: (F,3) torch tensors. Returns (F,3,3) body→world rotations with
    columns (X+ right, Y+ down, Z+ forward)."""
    if not _HAS_TORCH:
        raise RuntimeError("torch not available")
    upv = torch.tensor([0.0, 1.0, 0.0], dtype=pos.dtype, device=pos.device) \
        if world_up is None else world_up
    f = look - pos
    z_fwd = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    cross = torch.cross(z_fwd, upv.expand_as(z_fwd), dim=-1)
    degenerate = cross.norm(dim=-1, keepdim=True) < 1e-6
    up_adj = torch.where(degenerate,
                         upv + torch.tensor([1e-3, 0.0, 0.0],
                                            dtype=pos.dtype, device=pos.device),
                         upv).expand_as(z_fwd)
    up_adj = up_adj / up_adj.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x_right = torch.cross(z_fwd, up_adj, dim=-1)
    x_right = x_right / x_right.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    y_down = -torch.cross(x_right, z_fwd, dim=-1)
    return torch.stack([x_right, y_down, z_fwd], dim=-1)


def quaternion_robot(pos, look, world_up=WORLD_UP) -> tuple:
    """(w, x, y, z) body→world quaternion for the robot camera frame
    (X+ right, Y+ down, Z+ forward) looking from ``pos`` at ``look``."""
    return _quat_from_rotation(rotation_robot(pos, look, world_up))


def pose_robot(pos, look, world_up=WORLD_UP) -> dict:
    """One robot-frame pose: {position, quaternion (w,x,y,z), rotation (3×3)}."""
    R = rotation_robot(pos, look, world_up)
    return {"position": [float(v) for v in np.asarray(pos).reshape(3)],
            "quaternion_wxyz": _quat_from_rotation(R),
            "rotation_body_to_world": R.tolist(),
            "frame": "robot_camera: X+ right, Y+ down, Z+ forward"}


def _quat_from_rotation(R: np.ndarray) -> tuple:
    """(w, x, y, z) from a right-handed rotation matrix (columns = axes)."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return s / 4.0, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    if m00 >= m11 and m00 >= m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    if m11 >= m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
