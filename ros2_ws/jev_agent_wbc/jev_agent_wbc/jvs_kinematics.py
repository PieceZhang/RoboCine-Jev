"""Kinematics and trajectory helpers for the JVS mock camera arm.

This module deliberately has no ROS imports so the JSON contract and IK can be
tested without a running ROS graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
# Keep these identical to the revolute limits in urdf/jev_agent_wbc.urdf.xacro.
# Wider planning bounds make the Isaac drive hit its physical stop while the
# controller continues integrating an unreachable servo target.
JOINT_LOWER = np.array([
    -1.3962634016,
    -1.18682389136,
    -0.523598775598,
    -1.57079632679,
    -2.61799387799,
    -0.698131700798,
])
JOINT_UPPER = np.array([
    1.3962634016,
    0.698131700798,
    2.61799387799,
    1.57079632679,
    0.03490658504,
    0.698131700798,
])

# URDF joint origins, expressed in each joint's parent frame.  The fixed
# base_link -> neck_base_link offset is folded into the first entry.
JOINT_ORIGINS = np.array([
    [0.098, 0.0, 0.266],
    [0.0, 0.0, 0.100],
    [0.0, 0.0195, 0.20429],
    [-0.082, -0.020, 0.168],
    [0.0, 0.034, 0.088],
    [0.068, -0.035, 0.036],
])
JOINT_AXES = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])
CAMERA_OFFSET = np.array([0.038, 0.0, 0.1165])


@dataclass(frozen=True)
class TaskSpaceSample:
    """One relative camera pose sample in metres, radians, and seconds."""

    time_from_start: float
    pose: np.ndarray


@dataclass(frozen=True)
class RelativeTrajectory:
    """Validated relative task-space trajectory loaded from JSON."""

    samples: tuple[TaskSpaceSample, ...]
    sample_rate_hz: float
    frame_id: str
    source: Path

    @property
    def duration_s(self) -> float:
        return self.samples[-1].time_from_start


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a 3x3 active rotation matrix for a unit axis."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    v = 1.0-c
    return np.array([[c+x*x*v, x*y*v-z*s, x*z*v+y*s],
                     [y*x*v+z*s, c+y*y*v, y*z*v-x*s],
                     [z*x*v-y*s, z*y*v+x*s, c+z*z*v]])


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ROS fixed-axis roll/pitch/yaw rotation (Rz(yaw) Ry(pitch) Rx(roll))."""
    return (
        _rotation(np.array([0.0, 0.0, 1.0]), yaw)
        @ _rotation(np.array([0.0, 1.0, 0.0]), pitch)
        @ _rotation(np.array([1.0, 0.0, 0.0]), roll)
    )


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """SO(3) logarithm as an axis-angle vector."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    if angle < 1.0e-8:
        return 0.5 * vee
    if math.pi - angle < 1.0e-5:
        # This branch is not normally reached by sequential IK, but keeps the
        # function well behaved for user trajectories with a 180-degree step.
        diagonal = np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if largest == 0:
            axis[1] = math.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
            axis[2] = math.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
        elif largest == 1:
            axis[0] = math.copysign(axis[0], rotation[0, 1] + rotation[1, 0])
            axis[2] = math.copysign(axis[2], rotation[1, 2] + rotation[2, 1])
        else:
            axis[0] = math.copysign(axis[0], rotation[0, 2] + rotation[2, 0])
            axis[1] = math.copysign(axis[1], rotation[1, 2] + rotation[2, 1])
        norm = float(np.linalg.norm(axis))
        return angle * axis / max(norm, 1.0e-12)
    return (angle / (2.0 * math.sin(angle))) * vee


def forward_kinematics(
    joint_positions: Sequence[float],
    with_jacobian: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the camera_link pose in base_link, plus its spatial Jacobian."""
    q = np.asarray(joint_positions, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("joint_positions must contain six finite values")

    position = np.zeros(3)
    orientation = np.eye(3)
    origins_world: list[np.ndarray] = []
    axes_world: list[np.ndarray] = []
    for origin, axis, angle in zip(JOINT_ORIGINS, JOINT_AXES, q):
        position = position + orientation @ origin
        origins_world.append(position.copy())
        axes_world.append(orientation @ axis)
        orientation = orientation @ _rotation(axis, float(angle))
    position = position + orientation @ CAMERA_OFFSET

    if not with_jacobian:
        return position, orientation
    jacobian = np.zeros((6, 6))
    axes_array = np.asarray(axes_world)
    jacobian[:3] = np.cross(axes_array, position-np.asarray(origins_world)).T
    jacobian[3:] = axes_array.T
    return position, orientation, jacobian


def solve_ik(
    target_position: Sequence[float],
    target_orientation: np.ndarray,
    seed: Sequence[float],
    *,
    max_iterations: int = 200,
    position_tolerance: float = 2.0e-4,
    orientation_tolerance: float = 2.0e-3,
    damping: float = 1.0e-3,
) -> np.ndarray | None:
    """Solve one camera pose with bounded damped least-squares IK.

    ``None`` is returned when the requested pose cannot be reached within the
    supplied iteration budget and tolerances.
    """
    target_p = np.asarray(target_position, dtype=float)
    target_r = np.asarray(target_orientation, dtype=float)
    q = np.asarray(seed, dtype=float).copy()
    if target_p.shape != (3,) or not np.all(np.isfinite(target_p)):
        raise ValueError("target_position must contain three finite values")
    if target_r.shape != (3, 3) or not np.all(np.isfinite(target_r)):
        raise ValueError("target_orientation must be a finite 3x3 matrix")
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("seed must contain six finite values")
    if max_iterations <= 0 or position_tolerance <= 0.0 or orientation_tolerance <= 0.0:
        raise ValueError("IK iteration count and tolerances must be positive")
    if damping <= 0.0:
        raise ValueError("IK damping must be positive")

    q = np.clip(q, JOINT_LOWER, JOINT_UPPER)
    # Rotational rows are converted to metre-like scale so translation does
    # not get sacrificed near a wrist singularity.
    orientation_weight = 0.20
    previous_cost = math.inf
    for _ in range(max_iterations):
        position, orientation, jacobian = forward_kinematics(q, with_jacobian=True)
        position_error = target_p - position
        rotation_error = _rotation_vector(target_r @ orientation.T)
        if (
            np.linalg.norm(position_error) <= position_tolerance
            and np.linalg.norm(rotation_error) <= orientation_tolerance
        ):
            return q

        error = np.concatenate((position_error, orientation_weight * rotation_error))
        weighted_jacobian = jacobian.copy()
        weighted_jacobian[3:, :] *= orientation_weight
        lhs = weighted_jacobian @ weighted_jacobian.T + (damping ** 2) * np.eye(6)
        try:
            delta = weighted_jacobian.T @ np.linalg.solve(lhs, error)
        except np.linalg.LinAlgError:
            return None
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > 0.20:
            delta *= 0.20 / delta_norm

        cost = float(error @ error)
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = np.clip(q + scale * delta, JOINT_LOWER, JOINT_UPPER)
            candidate_p, candidate_r = forward_kinematics(candidate)
            candidate_error = np.concatenate((
                target_p - candidate_p,
                orientation_weight * _rotation_vector(target_r @ candidate_r.T),
            ))
            if float(candidate_error @ candidate_error) < cost:
                q = candidate
                accepted = True
                break
        if not accepted or abs(previous_cost - cost) < 1.0e-16:
            break
        previous_cost = cost
    return None


def load_relative_trajectory(path: str | Path) -> RelativeTrajectory:
    """Load and validate a ``jvs_relative_task_space_trajectory/v1`` file."""
    source = Path(path).expanduser()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read trajectory file {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc

    expected_format = "jvs_relative_task_space_trajectory/v1"
    if document.get("format") != expected_format:
        raise ValueError(f"format must be {expected_format!r}")
    if document.get("relative_to") != "current_camera_link":
        raise ValueError("relative_to must be 'current_camera_link'")
    frame_id = str(document.get("frame_id", ""))
    if frame_id not in {"base_link", "initial_camera_link"}:
        raise ValueError("frame_id must be 'base_link' or 'initial_camera_link'")
    if document.get("dof_order") != ["x", "y", "z", "roll", "pitch", "yaw"]:
        raise ValueError("dof_order must be [x, y, z, roll, pitch, yaw]")
    units = document.get("units", {})
    if units.get("translation") != "m" or units.get("rotation") != "rad":
        raise ValueError("trajectory units must be metres and radians")

    try:
        sample_rate_hz = float(document["sample_rate_hz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("sample_rate_hz must be present and numeric") from exc
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite")

    raw_samples = document.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) < 2:
        raise ValueError("samples must contain at least two trajectory points")
    samples: list[TaskSpaceSample] = []
    for index, item in enumerate(raw_samples):
        try:
            timestamp = float(item["time_from_start"])
            pose = np.asarray(item["pose"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"sample {index} must contain numeric time_from_start and pose"
            ) from exc
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(f"sample {index} has an invalid timestamp")
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise ValueError(f"sample {index} pose must contain six finite values")
        if samples and timestamp <= samples[-1].time_from_start:
            raise ValueError("sample timestamps must be strictly increasing")
        samples.append(TaskSpaceSample(timestamp, pose))

    if abs(samples[0].time_from_start) > 1.0e-9:
        raise ValueError("the first sample must start at time 0")
    if not np.allclose(samples[0].pose, 0.0, atol=1.0e-9):
        raise ValueError("the first relative pose must be all zeros")
    if "duration_s" in document:
        duration_s = float(document["duration_s"])
        if not math.isclose(duration_s, samples[-1].time_from_start, abs_tol=1.0e-6):
            raise ValueError("duration_s must equal the final sample timestamp")
    return RelativeTrajectory(tuple(samples), sample_rate_hz, frame_id, source)


def relative_pose_target(
    initial_position: np.ndarray,
    initial_orientation: np.ndarray,
    relative_pose: Sequence[float],
    frame_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose one relative xyz/RPY sample with the captured initial pose."""
    relative = np.asarray(relative_pose, dtype=float)
    if relative.shape != (6,):
        raise ValueError("relative_pose must contain six values")
    delta_position = relative[:3]
    delta_rotation = rpy_matrix(*relative[3:])
    if frame_id == "base_link":
        return initial_position + delta_position, delta_rotation @ initial_orientation
    if frame_id == "initial_camera_link":
        return (
            initial_position + initial_orientation @ delta_position,
            initial_orientation @ delta_rotation,
        )
    raise ValueError(f"unsupported trajectory frame_id: {frame_id!r}")
