"""Soft arm workspace and conditioning preferences for mobile camera tracking."""
import numpy as np
from scipy.spatial.transform import Rotation

from .jvs_kinematics import forward_kinematics, JOINT_LOWER, JOINT_UPPER


COMFORT_POSE = np.array([0., -.3526, .62, 0., -.2674, 0.])
COMFORT_REACH = .25


def arm_metrics(q):
    position, _, jacobian = forward_kinematics(q, with_jacobian=True)
    scaled = jacobian.copy()
    scaled[3:] *= .3  # characteristic arm length, metres per angular radian
    singular = np.linalg.svd(scaled, compute_uv=False)
    return dict(arm_horizontal_reach_m=float(np.linalg.norm(position[:2]-[.098, 0.])),
                arm_sigma_min=float(singular[-1]),
                arm_joint_reserve_rad=float(np.min(np.minimum(q-JOINT_LOWER, JOINT_UPPER-q))))


def preferred_posture(yaw, camera_rotation, camera_height, initial_q=None):
    """Comfortable arm branch at the required height/orientation, XY free.

    The base is responsible for the resulting camera XY offset. This prevents
    a fixed world-frame base offset from forcing the wrist onto its upper stop
    when the camera pitches or the chassis changes heading.
    """
    # Measured joints seed the execution preference on its current IK branch.
    # Restarting at the neutral pose on every small chassis turn can flip the
    # SO(3) half-turn branch and abruptly move the preferred camera XY offset.
    q = COMFORT_POSE.copy() if initial_q is None else np.asarray(initial_q, float).copy()
    rotation = Rotation.from_euler('z', yaw).as_matrix()
    weights = np.array([2500., 400., 400., 400.])
    regularizer = np.array([.8, 1., 1., .8, 1., .8])
    for _ in range(8):
        p, r, j = forward_kinematics(q, with_jacobian=True)
        p, r = rotation@p, rotation@r
        j = np.vstack([rotation@j[:3], rotation@j[3:]])
        error = np.r_[camera_height-p[2], Rotation.from_matrix(camera_rotation@r.T).as_rotvec()]
        rows = j[[2,3,4,5]]
        step = np.linalg.solve(rows.T@(weights[:,None]*rows)+np.diag(regularizer),
            rows.T@(weights*error)-regularizer*(q-COMFORT_POSE))
        step *= min(1., .25/max(np.max(np.abs(step)), 1e-8))
        q = np.clip(q+step, JOINT_LOWER+.15, JOINT_UPPER-.15)
    offset = rotation@forward_kinematics(q)[0]
    return q, offset


def posture_velocity(q, velocity_limits, preferred=COMFORT_POSE):
    """A bounded preference, projected into camera-task nullspace by the QP."""
    desired = 1.5*(preferred-q)
    metrics = arm_metrics(q)
    if metrics['arm_sigma_min'] < .02:
        # Smooth log-volume gradient stays finite at singular configurations.
        # It supplements the bent posture seed, whose direction is defined even
        # at a perfectly straight, symmetric reset configuration.
        def volume(joints):
            jacobian = forward_kinematics(joints, with_jacobian=True)[2]
            jacobian[3:] *= .3
            singular = np.linalg.svd(jacobian, compute_uv=False)
            return float(np.sum(np.log(singular**2+.01**2)))
        step = 1e-3
        gradient = np.array([(volume(q+np.eye(6)[j]*step)-volume(q-np.eye(6)[j]*step))/(2*step)
                             for j in range(6)])
        desired += .025*gradient
    scale = .6*np.asarray(velocity_limits)
    return scale*np.tanh(desired/scale), metrics
