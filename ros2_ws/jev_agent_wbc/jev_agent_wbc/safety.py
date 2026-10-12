"""Direct swept-clearance fallback when sufficient CBF halfspaces conflict.

CBFs are a sufficient local construction, not the physical safety definition.
The fallback preserves every actuator bound and MUST pass a continuous dynamic
distance certificate. If neither construction succeeds, execution faults.
"""
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation

from .shot_predictive_qp import integration_matrix
from .whole_body_qp import whole_body_kinematics


def certified_escape(controller, state, target, rotation, dt, *, base_z=0., base_rotation=None, commanded_q=None):
    collision = controller.collision
    previous = controller.previous.copy()
    A, lower, upper = controller.bounds(np.asarray(state)[3:], dt, commanded_q)
    if np.any(lower > upper):
        raise RuntimeError('no actuator-feasible fallback')
    centers, radii = collision.active_obstacles(state, base_z, base_rotation, per_body=None)
    distances, gradients = collision.geometry.primitive_distances(state, centers, base_z, base_rotation)
    distances -= radii[:, None]
    rows = gradients@integration_matrix(state[2])
    motion = collision.primitive_motion(state, centers, base_z, base_rotation)
    requested = -1.0*(distances-.08)-motion
    approaching = (distances < .45) & ((rows@previous+motion < .03) | (distances < .10))
    reserve, rate = rows[approaching], requested[approaching]
    p, r, j = whole_body_kinematics(state, base_z, base_rotation)
    error = np.r_[np.asarray(target)-p, Rotation.from_matrix(rotation@r.T).as_rotvec()]
    candidates = []
    for weight in (1e3, 1e5):
        H = np.eye(8)+.2*j.T@j
        g = -.3*previous-.2*j.T@(2.*error)
        if len(reserve):
            H += weight/len(reserve)*reserve.T@reserve
            g -= weight/len(reserve)*reserve.T@rate
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(H)), q=g, A=sparse.csc_matrix(A), l=lower, u=upper,
                     verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=8000, polish=True)
        result = solver.solve()
        if result.info.status_val not in (1, 2) or result.x is None or not np.isfinite(result.x).all():
            continue
        command = result.x
        residual = float(max(0., np.max(lower-A@command), np.max(A@command-upper)))
        if residual > 1e-6:
            continue
        proof = collision.sweep(state, command, dt, base_z, tolerance=.0005, max_samples=33,
                                base_rotation=base_rotation)
        clearance = float(proof['lower'].min())
        if controller.limits.command_ramp:
            # Refine a loose constant-command perturbation bound into short
            # certified ramp intervals before declaring the fallback impossible.
            from .actuation import certify_control_ramp
            try:
                clearance = certify_control_ramp(collision.scene, state, previous, command,
                    collision.physical_time, dt, base_z, steps=max(4, int(np.ceil(dt/.002))),
                    constant_certificate=clearance, base_rotation=base_rotation)
            except RuntimeError:
                continue
        if clearance < collision.margin:
            continue
        future = collision.sweep(state, command, .2, base_z, max_samples=9, base_rotation=base_rotation)
        score = float(future['lower'].min())
        candidates.append((score, command.copy(), clearance, residual))
    if not candidates:
        raise RuntimeError('no actuator-bounded command passes direct dynamic sweep certificate')
    score, command, clearance, residual = max(candidates, key=lambda value: value[0])
    controller.previous = command.copy()
    return command, dict(position_error=float(np.linalg.norm(error[:3])),
        orientation_error=float(np.linalg.norm(error[3:])), qp_violation=residual,
        command_sweep_lower_m=clearance, command_diff_drive_violation=0.,
        certified_escape=True, barrier_halfspaces_infeasible=True,
        emergency_future_clearance_diagnostic_m=score,
        dynamic_constraint_count=int(np.sum(approaching)),
        certificate_scope='current control interval; future value is a diagnostic, not a guarantee')


def solve_with_certificate(controller, state, target, rotation, feedforward, dt, **kwargs):
    try:
        return controller.solve(state, target, rotation, feedforward, dt, **kwargs)
    except RuntimeError as primary:
        try:
            braking = {**kwargs, 'brake': True}
            command, info = controller.solve(state, target, rotation, np.zeros(6), dt, **braking)
            info['safety_brake'] = True
        except RuntimeError:
            command, info = certified_escape(controller, state, target, rotation, dt,
                **{key: kwargs[key] for key in ('base_z', 'base_rotation', 'commanded_q') if key in kwargs})
        info['primary_qp_failure'] = str(primary)
        return command, info
