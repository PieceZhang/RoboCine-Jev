#!/usr/bin/env python3
"""Genuine Giskard solver + real TIAGo URDF task, CPU component validation.

Not the complete Giskard World/Executor; not a physical closed loop. No fallback
optimizer. The only solver is upstream QPSolverPIQP from the locked source.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wbc_backend import (TimedCameraReference,add_pinned_core_paths,
                         differential_wheel_rates)
from kinematics import TiagoKinematics,pose_error,transform


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GenuineGiskardRateTask:
    """TIAGo task/constraints adapter to Giskard's explicit solver interface.

    Six equality slack variables permit unreachable camera targets to yield a
    residual rather than relax hard actuator constraints. Source position,
    velocity and wheel velocity bounds are retained. Additional acceleration
    and base speed bounds are conservative fixture policy, not manufacturer
    acceleration/torque certification. No collision constraint is provided.
    """
    def __init__(self,model,dt=.01,arm_only=False):
        source,self.commit = add_pinned_core_paths()
        from giskardpy.qp.qp_data import QPDataExplicit
        from giskardpy.qp.solvers.qp_solver_piqp import QPSolverPIQP
        self.Data = QPDataExplicit
        self.solver = QPSolverPIQP()  # big_ball_mode stays FALSE
        self.source_path = Path(inspect.getfile(QPSolverPIQP)).resolve()
        if not self.source_path.is_relative_to(source.resolve()):
            raise RuntimeError('Giskard solver does not come from locked upstream tree')
        self.model,self.dt,self.arm_only = model,dt,arm_only
        self.acceleration = np.array([.45,.65,.20,*([1.0]*7)])
        self.rate_limit = np.r_[.22,.35,model.velocity]
        radius,separation = model.profile['wheel_radius_m'],model.profile['wheel_separation_m']
        self.wheel_map = np.zeros((2,10))
        self.wheel_map[:,:2] = np.array([[1.,-separation/2],[1.,separation/2]])/radius

    def bounds(self,q,previous):
        dt = self.dt
        lo = np.maximum(-self.rate_limit,previous-self.acceleration*dt)
        hi = np.minimum(self.rate_limit,previous+self.acceleration*dt)
        # One-step source position limits plus a conservative braking envelope.
        a = self.acceleration[2:]
        lower_distance = np.maximum(0.,q-self.model.lower)
        upper_distance = np.maximum(0.,self.model.upper-q)
        brake_lo = a*dt-np.sqrt((a*dt)**2+2*a*lower_distance)
        brake_hi = -a*dt+np.sqrt((a*dt)**2+2*a*upper_distance)
        lo[2:] = np.maximum.reduce([lo[2:],(self.model.lower-q)/dt,brake_lo])
        hi[2:] = np.minimum.reduce([hi[2:],(self.model.upper-q)/dt,brake_hi])
        if self.arm_only:
            lo[:2],hi[:2] = 0.,0.
        if np.any(lo>hi+1e-10):
            raise RuntimeError('Empty hard rate/braking bounds; no relaxed fallback')
        return lo,hi

    def solve(self,J,desired_twist,q,previous):
        lo,hi = self.bounds(q,previous)
        # Diagonal upstream QP Hessian: express tracking through explicit slack,
        # rather than silently discard off-diagonal terms of J.T@J.
        data = self.Data(num_equality_slack_variables=6,num_inequality_slack_variables=0,
            quadratic_weights=np.r_[.002,.002,*([.02]*8),*([100.]*6)],
            linear_weights=np.zeros(16),box_lower_constraints=np.r_[lo,*([-np.inf]*6)],
            box_upper_constraints=np.r_[hi,*([np.inf]*6)],
            equality_matrix=sparse.csc_matrix(np.c_[J,-np.eye(6)]),
            equality_bounds=desired_twist,
            inequality_matrix=sparse.csc_matrix(np.c_[self.wheel_map,np.zeros((2,6))]),
            inequality_lower_bounds=np.full(2,-self.model.wheel_velocity),
            inequality_upper_bounds=np.full(2,self.model.wheel_velocity))
        started = time.perf_counter()
        solution = self.solver.solver_call(data)
        elapsed = time.perf_counter()-started
        residual = float(np.max(np.abs(data.equality_matrix@solution-desired_twist)))
        command = solution[:10].copy()
        violation = max(float(np.max(lo-command)),float(np.max(command-hi)),
            float(np.max(np.abs(self.wheel_map@command)-self.model.wheel_velocity)),0.)
        return command,elapsed,residual,violation,solution[10:].copy()


def fixture_reference(initial,duration=4.):
    # Authored timestamps/velocities remain explicit. No timing compression.
    times = np.linspace(0.,duration,9)
    samples = []
    for t in times:
        phase = 2*np.pi*t/duration
        offset = np.array([.065*(1-np.cos(phase)),.045*np.sin(phase),.015*(1-np.cos(phase))])
        velocity = np.array([.065*np.sin(phase),.045*np.cos(phase),.015*np.sin(phase)])*2*np.pi/duration
        rotation = Rotation.from_rotvec(np.array([.025,-.035,.055])*np.sin(phase))
        rotation = rotation*Rotation.from_matrix(initial[:3,:3])
        samples.append(dict(t_s=float(t),position_m=(initial[:3,3]+offset).tolist(),
            linear_velocity_world_mps=velocity.tolist(),quaternion_xyzw=rotation.as_quat().tolist()))
    return samples


def run_case(model,arm_only):
    base = transform([0.,0.,.005],[0.,0.,-.6])
    q,previous = model.home.copy(),np.zeros(10)
    initial,_ = model.pose_and_jacobian(base,q)
    reference_knots = fixture_reference(initial)
    reference = TimedCameraReference(reference_knots)
    task = GenuineGiskardRateTask(model,arm_only=arm_only)
    records = []
    for step in range(400):
        t = step*task.dt
        target = reference.evaluate(t)
        camera,J = model.pose_and_jacobian(base,q)
        error = pose_error(target.T_world_optical,camera)
        command,elapsed,residual,violation,slack = task.solve(J,target.twist_world+4.*error,q,previous)
        records.append(dict(t_s=t,base=base.tolist(),q=q.tolist(),camera=camera.tolist(),
            target=target.T_world_optical.tolist(),command=command.tolist(),error=error.tolist(),
            equality_residual=residual,bound_violation=violation,solve_wall_s=elapsed,
            tracking_slack=slack.tolist(),wheel_rates=differential_wheel_rates(*command[:2]).tolist()))
        base,q = model.integrate(base,q,command,task.dt)
        previous = command
    errors = np.array([r['error'] for r in records])
    all_q = np.array([r['q'] for r in records]+[q.tolist()])
    rates = np.array([r['command'] for r in records])
    acceleration = np.diff(np.vstack([np.zeros(10),rates]),axis=0)/task.dt
    latencies = np.array([r['solve_wall_s'] for r in records])
    base_positions = np.array([np.array(r['base'])[:3,3] for r in records])
    summary = dict(arm_only=arm_only,steps=len(records),logical_dt_s=task.dt,
        camera_position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:,:3]**2,axis=1)))),
        camera_orientation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:,3:]**2,axis=1)))),
        max_equality_residual=max(r['equality_residual'] for r in records),
        max_hard_bound_violation=max(r['bound_violation'] for r in records),
        max_acceleration_excess=float(max(0.,np.max(np.abs(acceleration)-task.acceleration))),
        source_position_limits_respected=bool(np.all(all_q>=model.lower-1e-8) and np.all(all_q<=model.upper+1e-8)),
        source_velocity_limits_respected=bool(np.all(np.abs(rates[:,2:])<=model.velocity+1e-8)),
        base_position_span_m=float(np.linalg.norm(np.ptp(base_positions,axis=0))),
        max_base_forward_mps=float(np.max(np.abs(rates[:,0]))),
        max_base_yaw_radps=float(np.max(np.abs(rates[:,1]))),
        max_joint_displacement=np.max(np.abs(all_q-model.home),axis=0).tolist(),
        solver_median_wall_s=float(np.median(latencies)),solver_p99_wall_s=float(np.quantile(latencies,.99)),
        big_ball_mode=task.solver.big_ball_mode)
    summary['passed'] = bool(summary['camera_position_rmse_m']<.01 and summary['camera_orientation_rmse_rad']<.01
        and summary['max_equality_residual']<1e-5 and summary['max_hard_bound_violation']<1e-6
        and summary['max_acceleration_excess']<1e-5 and summary['source_position_limits_respected']
        and summary['source_velocity_limits_respected'] and not summary['big_ball_mode']
        and (summary['base_position_span_m']<1e-8 if arm_only else summary['base_position_span_m']>.005))
    return summary,records,reference_knots,task


def check_reject_infeasible(task):
    from giskardpy.qp.exceptions import InfeasibleException
    data = task.Data(num_equality_slack_variables=0,num_inequality_slack_variables=0,
        quadratic_weights=np.ones(1),linear_weights=np.zeros(1),
        box_lower_constraints=np.zeros(1),box_upper_constraints=np.zeros(1),
        equality_matrix=sparse.csc_matrix([[1.]]),equality_bounds=np.ones(1),
        inequality_matrix=sparse.csc_matrix((0,1)),inequality_lower_bounds=np.zeros(0),inequality_upper_bounds=np.zeros(0))
    try:
        task.solver.solver_call(data)
    except InfeasibleException:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model = TiagoKinematics()
    cases = []
    for arm_only in (True,False):
        summary,records,knots,task = run_case(model,arm_only)
        name = 'arm_only' if arm_only else 'coordinated'
        (args.output/f'{name}_trace.json').write_text(json.dumps(records,indent=2)+'\n')
        (args.output/f'{name}_reference.json').write_text(json.dumps(knots,indent=2)+'\n')
        cases.append(summary)
    rejected = check_reject_infeasible(task)
    source_paths = [ROOT/'wbc_backend.py',ROOT/'wbc/kinematics.py',Path(__file__),
        ROOT/'robot.json',ROOT/'sensors.json',ROOT/'urdf/tiago_physics.urdf',task.source_path,
        task.source_path.parents[1]/'qp_data.py']
    result = dict(schema='tiago_genuine_giskard_core_fixture/v1',
        execution_kind='CPU kinematic task fixture using upstream Giskard explicit QP solver; no PhysX',
        solver_class='giskardpy.qp.solvers.qp_solver_piqp.QPSolverPIQP',
        upstream_commit=task.commit,solver_source=str(task.source_path),
        command_coordinates=['body_forward_mps','body_yaw_radps',*['torso_lift_joint',*(f'arm_{i}_joint' for i in range(1,8))]],
        camera_frame='tiago_camera_optical_frame',cases=cases,infeasible_problem_rejected=rejected,
        continuous_reference_interpolation='authored-time CubicHermiteSpline + RotationSpline, world spatial twist',
        environment_constraints_verified=False,full_world_executor_verified=False,
        physical_tracking_verified=False,representation_ingress_verified=False,
        pace='offline CPU component benchmark, no deliberate wall-clock delay, no simulation pacing claim',
        source_files_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths},
        passed=bool(all(c['passed'] for c in cases) and rejected))
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],cases=cases,infeasible_problem_rejected=rejected),indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
