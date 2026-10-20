#!/usr/bin/env python3
"""Actual Giskard availability/source/model preflight; never dispatch motion."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ['XDG_CACHE_HOME'] = str(ROOT/'wbc/cache')
os.environ['MPLCONFIGDIR'] = str(ROOT/'wbc/cache/matplotlib')
sys.path.insert(0,str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(function):
    try:
        return dict(passed=True,result=function())
    except Exception as exc:
        return dict(passed=False,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())


def source_probe():
    lock = json.loads((ROOT/'wbc/FRAMEWORK_SOURCE_LOCK.json').read_text())
    entries = []
    for entry in lock['repositories']:
        path = ROOT/'wbc/vendor'/entry['name']
        if not path.exists() and entry['name']!='cognitive_robot_abstract_machine':
            entries.append(dict(name=entry['name'],optional_legacy_source_present=False))
            continue
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
        modified = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=path,text=True).strip()
        if commit!=entry['commit'] or modified:
            raise RuntimeError(f'{entry["name"]}: source differs from pinned checkout')
        entries.append(dict(name=entry['name'],commit=commit,tracked_source_unmodified=True))
    return entries


def solver_probe():
    from wbc_backend import add_pinned_core_paths
    source,commit = add_pinned_core_paths()
    from giskardpy.qp.solvers.qp_solver_piqp import QPSolverPIQP
    import piqp
    solver = QPSolverPIQP()
    path = Path(inspect.getfile(QPSolverPIQP)).resolve()
    if not path.is_relative_to(source.resolve()) or solver.big_ball_mode:
        raise RuntimeError('Unexpected solver origin or unsafe big_ball_mode')
    return dict(class_name=type(solver).__module__+'.'+type(solver).__name__,
        source=str(path.relative_to(ROOT)),sha256=digest(path),upstream_commit=commit,
        native_piqp_version=piqp.__version__,big_ball_mode=solver.big_ball_mode)


def model_probe():
    from kinematics import TiagoKinematics,transform
    model = TiagoKinematics()
    pose,J = model.pose_and_jacobian(transform(),model.home)
    return dict(single_arm_source_joints=[j['name'] for j in model.profile['joints']],
        controlled_coordinates=10,jacobian_shape=list(J.shape),home_optical_pose=pose.tolist(),
        source_urdf_sha256=digest(ROOT/'urdf/tiago_physics.urdf'),
        original_position_velocity_effort_limits_retained=True,
        differential_base_lateral_command_exists=False)


def world_probe():
    import numpy as np
    import time
    from full_framework import FullGiskardExecutor
    from source_state import source_state
    spec=json.loads((ROOT/'robot.json').read_text())
    metadata=json.loads((ROOT/'wbc/results/collision_geometry_01.json').read_text())
    solver=FullGiskardExecutor(spec,metadata,ROOT/'wbc/cache/preflight_world')
    q=np.array([j['home'] for j in spec['joints']]);poses,js=source_state(spec,q)
    request=dict(q=q,previous=np.zeros(10),measured_rates=np.zeros(10),poses=poses,body_jacobians=js,
        mode='actuator',desired_rates=np.r_[.1,0.,np.zeros(8)],sample_s=0.,phase='cpu_preflight')
    start=time.monotonic();actuator=solver.solve(request)
    remaining=start+.01-time.monotonic()
    if remaining>0:time.sleep(remaining)
    optical=solver.world.compute_forward_kinematics_np(solver.world.root,solver.camera)
    reference=optical.copy();reference[0,3]+=.001
    request.update(sample_s=.01,mode='camera',reference=reference,feedforward=np.zeros(6))
    camera=solver.solve(request)
    if not actuator['full_world_executor'] or not camera['full_world_executor'] or actuator['command'][0]<=0.:
        raise RuntimeError('Actual World/Executor did not execute actuator and optical camera goals')
    return dict(world_type=type(solver.config).__name__,
        executor_class=type(solver.executor).__module__+'.'+type(solver.executor).__name__,
        real_goal_class=type(solver.binding.goal).__module__+'.'+type(solver.binding.goal).__name__,
        compiled_executor_executed=True,actuator_and_optical_camera_goals_executed=True,
        active_coordinates=10,model_manifest=solver.model_manifest,
        native_actuator_proposal=actuator['upstream_proposal'].tolist(),
        native_camera_proposal=camera['upstream_proposal'].tolist(),
        maximum_hard_policy_violation=max(actuator['hard_constraint_violation'],camera['hard_constraint_violation']),
        measured_physical_tracking_verified=False,
        warning='CPU source-FK fixture only. No physical runtime or environment success inferred.')


def evidence_probe(path):
    report = json.loads(path.read_text())
    if report.get('passed') is not True or report.get('full_world_executor_verified') is not False:
        raise RuntimeError('Missing/failed or incorrectly classified component evidence')
    changed = [name for name,expected in report['source_files_sha256'].items()
        if not (ROOT/name).is_file() or digest(ROOT/name)!=expected]
    if changed:
        raise RuntimeError('Component result stale after source changes: '+', '.join(changed))
    return dict(path=str(path.relative_to(ROOT)),sha256=digest(path),
        genuine_core_solver_fixture_passed=True,physical_tracking_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--component-result',type=Path,default=ROOT/'wbc/results/core_04/summary.json')
    args = parser.parse_args()
    results = dict(schema='tiago_wbc_preflight/v1',python=sys.version,executable=sys.executable,
        platform=platform.platform(),ros_distro=os.environ.get('ROS_DISTRO'),
        rclpy_discoverable=importlib.util.find_spec('rclpy') is not None,
        sources=probe(source_probe),genuine_solver=probe(solver_probe),
        real_robot_model=probe(model_probe),full_world_goal=probe(world_probe),
        component_evidence=probe(lambda:evidence_probe(args.component_result)),
        motion_dispatched=False,backend_runtime_ready=False,
        representation_ingress_verified=False,continuous_camera_tracking_verified=False,
        environment_constraints_verified=False,sensors_avoidance_verified=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(results,stream,indent=2)
        stream.write('\n')
    print(json.dumps({key:results[key]['passed'] for key in
        ('sources','genuine_solver','real_robot_model','full_world_goal','component_evidence')},indent=2))
    # Installed components are valuable but cannot turn a missing physical backend into readiness.
    return 0 if all(results[key]['passed'] for key in ('sources','genuine_solver','real_robot_model','full_world_goal','component_evidence')) else 1


if __name__=='__main__':
    raise SystemExit(main())
