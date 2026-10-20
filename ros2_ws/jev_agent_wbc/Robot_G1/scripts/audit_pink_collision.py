#!/usr/bin/env python3
"""CPU model audit against historical, measured native USD collision bodies.

This replays measured transforms for FK/distance comparison. It does not turn a
failed physical trial into a passing one, integrate physics, or certify tracking.
Run using the isolated G1 Pin/HPP-FCL interpreter.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
import yaml


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    if a.output.exists():raise FileExistsError(a.output)
    rows=[json.loads(x) for x in (a.trace/'measurements.jsonl').read_text().splitlines()];bytime={r['stamp_s']:r for r in rows}
    evidence={r['fine_collision_evidence']['stamp_s']:r['fine_collision_evidence'] for r in rows if 'fine_collision_evidence' in r}
    if (a.trace/'constraint_abort.json').is_file():
        abort=json.loads((a.trace/'constraint_abort.json').read_text())['collision'];evidence[abort['stamp_s']]=abort
    first=evidence[min(evidence)];r=bytime[first['stamp_s']]
    backend=load('actual_pink_backend',root/'wbc_backend.py');rig=load('actual_rig',root.parent/'jev_agent_wbc/robot_sensor_rig.py')
    io=yaml.safe_load((root/'config/height_policy_io.yaml').read_text());sensor,_=rig.load_config('g1',root.parent/'config/robot_sensor_mounts.json')
    solver=backend.PinkUpperBody(io['actions'][0]['joint_names'],io['articulations']['robot']['joint_names'],rig.optical_transform(sensor['camera']),
        root/'kinematics/g1_29dof.urdf',posture_home=dict(zip(io['articulations']['robot']['joint_names'],io['articulations']['robot']['default_joint_pos'])),posture_cost=.03,damping_cost=.02)
    metadata=json.loads((a.trace/'fine_collision_geometry.json').read_text());q=dict(zip(r['joint_names'],r['joint_positions']))
    initialization=solver.configure_collisions(metadata,q,r['T_world_base'],first['T_world_collision_bodies'])
    errors=[];distance_comparisons=[];start=time.monotonic()
    chosen=set(np.linspace(0,len(evidence)-1,min(21,len(evidence)),dtype=int))
    for i,(stamp,e) in enumerate(sorted(evidence.items())):
        measured=bytime[stamp];joints=dict(zip(measured['joint_names'],measured['joint_positions']))
        predicted=solver.collision_helper.predicted_body_poses(solver.model,joints,measured['T_world_base'],solver.collision_offsets)
        for name,P in predicted.items():
            P=np.asarray(P);T=np.asarray(e['T_world_collision_bodies'][name])
            errors.append((np.linalg.norm(P[:3,3]-T[:3,3]),Rotation.from_matrix(P[:3,:3]@T[:3,:3].T).magnitude()))
        if i in chosen:
            geom=solver.collision_geometry;data=solver.model.createData();gd=geom.createData()
            pin.computeDistances(solver.model,data,geom,gd,solver.collision_helper.full_configuration(solver.model,joints))
            actual=float(min(x.min_distance for x in gd.distanceResults));native=e['minimum_clearance_lower_bound_m']
            distance_comparisons.append(dict(stamp_s=stamp,pin_hppfcl_minimum_m=actual,measured_native_fcl_minimum_m=native,difference_m=actual-native))
    solve_start=time.monotonic();solver.velocity(q,r['T_world_base'],r['T_world_reference_optical'],.02);solve_ms=(time.monotonic()-solve_start)*1000
    values=np.max(errors,axis=0);delta=max(abs(x['difference_m']) for x in distance_comparisons)
    source_paths=[Path(__file__).resolve(),root/'pink_collision.py',root/'wbc_backend.py',root/'kinematics/g1_29dof.urdf',a.trace/'measurements.jsonl',a.trace/'fine_collision_geometry.json']
    result=dict(schema='g1_actual_geometry_pink_cpu_audit/v1',scope='CPU FK and distance agreement on failed historical physical trace; not new physics/trajectory acceptance',
        measured_frame_samples=len(errors),physical_shape_count=initialization['shape_count'],all_source_nonfiltered_pairs=initialization['all_nonfiltered_pairs'],
        initial_calibration_stamp_s=first['stamp_s'],maximum_frame_position_error_m=float(values[0]),maximum_frame_rotation_error_rad=float(values[1]),
        maximum_fcl_library_distance_difference_m=delta,distance_comparisons=distance_comparisons,genuine_pink_barrier_qp=solver.last_collision_report,
        first_qp_wall_ms=solve_ms,wall_seconds=time.monotonic()-start,passed=bool(values[0]<.001 and values[1]<.001 and delta<.001),
        source_sha256={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in source_paths},initialization=initialization)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('initialization','source_sha256','distance_comparisons')}))
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
