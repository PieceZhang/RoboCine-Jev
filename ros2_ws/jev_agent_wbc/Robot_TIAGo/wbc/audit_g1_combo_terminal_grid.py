#!/usr/bin/env python3
"""Real Pink terminal soft-cost grid; measured-state and ideal CPU comparisons.

Original WORLD camera target and all hard limits/barriers are preserved. No
physical state is executed. Recursive ideal upper motion has base/legs frozen.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial.transform import Rotation


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod);return mod


def error(actual,target):
    return [float(np.linalg.norm(actual[:3,3]-target[:3,3])),
        float(Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude())]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recorded',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();folder=a.recorded.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    frozen=out/'sources';frozen.mkdir()
    for name in ('g1_backend.py','pink_collision.py','g1_29dof.urdf','height_policy_io.yaml'):
        shutil.copy2(folder/'sources'/name,frozen/name)
    shutil.copy2(Path(__file__),frozen/Path(__file__).name)
    backend=load('grid_frozen_backend',frozen/'g1_backend.py');collision=load('grid_frozen_collision',frozen/'pink_collision.py')
    rows=[json.loads(line) for line in (folder/'measurements.jsonl').open()]
    names=rows[0]['joint_names'];legs=[n for n in names if any(t in n for t in ('hip','knee','ankle'))]
    import yaml
    io=yaml.safe_load((frozen/'height_policy_io.yaml').read_text())['articulations']['robot']
    home=dict(zip(io['joint_names'],io['default_joint_pos']))
    camera=json.loads((folder/'sensor_mounts.json').read_text())['camera']['T_parent_optical']
    metadata=json.loads((folder/'fine_collision_geometry.json').read_text())
    offsets=json.loads((folder/'pink_collision_initialization.json').read_text())['physical_body_frame_offsets']
    parameters=((.03,.2),(.1,.2),(.03,.5),(.03,1.),(.1,.5),(.1,1.))
    modes=('measured_q_recorded_previous','measured_q_recursive_previous','ideal_recursive_upper_frozen_lower_base')
    reports=[]
    with (out/'trace.jsonl').open('x') as stream:
        for posture,damping in parameters:
            solver=backend.PinkUpperBody(legs,names,camera,frozen/'g1_29dof.urdf',posture_home=home,
                posture_cost=posture,waist_posture_cost=.15,damping_cost=damping,collision_gain=.5)
            solver.collision_helper=collision;solver.collision_offsets=offsets
            solver.collision_geometry=collision.geometry_model(solver.model,metadata,offsets)
            geom_data=solver.collision_geometry.createData();full_data=solver.model.createData()
            def distance(joints):
                q=collision.full_configuration(solver.model,joints)
                solver.pin.forwardKinematics(solver.model,full_data,q)
                solver.pin.updateGeometryPlacements(solver.model,full_data,solver.collision_geometry,geom_data,q)
                solver.pin.computeDistances(solver.model,full_data,solver.collision_geometry,geom_data,q)
                return float(min(d.min_distance for d in geom_data.distanceResults)),float(min(np.min(q-solver.model.lowerPositionLimit),np.min(solver.model.upperPositionLimit-q)))
            for case in ('left','camera_shot','yaw_left','yaw_right'):
                terminal=[r for r in rows if r['case']==case and r['phase']=='stop']
                terminal=[r for r in terminal if r['stamp_s']>=terminal[-1]['stamp_s']-.5-1e-8]
                first=terminal[0];initial=dict(zip(names,first['joint_positions']))
                for mode in modes:
                    projected=initial.copy();ideal=initial.copy();frozen_base=np.asarray(first['T_world_base'])
                    previous={n:first['pink_upper_velocity'].get(n,0.) for n in solver.previous};solver.previous=previous.copy()
                    positions=[[projected[n] for n in solver.previous]];samples=[];failed=None;started=time.monotonic()
                    for i,row in enumerate(terminal[:-1]):
                        delay=started+i*.02-time.monotonic()
                        if delay>0:time.sleep(delay)
                        measured=dict(zip(names,row['joint_positions']))
                        base=np.asarray(row['T_world_base'])
                        if mode=='ideal_recursive_upper_frozen_lower_base':measured=ideal.copy();base=frozen_base
                        if mode=='measured_q_recorded_previous':
                            solver.previous={n:row['pink_upper_velocity'].get(n,0.) for n in solver.previous}
                        before_previous=solver.previous.copy()
                        target=np.asarray(row['T_world_reference_optical'])
                        try:
                            velocity=solver.velocity(measured,base,target,.02,
                                posture_reference=row['pink_parking_posture']['targets'])
                        except Exception as exc:
                            failed=dict(stamp_s=row['stamp_s'],error=repr(exc));break
                        after=measured.copy()
                        for n,v in velocity.items():
                            after[n]+=.02*v;projected[n]+=.02*v
                        if mode=='ideal_recursive_upper_frozen_lower_base':ideal=after.copy();projected=ideal.copy()
                        positions.append([projected[n] for n in solver.previous])
                        d,margin=distance(after);predicted=solver.predicted_camera(after,base)
                        item=dict(case=case,mode=mode,posture_cost=posture,damping_cost=damping,
                            stamp_s=row['stamp_s'],native_velocity=velocity,
                            peak_velocity_rad_s=max(map(abs,velocity.values())),
                            peak_acceleration_rad_s2=max(abs(velocity[n]-before_previous[n])/.02 for n in velocity),
                            ideal_or_micro_camera_error=error(predicted,target),
                            full548_predicted_distance_m=d,original_position_margin_rad=margin)
                        samples.append(item);stream.write(json.dumps(item,allow_nan=False)+'\n')
                    spans=np.ptp(np.asarray(positions),axis=0)
                    report=dict(case=case,mode=mode,posture_cost=posture,damping_cost=damping,
                        samples=len(samples),window_s=terminal[-1]['stamp_s']-first['stamp_s'],
                        upper_names=list(solver.previous),projected_upper_position_span_rad=spans.tolist(),
                        maximum_upper_position_span_rad=float(max(spans)),
                        upper_position_span_lt10mrad=bool(max(spans)<.01),failed=failed,
                        maximum_camera_error_m=max((x['ideal_or_micro_camera_error'][0] for x in samples),default=None),
                        maximum_camera_error_rad=max((x['ideal_or_micro_camera_error'][1] for x in samples),default=None),
                        minimum_predicted_full548_distance_m=min((x['full548_predicted_distance_m'] for x in samples),default=None),
                        maximum_velocity_rad_s=max((x['peak_velocity_rad_s'] for x in samples),default=None),
                        maximum_acceleration_rad_s2=max((x['peak_acceleration_rad_s2'] for x in samples),default=None),
                        minimum_original_position_margin_rad=min((x['original_position_margin_rad'] for x in samples),default=None),
                        camera_original_tolerance_pass=bool(samples and all(x['ideal_or_micro_camera_error'][0]<=.05 and x['ideal_or_micro_camera_error'][1]<=.15 for x in samples)),
                        full_hard_model_check_pass=bool(len(samples)==25 and all(x['peak_velocity_rad_s']<=.500001 and x['peak_acceleration_rad_s2']<=1.000001 and x['original_position_margin_rad']>=-1e-8 and x['full548_predicted_distance_m']>=.0065 for x in samples)),
                        wall_s=time.monotonic()-started)
                    reports.append(report)
                    print(json.dumps({k:report[k] for k in ('case','mode','posture_cost','damping_cost','maximum_upper_position_span_rad','camera_original_tolerance_pass','full_hard_model_check_pass')}),flush=True)
    paths=[folder/'measurements.jsonl',folder/'sensor_mounts.json',folder/'fine_collision_geometry.json',folder/'pink_collision_initialization.json']+[p for p in frozen.iterdir() if p.is_file()]
    summary=dict(schema='independent_g1_combo_terminal_grid/v1',physical_execution=False,
        scope='25 genuine native Pink solves per original last0.5s. Measured-q modes integrate proposals only as a projected velocity diagnostic, not a physical or kinematic rollout. Ideal recursive mode integrates all15upper joints with base and lower joints fixed at first measured terminal pose. Original WORLD targets/15DoF/native limits/473barriers/548FCL unchanged; no actual gravity/servo/contact simulation.',
        hard_limits=dict(speed=.5,acceleration=1.,barrier_d_min_m=.0065,external_FCL_min_m=.005,collision_gain=.5),
        walking_posture_unchanged=.03,waist_posture_cost=.15,pace=1,reports=reports,
        evidence_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
