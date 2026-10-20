#!/usr/bin/env python3
"""Direct FCL distance finite differences against captured genuine Pink rows."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import hppfcl


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--recorded',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.diagnostic.resolve();recorded=a.recorded.resolve()
    report=json.loads((root/'summary.json').read_text())
    rows=[json.loads(l) for l in (recorded/'measurements.jsonl').open()]
    initialization=json.loads((recorded/'pink_collision_initialization.json').read_text())
    metadata=json.loads((recorded/'fine_collision_geometry.json').read_text())
    backend=load('gradient_frozen_backend',root/'sources/g1_backend.py')
    collision=load('gradient_frozen_collision',root/'sources/pink_collision.py')
    names=rows[0]['joint_names'];legs=[n for n in names if any(x in n for x in ('hip','knee','ankle'))]
    camera=json.loads((recorded/'sensor_mounts.json').read_text())['camera']['T_parent_optical']
    solver=backend.PinkUpperBody(legs,names,camera,root/'sources/g1_29dof.urdf')
    solver.collision_helper=collision;solver.collision_offsets=initialization['physical_body_frame_offsets']
    solver.collision_geometry=collision.geometry_model(solver.model,metadata,solver.collision_offsets)
    samples=[]
    for item in report['samples']:
        if item['previous_mode']!='last_logged_native' or abs(item['nearby_measurement_s']-item['failure_reported_at_s']+.02)>1e-7:continue
        row=min(rows,key=lambda x:abs(x['stamp_s']-item['nearby_measurement_s']))
        measured=dict(zip(names,row['joint_positions']))
        configuration,reduced_names=solver.configuration(measured)
        matrix=np.load(root/item['saved_matrix'])
        for conflict in item['single_barrier_command_box_conflicts']:
            pair_index=conflict['pair_index'];data=configuration.collision_model.createData();kin=configuration.model.createData()
            def distance(q):
                solver.pin.forwardKinematics(configuration.model,kin,q)
                solver.pin.updateGeometryPlacements(configuration.model,kin,configuration.collision_model,data,q)
                pair=configuration.collision_model.collisionPairs[pair_index]
                first=configuration.collision_model.geometryObjects[pair.first]
                second=configuration.collision_model.geometryObjects[pair.second]
                first_pose=data.oMg[pair.first];second_pose=data.oMg[pair.second]
                request=hppfcl.DistanceRequest();request.gjk_tolerance=1e-12;request.enable_cached_gjk_guess=False
                result=hppfcl.DistanceResult()
                hppfcl.distance(first.geometry,hppfcl.Transform3f(first_pose.rotation,first_pose.translation),
                    second.geometry,hppfcl.Transform3f(second_pose.rotation,second_pose.translation),request,result)
                return result.min_distance
            q=configuration.q.copy();fd=[]
            for i in range(len(q)):
                upper=q.copy();lower=q.copy();upper[i]+=1e-6;lower[i]-=1e-6
                fd.append((distance(upper)-distance(lower))/2e-6)
            native=-matrix['G'][conflict['row']]*.02
            samples.append(dict(stamp_s=row['stamp_s'],pair_index=pair_index,
                pair=[conflict['first'],conflict['second']],distance_m=distance(q),
                joint_names=reduced_names,native_distance_J_m_per_rad=native.tolist(),
                independent_FCL_distance_FD_m_per_rad=fd,
                maximum_absolute_error_m_per_rad=float(np.max(np.abs(native-np.array(fd))))))
    result=dict(schema='independent_g1_barrier_gradient/v1',scope='Genuine Pink gradient vs independent direct reduced-model HPP-FCL distance finite differences. Frozen lower configuration; not physical actor/COM Jacobian validation.',
        epsilon_rad=1e-6,independent_distance_request_gjk_tolerance=1e-12,
        independent_distance_cached_guess=False,
        passed=all(x['maximum_absolute_error_m_per_rad']<1e-5 for x in samples),samples=samples,
        evidence_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [root/'summary.json',root/'sources/g1_backend.py',root/'sources/pink_collision.py',Path(__file__).resolve()]})
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=result['passed'],samples=len(samples),max_error=max(x['maximum_absolute_error_m_per_rad'] for x in samples))))


if __name__=='__main__':main()
