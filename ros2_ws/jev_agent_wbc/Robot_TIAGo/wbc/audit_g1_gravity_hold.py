#!/usr/bin/env python3
"""True Pin actual-mass comparison and short native CPU timing; no Kit."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-dir',type=Path,required=True);p.add_argument('--recorded',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);sources=a.output/'sources';sources.mkdir()
    urdf=a.robot_dir/'kinematics/g1_29dof.urdf';metadata_path=a.robot_dir/'config/actual_gravity_model.json'
    for source in [a.robot_dir/'gravity_hold.py',a.robot_dir/'gravity_model.py',urdf,metadata_path,Path(__file__)]:
        shutil.copy2(source,sources/source.name)
    prototype=load('fresh_measured_numpy_gravity',sources/'gravity_hold.py')
    metadata=json.loads((sources/metadata_path.name).read_text())
    started=time.perf_counter_ns();model=prototype.GravityHold(sources/urdf.name,metadata)
    construction_ms=(time.perf_counter_ns()-started)/1e6
    imports=[]
    for node in ast.walk(ast.parse((sources/'gravity_hold.py').read_text())):
        if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module)
    assert not any(x.split('.')[0]in ('pinocchio','scipy','torch','omni','isaacsim','isaaclab')for x in imports)
    pin_module=load('unchanged_actual_pointmass_pin_model',sources/'gravity_model.py')
    reference=pin_module.ActualGravity(sources/urdf.name,metadata)
    from scipy.spatial.transform import Rotation
    rows=[json.loads(line)for line in(a.recorded/'measurements.jsonl').open()]
    cases=[]
    for index in np.linspace(0,len(rows)-1,10,dtype=int):
        row=rows[index];cases.append(dict(label='measured_'+str(index),stamp_s=row['stamp_s'],
            q=dict(zip(row['joint_names'],row['joint_positions'])),T=np.asarray(row['T_world_base'])))
    rng=np.random.default_rng(723901);limits={}
    for joint in ET.parse(sources/urdf.name).getroot().findall('joint'):
        if joint.get('type')in ('revolute','continuous'):
            limit=joint.find('limit');limits[joint.get('name')]=(float(limit.get('lower')),float(limit.get('upper')))
    for index in range(10):
        q={n:float(np.clip(metadata['calibration_joint_positions'][n]+rng.uniform(-.3,.3),limits[n][0]+.001,limits[n][1]-.001))for n in model.joint_names}
        T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',rng.uniform([-.4,-.4,-3.],[.4,.4,3.])).as_matrix()
        T[:3,3]=rng.uniform(-1.,1.,3)
        cases.append(dict(label='synthetic_full_pose_'+str(index),q=q,T=T))
    comparisons=[]
    for case in cases:
        expected=reference.torque(case['q'],case['T']);actual=model.torque(case['q'],case['T'])
        upper=model.upper_torque(case['q'],case['T'][:3,:3])
        vector=model.torque_vector(np.asarray([case['q'][n]for n in model.joint_names]),case['T'])
        errors={n:abs(actual[n]-expected[n])for n in model.joint_names}
        assert set(upper)==set(prototype.UPPER_JOINT_NAMES)
        np.testing.assert_array_equal(vector,np.asarray([actual[n]for n in model.joint_names]))
        comparisons.append(dict(label=case['label'],max_all29_error_Nm=float(max(errors.values())),
            max_upper15_error_Nm=float(max(errors[n]for n in upper)),
            named_torque_Nm=actual,pin_actual_mass_torque_Nm=expected))
    checks=[]
    def rejected(name,callback):
        try:callback();ok=False
        except (ValueError,TypeError,KeyError):ok=True
        checks.append(dict(name=name,rejected=ok));assert ok,name
    q=cases[0]['q'];T=cases[0]['T'];name=model.joint_names[0]
    for label,value in [('boolean',True),('string','0.1'),('nan',np.nan),('inf',np.inf)]:
        bad=q.copy();bad[name]=value;rejected('joint_'+label,lambda:model.torque(bad,T))
    bad=q.copy();bad.pop(name);rejected('missing_joint',lambda:model.torque(bad,T))
    bad=q.copy();bad['foreign_joint']=0.;rejected('extra_joint',lambda:model.torque(bad,T))
    rejected('wrong_vector_size',lambda:model.torque_vector(np.zeros(28),T))
    reflected=T.copy();reflected[:3,0]*=-1;rejected('reflected_pelvis',lambda:model.torque(q,reflected))
    scaled=T.copy();scaled[0,0]+=.01;rejected('nonrotation_pelvis',lambda:model.torque(q,scaled))
    rejected('boolean_pelvis',lambda:model.torque(q,np.eye(3,dtype=bool)))
    badT=T.copy();badT[3,0]=.01;rejected('invalid_bottom_row',lambda:model.torque(q,badT))
    badT=T.copy();badT[2,3]=np.nan;rejected('nonfinite_pelvis',lambda:model.torque(q,badT))
    for label,edit in [('wrong_urdf_binding',lambda m:m.update(source_urdf_sha256='0'*64)),
            ('missing_actual_body',lambda m:m['bodies'].pop('pelvis')),
            ('negative_mass',lambda m:m['bodies']['pelvis'].update(mass_kg=-1.)),
            ('wrong_actual_mass_total',lambda m:m.update(total_actual_mass_kg=m['total_actual_mass_kg']+.1)),
            ('wrong_parent_joint',lambda m:m['bodies']['pelvis'].update(parent_joint='waist_yaw_joint'))]:
        bad=copy.deepcopy(metadata);edit(bad)
        rejected(label,lambda:prototype.GravityHold(sources/urdf.name,bad))
    baseline=model.torque(q,T);changed=q.copy();changed['right_elbow_joint']+=.1
    changed_tau=model.torque(changed,T)
    assert max(abs(changed_tau[n]-baseline[n])for n in baseline)>.001
    np.testing.assert_array_equal(list(model.torque(q,T).values()),list(baseline.values()))
    # Immutable compiled parameters; no torque/output or prior input cache.
    mutated=copy.deepcopy(metadata);independent=prototype.GravityHold(sources/urdf.name,mutated)
    mutated['bodies']['torso_link']['mass_kg']=1e6
    np.testing.assert_array_equal(list(independent.torque(q,T).values()),list(baseline.values()))
    calls=1000;timings={}
    ordered=[np.asarray([c['q'][n]for n in model.joint_names])for c in cases]
    functions={
        'named29':lambda i:model.torque(cases[i]['q'],cases[i]['T']),
        'upper15':lambda i:model.upper_torque(cases[i]['q'],cases[i]['T'][:3,:3]),
        'vector29':lambda i:model.torque_vector(ordered[i],cases[i]['T'][:3,:3]),
    }
    for label,function in functions.items():
        samples=[]
        for i in range(calls):
            begin=time.perf_counter_ns();function(i%20);samples.append((time.perf_counter_ns()-begin)/1e6)
        timings[label]=dict(calls=calls,median_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),
            p99_ms=float(np.percentile(samples,99)),maximum_ms=float(max(samples)),
            count_over_half_ms=sum(v>=.5 for v in samples))
        np.save(a.output/(label+'_call_ms.npy'),samples)
    maximum=max(c['max_all29_error_Nm']for c in comparisons)
    report=dict(schema='g1_numpy_gravity_hold_cpu/v1',prototype_only=True,connected_to_runner=False,
        physical_execution=False,construction_wall_ms=construction_ms,pose_count=len(cases),
        compared_all29=len(cases)*29,compared_upper15=len(cases)*15,maximum_error_Nm=maximum,
        numerical_threshold_Nm=1e-8,numerical_pass=maximum<1e-8,
        negative_checks=checks,no_stale_q_or_torque_reuse=True,external_metadata_mutation_isolated=True,
        module_imports=imports,depends_on_pin_in_kit=False,
        model=model.manifest(),pose_comparisons=comparisons,timings=timings,
        observed_all_calls_under_half_ms=all(v['count_over_half_ms']==0 for v in timings.values()),
        timing_scope='Natural CPU wall-call time with no deliberate sleep. Finite observations are not a hard real-time guarantee or physical stability proof.',
        input_sha256={str((a.recorded/'measurements.jsonl').resolve()):hashlib.sha256((a.recorded/'measurements.jsonl').read_bytes()).hexdigest()},
        frozen_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sources.iterdir()if p.is_file()})
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ('pose_comparisons','negative_checks','model')},indent=2))


if __name__=='__main__':main()
