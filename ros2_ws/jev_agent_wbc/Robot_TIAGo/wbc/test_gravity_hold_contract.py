"""Synthetic write metadata checked against saved genuine Pin CPU expectations."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    root=Path(__file__).resolve().parents[2];out=Path(__file__).with_name('results')/'g1_gravity_hold_contract01'
    out.mkdir(exist_ok=False);sources=out/'sources';sources.mkdir()
    source=Path(__file__).with_name('gravity_hold_contract.py')
    shutil.copy2(source,sources/source.name);shutil.copy2(__file__,sources/Path(__file__).name)
    spec=importlib.util.spec_from_file_location('frozen_gravity_contract',sources/source.name)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cpu_path=out.parent/'g1_gravity_hold_cpu01/summary.json';cpu=json.loads(cpu_path.read_text())
    selected=cpu['pose_comparisons'][:4];indices=[int(x['label'].split('_')[1])for x in selected]
    raw=root/'Robot_G1/results/adaptive_park_posture_damping03_01/measurements.jsonl'
    captured={}
    for i,line in enumerate(raw.open()):
        if i in indices:captured[i]=json.loads(line)
        if i==max(indices):break
    names=captured[indices[0]]['joint_names'];binding='a'*64;expected=[];writes=[]
    for s,(index,reference)in enumerate(zip(indices,selected)):
        row=captured[index];q=row['joint_positions'];R=np.asarray(row['T_world_base'])[:3,:3].tolist()
        computed=[reference['named_torque_Nm'][n]for n in m.OWNERS]
        g=np.asarray(computed,np.float32).astype(float).tolist();expected.append({n:reference['pin_actual_mass_torque_Nm'][n]for n in m.OWNERS})
        stamp=10+s*.005;seq=2000+s
        writes.append(dict(substep=s,physics_state_seq=seq,measured_stamp_s=stamp,write_stamp_s=stamp,
            motion_ready=False,source='local_static_model',status='computed',reason='same_current_measured_state',
            position_target15=[0.]*15,held_reference15=[0.]*15,velocity_target15=[0.]*15,
            worker_gravity15=[0.]*15,local_gravity15=g,applied_gravity15=g.copy(),computed_gravity15=computed,
            trajectory_reused=False,effort_composition='existing_explicit_pd_plus_g_single_clamp',
            joint_names=names,measured_q29=q,pelvis_R=R,measurement_source='physx_sdk_current_after_update',
            sample_token=m.sample_token(physics_state_seq=seq,stamp_s=stamp,joint_names=names,positions_rad=q,pelvis_R=R)))
    record=dict(schema='g1_upper_static_gravity_support/v1',model_binding_id=binding,physics_dt_s=.005,writes=writes)
    kwargs=dict(control_start_s=10.,first_physics_state_seq=2000,plant_joint_names=names,model_binding_id=binding,
        previous_position_target15=[0.]*15,motion_lease_report=None,expected_gravity15=expected,
        expected_initial_q29=writes[0]['measured_q29'],expected_initial_pelvis_R=writes[0]['pelvis_R'])
    results=[]
    def test(name,data,valid,**updates):
        args=dict(kwargs);args.update(updates);result=m.validate_tick(data,**args)
        results.append(dict(name=name,expected_valid=valid,passed=result['valid']is valid,result=result))
    test('all_four_current_static_Pin_references',record,True)
    fresh=copy.deepcopy(record)
    for w in fresh['writes']:
        w.update(motion_ready=True,source='worker_motion',status='worker',computed_gravity15=None,
            local_gravity15=[0.]*15,worker_gravity15=[.125]*15,applied_gravity15=[.125]*15)
    lease=dict(checks=[dict(valid=True,sim_age_s=.02+s*.005,wall_age_s=.02+s*.005,execution_horizon_s=.005)for s in range(4)])
    test('four_original_fresh_worker_writes',fresh,True,motion_lease_report=lease)
    mixed=copy.deepcopy(record);mixed['writes'][:2]=copy.deepcopy(fresh['writes'][:2])
    test('fresh_prefix_then_expired_static_suffix',mixed,True,motion_lease_report=lease)
    zero=copy.deepcopy(record)
    for w in zero['writes']:w.update(source='zero_fallback',status='rejected',reason='invalid_measured_input',
        local_gravity15=[0.]*15,applied_gravity15=[0.]*15,computed_gravity15=None,measured_q29=None,pelvis_R=None,sample_token=None)
    test('invalid_input_zero_without_any_cached_gravity',zero,True)
    mutations={
        'missing_write':lambda r:r['writes'].pop(),
        'stale_sample_clock':lambda r:r['writes'][2].update(measured_stamp_s=10.),
        'future_write_clock':lambda r:r['writes'][2].update(write_stamp_s=10.02),
        'reused_sample_token':lambda r:r['writes'][2].update(sample_token=r['writes'][1]['sample_token']),
        'wrong_plant_order':lambda r:r['writes'][2].update(joint_names=list(reversed(names))),
        'old_worker_gravity_reused':lambda r:r['writes'][2].update(worker_gravity15=[.1]*15),
        'nonfresh_velocity':lambda r:r['writes'][2].update(velocity_target15=[.01]*15),
        'uncommitted_position':lambda r:r['writes'][2].update(position_target15=[float(np.float32(.01))]*15),
        'wrong_model_binding':lambda r:r.update(model_binding_id='b'*64),
        'clipped_local_torque':lambda r:r['writes'][2].update(local_gravity15=[0.]*15,applied_gravity15=[0.]*15),
        'wrong_pose_rotation':lambda r:r['writes'][2].update(pelvis_R=(-np.eye(3)).tolist()),
        'bool_sequence':lambda r:r['writes'][2].update(physics_state_seq=True),
        'extra_clamp_mode':lambda r:r['writes'][2].update(effort_composition='preclip_g_then_total_clip'),
        'trajectory_reused':lambda r:r['writes'][2].update(trajectory_reused=True),
    }
    for name,mutate in mutations.items():
        bad=copy.deepcopy(record);mutate(bad);test(name,bad,False)
    bad=copy.deepcopy(record);w=bad['writes'][2];w['computed_gravity15'][3]+=1.
    w['local_gravity15']=np.asarray(w['computed_gravity15'],np.float32).astype(float).tolist();w['applied_gravity15']=w['local_gravity15'].copy()
    test('self_consistent_wrong_g_fails_independent_Pin',bad,False)
    test('local_without_independent_reference',record,False,expected_gravity15=None)
    expired=copy.deepcopy(lease);expired['checks'][2]['sim_age_s']=.06
    test('worker_does_not_gain_new_lease_from_static_profile',fresh,False,motion_lease_report=expired)
    (out/'synthetic_write_fixture.json').write_text(json.dumps(dict(record=record,independent_Pin_expected15=expected),indent=2)+'\n')
    report=dict(schema='g1_gravity_hold_contract_cpu/v1',passed=all(r['passed']for r in results),checks=len(results),
        results=results,physical_execution=False,metadata_only=True,
        independent_reference='Saved genuine actual-mass Pin CPU01 torques, not gravity_hold computation inside this validator.',
        saved_Pin_summary_sha256=hashlib.sha256(cpu_path.read_bytes()).hexdigest(),
        scope='Synthetic chosen metadata and sparse original input poses. No physical replay or source/sensor/clamp certification.',
        frozen_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sources.iterdir()if p.is_file()})
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=report['passed'],checks=len(results),failed=[r['name']for r in results if not r['passed']]),indent=2))


if __name__=='__main__':main()
