#!/usr/bin/env python3
"""Independent bounded terminal-hold transactions; synthetic CPU, no physics."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--robot-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);frozen=a.output/'sources';frozen.mkdir()
    root=a.robot_dir.resolve().parent;sys.path.insert(0,str(root))
    from jev_agent_wbc.wbc_protocol import TimedTask,CameraTrajectory,semantic_camera
    from jev_agent_wbc.representation import Pose
    for name in ('terminal_native_hold.py','world_z_bumpless.py','world_z_feedback.py','adaptive_runner.py','wbc_backend.py'):
        shutil.copy2(a.robot_dir/name,frozen/name)
    shutil.copy2(__file__,frozen/Path(__file__).name)
    h=load('independent_actual_hold',frozen/'terminal_native_hold.py')
    z=load('independent_held_window_seed',frozen/'world_z_bumpless.py')
    backend=load('independent_hold_domain',a.robot_dir/'wbc_backend.py')
    domain=backend.PolicyDomain();camera=np.eye(4);camera[:3,3]=[.5,.1,.52]
    pose=semantic_camera(Pose.from_rt(camera[:3,:3],camera[:3,3]))
    old=TimedTask('old-task',0.,'old-obs',CameraTrajectory(0.,(0.,2.),(pose,pose)),.05,.15)
    new=TimedTask('new-task',7.,'new-obs',CameraTrajectory(7.,(0.,1.),(pose,pose)),.05,.15)
    native=np.array([.5,.1,.5,1.,0.,0.,0.],np.float32);candidate=native.astype(float)+[.01,0,.02,0,0,0,0]
    checks=[]
    def check(name,passed,**details):checks.append(dict(check=name,passed=bool(passed),**details))
    def select_args(stamp=4.,seq=200):
        return dict(stamp_s=stamp,measurement_stamp_s=stamp,previous_consumed_native=native.copy(),
            previous_issue_stamp_s=stamp-.02,previous_consumption_seq=seq-1,
            previous_body_input=np.zeros(3),issued_body_input=np.zeros(3),domain=domain,
            collision_evidence=dict(status='pass',stamp_s=stamp),original_support_ok=True,
            world_position_error_m=.01,world_orientation_error_rad=.02)
    hold=h.TerminalNativeHold(old);issued=[];rows=[]
    actual,before=hold.select(candidate,**select_args(3.98,199))
    check('before_fixed_end2_unchanged',not before['selected']and np.array_equal(actual,candidate))
    for seq in range(200,350):
        stamp=seq*.02;actual,report=hold.select(candidate,**select_args(stamp,seq))
        check('prepared_first_tick_not_active',not report['active'])if seq==200 else None
        # Synchronous stand-in for policy feature consumption, with actual f32.
        feature=np.asarray(actual,np.float32);issued.append(feature.copy())
        confirmed=hold.confirm(report,actual_native=feature,actual_body_input=np.zeros(3),stamp_s=stamp,
            consumption_seq=seq,override_reasons=['terminal_native_hold'])
        assert confirmed and report['active']
        if seq>=324:
            end=stamp+.02
            rows.append(dict(stamp_s=end,T_world_camera_optical=camera.tolist(),actual_native_command=feature.astype(float).tolist(),
                applied_policy_velocity=[0.,0.,0.],terminal_native_hold=copy.deepcopy(report),
                task_feedback=dict(task_id=old.task_id,feasible=None,constraints={k:dict(status='pass',stamp_s=end)for k in
                    ('configuration','actuation','self_collision','policy_domain')}),
                fine_collision_evidence=dict(status='pass',stamp_s=stamp),native_domain_recovery=dict(active=False),
                native_camera_parking=dict(active=False),requested_native_domain=dict(status='inside_outer_domain_unverified'),
                actual_native_domain=dict(status='inside_outer_domain_unverified',world_z_feedback=dict(
                    integration_allowed=False,externally_rolled_back=True,external_rollback_reason='terminal_native_hold',
                    native_correction_accepted=False,integral_before_m=-.02,integral_after_m=-.02,sample_seq=seq))))
    check('complete_fixed_3_seconds_actual_bits',len(issued)==150 and all(h.same_native_bits(v,native)for v in issued),
          first_issue_s=4.,last_issue_s=6.98,policy_intervals=150)
    check('full_original_task_unchanged',hold.manifest()['task']==old.to_dict())
    after,report=hold.select(candidate,**select_args(7.,350))
    check('stop_end_does_not_reuse_old_hold',not report['selected']and np.array_equal(after,candidate))
    manifest=hold.manifest()
    context=dict(now_s=7.,observation_stamp_s=7.,reference_optical=camera.copy(),measured_optical=camera.copy(),
        link0=np.eye(4),extrinsic=np.eye(4),previous_applied_native=native.copy(),previous_issue_stamp_s=6.98,
        history_rows=rows,domain=domain,previous_terminal_hold=manifest)
    f=z.BumplessWorldZFeedback(enabled=True);admission=f.admit(backend,new,**context)
    check('only_verified_full_hold_window_can_seed',admission['eligible']and
        admission['prior_calibration_source']=='verified_fixed_terminal_actual_native_hold'and f.integral_m==0.)
    seeded,seed_report=f.feedback_native_command(backend,camera,camera,np.eye(4),np.eye(4),
        task_id=new.task_id,stamp_s=7.,now_s=7.,dt_s=.02,sample_seq=350,fresh=True,integration_allowed=True,domain=domain)
    check('seed_shares_original_3d25mm_cap',seed_report['world_z_feedback']['proposed_position_correction_norm_m']<=.025+1e-12)
    f.rollback_last_update('outer_same_tick_veto',update_id=seed_report['world_z_feedback']['update_id'])
    check('new_task_seed_outer_veto_is_zero',f.integral_m==0. and not f.finish_seed_update(
        seed_report['world_z_feedback']['update_id'],applied_native=seeded))
    for fault in ('missing_manifest','old_parking','recovery','fault_override','copied_old_report','uncommitted',
                  'changed_TaskSHA','changed_capture_clock','changed_native_bit','moving_body','stale_FCL',
                  'domain_failed','unstable_actual_EEz','expired_prior','changed_sequence'):
        ctx=copy.deepcopy(context);row=ctx['history_rows'][3]
        if fault=='missing_manifest':ctx['previous_terminal_hold']=None
        elif fault=='old_parking':row['native_camera_parking']['active']=True
        elif fault=='recovery':row['native_domain_recovery']['active']=True
        elif fault=='fault_override':row['actual_native_domain']['world_z_feedback']['external_rollback_reason']+=',collision_freshness_veto'
        elif fault=='copied_old_report':row['terminal_native_hold']['selected_issue_stamp_s']-=.02
        elif fault=='uncommitted':ctx['previous_terminal_hold']['first_consumption']=None
        elif fault=='changed_TaskSHA':row['terminal_native_hold']['task_sha256']='0'*64
        elif fault=='changed_capture_clock':ctx['previous_terminal_hold']['capture']['stamp_s']+=.02
        elif fault=='changed_native_bit':row['actual_native_command'][0]=float(np.nextafter(np.float32(.5),np.float32(1)))
        elif fault=='moving_body':row['applied_policy_velocity'][0]=.0001
        elif fault=='stale_FCL':row['task_feedback']['constraints']['self_collision']['stamp_s']-=.1
        elif fault=='domain_failed':row['requested_native_domain']['status']='out_of_policy_domain'
        elif fault=='unstable_actual_EEz':row['T_world_camera_optical'][2][3]+=.002
        elif fault=='expired_prior':ctx['previous_issue_stamp_s']=6.9
        elif fault=='changed_sequence':row['terminal_native_hold']['selected_consumption_seq']-=1
        rejected=z.BumplessWorldZFeedback(enabled=True);r=rejected.admit(backend,new,**ctx)
        check('calibration_rejects_'+fault,not r['eligible']and rejected.integral_m==0.)
    for fault in ('not_consumed_next_tick','different_Task','late_confirmation','actual_bits_changed','later_recovery','domain_invalid'):
        hold=h.TerminalNativeHold(old);args=select_args();actual,r=hold.select(candidate,**args)
        stamp=4.;seq=200;reasons=['terminal_native_hold'];value=actual.copy()
        if fault=='not_consumed_next_tick':hold.select(candidate,**select_args(4.02,201))
        elif fault=='different_Task':r['task_sha256']='0'*64
        elif fault=='late_confirmation':stamp=4.02
        elif fault=='actual_bits_changed':value[0]=float(np.nextafter(np.float32(value[0]),np.float32(1)))
        elif fault=='later_recovery':reasons.append('native_domain_recovery')
        else:
            class InvalidDomain:
                def check(self,value):return dict(status='out_of_policy_domain')
            args['domain']=InvalidDomain();args.update(stamp_s=4.02,measurement_stamp_s=4.02,previous_consumption_seq=200)
            hold.select(candidate,**args)
        accepted=hold.confirm(r,actual_native=value,actual_body_input=np.zeros(3),stamp_s=stamp,
            consumption_seq=seq,override_reasons=reasons)
        check('hold_rejects_'+fault,not accepted and not r['active'])
    source=(frozen/'adaptive_runner.py').read_text()
    select=source.index('native,terminal_hold_report=terminal_native_hold.select(')
    domain_guard=source.index('native,recovery_command,recovery_report=domain_recovery.update(',select)
    finalize=source.index('corrected=finalize_native_feedback(',domain_guard)
    f32=source.index('ee=torch.tensor(native,dtype=torch.float32)',finalize)
    policy=source.index('action=policy(torch.cat(histories',f32)
    confirm=source.index('terminal_native_hold.confirm(',policy)
    physics=source.index('sim.step(render=False)',confirm)
    check('actual_runtime_guard_consume_confirm_order',select<domain_guard<finalize<f32<policy<confirm<physics)
    check('original_world_feedback_and_endpoint_stay_live','hold_terminal=True,constraints=constraints'in source and
        'terminal_endpoint=optical_sampler.sample(task_end)'in source)
    (a.output/'synthetic_original_hold_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    report=dict(schema='go2_terminal_hold_seed_independent/v1',passed=all(v['passed']for v in checks),
        checks=len(checks),failed_checks=[v['check']for v in checks if not v['passed']],records=checks,
        physical_execution=False,scope='Synthetic 150 actual-consumption callbacks plus narrowly qualified26-row seed evidence; source-order review. No learned policy inference, physics, contact or tracking certificate.',
        frozen_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in frozen.iterdir()if p.is_file()})
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='records'},indent=2))


if __name__=='__main__':main()
