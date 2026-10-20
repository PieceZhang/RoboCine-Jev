#!/usr/bin/env python3
"""Independent admission/commit boundary checks, with no policy or physics claim."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.robot_dir.resolve().parent
    sys.path.insert(0, str(root))
    from jev_agent_wbc.wbc_protocol import TimedTask, CameraTrajectory, semantic_camera
    from jev_agent_wbc.representation import Pose
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out/'sources'
    frozen.mkdir()
    for name in ('world_z_bumpless.py', 'world_z_feedback.py', 'wbc_backend.py', 'adaptive_runner.py'):
        shutil.copy2(a.robot_dir/name, frozen/name)
    shutil.copy2(__file__, frozen/Path(__file__).name)
    wrapper = load('independent_frozen_seed', frozen/'world_z_bumpless.py')
    backend = load('independent_seed_backend', a.robot_dir/'wbc_backend.py')
    driver = load('independent_seed_driver', frozen/'adaptive_runner.py')
    results = []

    def fixture(z=.5):
        camera = np.eye(4)
        camera[:3, 3] = [.5, .1, .52]
        native = np.array([.5, .1, z, 1., 0., 0., 0.], dtype=np.float32)
        pose = semantic_camera(Pose.from_rt(camera[:3, :3], camera[:3, 3]))
        task = TimedTask('reused-human-id', 5., 'fresh-observation',
                         CameraTrajectory(5., (0., 1.), (pose, pose)), .05, .15)
        rows = []
        for i in range(26):
            stamp = 4.5+i*.02
            rows.append(dict(stamp_s=stamp, actual_native_command=native.astype(float).tolist(),
                T_world_camera_optical=camera.tolist(),
                task_feedback=dict(task_id='completed-task', feasible=None, constraints={
                    k: dict(status='pass', stamp_s=stamp) for k in
                    ('configuration', 'actuation', 'self_collision', 'policy_domain')}),
                fine_collision_evidence=dict(status='pass'),
                native_domain_recovery=dict(active=False), native_camera_parking=dict(active=False),
                actual_native_domain=dict(status='inside_outer_domain_unverified', world_z_feedback=dict(
                    externally_rolled_back=False, native_correction_accepted=True)),
                requested_native_domain=dict(status='inside_outer_domain_unverified')))
        return task, dict(now_s=5., observation_stamp_s=5., reference_optical=camera.copy(),
            measured_optical=camera.copy(), link0=np.eye(4), extrinsic=np.eye(4),
            previous_applied_native=native.copy(), previous_issue_stamp_s=4.98,
            history_rows=rows, domain=backend.PolicyDomain())

    def feedback(f, task, context, stamp=5., seq=250, **overrides):
        args = dict(task_id=task.task_id, stamp_s=stamp, now_s=stamp, dt_s=.02,
                    sample_seq=seq, fresh=True, integration_allowed=True, domain=context['domain'])
        args.update(overrides)
        return f.feedback_native_command(backend, context['reference_optical'], context['measured_optical'],
                                        context['link0'], context['extrinsic'], **args)

    def record(name, passed, **details):
        results.append(dict(check=name, passed=bool(passed), **details))

    # A prepared value must not become historical calibration before actual issue.
    task, context = fixture()
    f = wrapper.BumplessWorldZFeedback(enabled=True)
    admission = f.admit(backend, task, **context)
    native, check = feedback(f, task, context)
    token = check['world_z_feedback']['update_id']
    record('prepared_is_not_committed', admission['transaction_status']=='prepared_waiting_actual_issue')
    actual = np.asarray(native, dtype=np.float32)
    committed = f.finish_seed_update(token, applied_native=actual)
    record('matching_actual_f32_commit', committed and admission['transaction_status']=='committed_to_native_input')
    record('original_task_unchanged', wrapper.canonical_task(task)[1]==admission['task_sha256'])

    # An unconfirmed first tick is not permission to carry a bias or confirm later.
    task, context = fixture()
    f = wrapper.BumplessWorldZFeedback(enabled=True)
    admission = f.admit(backend, task, **context)
    native, check = feedback(f, task, context)
    old_token = check['world_z_feedback']['update_id']
    _, later = feedback(f, task, context, stamp=5.02, seq=251)
    before_late_finish = f.integral_m
    late_finish = f.finish_seed_update(old_token, applied_native=native)
    record('unconfirmed_seed_does_not_propagate', abs(before_late_finish)<1e-12,
           integral_at_next_tick_m=before_late_finish)
    record('previous_tick_seed_confirmation_rejected', not late_finish,
           delayed_confirmation_accepted=late_finish, transaction_status=admission['transaction_status'])

    for reason in ('native_parking_hold', 'native_domain_recovery', 'collision_freshness_veto',
                   'original_camera_support_rejected', 'terminal_rejected_forecast_body_brake'):
        task, context = fixture()
        f = wrapper.BumplessWorldZFeedback(enabled=True)
        admission = f.admit(backend, task, **context)
        native, check = feedback(f, task, context)
        driver.finalize_native_feedback(f, check, context['domain'].check(native), [reason])
        record('outer_rollback_'+reason, f.integral_m==0. and
               admission['transaction_status']=='outer_override_rolled_back_reset_zero')

    for native_z in (.48, .49, .50, .54, .56):
        task, context = fixture(native_z)
        frozen_task = copy.deepcopy(task.to_dict())
        f = wrapper.BumplessWorldZFeedback(enabled=True)
        admission = f.admit(backend, task, **context)
        native, check = feedback(f, task, context)
        cap = check['world_z_feedback']['proposed_position_correction_norm_m']
        record('seed_cap_'+str(native_z), abs(f.integral_m)<=.025+1e-12 and cap<=.025+1e-12
               and task.to_dict()==frozen_task, integral_m=f.integral_m, total_correction_norm_m=cap)
        f.finish_seed_update(check['world_z_feedback']['update_id'], applied_native=np.asarray(native, np.float32))
        context['measured_optical'][:3, 3] += [-.1, .08, -.04]
        _, check2 = feedback(f, task, context, stamp=5.02, seq=251)
        cap2 = check2['world_z_feedback']['proposed_position_correction_norm_m']
        record('whole_xyz_cap_'+str(native_z), cap2<=.025+1e-12 and abs(f.integral_m)<=.025+1e-12,
               total_correction_norm_m=cap2)

    for failure in ('first_task', 'expired_prior', 'stale_admission', 'different_task_sha', 'domain_reject'):
        task, context = fixture()
        f = wrapper.BumplessWorldZFeedback(enabled=True)
        if failure=='first_task': context['previous_applied_native']=None
        elif failure=='expired_prior': context['previous_issue_stamp_s']=4.90
        elif failure=='stale_admission': context['observation_stamp_s']=4.98
        if failure=='different_task_sha':
            old=f.admit(backend, task, **context)
            old_native,old_check=feedback(f,task,context)
            f.finish_seed_update(old_check['world_z_feedback']['update_id'],applied_native=old_native)
            data=task.to_dict();data['camera_trajectory']['samples'][0]['pose']['translation_m'][2]+=.001
            task=TimedTask.from_dict(data)
        admission=f.admit(backend, task, **context)
        if failure=='domain_reject':
            original=context['domain']
            class RejectSeed:
                def check(self, command):
                    result=original.check(command)
                    if command[2]<.51:result['status']='out_of_policy_domain'
                    return result
            context['domain']=RejectSeed()
        native,check=feedback(f,task,context)
        raw=backend.camera_to_mixed(context['reference_optical'],context['link0'],context['extrinsic'])
        record('reset_zero_'+failure, f.integral_m==0. and np.array_equal(native,raw),
               transaction_status=admission['transaction_status'],native_z_m=float(native[2]))

    if hasattr(driver,'consume_policy_with_seed_confirmation'):
        task,context=fixture();f=wrapper.BumplessWorldZFeedback(enabled=True)
        admission=f.admit(backend,task,**context);native,check=feedback(f,task,context)
        actual=driver.finalize_native_feedback(f,check,context['domain'].check(native),[])
        called=[]
        def consume():
            called.append(admission['transaction_status'])
            return 'native-policy-return-value'
        returned=driver.consume_policy_with_seed_confirmation(consume,f,actual,np.asarray(native,np.float32))
        record('policy_callback_sees_only_prepared_seed', called==['prepared_waiting_actual_issue'] and
            returned=='native-policy-return-value' and admission['transaction_status']=='committed_to_native_input')
        task,context=fixture();f=wrapper.BumplessWorldZFeedback(enabled=True)
        admission=f.admit(backend,task,**context);native,check=feedback(f,task,context)
        actual=driver.finalize_native_feedback(f,check,context['domain'].check(native),[])
        propagated=False
        def failing_consume():raise RuntimeError('independent synthetic policy failure')
        try:driver.consume_policy_with_seed_confirmation(failing_consume,f,actual,np.asarray(native,np.float32))
        except RuntimeError as exc:propagated=str(exc)=='independent synthetic policy failure'
        record('policy_exception_rolls_back_seed', propagated and f.integral_m==0. and
            admission['transaction_status']=='outer_override_rolled_back_reset_zero')
    source=(frozen/'adaptive_runner.py').read_text()
    # Check the actual main-loop ingress, then the commit helper's ordering.
    finalize_site=source.find('corrected=finalize_native_feedback(world_z_feedback,feedback_check,corrected,feedback_overrides')
    f32_site=source.find('ee=torch.tensor(native,dtype=torch.float32)',finalize_site)
    policy_site=source.find('action=consume_policy_with_seed_confirmation(',f32_site)
    tree=ast.parse(source)
    consume_node=next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='consume_policy_with_seed_confirmation'),None)
    consumer=ast.get_source_segment(source,consume_node) if consume_node is not None else ''
    record('runtime_commit_after_actual_policy_consumption',finalize_site<f32_site<policy_site and
           0<=consumer.find('result=invoke_policy()')<consumer.find('controller.finish_seed_update(') and
           'world_z_feedback,corrected,ee[0].numpy()' in source[policy_site:policy_site+400],
           finalize_offset=finalize_site,f32_assignment_offset=f32_site,policy_consumption_offset=policy_site,
           scope='Actual loop supplies its real float32 EE feature to post-success commit; helper callback checked independently.')
    report=dict(schema='go2_bumpless_seed_independent/v1', passed=all(r['passed'] for r in results),
        checks=len(results),failed_checks=[r['check'] for r in results if not r['passed']],results=results,
        physical_execution=False,scope='Independent synthetic measured-window transactions and source ingress order. No NN convergence, contact or physical tracking certificate.',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.iterdir() if p.is_file()})
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
