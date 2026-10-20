#!/usr/bin/env python3
"""Small synthetic falsification of the independent XY stand-input auditor."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np


def fixture():
    task=dict(task_id='independent_xy',camera_trajectory=dict(start_s=0.,samples=[dict(t_s=0.),dict(t_s=1.)]))
    rows=[dict(stamp_s=1.,phase='move',measured_body_velocity=[.08,-.04,.07],
               applied_policy_velocity=[0.,0.,0.])]
    for i in range(250):
        now=1.+i*.02;late=now>=3.-1e-8;prior=np.asarray(rows[-1]['applied_policy_velocity'])
        measured=np.asarray(rows[-1]['measured_body_velocity']);raw=np.array([-.5*measured[0],-.5*measured[1],0.])
        target=np.clip(raw,[-.03,-.03,0.],[.03,.03,0.]);first=np.clip(target,prior-np.array([.5,.5,.8])*.02,prior+np.array([.5,.5,.8])*.02)
        issued=first if late else np.zeros(3)
        horizon=min(1.2,6.-now);nodes=np.unique(np.round(np.r_[np.linspace(0.,min(horizon,.8),5),np.linspace(min(horizon,.8),horizon,3)],8))
        rows.append(dict(stamp_s=now+.02,phase='stop',requested_body_velocity=[0.,0.,0.],
            measured_body_velocity=[.08,-.04,.07],applied_policy_velocity=issued.tolist(),
            native_standing_mode=bool(np.all(issued==0)),task_feedback=dict(task_id=task['task_id'],feasible=None,reason='accepted'),
            terminal_allocation=dict(profile='vxy_p05_v1',phase='active_planar_brake'if late else'allocating',
                original_world_task_modified=False,all18_learned_actions_unchanged=True,
                reference_mode='explicit_original_endpoint_hold',stamp_s=now,reference_stamp_s=1.,
                final_body_input=issued.tolist(),zero_input_deadline_s=3.,measurement_stamp_s=now,
                measured_body_velocity=measured.tolist(),previous_actual_input=prior.tolist(),feedback_gain=.5,
                assist_vy_cap_mps=.03,assist_vx_cap_mps=.03,assist_caps_mps=[.03,.03,0.],feedback_axes=['body_vx','body_vy'],
                requested_stop_velocity=[0.,0.,0.],raw_feedback_input=raw.tolist(),bounded_feedback_target=target.tolist(),
                cap_slew_admissible_first=first.tolist(),future_input_assumption='Frozen current measured feedback target; next tick remeasured',
                prediction_model=dict(profile='bounded_ramp_history_v2'),veto_reasons=[],final_feedback_only=True),
            camera_support=dict(planned_slew_brake_model=True,prediction_horizon_s=horizon,
                prediction_intervals_s=nodes.tolist(),locked_input=True,feasible=True),
            fine_collision_evidence=dict(status='pass',stamp_s=now),native_domain_recovery=dict(active=False),
            actual_native_domain=dict(world_z_feedback=dict(integral_before_m=.01,integral_after_m=.01,
                integral_updated=False,measurement_fresh_and_clock_valid=True,externally_rolled_back=False,native_correction_accepted=True))))
    return rows,[task]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--auditor',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();begin=time.perf_counter()
    a.output.mkdir(parents=True,exist_ok=False);sources=a.output/'sources';sources.mkdir()
    for source in [a.auditor,Path(__file__)]:shutil.copy2(source,sources/source.name)
    spec=importlib.util.spec_from_file_location('frozen_planar_input_auditor',sources/a.auditor.name)
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    rows,tasks=fixture();positive=auditor.audit_rows(rows,tasks,True);assert positive['passed']
    tests=[];active=101
    def check(name,change):
        bad=copy.deepcopy(rows);change(bad);result=auditor.audit_rows(bad,tasks,True)
        tests.append(dict(name=name,rejected=not result['passed'],result=result))
    check('measurement_clock_is_not_issue_clock',lambda r:r[active]['terminal_allocation'].update(measurement_stamp_s=r[active]['stamp_s']))
    check('wrong_previous_actual_input',lambda r:r[active]['terminal_allocation'].update(previous_actual_input=[.01,0.,0.]))
    check('removed_reserved_forecast_node',lambda r:r[active]['camera_support']['prediction_intervals_s'].pop())
    check('undeclared_forecast_rejection',lambda r:r[active]['camera_support'].update(feasible=False))
    check('stale_FCL_even_with_pass_status',lambda r:r[active]['fine_collision_evidence'].update(stamp_s=r[active]['stamp_s']-.10))
    check('undeclared_native_recovery',lambda r:r[active]['native_domain_recovery'].update(active=True))
    check('wrong_X_feedback_sign',lambda r:r[active]['terminal_allocation']['raw_feedback_input'].__setitem__(0,.04))
    check('old_vy_profile_does_not_prove_XY',lambda r:r[active]['terminal_allocation'].update(profile='vy_p05_v1',phase='active_lateral_brake',feedback_axes=['body_vy']))
    check('X_cap_cannot_be_changed_to_40mmps',lambda r:r[active]['terminal_allocation'].update(assist_vx_cap_mps=.04,assist_caps_mps=[.04,.03,0.]))
    # Preserve250 stamps/count while moving the first stop before the endpoint.
    def wrong_stop(r):
        stamp=r[0]['stamp_s'];r[0]=copy.deepcopy(r[1]);r[0]['stamp_s']=stamp
        r[0]['terminal_allocation']['stamp_s']=stamp-.02;r[1]['phase']='move'
    check('250_rows_do_not_excuse_wrong_stop_start',wrong_stop)
    veto=copy.deepcopy(rows);row=veto[-1];prior=np.asarray(veto[-2]['applied_policy_velocity'])
    safe=np.clip(np.zeros(3),prior-np.array([.5,.5,.8])*.02,prior+np.array([.5,.5,.8])*.02)
    row['applied_policy_velocity']=safe.tolist();row['native_standing_mode']=bool(np.all(safe==0))
    row['terminal_allocation'].update(veto_reasons=['original_forecast_rejected'],final_body_input=safe.tolist(),final_feedback_only=False)
    row['camera_support']['feasible']=False;row['task_feedback'].update(feasible=False,reason='tracking_limit')
    row['actual_native_domain']['world_z_feedback'].update(externally_rolled_back=True,native_correction_accepted=False)
    legal_veto=auditor.audit_rows(veto,tasks,True);assert legal_veto['passed']
    missing=copy.deepcopy(veto);missing[-1]['actual_native_domain']['world_z_feedback']['externally_rolled_back']=False
    result=auditor.audit_rows(missing,tasks,True)
    tests.append(dict(name='fresh_veto_missing_ZI_rollback',rejected=not result['passed'],result=result))
    report=dict(schema='go2_planar_stand_input_auditor_independent/v1',synthetic_positive=positive,
        legal_synthetic_veto_rollback_positive=legal_veto,negative_cases=tests,
        passed=all(t['rejected']for t in tests),physical_execution=False,
        physical_acceptance_reclassified=False,scope='Synthetic input audit only; no plant prediction or physical acceptance.',
        frozen_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sources.iterdir()if p.is_file()},
        audit_wall_seconds=time.perf_counter()-begin)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
