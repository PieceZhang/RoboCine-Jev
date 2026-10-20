#!/usr/bin/env python3
"""Independent mutation checks of the root input auditor, no physical replay."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--auditor',type=Path,required=True)
    p.add_argument('--recorded',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sources=a.output/'sources';sources.mkdir()
    shutil.copy2(a.auditor,sources/a.auditor.name);shutil.copy2(__file__,sources/Path(__file__).name)
    spec=importlib.util.spec_from_file_location('frozen_root_active_auditor',sources/a.auditor.name)
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    rows=[json.loads(line)for line in(a.recorded/'measurements.jsonl').open()]
    tasks=json.loads((a.recorded/'timed_tasks.json').read_text());summary=json.loads((a.recorded/'summary.json').read_text())
    positive=auditor.audit_rows(rows,tasks,summary['run_complete']);assert positive['passed']
    active=next(i for i,r in enumerate(rows)if(r.get('terminal_allocation')or{}).get('phase')=='active_lateral_brake')
    tests=[]
    def record(name,mutated):
        result=auditor.audit_rows(mutated,tasks,True)
        tests.append(dict(name=name,rejected=not result['passed'],auditor_result=result))
    altered=copy.deepcopy(rows);altered[active]['camera_support']['feasible']=False
    record('undeclared_actual_forecast_veto',altered)
    altered=copy.deepcopy(rows);altered[active]['fine_collision_evidence']['status']='unknown'
    record('undeclared_actual_FCL_veto',altered)
    altered=copy.deepcopy(rows);altered[active]['native_domain_recovery']['active']=True
    record('undeclared_actual_recovery_veto',altered)
    # Keep250 stop rows/finalstamp but substitute the20ms before the true stop.
    altered=copy.deepcopy(rows);first=next(i for i,r in enumerate(altered)if r.get('phase')=='stop'and
        r.get('task_feedback',{}).get('task_id')==tasks[0]['task_id'])
    replacement=altered[first-1];replacement['phase']='stop';replacement['requested_body_velocity']=[0.,0.,0.]
    replacement['terminal_allocation']=copy.deepcopy(altered[first]['terminal_allocation'])
    replacement['terminal_allocation']['stamp_s']=replacement['stamp_s']-.02
    replacement['terminal_allocation']['final_body_input']=copy.deepcopy(replacement['applied_policy_velocity'])
    replacement['camera_support']=copy.deepcopy(altered[first]['camera_support'])
    altered[first]['phase']='move'
    record('noncontiguous_stop_wrong_start_same250count',altered)
    # Last row prevents a synthetic changed output from requiring a fabricated
    # later physical measurement. This checks legal veto, not physical safety.
    veto=copy.deepcopy(rows);row=veto[-1];previous=veto[-2]
    brake=np.clip(np.zeros(3),np.asarray(previous['applied_policy_velocity'])-np.array([.5,.5,.8])*.02,
                  np.asarray(previous['applied_policy_velocity'])+np.array([.5,.5,.8])*.02)
    row['applied_policy_velocity']=brake.tolist();row['native_standing_mode']=bool(np.all(brake==0))
    report=row['terminal_allocation'];report.update(veto_reasons=['original_forecast_rejected'],
        final_body_input=brake.tolist(),final_feedback_only=False)
    row['camera_support'].update(feasible=False,camera_supported=False,base_velocity_feasible=False)
    row['task_feedback'].update(feasible=False,reason='tracking_limit')
    zi=row['actual_native_domain']['world_z_feedback']
    zi.update(integral_after_m=zi['integral_before_m'],integral_updated=False,
              externally_rolled_back=True,native_correction_accepted=False,
              external_rollback_reason='active_brake_safety_veto,original_camera_support_rejected')
    legal_veto=auditor.audit_rows(veto,tasks,True)
    broken=copy.deepcopy(veto);broken[-1]['actual_native_domain']['world_z_feedback'].update(
        externally_rolled_back=False,native_correction_accepted=True)
    record('missing_fresh_clock_outer_rollback',broken)
    invalid_clock=copy.deepcopy(veto);invalid_clock[-1]['actual_native_domain']['world_z_feedback'].update(
        measurement_fresh_and_clock_valid=False,externally_rolled_back=False,native_correction_accepted=False)
    legal_freeze=auditor.audit_rows(invalid_clock,tasks,True)
    report=dict(schema='go2_active_stand_auditor_independent/v1',actual_run_positive=positive,
        negative_cases=tests,all_negative_cases_rejected=all(t['rejected']for t in tests),
        correct_synthetic_veto_rollback_positive=legal_veto,invalid_clock_freeze_positive=legal_freeze,
        review_passed=all(t['rejected']for t in tests)and legal_veto['passed']and legal_freeze['passed'],
        runtime_bug_claimed=False,physical_execution=False,
        scope='Existing physical raw read only. Mutations are synthetic auditor falsification, not actual run faults.',
        input_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()for p in
            [a.recorded/'measurements.jsonl',a.recorded/'timed_tasks.json',a.recorded/'summary.json']},
        frozen_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sources.iterdir()if p.is_file()})
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
