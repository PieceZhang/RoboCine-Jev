#!/usr/bin/env python3
"""Tiny recovery counterexamples and honest unknown-transport recovery positives."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);sources=a.output/'sources';sources.mkdir()
    for source in (a.root/'jev_agent_wbc/wbc_recovery_audit.py',a.root/'test/test_wbc_recovery_audit.py',Path(__file__)):
        shutil.copy2(source,sources/source.name)
    auditor=load('frozen_recovery_semantics',sources/'wbc_recovery_audit.py')
    # The fixture imports constants by the normal package name; bind that name
    # to the exact frozen auditor, not a possibly changing workspace module.
    sys.path.insert(0,str(a.root.resolve()))
    sys.modules['jev_agent_wbc.wbc_recovery_audit']=auditor
    fixtures=load('frozen_original_recovery_fixture',sources/'test_wbc_recovery_audit.py')
    original,tasks=fixtures.fixture();baseline=auditor.audit_recovery_rows(original,tasks)
    assert baseline['passed']
    results=[]
    def test(name,rows):
        result=auditor.audit_recovery_rows(rows,tasks)
        results.append(dict(name=name,rejected=not result['passed'],result=result,synthetic_rows=rows))
    bad=copy.deepcopy(original)
    bad[1]['task_feedback'].update(reason='command_stale',feasible=None)
    test('actual_solver_failure_masked_in_both_public_fields',bad)
    bad=copy.deepcopy(original);ghost=copy.deepcopy(bad[1]);ghost.update(stamp_s=1.04,task_feedback=None,
        pink_upper_velocity={auditor.UPPER[0]:.1},applied_policy_velocity=[0.,0.,.2])
    ghost['applied_joint_position_target'][0]=.2
    bad[-1]['stamp_s']=1.06;bad[-1]['pink_runtime_recovery'].update(acquisition_stamp_s=1.02,command_sent_wall_s=100.06)
    bad.insert(2,ghost)
    test('active_missing_feedback_hides_nonfresh_motion',bad)
    bad=copy.deepcopy(original);bad[1]['pink_runtime_recovery']['dropped_seq']=1
    bad[-1]['pink_runtime_recovery']['seq']=2
    test('resumption_sequence_older_than_prefault_execution',bad)
    stale=copy.deepcopy(original)
    stale[1]['pink_runtime_recovery']['failure_kind']='stale_or_mismatched_transport'
    stale[1]['task_feedback'].update(reason='command_stale',feasible=None)
    honest_stale=auditor.audit_recovery_rows(stale,tasks)
    positive=[dict(name='bounded_stale_transport_remains_unknown',passed=honest_stale['passed'],
        result=honest_stale,synthetic_rows=stale)]
    bad=copy.deepcopy(stale);bad[1]['applied_policy_velocity'][2]=.04
    test('honest_stale_transport_still_requires_actual_slew_brake',bad)
    bad=copy.deepcopy(stale);bad[-1]['pink_runtime_recovery']['acquisition_stamp_s']=.98
    test('honest_stale_transport_cannot_resume_old_acquisition',bad)
    bad=copy.deepcopy(original);bad[1].pop('task_feedback')
    test('raw_fault_with_missing_public_feedback',bad)
    report=dict(schema='g1_recovery_counterexamples_independent/v1',baseline=baseline,cases=results,
        positive_cases=positive,all_positive_cases_passed=all(r['passed']for r in positive),
        all_counterexamples_rejected=all(r['rejected']for r in results),physical_execution=False,
        scope='Three/four synthetic samples only. No actual failure/physical acceptance is reclassified.',
        frozen_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sources.iterdir()if p.is_file()})
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(baseline_passed=baseline['passed'],cases=[dict(name=r['name'],rejected=r['rejected'])for r in results],frozen_sha256=report['frozen_sha256']),indent=2))


if __name__=='__main__':main()
