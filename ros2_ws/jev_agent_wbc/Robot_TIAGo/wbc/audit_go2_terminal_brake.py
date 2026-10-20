#!/usr/bin/env python3
"""Read-only final terminal-brake priority and genuine World-Z rollback."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--robot-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False);frozen=out/'sources';frozen.mkdir()
    for name in ['adaptive_runner.py','world_z_feedback.py','wbc_backend.py']:
        shutil.copy2(a.robot_dir/name,frozen/name)
    shutil.copy2(__file__,frozen/Path(__file__).name)
    text=(frozen/'adaptive_runner.py').read_text();tree=ast.parse(text)
    names=['final_terminal_body_command','finalize_native_feedback']
    subset=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name in names],type_ignores=[])
    namespace={};exec(compile(subset,str(frozen/'adaptive_runner.py'),'exec'),namespace)
    final=namespace[names[0]];rollback=namespace[names[1]]
    class ForbiddenContext:
        def guard_applied(self,command):raise AssertionError('Failed context must not authorize recovery motion')
    rng=np.random.default_rng(9241);tested=0
    for error,latched in [('missing history',False),(None,True),('late deadline',True)]:
        for _ in range(200):
            previous=rng.uniform(-.2,.2,3);recovery=rng.uniform(-.2,.2,3)
            issued,forced=final(recovery,previous,.02,ForbiddenContext(),error,latched)
            expected=np.clip(np.zeros(3),previous-np.array([.5,.5,.8])*.02,previous+np.array([.5,.5,.8])*.02)
            np.testing.assert_array_equal(issued,expected)
            assert forced and np.all(abs(issued)<=.2+1e-15)
            assert np.all(abs(issued-previous)<=np.array([.5,.5,.8])*.02+1e-15)
            tested+=1
    backend=load('terminal_brake_actual_backend',a.robot_dir/'wbc_backend.py')
    feedback=load('terminal_brake_frozen_ZI',frozen/'world_z_feedback.py')
    controller=feedback.WorldZFeedback(enabled=True);domain=backend.PolicyDomain()
    reference=np.eye(4);reference[:3,3]=[.5,0.,.5];measured=reference.copy();measured[2,3]-=.01
    original=reference.copy()
    for index in range(2):
        _,check=controller.feedback_native_command(backend,reference,measured,np.eye(4),np.eye(4),
            task_id='terminal',stamp_s=1.+index*.02,now_s=1.+index*.02,dt_s=.02,sample_seq=index,
            fresh=True,integration_allowed=True,domain=domain)
    before=controller.integral_m;old=check['world_z_feedback']['integral_before_m'];assert before>old
    actual=domain.check(backend.camera_to_mixed(reference,np.eye(4),np.eye(4)))
    rolled=rollback(controller,check,actual,['terminal_rejected_forecast_body_brake'])
    assert controller.integral_m==old and rolled['world_z_feedback']['externally_rolled_back']
    np.testing.assert_array_equal(reference,original)
    recovery=text.index('native,recovery_command,recovery_report=domain_recovery.update(')
    fcl=text.index("if a.collision and collision_evidence['status']!='pass':",recovery)
    final_index=text.index('issued,forced_brake=final_terminal_body_command(policy_command',fcl)
    rollback_index=text.index('corrected=finalize_native_feedback(world_z_feedback,feedback_check,corrected,feedback_overrides)',final_index)
    report=dict(schema='go2_final_terminal_brake_independent/v1',physical_execution=False,passed=True,
        cases=tested,actual_call_order=['native_domain_recovery','FCL_veto','final_terminal_body_command','World_Z_transaction_rollback'],
        source_order_verified=bool(recovery<fcl<final_index<rollback_index),
        native7_legal_domain_preserved=actual['status'],integral_before_m=before,
        integral_expected_prior_committed_m=old,integral_after_m=controller.integral_m,
        original_reference_unchanged=True,scope='Final unknown/failed terminal forecast only; no new model/NN/physical certification.',
        evidence_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in frozen.iterdir()if path.is_file()})
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
