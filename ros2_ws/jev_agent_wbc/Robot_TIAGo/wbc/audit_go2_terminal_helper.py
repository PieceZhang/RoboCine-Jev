#!/usr/bin/env python3
"""Independent CPU invariants for the private Go2 terminal input model."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.package.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);sources=out/'sources';sources.mkdir()
    for src in (root/'Robot_Go2/terminal_allocator.py',root/'Robot_Go2/wbc_backend.py',Path(__file__).resolve()):shutil.copy2(src,sources/src.name)
    helper=load('independent_terminal_helper',sources/'terminal_allocator.py');backend=load('independent_terminal_backend',sources/'wbc_backend.py')
    dt=.02;end=5.003;checks=[]
    body=np.eye(4);body[:3,:3]=Rotation.from_euler('xyz',[.1,-.08,.5]).as_matrix();body[:3,3]=[.1,-.2,.3]
    offset=np.eye(4);offset[:3,3]=[.02,0,.07];link=body@offset
    endpoint=np.eye(4);endpoint[:3,:3]=Rotation.from_euler('xyz',[.05,.1,.3]).as_matrix();endpoint[:3,3]=[.55,.25,.5]
    def context(stamp,previous=None,history=None):
        previous=np.zeros(3) if previous is None else np.asarray(previous,float)
        return helper.TerminalContext(backend,stamp_s=stamp,task_end_s=end,stop_seconds=5.,endpoint_optical=endpoint,
            extrinsic=np.eye(4),body=body,link=link,measured_velocity=np.array([.08,.01,.01]),previous_applied=previous,
            input_history=[(stamp-k*dt,previous.copy()) for k in range(32,0,-1)] if history is None else history)

    def expect_reject(name,call):
        try:call()
        except ValueError as error:checks.append(dict(case=name,passed=True,error=repr(error)));return error
        raise AssertionError(f'{name} was accepted')

    late=expect_reject('non_grid_absolute_deadline_viability',lambda:context(7.,[.0115,0,0]))
    assert isinstance(late,helper.TerminalDeadlineInfeasible)
    assert np.all(np.abs(late.safe_brake_command-[.0115,0,0])<=helper.SLEW*dt+1e-12)
    expect_reject('aligned_unreachable_deadline',lambda:context(end+1.98,[.2,0,0]))
    stamp=end+.1
    history=[(stamp-k*dt,[0,0,0]) for k in range(32,0,-1)]
    expect_reject('stale_latest_actual_input',lambda:context(stamp,history=[(t-dt,u) for t,u in history]))
    expect_reject('future_actual_input',lambda:context(stamp,history=[(t+dt,u) for t,u in history]))
    expect_reject('last_actual_input_disagrees_with_previous',lambda:context(stamp,[.01,0,0],history=history))
    expect_reject('insufficient_delay_history',lambda:context(stamp,history=history[-5:]))
    c=context(stamp);c.solve()
    expect_reject('predicted_first_would_violate_original_slew',lambda:c.predict_link([.2,0,0],.2))
    positive=context(stamp,[.1,0,0]);positive.solve()
    minus=np.array([-1e-9,0,0]);plus=-minus
    left=positive._trajectory(positive._first(minus),minus)['xyz'][-1]
    right=positive._trajectory(positive._first(plus),plus)['xyz'][-1]
    jump=float(np.linalg.norm(left-right));assert jump<1e-7
    checks.append(dict(case='zero_crossing_continuity_same_state_and_history',passed=True,predicted_xy_jump_m=jump))

    # Test the literal zero deadline between control ticks, without moving the
    # original Task.end onto the control grid. These are command fixtures, not
    # a robot/dynamics simulation.
    previous=np.array([.2,-.2,.2]);first_stamp=5.02
    history=[(first_stamp-k*dt,previous.copy()) for k in range(32,0,-1)]
    cap_peak=np.zeros(3);slew_peak=np.zeros(3);near_deadline=[]
    for step in range(250):
        stamp=round(first_stamp+step*dt,8)
        c=context(stamp,previous,history)
        desired=np.array([.2,-.2,.2]) if step%2 else np.array([-.2,.2,-.2])
        command=c.guard_applied(desired)
        assert np.all(abs(command)<=helper.CAPS+1e-12)
        assert np.all(abs(command-previous)<=helper.SLEW*dt+1e-9)
        cap_peak=np.maximum(cap_peak,abs(command));slew_peak=np.maximum(slew_peak,abs(command-previous)/dt)
        if stamp<=end+2.<stamp+dt or stamp>=end+2.:
            assert np.all(command==0), (stamp,command)
        if abs(stamp-(end+2.))<.03:near_deadline.append(dict(stamp_s=stamp,command=command.tolist()))
        # Model first input must be the exact command authorized by the guard.
        c.desired=desired;c.proposal=command.copy()
        data=c._trajectory(command,desired if c.phase=='allocate' else np.zeros(3))
        np.testing.assert_allclose(data['issued'][0],command,atol=1e-12,rtol=0)
        np.testing.assert_allclose(c.predict_link(command,0.),c.link,atol=1e-12,rtol=0)
        previous=command;history.append((stamp,command.copy()));history=history[-32:]
    checks.append(dict(case='250_control_ticks_original_non_grid_end_zero_deadline',passed=True,task_end_s=end,
        zero_deadline_s=end+2.,near_deadline=near_deadline,maximum_absolute_command=cap_peak.tolist(),
        maximum_slew=slew_peak.tolist(),original_endpoint_unchanged=bool(np.array_equal(c.endpoint,endpoint))))
    paths=[p for p in sources.iterdir() if p.is_file()]
    report=dict(schema='independent_go2_terminal_helper/v1',passed=True,physical_execution=False,
        scope='Static CPU input/model invariants only. No learned18, physical motion, strict-stop or environment claim. Existing world-Z feedback is outside this helper and must retain its original final domain check/rollback.',
        checks=checks,evidence_sha256={str(path.relative_to(out)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        source_unchanged_during_test=hashlib.sha256((root/'Robot_Go2/terminal_allocator.py').read_bytes()).hexdigest()==hashlib.sha256((sources/'terminal_allocator.py').read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
