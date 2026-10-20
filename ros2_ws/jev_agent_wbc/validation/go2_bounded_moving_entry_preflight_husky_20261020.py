#!/usr/bin/env python3
"""Current optional predictor on unchanged physical moving snapshots; CPU only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_terminal_01'
OUT=ROOT/'Robot_Go2/results/bounded_moving_entry_cpu_01'
sys.path.insert(0,str(RUN/'sources'))
from jev_agent_wbc.wbc_protocol import TimedTask


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False);frozen=OUT/'sources';frozen.mkdir()
    files=[ROOT/'Robot_Go2'/name for name in ['bounded_body_prediction.py','camera_coordinator.py','adaptive_runner.py']]+[Path(__file__)]
    for path in files:shutil.copy2(path,frozen/path.name)
    backend=load('bounded_moving_backend',RUN/'sources/go2_backend.py');backend.UPSTREAM_MDP=RUN/'sources/go2_upstream'
    coord=load('bounded_moving_coordinator',frozen/'camera_coordinator.py');sampler=load('bounded_moving_sampler',RUN/'sources/optical_sampler.py')
    rows=[json.loads(line)for line in(RUN/'measurements.jsonl').read_text().splitlines()]
    tasks=[TimedTask.from_dict(value)for value in json.loads((RUN/'timed_tasks.json').read_text())]
    samplers=[sampler.OpticalSampler(task)for task in tasks];cases=[case['case']for case in json.loads((RUN/'summary.json').read_text())['cases']]
    extrinsic=np.asarray(json.loads((RUN/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
    c=coord.CameraCoordinator(extrinsic,backend=backend,response_profile='measured_backward_v1',prediction_profile='bounded_ramp_history_v2')
    history=[];checked=[];durations=[];started=time.monotonic()
    for index,row in enumerate(rows):
        stamp=row['stamp_s']-.02
        if row['case']!='settle' and row['phase']=='move':
            task_index=cases.index(row['case']);task=tasks[task_index];sample=samplers[task_index];previous=rows[index-1]
            body=np.asarray(previous['T_world_base']);link=body.copy();link[:3,3]+=body[:3,:3]@np.array([0,0,.07])
            last=np.asarray(previous['applied_policy_velocity']);requested=np.asarray(row['requested_body_velocity']);old=row['camera_support']
            preferred=np.asarray(old.get('preferred_base_command',np.clip(requested,last-np.array([.5,.5,.8])*.02,last+np.array([.5,.5,.8])*.02)))
            future=requested*np.where(requested<0,[1.8181818181818181,1,1],[1,1,1])
            c.input_history=[(t,u.copy())for t,u in history]
            tick=time.perf_counter()
            support=c.command(stamp,sample.sample,link,body,preferred,np.asarray(previous['measured_body_velocity']),
                max_lookahead_s=max(0.,task.trajectory.end_s-stamp),future_policy_velocity=future,
                previous_applied=last,command_dt=.02,native_posture_x_m=.58,
                posture_weight=coord.operating_posture_weight(stamp,task.trajectory.start_s,task.trajectory.end_s))
            durations.append(time.perf_counter()-tick);actual=np.asarray(support['base_command'])
            checked.append(dict(case=row['case'],stamp_s=stamp,old_reason=old['reason'],new_reason=support['reason'],
                old_feasible=old['feasible'],new_feasible=support['feasible'],
                old_margin=old.get('predicted_min_domain_margin'),new_margin=support.get('predicted_min_domain_margin'),
                new_proposal=actual.tolist(),original_previous_applied=last.tolist(),
                original_source_preferred_available='preferred_base_command'in old,
                original_reference_native_max_difference=float(np.max(abs(np.asarray(support['native_command'])-row['requested_native_command']))),
                command_caps_pass=bool(np.all(abs(actual)<=.2+1e-9)),
                command_slew_pass=bool(np.all(abs(actual-last)<=np.array([.5,.5,.8])*.02+1e-9))))
        history.append((stamp,np.asarray(row['applied_policy_velocity'])));history=history[-32:]
    result=dict(schema='go2_bounded_moving_entry_cpu/v1',samples=len(checked),
        scope='Original separate measured moving snapshots and actual prior inputs; new proposals never replace later recorded states/history.38 old instantaneous-domain failures use a bounded fallback preferred because original support does not log it. Not a physical or closed-loop trial.',
        cap_or_slew_violations=sum(not row['command_caps_pass']or not row['command_slew_pass']for row in checked),
        original_reference_native_max_difference=max(row['original_reference_native_max_difference']for row in checked),
        old_false_count=sum(row['old_feasible']is False for row in checked),new_false_count=sum(row['new_feasible']is False for row in checked),
        old_first_diagR=[row for row in checked if row['case']=='diagonal_right'][0],
        control_only_wall_ms=dict(mean=float(np.mean(durations)*1000),p95=float(np.quantile(durations,.95)*1000),max=float(max(durations)*1000)),
        wall_s=time.monotonic()-started,physical_verified=False,
        cpu_gate_pass=all(row['command_caps_pass']and row['command_slew_pass']and row['original_reference_native_max_difference']<2e-6 for row in checked),
        files_sha256={str(path.relative_to(ROOT)):sha(path)for path in files+[RUN/'measurements.jsonl',RUN/'timed_tasks.json']})
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'rows.json').write_text(json.dumps(checked,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
