#!/usr/bin/env python3
"""Exact old false replay and bounded all-axis input-plan workspace comparison."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_terminal_01'
OUT=ROOT/'Robot_Go2/results/bounded_ramp_false_cpu_01'
sys.path.insert(0,str(RUN/'sources'))
from jev_agent_wbc.wbc_protocol import TimedTask


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False);frozen=OUT/'sources';frozen.mkdir()
    paths=[ROOT/'Robot_Go2/bounded_body_prediction.py',ROOT/'test/test_wbc_go2_bounded_body_prediction.py',Path(__file__)]
    for path in paths:shutil.copy2(path,frozen/path.name)
    model=load('bounded_allaxis_input_model',frozen/'bounded_body_prediction.py')
    backend=load('bounded_original_backend',RUN/'sources/go2_backend.py');backend.UPSTREAM_MDP=RUN/'sources/go2_upstream'
    coord=load('bounded_original_coordinator',RUN/'sources/camera_coordinator.py')
    sampler=load('bounded_original_sampler',RUN/'sources/optical_sampler.py')
    rows=[json.loads(line)for line in(RUN/'measurements.jsonl').read_text().splitlines()]
    summary=json.loads((RUN/'summary.json').read_text());tasks=[TimedTask.from_dict(value)for value in json.loads((RUN/'timed_tasks.json').read_text())]
    cases=[value['case']for value in summary['cases']];samples=[sampler.OpticalSampler(task)for task in tasks]
    extrinsic=np.asarray(json.loads((RUN/'sensor_mounts.json').read_text())['camera']['T_parent_optical']);inverse=np.linalg.inv(extrinsic)
    c=coord.CameraCoordinator(extrinsic,backend=backend,response_profile='measured_backward_v1',prediction_profile='persistent_bias_12_v1')
    results=[];history=[]
    for index,row in enumerate(rows):
        stamp=row['stamp_s']-.02
        if 'camera_support'in row and row['camera_support']['feasible']is False:
            previous=rows[index-1];case=cases.index(row['case']);task=tasks[case];sample=samples[case];old=row['camera_support']
            body=np.asarray(previous['T_world_base']);link=body.copy();link[:3,3]+=body[:3,:3]@np.array([0,0,.07])
            record=dict(case=row['case'],stamp_s=row['stamp_s'],issue_stamp_s=stamp,old_reason=old['reason'])
            if old['reason']=='out_of_policy_domain':
                record.update(current_domain_failure=old.get('current_domain'),
                    original_current_domain_cannot_be_changed_instantly=True,
                    new_plan_claim='none: already outside original instantaneous native domain on this frozen state')
            else:
                preferred=np.asarray(old['preferred_base_command']);measured=np.asarray(previous['measured_body_velocity'])
                previous_applied=np.asarray(previous['applied_policy_velocity']);horizon=min(1.2,task.trajectory.end_s-stamp)
                intervals=c.prediction_intervals(horizon);targets=[sample.sample(stamp+t)@inverse for t in intervals]
                future=np.asarray(row['requested_body_velocity'])*np.where(np.asarray(row['requested_body_velocity'])<0,[1.8181818181818181,1.,1.],[1.,1.,1.])
                c.input_history=[(t,u.copy())for t,u in history]
                weight=old['native_posture_preference']['weight']
                replay=c.command(stamp,sample.sample,link,body,preferred,measured,max_lookahead_s=horizon,
                    future_policy_velocity=future,previous_applied=previous_applied,command_dt=.02,
                    native_posture_x_m=.58,posture_weight=weight)
                record.update(old_replay_reason=replay['reason'],old_margin=old['predicted_min_domain_margin'],old_replay_margin=replay['predicted_min_domain_margin'],
                    old_replay_applied_exact=bool(np.allclose(replay['base_command'],row['applied_policy_velocity'],atol=1e-10,rtol=0)),
                    old_replay_native_error=float(np.max(abs(np.asarray(replay['native_command'])-row['requested_native_command']))),
                    measured_velocity=measured.tolist(),previous_applied=previous_applied.tolist(),preferred_first=preferred.tolist(),future_target=future.tolist())
                low=np.maximum(-model.CAPS,previous_applied-model.SLEW*.02);high=np.minimum(model.CAPS,previous_applied+model.SLEW*.02)
                candidates=[preferred,np.clip(np.zeros(3),low,high),*map(np.asarray,itertools.product(*zip(low,high)))]
                record['new_models']={}
                for mode in ['linear_residual','measured_hold']:
                    forecast=model.BoundedBodyPrediction(backend,stamp_s=stamp,body=body,link=link,
                        measured_velocity=measured,previous_applied=previous_applied,input_history=history,
                        prediction_intervals=intervals,pending_mode=mode)
                    trial=[]
                    for first in candidates:
                        target=c.future_plan(first,preferred,future,weight);values=[]
                        for elapsed,world_target in zip(intervals,targets):
                            predicted=forecast.predict_link(first,target,float(elapsed))
                            rotation=predicted[:3,:3].T
                            position=np.r_[(rotation@(world_target[:3,3]-predicted[:3,3]))[:2],world_target[2,3]]
                            angles=Rotation.from_matrix(rotation@world_target[:3,:3]).as_euler('xyz');reserve=min(1.,elapsed/.8)
                            values.extend(position-c.xyz[:,0]-c.reserve*reserve);values.extend(c.xyz[:,1]-position-c.reserve*reserve)
                            values.extend(angles-c.euler[0]-.015*reserve);values.extend(c.euler[1]-angles-.015*reserve)
                        data=forecast.trajectory(first,target);issued=np.vstack([previous_applied,data['issued']])
                        trial.append(dict(first=first.tolist(),future_target=target.tolist(),margin=float(min(values)),
                            cap_pass=bool(np.all(abs(issued)<=model.CAPS+1e-9)),slew_pass=bool(np.all(abs(np.diff(issued,axis=0))<=model.SLEW*.02+1e-9)),
                            first_exact=bool(np.array_equal(data['issued'][0],first)),
                            initial_transform_exact=bool(np.array_equal(forecast.predict_link(first,target,0),link)),
                            predicted_body_end_position=data['roots'][-1,:3,3].tolist()))
                    record['new_models'][mode]=dict(preferred=trial[0],best=max(trial,key=lambda value:value['margin']),
                        positive_candidate_count=sum(value['margin']>=-1e-7 for value in trial),candidate_count=len(trial),all_trials_input_invariants=all(value['cap_pass']and value['slew_pass']and value['first_exact']and value['initial_transform_exact']for value in trial))
            results.append(record)
        history.append((stamp,np.asarray(row['applied_policy_velocity'])));history=history[-32:]
    solved=[value for value in results if 'new_models'in value]
    report=dict(schema='go2_bounded_plan_false_cpu/v1',false_samples=len(results),solver_false_samples=len(solved),current_outer_domain_samples=len(results)-len(solved),
        old_replay_reason_mismatch_count=sum(value['old_reason']!=value['old_replay_reason']for value in solved),
        old_replay_margin_max_abs_error=max(abs(value['old_margin']-value['old_replay_margin'])for value in solved),
        old_replay_applied_mismatch_count=sum(not value['old_replay_applied_exact']for value in solved),
        old_replay_native_max_error=max(value['old_replay_native_error']for value in solved),
        new_models={mode:dict(preferred_plan_feasible_count=sum(value['new_models'][mode]['preferred']['margin']>=-1e-7 for value in solved),
            any_plan_feasible_count=sum(value['new_models'][mode]['best']['margin']>=-1e-7 for value in solved),
            all_input_invariants=all(value['new_models'][mode]['all_trials_input_invariants']for value in solved))for mode in ['linear_residual','measured_hold']},
        first_source_false=results[0],scope='Fixed original states/history/Task; original nodes and35/35/15mm/15mrad reserves. Positive result certifies only this explicit operating-model plan; not learned18/contact/physical feasibility. Later frozen current-domain failures remain failures.',
        physical_verified=False,runtime_modified=False,
        files_sha256={str(path.relative_to(ROOT)):sha(path)for path in paths+[RUN/'measurements.jsonl',RUN/'timed_tasks.json',RUN/'sources/camera_coordinator.py',RUN/'sources/go2_backend.py']})
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'rows.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
