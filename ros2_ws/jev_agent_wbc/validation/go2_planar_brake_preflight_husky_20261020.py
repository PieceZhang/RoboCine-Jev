#!/usr/bin/env python3
"""Light measured-snapshot check, without replaying policy or contact physics."""
import hashlib,importlib.util,json,shutil,time,sys
from pathlib import Path
import numpy as np
from scipy.signal import detrend

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'Robot_Go2/results/adaptive_suite_output_continuity_01'
OUT=ROOT/'Robot_Go2/results/planar_stand_brake_cpu_02'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def corr(a,b):
    a=detrend(a);b=detrend(b)
    return float(np.corrcoef(a,b)[0,1])if np.std(a)*np.std(b)>1e-12 else None

def main():
    OUT.mkdir(exist_ok=False);f=OUT/'sources'
    shutil.copytree(ROOT/'Robot_Go2/results/source_pd_full_preparation_01/sources',f)
    paths=[ROOT/'Robot_Go2'/n for n in ['lateral_stand_brake.py','adaptive_runner.py']]+[ROOT/'test/test_wbc_go2_planar_stand_brake.py',Path(__file__)]
    for p in paths:shutil.copy2(p,f/p.name)
    backend=load('xy_snapshot_backend',f/'go2_backend.py');backend.UPSTREAM_MDP=f/'go2_upstream'
    coord=load('xy_snapshot_coordinator',f/'camera_coordinator.py')
    brake=load('xy_snapshot_brake',f/'lateral_stand_brake.py')
    sampler=load('xy_snapshot_sampler',f/'optical_sampler.py')
    from jev_agent_wbc.wbc_protocol import TimedTask
    tasks={t.task_id:t for t in map(TimedTask.from_dict,json.loads((RUN/'timed_tasks.json').read_text()))}
    ext=np.asarray(json.loads((RUN/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
    rows=[json.loads(x)for x in(RUN/'measurements.jsonl').read_text().splitlines()]
    reports=[];walls=[];correlations=[]
    for name in ['yaw_left','camera_shot']:
        task=next(t for t in tasks.values()if name in t.task_id)
        endpoint=sampler.OpticalSampler(task).sample(task.trajectory.end_s)
        co=coord.CameraCoordinator(ext,backend=backend,response_profile='measured_backward_v1',prediction_profile='bounded_ramp_history_v2')
        selected=[i for i,row in enumerate(rows)if row.get('task_feedback',{}).get('task_id')==task.task_id and row['phase']=='stop'and row['stamp_s']-.02>=task.trajectory.end_s+2-1e-9]
        part=[rows[i]for i in selected];names=part[0]['joint_names'];ids=[names.index(n)for n in ['FR_calf_joint','FL_calf_joint']]
        common_qd=np.asarray([r['measured_joint_velocity']for r in part])[:,ids].mean(1)
        vx=np.asarray([r['measured_body_velocity'][0]for r in part])
        gyro_y=np.asarray([(np.asarray(r['T_world_base'])[:3,:3].T@r['base_angular_velocity_world'])[1]for r in part])
        lagged=[]
        for k in range(-10,11):
            if k>0:vs= vx[:-k];gs=gyro_y[:-k];qs=common_qd[k:]
            elif k<0:vs=vx[-k:];gs=gyro_y[-k:];qs=common_qd[:k]
            else:vs=vx;gs=gyro_y;qs=common_qd
            lagged.append(dict(signal_leads_common_qd_s=k*.02,vx_corr=corr(vs,qs),pitch_gyro_corr=corr(gs,qs)))
        correlations.append(dict(case=name,samples=len(part),same_tick_common_qd_vs_vx=corr(vx,common_qd),
            same_tick_common_qd_vs_pitch_gyro=corr(gyro_y,common_qd),lagged=lagged,
            interpretation='Closed-loop correlation only; negative actual vx feedback opposes measured translational velocity. No causal pitch gain or learned/contact stability claim.'))
        for i in selected:
            row=rows[i];before=rows[i-1];stamp=row['stamp_s']-.02
            previous=np.asarray(before['applied_policy_velocity']);measured=np.asarray(before['measured_body_velocity'])
            history=[(r['stamp_s']-.02,np.asarray(r['applied_policy_velocity']))for r in rows[max(0,i-32):i]]
            poses=before['fine_collision_evidence']['T_world_collision_bodies']
            body=np.asarray(poses['base']);link=np.asarray(poses['link0'])
            horizon=min(1.2,task.trajectory.end_s+5-stamp)
            begin=time.perf_counter()
            ctx=brake.LateralStandBrake(backend,stamp_s=stamp,measurement_stamp_s=before['stamp_s'],task_end_s=task.trajectory.end_s,
                stop_seconds=5.,body=body,link=link,measured_velocity=measured,previous_applied=previous,input_history=history,
                prediction_intervals=co.prediction_intervals(horizon),profile='vxy_p05_v1')
            support=co.command(stamp,lambda t:endpoint.copy(),link,body,ctx.first,measured,max_lookahead_s=horizon,
                previous_applied=previous,command_dt=.02,locked_input=True,prediction_provider=ctx.predict_link)
            issued=ctx.finalize(support['base_command'],veto_reasons=[]if support.get('feasible')is not False else['original_forecast_rejected'])
            walls.append(time.perf_counter()-begin)
            reports.append(dict(case=name,stamp_s=stamp,raw=ctx.raw.tolist(),target=ctx.target.tolist(),first=ctx.first.tolist(),issued=issued.tolist(),
                feasible=support['feasible'],reason=support['reason'],reserved_min_margin_m=support['predicted_min_domain_margin'],
                nodes_s=support['prediction_intervals_s'],
                cap_slew_pass=bool(np.all(abs(issued[:2])<=.03+1e-12)and np.all(abs(issued-previous)<=brake.SLEW*.02+1e-12)and issued[2]==0),
                raw_negative_velocity_pass=bool(np.allclose(ctx.raw[:2],-.5*measured[:2],atol=0,rtol=0))))
    report=dict(schema='go2_planar_stand_brake_snapshots/v1',snapshots=len(reports),correlations=correlations,
        cpu_gate_pass=all(r['cap_slew_pass']and r['raw_negative_velocity_pass']and r['feasible']is not False for r in reports),
        forecast_rejections=sum(r['feasible']is False for r in reports),minimum_reserved_margin_m=min(r['reserved_min_margin_m']for r in reports),
        control_only_wall_ms=dict(mean=float(np.mean(walls)*1000),p95=float(np.percentile(walls,95)*1000),max=float(max(walls)*1000)),
        scope='300 independent old D3 measured late-stop snapshots. No integrated state propagation, neural-policy execution, contact simulation or physical efficacy inferred.',
        original_world_reference_modified=False,physical_verified=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths},
        measured_evidence_sha256={str(p.relative_to(ROOT)):sha(p)for p in [RUN/'measurements.jsonl',RUN/'timed_tasks.json',RUN/'sensor_mounts.json']},
        frozen_sha256={str(p.relative_to(f)):sha(p)for p in f.rglob('*')if p.is_file()})
    (OUT/'rows.json').write_text(json.dumps(reports,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ['frozen_sha256','correlations','source_sha256','measured_evidence_sha256']},indent=2))

if __name__=='__main__':main()
