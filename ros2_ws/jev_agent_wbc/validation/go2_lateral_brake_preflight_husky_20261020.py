#!/usr/bin/env python3
"""Separate frozen late-stop measured states; no policy/physics propagation."""
import hashlib,importlib.util,json,shutil,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_bumpless_z_01'
OUT=ROOT/'Robot_Go2/results/lateral_stand_brake_cpu_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 OUT.mkdir(exist_ok=False);f=OUT/'sources';shutil.copytree(ROOT/'Robot_Go2/results/terminal_hold_full_preparation_01/sources',f)
 paths=[ROOT/'Robot_Go2'/n for n in ['lateral_stand_brake.py','adaptive_runner.py','bounded_body_prediction.py','camera_coordinator.py','world_z_bumpless.py','world_z_feedback.py']]+[ROOT/'test/test_wbc_go2_lateral_stand_brake.py',Path(__file__)]
 for p in paths:shutil.copy2(p,f/p.name)
 b=load('lateral_replay_backend',f/'go2_backend.py');b.UPSTREAM_MDP=f/'go2_upstream'
 c=load('lateral_replay_coordinator',f/'camera_coordinator.py');a=load('lateral_replay_brake',f/'lateral_stand_brake.py')
 sam=load('lateral_replay_sampler',f/'optical_sampler.py')
 from jev_agent_wbc.wbc_protocol import TimedTask
 tasks=[TimedTask.from_dict(x)for x in json.loads((RUN/'timed_tasks.json').read_text())];task=next(t for t in tasks if 'yaw_left'in t.task_id)
 endpoint=sam.OpticalSampler(task).sample(task.trajectory.end_s)
 ext=np.asarray(json.loads((RUN/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
 rows=[json.loads(x)for x in(RUN/'measurements.jsonl').read_text().splitlines()];co=c.CameraCoordinator(ext,backend=b,response_profile='measured_backward_v1',prediction_profile='bounded_ramp_history_v2')
 reports=[];wall=[]
 selected=[i for i,row in enumerate(rows)if row.get('task_feedback',{}).get('task_id')==task.task_id and row['stamp_s']-.02>=task.trajectory.end_s+2-1e-9]
 for index in selected:
  row=rows[index];before=rows[index-1];stamp=row['stamp_s']-.02;previous=np.asarray(before['applied_policy_velocity']);measured=np.asarray(before['measured_body_velocity'])
  history=[(r['stamp_s']-.02,np.asarray(r['applied_policy_velocity']))for r in rows[max(0,index-32):index]]
  body=np.asarray(before['T_world_base']);link=body.copy();link[:3,3]+=body[:3,:3]@np.array([0,0,.07])
  horizon=min(1.2,task.trajectory.end_s+5-stamp)
  begin=time.perf_counter();ctx=a.LateralStandBrake(b,stamp_s=stamp,measurement_stamp_s=before['stamp_s'],task_end_s=task.trajectory.end_s,stop_seconds=5.,
   body=body,link=link,measured_velocity=measured,previous_applied=previous,input_history=history,prediction_intervals=co.prediction_intervals(horizon))
  support=co.command(stamp,lambda t:endpoint.copy(),link,body,ctx.first,measured,max_lookahead_s=horizon,
   previous_applied=previous,command_dt=.02,locked_input=True,prediction_provider=ctx.predict_link)
  issued=ctx.finalize(support['base_command'],veto_reasons=[]if support.get('feasible')is not False else['original_forecast_rejected'])
  wall.append(time.perf_counter()-begin)
  reports.append(dict(stamp_s=stamp,raw=ctx.raw.tolist(),target=ctx.target.tolist(),first=ctx.first.tolist(),issued=issued.tolist(),
   feasible=support['feasible'],reason=support['reason'],reserved_min_margin_m=support['predicted_min_domain_margin'],nodes_s=support['prediction_intervals_s'],
   cap_slew_pass=bool(abs(issued[1])<=.03+1e-12 and np.all(abs(issued-previous)<=a.SLEW*.02+1e-12)and issued[0]==issued[2]==0),
   nonzero_prediction=bool(np.max(abs(ctx.model.trajectory(ctx.first,ctx.target)['roots'][-1,:2,3]-body[:2,3]))>1e-12)))
 result=dict(schema='go2_lateral_stand_brake_preflight/v1',snapshots=len(reports),scope='150 independent actual old measured states. No propagation, invented measurements, learned policy or physical brake efficacy proof.',
  cpu_gate_pass=all(r['cap_slew_pass']and r['feasible']is not False for r in reports),
  forecast_rejections=sum(r['feasible']is False for r in reports),minimum_reserved_margin_m=min(r['reserved_min_margin_m']for r in reports),
  control_only_wall_ms=dict(mean=float(np.mean(wall)*1000),p95=float(np.percentile(wall,95)*1000),max=float(max(wall)*1000)),
  reference_modified=False,physical_verified=False,source_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths},
  frozen_sha256={str(p.relative_to(f)):sha(p)for p in f.rglob('*')if p.is_file()})
 (OUT/'rows.json').write_text(json.dumps(reports,indent=2)+'\n');(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
