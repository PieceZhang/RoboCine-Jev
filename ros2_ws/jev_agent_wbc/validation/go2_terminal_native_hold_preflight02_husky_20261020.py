#!/usr/bin/env python3
"""Short immutable measured snapshot checks, not a learned-policy rollout."""
import hashlib,importlib.util,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from jev_agent_wbc.wbc_protocol import TimedTask,Capabilities
from jev_agent_wbc.wbc_physical import LiveTaskSession
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_bumpless_z_01'
OUT=ROOT/'Robot_Go2/results/terminal_native_hold_cpu_02'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 OUT.mkdir(exist_ok=False);f=OUT/'sources';f.mkdir()
 paths=[ROOT/'Robot_Go2'/n for n in ['terminal_native_hold.py','world_z_bumpless.py','world_z_feedback.py','adaptive_runner.py','optical_sampler.py','bounded_body_prediction.py']]+[Path(__file__)]
 for path in paths:shutil.copy2(path,f/path.name)
 h=load('preflight_terminal_native_hold',f/'terminal_native_hold.py');sam=load('preflight_terminal_sampler',f/'optical_sampler.py');b=load('preflight_hold_backend',RUN/'sources/go2_backend.py');b.UPSTREAM_MDP=RUN/'sources/go2_upstream';domain=b.PolicyDomain()
 tasks=[TimedTask.from_dict(x)for x in json.loads((RUN/'timed_tasks.json').read_text())];rows=[json.loads(x)for x in(RUN/'measurements.jsonl').read_text().splitlines()]
 ext=np.asarray(json.loads((RUN/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
 records=[];errors=[];negative=0;count=0
 for task in tasks:
  hold=h.TerminalNativeHold(task);sampler=sam.OpticalSampler(task);session=LiveTaskSession(Capabilities('cpu_only',True,frozenset(['timed_camera_se3'])));session._task=task
  # The input Task was admitted in the frozen physical run. This is only an
  # independent pose-equality check, not a substitute admission route.
  times=list(task.trajectory.times_s)
  times=[task.trajectory.start_s+x for x in times]
  rng=np.random.default_rng(749);times+=list(rng.uniform(task.trajectory.start_s,task.trajectory.end_s,100))+[task.trajectory.start_s,task.trajectory.end_s]
  for t in times:
   a=session.reference(float(t))[0];c=sampler.sample(float(t));errors.append(float(np.max(np.abs(a-c))));count+=1
  for t in [task.trajectory.start_s-.01,task.trajectory.end_s+.01]:
   old=new=False
   try:session.reference(t)
   except ValueError:old=True
   try:sampler.sample(t)
   except ValueError:new=True
   assert old and new;negative+=1
  index=next(i for i,row in enumerate(rows)if row['task_feedback']['task_id']==task.task_id and row['stamp_s']-.02>=task.trajectory.end_s+2-1e-9)
  current=rows[index];previous=rows[index-1];stamp=current['stamp_s']-.02
  candidate=np.asarray(current['actual_native_command']);consumed=np.asarray(previous['actual_native_command'],dtype=np.float32)
  selected,report=hold.select(candidate,stamp_s=stamp,measurement_stamp_s=previous['stamp_s'],previous_consumed_native=consumed,
   previous_issue_stamp_s=stamp-.02,previous_consumption_seq=round(stamp/.02)-1,previous_body_input=previous['applied_policy_velocity'],issued_body_input=current['applied_policy_velocity'],
   domain=domain,collision_evidence=current['fine_collision_evidence'],original_support_ok=current['camera_support']['feasible']is not False,
   world_position_error_m=previous['position_error_m'],world_orientation_error_rad=previous['orientation_error_rad'])
  confirmed=hold.confirm(report,actual_native=selected,actual_body_input=current['applied_policy_velocity'],stamp_s=stamp,consumption_seq=round(stamp/.02),override_reasons=['terminal_native_hold'])
  drift=[]
  for row in rows[index:]:
   if row['task_feedback']['task_id']!=task.task_id:break
   link=np.asarray(row['T_world_base']).copy();link[:3,3]+=link[:3,:3]@np.array([0,0,.07]);target=b.mixed_to_world_camera(selected,link,ext);reference=np.asarray(row['T_world_reference_optical'])
   drift.append([float(np.linalg.norm(target[:3,3]-reference[:3,3])),float(Rotation.from_matrix(target[:3,:3]@reference[:3,:3].T).magnitude())])
  records.append(dict(task_id=task.task_id,capture_selected=report['selected'],actual_f32_confirmed=confirmed,capture_stamp_s=stamp,
   native_exact_previous_consumed=h.same_native_bits(selected,consumed),capture_native7=selected.tolist(),original_current_error_m=previous['position_error_m'],
   original_current_error_rad=previous['orientation_error_rad'],frozen_old_body_path_max_held_desired_world_error=np.max(drift,axis=0).tolist(),
   scope='Re-expressed desired native7 on the old measured body path only; neither actual policy response nor counterfactual body path is simulated.'))
 result=dict(schema='go2_terminal_native_hold_cpu/v1',cpu_gate_pass=all(r['capture_selected']and r['actual_f32_confirmed']and r['native_exact_previous_consumed']for r in records)and max(errors)<2e-14,
  records=records,pose_comparisons=count,max_pose_matrix_abs_error=max(errors),clock_negative_comparisons=negative,
  source_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths},frozen_sha256={str(p.relative_to(f)):sha(p)for p in f.rglob('*')if p.is_file()},
  physical_verified=False,scope='Nine fresh frozen end+2 measurements only, fullTask unchanged; no propagation/new physics or all18qstop claim.')
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
