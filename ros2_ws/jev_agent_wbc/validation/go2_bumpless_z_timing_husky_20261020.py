#!/usr/bin/env python3
"""Paced paired-call microbenchmark, separate from physical acceptance."""
import hashlib,importlib.util,json,shutil,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_bumpless_z_01'
OLD=ROOT/'Robot_Go2/results/adaptive_suite_terminal_pose_01'
OUT=ROOT/'Robot_Go2/results/bumpless_z_timing_cpu_01'

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def stats(a):return dict(mean_us=float(np.mean(a)*1e6),median_us=float(np.median(a)*1e6),p95_us=float(np.percentile(a,95)*1e6))

def main():
 OUT.mkdir(exist_ok=False);f=OUT/'sources';f.mkdir()
 for name in ['world_z_bumpless.py','world_z_feedback.py','wbc_backend.py','optical_sampler.py']:shutil.copy2(RUN/'sources'/('go2_backend.py'if name=='wbc_backend.py'else name),f/name)
 shutil.copy2(Path(__file__),f/Path(__file__).name)
 b=load('timing_backend',f/'wbc_backend.py');b.UPSTREAM_MDP=RUN/'sources/go2_upstream'
 w=load('timing_seed',f/'world_z_bumpless.py');sam=load('timing_optical_sampler',f/'optical_sampler.py');domain=b.PolicyDomain()
 from jev_agent_wbc.wbc_protocol import TimedTask,Capabilities
 from jev_agent_wbc.wbc_physical import LiveTaskSession
 tasks=[TimedTask.from_dict(v)for v in json.loads((RUN/'timed_tasks.json').read_text())]
 rows=[json.loads(x)for x in(RUN/'measurements.jsonl').read_text().splitlines()];summ=json.loads((RUN/'summary.json').read_text());old=json.loads((OLD/'summary.json').read_text())
 ext=np.asarray(json.loads((RUN/'sensor_mounts.json').read_text())['camera']['T_parent_optical']);cal=json.loads((RUN/'world_z_seed_calibrations.json').read_text())['records']
 base=w._base.WorldZFeedback(enabled=True);wrap=w.BumplessWorldZFeedback(enabled=True)
 base_times=[];wrapper_times=[];json_base=[];json_extra=[];public=[];cached=[];equivalent=True
 selected=[i for i in range(101,len(rows),10)]
 sessions=[];samplers=[]
 for task in tasks:
  session=LiveTaskSession(Capabilities('timing_only',True,frozenset(['timed_camera_se3'])));session._task=task
  sessions.append(session);samplers.append(sam.OpticalSampler(task))
 begin=time.monotonic()
 for number,i in enumerate(selected):
  row=rows[i];prev=rows[i-1];case=next(j for j,t in enumerate(tasks)if t.task_id==row['task_feedback']['task_id']);task=tasks[case]
  stamp=row['stamp_s']-.02;camera=np.asarray(prev['T_world_camera_optical']);link=np.asarray(prev['T_world_base']).copy();link[:3,3]+=link[:3,:3]@np.array([0,0,.07])
  ref=samplers[case].sample(min(stamp,task.trajectory.end_s));args=dict(task_id=task.task_id,stamp_s=stamp,now_s=stamp,dt_s=.02,sample_seq=round(stamp/.02),fresh=True,integration_allowed=True,domain=domain)
  results={}
  for kind in (['base','wrapper']if number%2==0 else['wrapper','base']):
   c=base if kind=='base'else wrap;c.reset(task.task_id);c.integral_m=row['actual_native_domain']['world_z_feedback']['integral_before_m'];c._last_stamp_s=stamp-.02;c._last_sample_seq=round(stamp/.02)-1;c._first_in_task=False
   if kind=='wrapper':c.admission=cal[case];c._seed_pending=None;c._seed_update_id=None
   t=time.perf_counter();results[kind]=c.feedback_native_command(b,ref,camera,link,ext,**args);elapsed=time.perf_counter()-t
   (base_times if kind=='base'else wrapper_times).append(elapsed)
  equivalent=bool(equivalent and np.array_equal(results['base'][0],results['wrapper'][0]))
  oldreport=results['base'][1]['world_z_feedback'];newreport=results['wrapper'][1]['world_z_feedback']
  for kind in (['old','new']if number%2==0 else['new','old']):
   t=time.perf_counter();json.dumps(oldreport if kind=='old'else newreport,allow_nan=False);elapsed=time.perf_counter()-t
   (json_base if kind=='old'else json_extra).append(elapsed)
  for kind in (['public','cached']if number%2==0 else['cached','public']):
   t=time.perf_counter();value=sessions[case].reference(min(stamp,task.trajectory.end_s))[0]if kind=='public'else samplers[case].sample(min(stamp,task.trajectory.end_s));elapsed=time.perf_counter()-t
   (public if kind=='public'else cached).append(elapsed)
   if kind=='public':reference=value
   else:fast=value
  equivalent=bool(equivalent and np.allclose(reference,fast,atol=2e-14,rtol=0))
  delay=begin+(number+1)*.02-time.monotonic()
  if delay>0:time.sleep(delay)
 admission=[]
 for j,task in enumerate(tasks):
  index=next(i for i,row in enumerate(rows)if row.get('task_feedback',{}).get('task_id')==task.task_id);prev=rows[index-1];stamp=prev['stamp_s'];link=np.asarray(prev['T_world_base']).copy();link[:3,3]+=link[:3,:3]@np.array([0,0,.07]);c=w.BumplessWorldZFeedback(enabled=True)
  t=time.perf_counter();c.admit(b,task,now_s=stamp,observation_stamp_s=stamp,reference_optical=samplers[j].sample(stamp),measured_optical=prev['T_world_camera_optical'],link0=link,extrinsic=ext,previous_applied_native=np.asarray(prev['actual_native_command'],dtype=np.float32)if j else None,previous_issue_stamp_s=stamp-.02 if j else None,history_rows=rows[max(0,index-26):index],domain=domain);admission.append(time.perf_counter()-t)
 phases={k:dict(old_ms=v,new_ms=summ['mean_control_phase_wall_ms'][k],delta_ms=summ['mean_control_phase_wall_ms'][k]-v,estimated_4600tick_delta_s=(summ['mean_control_phase_wall_ms'][k]-v)*4.6)for k,v in old['mean_control_phase_wall_ms'].items()}
 result=dict(schema='go2_bumpless_timing_cpu/v1',pace=1,samples=len(selected),paired_alternating_order=True,numerical_outputs_equivalent=equivalent,physical_verified=False,
  base_feedback=stats(base_times),wrapper_feedback=stats(wrapper_times),mean_wrapper_delta_us=float((np.mean(wrapper_times)-np.mean(base_times))*1e6),estimated4500call_extra_s=float((np.mean(wrapper_times)-np.mean(base_times))*4500),
  report_json_base=stats(json_base),report_json_with_metadata=stats(json_extra),estimated4500metadata_json_extra_s=float((np.mean(json_extra)-np.mean(json_base))*4500),
  nine_admissions_total_s=float(sum(admission)),admissions_s=admission,public_pose_and_unused_twist=stats(public),cached_original_pose=stats(cached),estimated9000pose_only_saving_s=float((np.mean(public)-np.mean(cached))*9000),
  measured_physical_old_wall_s=old['wall_seconds'],measured_physical_new_wall_s=summ['wall_seconds'],physical_phase_deltas=phases,
  scope='Paired CPU call cost on identical frozen measured states,50Hz paced; not a rerun of learned policy or PhysX. Estimated sums do not prove future realtime acceptance.',
  source_sha256={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),RUN/'summary.json',OLD/'summary.json']},frozen_sha256={str(p.relative_to(f)):sha(p)for p in f.rglob('*')if p.is_file()})
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
