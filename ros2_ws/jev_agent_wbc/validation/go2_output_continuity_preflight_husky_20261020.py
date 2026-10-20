#!/usr/bin/env python3
"""Bounded fresh output-continuity arithmetic on unchanged actual Task boundaries."""
import hashlib,importlib.util,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_lateral_brake_01'
OUT=ROOT/'Robot_Go2/results/output_continuity_cpu_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 OUT.mkdir(exist_ok=False);f=OUT/'sources';f.mkdir()
 paths=[ROOT/'Robot_Go2/world_z_bumpless.py',ROOT/'Robot_Go2/world_z_feedback.py',ROOT/'test/test_wbc_go2_output_continuity.py',Path(__file__)]
 for p in paths:shutil.copy2(p,f/p.name)
 w=load('output_continuity_candidate',f/'world_z_bumpless.py');b=load('continuity_original_backend',RUN/'sources/go2_backend.py');b.UPSTREAM_MDP=RUN/'sources/go2_upstream'
 sam=load('continuity_original_sampler',RUN/'sources/optical_sampler.py');domain=b.PolicyDomain()
 from jev_agent_wbc.wbc_protocol import TimedTask
 tasks=[TimedTask.from_dict(x)for x in json.loads((RUN/'timed_tasks.json').read_text())];rows=[json.loads(x)for x in(RUN/'measurements.jsonl').read_text().splitlines()]
 ext=np.asarray(json.loads((RUN/'sensor_mounts.json').read_text())['camera']['T_parent_optical']);records=[]
 for number,task in enumerate(tasks):
  index=next(i for i,row in enumerate(rows)if row.get('task_feedback',{}).get('task_id')==task.task_id);previous=rows[index-1];stamp=previous['stamp_s'];camera=np.asarray(previous['T_world_camera_optical']);reference=sam.OpticalSampler(task).sample(stamp)
  body=np.asarray(previous['T_world_base']);link=body.copy();link[:3,3]+=body[:3,:3]@np.array([0,0,.07]);actual=np.asarray(previous['actual_native_command'],dtype=np.float32)if number else None
  context=dict(now_s=stamp,observation_stamp_s=stamp,reference_optical=reference,measured_optical=camera,link0=link,extrinsic=ext,
   previous_applied_native=actual,previous_issue_stamp_s=stamp-.02 if number else None,history_rows=rows[max(0,index-26):index],domain=domain)
  record=dict(task_id=task.task_id,task_sha_preserved=True,modes={})
  for profile in ['fresh_steady_native_z_v1','fresh_actual_output_continuity_v2']:
   c=w.BumplessWorldZFeedback(enabled=True,initialization_profile=profile);decision=c.admit(b,task,**context)
   native,check=c.feedback_native_command(b,reference,camera,link,ext,task_id=task.task_id,stamp_s=stamp,now_s=stamp,dt_s=.02,sample_seq=round(stamp/.02),fresh=True,integration_allowed=True,domain=domain)
   if check['world_z_feedback'].get('seed_pending_commit'):assert c.finish_seed_update(check['world_z_feedback']['update_id'],applied_native=np.asarray(native,dtype=np.float32))
   record['modes'][profile]=dict(decision=decision,first_native_z_m=float(np.float32(native[2])),
    jump_from_previous_actual_m=float(np.float32(native[2])-actual[2])if number else None,
    original_total_correction_norm_m=check['world_z_feedback']['proposed_position_correction_norm_m'],domain=check['status'])
   record['task_sha_preserved']&=decision['task_sha256']==w.canonical_task(task)[1]
  records.append(record)
 new=[r['modes']['fresh_actual_output_continuity_v2']for r in records]
 result=dict(schema='go2_fresh_output_continuity_cpu/v1',cpu_gate_pass=all(r['task_sha_preserved']for r in records)and all(r['original_total_correction_norm_m']<=.025+1e-12 and r['domain']=='inside_outer_domain_unverified'for r in new),
  subsequent_boundaries=8,new_eligible_count=sum(r['decision']['eligible']for r in new),records=records,
  exact_output_continuity_only=True,steady_bias_or_training_claim=False,physical_verified=False,runner_changed=False,
  scope='Independent actual boundary snapshots only; no command propagation, NN/PhysX or all18qstop effect claimed. Legacy steady1mm profile unchanged; new mode keeps those spans as diagnostics.',
  source_sha256={str(p.relative_to(ROOT)):sha(p)for p in paths+[RUN/'measurements.jsonl',RUN/'timed_tasks.json',RUN/'provenance.json']})
 assert result['new_eligible_count']==8
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
