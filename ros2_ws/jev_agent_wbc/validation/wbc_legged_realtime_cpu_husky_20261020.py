#!/usr/bin/env python3
"""CPU-only, no pacing delays, replay/microbenchmarks; never starts Isaac/PhysX."""
import cProfile
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pstats
import sys
import tempfile
import time
from types import SimpleNamespace

os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy.spatial.transform import Rotation
PACKAGE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PACKAGE))
from jev_agent_wbc.wbc_protocol import TimedTask,Capabilities
from jev_agent_wbc.wbc_physical import LiveTaskSession
from jev_agent_wbc.representation import Observation

def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def timed(f,repeats=3):
 values=[]
 for _ in range(repeats):
  started=time.perf_counter();f();values.append(time.perf_counter()-started)
 return dict(wall_s=values,median_s=float(np.median(values)))

def main():
 root=PACKAGE/'Robot_Go2/results/adaptive_backward_precision_01';frozen=root/'sources'
 for n in ['wbc_protocol.py','wbc_physical.py','representation.py']:
  assert sha(frozen/'jev_agent_wbc'/n)==sha(PACKAGE/'jev_agent_wbc'/n),'public sources differ from preserved run'
 rows=[json.loads(line) for line in (root/'measurements.jsonl').open()]
 task=TimedTask.from_dict(json.loads((root/'timed_tasks.json').read_text())[0]);initial=next(r for r in rows if abs(r['stamp_s']-task.trajectory.start_s)<1e-8)
 session=LiveTaskSession(Capabilities('go2_benchmark',True,frozenset(['timed_camera_se3'])))
 assert session.accept(task,Observation.from_dict(initial['measured_observation']),task.trajectory.start_s)['supported']
 sampler_module=load('bench_sampler',PACKAGE/'Robot_Go2/optical_sampler.py');sampler=sampler_module.OpticalSampler(task)
 stamps=np.linspace(task.trajectory.start_s,task.trajectory.end_s,500)
 baseline=lambda t:session.reference(float(t))[0]
 optimized=lambda t:sampler.sample(float(t))
 errors=[float(np.max(abs(baseline(t)-optimized(t)))) for t in stamps]
 def samples(f):
  for t in stamps:f(t)
 sampler_times={name:timed(lambda f=fn:samples(f),5) for name,fn in [('public_reference',baseline),('optical_pose_only',optimized)]}
 # Full unchanged CameraCoordinator replay. Link0 is fixed +.07m above base
 # in the saved source joint; both paths use identical reconstructed input.
 backend=load('bench_frozen_go2',frozen/'go2_backend.py');backend.UPSTREAM_MDP=frozen/'go2_upstream'
 coordinator_module=load('bench_frozen_coordinator',frozen/'camera_coordinator.py')
 extrinsic=np.array(json.loads((root/'sensor_mounts.json').read_text())['camera']['T_parent_optical'])
 active=[i for i,r in enumerate(rows) if r['case']=='backward'][::5]
 def coordinator_replay(sample,save=False):
  coord=coordinator_module.CameraCoordinator(extrinsic,backend=backend,response_profile='measured_backward_v1');out=[]
  # Keep the complete original input history, while benchmarking every fifth
  # callback. This is not a closed-loop accuracy/worker-timing experiment.
  for i in range(1,len(rows)):
   r,previous=rows[i],rows[i-1];stamp=r['stamp_s']-.02
   if i in active:
    body=np.array(previous['T_world_base']);link=body.copy();link[:3,3]+=body[:3,:3]@np.array([0,0,.07])
    support=r['camera_support'];preferred=np.array(support['preferred_base_command']);future=np.array(r['requested_body_velocity']);future*=np.where(future<0,np.array([1/.55,1,1]),1.)
    result=coord.command(stamp,lambda t:sample(float(min(t,task.trajectory.end_s))),link,body,preferred,np.array(previous['measured_body_velocity']),max_lookahead_s=max(0.,task.trajectory.end_s-stamp),future_policy_velocity=np.clip(future,-.2,.2),previous_applied=np.array(previous['applied_policy_velocity']),command_dt=.02)
    if save:out.append(result)
   coord.observe_applied_input(stamp,np.array(r['applied_policy_velocity']))
  return out
 old=coordinator_replay(baseline,True);new=coordinator_replay(optimized,True)
 command_error=max(float(np.max(abs(a['base_command']-b['base_command']))) for a,b in zip(old,new))
 margins_error=max(abs(a['predicted_min_domain_margin']-b['predicted_min_domain_margin']) for a,b in zip(old,new))
 statuses_equal=all((a['camera_supported'],a['feasible'],a['reason'])==(b['camera_supported'],b['feasible'],b['reason']) for a,b in zip(old,new))
 coordinator_times={name:timed(lambda f=fn:coordinator_replay(f),3) for name,fn in [('public_reference',baseline),('optical_pose_only',optimized)]}
 profile=cProfile.Profile();profile.enable();coordinator_replay(baseline);profile.disable();stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(22)
 # Batch pose conversions use a cached body-name index; pure CPU transforms
 # only. PhysX reads are not measured or bypassed by this microbenchmark.
 rig=load('bench_rig',frozen/'jev_agent_wbc/robot_sensor_rig.py')
 collision=next(r['fine_collision_evidence'] for r in rows if r.get('fine_collision_evidence',{}).get('T_world_collision_bodies'))
 poses=collision['T_world_collision_bodies'];names=list(poses);matrices=np.array([poses[n] for n in names]);quats=Rotation.from_matrix(matrices[:,:3,:3]).as_quat()[:,[3,0,1,2]]
 states=np.column_stack([matrices[:,:3,3],quats,np.zeros((len(names),6))]);indices=np.arange(len(names))
 def per_body():return {n:rig.pose_transform(states[names.index(n),:3],states[names.index(n),3:7]) for n in names}
 def batch():
  selected=states[indices];out=np.broadcast_to(np.eye(4),(len(names),4,4)).copy();out[:,:3,:3]=Rotation.from_quat(selected[:,[4,5,6,3]]).as_matrix();out[:,:3,3]=selected[:,:3]
  if not np.isfinite(out).all():raise ValueError('nonfinite sensor transform')
  return {n:out[i] for i,n in enumerate(names)}
 oldp,newp=per_body(),batch();pose_error=max(float(np.max(abs(oldp[n]-newp[n]))) for n in names)
 def many(f):
  for _ in range(1000):f()
 pose_times={name:timed(lambda f=fn:many(f),3) for name,fn in [('per_body_current',per_body),('cached_indices_batch_rotation',batch)]}
 # Preserve every journal field; compare JSON semantic equality. Raw log bytes
 # are not hashes of a canonical wire format. No old files are modified.
 json_times={name:timed(lambda options=opts:[json.dumps(r,allow_nan=False,**options) for r in rows],3) for name,opts in [('current',{}),('compact_separators',{'separators':(',',':')})]}
 encoded=[json.dumps(r,allow_nan=False)+'\n' for r in rows]
 def write(buffering):
  with tempfile.TemporaryDirectory(prefix='jev-journal-bench-') as folder:
   with (Path(folder)/'raw.jsonl').open('w',buffering=buffering) as f:
    for line in encoded:f.write(line)
 write_times={name:timed(lambda b=b:write(b),3) for name,b in [('line_buffered',1),('block_buffered_64KiB',65536)]}
 summaries={}
 for p in [root,PACKAGE/'Robot_G1/results/adaptive_recovery_sameprofile_01']:
  s=json.loads((p/'summary.json').read_text());means=s['mean_control_phase_wall_ms'];count=round(s['sim_seconds']/.02);total=sum(means.values())*.001*count
  summaries[p.name]=dict(logged_control_ticks=count,phase_means_ms=means,sum_phase_mean_ms=sum(means.values()),sum_accounted_phase_wall_s=total,actual_wall_s=s['wall_seconds'],unaccounted_wall_including_absolute_pace_wait_and_partial_failed_tick_s=s['wall_seconds']-total,run_complete=s['run_complete'],timing_pass=s['timing_pass'])
 report=dict(schema='legged_realtime_cpu_diagnostic/v1',scope='CPU-only preserved-data microbenchmarks with no deliberate slowdown, no Kit/PhysX, no controller/source/log modifications; timings are local measured medians, not a physical realtime pass',summaries=summaries,sampler=dict(sample_count=len(stamps),maximum_optical_matrix_difference=max(errors),times=sampler_times,estimated_six_samples_saved_ms=6*1000*(sampler_times['public_reference']['median_s']-sampler_times['optical_pose_only']['median_s'])/len(stamps)),coordinator=dict(replay_calls=len(active),same_statuses=statuses_equal,maximum_base_command_difference=command_error,maximum_domain_margin_difference=margins_error,times=coordinator_times,baseline_profile=stream.getvalue()),body_pose_conversion=dict(body_count=len(names),maximum_matrix_difference=pose_error,repetitions=1000,times=pose_times),json_serialization=dict(rows=len(rows),original_bytes=(root/'measurements.jsonl').stat().st_size,times=json_times),journal_write_only=dict(serialization_excluded=True,rows=len(rows),times=write_times),source_sha256={})
 source_paths=[Path(__file__),PACKAGE/'Robot_Go2/optical_sampler.py',PACKAGE/'test/test_wbc_optical_sampler.py',root/'measurements.jsonl',root/'summary.json',root/'timed_tasks.json',root/'provenance.json',PACKAGE/'Robot_G1/results/adaptive_recovery_sameprofile_01/summary.json',frozen/'adaptive_runner.py',frozen/'camera_coordinator.py',frozen/'go2_backend.py',frozen/'legged_collision.py']
 source_paths += [frozen/'jev_agent_wbc'/n for n in ['wbc_protocol.py','wbc_physical.py','representation.py','robot_sensor_rig.py']]
 source_paths += [PACKAGE/'Robot_Go2/vendor/IsaacLab/source/isaaclab/isaaclab/assets/articulation'/n for n in ['articulation.py','articulation_data.py']]
 report['source_sha256']={str(p.relative_to(PACKAGE)):sha(p) for p in source_paths}
 output=PACKAGE/'validation/wbc_legged_realtime_cpu_husky_20261020.json'
 if output.exists():raise FileExistsError(output)
 output.write_text(json.dumps(report,indent=2)+'\n');print(output)
 print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','summaries']},indent=2)[:3500])
if __name__=='__main__':main()
