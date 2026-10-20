#!/usr/bin/env python3
"""Candidate future-plan consistency on preserved state traces; no controller edits."""
from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
P=Path(__file__).resolve().parents[1];R=P/'Robot_Go2/results/adaptive_camera6d_softpose_01';F=R/'sources';sys.path.insert(0,str(P))
from jev_agent_wbc.wbc_protocol import TimedTask

def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 started=time.monotonic();output=P/'validation/go2_camera6d_persistent_plan_husky_20261020';assert not output.exists();output.mkdir();source=(F/'camera_coordinator.py').read_text();old='''            if not posture_weight:return future_policy_velocity
            # The from-now input bias persists through this short prediction;''';new='''            if future_policy_velocity is None and not posture_weight:return None
            # The from-now input bias persists through this short prediction;''';assert source.count(old)==1;candidate=source.replace(old,new);candidate_path=output/'candidate_coordinator.py';candidate_path.write_text(candidate)
 rows=[json.loads(s) for s in (R/'measurements.jsonl').open()];task=TimedTask.from_dict(json.loads((R/'timed_tasks.json').read_text())[0]);sampler=load('persistent_sampler',F/'optical_sampler.py').OpticalSampler(task);backend=load('persistent_backend',F/'go2_backend.py');backend.UPSTREAM_MDP=F/'go2_upstream';module=load('persistent_candidate',candidate_path);E=np.array(json.loads((R/'sensor_mounts.json').read_text())['camera']['T_parent_optical']);coord=module.CameraCoordinator(E,backend=backend,response_profile='measured_backward_v1');captured={};code=type(coord).command.__code__;output_rows=[]
 def trace(frame,event,arg):
  if frame.f_code is code and event=='return':captured.clear();captured.update(frame.f_locals)
  return trace
 for i,r in enumerate(rows):
  stamp=r['stamp_s']-.02
  if r['case']!='settle':
   previous=rows[i-1];body=np.array(previous['T_world_base']);link=body.copy();link[:3,3]+=body[:3,:3]@np.array([0,0,.07]);original=r['camera_support'];preferred=np.array(original['preferred_base_command']);future=np.array(r['requested_body_velocity']);future*=np.where(future<0,np.array([1/.55,1,1]),1.);future=np.clip(future,-.2,.2);prev=np.array(previous['applied_policy_velocity']);sys.settrace(trace)
   result=coord.command(stamp,lambda t:sampler.sample(min(float(t),task.trajectory.end_s)),link,body,preferred,np.array(previous['measured_body_velocity']),max_lookahead_s=max(0.,task.trajectory.end_s-stamp),future_policy_velocity=future,previous_applied=prev,command_dt=.02);sys.settrace(None)
   applied=np.array(result['base_command']);margins=captured['margins'](applied);intervals=captured['intervals'];j=int(np.argmin(margins));labels=['x_lower','y_lower','z_lower','x_upper','y_upper','z_upper','roll_lower','pitch_lower','yaw_lower','roll_upper','pitch_upper','yaw_upper'];plan=np.clip(future+applied-preferred,-coord.limits,coord.limits)
   output_rows.append(dict(row=i,measurement_end_stamp_s=r['stamp_s'],command_stamp_s=stamp,original_feasible=original['feasible'],original_reason=original['reason'],candidate_feasible=result['feasible'],candidate_reason=result['reason'],actual_non_linear_min_margin=float(np.min(margins)),active_bound=labels[j%12],active_lookahead_s=float(intervals[j//12]),candidate_first_command=applied.tolist(),preferred=preferred.tolist(),candidate_future_plan=plan.tolist(),first_input_within_original_caps=bool(np.all(abs(applied)<=coord.limits+1e-9)),first_input_within_original_slew=bool(np.all(abs(applied-prev)<=np.array([.5,.5,.8])*.02+1e-9)),future_plan_within_original_caps=bool(np.all(abs(plan)<=coord.limits+1e-9)),native_current_xyz=np.asarray(result['native_command'])[:3].tolist()))
  coord.observe_applied_input(stamp,np.array(r['applied_policy_velocity']))
 false=[r for r in output_rows if r['candidate_feasible'] is False];recovered=[r for r in output_rows if r['original_feasible'] is False and r['candidate_feasible'] is not False];new_fail=[r for r in output_rows if r['original_feasible'] is not False and r['candidate_feasible'] is False];invalid_pass=[r for r in output_rows if r['candidate_feasible'] is not False and r['actual_non_linear_min_margin']< -1e-7]
 report=dict(schema='go2_persistent_future_plan_cpu/v1',scope='Preserved measured-state replay only, not CPU closed-loop and not a physical acceptance. Actual input history remains the original actually applied history; newly proposed commands are never applied to later frozen measurements.',change='Candidate−preferred input bias persists into the existing bounded future plan even when posture_weight=0. Original futureNone/no-posture semantics retained. No other coordinator changes.',horizon_s=.8,horizon_samples=5,position_reserve_m=[.035,.035,.015],input_caps=[.2,.2,.2],input_slew=[.5,.5,.8],original_false_count=131,candidate_false_count=len(false),original_false_recovered_count=len(recovered),new_false_count=len(new_fail),falsely_accepted_nonlinear_count=len(invalid_pass),caps_or_slew_violations=sum(not(r['first_input_within_original_caps'] and r['first_input_within_original_slew'] and r['future_plan_within_original_caps']) for r in output_rows),candidate_failed_rows=false,candidate_recovered_rows=recovered,cpu_wall_s=time.monotonic()-started,source_sha256={})
 files=[Path(__file__),candidate_path,R/'measurements.jsonl',R/'timed_tasks.json',R/'provenance.json',F/'camera_coordinator.py',F/'go2_backend.py',F/'optical_sampler.py'];report['source_sha256']={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};(output/'replay.json').write_text(json.dumps(report,indent=2)+'\n');(output/'all_rows.json').write_text(json.dumps(output_rows,indent=2)+'\n');print(output);print({k:v for k,v in report.items() if k not in ['candidate_failed_rows','candidate_recovered_rows','source_sha256']})
 print('Remaining failures',[(r['measurement_end_stamp_s'],r['actual_non_linear_min_margin'],r['active_bound'],r['active_lookahead_s']) for r in false[::max(1,len(false)//15)]])
if __name__=='__main__':main()
