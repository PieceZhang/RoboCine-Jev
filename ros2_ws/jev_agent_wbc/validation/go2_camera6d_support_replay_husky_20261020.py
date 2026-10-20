#!/usr/bin/env python3
"""Read-only same-coordinator replay and bounded mathematical feasibility probes."""
from pathlib import Path
import sys,json,hashlib,importlib.util,itertools,time
import numpy as np
from scipy.optimize import minimize
P=Path(__file__).resolve().parents[1];R=P/'Robot_Go2/results/adaptive_camera6d_softpose_01';F=R/'sources';sys.path.insert(0,str(P))
from jev_agent_wbc.wbc_protocol import TimedTask

def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 start=time.monotonic();rows=[json.loads(s) for s in (R/'measurements.jsonl').open()];task=TimedTask.from_dict(json.loads((R/'timed_tasks.json').read_text())[0]);backend=load('replay_backend',F/'go2_backend.py');backend.UPSTREAM_MDP=F/'go2_upstream';module=load('replay_coordinator',F/'camera_coordinator.py');sampler=load('replay_optical',F/'optical_sampler.py').OpticalSampler(task);E=np.array(json.loads((R/'sensor_mounts.json').read_text())['camera']['T_parent_optical']);coord=module.CameraCoordinator(E,backend=backend,response_profile='measured_backward_v1')
 bad=[i for i,r in enumerate(rows) if r.get('task_feedback',{}).get('feasible') is False];worst=min(bad,key=lambda i:rows[i]['camera_support']['predicted_min_domain_margin']);selected=set([bad[0],bad[-1],worst]+[bad[round((len(bad)-1)*x)] for x in [.125,.25,.375,.5,.625,.75,.875]])
 capture={};code=type(coord).command.__code__
 def trace(frame,event,arg):
  if frame.f_code is code and event=='return':capture.update(frame.f_locals)
  return trace
 def max_margin(fn,low,high,seeds):
  results=[]
  for v in seeds:
   x=np.r_[np.clip(v,low,high),min(0.,float(np.min(fn(np.clip(v,low,high)))))]
   res=minimize(lambda z:-z[3],x,method='SLSQP',bounds=list(zip(low,high))+[(-5.,1.)],constraints=[dict(type='ineq',fun=lambda z:fn(z[:3])-z[3])],options=dict(maxiter=80,ftol=1e-10))
   margin=float(np.min(fn(res.x[:3])));results.append(dict(v=res.x[:3].tolist(),margin=margin,solver_success=bool(res.success),message=res.message))
  return max(results,key=lambda r:r['margin']),results
 report=dict(schema='go2_camera6d_support_replay/v1',scope='CPU frozen-source replay from preserved measured base/root velocities and exact original TimedTask; link0 reconstructed through source fixed+.07m joint. Optimization probes are model mathematics, not physical feasibility certificates or a controller change.',original_false_samples=len(bad),replay_counts=dict(),selected={},source_sha256={});status_mismatch=[];maxnative=maxcmd=maxmargin=0.;false_count=0
 for i,r in enumerate(rows):
  stamp=r['stamp_s']-.02;future=np.array(r['requested_body_velocity'],float)
  if r['case']!='settle':
   prev=rows[i-1];body=np.array(prev['T_world_base']);link=body.copy();link[:3,3]+=body[:3,:3]@np.array([0,0,.07]);support=r['camera_support'];preferred=np.array(support['preferred_base_command']);previous=np.array(prev['applied_policy_velocity']);future*=np.where(future<0,np.array([1/.55,1,1]),1.);future=np.clip(future,-.2,.2)
   if i in selected:sys.settrace(trace)
   result=coord.command(stamp,lambda t:sampler.sample(min(float(t),task.trajectory.end_s)),link,body,preferred,np.array(prev['measured_body_velocity']),max_lookahead_s=max(0.,task.trajectory.end_s-stamp),future_policy_velocity=future,previous_applied=previous,command_dt=.02)
   if i in selected:sys.settrace(None)
   maxnative=max(maxnative,float(np.max(abs(result['native_command']-support['native_command']))));maxcmd=max(maxcmd,float(np.max(abs(result['base_command']-support['base_command']))));maxmargin=max(maxmargin,abs(result['predicted_min_domain_margin']-support['predicted_min_domain_margin']))
   false_count+=result['feasible'] is False
   if (result['feasible'],result['reason'])!=(support['feasible'],support['reason']):status_mismatch.append(dict(row=i,recorded=support['reason'],replay=result['reason']))
   if i in selected:
    fn=capture['margins'];low,high=capture['low'],capture['high'];horizon=capture['horizon'];intervals=capture['intervals'];seeds=[preferred,np.zeros(3),result['base_command'],low,high,(low+high)/2]
    seeds += [np.where(np.array(s),high,low) for s in itertools.product([False,True],repeat=3)]
    best,attempts=max_margin(fn,low,high,seeds);sampled=np.array([fn(np.array(v)) for v in itertools.product(*[np.linspace(a,b,5) for a,b in zip(low,high)])]);gridmax=float(np.max(np.min(sampled,axis=1)));margin=fn(np.array(best['v']));j=int(np.argmin(margin));elapsed=float(intervals[j//12]);k=j%12;labels=['x_lower','y_lower','z_lower','x_upper','y_upper','z_upper','roll_lower','pitch_lower','yaw_lower','roll_upper','pitch_upper','yaw_upper'];reserve=np.tile(np.r_[coord.reserve,coord.reserve,[.015]*6],len(intervals))*np.repeat(np.minimum(1.,intervals/max(horizon,1e-9)),12);outer=margin+reserve
    entry=dict(row=i,measurement_end_stamp_s=r['stamp_s'],command_stamp_s=stamp,recorded_reason=support['reason'],recorded_min_margin_mixed_units=support['predicted_min_domain_margin'],native_current_xyz=backend.camera_to_mixed(sampler.sample(stamp),link,E)[:3].tolist(),preferred=preferred.tolist(),previous_applied=previous.tolist(),first_command_bounds_low=low.tolist(),first_command_bounds_high=high.tolist(),original_fixed_future_plan=future.tolist(),original_model_maxmin=best,multistart_attempt_count=len(attempts),grid5cube_best_margin=gridmax,active_bound=dict(name=labels[k],lookahead_s=elapsed,margin_mixed_units=float(margin[j])),same_best_command_min_outer_bounds_margin=float(np.min(outer)),original_model_feasible_found=best['margin']>=-1e-7,counterfactual_future_x_plans=[])
    for fx in [-.05,-.10,-.15,-.20]:
     future[0]=fx
     alternative,_=max_margin(fn,low,high,[best['v'],low,(low+high)/2])
     entry['counterfactual_future_x_plans'].append(dict(future_x=fx,original_first_slew_and_caps_unchanged=True,future_x_ramp_source_slew=.5,changes_forecast_not_Task=True,maxmin=alternative,scope='Different future input plan in the same source delayed/ramped prediction; not feasible under the original fixed-zero future plan and not a physically executed controller'))
    future[0]=0.;report['selected'][str(i)]=entry
  coord.observe_applied_input(stamp,np.array(r['applied_policy_velocity']))
 report['replay_counts']=dict(active_rows=len(rows)-100,original_false=len(bad),replayed_false=false_count,status_mismatches=status_mismatch,maximum_current_native_command_difference=maxnative,maximum_base_command_difference=maxcmd,maximum_original_margin_difference=maxmargin)
 report['cpu_wall_s']=time.monotonic()-start
 files=[Path(__file__),R/'summary.json',R/'measurements.jsonl',R/'timed_tasks.json',R/'provenance.json',R/'sensor_mounts.json',F/'camera_coordinator.py',F/'go2_backend.py',F/'optical_sampler.py']
 report['source_sha256']={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};out=P/'validation/go2_camera6d_support_replay_husky_20261020.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(out);print(report['replay_counts'])
 for e in report['selected'].values():print(e['measurement_end_stamp_s'],'orig',round(e['original_model_maxmin']['margin'],7),'outer',round(e['same_best_command_min_outer_bounds_margin'],7),'bound',e['active_bound'],'futureX',[round(d['maxmin']['margin'],6) for d in e['counterfactual_future_x_plans']])
if __name__=='__main__':main()
