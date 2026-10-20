from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
p=Path(__file__).resolve().parents[2];run=p/'results/ocs2_acceptance_02';rows=[json.loads(l) for l in (run/'samples.jsonl').open()];moving=[r for r in rows if r['case']=='ocs2_camera'];task=json.loads((run/'task.json').read_text());t0=task['camera_trajectory']['start_s'];tol=task['position_tolerance_m'];bad=[r for r in moving if r['position_error_m']>tol];groups=[]
for r in bad:
 if not groups or r['stamp_s']-groups[-1][-1]['stamp_s']>.010001:groups.append([])
 groups[-1].append(r)
peak=max(moving,key=lambda r:r['position_error_m']);journal=[json.loads(l) for l in (run/'native_mpc.jsonl').open()];policies=[j for j in journal if j['result'].get('ok') is True]
matching=[]
for j in policies:
 req,res=j['request'],j['result'];t=peak['stamp_s']-.01
 if req['state_stamp_s']<=t<=req['state_stamp_s']+.200001:
  states=np.array(res['states']);times=res['times_s'];u=np.array([np.interp(t,times,states[:,9+k]) for k in range(8)])
  err=float(np.linalg.norm(u[:2]-np.array(peak['requested_body_velocity'])[[0,2]]))
  if err<1e-8:matching.append(j)
sp=importlib.util.spec_from_file_location('a',p/'ocs2/adapter.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);adapter=m.WheeledBackendAdapter();ids=[peak['joint_names'].index(n) for n in adapter.joint_names]
metrics=dict(scope='post hoc analysis of preserved full02 raw measurements; no Task, tolerance, trajectory, controller or acceptance modification',position_tolerance_m=tol,moving_samples=len(moving),pointwise_position_exceeding_samples=len(bad),pointwise_position_exceeding_fraction=len(bad)/len(moving),pointwise_position_exceeding_total_sample_duration_s=len(bad)*.01,pointwise_orientation_exceeding_samples=sum(r['orientation_error_rad']>task['orientation_tolerance_rad'] for r in moving),intervals=[dict(start_s=g[0]['stamp_s'],end_s=g[-1]['stamp_s'],start_task_relative_s=g[0]['stamp_s']-t0,end_task_relative_s=g[-1]['stamp_s']-t0,samples=len(g),maximum_error_m=max(r['position_error_m'] for r in g)) for g in groups],peak=dict(stamp_s=peak['stamp_s'],task_relative_s=peak['stamp_s']-t0,position_error_m=peak['position_error_m'],orientation_error_rad=peak['orientation_error_rad'],actual_minus_reference_m=(np.array(peak['T_world_camera_optical'])[:3,3]-np.array(peak['T_world_reference_optical'])[:3,3]).tolist(),requested_body_velocity=peak['requested_body_velocity'],measured_body_velocity=peak['measured_body_velocity']),native_failed_requests=sum(j['result'].get('ok') is not True for j in journal),native_max_transport_wall_s=max(j['result']['transport_wall_s'] for j in journal),matching_active_policy_ids=[j['request']['request_id'] for j in matching])
if len(matching)==1:
 j=matching[0];req,res=j['request'],j['result'];t=peak['stamp_s']-.01;states=np.array(res['states']);x=np.array([np.interp(t,res['times_s'],states[:,k]) for k in range(states.shape[1])]);T=np.array(req['T_world_base']);Rz=lambda yaw:Rotation.from_euler('z',yaw).as_matrix();T[:3,:3]=Rz(x[2])@Rz(-req['state'][2])@T[:3,:3];T[:2,3]=x[:2]
 predicted=adapter.measured_camera_pose(T,x[3:9]);actualbase=np.array(peak['T_world_base']);actual=np.array(peak['T_world_camera_optical']);ref=np.array(peak['T_world_reference_optical']);hybrid=adapter.measured_camera_pose(actualbase,x[3:9])
 metrics['peak_policy_comparison']=dict(request_id=req['request_id'],policy_age_sim_s=t-req['state_stamp_s'],transport_wall_s=res['transport_wall_s'],predicted_camera_minus_reference_m=(predicted[:3,3]-ref[:3,3]).tolist(),predicted_position_error_m=float(np.linalg.norm(predicted[:3,3]-ref[:3,3])),base_pose_deviation_camera_component_m=(hybrid[:3,3]-predicted[:3,3]).tolist(),arm_configuration_deviation_camera_component_m=(actual[:3,3]-hybrid[:3,3]).tolist(),measured_minus_predicted_base_xy_m=(actualbase[:2,3]-x[:2]).tolist(),measured_minus_predicted_yaw_rad=float(np.arctan2(actualbase[1,0],actualbase[0,0])-x[2]),maximum_arm_position_prediction_error_rad=float(np.max(np.abs(np.array(peak['q'])[ids]-x[3:9]))))
stopped=[r for r in rows if r['case']=='final_stop']
metrics['terminal_evolution']=[dict(stamp_s=r['stamp_s'],seconds_after_task_end_s=r['stamp_s']-moving[-1]['stamp_s'],position_error_m=r['position_error_m'],orientation_error_rad=r['orientation_error_rad'],requested_body_velocity=r['requested_body_velocity'],measured_body_velocity=r['measured_body_velocity']) for r in [moving[-1],stopped[0],stopped[9],stopped[24],stopped[49],stopped[99],stopped[199],stopped[299],stopped[-1]]]
metrics['final_task_tolerance_pass']=bool(stopped[-1]['position_error_m']<tol and stopped[-1]['orientation_error_rad']<task['orientation_tolerance_rad'])
metrics['historical_acceptance_scope']='Historical acceptance.json only gated RMS and strict motion stop; final Task tolerance is now independently checked and fails. The historical file is retained unchanged.'
perf=json.loads((run/'timing_breakdown.json').read_text())['rows'];pm=next(r for r in perf if r['stamp_s']==peak['stamp_s']);metrics['peak_loop_timing']=pm
metrics['interpretation']='The saved RMS acceptance is distinct from per-sample tolerance feedback. Decomposition compares the exact matched native predicted pose with actual base/arm measurements; it indicates contributions, not a unique causal experiment or a continuous-domain guarantee.'
metrics['source_files_sha256']={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in ['samples.jsonl','task.json','native_mpc.jsonl','timing_breakdown.json','acceptance.json']}
(run/'pointwise_terminal_analysis_v2.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
