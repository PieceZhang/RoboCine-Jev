#!/usr/bin/env python3
"""Read-only post hoc Go2 FULL stop/native-input coupling comparisons."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE/'Robot_Go2/results/adaptive_suite_softpose_01'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def correlated(a,b):return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-10 and np.std(b)>1e-10 else None

def main():
 rows=[json.loads(line) for line in (ROOT/'measurements.jsonl').open()];summary=json.loads((ROOT/'summary.json').read_text());cases={c['case']:c for c in summary['cases']}
 report=dict(schema='go2_full_stop_coupling_readonly/v1',scope='Measured92s FULL preserved data only; no new physics, no source/Task edits. Cross-case associations are confounded by case ordering/initial state and are not causal proof or knowledge of checkpoint training density.',cases={},source_sha256={})
 def stats(selected):
  native=np.array([r['actual_native_command'] for r in selected]);body=np.array([r['T_world_base'] for r in selected]);camera=np.array([r['T_world_camera_optical'] for r in selected]);target=np.array([r['applied_joint_position_target'] for r in selected]);q=np.array([r['joint_positions'] for r in selected]);qd=np.array([r['measured_joint_velocity'] for r in selected]);names=selected[0]['joint_names'];euler=Rotation.from_quat(native[:,[4,5,6,3]]).as_euler('xyz');base_angles=Rotation.from_matrix(body[:,:3,:3]).as_euler('xyz')
  link=body.copy();link[:,:3,3]+=body[:,:3,:3]@np.array([0,0,.07])
  # Reproduce the pinned sampler's zero angular-offset bearing construction.
  # This is a geometric comparison, not a certificate of policy training density.
  planar=np.c_[native[:,:2],np.zeros(len(native))];target_world=link[:,:3,3]+np.einsum('nij,nj->ni',link[:,:3,:3],planar);target_world[:,2]=native[:,2];delta=np.einsum('nji,nj->ni',link[:,:3,:3],target_world-link[:,:3,3]);bearing=np.c_[np.zeros(len(native)),-np.arctan2(delta[:,2],np.linalg.norm(delta[:,:2],axis=1)),np.arctan2(delta[:,1],delta[:,0])];bearing=np.clip(bearing,-np.pi/4,np.pi/3)
  orientation_deviation=(Rotation.from_euler('xyz',bearing).inv()*Rotation.from_euler('xyz',euler)).magnitude()
  tops=lambda a:[dict(joint=names[i],span_rad=float(np.ptp(a[:,i]))) for i in np.argsort(np.ptp(a,axis=0))[-6:][::-1]]
  v=np.array([r['applied_policy_velocity'] for r in selected]);angular=np.array([r['base_angular_velocity_world'] for r in selected]);domain=[r['actual_native_domain'] for r in selected];posture=[r['camera_support']['native_posture_preference'] for r in selected]
  return dict(samples=len(selected),stamp_start_s=selected[0]['stamp_s'],stamp_end_s=selected[-1]['stamp_s'],native_xyz_mean=np.mean(native[:,:3],axis=0).tolist(),native_xyz_min=np.min(native[:,:3],axis=0).tolist(),native_xyz_max=np.max(native[:,:3],axis=0).tolist(),native_xyz_span=np.ptp(native[:,:3],axis=0).tolist(),native_euler_mean_rad=np.mean(euler,axis=0).tolist(),native_euler_span_rad=np.ptp(euler,axis=0).tolist(),mean_native_bearing_euler_rad=np.mean(bearing,axis=0).tolist(),mean_native_minus_bearing_euler_rad=np.mean(euler-bearing,axis=0).tolist(),native_orientation_bearing_deviation_mean_rad=float(np.mean(orientation_deviation)),base_rpy_mean_rad=np.mean(base_angles,axis=0).tolist(),base_rpy_span_rad=np.ptp(base_angles,axis=0).tolist(),native_vs_base_rpy_correlation=[correlated(euler[:,i],base_angles[:,i]) for i in range(3)],base_height_mean_m=float(np.mean(body[:,2,3])),base_height_span_m=float(np.ptp(body[:,2,3])),base_angular_velocity_rms_radps=np.sqrt(np.mean(angular**2,axis=0)).tolist(),base_angular_velocity_final_world_radps=angular[-1].tolist(),exact_zero_policy_velocity_fraction=float(np.mean(np.all(v==0,axis=1))),max_abs_policy_velocity=np.max(abs(v),axis=0).tolist(),standing_mode_fraction=float(np.mean([r['native_standing_mode'] for r in selected])),posture_preference_enabled_fraction=float(np.mean([r['enabled'] for r in posture])),posture_preference_mean_weight=float(np.mean([r['weight'] for r in posture])),largest_measured_q_spans=tops(q),largest_learned_pd_target_spans=tops(target),max_joint_speed_radps=float(np.max(abs(qd))),camera_error_position_mean_m=float(np.mean([r['position_error_m'] for r in selected])),camera_error_orientation_mean_rad=float(np.mean([r['orientation_error_rad'] for r in selected])))
 for name,c in cases.items():
  case=[r for r in rows if r['case']==name];stop=[r for r in case if r['phase']=='stop'];end=stop[-1]['stamp_s'];moving=[r for r in case if r['phase']=='move'];tail=[r for r in stop if r['stamp_s']>=end-.5-1e-8];late=[r for r in stop if r['stamp_s']>=end-2.-1e-8]
  report['cases'][name]=dict(passed=c['strict_stop']['passed'],recorded_stop=c['strict_stop'],moving_last_half_second=stats([r for r in moving if r['stamp_s']>=moving[-1]['stamp_s']-.5-1e-8]),stop_first_second=stats([r for r in stop if r['stamp_s']<=stop[0]['stamp_s']+1.+1e-8]),stop_last_two_seconds=stats(late),stop_last_half_second=stats(tail),parking_events=[dict(stamp_s=r['stamp_s'],**r['parking_transition']) for r in stop if r['parking_transition'] and (r['parking_transition']['entered'] or r['parking_transition']['exited'])])
 for f in [Path(__file__),ROOT/'summary.json',ROOT/'measurements.jsonl',ROOT/'timed_tasks.json',ROOT/'provenance.json',ROOT/'sources/camera_coordinator.py',ROOT/'sources/go2_upstream/mdp/pose_command_wbc.py',ROOT/'sources/go2_upstream/config/go2_piper/wbc_env_cfg.py']:
  report['source_sha256'][str(f.relative_to(PACKAGE))]=digest(f)
 output=PACKAGE/'validation/go2_full_stop_coupling_husky_20261020.json';assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n')
 print(output)
 for name,c in report['cases'].items():
  x=c['stop_last_half_second'];print(name,'pass',c['passed'],'xyz',np.round(x['native_xyz_mean'],4),'rpy',np.round(x['native_euler_mean_rad'],4),'bearingDelta',np.round(x['mean_native_minus_bearing_euler_rad'],4),'baseRP',np.round(x['base_rpy_mean_rad'][:2],4),'zero',x['exact_zero_policy_velocity_fraction'],'mode',x['standing_mode_fraction'],'qtargetmax',x['largest_learned_pd_target_spans'][0])
if __name__=='__main__':main()
