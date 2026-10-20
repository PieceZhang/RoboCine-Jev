#!/usr/bin/env python3
"""Read-only comparison of three preserved Go2 stop failures; no control changes."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
PACKAGE=Path(__file__).resolve().parents[1]
NAMES=('response','ground','precision')

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def span_rotation(R):
 q=Rotation.from_matrix(R).as_quat();return float(2*np.arccos(np.clip(np.min(np.abs(q@q.T)),0,1)))
def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-10 and np.std(b)>1e-10 else None
def top_spans(names,arr):return [dict(joint=names[i],span_rad=float(np.ptp(arr[:,i]))) for i in np.argsort(np.ptp(arr,axis=0))[-6:][::-1]]
def main():
 out={'schema':'go2_stop_readonly_diagnostic/v1','scope':'Preserved low-damping run measurements only, no new physics, no source or Task edits, no claim about checkpoint training distribution.','runs':{},'source_sha256':{str(Path(__file__).relative_to(PACKAGE)):digest(Path(__file__))}}
 for n in NAMES:
  root=PACKAGE/f'Robot_Go2/results/adaptive_backward_{n}_01';rows=[json.loads(s) for s in (root/'measurements.jsonl').open()];stop=[r for r in rows if r['case']=='backward' and r['phase']=='stop'];tail=[r for r in stop if r['stamp_s']>=stop[-1]['stamp_s']-.5-1e-9];late=[r for r in stop if r['stamp_s']>=stop[-1]['stamp_s']-2-1e-9]
  def calculate(s):
   names=s[0]['joint_names'];q=np.array([r['joint_positions'] for r in s]);qd=np.array([r['measured_joint_velocity'] for r in s]);targets=np.array([r['applied_joint_position_target'] for r in s]);v=np.array([r['applied_policy_velocity'] for r in s]);native=np.array([r['actual_native_command'] for r in s]);base=np.array([r['T_world_base'] for r in s]);cam=np.array([r['T_world_camera_optical'] for r in s]);angles=Rotation.from_matrix(base[:,:3,:3]).as_euler('xyz');ne=Rotation.from_quat(native[:,[4,5,6,3]]).as_euler('xyz')
   # Flattened tested_scene fixed joint base->link0 is translation(0,0,.07), identity rotation.
   link=base.copy();link[:,:3,3]+=base[:,:3,:3]@np.array([0.,0.,.07])
   return dict(samples=len(s),stamp_start=s[0]['stamp_s'],stamp_end=s[-1]['stamp_s'],all_applied_base_velocity_exact_zero=bool(np.all(v==0)),zero_input_fraction=float(np.mean(np.all(v==0,axis=1))),max_abs_policy_body_velocity=np.max(abs(v),axis=0).tolist(),native_standing_fraction=float(np.mean([r['native_standing_mode'] for r in s])),base_rotation_span_rad=span_rotation(base[:,:3,:3]),camera_rotation_span_rad=span_rotation(cam[:,:3,:3]),base_euler_span_rad=np.ptp(angles,axis=0).tolist(),native_ee_xyz_span=np.ptp(native[:,:3],axis=0).tolist(),native_ee_euler_span_rad=np.ptp(ne,axis=0).tolist(),native_ee_xyz_mean=np.mean(native[:,:3],axis=0).tolist(),native_ee_euler_mean_rad=np.mean(ne,axis=0).tolist(),native_vs_base_euler_correlation=[corr(ne[:,i],angles[:,i]) for i in range(3)],largest_measured_joint_spans=top_spans(names,q),largest_policy_position_target_spans=top_spans(names,targets),max_abs_measured_qd=np.max(abs(qd),axis=0).tolist(),joint_names=names,measured_target_error_rmse_rad=np.sqrt(np.mean((targets-q)**2,axis=0)).tolist(),target_vs_measured_each_axis_correlation=[corr(targets[:,i],q[:,i]) for i in range(len(names))],link0_pose_provenance='reconstructed base SE3 @ fixed source joint (0,0,.07) identity; not separately logged measured link0 body',link0_world_translation_span_m=np.ptp(link[:,:3,3],axis=0).tolist(),final_base_linear_speed=float(np.linalg.norm(s[-1]['base_linear_velocity_world'])),final_base_angular_speed=float(np.linalg.norm(s[-1]['base_angular_velocity_world'])))
  transitions=[dict(stamp_s=r['stamp_s'],**r['parking_transition']) for r in stop if r['parking_transition'] and (r['parking_transition']['entered'] or r['parking_transition']['exited'])]
  zero=[r['stamp_s'] for r in stop if np.all(np.array(r['applied_policy_velocity'])==0)]
  out['runs'][n]=dict(full_stop_start=stop[0]['stamp_s'],full_stop_end=stop[-1]['stamp_s'],first_exact_zero_input_s=min(zero) if zero else None,parking_transitions=transitions,tail_half_second=calculate(tail),last_two_seconds=calculate(late),recorded_strict_stop=json.loads((root/'summary.json').read_text())['cases'][0]['strict_stop'])
  for rel in ('measurements.jsonl','summary.json','provenance.json','sensor_mounts.json','tested_scene.usda','sources/adaptive_runner.py','sources/go2_backend.py','sources/velocity_servo.py','sources/go2_upstream/config/go2_piper/wbc_env_cfg.py','sources/go2_upstream/mdp/pose_command_wbc.py'):
   p=root/rel;out['source_sha256'][str(p.relative_to(PACKAGE))]=digest(p)
 out['findings']=[
  'Literal zero base velocity input is not a static full-robot action: all18 actions are recomputed by the learned policy every tick, including legs and arm; PD joint targets continue changing. No frozen18action substitute is proposed.',
  'The public world-fixed endpoint requires native EE commands to change with measured link0 roll/pitch/translation. Measured camera orientation can remain much steadier than base/arm while native targets and joints continue moving. Command=zero for base cannot imply allDOF stop.',
  'Parking hysteresis re-exits learned standing mode when the original anchor drift crosses25mm or50mrad; the mode and velocity servo then resume corrections. This is an actual recorded mechanism, not merely residual numerical noise.',
  'Ground/precision changes did not remove the stop failure in these preserved runs. This comparison is observational, not a one-factor causal proof because complete frozen configurations differ.',
  'Pinned upstream EE sampler is mixed frame: xy in link0, z world; orientation includes bearing pitch/yaw toward the sampled target. Config ranges and limit_ranges define an outer envelope, not the actual checkpoint training density or a known static equilibrium. A soft native pose preference is testable but its stability is not established by envelope membership.'
 ]
 target=PACKAGE/'validation/go2_stop_diagnostic_husky_20261020.json'
 if target.exists():raise FileExistsError(target)
 target.write_text(json.dumps(out,indent=2)+'\n');print(target)
 for n,d in out['runs'].items():
  t=d['tail_half_second'];print(n,'zeroFraction',t['zero_input_fraction'],'targetSpanTop',t['largest_policy_position_target_spans'][:2],'qSpanTop',t['largest_measured_joint_spans'][:2],'native/baseEulerCorrelation',t['native_vs_base_euler_correlation'],'standingTransitions',len(d['parking_transitions']))
if __name__=='__main__':main()
