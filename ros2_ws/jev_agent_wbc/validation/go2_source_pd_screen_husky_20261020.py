#!/usr/bin/env python3
"""Onefactor source-PD motivation; illustrative eigen screen is not PhysX proof."""
from pathlib import Path
import ast,hashlib,json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'Robot_Go2/results/adaptive_suite_output_continuity_01'
OUT=ROOT/'Robot_Go2/results/source_pd_screen_cpu_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 stage=RUN/'tested_scene.usda';props={};owner=None
 for line in stage.read_text().splitlines():
  m=re.match(r'        def Xform "([^"]+)"',line)
  if m:owner=m.group(1)
  if owner is not None:
   for key in ['mass','centerOfMass','diagonalInertia']:
    m=re.match(r'\s+(?:float|point3f|float3) physics:'+key+r' = (.*)',line)
    if m:props.setdefault(owner,{})[key]=ast.literal_eval(m.group(1))
 props={n:v for n,v in props.items()if 'mass'in v};assert len(props)==25
 rows=[json.loads(x)for x in(RUN/'measurements.jsonl').read_text().splitlines()];coms=[]
 for case in ['forward','backward','left','right','diagonal_left','diagonal_right','yaw_left','yaw_right','camera_shot']:
  row=next(r for r in reversed(rows)if r['case']==case);poses=row['fine_collision_evidence']['T_world_collision_bodies'];B=np.linalg.inv(np.asarray(poses['base']))
  total=np.zeros(3);arm=np.zeros(3);mass=arm_mass=0.
  for n,p in props.items():
   T=B@np.asarray(poses[n]);point=T[:3,:3]@p.get('centerOfMass',[0,0,0])+T[:3,3];total+=p['mass']*point;mass+=p['mass']
   if re.fullmatch(r'link\d+',n):arm+=p['mass']*point;arm_mass+=p['mass']
  coms.append(dict(case=case,authored_mass_sum_kg=mass,authored_arm_mass_sum_kg=arm_mass,
   authored_model_com_in_measured_base_m=(total/mass).tolist(),authored_arm_com_in_measured_base_m=(arm/arm_mass).tolist(),
   pose_stamp_s=row['fine_collision_evidence']['stamp_s'],scope='USD authored mass/COM transformed by actual measured FCL body poses; actual native mass tensors were not saved in this old run. Not a verified fused PhysX COM.'))
 screen=[];h=.005;kp=30.;omega=2*np.pi*2.
 for J in [.01,.02,.05,.1,.2,.3]:
  values=[]
  for scale in [1.,3.]:
   kd=.6*scale;A=np.array([[1-kp*h*h/J,h*(1-kd*h/J)],[-kp*h/J,1-kd*h/J]])
   D=np.array([kp*h*h/J,kp*h/J]);z=np.exp(1j*omega*h);H=(np.linalg.solve(z*np.eye(2)-A,D))[0]
   values.append(dict(park_scale=scale,kd=kd,eigenvalue_abs=np.abs(np.linalg.eigvals(A)).tolist(),
    transfer_magnitude_2Hz=float(abs(H)),transfer_phase_degrees_2Hz=float(np.angle(H,deg=True)),damping_ratio=float(kd/(2*np.sqrt(kp*J)))))
  screen.append(dict(illustrative_effective_joint_inertia_kgm2=J,models=values))
 report=dict(schema='go2_source_pd_candidate_screen/v1',source_native_stiffness=30.,source_native_damping=.6,
  training_gain_multiplier=[.8,1.2],training_damping_range=[.48,.72],previous_park_damping=1.8,proposed_park_damping=.6,
  delayed_target_ticks=[0,4],physics_dt_s=.005,old_actual_sampled_delay='unknown: no readback in old journals',
  actual_actuator_model='explicit DelayedPD: current q/qd, delayed target, original motor saturation; D term alone dissipative',
  scalar_screen=screen,authored_model_com= coms,
  limitations=['Illustrative isolated semiimplicit joint model; effective contact/articulation inertia is unknown and grid is not an inferred bound.',
   'D3 is locally dissipative and both scalar models stable here; neither causality nor full-contact/learned-loop stability is certified.',
   'At2Hz phase effect depends on effective inertia; matching native trained gains is the candidate rationale, not a claim D3 is intrinsically unstable.',
   'Only complete sameprofile sourceD1 physical trial can determine all18qstop/Task/FCL/timing.'],
  source_sha256={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),stage,RUN/'actuator_audit.json',RUN/'measurements.jsonl',
   ROOT/'Robot_Go2/vendor/LeggedManip_Lab/source/LeggedManip_Lab/LeggedManip_Lab/assets/go2_piper/go2_piper_articulation_cfg.py',
   ROOT/'Robot_Go2/vendor/LeggedManip_Lab/source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/leggedmanip_lab_env_cfg.py',
   ROOT/'Robot_Go2/vendor/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py']},physical_verified=False)
 OUT.mkdir(exist_ok=False);(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
